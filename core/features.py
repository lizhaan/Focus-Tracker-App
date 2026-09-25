from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

import numpy as np
import cv2


RIGHT_EYE = {"P1": 33, "P2": 160, "P3": 158, "P4": 133, "P5": 153, "P6": 144}
LEFT_EYE = {"P1": 362, "P2": 385, "P3": 387, "P4": 263, "P5": 373, "P6": 380}

IRIS_CENTER = {"right": 473, "left": 468}

HEAD_POSE_LANDMARKS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 362,
    "right_eye_outer": 33,
    "left_mouth_corner": 291,
    "right_mouth_corner": 61,
}

MODEL_POINTS_3D = np.array(
    [
        (0.0, 0.0, 0.0),  # Tip of the nose
        (0.0, -63.6, -12.5),  # Chin
        (-43.3, 32.7, -26.0),  # Left outer eye corner
        (43.3, 32.7, -26.0),  # Right outer eye corner
        (-28.9, -28.9, -24.1),  # Left corner of the mouth
        (28.9, -28.9, -24.1),  # Right corner of the mouth
    ],
    dtype=np.float64,
)

_EPS = 1e-9


def _dist(a: np.ndarray, b: np.ndarray, use_z: bool = True) -> float:
    """Euclidean distance between two points"""
    if use_z:
        diff = a - b
    else:
        diff = a[:2] - b[:2]
    return float(np.linalg.norm(diff))

# 1. EAR

def _eye_aspect_ratio(landmarks: np.ndarray, eye: Dict[str, int]) -> float:
    p1, p2, p3 = landmarks[eye["P1"]], landmarks[eye["P2"]], landmarks[eye["P3"]]
    p4, p5, p6 = landmarks[eye["P4"]], landmarks[eye["P5"]], landmarks[eye["P6"]]

    horizontal = _dist(p1, p4, use_z=False)
    if horizontal < _EPS:
        return 0.0

    vertical = _dist(p2, p6, use_z=False) + _dist(p3, p5, use_z=False)
    return vertical / (2.0 * horizontal)


def compute_ear(landmarks: np.ndarray) -> float:
    ear_right = _eye_aspect_ratio(landmarks, RIGHT_EYE)
    ear_left = _eye_aspect_ratio(landmarks, LEFT_EYE)
    return (ear_right + ear_left) / 2.0


# 2. Head Pose — Yaw / Pitch / Roll

@dataclass
class HeadPose:
    yaw: float
    pitch: float
    roll: float
    rvec: np.ndarray
    tvec: np.ndarray


def _camera_matrix(frame_width: int, frame_height: int) -> np.ndarray:
    focal_length = float(frame_width)  # f ≈ W (без окремого калібрування камери)
    return np.array(
        [
            [focal_length, 0.0, frame_width / 2.0],
            [0.0, focal_length, frame_height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def compute_head_pose(
        landmarks: np.ndarray, frame_width: int, frame_height: int
) -> Optional[HeadPose]:
    image_points_px = np.array(
        [
            (
                landmarks[idx][0] * frame_width,
                landmarks[idx][1] * frame_height,
            )
            for idx in HEAD_POSE_LANDMARKS.values()
        ],
        dtype=np.float64,
    )

    camera_matrix = _camera_matrix(frame_width, frame_height)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    try:
        success, rvec, tvec = cv2.solvePnP(
            MODEL_POINTS_3D,
            image_points_px,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error:
        return None
    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    r = rotation_matrix

    pitch = np.arctan2(r[2, 1], r[2, 2])
    yaw = np.arctan2(-r[2, 0], np.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2))
    roll = np.arctan2(r[1, 0], r[0, 0])

    to_deg = 180.0 / np.pi
    return HeadPose(
        yaw=float(yaw * to_deg),
        pitch=float(pitch * to_deg),
        roll=float(roll * to_deg),
        rvec=rvec,
        tvec=tvec,
    )

#Gaze_raw / Gaze_vertical_raw
def _gaze_ratio_for_eye(
        landmarks: np.ndarray, eye: Dict[str, int], iris_idx: int
) -> tuple[float, float]:
    iris = landmarks[iris_idx]
    outer, inner = landmarks[eye["P1"]], landmarks[eye["P4"]]
    upper = (landmarks[eye["P2"]] + landmarks[eye["P3"]]) / 2.0
    lower = (landmarks[eye["P5"]] + landmarks[eye["P6"]]) / 2.0

    d_out = _dist(iris, outer, use_z=False)
    d_in = _dist(iris, inner, use_z=False)
    d_up = _dist(iris, upper, use_z=False)
    d_down = _dist(iris, lower, use_z=False)

    horizontal = d_out / d_in if d_in > _EPS else 0.0
    vertical = d_down / d_up if d_up > _EPS else 0.0
    return horizontal, vertical


def compute_gaze_raw(landmarks: np.ndarray) -> tuple[float, float]:

    h_right, v_right = _gaze_ratio_for_eye(landmarks, RIGHT_EYE, IRIS_CENTER["right"])
    h_left, v_left = _gaze_ratio_for_eye(landmarks, LEFT_EYE, IRIS_CENTER["left"])

    gaze_raw = (h_right + h_left) / 2.0
    gaze_vertical_raw = (v_right + v_left) / 2.0
    return gaze_raw, gaze_vertical_raw


#Smoothing

@dataclass
class _SignalState:
    window: Deque[float]
    ema_value: Optional[float] = None


class MetricsSmoother:
    def __init__(self, window_size: int = 3, alpha: float = 0.3):
        if not 3 <= window_size <= 5:
            raise ValueError("window_size (k) має бути в діапазоні 3..5")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha має бути в діапазоні (0, 1]")

        self.window_size = window_size
        self.alpha = alpha
        self._signals: Dict[str, _SignalState] = {}

    def _state_for(self, name: str) -> _SignalState:
        if name not in self._signals:
            self._signals[name] = _SignalState(window=deque(maxlen=self.window_size))
        return self._signals[name]

    def update(self, name: str, raw_value: float) -> float:
        state = self._state_for(name)
        state.window.append(raw_value)

        x_med = float(np.median(state.window))

        if state.ema_value is None:
            # x̂(0) = x_med(0)
            state.ema_value = x_med
        else:
            state.ema_value = self.alpha * x_med + (1.0 - self.alpha) * state.ema_value

        return state.ema_value

    def reset(self, name: Optional[str] = None) -> None:
        if name is None:
            self._signals.clear()
        else:
            self._signals.pop(name, None)


#Composite image of metrics from a single frame
@dataclass
class FrameMetrics:
    ear_raw: float
    yaw_raw: Optional[float]
    pitch_raw: Optional[float]
    roll_raw: Optional[float]
    gaze_raw: float
    gaze_vertical_raw: float

    ear_smoothed: float = field(default=0.0)
    yaw_smoothed: float = field(default=0.0)
    pitch_smoothed: float = field(default=0.0)
    gaze_smoothed: float = field(default=0.0)
    gaze_vertical_smoothed: float = field(default=0.0)


def compute_frame_metrics(
        landmarks: np.ndarray,
        frame_width: int,
        frame_height: int,
        smoother: MetricsSmoother,
) -> FrameMetrics:
    ear = compute_ear(landmarks)
    gaze_raw, gaze_vertical_raw = compute_gaze_raw(landmarks)
    pose = compute_head_pose(landmarks, frame_width, frame_height)

    yaw = pose.yaw if pose is not None else None
    pitch = pose.pitch if pose is not None else None
    roll = pose.roll if pose is not None else None

    metrics = FrameMetrics(
        ear_raw=ear,
        yaw_raw=yaw,
        pitch_raw=pitch,
        roll_raw=roll,
        gaze_raw=gaze_raw,
        gaze_vertical_raw=gaze_vertical_raw,
    )

    metrics.ear_smoothed = smoother.update("ear", ear)
    metrics.gaze_smoothed = smoother.update("gaze_raw", gaze_raw)
    metrics.gaze_vertical_smoothed = smoother.update(
        "gaze_vertical_raw", gaze_vertical_raw
    )
    if yaw is not None:
        metrics.yaw_smoothed = smoother.update("yaw", yaw)
    if pitch is not None:
        metrics.pitch_smoothed = smoother.update("pitch", pitch)

    return metrics