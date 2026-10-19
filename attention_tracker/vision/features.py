"""Сирі ознаки кадру: EAR, поза голови, погляд.

Усі геометричні обчислення — у пікселях. Функції чисті; стан (попередні
rvec/tvec для solvePnP) тримає тільки `FeatureExtractor`.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from attention_tracker.models import FrameFeatures
from attention_tracker.vision.landmarks import (
    FACE_MODEL_3D,
    HEAD_POSE_INDICES,
    LEFT_EYE_EAR,
    LEFT_EYE_GAZE,
    LEFT_IRIS,
    RIGHT_EYE_EAR,
    RIGHT_EYE_GAZE,
    RIGHT_IRIS,
)

_EPS = 1e-9  # захист від ділення на нуль у виродженій геометрії


def wrap(theta: float) -> float:
    """wrap(θ) = ((θ + 180) mod 360) − 180, градуси."""
    return ((theta + 180.0) % 360.0) - 180.0


def to_pixels(landmarks: np.ndarray, frame_w: int, frame_h: int) -> np.ndarray:
    """pᵢ = (xᵢ·W, yᵢ·H). Повертає масив (N, 2); z відкидається."""
    return landmarks[:, :2] * np.array([frame_w, frame_h], dtype=np.float64)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    """dist(a, b) = ‖a − b‖₂."""
    return float(np.linalg.norm(a - b))


# --- EAR --------------------------------------------------------------


def eye_aspect_ratio(px: np.ndarray, eye: tuple[int, int, int, int, int, int]) -> float:
    """EAR_eye = (dist(P2,P6) + dist(P3,P5)) / (2·dist(P1,P4))."""
    p1, p2, p3, p4, p5, p6 = (px[i] for i in eye)
    horizontal = _dist(p1, p4)
    if horizontal < _EPS:
        return 0.0
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * horizontal)


def ear(px: np.ndarray) -> float:
    """EAR(t) = (EAR_right + EAR_left) / 2."""
    return (eye_aspect_ratio(px, RIGHT_EYE_EAR) + eye_aspect_ratio(px, LEFT_EYE_EAR)) / 2.0


# --- Head pose ----------------------------------------------------------


@dataclass(frozen=True)
class HeadPose:
    """кути в градусах + розвʼязок solvePnP для наступного кадру."""

    yaw: float
    pitch: float
    roll: float
    rvec: np.ndarray
    tvec: np.ndarray


def camera_matrix(frame_w: int, frame_h: int) -> np.ndarray:
    """f = W, головна точка в центрі кадру."""
    f = float(frame_w)
    return np.array(
        [[f, 0.0, frame_w / 2.0], [0.0, f, frame_h / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def angles_from_rotation(r: np.ndarray) -> tuple[float, float, float]:
    """(yaw, pitch, roll) у градусах з матриці повороту R."""
    pitch = np.arctan2(r[2, 1], r[2, 2])
    yaw = np.arctan2(-r[2, 0], np.sqrt(r[2, 1] ** 2 + r[2, 2] ** 2))
    roll = np.arctan2(r[1, 0], r[0, 0])
    return float(np.degrees(yaw)), float(np.degrees(pitch)), float(np.degrees(roll))


def head_pose(
    px: np.ndarray,
    frame_w: int,
    frame_h: int,
    prev: tuple[np.ndarray, np.ndarray] | None = None,
) -> HeadPose | None:
    """`None`, якщо solvePnP не зійшовся (кадр невалідний)."""
    image_pts = np.array([px[i] for i in HEAD_POSE_INDICES], dtype=np.float64)
    k = camera_matrix(frame_w, frame_h)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    try:
        if prev is not None:
            ok, rvec, tvec = cv2.solvePnP(
                FACE_MODEL_3D,
                image_pts,
                k,
                dist_coeffs,
                rvec=prev[0].copy(),
                tvec=prev[1].copy(),
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        else:
            ok, rvec, tvec = cv2.solvePnP(
                FACE_MODEL_3D, image_pts, k, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )
    except cv2.error:
        return None
    if not ok:
        return None

    r, _ = cv2.Rodrigues(rvec)
    yaw, pitch, roll = angles_from_rotation(r)
    return HeadPose(yaw=yaw, pitch=pitch, roll=roll, rvec=rvec, tvec=tvec)


# --- Gaze -----------------------------------------------------------


def eye_gaze(c_l: np.ndarray, c_r: np.ndarray, iris: np.ndarray) -> tuple[float, float] | None:
    """(g_eye, v_eye) для одного ока. `None` для виродженої осі ока."""
    e = c_r - c_l
    w = float(np.linalg.norm(e))
    if w < _EPS:
        return None
    u = e / w
    n = np.array([-u[1], u[0]])  # перпендикуляр, напрямлений вниз у кадрі
    m = (c_l + c_r) / 2.0
    g = float(np.dot(iris - c_l, e)) / (w * w)
    v = float(np.dot(iris - m, n)) / w
    return g, v


def gaze_raw(px: np.ndarray) -> tuple[float, float] | None:
    """Gaze_h = (g_right + g_left)/2, Gaze_v = (v_right + v_left)/2."""
    right = eye_gaze(*(px[i] for i in RIGHT_EYE_GAZE))
    left = eye_gaze(*(px[i] for i in LEFT_EYE_GAZE))
    if right is None or left is None:
        return None
    return (right[0] + left[0]) / 2.0, (right[1] + left[1]) / 2.0


def iris_distance_px(px: np.ndarray) -> float:
    """dist(p468, p473)."""
    return _dist(px[RIGHT_IRIS], px[LEFT_IRIS])


# --- Екстрактор ----------------------------------------------------------------


class FeatureExtractor:
    """landmarks кадру → `FrameFeatures` (сирі значення).

    Тримає попередні rvec/tvec для `useExtrinsicGuess` і скидає їх,
    коли обличчя втрачено або solvePnP не зійшовся.
    """

    def __init__(self) -> None:
        self._prev_pose: tuple[np.ndarray, np.ndarray] | None = None

    def reset(self) -> None:
        self._prev_pose = None

    def extract(
        self, landmarks: np.ndarray | None, ts: float, frame_w: int, frame_h: int
    ) -> FrameFeatures:
        if landmarks is None:
            self.reset()
            return FrameFeatures.invalid(ts)

        px = to_pixels(landmarks, frame_w, frame_h)
        pose = head_pose(px, frame_w, frame_h, self._prev_pose)
        if pose is None:
            self.reset()
            return FrameFeatures.invalid(ts)
        self._prev_pose = (pose.rvec, pose.tvec)

        gaze = gaze_raw(px)
        return FrameFeatures(
            ts=ts,
            valid=True,
            ear=ear(px),
            yaw=pose.yaw,
            pitch=pose.pitch,
            roll=pose.roll,
            gaze_h=gaze[0] if gaze is not None else None,
            gaze_v=gaze[1] if gaze is not None else None,
            iris_distance_px=iris_distance_px(px),
        )
