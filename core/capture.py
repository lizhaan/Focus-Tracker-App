from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import mediapipe as mp
import numpy as np

from core.features import FrameMetrics, MetricsSmoother, compute_frame_metrics

FrameCallback = Callable[["CaptureFrame"], None]


@dataclass
class CaptureFrame:

    timestamp: float
    frame_width: int
    frame_height: int
    landmarks: Optional[np.ndarray]
    metrics: Optional[FrameMetrics]


class FaceMeshCapture:

    def __init__(
            self,
            on_frame: Optional[FrameCallback] = None,
            camera_index: int = 0,
            max_num_faces: int = 1,
            min_detection_confidence: float = 0.5,
            min_tracking_confidence: float = 0.5,
            smoother_window: int = 3,
            smoother_alpha: float = 0.3,
    ) -> None:
        self.on_frame = on_frame
        self.camera_index = camera_index

        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self._smoother = MetricsSmoother(window_size=smoother_window, alpha=smoother_alpha)
        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False


    def start(self) -> None:
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open the camera #{self.camera_index}")
        self._running = True
        self.run()

    def stop(self) -> None:
        self._running = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def reset_smoothing(self) -> None:
        self._smoother.reset()


    def run(self) -> None:
        assert self._cap is not None, "First start()"

        while self._running:
            ok, frame_bgr = self._cap.read()
            if not ok:
                break

            capture_frame = self.process_frame(frame_bgr)
            if self.on_frame is not None:
                self.on_frame(capture_frame)

    def process_frame(self, frame_bgr: np.ndarray) -> CaptureFrame:
        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        results = self._face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return CaptureFrame(
                timestamp=time.time(),
                frame_width=width,
                frame_height=height,
                landmarks=None,
                metrics=None,
            )

        face_landmarks = results.multi_face_landmarks[0]
        landmarks = np.array(
            [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark],
            dtype=np.float64,
        )

        metrics = compute_frame_metrics(landmarks, width, height, self._smoother)

        return CaptureFrame(
            timestamp=time.time(),
            frame_width=width,
            frame_height=height,
            landmarks=landmarks,
            metrics=metrics,
        )

    def close(self) -> None:
        self.stop()
        self._face_mesh.close()
