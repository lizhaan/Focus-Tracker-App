"""MediaPipe FaceLandmarker, Tasks API."""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

from attention_tracker.config import LandmarkerConfig


class FaceLandmarkerWrapper:
    """`detect(frame_bgr, ts_ms)` → масив (478, 3) нормалізованих точок або `None`.

    Відсутність обличчя — звичайна ситуація (`None`), а не виняток.
    """

    def __init__(self, cfg: LandmarkerConfig) -> None:
        if not cfg.model_path.is_file():
            raise FileNotFoundError(f"Немає моделі FaceLandmarker: {cfg.model_path}")
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(cfg.model_path)),
            running_mode=RunningMode.VIDEO,
            num_faces=cfg.num_faces,
            min_face_detection_confidence=cfg.min_face_detection_confidence,
            min_face_presence_confidence=cfg.min_face_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._last_ts_ms: int | None = None

    def detect(self, frame_bgr: np.ndarray, ts_ms: int) -> np.ndarray | None:
        # VIDEO-режим вимагає строго зростаючих міток часу
        if self._last_ts_ms is not None and ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, ts_ms)
        if not result.face_landmarks:
            return None
        return np.array(
            [(p.x, p.y, p.z) for p in result.face_landmarks[0]], dtype=np.float64
        )

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> FaceLandmarkerWrapper:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
