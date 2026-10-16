"""PipelineWorker (QThread): камера → landmarker → features → filter (PROJECT_PLAN §5.2–5.3).

Етап 2, крок 1: тільки ознаки для debug_view. Детектор і машина станів
додаються в кроках 3–4. Назовні — тільки Qt-сигнали; у БД нічого не пише.
Кадри передаються лише в памʼяті для превʼю і ніде не зберігаються.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QThread, Signal

from attention_tracker.capture.camera import Camera, CameraError
from attention_tracker.capture.landmarker import FaceLandmarkerWrapper
from attention_tracker.config import AppConfig
from attention_tracker.models import FrameFeatures, FrameResult
from attention_tracker.vision.features import FeatureExtractor
from attention_tracker.vision.filters import SignalFilter

_MS_PER_S = 1000.0


@dataclass(frozen=True)
class DebugFrame:
    """Payload сигналу debug_frame (FR-11): кадр, точки і сирі ознаки."""

    frame_bgr: np.ndarray
    landmarks: np.ndarray | None
    raw: FrameFeatures


class PipelineWorker(QThread):
    frame_processed = Signal(object)  # FrameResult (ознаки після SignalFilter)
    debug_frame = Signal(object)  # DebugFrame
    camera_error = Signal(str)
    fps_updated = Signal(float)

    def __init__(self, cfg: AppConfig, parent: object | None = None) -> None:
        super().__init__(parent)
        self._cfg = cfg
        self._running = False

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        camera = Camera(self._cfg.camera)
        try:
            camera.open()
            landmarker = FaceLandmarkerWrapper(self._cfg.landmarker)
        except (CameraError, FileNotFoundError) as exc:
            camera.release()
            self.camera_error.emit(str(exc))
            return

        extractor = FeatureExtractor()
        signal_filter = SignalFilter(self._cfg.filters)
        fps_window_start: float | None = None
        fps_frames = 0
        self._running = True
        try:
            while self._running:
                item = camera.read()
                if item is None:
                    self.camera_error.emit("Не вдалося отримати кадр з камери")
                    break
                frame, ts = item
                h, w = frame.shape[:2]

                landmarks = landmarker.detect(frame, int(ts * _MS_PER_S))
                raw = extractor.extract(landmarks, ts, w, h)
                smoothed = signal_filter.update(raw)
                self.frame_processed.emit(FrameResult(smoothed, None, None, None))
                self.debug_frame.emit(DebugFrame(frame, landmarks, raw))

                if fps_window_start is None:
                    fps_window_start = ts
                    continue
                fps_frames += 1
                if ts - fps_window_start >= self._cfg.ui.fps_report_interval:
                    self.fps_updated.emit(fps_frames / (ts - fps_window_start))
                    fps_window_start, fps_frames = ts, 0
        finally:
            landmarker.close()
            camera.release()
