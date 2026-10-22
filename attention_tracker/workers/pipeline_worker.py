"""PipelineWorker (QThread): камера → landmarker → ознаки / FrameProcessor → Qt-сигнали.

Один запуск потоку — одна сесія або одне калібрування: камера вмикається в run() і звільняється
після stop(). Назовні — тільки Qt-сигнали; у БД нічого не пише. Кадри
передаються лише в памʼяті для превʼю і ніде не зберігаються.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum

import numpy as np
from PySide6.QtCore import QThread, Signal

from attention_tracker.calibration.profile import CalibrationProfile
from attention_tracker.calibration.recenter import RecenterState
from attention_tracker.capture.camera import Camera, CameraError
from attention_tracker.capture.landmarker import FaceLandmarkerWrapper
from attention_tracker.config import AppConfig
from attention_tracker.models import FrameFeatures
from attention_tracker.vision.features import FeatureExtractor
from attention_tracker.workers.frame_processor import FrameProcessor, ProcessOutput

_MS_PER_S = 1000.0


@dataclass(frozen=True)
class DebugFrame:
    """Payload сигналу debug_frame: кадр, точки і сирі ознаки."""

    frame_bgr: np.ndarray
    landmarks: np.ndarray | None
    raw: FrameFeatures


class WorkerMode(Enum):
    CALIBRATION = "CALIBRATION"  # тільки сирі ознаки для майстра калібрування
    SESSION = "SESSION"  # повний пайплайн з детектором і машиною станів


class PipelineWorker(QThread):
    camera_opened = Signal(str, int, int)  # camera_id, ширина, висота кадру
    frame_processed = Signal(object)  # FrameResult (SESSION)
    state_changed = Signal(object)  # StateChange (SESSION)
    recenter_done = Signal(object)  # RecenterResult (SESSION)
    recenter_retry = Signal()  # перецентрування починається знову (SESSION)
    calibration_sample = Signal(object)  # сирі FrameFeatures (CALIBRATION)
    debug_frame = Signal(object)  # DebugFrame
    camera_error = Signal(str)
    fps_updated = Signal(float)

    def __init__(
        self,
        cfg: AppConfig,
        mode: WorkerMode = WorkerMode.SESSION,
        profile: CalibrationProfile | None = None,
        parent: object | None = None,
    ) -> None:
        super().__init__(parent)
        if mode is WorkerMode.SESSION and profile is None:
            raise ValueError("для сесії потрібен профіль калібрування")
        self._cfg = cfg
        self._mode = mode
        self._profile = profile
        self._running = False
        self._lock = threading.Lock()
        self._pending_timings: tuple[float, float] | None = None

    def stop(self) -> None:
        self._running = False

    def set_timings(self, n: float, y: float) -> None:
        """Можна викликати з UI-потоку; застосовується з наступного кадру."""
        focus = self._cfg.focus
        if not (focus.n_min <= n <= focus.n_max and focus.y_min <= y <= focus.y_max):
            raise ValueError(f"n={n}, y={y} поза допустимими межами")  # не валимо потік
        with self._lock:
            self._pending_timings = (n, y)

    def run(self) -> None:
        camera = Camera(self._cfg.camera)
        try:
            camera.open()
            landmarker = FaceLandmarkerWrapper(self._cfg.landmarker)
        except (CameraError, FileNotFoundError) as exc:
            camera.release()
            self.camera_error.emit(str(exc))
            return

        w, h = camera.frame_size
        self.camera_opened.emit(camera.camera_id, w, h)
        extractor = FeatureExtractor()
        processor: FrameProcessor | None = None
        if self._mode is WorkerMode.SESSION:
            processor = FrameProcessor(self._cfg, self._profile)
            processor.start_session(time.monotonic())
        fps_window_start: float | None = None
        fps_frames = 0
        self._running = True
        try:
            while self._running:
                self._apply_pending_timings(processor)
                item = camera.read()
                if item is None:
                    self.camera_error.emit("Не вдалося отримати кадр з камери")
                    break
                frame, ts = item
                h, w = frame.shape[:2]

                landmarks = landmarker.detect(frame, int(ts * _MS_PER_S))
                if processor is None:
                    raw = extractor.extract(landmarks, ts, w, h)
                    self.calibration_sample.emit(raw)  # сирі, без згладжування
                else:
                    out = processor.process(landmarks, ts, w, h)
                    raw = out.raw
                    self._emit_session(out)
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

    def _emit_session(self, out: ProcessOutput) -> None:
        if out.recenter_event is RecenterState.DONE:
            self.recenter_done.emit(out.recenter_result)
        elif out.recenter_event is RecenterState.RETRY:
            self.recenter_retry.emit()
        self.frame_processed.emit(out.result)
        if out.change is not None:
            self.state_changed.emit(out.change)

    def _apply_pending_timings(self, processor: FrameProcessor | None) -> None:
        with self._lock:
            timings, self._pending_timings = self._pending_timings, None
        if timings is not None and processor is not None:
            processor.set_timings(*timings)
