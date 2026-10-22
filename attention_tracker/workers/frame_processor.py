"""Обробка одного кадру сесії без Qt: ознаки → фільтр → детектор → машина станів.

Винесено з PipelineWorker, щоб покрити тестами без камери й потоків.
"""

from __future__ import annotations

import numpy as np

from attention_tracker.config import AppConfig
from attention_tracker.focus.state_machine import FocusStateMachine
from attention_tracker.models import FrameFeatures, FrameResult, StateChange
from attention_tracker.vision.distraction import DistractionDetector
from attention_tracker.vision.features import FeatureExtractor
from attention_tracker.vision.filters import SignalFilter


class FrameProcessor:
    def __init__(self, cfg: AppConfig) -> None:
        self._extractor = FeatureExtractor()
        self._filter = SignalFilter(cfg.filters)
        self._detector = DistractionDetector(cfg.distraction, cfg.alpha_thresholds)
        self._state_machine = FocusStateMachine(cfg.focus)

    @property
    def state_machine(self) -> FocusStateMachine:
        return self._state_machine

    @property
    def baseline_ready(self) -> bool:
        return self._detector.baseline_ready

    def start_session(self, now: float) -> None:
        """Новий baseline, порожні фільтри, стан FOCUSED."""
        self._extractor.reset()
        self._filter.reset()
        self._detector.reset(now)
        self._state_machine.reset(now)

    def set_timings(self, n: float, y: float) -> None:
        self._state_machine.set_timings(n, y)

    def process(
        self, landmarks: np.ndarray | None, ts: float, frame_w: int, frame_h: int
    ) -> tuple[FrameFeatures, FrameResult, StateChange | None]:
        """Кадр → (сирі ознаки, результат для UI, перехід стану або None)."""
        raw = self._extractor.extract(landmarks, ts, frame_w, frame_h)
        result, change = self.process_features(raw)
        return raw, result, change

    def process_features(self, raw: FrameFeatures) -> tuple[FrameResult, StateChange | None]:
        smoothed = self._filter.update(raw)
        distraction = self._detector.update(raw, smoothed)
        if self._detector.consume_filter_reset():
            self._filter.reset()
        change = self._state_machine.update(distraction.distracted, distraction.reason, raw.ts)
        result = FrameResult(
            features=smoothed,
            distraction=distraction,
            state=self._state_machine.state,
            recovering_progress=self._state_machine.recovering_progress(raw.ts),
        )
        return result, change
