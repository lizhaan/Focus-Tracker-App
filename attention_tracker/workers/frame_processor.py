"""Обробка одного кадру сесії без Qt: ознаки → фільтр → детектор → машина станів.

Сесія починається з перецентрування (погляд у центр); поки воно триває,
детектор і машина станів не оновлюються. Невдале перецентрування
автоматично починається знову. На перервах і ручній паузі трекінг вимкнено:
детектор працює, машина станів — ні.

Винесено з PipelineWorker, щоб покрити тестами без камери й потоків.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from attention_tracker.calibration.profile import CalibrationProfile
from attention_tracker.calibration.recenter import Recenterer, RecenterResult, RecenterState
from attention_tracker.config import AppConfig
from attention_tracker.focus.state_machine import FocusStateMachine
from attention_tracker.models import FrameFeatures, FrameResult, StateChange
from attention_tracker.vision.distraction import DistractionDetector
from attention_tracker.vision.features import FeatureExtractor
from attention_tracker.vision.filters import SignalFilter


@dataclass(frozen=True)
class ProcessOutput:
    raw: FrameFeatures
    result: FrameResult
    change: StateChange | None
    recenter_event: RecenterState | None = None  # DONE / RETRY на кадрі, де перецентрування завершилось
    recenter_result: RecenterResult | None = None


class FrameProcessor:
    def __init__(self, cfg: AppConfig, profile: CalibrationProfile) -> None:
        self._extractor = FeatureExtractor()
        self._filter = SignalFilter(cfg.filters)
        self._detector = DistractionDetector(cfg.distraction, profile)
        self._recenterer = Recenterer(cfg.calibration, profile)
        self._state_machine = FocusStateMachine(cfg.focus)
        self._recentering = False
        self._tracking = True

    @property
    def state_machine(self) -> FocusStateMachine:
        return self._state_machine

    @property
    def detector(self) -> DistractionDetector:
        return self._detector

    @property
    def recentering(self) -> bool:
        return self._recentering

    def start_session(self, now: float) -> None:
        """Порожні фільтри, перецентрування, потім стан FOCUSED."""
        self._extractor.reset()
        self._filter.reset()
        self._recenterer.start(now)
        self._recentering = True

    @property
    def tracking(self) -> bool:
        return self._tracking

    def set_tracking(self, active: bool, now: float) -> None:
        """Перерва або ручна пауза — машина станів не оновлюється; після — знову FOCUSED."""
        if active and not self._tracking:
            self._detector.reset(now)
            self._state_machine.reset(now)
        self._tracking = active

    def set_timings(self, n: float, y: float) -> None:
        self._state_machine.set_timings(n, y)

    def process(self, landmarks: np.ndarray | None, ts: float, frame_w: int, frame_h: int) -> ProcessOutput:
        return self.process_features(self._extractor.extract(landmarks, ts, frame_w, frame_h))

    def process_features(self, raw: FrameFeatures) -> ProcessOutput:
        smoothed = self._filter.update(raw)
        if self._recentering:
            return self._recenter_step(raw, smoothed)

        distraction = self._detector.update(raw, smoothed)
        if self._detector.consume_filter_reset():
            self._filter.reset()
        if not self._tracking:
            result = FrameResult(features=smoothed, distraction=distraction, state=None, recovering_progress=None)
            return ProcessOutput(raw, result, None)
        change = self._state_machine.update(distraction.distracted, distraction.reason, raw.ts)
        result = FrameResult(
            features=smoothed,
            distraction=distraction,
            state=self._state_machine.state,
            recovering_progress=self._state_machine.recovering_progress(raw.ts),
        )
        return ProcessOutput(raw, result, change)

    def _recenter_step(self, raw: FrameFeatures, smoothed: FrameFeatures) -> ProcessOutput:
        state = self._recenterer.feed(raw)
        result = FrameResult(features=smoothed, distraction=None, state=None, recovering_progress=None)
        if state is RecenterState.DONE:
            recenter = self._recenterer.result
            self._detector.apply_recenter(recenter.adjustment)
            self._detector.reset(raw.ts)
            self._state_machine.reset(raw.ts)
            self._recentering = False
            return ProcessOutput(raw, result, None, state, recenter)
        if state is RecenterState.RETRY:
            self._recenterer.start(raw.ts)
            return ProcessOutput(raw, result, None, state)
        return ProcessOutput(raw, result, None)
