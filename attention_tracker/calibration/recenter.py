"""Швидке перецентрування на початку сесії: 2 с погляду в центр робочого монітора.

Оновлює baseline Yaw/Pitch і поправку дрейфу точки погляду (off_x, off_y)
тільки для поточної сесії — профіль не змінюється.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from attention_tracker.calibration.fitting import angle_robust_mean, robust_mean
from attention_tracker.calibration.profile import CalibrationProfile
from attention_tracker.config import CalibrationConfig
from attention_tracker.models import FrameFeatures
from attention_tracker.vision.gaze_model import phi_x, phi_y

_SCREEN_CENTER = 0.5


@dataclass(frozen=True)
class SessionAdjustment:
    """Поправки профілю на одну сесію."""

    yaw_baseline: float
    pitch_baseline: float
    off_x: float = 0.0
    off_y: float = 0.0

    @classmethod
    def from_profile(cls, profile: CalibrationProfile) -> SessionAdjustment:
        return cls(profile.yaw_baseline, profile.pitch_baseline)


class RecenterState(Enum):
    COLLECTING = "COLLECTING"
    DONE = "DONE"
    RETRY = "RETRY"  # < min_valid_ratio валідних кадрів


@dataclass(frozen=True)
class RecenterResult:
    adjustment: SessionAdjustment
    large_shift: bool  # положення сильно змінилось — рекомендуємо повне калібрування


class Recenterer:
    def __init__(self, cfg: CalibrationConfig, profile: CalibrationProfile) -> None:
        self._cfg = cfg
        self._profile = profile
        self._t0: float | None = None
        self._total = 0
        self._valid: list[FrameFeatures] = []
        self._state = RecenterState.COLLECTING
        self._result: RecenterResult | None = None

    @property
    def state(self) -> RecenterState:
        return self._state

    @property
    def result(self) -> RecenterResult | None:
        return self._result

    def start(self, now: float) -> None:
        self._t0 = now
        self._total = 0
        self._valid = []
        self._state = RecenterState.COLLECTING
        self._result = None

    def progress(self, now: float) -> float:
        if self._t0 is None:
            return 0.0
        return min(1.0, max(0.0, (now - self._t0) / self._cfg.recenter_duration))

    def feed(self, raw: FrameFeatures) -> RecenterState:
        if self._state is not RecenterState.COLLECTING:
            return self._state
        if self._t0 is None:
            self.start(raw.ts)
        elapsed = raw.ts - self._t0
        if elapsed >= self._cfg.t_settle:
            self._total += 1
            if raw.valid and raw.gaze_h is not None and raw.gaze_v is not None:
                self._valid.append(raw)
        if elapsed >= self._cfg.recenter_duration:
            self._finish()
        return self._state

    def _finish(self) -> None:
        cfg = self._cfg
        if self._total == 0 or len(self._valid) / self._total < cfg.min_valid_ratio:
            self._state = RecenterState.RETRY
            return
        frames = self._valid
        yaw_b = angle_robust_mean([f.yaw for f in frames], cfg.mad_k)
        pitch_b = angle_robust_mean([f.pitch for f in frames], cfg.mad_k)
        gaze_h = robust_mean([f.gaze_h for f in frames], cfg.mad_k)
        gaze_v = robust_mean([f.gaze_v for f in frames], cfg.mad_k)
        sx_c = self._profile.regressor_x.predict(phi_x(gaze_h, 0.0))
        sy_c = self._profile.regressor_y.predict(phi_y(gaze_v, 0.0))
        off_x = _SCREEN_CENTER - sx_c
        off_y = _SCREEN_CENTER - sy_c
        self._result = RecenterResult(
            adjustment=SessionAdjustment(yaw_b, pitch_b, off_x, off_y),
            large_shift=abs(off_x) > cfg.recenter_max_offset or abs(off_y) > cfg.recenter_max_offset,
        )
        self._state = RecenterState.DONE
