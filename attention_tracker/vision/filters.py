"""Згладжування сигналів: median + EMA (MATH §5)."""

from __future__ import annotations

import dataclasses
from collections import deque

import numpy as np

from attention_tracker.config import FilterConfig
from attention_tracker.models import FrameFeatures


class MedianEmaFilter:
    """MATH §5: x_med = median(останні k), x̂ = α·x_med + (1−α)·x̂(t−1).

    Перший вихід x̂ = x_med(перший).
    """

    def __init__(self, window: int, alpha: float) -> None:
        if window < 1:
            raise ValueError("window (k) має бути ≥ 1")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha має бути в діапазоні (0, 1]")
        self._alpha = alpha
        self._window: deque[float] = deque(maxlen=window)
        self._value: float | None = None

    def update(self, x: float) -> float:
        self._window.append(x)
        x_med = float(np.median(self._window))
        if self._value is None:
            self._value = x_med
        else:
            self._value = self._alpha * x_med + (1.0 - self._alpha) * self._value
        return self._value

    def reset(self) -> None:
        self._window.clear()
        self._value = None


class SignalFilter:
    """MATH §5: згладжує EAR, Yaw, Pitch, Gaze_h, Gaze_v у `FrameFeatures`.

    Yaw/Pitch згладжуються сирими: median і EMA лінійні, тому
    згладити Yaw і потім відняти baseline — те саме, що згладити ΔYaw.
    Відрізнялося б тільки біля ±180°, але такі кадри невалідні (§4.1).

    Невалідні кадри повертаються без змін і не оновлюють фільтри. Скидання
    після `T_noface` (§4.1) викликає детектор через `reset()`.
    """

    FIELDS = ("ear", "yaw", "pitch", "gaze_h", "gaze_v")

    def __init__(self, cfg: FilterConfig) -> None:
        self._filters = {
            name: MedianEmaFilter(cfg.median_window, cfg.ema_alpha) for name in self.FIELDS
        }

    def update(self, features: FrameFeatures) -> FrameFeatures:
        if not features.valid:
            return features
        smoothed: dict[str, float] = {}
        for name, flt in self._filters.items():
            value = getattr(features, name)
            if value is not None:
                smoothed[name] = flt.update(value)
        return dataclasses.replace(features, **smoothed)

    def reset(self) -> None:
        for flt in self._filters.values():
            flt.reset()
