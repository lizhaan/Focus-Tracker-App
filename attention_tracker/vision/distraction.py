"""Детектор відволікання на тимчасових порогах (Етап 2, без калібрування).

Кадр → `distracted` + причина:
- перші `baseline_duration` секунд сесії збирається baseline Yaw, Pitch і Gaze_h;
- кадр невалідний, якщо обличчя немає, solvePnP не зійшовся або
  |ΔYaw| > 45° / |ΔPitch| > 40°; короткий збій (≤ T_noface від останнього
  валідного кадру) тримає попереднє рішення, довгий → FACE_ABSENT / HEAD_AWAY;
- gaze_out = |Gaze_h − gaze_h_baseline| > 0.15 OR |ΔYaw| > 25°.

Валідність перевіряється на сирих значеннях, пороги погляду — на згладжених.
Детекція телефона (PHONE_EYES / PHONE_HEAD) з'явиться разом з калібруванням.
"""

from __future__ import annotations

from attention_tracker.config import AlphaThresholdsConfig, DistractionConfig
from attention_tracker.models import DistractionResult, FrameFeatures, Reason
from attention_tracker.vision.features import wrap


def _angle_mean(values: list[float]) -> float:
    """Середнє кутів без стрибка через ±180°: усереднюємо відхилення від першого."""
    ref = values[0]
    return ref + sum(wrap(v - ref) for v in values) / len(values)


class DistractionDetector:
    def __init__(self, cfg: DistractionConfig, thresholds: AlphaThresholdsConfig) -> None:
        self._cfg = cfg
        self._thr = thresholds
        self.reset(0.0)

    def reset(self, now: float) -> None:
        """Початок сесії: новий збір baseline, попереднє рішення — «не відволікся»."""
        self._baseline_start = now
        self._samples: list[tuple[float, float, float | None]] = []
        self._yaw_baseline: float | None = None
        self._pitch_baseline: float | None = None
        self._gaze_h_baseline: float | None = None
        self._last_valid_ts = now
        self._invalid_long = False
        self._filters_reset_pending = False
        self._last = (False, Reason.NONE)

    @property
    def baseline_ready(self) -> bool:
        return self._yaw_baseline is not None

    @property
    def baseline(self) -> tuple[float, float, float | None] | None:
        """(yaw, pitch, gaze_h) baseline або `None`, поки збирається."""
        if self._yaw_baseline is None:
            return None
        return self._yaw_baseline, self._pitch_baseline, self._gaze_h_baseline

    def consume_filter_reset(self) -> bool:
        """True один раз після початку довгої втрати обличчя: час скинути SignalFilter."""
        pending = self._filters_reset_pending
        self._filters_reset_pending = False
        return pending

    def update(self, raw: FrameFeatures, smoothed: FrameFeatures) -> DistractionResult:
        now = raw.ts
        invalid_reason = self._invalid_reason(raw)

        if invalid_reason is not None:
            if now - self._last_valid_ts <= self._cfg.t_noface:
                return self._result(now, *self._last)  # короткий збій — тримаємо останнє
            if not self._invalid_long:
                self._invalid_long = True
                self._filters_reset_pending = True
            return self._result(now, True, invalid_reason)

        self._last_valid_ts = now
        self._invalid_long = False

        if not self.baseline_ready:
            self._collect_baseline(raw, now)
            return self._result(now, False, Reason.NONE)

        if self._gaze_out(smoothed):
            return self._result(now, True, Reason.GAZE_OUT)
        return self._result(now, False, Reason.NONE)

    # --- внутрішнє ---------------------------------------------------------

    def _invalid_reason(self, raw: FrameFeatures) -> Reason | None:
        if not raw.valid:
            return Reason.FACE_ABSENT
        if self.baseline_ready:
            d_yaw = wrap(raw.yaw - self._yaw_baseline)
            d_pitch = wrap(raw.pitch - self._pitch_baseline)
            if abs(d_yaw) > self._cfg.invalid_yaw_deg or abs(d_pitch) > self._cfg.invalid_pitch_deg:
                return Reason.HEAD_AWAY
        return None

    def _collect_baseline(self, raw: FrameFeatures, now: float) -> None:
        self._samples.append((raw.yaw, raw.pitch, raw.gaze_h))
        if now - self._baseline_start < self._thr.baseline_duration:
            return
        self._yaw_baseline = _angle_mean([s[0] for s in self._samples])
        self._pitch_baseline = _angle_mean([s[1] for s in self._samples])
        gazes = [s[2] for s in self._samples if s[2] is not None]
        self._gaze_h_baseline = sum(gazes) / len(gazes) if gazes else None
        self._samples.clear()

    def _gaze_out(self, s: FrameFeatures) -> bool:
        if s.yaw is not None and abs(wrap(s.yaw - self._yaw_baseline)) > self._thr.yaw_delta_deg:
            return True
        if s.gaze_h is not None and self._gaze_h_baseline is not None:
            return abs(s.gaze_h - self._gaze_h_baseline) > self._thr.gaze_h_delta
        return False

    def _result(self, now: float, distracted: bool, reason: Reason) -> DistractionResult:
        self._last = (distracted, reason)
        return DistractionResult(ts=now, distracted=distracted, reason=reason, screen_point=None)
