"""Детектор відволікання на профілі калібрування.

Кадр → `distracted` + причина:
- кадр невалідний, якщо обличчя немає, solvePnP не зійшовся або
  |ΔYaw| > 45° / |ΔPitch| > 40°; короткий збій (≤ T_noface від останнього
  валідного кадру) тримає попереднє рішення, довгий → FACE_ABSENT / HEAD_AWAY
  і скидання фільтрів та гістерезису;
- точка погляду (sx, sy) — регресія з профілю + поправка перецентрування;
  під час моргання (EAR < ear_freeze) Gaze_h/Gaze_v заморожуються;
- gaze_out — точка поза робочим монітором з гістерезисом ε_out / ε_in;
- телефон: очі прикриті (EAR < ear_threshold) або голова вниз
  (pitch_down > pitch_threshold) безперервно ≥ T_blink.
Пріоритет причин: FACE_ABSENT / HEAD_AWAY > PHONE_HEAD > PHONE_EYES > GAZE_OUT.

Валідність перевіряється на сирих значеннях, решта — на згладжених.
"""

from __future__ import annotations

from attention_tracker.calibration.profile import CalibrationProfile
from attention_tracker.calibration.recenter import SessionAdjustment
from attention_tracker.config import DistractionConfig
from attention_tracker.models import DistractionResult, FrameFeatures, Reason
from attention_tracker.vision.features import wrap
from attention_tracker.vision.gaze_model import phi_x, phi_y


class _HeldCondition:
    """Умова, що має триматися безперервно щонайменше `duration` секунд."""

    def __init__(self, duration: float) -> None:
        self._duration = duration
        self._since: float | None = None

    def reset(self) -> None:
        self._since = None

    def update(self, condition: bool, now: float) -> bool:
        if not condition:
            self._since = None
            return False
        if self._since is None:
            self._since = now
        return now - self._since >= self._duration


class DistractionDetector:
    def __init__(self, cfg: DistractionConfig, profile: CalibrationProfile) -> None:
        self._cfg = cfg
        self._profile = profile
        self._adj = SessionAdjustment.from_profile(profile)
        self.reset(0.0)

    def reset(self, now: float) -> None:
        """Початок сесії: попереднє рішення — «не відволікся», погляд на екрані."""
        self._last_valid_ts = now
        self._invalid_long = False
        self._filters_reset_pending = False
        self._last = (False, Reason.NONE)
        self._on_screen = True
        self._eyes_low = _HeldCondition(self._cfg.t_blink)
        self._head_down = _HeldCondition(self._cfg.t_blink)
        self._frozen_gaze: tuple[float, float] | None = None

    def apply_recenter(self, adjustment: SessionAdjustment) -> None:
        """Поправки з перецентрування на цю сесію."""
        self._adj = adjustment

    @property
    def adjustment(self) -> SessionAdjustment:
        return self._adj

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
                return self._result(now, *self._last, None)  # короткий збій — тримаємо останнє
            if not self._invalid_long:
                self._invalid_long = True
                self._filters_reset_pending = True
                self._on_screen = True  # гістерезис скидається
                self._eyes_low.reset()
                self._head_down.reset()
                self._frozen_gaze = None
            return self._result(now, True, invalid_reason, None)

        self._last_valid_ts = now
        self._invalid_long = False

        d_yaw = wrap(smoothed.yaw - self._adj.yaw_baseline)
        d_pitch = wrap(smoothed.pitch - self._adj.pitch_baseline)
        point = self._screen_point(smoothed, d_yaw, d_pitch)
        gaze_out = point is not None and not self._update_on_screen(point)
        pitch_down = self._profile.s_pitch * d_pitch
        eyes_low = (
            self._profile.ear_enabled
            and smoothed.ear is not None
            and smoothed.ear < self._profile.ear_threshold
        )
        phone_head = self._head_down.update(pitch_down > self._profile.pitch_threshold, now)
        phone_eyes = self._eyes_low.update(eyes_low, now)
        head = (d_yaw, d_pitch, pitch_down)

        if phone_head:
            return self._result(now, True, Reason.PHONE_HEAD, point, head)
        if phone_eyes:
            return self._result(now, True, Reason.PHONE_EYES, point, head)
        if gaze_out:
            return self._result(now, True, Reason.GAZE_OUT, point, head)
        return self._result(now, False, Reason.NONE, point, head)

    # --- внутрішнє ---------------------------------------------------------

    def _invalid_reason(self, raw: FrameFeatures) -> Reason | None:
        if not raw.valid:
            return Reason.FACE_ABSENT
        d_yaw = wrap(raw.yaw - self._adj.yaw_baseline)
        d_pitch = wrap(raw.pitch - self._adj.pitch_baseline)
        if abs(d_yaw) > self._cfg.invalid_yaw_deg or abs(d_pitch) > self._cfg.invalid_pitch_deg:
            return Reason.HEAD_AWAY
        return None

    def _gaze(self, s: FrameFeatures) -> tuple[float, float] | None:
        """Gaze_h, Gaze_v з заморожуванням під час моргання."""
        blinking = s.ear is not None and s.ear < self._profile.ear_freeze
        if s.gaze_h is not None and s.gaze_v is not None and not blinking:
            self._frozen_gaze = (s.gaze_h, s.gaze_v)
            return self._frozen_gaze
        if self._frozen_gaze is not None:
            return self._frozen_gaze
        if s.gaze_h is not None and s.gaze_v is not None:
            return s.gaze_h, s.gaze_v  # моргання з першого кадру — заморожувати нічого
        return None

    def _screen_point(self, s: FrameFeatures, d_yaw: float, d_pitch: float) -> tuple[float, float] | None:
        gaze = self._gaze(s)
        if gaze is None:
            return None
        sx = self._profile.regressor_x.predict(phi_x(gaze[0], d_yaw)) + self._adj.off_x
        sy = self._profile.regressor_y.predict(phi_y(gaze[1], d_pitch)) + self._adj.off_y
        return sx, sy

    def _update_on_screen(self, point: tuple[float, float]) -> bool:
        eps = self._cfg.eps_out if self._on_screen else self._cfg.eps_in
        sx, sy = point
        self._on_screen = -eps <= sx <= 1.0 + eps and -eps <= sy <= 1.0 + eps
        return self._on_screen

    def _result(
        self,
        now: float,
        distracted: bool,
        reason: Reason,
        point: tuple[float, float] | None,
        head: tuple[float, float, float] | None = None,
    ) -> DistractionResult:
        self._last = (distracted, reason)
        d_yaw, d_pitch, pitch_down = head if head is not None else (None, None, None)
        return DistractionResult(
            ts=now, distracted=distracted, reason=reason, screen_point=point,
            d_yaw=d_yaw, d_pitch=d_pitch, pitch_down=pitch_down,
        )
