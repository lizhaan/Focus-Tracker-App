"""Процедура калібрування без Qt: кроки, збір кадрів, робастні середні, перевірки.

UI показує ціль поточного кроку, викликає `start_step(now)` після відліку і
передає кожен сирий кадр у `feed()`. Час — мітки кадрів, не кількість кадрів.

Кроки: перевірка перед стартом → центр → сітка (8) → повороти голови (12) →
низ екрана → телефон → перевірка якості (4 випадкові точки).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from attention_tracker.calibration.fitting import (
    Quality,
    RegressionSample,
    angle_robust_mean,
    Thresholds,
    compute_thresholds,
    fit_regressors,
    quality_of,
    robust_mean,
    s_pitch_from,
)
from attention_tracker.calibration.profile import CalibrationProfile
from attention_tracker.config import CalibrationConfig
from attention_tracker.models import FrameFeatures
from attention_tracker.vision.features import wrap
from attention_tracker.vision.gaze_model import GazeRegressor, phi_x, phi_y


class StepKind(Enum):
    PRECHECK = "PRECHECK"
    CENTER = "CENTER"
    GRID = "GRID"
    HEAD = "HEAD"
    BOTTOM = "BOTTOM"
    PHONE = "PHONE"
    CHECK = "CHECK"


class HeadPose(Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UP = "UP"
    DOWN = "DOWN"


class StepState(Enum):
    READY = "READY"  # крок ще не запущено (UI показує інструкцію/відлік)
    WAITING_POSE = "WAITING_POSE"  # поворот голови: чекаємо кут у діапазоні
    COLLECTING = "COLLECTING"
    DONE = "DONE"
    RETRY = "RETRY"  # крок треба повторити, причина — `retry_reason`


class RetryReason(Enum):
    LOW_VALID = "LOW_VALID"  # < 70 % валідних кадрів: світло, відстань, не рухатись
    FACE_RATIO = "FACE_RATIO"  # обличчя видно рідше ніж на 90 % кадрів
    TOO_FAR = "TOO_FAR"  # сядьте ближче
    TOO_CLOSE = "TOO_CLOSE"  # сядьте далі
    LOW_FPS = "LOW_FPS"
    PITCH_TOO_SMALL = "PITCH_TOO_SMALL"  # нахил вниз замалий


@dataclass(frozen=True)
class CalibrationStep:
    kind: StepKind
    target: tuple[float, float] | None  # ціль погляду на робочому моніторі
    pose: HeadPose | None = None


@dataclass
class _Collected:
    """Кадри поточного кроку після T_settle."""

    total: int = 0
    valid: list[FrameFeatures] = field(default_factory=list)
    first_ts: float | None = None
    last_ts: float | None = None


class CalibrationProcedure:
    def __init__(
        self,
        cfg: CalibrationConfig,
        ear_freeze_factor: float,
        frame_w: int,
        frame_h: int,
        rng: random.Random | None = None,
    ) -> None:
        self._cfg = cfg
        self._ear_freeze_factor = ear_freeze_factor
        self._frame_w = frame_w
        self._frame_h = frame_h
        self.steps: list[CalibrationStep] = self._build_steps(rng or random.Random())
        self._index = 0
        self._state = StepState.READY
        self._retry_reason: RetryReason | None = None
        self._t0: float | None = None
        self._collected = _Collected()

        self._yaw_baseline: float | None = None
        self._pitch_baseline: float | None = None
        self._ear_center: float | None = None
        self._yaw_sign_left: int | None = None
        self._pitch_sign_up: int | None = None
        self._s_pitch: int | None = None
        self._ear_bottom: float | None = None
        self._pd_bottom: float | None = None
        self._ear_phone: float | None = None
        self._pd_phone: float | None = None
        self._samples: list[RegressionSample] = []
        self._thresholds: Thresholds | None = None
        self._regressors: tuple[GazeRegressor, GazeRegressor] | None = None
        self._check_errors: list[float] = []

    # --- побудова кроків ------------------------------------------------------

    def _build_steps(self, rng: random.Random) -> list[CalibrationStep]:
        cfg = self._cfg
        steps = [CalibrationStep(StepKind.PRECHECK, None), CalibrationStep(StepKind.CENTER, cfg.center_target)]
        steps += [CalibrationStep(StepKind.GRID, t) for t in cfg.grid_targets]
        for pose, targets in (
            (HeadPose.LEFT, cfg.head_yaw_targets),
            (HeadPose.RIGHT, cfg.head_yaw_targets),
            (HeadPose.UP, cfg.head_pitch_targets),
            (HeadPose.DOWN, cfg.head_pitch_targets),
        ):
            steps += [CalibrationStep(StepKind.HEAD, t, pose) for t in targets]
        steps.append(CalibrationStep(StepKind.BOTTOM, cfg.bottom_target))
        steps.append(CalibrationStep(StepKind.PHONE, None))
        steps += [CalibrationStep(StepKind.CHECK, t) for t in self._check_targets(rng)]
        return steps

    def _check_targets(self, rng: random.Random) -> list[tuple[float, float]]:
        """Випадкові точки, не ближче check_point_min_dist до цілей навчання."""
        cfg = self._cfg
        training = [cfg.center_target, *cfg.grid_targets, *cfg.head_yaw_targets, *cfg.head_pitch_targets]
        lo, hi = cfg.check_point_range
        points: list[tuple[float, float]] = []
        while len(points) < cfg.n_check_points:
            p = (rng.uniform(lo, hi), rng.uniform(lo, hi))
            if all(math.dist(p, t) >= cfg.check_point_min_dist for t in training + points):
                points.append(p)
        return points

    # --- стан ------------------------------------------------------------------

    @property
    def current(self) -> CalibrationStep | None:
        return self.steps[self._index] if self._index < len(self.steps) else None

    @property
    def index(self) -> int:
        return self._index

    @property
    def finished(self) -> bool:
        return self._index >= len(self.steps)

    @property
    def state(self) -> StepState:
        return self._state

    @property
    def retry_reason(self) -> RetryReason | None:
        return self._retry_reason

    @property
    def calib_error(self) -> float | None:
        if not self._check_errors or len(self._check_errors) < self._cfg.n_check_points:
            return None
        return float(np.mean(self._check_errors))

    @property
    def quality(self) -> Quality | None:
        error = self.calib_error
        return None if error is None else quality_of(error, self._cfg)

    def progress(self, now: float) -> float | None:
        """Частка часу збору поточного кроку (0..1) для анімації цілі."""
        if self._state is not StepState.COLLECTING or self._t0 is None:
            return None
        return min(1.0, max(0.0, (now - self._t0) / self._duration()))

    # --- керування кроком ------------------------------------------------------

    def start_step(self, now: float) -> None:
        """Почати (або повторити після RETRY) поточний крок."""
        step = self.current
        if step is None:
            raise RuntimeError("калібрування вже завершено")
        self._collected = _Collected()
        self._retry_reason = None
        if step.kind is StepKind.HEAD:
            self._state = StepState.WAITING_POSE
            self._t0 = None
        else:
            self._state = StepState.COLLECTING
            self._t0 = now

    def next(self) -> None:
        """Перейти до наступного кроку після DONE."""
        if self._state is not StepState.DONE:
            raise RuntimeError(f"крок не завершено: {self._state}")
        self._index += 1
        self._state = StepState.READY

    def pose_ok(self, raw: FrameFeatures) -> bool | None:
        """Для кроку повороту голови: чи кут у цільовому діапазоні; інакше None."""
        step = self.current
        if step is None or step.kind is not StepKind.HEAD:
            return None
        if not raw.valid or self._yaw_baseline is None:
            return False
        if step.pose in (HeadPose.LEFT, HeadPose.RIGHT):
            delta = wrap(raw.yaw - self._yaw_baseline)
            lo, hi = self._cfg.head_yaw_range_deg
            first_sign = self._yaw_sign_left
            needs_opposite = step.pose is HeadPose.RIGHT
        else:
            delta = wrap(raw.pitch - self._pitch_baseline)
            lo, hi = self._cfg.head_pitch_range_deg
            first_sign = self._pitch_sign_up
            needs_opposite = step.pose is HeadPose.DOWN
        if not lo <= abs(delta) <= hi:
            return False
        if first_sign is None:
            return True
        sign = 1 if delta > 0 else -1
        return sign == (-first_sign if needs_opposite else first_sign)

    def feed(self, raw: FrameFeatures) -> StepState:
        step = self.current
        if step is None or self._state not in (StepState.WAITING_POSE, StepState.COLLECTING):
            return self._state

        pose_ok = self.pose_ok(raw)
        if self._state is StepState.WAITING_POSE:
            if not pose_ok:
                return self._state
            self._state = StepState.COLLECTING
            self._t0 = raw.ts

        elapsed = raw.ts - self._t0
        if step.kind is StepKind.PRECHECK or elapsed >= self._cfg.t_settle:
            self._record(raw, raw.valid and pose_ok is not False)
        if elapsed >= self._duration():
            self._finish_step()
        return self._state

    def _duration(self) -> float:
        step = self.current
        if step is not None and step.kind is StepKind.PRECHECK:
            return self._cfg.precheck_duration
        return self._cfg.t_calib

    def _record(self, raw: FrameFeatures, valid: bool) -> None:
        c = self._collected
        c.total += 1
        if valid:
            c.valid.append(raw)
        if c.first_ts is None:
            c.first_ts = raw.ts
        c.last_ts = raw.ts

    def _retry(self, reason: RetryReason) -> None:
        self._state = StepState.RETRY
        self._retry_reason = reason

    # --- завершення кроку ------------------------------------------------------

    def _finish_step(self) -> None:
        step = self.current
        if step.kind is StepKind.PRECHECK:
            self._finish_precheck()
            return

        c = self._collected
        if c.total == 0 or len(c.valid) / c.total < self._cfg.min_valid_ratio:
            self._retry(RetryReason.LOW_VALID)
            return

        handler = {
            StepKind.CENTER: self._finish_center,
            StepKind.GRID: self._finish_target,
            StepKind.HEAD: self._finish_head,
            StepKind.BOTTOM: self._finish_bottom,
            StepKind.PHONE: self._finish_phone,
            StepKind.CHECK: self._finish_check,
        }[step.kind]
        if handler(step, c.valid) is not False:
            self._state = StepState.DONE

    def _finish_precheck(self) -> None:
        cfg = self._cfg
        c = self._collected
        if c.total == 0 or len(c.valid) / c.total < cfg.precheck_face_ratio:
            self._retry(RetryReason.FACE_RATIO)
            return
        iris = robust_mean([f.iris_distance_px for f in c.valid], cfg.mad_k) / self._frame_w
        if iris < cfg.iris_distance_min_frac:
            self._retry(RetryReason.TOO_FAR)
            return
        if iris > cfg.iris_distance_max_frac:
            self._retry(RetryReason.TOO_CLOSE)
            return
        span = c.last_ts - c.first_ts
        fps = (c.total - 1) / span if span > 0 else 0.0
        if fps < cfg.min_fps:
            self._retry(RetryReason.LOW_FPS)
            return
        self._state = StepState.DONE

    def _mean(self, frames: list[FrameFeatures], name: str) -> float:
        return robust_mean([getattr(f, name) for f in frames if getattr(f, name) is not None], self._cfg.mad_k)

    def _deltas(self, frames: list[FrameFeatures]) -> tuple[float, float]:
        d_yaw = robust_mean([wrap(f.yaw - self._yaw_baseline) for f in frames], self._cfg.mad_k)
        d_pitch = robust_mean([wrap(f.pitch - self._pitch_baseline) for f in frames], self._cfg.mad_k)
        return d_yaw, d_pitch

    def _finish_center(self, step: CalibrationStep, frames: list[FrameFeatures]) -> None:
        self._yaw_baseline = angle_robust_mean([f.yaw for f in frames], self._cfg.mad_k)
        self._pitch_baseline = angle_robust_mean([f.pitch for f in frames], self._cfg.mad_k)
        self._ear_center = self._mean(frames, "ear")
        self._add_sample(step, frames, d_yaw=0.0, d_pitch=0.0)

    def _finish_target(self, step: CalibrationStep, frames: list[FrameFeatures]) -> None:
        d_yaw, d_pitch = self._deltas(frames)
        self._add_sample(step, frames, d_yaw, d_pitch)

    def _finish_head(self, step: CalibrationStep, frames: list[FrameFeatures]) -> bool:
        d_yaw, d_pitch = self._deltas(frames)
        if step.pose is HeadPose.DOWN:
            s_pitch = s_pitch_from(d_pitch, self._cfg.s_pitch_min_deg)
            if s_pitch is None:
                self._retry(RetryReason.PITCH_TOO_SMALL)
                return False
            if self._s_pitch is None:
                self._s_pitch = s_pitch
        if step.pose is HeadPose.LEFT and self._yaw_sign_left is None:
            self._yaw_sign_left = 1 if d_yaw > 0 else -1
        if step.pose is HeadPose.UP and self._pitch_sign_up is None:
            self._pitch_sign_up = 1 if d_pitch > 0 else -1
        self._add_sample(step, frames, d_yaw, d_pitch)
        return True

    def _finish_bottom(self, step: CalibrationStep, frames: list[FrameFeatures]) -> None:
        self._ear_bottom = self._mean(frames, "ear")
        self._pd_bottom = self._pitch_down(frames)

    def _finish_phone(self, step: CalibrationStep, frames: list[FrameFeatures]) -> None:
        self._ear_phone = self._mean(frames, "ear")
        self._pd_phone = self._pitch_down(frames)
        self._thresholds = compute_thresholds(
            self._ear_bottom, self._ear_phone, self._pd_bottom, self._pd_phone,
            self._cfg, self._ear_freeze_factor,
        )
        self._regressors = fit_regressors(self._samples, self._cfg)

    def _finish_check(self, step: CalibrationStep, frames: list[FrameFeatures]) -> None:
        d_yaw, d_pitch = self._deltas(frames)
        reg_x, reg_y = self._regressors
        sx = reg_x.predict(phi_x(self._mean(frames, "gaze_h"), d_yaw))
        sy = reg_y.predict(phi_y(self._mean(frames, "gaze_v"), d_pitch))
        self._check_errors.append(math.dist((sx, sy), step.target))

    def _pitch_down(self, frames: list[FrameFeatures]) -> float:
        return robust_mean(
            [self._s_pitch * wrap(f.pitch - self._pitch_baseline) for f in frames], self._cfg.mad_k
        )

    def _add_sample(
        self, step: CalibrationStep, frames: list[FrameFeatures], d_yaw: float, d_pitch: float
    ) -> None:
        self._samples.append(
            RegressionSample(
                gaze_h=self._mean(frames, "gaze_h"),
                gaze_v=self._mean(frames, "gaze_v"),
                d_yaw=d_yaw,
                d_pitch=d_pitch,
                sx=step.target[0],
                sy=step.target[1],
            )
        )

    # --- результат -------------------------------------------------------------

    def result(self, user_id: int, monitor_id: str, camera_id: str, created_at: str) -> CalibrationProfile:
        if not self.finished:
            raise RuntimeError("калібрування не завершено")
        reg_x, reg_y = self._regressors
        t = self._thresholds
        return CalibrationProfile(
            user_id=user_id,
            monitor_id=monitor_id,
            camera_id=camera_id,
            frame_w=self._frame_w,
            frame_h=self._frame_h,
            yaw_baseline=self._yaw_baseline,
            pitch_baseline=self._pitch_baseline,
            ear_center=self._ear_center,
            s_pitch=self._s_pitch,
            ear_bottom=self._ear_bottom,
            pd_bottom=self._pd_bottom,
            ear_phone=self._ear_phone,
            pd_phone=self._pd_phone,
            ear_enabled=t.ear_enabled,
            ear_threshold=t.ear_threshold,
            ear_freeze=t.ear_freeze,
            pitch_enabled=t.pitch_enabled,
            pitch_threshold=t.pitch_threshold,
            regressor_x=reg_x,
            regressor_y=reg_y,
            calib_error=self.calib_error,
            created_at=created_at,
        )
