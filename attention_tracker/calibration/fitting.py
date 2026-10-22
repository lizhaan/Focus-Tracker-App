"""Обчислення з даних калібрування: робастне середнє, пороги, регресія, якість."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from attention_tracker.config import CalibrationConfig
from attention_tracker.vision.gaze_model import GazeRegressor, phi_x, phi_y


def robust_mean(values: list[float] | np.ndarray, mad_k: float) -> float:
    """Відкинути точки з |x − median| > k·MAD, потім mean від решти."""
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        raise ValueError("робастне середнє від порожнього набору")
    med = np.median(x)
    dev = np.abs(x - med)
    mad = np.median(dev)
    kept = x[dev <= mad_k * mad]
    return float(kept.mean())


@dataclass(frozen=True)
class Thresholds:
    ear_enabled: bool
    ear_threshold: float | None  # None, якщо EAR не розрізняє низ екрана і телефон
    ear_freeze: float
    pitch_threshold: float


def compute_thresholds(
    ear_bottom: float,
    ear_phone: float,
    pd_bottom: float,
    pd_phone: float,
    cfg: CalibrationConfig,
    ear_freeze_factor: float,
) -> Thresholds:
    """Пороги посередині між «низ екрана» і «телефон»."""
    ear_enabled = ear_bottom - ear_phone >= cfg.ear_min_gap
    ear_threshold = (ear_bottom + ear_phone) / 2.0 if ear_enabled else None

    if pd_phone - pd_bottom >= cfg.pitch_min_gap_deg:
        pitch_threshold = (pd_bottom + pd_phone) / 2.0
    else:
        pitch_threshold = pd_bottom + cfg.pitch_fallback_margin_deg
    pitch_threshold = max(pitch_threshold, pd_bottom + cfg.pitch_min_margin_deg)

    return Thresholds(
        ear_enabled=ear_enabled,
        ear_threshold=ear_threshold,
        ear_freeze=ear_freeze_factor * ear_bottom,
        pitch_threshold=pitch_threshold,
    )


def s_pitch_from(mean_d_pitch_down: float, min_abs_deg: float) -> int | None:
    """Знак нахилу «вниз»; None — нахил замалий, крок треба повторити."""
    if abs(mean_d_pitch_down) < min_abs_deg:
        return None
    return 1 if mean_d_pitch_down > 0 else -1


@dataclass(frozen=True)
class RegressionSample:
    """Одна точка набору регресії: ознаки → ціль на екрані."""

    gaze_h: float
    gaze_v: float
    d_yaw: float
    d_pitch: float
    sx: float
    sy: float


def fit_regressors(
    samples: list[RegressionSample], cfg: CalibrationConfig
) -> tuple[GazeRegressor, GazeRegressor]:
    phi_xs = np.array([phi_x(s.gaze_h, s.d_yaw) for s in samples])
    phi_ys = np.array([phi_y(s.gaze_v, s.d_pitch) for s in samples])
    reg_x = GazeRegressor.fit(phi_xs, np.array([s.sx for s in samples]), cfg.ridge_lambda, cfg.sigma_min)
    reg_y = GazeRegressor.fit(phi_ys, np.array([s.sy for s in samples]), cfg.ridge_lambda, cfg.sigma_min)
    return reg_x, reg_y


class Quality(Enum):
    OK = "OK"  # профіль зберігається
    WARN = "WARN"  # зберегти з попередженням «точність знижена»
    RETRY = "RETRY"  # запропонувати повторити калібрування


def quality_of(calib_error: float, cfg: CalibrationConfig) -> Quality:
    if calib_error <= cfg.error_ok:
        return Quality.OK
    if calib_error <= cfg.error_warn:
        return Quality.WARN
    return Quality.RETRY
