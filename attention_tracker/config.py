"""Усі числові параметри застосунку (MATH §9, PROJECT_PLAN §8, §11).

Логіка отримує потрібну секцію `AppConfig` як аргумент, а не читає
константи напряму. Зміна налаштувань користувачем створює новий обʼєкт
через `dataclasses.replace`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent / "assets"


@dataclass(frozen=True)
class CameraConfig:
    """PROJECT_PLAN §8.5: камера (індекс + роздільна здатність)."""

    index: int = 0
    width: int = 1280
    height: int = 720
    fallback_width: int = 640
    fallback_height: int = 480


@dataclass(frozen=True)
class LandmarkerConfig:
    """MATH §0: FaceLandmarker, running_mode=VIDEO, num_faces=1.

    Пороги впевненості — стандартні значення MediaPipe (документи їх не задають).
    """

    model_path: Path = ASSETS_DIR / "face_landmarker.task"
    num_faces: int = 1
    min_face_detection_confidence: float = 0.5
    min_face_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass(frozen=True)
class FilterConfig:
    """MATH §5: median (k кадрів) + EMA (α)."""

    median_window: int = 5
    ema_alpha: float = 0.35


@dataclass(frozen=True)
class DistractionConfig:
    """MATH §3.2, §3.4, §4."""

    t_blink: float = 0.5  # с, §4.2
    t_noface: float = 0.5  # с, §4.1
    invalid_yaw_deg: float = 45.0  # §4.1
    invalid_pitch_deg: float = 40.0  # §4.1
    eps_out: float = 0.05  # §3.4
    eps_in: float = 0.02  # §3.4
    ear_freeze_factor: float = 0.7  # §3.2: ear_freeze = 0.7 · EAR_bottom


@dataclass(frozen=True)
class AlphaThresholdsConfig:
    """PROJECT_PLAN §11, Етап 2 п.3: тимчасові пороги без регресії.

    gaze_out = |Gaze_h − gaze_h_center| > gaze_h_delta OR |ΔYaw| > yaw_delta_deg,
    baseline береться з перших baseline_duration секунд сесії.
    """

    gaze_h_center: float = 0.5
    gaze_h_delta: float = 0.15
    yaw_delta_deg: float = 25.0
    baseline_duration: float = 2.0  # с


@dataclass(frozen=True)
class FocusConfig:
    """MATH §6: таймери машини станів. n і y налаштовує користувач (§8.5)."""

    n: float = 3.0  # с
    y: float = 2.0  # с
    tau: float = 0.3  # с
    n_min: float = 1.0
    n_max: float = 30.0
    y_min: float = 0.5
    y_max: float = 15.0
    ui_step: float = 0.5


@dataclass(frozen=True)
class CalibrationConfig:
    """MATH §7: збір даних, пороги, регресія, перевірка якості."""

    # §7.0 перевірка перед стартом
    precheck_duration: float = 2.0  # с
    precheck_face_ratio: float = 0.90
    iris_distance_min_frac: float = 0.08  # · W
    iris_distance_max_frac: float = 0.30  # · W
    min_fps: float = 15.0

    # §7.1 збір даних на кроці
    t_calib: float = 2.0  # с
    t_settle: float = 0.6  # с
    min_valid_ratio: float = 0.70
    mad_k: float = 3.0

    # §7 крок 3: повороти голови
    head_yaw_range_deg: tuple[float, float] = (10.0, 25.0)
    head_pitch_range_deg: tuple[float, float] = (8.0, 20.0)
    s_pitch_min_deg: float = 5.0

    # §7.2 пороги
    ear_min_gap: float = 0.03
    pitch_min_gap_deg: float = 5.0
    pitch_fallback_margin_deg: float = 8.0
    pitch_min_margin_deg: float = 4.0

    # §3.3, §7.4 регресія
    ridge_lambda: float = 0.05
    sigma_min: float = 1e-6

    # §7.5 перевірка якості
    n_check_points: int = 4
    error_ok: float = 0.08
    error_warn: float = 0.15

    # §7.6 перецентрування
    recenter_duration: float = 2.0  # с
    recenter_max_offset: float = 0.15


@dataclass(frozen=True)
class MetricsConfig:
    """MATH §8: біни графіка концентрації."""

    live_bin: float = 10.0  # с
    history_bin: float = 60.0  # с


@dataclass(frozen=True)
class UiConfig:
    """PROJECT_PLAN §5.2, §5.3, §8.3."""

    frame_signal_hz: float = 10.0  # проріджування frame_processed для UI
    fps_report_interval: float = 1.0  # с
    pomodoro_tick: float = 1.0  # с
    overlay_fade_ms: int = 200


@dataclass(frozen=True)
class AppConfig:
    """Корінь конфігурації (MATH §9)."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    landmarker: LandmarkerConfig = field(default_factory=LandmarkerConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    distraction: DistractionConfig = field(default_factory=DistractionConfig)
    alpha_thresholds: AlphaThresholdsConfig = field(default_factory=AlphaThresholdsConfig)
    focus: FocusConfig = field(default_factory=FocusConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    ui: UiConfig = field(default_factory=UiConfig)
