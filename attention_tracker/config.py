from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent / "assets"


@dataclass(frozen=True)
class CameraConfig:
    """камера (індекс + роздільна здатність)."""

    index: int = 0
    width: int = 1280
    height: int = 720
    fallback_width: int = 640
    fallback_height: int = 480


@dataclass(frozen=True)
class LandmarkerConfig:
    """FaceLandmarker, running_mode=VIDEO, num_faces=1.

    Пороги впевненості — стандартні значення MediaPipe (документи їх не задають).
    """

    model_path: Path = ASSETS_DIR / "face_landmarker.task"
    num_faces: int = 1
    min_face_detection_confidence: float = 0.5
    min_face_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass(frozen=True)
class FilterConfig:
    """median (k кадрів) + EMA (α)."""

    median_window: int = 5
    ema_alpha: float = 0.35


@dataclass(frozen=True)
class DistractionConfig:

    t_blink: float = 0.5  # с
    t_noface: float = 0.5  # с
    invalid_yaw_deg: float = 45.0
    invalid_pitch_deg: float = 40.0
    eps_out: float = 0.05
    eps_in: float = 0.02
    ear_freeze_factor: float = 0.7  # ear_freeze = 0.7 · EAR_bottom


@dataclass(frozen=True)
class FocusConfig:
    """таймери машини станів. n і y налаштовує користувач."""

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
    """збір даних, пороги, регресія, перевірка якості."""

    # перевірка перед стартом
    precheck_duration: float = 2.0  # с
    precheck_face_ratio: float = 0.90
    iris_distance_min_frac: float = 0.08  # · W
    iris_distance_max_frac: float = 0.30  # · W
    min_fps: float = 15.0

    # збір даних на кроці
    t_calib: float = 2.0  # с
    t_settle: float = 0.6  # с
    min_valid_ratio: float = 0.70
    mad_k: float = 3.0

    # цілі погляду, нормалізовані координати робочого монітора
    center_target: tuple[float, float] = (0.5, 0.5)
    grid_targets: tuple[tuple[float, float], ...] = (
        (0.05, 0.05), (0.5, 0.05), (0.95, 0.05),
        (0.05, 0.5), (0.95, 0.5),
        (0.05, 0.95), (0.5, 0.95), (0.95, 0.95),
    )
    head_yaw_targets: tuple[tuple[float, float], ...] = ((0.05, 0.5), (0.5, 0.5), (0.95, 0.5))
    head_pitch_targets: tuple[tuple[float, float], ...] = ((0.5, 0.05), (0.5, 0.5), (0.5, 0.95))
    bottom_target: tuple[float, float] = (0.5, 0.97)
    # точки перевірки якості: випадкові, не ближче min_dist до цілей навчання
    check_point_range: tuple[float, float] = (0.15, 0.85)
    check_point_min_dist: float = 0.1

    # крок 3: повороти голови
    head_yaw_range_deg: tuple[float, float] = (10.0, 25.0)
    head_pitch_range_deg: tuple[float, float] = (8.0, 20.0)
    s_pitch_min_deg: float = 5.0

    # пороги
    ear_min_gap: float = 0.03
    pitch_min_gap_deg: float = 5.0
    pitch_fallback_margin_deg: float = 8.0
    pitch_min_margin_deg: float = 4.0

    # регресія
    ridge_lambda: float = 0.05
    sigma_min: float = 1e-6

    # перевірка якості
    n_check_points: int = 4
    error_ok: float = 0.08
    error_warn: float = 0.15

    # перецентрування
    recenter_duration: float = 2.0  # с
    recenter_max_offset: float = 0.15


@dataclass(frozen=True)
class MetricsConfig:
    """біни графіка концентрації."""

    live_bin: float = 10.0  # с
    history_bin: float = 60.0  # с


@dataclass(frozen=True)
class UiConfig:

    frame_signal_hz: float = 10.0  # проріджування frame_processed для UI
    fps_report_interval: float = 1.0  # с
    pomodoro_tick: float = 1.0  # с
    overlay_fade_ms: int = 200
    calibration_countdown: float = 3.0  # с, відлік перед кроком з новою інструкцією
    calibration_lead_in: float = 1.0  # с, пауза перед наступною точкою того ж кроку


@dataclass(frozen=True)
class AppConfig:
    """Корінь конфігурації."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    landmarker: LandmarkerConfig = field(default_factory=LandmarkerConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    distraction: DistractionConfig = field(default_factory=DistractionConfig)
    focus: FocusConfig = field(default_factory=FocusConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    ui: UiConfig = field(default_factory=UiConfig)
