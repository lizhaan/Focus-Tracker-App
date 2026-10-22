import dataclasses

import numpy as np
import pytest

from attention_tracker.calibration.profile import CalibrationProfile
from attention_tracker.calibration.recenter import SessionAdjustment
from attention_tracker.config import DistractionConfig
from attention_tracker.models import FrameFeatures, Reason
from attention_tracker.vision.distraction import DistractionDetector
from attention_tracker.vision.gaze_model import GazeRegressor

DT = 0.05
# Прозора регресія (μ = 0, σ = 1):
#   sx = 0.5 + 4·(Gaze_h − 0.5) + 0.02·ΔYaw     → поворот голови компенсується очима
#   sy = 0.5 + 5·Gaze_v + 0.02·ΔPitch
REG_X = GazeRegressor(-1.5, np.array([4.0, 0.02, 0.0, 0.0]), np.zeros(4), np.ones(4))
REG_Y = GazeRegressor(0.5, np.array([5.0, 0.02, 0.0, 0.0]), np.zeros(4), np.ones(4))


def make_profile(**overrides) -> CalibrationProfile:
    base = dict(
        user_id=1, monitor_id="mon", camera_id="index:0", frame_w=1280, frame_h=720,
        yaw_baseline=0.0, pitch_baseline=0.0, ear_center=0.3, s_pitch=1,
        ear_bottom=0.26, pd_bottom=3.0, ear_phone=0.18, pd_phone=20.0,
        ear_enabled=True, ear_threshold=0.22, ear_freeze=0.18, pitch_threshold=11.5,
        regressor_x=REG_X, regressor_y=REG_Y, calib_error=0.05, created_at="2026-10-22T10:00:00",
    )
    base.update(overrides)
    return CalibrationProfile(**base)


def gaze_h_for(sx: float, d_yaw: float = 0.0) -> float:
    """Gaze_h, який дає точку sx при заданому повороті голови."""
    return 0.5 + (sx - 0.5 - 0.02 * d_yaw) / 4.0


def ff(ts, sx=0.5, sy=0.5, yaw=0.0, pitch=0.0, ear=0.3, gaze_h=None, valid=True) -> FrameFeatures:
    if not valid:
        return FrameFeatures.invalid(ts)
    gh = gaze_h if gaze_h is not None else gaze_h_for(sx, yaw)
    gv = (sy - 0.5 - 0.02 * pitch) / 5.0
    return FrameFeatures(ts=ts, valid=True, ear=ear, yaw=yaw, pitch=pitch, roll=0.0,
                         gaze_h=gh, gaze_v=gv, iris_distance_px=120.0)


def step(det, frame):
    return det.update(frame, frame)


def hold(det, t0, t1, **kwargs):
    """Однакові кадри на [t0, t1]; повертає останній результат."""
    result = None
    for i in range(round((t1 - t0) / DT) + 1):
        result = step(det, ff(round(t0 + i * DT, 6), **kwargs))
    return result


@pytest.fixture
def det() -> DistractionDetector:
    d = DistractionDetector(DistractionConfig(), make_profile())
    d.reset(0.0)
    return d


# --- точка погляду і гістерезис ---------------------------------------------


def test_center_gaze_is_on_screen(det):
    result = step(det, ff(0.0))
    assert not result.distracted and result.reason is Reason.NONE
    assert result.screen_point == pytest.approx((0.5, 0.5))


def test_hysteresis_on_right_edge(det):
    assert not step(det, ff(0.0, sx=1.04)).distracted  # усередині ε_out = 0.05
    assert step(det, ff(0.1, sx=1.06)).reason is Reason.GAZE_OUT
    assert step(det, ff(0.2, sx=1.03)).distracted  # щоб повернутись — ε_in = 0.02
    assert not step(det, ff(0.3, sx=1.01)).distracted


@pytest.mark.parametrize("sx, sy", [(-0.1, 0.5), (0.5, -0.1), (0.5, 1.1)])
def test_gaze_outside_any_edge(det, sx, sy):
    assert step(det, ff(0.0, sx=sx, sy=sy)).reason is Reason.GAZE_OUT


def test_head_turn_with_eyes_on_screen_is_not_distraction(det):
    result = step(det, ff(0.0, sx=0.5, yaw=20.0))  # голова вбік, очі компенсують
    assert not result.distracted
    assert result.screen_point[0] == pytest.approx(0.5)


# --- телефон -------------------------------------------------------------


def test_head_down_needs_t_blink(det):
    assert not hold(det, 0.0, 0.4, pitch=15.0).distracted
    assert hold(det, 0.45, 0.6, pitch=15.0).reason is Reason.PHONE_HEAD


def test_bottom_of_screen_is_not_phone(det):
    assert not hold(det, 0.0, 2.0, sy=0.95, pitch=3.0).distracted  # PD_bottom = 3 < 11.5


def test_head_down_ignored_when_pitch_disabled():
    d = DistractionDetector(DistractionConfig(), make_profile(pitch_enabled=False, pitch_threshold=None))
    d.reset(0.0)
    result = hold(d, 0.0, 2.0, pitch=12.8)  # зміна пози, а не телефон
    assert result.reason is not Reason.PHONE_HEAD
    assert result.pitch_down == pytest.approx(12.8)


def test_result_carries_head_deltas(det):
    det.apply_recenter(SessionAdjustment(yaw_baseline=2.0, pitch_baseline=-1.0))
    result = step(det, ff(0.0, yaw=5.0, pitch=3.0, gaze_h=gaze_h_for(0.5, 3.0)))
    assert (result.d_yaw, result.d_pitch, result.pitch_down) == pytest.approx((3.0, 4.0, 4.0))


def test_eyes_low_is_phone_eyes(det):
    assert hold(det, 0.0, 0.6, ear=0.2).reason is Reason.PHONE_EYES


def test_blink_is_ignored(det):
    assert not hold(det, 0.0, 0.3, ear=0.1).distracted
    assert not hold(det, 0.35, 1.0).distracted


def test_eyes_low_ignored_when_ear_disabled():
    d = DistractionDetector(DistractionConfig(), make_profile(ear_enabled=False, ear_threshold=None))
    d.reset(0.0)
    assert not hold(d, 0.0, 1.0, ear=0.2).distracted


def test_reason_priority(det):
    assert hold(det, 0.0, 0.6, pitch=15.0, ear=0.2).reason is Reason.PHONE_HEAD
    det.reset(1.0)
    assert hold(det, 1.0, 1.6, sx=1.3, ear=0.2).reason is Reason.PHONE_EYES


def test_gaze_frozen_while_blinking(det):
    step(det, ff(0.0, sx=0.5))
    result = step(det, ff(0.05, ear=0.1, gaze_h=0.9))  # райдужка «стрибнула» під час моргання
    assert result.screen_point[0] == pytest.approx(0.5)
    assert not result.distracted


# --- невалідні кадри -------------------------------------------------------


def test_short_invalid_holds_previous(det):
    step(det, ff(0.0, sx=1.3))
    held = step(det, ff(0.3, valid=False))
    assert held.reason is Reason.GAZE_OUT and held.screen_point is None


def test_long_invalid_face_absent_resets_filters_and_hysteresis(det):
    step(det, ff(0.0, sx=1.3))  # погляд поза екраном
    result = step(det, ff(0.6, valid=False))
    assert result.reason is Reason.FACE_ABSENT
    assert det.consume_filter_reset() and not det.consume_filter_reset()
    assert not step(det, ff(0.7, sx=1.04)).distracted  # гістерезис знову з «на екрані»


@pytest.mark.parametrize("yaw, pitch", [(50.0, 0.0), (0.0, -45.0)])
def test_large_angles_head_away(det, yaw, pitch):
    step(det, ff(0.0))
    assert not step(det, ff(0.3, yaw=yaw, pitch=pitch)).distracted
    assert step(det, ff(0.6, yaw=yaw, pitch=pitch)).reason is Reason.HEAD_AWAY


# --- перецентрування -------------------------------------------------------


def test_recenter_shifts_baseline_and_point(det):
    det.apply_recenter(SessionAdjustment(yaw_baseline=30.0, pitch_baseline=0.0, off_x=0.1, off_y=-0.1))
    result = step(det, ff(0.0, sx=0.5, yaw=30.0, gaze_h=gaze_h_for(0.5)))
    assert not result.distracted  # 30° від профілю, але 0° від нового baseline
    assert result.screen_point == pytest.approx((0.6, 0.4))


def test_profile_baseline_used_by_default():
    d = DistractionDetector(DistractionConfig(), make_profile(yaw_baseline=10.0))
    assert d.adjustment == SessionAdjustment(10.0, 0.0)


def test_profile_is_frozen_dataclass():
    with pytest.raises(dataclasses.FrozenInstanceError):
        make_profile().yaw_baseline = 1.0  # type: ignore[misc]
