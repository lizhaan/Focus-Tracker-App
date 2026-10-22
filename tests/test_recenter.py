import pytest

from attention_tracker.calibration.recenter import Recenterer, RecenterState, SessionAdjustment
from attention_tracker.config import CalibrationConfig
from attention_tracker.models import FrameFeatures
from tests.test_distraction import ff, gaze_h_for, make_profile

CFG = CalibrationConfig()
FPS = 30.0


def run(rec, t0: float, seconds: float, **kwargs) -> RecenterState:
    state = rec.state
    for i in range(int(seconds * FPS) + 1):
        state = rec.feed(ff(round(t0 + i / FPS, 6), **kwargs))
        if state is not RecenterState.COLLECTING:
            break
    return state


def test_recenter_updates_baseline_and_offsets():
    rec = Recenterer(CFG, make_profile())
    rec.start(0.0)
    # голова зсунулась на 5°, а очі, дивлячись у центр, дають sx = 0.45 при ΔYaw = 0
    assert run(rec, 0.0, 3.0, yaw=5.0, gaze_h=gaze_h_for(0.45), sy=0.53) is RecenterState.DONE
    adj = rec.result.adjustment
    assert adj.yaw_baseline == pytest.approx(5.0)
    assert adj.pitch_baseline == pytest.approx(0.0)
    assert adj.off_x == pytest.approx(0.05)
    assert adj.off_y == pytest.approx(-0.03)
    assert not rec.result.large_shift


def test_large_shift_flagged():
    rec = Recenterer(CFG, make_profile())
    rec.start(0.0)
    run(rec, 0.0, 3.0, gaze_h=gaze_h_for(0.3))
    assert rec.result.large_shift
    assert rec.result.adjustment.off_x == pytest.approx(0.2)


def test_low_valid_ratio_retries():
    rec = Recenterer(CFG, make_profile())
    rec.start(0.0)
    state = RecenterState.COLLECTING
    for i in range(int(3 * FPS)):
        frame = ff(i / FPS) if i % 2 else FrameFeatures.invalid(i / FPS)
        state = rec.feed(frame)
        if state is not RecenterState.COLLECTING:
            break
    assert state is RecenterState.RETRY and rec.result is None


def test_progress():
    rec = Recenterer(CFG, make_profile())
    rec.start(10.0)
    assert rec.progress(11.0) == pytest.approx(0.5)
    assert rec.progress(20.0) == pytest.approx(1.0)


def test_adjustment_from_profile():
    assert SessionAdjustment.from_profile(make_profile(yaw_baseline=3.0, pitch_baseline=-2.0)) == \
        SessionAdjustment(3.0, -2.0, 0.0, 0.0)
