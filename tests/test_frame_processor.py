import pytest

from attention_tracker.config import AppConfig
from attention_tracker.models import FocusState, FrameFeatures, Reason
from attention_tracker.workers.frame_processor import FrameProcessor

DT = 0.05


def ff(ts: float, yaw: float = 0.0, gaze_h: float = 0.5, valid: bool = True) -> FrameFeatures:
    if not valid:
        return FrameFeatures.invalid(ts)
    return FrameFeatures(ts=ts, valid=True, ear=0.3, yaw=yaw, pitch=0.0, roll=0.0,
                         gaze_h=gaze_h, gaze_v=0.0, iris_distance_px=120.0)


def run(proc, t0: float, t1: float, **kwargs):
    """Кадри на [t0, t1); повертає (останній FrameResult, список переходів)."""
    changes, result = [], None
    for i in range(round((t1 - t0) / DT)):
        result, change = proc.process_features(ff(round(t0 + i * DT, 6), **kwargs))
        if change is not None:
            changes.append(change)
    return result, changes


@pytest.fixture
def proc() -> FrameProcessor:
    p = FrameProcessor(AppConfig())
    p.start_session(0.0)
    return p


def test_baseline_then_focused(proc):
    result, changes = run(proc, 0.0, 2.1)
    assert proc.baseline_ready
    assert changes == []
    assert result.state is FocusState.FOCUSED
    assert not result.distraction.distracted


def test_full_cycle_look_away_and_back(proc):
    run(proc, 0.0, 2.1)
    result, away = run(proc, 2.1, 6.0, gaze_h=0.9)  # дивимось убік ~4 с
    assert [c.new for c in away] == [FocusState.NOT_FOCUSED, FocusState.LOST_FOCUS]
    assert all(c.reason is Reason.GAZE_OUT for c in away)
    assert away[1].ts - away[0].ts == pytest.approx(3.0, abs=DT)  # n = 3 с
    assert result.state is FocusState.LOST_FOCUS

    result, back = run(proc, 6.0, 6.5)
    assert [c.new for c in back] == [FocusState.RECOVERING]
    assert 0.0 < result.recovering_progress < 1.0

    result, back2 = run(proc, 6.5, 9.0)
    assert [c.new for c in back2] == [FocusState.FOCUSED]
    assert back2[0].ts - back[0].ts == pytest.approx(2.0, abs=DT)  # y = 2 с
    assert result.recovering_progress is None


def test_face_absent_leads_to_lost_focus(proc):
    run(proc, 0.0, 2.1)
    result, changes = run(proc, 2.1, 6.5, valid=False)
    assert [c.new for c in changes] == [FocusState.NOT_FOCUSED, FocusState.LOST_FOCUS]
    assert changes[-1].reason is Reason.FACE_ABSENT


def test_long_face_absence_resets_filters(proc):
    run(proc, 0.0, 2.1, yaw=0.0)
    run(proc, 2.1, 3.0, valid=False)  # довше T_noface
    result, _ = proc.process_features(ff(3.0, yaw=10.0))
    assert result.features.yaw == pytest.approx(10.0)  # фільтр почав з нуля


def test_start_session_resets_everything(proc):
    run(proc, 0.0, 2.1)
    run(proc, 2.1, 6.0, gaze_h=0.9)
    assert proc.state_machine.state is FocusState.LOST_FOCUS
    proc.start_session(10.0)
    assert proc.state_machine.state is FocusState.FOCUSED
    assert not proc.baseline_ready


def test_set_timings_applies_to_state_machine(proc):
    proc.set_timings(5.0, 1.0)
    assert (proc.state_machine.n, proc.state_machine.y) == (5.0, 1.0)
