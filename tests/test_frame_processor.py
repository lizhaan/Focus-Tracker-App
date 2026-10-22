import pytest

from attention_tracker.calibration.recenter import RecenterState
from attention_tracker.config import AppConfig
from attention_tracker.models import FocusState, FrameFeatures, Reason
from attention_tracker.workers.frame_processor import FrameProcessor
from tests.test_distraction import ff, make_profile

DT = 0.05


def run(proc, t0: float, t1: float, **kwargs):
    """Кадри на [t0, t1); повертає (останній вихід, переходи, події перецентрування)."""
    out, changes, events = None, [], []
    for i in range(round((t1 - t0) / DT)):
        frame = ff(round(t0 + i * DT, 6), **kwargs) if kwargs.get("valid", True) else \
            FrameFeatures.invalid(round(t0 + i * DT, 6))
        out = proc.process_features(frame)
        if out.change is not None:
            changes.append(out.change)
        if out.recenter_event is not None:
            events.append(out.recenter_event)
    return out, changes, events


@pytest.fixture
def proc() -> FrameProcessor:
    p = FrameProcessor(AppConfig(), make_profile())
    p.start_session(0.0)
    return p


def test_session_starts_with_recentering(proc):
    out, changes, events = run(proc, 0.0, 1.0)
    assert proc.recentering
    assert out.result.state is None and out.result.distraction is None
    assert changes == [] and events == []

    out, _, events = run(proc, 1.0, 2.5)
    assert events == [RecenterState.DONE]
    assert not proc.recentering
    assert out.result.state is FocusState.FOCUSED
    assert out.result.distraction.screen_point is not None


def test_recenter_retry_restarts_automatically(proc):
    _, _, events = run(proc, 0.0, 2.5, valid=False)
    assert RecenterState.RETRY in events
    assert proc.recentering
    _, _, events = run(proc, 2.5, 5.0)
    assert events == [RecenterState.DONE]


def test_recenter_result_carried_on_done_frame(proc):
    outputs = [proc.process_features(ff(round(i * DT, 6), yaw=4.0)) for i in range(60)]
    done = [o for o in outputs if o.recenter_event is RecenterState.DONE]
    assert len(done) == 1
    assert done[0].recenter_result.adjustment.yaw_baseline == pytest.approx(4.0)


def test_look_away_after_recentering_loses_focus(proc):
    run(proc, 0.0, 2.5)
    _, changes, _ = run(proc, 2.5, 7.0, sx=1.3)
    assert [c.new for c in changes] == [FocusState.NOT_FOCUSED, FocusState.LOST_FOCUS]
    assert all(c.reason is Reason.GAZE_OUT for c in changes)

    _, back, _ = run(proc, 7.0, 10.0)
    assert [c.new for c in back] == [FocusState.RECOVERING, FocusState.FOCUSED]


def test_phone_leads_to_lost_focus(proc):
    run(proc, 0.0, 2.5)
    _, changes, _ = run(proc, 2.5, 7.0, pitch=15.0)
    assert [c.new for c in changes] == [FocusState.NOT_FOCUSED, FocusState.LOST_FOCUS]
    assert changes[-1].reason is Reason.PHONE_HEAD


def test_head_turn_with_eyes_on_screen_keeps_focus(proc):
    run(proc, 0.0, 2.5)
    _, changes, _ = run(proc, 2.5, 7.0, sx=0.5, yaw=20.0)
    assert changes == []


def test_long_face_absence_resets_filters(proc):
    run(proc, 0.0, 2.5, yaw=0.0)
    run(proc, 2.5, 3.5, valid=False)
    out = proc.process_features(ff(3.5, yaw=10.0))
    assert out.result.features.yaw == pytest.approx(10.0)


def test_set_timings(proc):
    proc.set_timings(5.0, 1.0)
    assert (proc.state_machine.n, proc.state_machine.y) == (5.0, 1.0)
