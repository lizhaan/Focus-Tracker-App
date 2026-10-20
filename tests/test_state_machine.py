import pytest

from attention_tracker.config import FocusConfig
from attention_tracker.focus.state_machine import FocusStateMachine
from attention_tracker.models import FocusState, Reason

F, NF, LOST, REC = (
    FocusState.FOCUSED,
    FocusState.NOT_FOCUSED,
    FocusState.LOST_FOCUS,
    FocusState.RECOVERING,
)
DT = 0.05  # крок кадрів для довгих відрізків, с


@pytest.fixture
def sm() -> FocusStateMachine:
    machine = FocusStateMachine(FocusConfig())  # n=3, y=2, τ=0.3
    machine.reset(0.0)
    return machine


def feed(sm, t0: float, t1: float, distracted: bool, reason=Reason.GAZE_OUT) -> list:
    """Кадри кожні DT на [t0, t1); повертає список переходів."""
    changes = []
    for i in range(round((t1 - t0) / DT)):
        change = sm.update(distracted, reason, round(t0 + i * DT, 6))
        if change is not None:
            changes.append(change)
    return changes


def to_lost(sm) -> float:
    """FOCUSED → NOT_FOCUSED → LOST_FOCUS; повертає час переходу в LOST_FOCUS."""
    sm.update(True, Reason.GAZE_OUT, 0.0)
    change = sm.update(True, Reason.GAZE_OUT, 3.0)
    assert change.new is LOST
    return change.ts


def to_recovering(sm) -> float:
    to_lost(sm)
    change = sm.update(False, Reason.NONE, 4.0)
    assert change.new is REC
    return change.ts


# --- FOCUSED / NOT_FOCUSED -----------------------------------------------------


def test_initial_state_and_no_change_when_focused(sm):
    assert sm.state is F
    assert sm.update(False, Reason.NONE, 0.1) is None
    assert sm.recovering_progress(0.1) is None


def test_focused_to_not_focused_carries_reason(sm):
    change = sm.update(True, Reason.HEAD_AWAY, 1.0)
    assert (change.old, change.new, change.reason, change.ts) == (F, NF, Reason.HEAD_AWAY, 1.0)


def test_not_focused_to_lost_after_n(sm):
    sm.update(True, Reason.GAZE_OUT, 0.0)
    assert sm.update(True, Reason.GAZE_OUT, 2.99) is None
    change = sm.update(True, Reason.PHONE_HEAD, 3.0)
    assert (change.old, change.new) == (NF, LOST)
    assert change.reason is Reason.PHONE_HEAD  # причина поточного кадру


def test_not_focused_back_to_focused_after_tau(sm):
    sm.update(True, Reason.GAZE_OUT, 0.0)
    assert sm.update(False, Reason.NONE, 1.0) is None
    assert sm.update(False, Reason.NONE, 1.29) is None
    change = sm.update(False, Reason.NONE, 1.31)
    assert (change.old, change.new, change.reason) == (NF, F, Reason.NONE)


def test_short_return_below_tau_does_not_reset_n(sm):
    feed(sm, 0.0, 1.0, True)
    assert feed(sm, 1.0, 1.2, False) == []  # 0.2 с < τ
    changes = feed(sm, 1.2, 3.2, True)
    assert [c.new for c in changes] == [LOST]
    assert changes[0].ts == pytest.approx(3.0)


def test_n_elapsed_during_short_return_goes_lost_on_next_distracted(sm):
    sm.update(True, Reason.GAZE_OUT, 0.0)
    sm.update(False, Reason.NONE, 2.9)
    assert sm.update(False, Reason.NONE, 3.1) is None  # n минуло, але погляд на екрані
    change = sm.update(True, Reason.GAZE_OUT, 3.15)
    assert change.new is LOST


def test_interrupted_clear_streak_restarts_tau(sm):
    sm.update(True, Reason.GAZE_OUT, 0.0)
    sm.update(False, Reason.NONE, 1.0)
    sm.update(True, Reason.GAZE_OUT, 1.2)
    assert sm.update(False, Reason.NONE, 1.4) is None
    assert sm.update(False, Reason.NONE, 1.6) is None  # 0.2 с від 1.4
    assert sm.update(False, Reason.NONE, 1.71).new is F


# --- LOST_FOCUS / RECOVERING ---------------------------------------------------


def test_lost_stays_while_distracted(sm):
    to_lost(sm)
    assert feed(sm, 3.0, 10.0, True) == []
    assert sm.state is LOST


def test_lost_to_recovering_immediately(sm):
    to_lost(sm)
    change = sm.update(False, Reason.NONE, 4.0)
    assert (change.old, change.new, change.reason) == (LOST, REC, Reason.NONE)


def test_recovering_to_focused_after_y(sm):
    t_return = to_recovering(sm)
    assert sm.update(False, Reason.NONE, t_return + 1.99) is None
    change = sm.update(False, Reason.NONE, t_return + 2.0)
    assert (change.old, change.new) == (REC, F)


def test_single_distracted_frame_does_not_reset_y(sm):
    t_return = to_recovering(sm)
    sm.update(False, Reason.NONE, t_return + 1.0)
    assert sm.update(True, Reason.GAZE_OUT, t_return + 1.05) is None  # < τ
    assert sm.update(False, Reason.NONE, t_return + 1.1) is None
    assert sm.update(False, Reason.NONE, t_return + 2.0).new is F


def test_relapse_in_recovering_goes_lost_and_restarts_y(sm):
    t_return = to_recovering(sm)
    sm.update(True, Reason.FACE_ABSENT, t_return + 1.0)
    change = sm.update(True, Reason.FACE_ABSENT, t_return + 1.31)
    assert (change.old, change.new, change.reason) == (REC, LOST, Reason.FACE_ABSENT)

    t_return2 = t_return + 2.0
    assert sm.update(False, Reason.NONE, t_return2).new is REC
    assert sm.update(False, Reason.NONE, t_return2 + 1.9) is None  # y рахується заново
    assert sm.update(False, Reason.NONE, t_return2 + 2.0).new is F


def test_lost_wins_when_y_and_tau_expire_together(sm):
    t_return = to_recovering(sm)
    sm.update(True, Reason.GAZE_OUT, t_return + 1.75)
    change = sm.update(True, Reason.GAZE_OUT, t_return + 2.1)  # і y, і τ минули
    assert change.new is LOST


def test_recovering_progress(sm):
    t_return = to_recovering(sm)
    assert sm.recovering_progress(t_return) == pytest.approx(0.0)
    assert sm.recovering_progress(t_return + 1.0) == pytest.approx(0.5)
    assert sm.recovering_progress(t_return + 5.0) == pytest.approx(1.0)
    sm.update(False, Reason.NONE, t_return + 2.0)
    assert sm.recovering_progress(t_return + 2.0) is None


# --- set_timings / reset -------------------------------------------------------


def test_lower_n_mid_not_focused_uses_accumulated_time(sm):
    sm.update(True, Reason.GAZE_OUT, 0.0)
    sm.update(True, Reason.GAZE_OUT, 2.0)
    sm.set_timings(n=1.5, y=2.0)
    assert sm.update(True, Reason.GAZE_OUT, 2.05).new is LOST


def test_raise_n_mid_not_focused_delays_lost(sm):
    sm.update(True, Reason.GAZE_OUT, 0.0)
    sm.set_timings(n=5.0, y=2.0)
    assert sm.update(True, Reason.GAZE_OUT, 3.0) is None
    assert sm.update(True, Reason.GAZE_OUT, 5.0).new is LOST


def test_change_y_mid_recovering(sm):
    t_return = to_recovering(sm)
    sm.update(False, Reason.NONE, t_return + 1.0)
    sm.set_timings(n=3.0, y=1.0)
    assert sm.recovering_progress(t_return + 1.0) == pytest.approx(1.0)
    assert sm.update(False, Reason.NONE, t_return + 1.05).new is F


@pytest.mark.parametrize("n, y", [(0.5, 2.0), (31.0, 2.0), (3.0, 0.4), (3.0, 16.0)])
def test_set_timings_rejects_out_of_range(sm, n, y):
    with pytest.raises(ValueError):
        sm.set_timings(n, y)
    assert (sm.n, sm.y) == (3.0, 2.0)


def test_reset_returns_to_focused(sm):
    to_recovering(sm)
    sm.reset(10.0)
    assert sm.state is F
    assert sm.recovering_progress(10.0) is None
    assert sm.update(True, Reason.GAZE_OUT, 10.0).new is NF


# --- Повний цикл ---------------------------------------------------------------


def test_full_cycle_with_frames(sm):
    changes = []
    changes += feed(sm, 0.0, 1.0, False)
    changes += feed(sm, 1.0, 5.0, True)  # відвернулась на 4 с
    changes += feed(sm, 5.0, 8.0, False)  # повернулась на 3 с
    assert [(c.old, c.new) for c in changes] == [(F, NF), (NF, LOST), (LOST, REC), (REC, F)]
    assert [c.ts for c in changes] == pytest.approx([1.0, 4.0, 5.0, 7.0])
