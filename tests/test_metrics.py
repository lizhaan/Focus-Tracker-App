import pytest

from attention_tracker.focus.metrics import (
    FocusEventRow,
    PomodoroEventRow,
    compute_metrics,
    focus_timeline,
    tracking_windows,
)
from attention_tracker.focus.pomodoro import PomodoroEvent as PE
from attention_tracker.models import FocusState, PomodoroPhase, Reason

F, NF, LOST, REC = FocusState.FOCUSED, FocusState.NOT_FOCUSED, FocusState.LOST_FOCUS, FocusState.RECOVERING
WORK, SHORT = PomodoroPhase.WORK, PomodoroPhase.SHORT_BREAK


def fe(t, old, new, reason=Reason.NONE):
    return FocusEventRow(t, old, new, reason)


def pe(t, event, phase=WORK):
    return PomodoroEventRow(t, event, phase)


# Одне відволікання з поверненням, робота 0–100 с
LOOK_AWAY = [
    fe(10, F, NF, Reason.GAZE_OUT),
    fe(13, NF, LOST, Reason.GAZE_OUT),
    fe(15, LOST, REC),
    fe(17, REC, F),
]
WORK_0_100 = [pe(0, PE.PHASE_START), pe(100, PE.STOP)]


def test_single_look_away():
    m = compute_metrics(LOOK_AWAY, WORK_0_100, end=100)
    assert m.t_focused == pytest.approx(93)  # 0–10 і 17–100
    assert m.t_not_focused == pytest.approx(3)
    assert m.t_lost == pytest.approx(4)  # LOST + RECOVERING
    assert m.t_tracked == pytest.approx(100)
    assert m.focus_ratio == pytest.approx(0.96)
    assert m.n_lost_episodes == 1
    assert m.mean_lost_duration == pytest.approx(4)
    assert m.longest_focus == pytest.approx(83)  # 17–100
    assert m.lost_by_reason == {"GAZE_OUT": 1}
    assert m.n_pomodoros == 0


def test_break_not_counted_and_state_resets_after_it():
    focus = [fe(37, F, NF, Reason.PHONE_EYES), fe(40, NF, LOST, Reason.PHONE_EYES)]
    pomodoro = [
        pe(0, PE.PHASE_START), pe(50, PE.PHASE_END), pe(50, PE.PHASE_START, SHORT),
        pe(60, PE.PHASE_END, SHORT), pe(60, PE.PHASE_START), pe(100, PE.STOP),
    ]
    m = compute_metrics(focus, pomodoro, end=100)
    assert m.t_tracked == pytest.approx(90)  # перерва 50–60 не рахується
    assert m.t_lost == pytest.approx(10)  # 40–50; після перерви знову FOCUSED
    assert m.t_focused == pytest.approx(77)  # 0–37 і 60–100
    assert m.n_pomodoros == 1
    assert m.lost_by_reason == {"PHONE_EYES": 1}
    assert m.longest_focus == pytest.approx(40)  # 0–40 перервано LOST, 60–100 = 40


def test_manual_pause_not_counted():
    pomodoro = [pe(0, PE.PHASE_START), pe(20, PE.MANUAL_PAUSE), pe(50, PE.MANUAL_RESUME), pe(80, PE.STOP)]
    m = compute_metrics([], pomodoro, end=80)
    assert m.t_tracked == pytest.approx(50)
    assert m.longest_focus == pytest.approx(30)  # пауза розриває відрізок


def test_auto_pause_time_is_counted_as_lost():
    focus = [fe(5, F, NF), fe(8, NF, LOST), fe(20, LOST, REC), fe(22, REC, F)]
    pomodoro = [pe(0, PE.PHASE_START), pe(8, PE.AUTO_PAUSE), pe(22, PE.AUTO_RESUME), pe(30, PE.STOP)]
    m = compute_metrics(focus, pomodoro, end=30)
    assert m.t_tracked == pytest.approx(30)
    assert m.t_lost == pytest.approx(14)


def test_session_without_timer_counts_until_end():
    m = compute_metrics([fe(10, F, NF)], [pe(0, PE.PHASE_START)], end=30)
    assert m.t_tracked == pytest.approx(30)
    assert m.t_not_focused == pytest.approx(20)


def test_empty_session():
    m = compute_metrics([], [], end=10)
    assert m.t_tracked == 0 and m.focus_ratio is None
    assert m.mean_lost_duration == 0.0 and m.lost_by_reason == {}


def test_events_outside_tracking_are_ignored():
    focus = [fe(55, F, NF), fe(58, NF, LOST)]  # під час перерви
    pomodoro = [pe(0, PE.PHASE_START), pe(50, PE.PHASE_END), pe(50, PE.PHASE_START, SHORT), pe(70, PE.STOP, SHORT)]
    m = compute_metrics(focus, pomodoro, end=70)
    assert m.n_lost_episodes == 0 and m.t_lost == 0


def test_tracking_windows():
    pomodoro = [pe(0, PE.PHASE_START), pe(20, PE.MANUAL_PAUSE), pe(30, PE.MANUAL_RESUME),
                pe(50, PE.PHASE_END), pe(50, PE.PHASE_START, SHORT), pe(60, PE.PHASE_END, SHORT),
                pe(60, PE.PHASE_START)]
    assert tracking_windows(pomodoro, end=70) == [(0, 20), (30, 50), (60, 70)]


def test_focus_timeline_bins():
    shares = focus_timeline(LOOK_AWAY, WORK_0_100, end=100, bin_s=10)
    assert len(shares) == 10
    assert shares[0] == pytest.approx(1.0)
    assert shares[1] == pytest.approx(0.6)  # 10–13 NF (фокус), 13–17 втрачено, 17–20 F
    assert shares[9] == pytest.approx(1.0)


def test_focus_timeline_break_is_none():
    pomodoro = [pe(0, PE.PHASE_START), pe(10, PE.PHASE_END), pe(10, PE.PHASE_START, SHORT),
                pe(20, PE.PHASE_END, SHORT), pe(20, PE.PHASE_START), pe(25, PE.STOP)]
    shares = focus_timeline([], pomodoro, end=25, bin_s=10)
    assert shares == [pytest.approx(1.0), None, pytest.approx(1.0)]
