import pytest

from attention_tracker.config import PomodoroConfig
from attention_tracker.focus.pomodoro import PomodoroEvent, PomodoroTimer
from attention_tracker.models import PomodoroPhase

CFG = PomodoroConfig(work=100.0, short_break=20.0, long_break=50.0, pomodoros_before_long_break=2)
W, S, L, IDLE = (PomodoroPhase.WORK, PomodoroPhase.SHORT_BREAK, PomodoroPhase.LONG_BREAK, PomodoroPhase.IDLE)


@pytest.fixture
def events():
    return []


@pytest.fixture
def timer(events) -> PomodoroTimer:
    return PomodoroTimer(CFG, events.append)


def kinds(events):
    return [(e.event, e.phase) for e in events]


def test_default_config_is_classic():
    cfg = PomodoroConfig()
    assert (cfg.work, cfg.short_break, cfg.long_break, cfg.pomodoros_before_long_break) == (1500.0, 300.0, 900.0, 4)


def test_start_begins_work(timer, events):
    assert timer.phase is IDLE and timer.remaining(0.0) == 0.0
    timer.start(0.0)
    assert timer.phase is W
    assert timer.remaining(30.0) == pytest.approx(70.0)
    assert kinds(events) == [(PomodoroEvent.PHASE_START, W)]


def test_phase_cycle_with_long_break(timer):
    timer.start(0.0)
    timer.tick(99.0)
    assert timer.phase is W
    timer.tick(100.0)
    assert timer.phase is S and timer.completed_pomodoros == 1
    timer.tick(120.0)
    assert timer.phase is W
    timer.tick(220.0)
    assert timer.phase is L and timer.completed_pomodoros == 2  # кожен 2-й — довга перерва
    timer.tick(270.0)
    assert timer.phase is W


def test_phase_events(timer, events):
    timer.start(0.0)
    timer.tick(100.0)
    assert kinds(events)[1:] == [(PomodoroEvent.PHASE_END, W), (PomodoroEvent.PHASE_START, S)]


def test_paused_time_not_counted(timer):
    timer.start(0.0)
    timer.pause_auto(10.0)
    assert timer.remaining(60.0) == pytest.approx(90.0)
    timer.tick(200.0)
    assert timer.phase is W  # на паузі фаза не завершується
    timer.resume_auto(60.0)
    assert timer.remaining(70.0) == pytest.approx(80.0)


def test_auto_resume_does_not_lift_manual_pause(timer, events):
    timer.start(0.0)
    timer.pause_auto(10.0)
    timer.pause_manual(20.0)
    timer.resume_auto(30.0)
    assert timer.paused and timer.paused_manual and not timer.paused_auto
    assert timer.remaining(100.0) == pytest.approx(90.0)
    assert PomodoroEvent.AUTO_RESUME in [e.event for e in events]


def test_manual_resume_clears_auto_pause(timer):
    timer.start(0.0)
    timer.pause_auto(10.0)
    timer.pause_manual(20.0)
    timer.resume_manual(40.0)
    assert not timer.paused
    assert timer.remaining(50.0) == pytest.approx(80.0)  # ішов 0–10 і 40–50


def test_auto_pause_only_in_work(timer, events):
    timer.start(0.0)
    timer.tick(100.0)
    assert timer.phase is S
    timer.pause_auto(105.0)
    assert not timer.paused
    assert PomodoroEvent.AUTO_PAUSE not in [e.event for e in events]


def test_manual_pause_in_break(timer):
    timer.start(0.0)
    timer.tick(100.0)
    timer.pause_manual(105.0)
    timer.tick(500.0)
    assert timer.phase is S
    timer.resume_manual(500.0)
    assert timer.remaining(500.0) == pytest.approx(15.0)


def test_repeated_pause_calls_are_ignored(timer, events):
    timer.start(0.0)
    timer.pause_auto(10.0)
    timer.pause_auto(20.0)
    timer.resume_auto(30.0)
    timer.resume_auto(40.0)
    assert [e.event for e in events].count(PomodoroEvent.AUTO_PAUSE) == 1
    assert [e.event for e in events].count(PomodoroEvent.AUTO_RESUME) == 1
    assert timer.remaining(30.0) == pytest.approx(90.0)  # пауза 10–30 не рахується


def test_skip_phase_counts_work_and_clears_pauses(timer):
    timer.start(0.0)
    timer.pause_manual(10.0)
    timer.skip_phase(20.0)
    assert timer.phase is S and timer.completed_pomodoros == 1 and not timer.paused
    assert timer.remaining(20.0) == pytest.approx(20.0)
    timer.skip_phase(25.0)
    assert timer.phase is W


def test_stop_returns_to_idle(timer, events):
    timer.start(0.0)
    timer.stop(30.0)
    assert timer.phase is IDLE and not timer.paused
    assert kinds(events)[-1] == (PomodoroEvent.STOP, W)
    timer.stop(40.0)
    assert len(events) == 2  # повторний стоп — без події


def test_start_again_resets_counter(timer):
    timer.start(0.0)
    timer.skip_phase(1.0)
    timer.stop(2.0)
    timer.start(3.0)
    assert timer.completed_pomodoros == 0 and timer.phase is W


def test_overshoot_starts_next_phase_from_tick_time(timer):
    timer.start(0.0)
    timer.tick(130.0)  # тік запізнився на 30 с
    assert timer.phase is S
    assert timer.remaining(130.0) == pytest.approx(20.0)
