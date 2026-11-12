import sqlite3

import pytest

from attention_tracker.focus.metrics import FocusEventRow, PomodoroEventRow
from attention_tracker.focus.pomodoro import PomodoroEvent as PE
from attention_tracker.models import FocusState, PomodoroPhase, Reason, StateChange
from attention_tracker.session.recorder import SessionRecorder, recover_unfinished
from attention_tracker.storage.storage import DEFAULT_USER, Storage
from tests.test_storage import make_profile

T0 = 1000.0  # monotonic на старті сесії
F, NF, LOST = FocusState.FOCUSED, FocusState.NOT_FOCUSED, FocusState.LOST_FOCUS


@pytest.fixture
def db():
    with Storage(":memory:") as s:
        yield s


@pytest.fixture
def user(db):
    return db.ensure_user(DEFAULT_USER, "2026-11-12T10:00:00")


def recorder(db, user, profile_id=None):
    return SessionRecorder(db, user, "mon", profile_id, {"n": 3.0, "pomodoro": True},
                           "2026-11-12T10:00:00", T0)


def test_records_events_with_offsets(db, user):
    rec = recorder(db, user)
    rec.on_pomodoro_event(PE.PHASE_START, PomodoroPhase.WORK, T0)
    rec.on_state_change(StateChange(T0 + 10, F, NF, Reason.GAZE_OUT))
    assert db.focus_events(rec.session_id) == [FocusEventRow(10.0, F, NF, Reason.GAZE_OUT)]
    assert db.pomodoro_events(rec.session_id) == [PomodoroEventRow(0.0, PE.PHASE_START, PomodoroPhase.WORK)]


def test_finish_adds_stop_once_and_saves_metrics(db, user):
    rec = recorder(db, user)
    rec.on_pomodoro_event(PE.PHASE_START, PomodoroPhase.WORK, T0)
    rec.on_state_change(StateChange(T0 + 10, F, NF, Reason.GAZE_OUT))
    rec.on_state_change(StateChange(T0 + 13, NF, LOST, Reason.GAZE_OUT))
    metrics = rec.finish(T0 + 20, "2026-11-12T10:00:20")
    assert rec.finished
    assert metrics.t_tracked == pytest.approx(20)
    assert metrics.n_lost_episodes == 1
    stops = [e for e in db.pomodoro_events(rec.session_id) if e.event is PE.STOP]
    assert len(stops) == 1 and stops[0].t == pytest.approx(20)

    [info] = db.list_sessions(user)
    assert info.ended_at == "2026-11-12T10:00:20"
    assert info.settings == {"n": 3.0, "pomodoro": True}
    assert info.metrics.n_lost_episodes == 1
    assert info.metrics.lost_by_reason == {"GAZE_OUT": 1}

    with pytest.raises(RuntimeError):
        rec.finish(T0 + 30, "2026-11-12T10:00:30")


def test_no_duplicate_stop_after_pomodoro_stop(db, user):
    rec = recorder(db, user)
    rec.on_pomodoro_event(PE.PHASE_START, PomodoroPhase.WORK, T0)
    rec.on_pomodoro_event(PE.STOP, PomodoroPhase.WORK, T0 + 5)
    rec.finish(T0 + 5, "2026-11-12T10:00:05")
    assert [e.event for e in db.pomodoro_events(rec.session_id)].count(PE.STOP) == 1


def test_events_after_finish_are_ignored(db, user):
    rec = recorder(db, user)
    rec.finish(T0 + 1, "2026-11-12T10:00:01")
    rec.on_state_change(StateChange(T0 + 2, F, NF, Reason.GAZE_OUT))
    rec.on_pomodoro_event(PE.STOP, PomodoroPhase.WORK, T0 + 2)
    assert db.focus_events(rec.session_id) == [] and db.pomodoro_events(rec.session_id) == []


def test_recover_unfinished_session(db, user):
    rec = recorder(db, user)
    rec.on_pomodoro_event(PE.PHASE_START, PomodoroPhase.WORK, T0)
    rec.on_state_change(StateChange(T0 + 90, F, NF, Reason.GAZE_OUT))
    # застосунок «упав» — finish() не викликано
    assert recover_unfinished(db) == 1
    [info] = db.list_sessions(user)
    assert info.ended_at == "2026-11-12T10:01:30"  # старт + остання подія
    assert info.metrics.t_tracked == pytest.approx(90)
    assert recover_unfinished(db) == 0


def test_session_keeps_history_when_profile_deleted(db, user):
    profile_id = db.save_profile(make_profile(user), "mon")
    rec = recorder(db, user, profile_id)
    rec.finish(T0 + 1, "2026-11-12T10:00:01")
    db.delete_profile(profile_id)
    [info] = db.list_sessions(user)
    assert info.profile_id is None


def test_delete_session_removes_events(db, user):
    rec = recorder(db, user)
    rec.on_pomodoro_event(PE.PHASE_START, PomodoroPhase.WORK, T0)
    rec.finish(T0 + 1, "2026-11-12T10:00:01")
    db.delete_session(rec.session_id)
    assert db.list_sessions(user) == []
    assert db.pomodoro_events(rec.session_id) == []


def test_migration_from_schema_v1(tmp_path):
    from pathlib import Path

    import attention_tracker.storage.storage as storage_module

    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.executescript((Path(storage_module.__file__).with_name("schema.sql")).read_text(encoding="utf-8"))
    con.execute("PRAGMA user_version = 1")
    con.execute("INSERT INTO settings (key, value) VALUES ('work_monitor', 'mon')")
    con.commit()
    con.close()

    with Storage(path) as db:
        assert db.schema_version == 2
        assert db.get_setting("work_monitor") == "mon"  # старі дані на місці
        user = db.ensure_user(DEFAULT_USER, "2026-11-12T10:00:00")
        assert db.list_sessions(user) == []
