import pytest

from attention_tracker.config import MetricsConfig
from attention_tracker.focus.pomodoro import PomodoroEvent as PE
from attention_tracker.models import FocusState as S, PomodoroPhase, Reason, StateChange
from attention_tracker.session.recorder import SessionRecorder
from attention_tracker.session.summary import session_duration, summary_from_storage
from attention_tracker.storage.storage import DEFAULT_USER, Storage

T0 = 500.0


@pytest.fixture
def db():
    with Storage(":memory:") as s:
        yield s


@pytest.fixture
def user(db):
    return db.ensure_user(DEFAULT_USER, "2026-11-12T10:00:00")


def record_session(db, user, started_at, ended_at, lost=0, length=120.0):
    rec = SessionRecorder(db, user, "mon", None, {"pomodoro": True}, started_at, T0)
    rec.on_pomodoro_event(PE.PHASE_START, PomodoroPhase.WORK, T0)
    for i in range(lost):
        t = T0 + 10 + i * 20
        rec.on_state_change(StateChange(t, S.FOCUSED, S.NOT_FOCUSED, Reason.GAZE_OUT))
        rec.on_state_change(StateChange(t + 3, S.NOT_FOCUSED, S.LOST_FOCUS, Reason.GAZE_OUT))
        rec.on_state_change(StateChange(t + 6, S.LOST_FOCUS, S.RECOVERING, Reason.NONE))
        rec.on_state_change(StateChange(t + 8, S.RECOVERING, S.FOCUSED, Reason.NONE))
    rec.finish(T0 + length, ended_at)
    return rec.session_id


def test_session_duration_and_summary_from_storage(db, user):
    record_session(db, user, "2026-11-12T10:00:00", "2026-11-12T10:02:00", lost=2)
    [info] = db.list_sessions(user)
    assert session_duration(db, info) == pytest.approx(120.0)
    s = summary_from_storage(db, info, MetricsConfig())
    assert s.metrics.n_lost_episodes == 2
    assert s.bin_s == 10.0 and len(s.shares) == 12


def test_session_row():
    from attention_tracker.ui.history_view import session_row

    from attention_tracker.focus.metrics import SessionMetrics
    from attention_tracker.storage.storage import SessionInfo

    info = SessionInfo(1, "2026-11-12T14:05:00", "2026-11-12T14:30:00", {"pomodoro": False}, "mon", None,
                       SessionMetrics(t_focused=900, t_not_focused=60, t_lost=40, focus_ratio=0.96,
                                      n_lost_episodes=3, n_pomodoros=1))
    assert session_row(info, 1500) == ["12 листопада, 14:05", "25 хв", "96%", "3", "—"]


def test_history_view_lists_and_selects(qapp, db, user):
    from attention_tracker.ui.history_view import HistoryView

    first = record_session(db, user, "2026-11-12T10:00:00", "2026-11-12T10:02:00", lost=1)
    second = record_session(db, user, "2026-11-12T11:00:00", "2026-11-12T11:02:00", lost=3)
    view = HistoryView(db, user, MetricsConfig())
    assert view.session_count == 2
    assert view.selected_session_id == second  # найновіша першою
    assert view.details.tile_values[1] == "3"

    view.select_row(1)
    assert view.selected_session_id == first
    assert view.details.tile_values[1] == "1"
    view.grab()


def test_history_view_empty_and_skips_unfinished(qapp, db, user):
    from attention_tracker.ui.history_view import HistoryView

    SessionRecorder(db, user, "mon", None, {}, "2026-11-12T10:00:00", T0)  # ще триває
    view = HistoryView(db, user, MetricsConfig())
    assert view.session_count == 0
    assert view.selected_session_id is None and view.details is None


def test_history_tab_in_main_window(main_window):
    from attention_tracker.calibration.recenter import RecenterResult, SessionAdjustment

    window, storage = main_window
    tabs = window._tabs
    assert [tabs.tabText(i) for i in range(tabs.count())] == ["Сесія", "Історія", "Налаштування", "Debug"]
    assert window._history.session_count == 0

    window._on_recenter_done(RecenterResult(SessionAdjustment(0.0, 0.0), large_shift=False))
    window._recorder._t0 -= 30.0
    window.stop_session()
    assert window._history.session_count == 1  # список оновився після сесії
    window._summary_window.close()
