import dataclasses

import pytest

from attention_tracker.config import AppConfig
from attention_tracker.models import FocusState
from attention_tracker.ui.state_style import IDLE_STYLE, STATE_STYLES, style_for


def test_every_state_has_style():
    assert set(STATE_STYLES) == set(FocusState)
    assert style_for(None) is IDLE_STYLE


@pytest.mark.parametrize(
    "state, text, progress",
    [
        (FocusState.FOCUSED, None, False),
        (FocusState.NOT_FOCUSED, None, False),
        (FocusState.LOST_FOCUS, "Фокус втрачено", False),
        (FocusState.RECOVERING, "Повертаємось у фокус…", True),
    ],
)
def test_overlay_content_has_no_reason(state, text, progress):
    style = style_for(state)
    assert style.overlay_text == text
    assert style.overlay_progress is progress


def test_overlay_window_does_not_take_focus(qapp):
    from PySide6.QtCore import Qt

    from attention_tracker.ui.overlay import FocusOverlay

    overlay = FocusOverlay(fade_ms=10)
    flags = overlay.windowFlags()
    for flag in (Qt.WindowType.WindowStaysOnTopHint, Qt.WindowType.FramelessWindowHint,
                 Qt.WindowType.Tool, Qt.WindowType.WindowTransparentForInput):
        assert flags & flag
    assert overlay.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    overlay.close()


def test_overlay_shows_and_hides(qapp):
    from PySide6.QtTest import QTest

    from attention_tracker.ui.overlay import FocusOverlay

    overlay = FocusOverlay(fade_ms=10)
    overlay.show_state(FocusState.NOT_FOCUSED, None)
    assert overlay.shown_state is None and not overlay.isVisible()

    overlay.show_state(FocusState.LOST_FOCUS, None)
    assert overlay.isVisible() and overlay.shown_state is FocusState.LOST_FOCUS

    overlay.show_state(FocusState.RECOVERING, 0.5)
    assert overlay._progress.value() == 50

    overlay.show_state(FocusState.FOCUSED, None)
    QTest.qWait(100)
    assert overlay.shown_state is None and not overlay.isVisible()
    overlay.close()


def test_main_window_idle_state(main_window):
    window, _ = main_window
    assert not window._stop.isEnabled()
    assert window._state_label.text() == IDLE_STYLE.label
    window._render_state(FocusState.LOST_FOCUS)
    assert window._state_label.text() == "Фокус втрачено"
    assert window._overlay.shown_state is FocusState.LOST_FOCUS
    window._render_state(None)


def test_start_disabled_without_profile(main_window):
    from attention_tracker.ui.main_window import ProfileStatus

    window, _ = main_window
    assert window.work_monitor is not None
    assert window.profile_state is ProfileStatus.MISSING
    assert not window._start.isEnabled()
    assert window._calibrate.isEnabled()


def test_saved_profile_enables_start_and_persists_monitor(main_window):
    from attention_tracker.ui.main_window import SETTING_WORK_MONITOR, ProfileStatus
    from tests.test_storage import make_profile

    window, storage = main_window
    monitor = window.work_monitor
    assert storage.get_setting(SETTING_WORK_MONITOR) == monitor.monitor_id
    profile = dataclasses.replace(make_profile(window._user_id, monitor_id=monitor.monitor_id), frame_w=1280, frame_h=720)
    window._on_profile_ready(profile)
    assert window.profile_state is ProfileStatus.VALID
    assert window._start.isEnabled()
    assert "Відкалібровано" in window._profile_label.text()


def test_recenter_window_cancel_and_finish(qapp):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from attention_tracker.ui.recenter_window import RecenterWindow

    cancelled = []
    window = RecenterWindow(duration=2.0, screen=None)
    window.cancelled.connect(lambda: cancelled.append(True))
    window.resize(800, 450)
    window.show()
    window.restart()
    window.grab()  # малювання з підказкою повтору
    QTest.keyClick(window, Qt.Key.Key_Escape)
    assert cancelled == [True]

    done = RecenterWindow(duration=2.0, screen=None)
    done.cancelled.connect(lambda: cancelled.append(True))
    done.show()
    done.finish()
    assert cancelled == [True]  # завершення — не скасування


@pytest.mark.parametrize(
    "seconds, text", [(1500.0, "25:00"), (59.2, "01:00"), (0.4, "00:01"), (0.0, "00:00"), (-3.0, "00:00")]
)
def test_format_remaining(seconds, text):
    from attention_tracker.ui.main_window import format_remaining

    assert format_remaining(seconds) == text


def test_pomodoro_controls_without_camera(main_window):
    import time

    from attention_tracker.models import PomodoroPhase
    from attention_tracker.ui.state_style import BREAK_STYLE, PAUSE_STYLE

    window, _ = main_window
    assert not window._pause.isEnabled() and not window._skip.isEnabled()
    window._pomodoro.start(time.monotonic())
    assert window._pause.isEnabled() and window._skip.isEnabled()
    assert window._state_label.text() == "У фокусі"
    assert "Робота" in window._phase_label.text()
    assert window._timer_label.text() == "25:00"

    window.toggle_pause()
    assert window._state_label.text() == PAUSE_STYLE.label
    assert window._pause.text() == "Продовжити"
    window.toggle_pause()
    assert window._state_label.text() == "У фокусі"

    window.skip_phase()
    assert window._pomodoro.phase is PomodoroPhase.SHORT_BREAK
    assert window._state_label.text() == BREAK_STYLE.label
    assert window._overlay.shown_state is None


@pytest.mark.parametrize(
    "seconds, text", [(0.0, "00:00"), (59.9, "00:59"), (754.0, "12:34"), (3725.0, "1:02:05")]
)
def test_format_elapsed(seconds, text):
    from attention_tracker.ui.main_window import format_elapsed

    assert format_elapsed(seconds) == text


def test_session_without_pomodoro(main_window):
    import time

    from attention_tracker.models import PomodoroPhase
    from attention_tracker.ui.main_window import SETTING_POMODORO_ENABLED
    from attention_tracker.ui.state_style import PAUSE_STYLE

    window, storage = main_window
    assert window._pomodoro_box.isChecked()  # за замовчуванням Pomodoro увімкнено
    window._pomodoro_box.setChecked(False)
    assert storage.get_setting(SETTING_POMODORO_ENABLED) == "0"
    assert window._timer_label.text() == "00:00"

    window._stopwatch.start(time.monotonic())  # як після перецентрування
    window._apply_phase_view()
    assert window._pomodoro.phase is PomodoroPhase.IDLE
    assert window._state_label.text() == "У фокусі"
    assert window._phase_label.text() == "Без таймера"
    assert window._pause.isEnabled() and not window._skip.isEnabled()

    window.toggle_pause()
    assert window._state_label.text() == PAUSE_STYLE.label
    assert "пауза" in window._phase_label.text()
    window.toggle_pause()
    assert window._state_label.text() == "У фокусі"


def test_pomodoro_choice_is_remembered(qapp):
    from attention_tracker.storage.storage import Storage
    from attention_tracker.ui.main_window import SETTING_POMODORO_ENABLED, MainWindow

    storage = Storage(":memory:")
    storage.set_setting(SETTING_POMODORO_ENABLED, "0")
    window = MainWindow(AppConfig(), storage)
    assert not window._pomodoro_box.isChecked()
    window.close()
    storage.close()


def test_lost_focus_auto_pauses_pomodoro(main_window):
    import time

    from attention_tracker.models import FocusState, Reason, StateChange

    window, _ = main_window
    window._pomodoro.start(time.monotonic())
    window._on_state_changed(StateChange(0.0, FocusState.NOT_FOCUSED, FocusState.LOST_FOCUS, Reason.GAZE_OUT))
    assert window._pomodoro.paused_auto
    assert "автопауза" in window._phase_label.text()
    window._on_state_changed(StateChange(1.0, FocusState.LOST_FOCUS, FocusState.RECOVERING, Reason.NONE))
    assert window._pomodoro.paused_auto  # RECOVERING — таймер ще стоїть
    window._on_state_changed(StateChange(2.0, FocusState.RECOVERING, FocusState.FOCUSED, Reason.NONE))
    assert not window._pomodoro.paused


def test_settings_view_emits_and_resets(qapp):
    from attention_tracker.storage.user_settings import UserSettings
    from attention_tracker.ui.settings_view import SettingsView

    defaults = UserSettings.defaults(AppConfig())
    view = SettingsView(AppConfig(), dataclasses.replace(defaults, n=7.0))
    received = []
    view.settings_changed.connect(received.append)
    assert view.settings().n == 7.0
    view._y.setValue(4.5)
    assert received[-1].y == 4.5 and received[-1].n == 7.0
    view.reset_to_defaults()
    assert received[-1] == defaults
    view.set_camera_editable(False)
    assert not view._camera_index.isEnabled() and not view._resolution.isEnabled()


def test_settings_change_applies_and_persists(main_window):
    from attention_tracker.storage.user_settings import load_settings
    from attention_tracker.ui.main_window import ProfileStatus

    window, storage = main_window
    view = window._settings_view
    view._work.setValue(50)
    assert window._cfg.pomodoro.work == 3000.0
    assert window._timer_label.text() == "50:00"
    assert load_settings(storage, AppConfig()).work_minutes == 50

    view._n.setValue(6.0)
    assert window._cfg.focus.n == 6.0

    view._camera_index.setValue(1)
    assert window._camera_id == "index:1"
    assert window.profile_state is ProfileStatus.MISSING


def test_saved_settings_used_on_start(qapp):
    from attention_tracker.storage.storage import Storage
    from attention_tracker.storage.user_settings import KEY_WORK
    from attention_tracker.ui.main_window import MainWindow

    storage = Storage(":memory:")
    storage.set_setting(KEY_WORK, "45")
    window = MainWindow(AppConfig(), storage)
    assert window._cfg.pomodoro.work == 2700.0
    assert window._timer_label.text() == "45:00"
    assert window._settings_view.settings().work_minutes == 45
    window.close()
    storage.close()


def test_theme_assets_exist_and_apply(qapp):
    import re
    from pathlib import Path

    from attention_tracker.ui.theme import STYLESHEET, apply_theme, camera_icon, eye_icon, timer_icon

    for path in re.findall(r"url\(([^)]+)\)", STYLESHEET):
        assert Path(path).is_file(), path
    apply_theme(qapp)
    assert qapp.styleSheet() == STYLESHEET
    for icon in (eye_icon(), timer_icon(), camera_icon()):
        assert not icon.isNull()
    qapp.setStyleSheet("")


def test_session_is_recorded(main_window):
    from attention_tracker.calibration.recenter import RecenterResult, SessionAdjustment
    from attention_tracker.models import FocusState, Reason, StateChange

    window, storage = main_window
    window._on_recenter_done(RecenterResult(SessionAdjustment(0.0, 0.0), large_shift=False))
    assert window._recorder is not None
    window._on_state_changed(StateChange(0.0, FocusState.FOCUSED, FocusState.NOT_FOCUSED, Reason.GAZE_OUT))
    window._recorder._t0 -= 30.0  # ніби сесія триває вже 30 с
    assert window._summary_window is None  # під час сесії підсумків немає
    window.stop_session()
    assert window._recorder is None

    summary = window._summary_window  # після «Стоп» — окреме вікно з підсумками
    assert summary is not None and summary.isVisible()
    assert summary.chart.segment_count >= 1
    assert summary.tile_values[1] == "0"  # втрат фокусу не було
    summary.close()

    [info] = storage.list_sessions(window._user_id)
    assert info.ended_at is not None and info.metrics is not None
    assert info.settings["pomodoro"] is True
    events = storage.pomodoro_events(info.id)
    assert [e.event.value for e in events] == ["phase_start", "stop"]
    assert len(storage.focus_events(info.id)) == 1


def test_no_summary_when_cancelled_before_tracking(main_window):
    window, storage = main_window
    window.stop_session()  # скасовано під час точки в центрі — запису ще немає
    assert window._summary_window is None
    assert storage.list_sessions(window._user_id) == []


def test_free_session_is_recorded(main_window):
    from attention_tracker.calibration.recenter import RecenterResult, SessionAdjustment

    window, storage = main_window
    window._pomodoro_box.setChecked(False)
    window._on_recenter_done(RecenterResult(SessionAdjustment(0.0, 0.0), large_shift=False))
    window.toggle_pause()
    window.toggle_pause()
    window.stop_session()
    [info] = storage.list_sessions(window._user_id)
    events = [e.event.value for e in storage.pomodoro_events(info.id)]
    assert events == ["phase_start", "manual_pause", "manual_resume", "stop"]
    assert info.settings["pomodoro"] is False


def test_unfinished_session_closed_on_start(qapp):
    from attention_tracker.session.recorder import SessionRecorder
    from attention_tracker.storage.storage import DEFAULT_USER, Storage
    from attention_tracker.ui.main_window import MainWindow

    storage = Storage(":memory:")
    user = storage.ensure_user(DEFAULT_USER, "2026-11-12T10:00:00")
    SessionRecorder(storage, user, "mon", None, {}, "2026-11-12T10:00:00", 0.0)  # «упала» без finish
    window = MainWindow(AppConfig(), storage)
    [info] = storage.list_sessions(user)
    assert info.ended_at is not None
    window.close()
    storage.close()


def test_profile_status_texts():
    from attention_tracker.ui.main_window import ProfileStatus, profile_status
    from tests.test_storage import make_profile

    profile = make_profile(1, monitor_id="mon", camera_id="index:0")
    assert profile_status(None, "mon", "index:0", (1280, 720))[0] is ProfileStatus.MISSING
    assert profile_status(profile, "mon", "index:0", (640, 480))[0] is ProfileStatus.INVALID
    assert profile_status(profile, "other", "index:0", (1280, 720))[0] is ProfileStatus.INVALID
    old_status, old_text = profile_status(dataclasses.replace(profile, schema_version=1), "mon", "index:0", (1280, 720))
    assert old_status is ProfileStatus.INVALID and "застарів" in old_text
    status, text = profile_status(profile, "mon", "index:0", (1280, 720))
    assert status is ProfileStatus.VALID
    assert "2026-10-22 10:00" in text and "0.050" in text
