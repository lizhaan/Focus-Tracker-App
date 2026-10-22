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


@pytest.fixture
def main_window(qapp):
    from attention_tracker.storage.storage import Storage
    from attention_tracker.ui.main_window import MainWindow

    storage = Storage(":memory:")
    window = MainWindow(AppConfig(), storage)
    yield window, storage
    window.close()
    storage.close()


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


def test_profile_status_texts():
    from attention_tracker.ui.main_window import ProfileStatus, profile_status
    from tests.test_storage import make_profile

    profile = make_profile(1, monitor_id="mon", camera_id="index:0")
    assert profile_status(None, "mon", "index:0", (1280, 720))[0] is ProfileStatus.MISSING
    assert profile_status(profile, "mon", "index:0", (640, 480))[0] is ProfileStatus.INVALID
    assert profile_status(profile, "other", "index:0", (1280, 720))[0] is ProfileStatus.INVALID
    status, text = profile_status(profile, "mon", "index:0", (1280, 720))
    assert status is ProfileStatus.VALID
    assert "2026-10-22 10:00" in text and "0.050" in text
