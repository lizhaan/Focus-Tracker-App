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


def test_main_window_idle_state(qapp):
    from attention_tracker.ui.main_window import MainWindow

    window = MainWindow(AppConfig())
    assert window._start.isEnabled() and not window._stop.isEnabled()
    assert window._state_label.text() == IDLE_STYLE.label
    window._render_state(FocusState.LOST_FOCUS)
    assert window._state_label.text() == "Фокус втрачено"
    assert window._overlay.shown_state is FocusState.LOST_FOCUS
    window._render_state(None)
    window.close()
