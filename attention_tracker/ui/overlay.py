"""Плашка поверх усіх вікон у LOST_FOCUS / RECOVERING.

Не забирає фокус введення і пропускає кліки мишею крізь себе.
Показується вгорі по центру основного монітора (вибір робочого монітора —
у наступному етапі), з'являється і зникає плавно.
"""

from __future__ import annotations

from PySide6.QtCore import QPropertyAnimation, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget

from attention_tracker.models import FocusState
from attention_tracker.ui.state_style import style_for

# Оформлення плашки
_WIDTH = 420
_TOP_MARGIN = 24
_PROGRESS_STEPS = 100
_STYLE = """
QWidget#overlayBody {
    background-color: rgba(28, 28, 28, 225);
    border-radius: 14px;
    border: 2px solid %(accent)s;
}
QLabel { color: white; font-size: 20px; font-weight: 600; }
QProgressBar {
    background-color: rgba(255, 255, 255, 40);
    border: none; border-radius: 4px; max-height: 8px;
}
QProgressBar::chunk { background-color: %(accent)s; border-radius: 4px; }
"""


class FocusOverlay(QWidget):
    def __init__(self, fade_ms: int) -> None:
        super().__init__(
            None,
            Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
            | Qt.WindowType.WindowTransparentForInput,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        body = QWidget(self)
        body.setObjectName("overlayBody")
        self._text = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self._progress = QProgressBar(textVisible=False)
        self._progress.setRange(0, _PROGRESS_STEPS)
        inner = QVBoxLayout(body)
        inner.setContentsMargins(24, 16, 24, 16)
        inner.addWidget(self._text)
        inner.addWidget(self._progress)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(body)
        self.setFixedWidth(_WIDTH)

        self._fade = QPropertyAnimation(self, b"windowOpacity", self)
        self._fade.setDuration(fade_ms)
        self._fade.finished.connect(self._on_fade_finished)
        self._shown_state: FocusState | None = None

    @property
    def shown_state(self) -> FocusState | None:
        """Стан, для якого плашка зараз показана (або з'являється); None — прихована."""
        return self._shown_state

    def show_state(self, state: FocusState | None, progress: float | None) -> None:
        style = style_for(state)
        if style.overlay_text is None:
            self._fade_out()
            return

        if state is not self._shown_state:
            self.setStyleSheet(_STYLE % {"accent": style.color})
            self._text.setText(style.overlay_text)
            self._progress.setVisible(style.overlay_progress)
            self.adjustSize()
        if style.overlay_progress:
            self._progress.setValue(round((progress or 0.0) * _PROGRESS_STEPS))
        if self._shown_state is None:
            self._fade_in()
        self._shown_state = state

    def _fade_in(self) -> None:
        self._place_on_screen()
        self._fade.stop()
        if not self.isVisible():
            self.setWindowOpacity(0.0)
            self.show()
        self._fade.setStartValue(self.windowOpacity())
        self._fade.setEndValue(1.0)
        self._fade.start()

    def _fade_out(self) -> None:
        if self._shown_state is None:
            return
        self._shown_state = None
        self._fade.stop()
        self._fade.setStartValue(self.windowOpacity())
        self._fade.setEndValue(0.0)
        self._fade.start()

    def _on_fade_finished(self) -> None:
        if self._shown_state is None:
            self.hide()

    def _place_on_screen(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()
        self.adjustSize()
        self.move(area.center().x() - self.width() // 2, area.top() + _TOP_MARGIN)
