"""Повноекранна ціль у центрі робочого монітора на час перецентрування.

Збір і обчислення — у worker-і (FrameProcessor); вікно лише показує точку,
що стискається за час збору, і підказку, якщо збір довелося повторити.
Esc — скасувати старт сесії.
"""

from __future__ import annotations

import time

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QKeyEvent, QPainter, QScreen
from PySide6.QtWidgets import QWidget

# Оформлення
_BG = QColor("#111418")
_TEXT = QColor("#e8eaed")
_HINT = QColor("#ffb74d")
_TARGET = QColor("#ffffff")
_TARGET_RADIUS = 28.0
_TARGET_MIN_FRACTION = 0.3
_DOT_RADIUS = 4.0
_TEXT_TOP = 0.12
_TICK_MS = 33

_TEXT_MAIN = "Дивіться на точку в центрі — налаштовуємось під ваше положення"
_TEXT_RETRY = "Не вдалося зібрати дані: перевірте світло і дивіться на точку"
_TEXT_FOOTER = "Esc — скасувати"


class RecenterWindow(QWidget):
    cancelled = Signal()

    def __init__(self, duration: float, screen: QScreen | None) -> None:
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self._duration = duration
        self._screen = screen
        self._t0 = time.monotonic()
        self._retry = False
        self._finished = False
        self.setCursor(Qt.CursorShape.BlankCursor)
        self._tick = QTimer(self)
        self._tick.setInterval(_TICK_MS)
        self._tick.timeout.connect(self.update)

    def open_on_screen(self) -> None:
        screen = self._screen or QGuiApplication.primaryScreen()
        if screen is not None:
            self.setGeometry(screen.geometry())
        self.showFullScreen()
        self.activateWindow()
        self._tick.start()

    def restart(self) -> None:
        """Збір починається знову (попередній був невдалим)."""
        self._t0 = time.monotonic()
        self._retry = True
        self.update()

    def finish(self) -> None:
        """Перецентрування завершено — закрити без сигналу скасування."""
        self._finished = True
        self.close()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 (Qt API)
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        self._tick.stop()
        if not self._finished:
            self._finished = True
            self.cancelled.emit()
        super().closeEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), _BG)
        w, h = self.width(), self.height()

        p.setPen(_TEXT)
        p.setFont(QFont(self.font().family(), 20))
        p.drawText(QRectF(0, h * _TEXT_TOP, w, h * 0.1), Qt.AlignmentFlag.AlignHCenter, _TEXT_MAIN)
        p.setPen(_HINT if self._retry else _TEXT)
        p.setFont(QFont(self.font().family(), 14))
        footer = _TEXT_RETRY if self._retry else _TEXT_FOOTER
        p.drawText(QRectF(0, h * (_TEXT_TOP + 0.08), w, h * 0.1), Qt.AlignmentFlag.AlignHCenter, footer)

        progress = min(1.0, (time.monotonic() - self._t0) / self._duration)
        radius = _TARGET_RADIUS * (1.0 - (1.0 - _TARGET_MIN_FRACTION) * progress)
        center = QPointF(w / 2, h / 2)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(_TARGET)
        p.drawEllipse(center, radius, radius)
        p.setBrush(_BG)
        p.drawEllipse(center, _DOT_RADIUS, _DOT_RADIUS)
        p.end()
