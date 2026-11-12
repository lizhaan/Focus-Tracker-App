"""Графік концентрації (pyqtgraph): частка фокусу по бінах і смуга станів.

Використовується наживо (біни 10 с) і в історії (біни 60 с). Вісь часу —
у хвилинах від початку сесії; біни без трекінгу (перерви, паузи) порожні.
"""

from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtWidgets import QVBoxLayout, QWidget

from attention_tracker.focus.metrics import Segment
from attention_tracker.ui import theme
from attention_tracker.ui.state_style import STATE_STYLES

# Оформлення
_SECONDS_PER_MINUTE = 60.0
_BAR_FILL = 0.8  # частка ширини біна під стовпчиком
_BARS_HEIGHT = 120
_STRIP_HEIGHT = 34
_Y_MAX = 1.05
_Y_TICKS = [(0.0, "0%"), (0.5, "50%"), (1.0, "100%")]
_MIN_SPAN_MINUTES = 1.0  # навіть на початку сесії вісь хоча б на хвилину


def _style_plot(plot: pg.PlotItem) -> None:
    plot.setMenuEnabled(False)
    plot.setMouseEnabled(x=False, y=False)
    plot.hideButtons()
    for name in ("left", "bottom"):
        axis = plot.getAxis(name)
        axis.setPen(pg.mkPen(theme.BORDER))
        axis.setTextPen(pg.mkPen(theme.MUTED))


class FocusChart(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._view = pg.GraphicsLayoutWidget()
        self._view.setBackground(theme.SURFACE)
        self._view.setFixedHeight(_BARS_HEIGHT + _STRIP_HEIGHT + 30)

        self._bars = self._view.addPlot(row=0, col=0)
        _style_plot(self._bars)
        self._bars.setYRange(0.0, _Y_MAX, padding=0)
        self._bars.getAxis("left").setTicks([_Y_TICKS])
        self._bars.getAxis("bottom").setStyle(showValues=False)
        self._bars.setMaximumHeight(_BARS_HEIGHT)

        self._strip = self._view.addPlot(row=1, col=0)
        _style_plot(self._strip)
        self._strip.setYRange(0.0, 1.0, padding=0)
        self._strip.getAxis("left").setTicks([[]])
        self._strip.getAxis("left").setWidth(self._bars.getAxis("left").width())
        self._strip.setLabel("bottom", "хв", color=theme.MUTED)
        self._strip.setMaximumHeight(_STRIP_HEIGHT + 24)
        self._strip.setXLink(self._bars)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)
        self.clear()

    def clear(self) -> None:
        self._bars.clear()
        self._strip.clear()
        self._bars.setXRange(0.0, _MIN_SPAN_MINUTES, padding=0)
        self.bar_count = 0
        self.segment_count = 0

    def set_data(
        self, shares: list[float | None], bin_s: float, segments: list[Segment], end: float
    ) -> None:
        """shares — частка фокусу по бінах (None — без трекінгу), segments — відрізки станів."""
        self._bars.clear()
        self._strip.clear()
        bin_min = bin_s / _SECONDS_PER_MINUTE

        filled = [(i, s) for i, s in enumerate(shares) if s is not None]
        if filled:
            self._bars.addItem(pg.BarGraphItem(
                x=[(i + 0.5) * bin_min for i, _ in filled],
                height=[s for _, s in filled],
                width=bin_min * _BAR_FILL,
                brush=theme.ACCENT,
                pen=pg.mkPen(None),
            ))

        if segments:
            self._strip.addItem(pg.BarGraphItem(
                x0=[s.start / _SECONDS_PER_MINUTE for s in segments],
                width=[s.duration / _SECONDS_PER_MINUTE for s in segments],
                y0=0.0,
                height=1.0,
                brushes=[STATE_STYLES[s.state].color for s in segments],
                pen=pg.mkPen(None),
            ))

        span = max(end / _SECONDS_PER_MINUTE, _MIN_SPAN_MINUTES)
        self._bars.setXRange(0.0, span, padding=0)
        self.bar_count = len(filled)
        self.segment_count = len(segments)
