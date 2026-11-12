"""Вікно «Підсумки сесії»: головні метрики, причини втрат фокусу, графік концентрації.

Відкривається після завершення сесії; те саме вікно показує сесію з історії.
"""

from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from attention_tracker.models import Reason
from attention_tracker.session.summary import SessionSummary
from attention_tracker.ui.charts import FocusChart

# Оформлення
_WINDOW_SIZE = (760, 720)
_MARGINS = (28, 24, 28, 24)
_TILE_COLUMNS = 3
_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 3600
_PERCENT = 100

REASON_NAMES: dict[str, str] = {
    Reason.GAZE_OUT.value: "Погляд поза екраном",
    Reason.PHONE_EYES.value: "Телефон (погляд униз)",
    Reason.PHONE_HEAD.value: "Телефон (нахил голови)",
    Reason.FACE_ABSENT.value: "Обличчя не видно",
    Reason.HEAD_AWAY.value: "Голова сильно відвернута",
}
_MONTHS = ("січня", "лютого", "березня", "квітня", "травня", "червня",
           "липня", "серпня", "вересня", "жовтня", "листопада", "грудня")


def format_duration(seconds: float) -> str:
    """45 с · 12 хв 30 с · 1 год 05 хв."""
    total = max(0, round(seconds))
    hours, rest = divmod(total, _SECONDS_PER_HOUR)
    minutes, secs = divmod(rest, _SECONDS_PER_MINUTE)
    if hours:
        return f"{hours} год {minutes:02d} хв"
    if minutes:
        return f"{minutes} хв {secs:02d} с" if secs else f"{minutes} хв"
    return f"{secs} с"


def format_started_at(iso: str) -> str:
    dt = datetime.fromisoformat(iso)
    return f"{dt.day} {_MONTHS[dt.month - 1]}, {dt:%H:%M}"


def _tile(value: str, label: str) -> QFrame:
    tile = QFrame(objectName="card")
    layout = QVBoxLayout(tile)
    layout.setContentsMargins(16, 14, 16, 14)
    value_label = QLabel(value, objectName="tileValue")
    caption = QLabel(label, objectName="tileLabel")
    caption.setWordWrap(True)
    layout.addWidget(value_label)
    layout.addWidget(caption)
    return tile


def _card(title: str, body: QWidget) -> QFrame:
    card = QFrame(objectName="card")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(16, 12, 16, 14)
    layout.addWidget(QLabel(title, objectName="cardTitle"))
    layout.addWidget(body)
    return card


class SessionSummaryView(QWidget):
    """Вміст підсумків: заголовок, плитки, причини, графік (у вікні й на вкладці «Історія»)."""

    def __init__(
        self, summary: SessionSummary, pomodoro: bool, title: str | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        m = summary.metrics

        subtitle = QLabel(
            f"{format_started_at(summary.started_at)}  ·  {format_duration(summary.duration)}"
            f"  (відстежено {format_duration(m.t_tracked)})",
            objectName="subtitle",
        )

        ratio = "—" if m.focus_ratio is None else f"{round(m.focus_ratio * _PERCENT)}%"
        tiles = [
            (ratio, "часу у фокусі"),
            (str(m.n_lost_episodes), "втрат фокусу"),
            (format_duration(m.mean_lost_duration) if m.n_lost_episodes else "—", "середня втрата фокусу"),
            (format_duration(m.longest_focus), "найдовше у фокусі без перерви"),
            (format_duration(m.t_lost), "загалом з утраченим фокусом"),
        ]
        if pomodoro:
            tiles.append((str(m.n_pomodoros), "помодоро"))
        self.tile_values = [value for value, _ in tiles]
        grid = QGridLayout()
        grid.setSpacing(12)
        for i, (value, label) in enumerate(tiles):
            grid.addWidget(_tile(value, label), i // _TILE_COLUMNS, i % _TILE_COLUMNS)

        self.reason_lines = [
            f"{REASON_NAMES.get(reason, reason)} — {count}"
            for reason, count in sorted(m.lost_by_reason.items(), key=lambda kv: -kv[1])
        ]
        reasons = QLabel("\n".join(self.reason_lines) or "Фокус не втрачався")

        self.chart = FocusChart()
        self.chart.set_data(summary.shares, summary.bin_s, summary.segments, summary.duration)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)
        if title:
            layout.addWidget(QLabel(title, objectName="windowTitle"))
        layout.addWidget(subtitle)
        layout.addLayout(grid)
        layout.addWidget(_card("Причини втрат фокусу", reasons))
        layout.addWidget(_card("Концентрація", self.chart))


class SessionSummaryWindow(QWidget):
    """Окреме вікно з підсумками щойно завершеної сесії."""

    def __init__(self, summary: SessionSummary, pomodoro: bool, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Підсумки сесії")
        self.resize(*_WINDOW_SIZE)
        self.view = SessionSummaryView(summary, pomodoro, title="Підсумки сесії")

        close = QPushButton("Закрити")
        close.clicked.connect(self.close)
        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close_row.addWidget(close)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(*_MARGINS)
        layout.addWidget(self.view)
        layout.addStretch(1)
        layout.addLayout(close_row)

    @property
    def tile_values(self) -> list[str]:
        return self.view.tile_values

    @property
    def reason_lines(self) -> list[str]:
        return self.view.reason_lines

    @property
    def chart(self) -> FocusChart:
        return self.view.chart
