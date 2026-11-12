"""Вкладка «Історія»: таблиця завершених сесій і підсумки вибраної сесії."""

from __future__ import annotations

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from attention_tracker.config import MetricsConfig
from attention_tracker.storage.storage import SessionInfo, Storage
from attention_tracker.session.summary import session_duration, summary_from_storage
from attention_tracker.ui.session_summary import SessionSummaryView, format_duration, format_started_at

# Оформлення
_MARGINS = (24, 20, 24, 20)
_TABLE_HEIGHT = 220
_PERCENT = 100
_EMPTY_TEXT = "Ще немає завершених сесій. Запустіть сесію на вкладці «Сесія»."
COLUMNS = ("Дата", "Тривалість", "У фокусі", "Втрат фокусу", "Помодоро")


def session_row(info: SessionInfo, duration: float) -> list[str]:
    """Текст рядка таблиці для сесії."""
    m = info.metrics
    ratio = "—" if m is None or m.focus_ratio is None else f"{round(m.focus_ratio * _PERCENT)}%"
    pomodoros = str(m.n_pomodoros) if m is not None and info.settings.get("pomodoro", True) else "—"
    return [
        format_started_at(info.started_at),
        format_duration(duration),
        ratio,
        str(m.n_lost_episodes) if m is not None else "—",
        pomodoros,
    ]


class HistoryView(QWidget):
    def __init__(self, storage: Storage, user_id: int, cfg: MetricsConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._storage = storage
        self._user_id = user_id
        self._cfg = cfg
        self._sessions: list[SessionInfo] = []
        self._details: SessionSummaryView | None = None

        self._table = QTableWidget(0, len(COLUMNS))
        self._table.setHorizontalHeaderLabels(COLUMNS)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setShowGrid(False)
        self._table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._table.setFixedHeight(_TABLE_HEIGHT)
        self._table.currentCellChanged.connect(self._on_row_changed)

        self._empty = QLabel(_EMPTY_TEXT, objectName="sessionHint")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._details_host = QWidget()
        self._details_layout = QVBoxLayout(self._details_host)
        self._details_layout.setContentsMargins(0, 8, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._details_host)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(*_MARGINS)
        layout.addWidget(self._table)
        layout.addWidget(self._empty)
        layout.addWidget(scroll, 1)
        self.refresh()

    @property
    def session_count(self) -> int:
        return len(self._sessions)

    @property
    def selected_session_id(self) -> int | None:
        row = self._table.currentRow()
        return self._sessions[row].id if 0 <= row < len(self._sessions) else None

    @property
    def details(self) -> SessionSummaryView | None:
        return self._details

    def refresh(self) -> None:
        """Перечитати сесії з БД; вибір — найновіша сесія."""
        self._sessions = [s for s in self._storage.list_sessions(self._user_id) if s.metrics is not None]
        self._table.blockSignals(True)
        self._table.setRowCount(len(self._sessions))
        for r, info in enumerate(self._sessions):
            for c, text in enumerate(session_row(info, session_duration(self._storage, info))):
                item = QTableWidgetItem(text)
                if c:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._table.setItem(r, c, item)
        self._table.blockSignals(False)
        has_sessions = bool(self._sessions)
        self._table.setVisible(has_sessions)
        self._empty.setVisible(not has_sessions)
        if has_sessions:
            self._table.setCurrentCell(0, 0)
            self._show(0)
        else:
            self._clear_details()

    def select_row(self, row: int) -> None:
        self._table.setCurrentCell(row, 0)

    @Slot(int, int, int, int)
    def _on_row_changed(self, row: int, _col: int, _prev_row: int, _prev_col: int) -> None:
        if 0 <= row < len(self._sessions):
            self._show(row)

    def _show(self, row: int) -> None:
        info = self._sessions[row]
        summary = summary_from_storage(self._storage, info, self._cfg)
        self._clear_details()
        self._details = SessionSummaryView(summary, pomodoro=info.settings.get("pomodoro", True))
        self._details_layout.addWidget(self._details)
        self._details_layout.addStretch(1)

    def _clear_details(self) -> None:
        while self._details_layout.count():
            widget = self._details_layout.takeAt(0).widget()
            if widget is not None:
                widget.setParent(None)  # одразу прибрати з екрана, не чекаючи deleteLater
                widget.deleteLater()
        self._details = None
