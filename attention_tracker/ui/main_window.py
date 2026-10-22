"""Головне вікно (альфа): Старт/Стоп, індикатор стану, плашка, вкладка Debug.

Pomodoro, історія, налаштування і калібрування — у наступному етапі.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from attention_tracker.config import AppConfig
from attention_tracker.models import FocusState, FrameResult, StateChange
from attention_tracker.ui.debug_view import DebugView
from attention_tracker.ui.overlay import FocusOverlay
from attention_tracker.ui.state_style import style_for
from attention_tracker.workers.pipeline_worker import PipelineWorker

# Оформлення
_WINDOW_SIZE = (960, 600)
_INDICATOR_SIZE = 96
_LABEL_STYLE = "font-size: 26px; font-weight: 600;"
_HINT_STYLE = "color: #757575; font-size: 14px;"
_MS_PER_S = 1000
_BASELINE_HINT = "Перші {sec:g} с дивіться на екран — запамʼятовуємо положення голови"


class MainWindow(QMainWindow):
    def __init__(self, cfg: AppConfig) -> None:
        super().__init__()
        self._cfg = cfg
        self._worker: PipelineWorker | None = None
        self._state: FocusState | None = None
        self.setWindowTitle("Attention Tracker")
        self.resize(*_WINDOW_SIZE)

        self._overlay = FocusOverlay(cfg.ui.overlay_fade_ms)
        self._debug = DebugView()

        self._indicator = QLabel()
        self._indicator.setFixedSize(_INDICATOR_SIZE, _INDICATOR_SIZE)
        self._state_label = QLabel()
        self._state_label.setStyleSheet(_LABEL_STYLE)
        self._state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hint = QLabel()
        self._hint.setStyleSheet(_HINT_STYLE)
        self._hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hint.setWordWrap(True)

        self._start = QPushButton("Старт")
        self._stop = QPushButton("Стоп")
        self._start.clicked.connect(self.start_session)
        self._stop.clicked.connect(self.stop_session)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self._start)
        buttons.addWidget(self._stop)
        buttons.addStretch(1)

        session = QWidget()
        col = QVBoxLayout(session)
        col.addStretch(1)
        col.addWidget(self._indicator, alignment=Qt.AlignmentFlag.AlignCenter)
        col.addWidget(self._state_label)
        col.addWidget(self._hint)
        col.addStretch(1)
        col.addLayout(buttons)

        tabs = QTabWidget()
        tabs.addTab(session, "Сесія")
        tabs.addTab(self._debug, "Debug")
        self.setCentralWidget(tabs)

        self._baseline_timer = QTimer(self, singleShot=True)
        self._baseline_timer.timeout.connect(lambda: self._hint.clear())
        self._render_state(None)
        self._update_buttons()

    # --- сесія -----------------------------------------------------------

    @Slot()
    def start_session(self) -> None:
        if self._worker is not None:
            return
        worker = PipelineWorker(self._cfg)
        worker.frame_processed.connect(self._on_frame_processed)
        worker.state_changed.connect(self._on_state_changed)
        worker.camera_error.connect(self._on_camera_error)
        worker.finished.connect(self._on_worker_finished)
        self._debug.attach(worker)
        self._worker = worker
        self.statusBar().clearMessage()
        self._render_state(FocusState.FOCUSED)
        duration = self._cfg.alpha_thresholds.baseline_duration
        self._hint.setText(_BASELINE_HINT.format(sec=duration))
        self._baseline_timer.start(int(duration * _MS_PER_S))
        self._update_buttons()
        worker.start()

    @Slot()
    def stop_session(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait()

    @Slot()
    def _on_worker_finished(self) -> None:
        self._debug.detach()
        self._worker = None
        self._baseline_timer.stop()
        self._hint.clear()
        self._render_state(None)
        self._update_buttons()

    # --- сигнали worker-а -----------------------------------------------

    @Slot(object)
    def _on_frame_processed(self, result: FrameResult) -> None:
        if result.state is FocusState.RECOVERING:
            self._overlay.show_state(result.state, result.recovering_progress)

    @Slot(object)
    def _on_state_changed(self, change: StateChange) -> None:
        self._render_state(change.new)

    @Slot(str)
    def _on_camera_error(self, message: str) -> None:
        self.statusBar().showMessage(message)
        self._debug.show_error(message)

    # --- відображення -------------------------------------------------------

    def _render_state(self, state: FocusState | None) -> None:
        self._state = state
        style = style_for(state)
        self._indicator.setStyleSheet(
            f"background-color: {style.color}; border-radius: {_INDICATOR_SIZE // 2}px;"
        )
        self._state_label.setText(style.label)
        self._overlay.show_state(state, None)

    def _update_buttons(self) -> None:
        running = self._worker is not None
        self._start.setEnabled(not running)
        self._stop.setEnabled(running)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        self.stop_session()
        self._overlay.close()
        super().closeEvent(event)
