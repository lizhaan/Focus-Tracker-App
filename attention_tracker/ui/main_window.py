"""Головне вікно: робочий монітор і профіль калібрування, Старт/Стоп, індикатор стану,
плашка, вкладка Debug.

Без дійсного профілю для робочого монітора кнопка «Старт» неактивна.
Pomodoro, налаштування й історія — у наступних кроках.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from attention_tracker.calibration.profile import CalibrationProfile
from attention_tracker.capture.camera import camera_id_for
from attention_tracker.config import AppConfig
from attention_tracker.models import FocusState, FrameResult, StateChange
from attention_tracker.platform.monitors import MonitorInfo, find_screen, list_monitors
from attention_tracker.storage.storage import DEFAULT_USER, Storage
from attention_tracker.ui.calibration_window import CalibrationWindow
from attention_tracker.ui.debug_view import DebugView
from attention_tracker.ui.overlay import FocusOverlay
from attention_tracker.ui.state_style import style_for
from attention_tracker.workers.pipeline_worker import PipelineWorker

# Оформлення
_WINDOW_SIZE = (960, 600)
_INDICATOR_SIZE = 96
_LABEL_STYLE = "font-size: 26px; font-weight: 600;"
_HINT_STYLE = "color: #757575; font-size: 14px;"
_OK_STYLE = "color: #2e7d32;"
_BAD_STYLE = "color: #c62828;"
_MS_PER_S = 1000
_BASELINE_HINT = "Перші {sec:g} с дивіться на екран — запамʼятовуємо положення голови"

# Ключі таблиці settings
SETTING_WORK_MONITOR = "work_monitor"
SETTING_FRAME_SIZE = "camera.frame_size"  # остання фактична роздільна здатність, "WxH"


class ProfileStatus(Enum):
    MISSING = "MISSING"
    INVALID = "INVALID"
    VALID = "VALID"


def profile_status(
    profile: CalibrationProfile | None, monitor_id: str, camera_id: str, frame_size: tuple[int, int]
) -> tuple[ProfileStatus, str]:
    """Статус профілю робочого монітора і текст для головного вікна."""
    if profile is None:
        return ProfileStatus.MISSING, "Не відкалібровано — натисніть «Калібрувати»"
    if not profile.is_valid_for(monitor_id, camera_id, *frame_size):
        return ProfileStatus.INVALID, "Профіль недійсний (змінилась камера або роздільна здатність) — відкалібруйте знову"
    date = profile.created_at.replace("T", " ")[:16]
    return ProfileStatus.VALID, f"✔ Відкалібровано {date}, похибка {profile.calib_error:.3f}"


def _parse_size(text: str | None) -> tuple[int, int] | None:
    if not text:
        return None
    w, _, h = text.partition("x")
    return int(w), int(h)


class MainWindow(QMainWindow):
    def __init__(self, cfg: AppConfig, storage: Storage) -> None:
        super().__init__()
        self._cfg = cfg
        self._storage = storage
        self._user_id = storage.ensure_user(DEFAULT_USER, datetime.now().isoformat(timespec="seconds"))
        self._camera_id = camera_id_for(cfg.camera.index)
        self._worker: PipelineWorker | None = None
        self._calibration: CalibrationWindow | None = None
        self._state: FocusState | None = None
        self._monitors: list[MonitorInfo] = []
        self._profile_status = ProfileStatus.MISSING
        self.setWindowTitle("Attention Tracker")
        self.resize(*_WINDOW_SIZE)

        self._overlay = FocusOverlay(cfg.ui.overlay_fade_ms)
        self._debug = DebugView()

        # робочий монітор і профіль
        self._monitor_box = QComboBox()
        self._monitor_box.currentIndexChanged.connect(self._on_monitor_changed)
        self._profile_label = QLabel()
        self._profile_label.setWordWrap(True)
        self._calibrate = QPushButton("Калібрувати")
        self._calibrate.clicked.connect(self.open_calibration)
        monitor_row = QHBoxLayout()
        monitor_row.addWidget(QLabel("Робочий монітор:"))
        monitor_row.addWidget(self._monitor_box, 1)
        monitor_row.addWidget(self._calibrate)

        # стан сесії
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
        col.addLayout(monitor_row)
        col.addWidget(self._profile_label)
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

        app = QGuiApplication.instance()
        app.screenAdded.connect(lambda _s: self.reload_monitors())
        app.screenRemoved.connect(lambda _s: self.reload_monitors())
        self.reload_monitors()
        self._render_state(None)

    # --- монітор і профіль --------------------------------------------------

    @property
    def work_monitor(self) -> MonitorInfo | None:
        index = self._monitor_box.currentIndex()
        return self._monitors[index] if 0 <= index < len(self._monitors) else None

    @property
    def profile_state(self) -> ProfileStatus:
        return self._profile_status

    def reload_monitors(self) -> None:
        """Заповнює список моніторів; вибір — збережений або основний."""
        self._monitors = list_monitors()
        saved = self._storage.get_setting(SETTING_WORK_MONITOR)
        ids = [m.monitor_id for m in self._monitors]
        if saved in ids:
            index = ids.index(saved)
        else:
            index = next((i for i, m in enumerate(self._monitors) if m.is_primary), 0)
        self._monitor_box.blockSignals(True)
        self._monitor_box.clear()
        self._monitor_box.addItems([m.name for m in self._monitors])
        self._monitor_box.setCurrentIndex(index)
        self._monitor_box.blockSignals(False)
        self._on_monitor_changed(index)

    @Slot(int)
    def _on_monitor_changed(self, _index: int) -> None:
        monitor = self.work_monitor
        if monitor is not None:
            self._storage.set_setting(SETTING_WORK_MONITOR, monitor.monitor_id)
            self._overlay.set_screen(find_screen(monitor.monitor_id))
        self.refresh_profile()

    def _frame_size(self) -> tuple[int, int]:
        saved = _parse_size(self._storage.get_setting(SETTING_FRAME_SIZE))
        return saved or (self._cfg.camera.width, self._cfg.camera.height)

    def refresh_profile(self) -> None:
        monitor = self.work_monitor
        profile = None
        if monitor is not None:
            profile = self._storage.load_profile(self._user_id, monitor.monitor_id, self._camera_id)
        self._profile_status, text = profile_status(
            profile, monitor.monitor_id if monitor else "", self._camera_id, self._frame_size()
        )
        self._profile_label.setText(text)
        self._profile_label.setStyleSheet(_OK_STYLE if self._profile_status is ProfileStatus.VALID else _BAD_STYLE)
        self._update_buttons()

    @Slot()
    def open_calibration(self) -> None:
        monitor = self.work_monitor
        if monitor is None or self._worker is not None or self._calibration is not None:
            return
        window = CalibrationWindow(self._cfg, self._user_id, monitor, find_screen(monitor.monitor_id))
        window.profile_ready.connect(self._on_profile_ready)
        window.closed.connect(self._on_calibration_closed)
        self._calibration = window
        self._update_buttons()
        window.open_on_screen()

    @Slot(object)
    def _on_profile_ready(self, profile: CalibrationProfile) -> None:
        monitor = self.work_monitor
        self._storage.save_profile(profile, monitor.name if monitor else None)
        self._storage.set_setting(SETTING_FRAME_SIZE, f"{profile.frame_w}x{profile.frame_h}")
        self.refresh_profile()

    @Slot()
    def _on_calibration_closed(self) -> None:
        self._calibration = None
        self._update_buttons()

    # --- сесія -----------------------------------------------------------

    @Slot()
    def start_session(self) -> None:
        if self._worker is not None or self._profile_status is not ProfileStatus.VALID:
            return
        worker = PipelineWorker(self._cfg)
        worker.camera_opened.connect(self._on_camera_opened)
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

    @Slot(str, int, int)
    def _on_camera_opened(self, camera_id: str, frame_w: int, frame_h: int) -> None:
        """Фактична роздільна здатність могла відрізнитись від профілю."""
        self._storage.set_setting(SETTING_FRAME_SIZE, f"{frame_w}x{frame_h}")
        self.refresh_profile()
        if self._profile_status is not ProfileStatus.VALID:
            self.statusBar().showMessage("Камера або її роздільна здатність змінились — потрібне калібрування")
            self.stop_session()

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
        calibrating = self._calibration is not None
        self._start.setEnabled(not running and not calibrating and self._profile_status is ProfileStatus.VALID)
        self._stop.setEnabled(running)
        self._calibrate.setEnabled(not running and not calibrating)
        self._monitor_box.setEnabled(not running and not calibrating)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        self.stop_session()
        if self._calibration is not None:
            self._calibration.close()
        self._overlay.close()
        super().closeEvent(event)
