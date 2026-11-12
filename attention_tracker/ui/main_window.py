"""Головне вікно: робочий монітор і профіль калібрування, Pomodoro (Старт / Пауза /
Стоп / Пропустити фазу), індикатор стану, плашка, вкладка Debug.

Без дійсного профілю для робочого монітора кнопка «Старт» неактивна.
У робочій фазі LOST_FOCUS ставить таймер на автопаузу, повернення в
FOCUSED після RECOVERING — знімає її. На перервах і ручній паузі трекінг вимкнено.
Pomodoro можна вимкнути: тоді сесія — лише трекінг, а таймер рахує її тривалість.
Налаштування й історія — у наступних кроках.
"""

from __future__ import annotations

import time
from datetime import datetime
from enum import Enum

from PySide6.QtCore import QPointF, Qt, QTimer, Slot
from PySide6.QtGui import QColor, QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from attention_tracker.calibration.profile import PROFILE_SCHEMA_VERSION, CalibrationProfile
from attention_tracker.calibration.recenter import RecenterResult
from attention_tracker.capture.camera import camera_id_for
from attention_tracker.config import AppConfig
from attention_tracker.focus.pomodoro import PomodoroEvent, PomodoroEventInfo, PomodoroTimer
from attention_tracker.focus.stopwatch import Stopwatch
from attention_tracker.models import FocusState, FrameResult, PomodoroPhase, StateChange
from attention_tracker.platform.monitors import MonitorInfo, find_screen, list_monitors
from attention_tracker.session.recorder import SessionRecorder, recover_unfinished
from attention_tracker.storage.storage import DEFAULT_USER, Storage
from attention_tracker.storage.user_settings import (
    UserSettings,
    apply_settings,
    load_settings,
    save_settings,
    validate,
)
from attention_tracker.ui.calibration_window import CalibrationWindow
from attention_tracker.ui.debug_view import DebugView
from attention_tracker.ui.overlay import FocusOverlay
from attention_tracker.ui.recenter_window import RecenterWindow
from attention_tracker.ui.settings_view import SettingsView
from attention_tracker.ui.state_style import BREAK_STYLE, PAUSE_STYLE, StateStyle, style_for
from attention_tracker.workers.pipeline_worker import PipelineWorker, WorkerMode

# Оформлення
_WINDOW_SIZE = (720, 760)
_INDICATOR_SIZE = 136
_GLOW_BLUR = 48
_GLOW_ALPHA = 110
_BUTTON_MIN_WIDTH = 84
_SECTION_SPACING = 16
_RECENTER_HINT = "Дивіться на точку в центрі робочого монітора"
_LARGE_SHIFT_HINT = "Положення сильно змінилось — рекомендуємо повне калібрування"
_FREE_SESSION_NAME = "Без таймера"
_MS_PER_S = 1000
_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 3600

PHASE_NAMES: dict[PomodoroPhase, str] = {
    PomodoroPhase.IDLE: "",
    PomodoroPhase.WORK: "Робота",
    PomodoroPhase.SHORT_BREAK: "Коротка перерва",
    PomodoroPhase.LONG_BREAK: "Довга перерва",
}

# Ключі таблиці settings
SETTING_WORK_MONITOR = "work_monitor"
SETTING_FRAME_SIZE = "camera.frame_size"  # остання фактична роздільна здатність, "WxH"
SETTING_POMODORO_ENABLED = "pomodoro.enabled"  # "1" / "0"


def format_remaining(seconds: float) -> str:
    """Залишок часу фази як ММ:СС (округлення вгору, щоб 0:00 було лише в кінці)."""
    total = max(0, int(-(-seconds // 1)))
    return f"{total // _SECONDS_PER_MINUTE:02d}:{total % _SECONDS_PER_MINUTE:02d}"


def format_elapsed(seconds: float) -> str:
    """Тривалість сесії: ММ:СС, від години — Г:ММ:СС."""
    total = max(0, int(seconds))
    hours, rest = divmod(total, _SECONDS_PER_HOUR)
    minutes, secs = divmod(rest, _SECONDS_PER_MINUTE)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


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
    if profile.schema_version != PROFILE_SCHEMA_VERSION:
        return ProfileStatus.INVALID, "Профіль застарів після оновлення програми — відкалібруйте знову"
    if not profile.is_valid_for(monitor_id, camera_id, *frame_size):
        return ProfileStatus.INVALID, "Профіль недійсний (змінилась камера або роздільна здатність) — відкалібруйте знову"
    date = profile.created_at.replace("T", " ")[:16]
    return ProfileStatus.VALID, f"✓ Відкалібровано {date}, похибка {profile.calib_error:.3f}"


def _parse_size(text: str | None) -> tuple[int, int] | None:
    if not text:
        return None
    w, _, h = text.partition("x")
    return int(w), int(h)


class MainWindow(QMainWindow):
    def __init__(self, cfg: AppConfig, storage: Storage) -> None:
        super().__init__()
        self._base_cfg = cfg  # стандартні значення
        self._settings = load_settings(storage, cfg)
        cfg = apply_settings(cfg, self._settings)  # з урахуванням налаштувань користувача
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
        self._settings_view = SettingsView(self._base_cfg, self._settings)
        self._settings_view.settings_changed.connect(self._on_settings_changed)

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

        # Pomodoro або секундомір сесії без таймера
        self._pomodoro = PomodoroTimer(cfg.pomodoro, self._on_pomodoro_event)
        self._stopwatch = Stopwatch()
        self._use_pomodoro = storage.get_setting(SETTING_POMODORO_ENABLED, "1") == "1"
        self._pomodoro_box = QCheckBox("Pomodoro-таймер")
        self._pomodoro_box.setChecked(self._use_pomodoro)
        self._pomodoro_box.toggled.connect(self._on_pomodoro_toggled)
        self._timer_label = QLabel(format_remaining(cfg.pomodoro.work), objectName="timer")
        self._timer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._phase_label = QLabel(objectName="phase")
        self._phase_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._tick = QTimer(self)
        self._tick.setInterval(int(cfg.ui.pomodoro_tick * _MS_PER_S))
        self._tick.timeout.connect(self._on_tick)

        # стан сесії
        self._indicator = QLabel()
        self._indicator.setFixedSize(_INDICATOR_SIZE, _INDICATOR_SIZE)
        self._glow = QGraphicsDropShadowEffect(self._indicator, blurRadius=_GLOW_BLUR, offset=QPointF(0, 0))
        self._indicator.setGraphicsEffect(self._glow)
        self._state_label = QLabel(objectName="stateLabel")
        self._state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hint = QLabel(objectName="sessionHint")
        self._hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hint.setWordWrap(True)

        self._start = QPushButton("Старт")
        self._pause = QPushButton("Пауза")
        self._stop = QPushButton("Стоп")
        self._skip = QPushButton("Пропустити фазу")
        self._start.clicked.connect(self.start_session)
        self._pause.clicked.connect(self.toggle_pause)
        self._stop.clicked.connect(self.stop_session)
        self._skip.clicked.connect(self.skip_phase)
        main_buttons = QHBoxLayout()
        main_buttons.addStretch(1)
        for button in (self._start, self._pause, self._stop):
            button.setMinimumWidth(_BUTTON_MIN_WIDTH)
            main_buttons.addWidget(button)
        main_buttons.addStretch(1)
        skip_row = QHBoxLayout()
        skip_row.addStretch(1)
        skip_row.addWidget(self._skip)
        skip_row.addStretch(1)

        session = QWidget()
        col = QVBoxLayout(session)
        col.setContentsMargins(24, 20, 24, 24)
        col.addLayout(monitor_row)
        col.addWidget(self._profile_label)
        col.addWidget(self._pomodoro_box)
        col.addStretch(1)
        col.addWidget(self._timer_label)
        col.addWidget(self._phase_label)
        col.addSpacing(_SECTION_SPACING * 2)
        col.addWidget(self._indicator, alignment=Qt.AlignmentFlag.AlignCenter)
        col.addSpacing(_SECTION_SPACING)
        col.addWidget(self._state_label)
        col.addWidget(self._hint)
        col.addStretch(1)
        col.addLayout(main_buttons)
        col.addLayout(skip_row)

        tabs = QTabWidget()
        tabs.addTab(session, "Сесія")
        tabs.addTab(self._settings_view, "Налаштування")
        tabs.addTab(self._debug, "Debug")
        self.setCentralWidget(tabs)

        self._recenter: RecenterWindow | None = None
        self._recorder: SessionRecorder | None = None
        recover_unfinished(storage)  # сесії, що обірвались при минулому запуску

        app = QGuiApplication.instance()
        app.screenAdded.connect(lambda _s: self.reload_monitors())
        app.screenRemoved.connect(lambda _s: self.reload_monitors())
        self.reload_monitors()
        self._render_state(None)
        self._update_timer()

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
        valid = self._profile_status is ProfileStatus.VALID
        self._profile_label.setObjectName("profileOk" if valid else "profileBad")
        self._profile_label.style().unpolish(self._profile_label)  # перечитати стиль за новим іменем
        self._profile_label.style().polish(self._profile_label)
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
        monitor = self.work_monitor
        profile = self._storage.load_profile(self._user_id, monitor.monitor_id, self._camera_id)
        worker = PipelineWorker(self._cfg, WorkerMode.SESSION, profile)
        worker.camera_opened.connect(self._on_camera_opened)
        worker.recenter_done.connect(self._on_recenter_done)
        worker.recenter_retry.connect(self._on_recenter_retry)
        worker.frame_processed.connect(self._on_frame_processed)
        worker.state_changed.connect(self._on_state_changed)
        worker.camera_error.connect(self._on_camera_error)
        worker.finished.connect(self._on_worker_finished)
        self._debug.attach(worker, profile)
        self._worker = worker
        self.statusBar().clearMessage()
        self._hint.setText(_RECENTER_HINT)

        recenter = RecenterWindow(self._cfg.calibration.recenter_duration, find_screen(monitor.monitor_id))
        recenter.cancelled.connect(self.stop_session)
        self._recenter = recenter
        self._update_buttons()
        worker.start()
        recenter.open_on_screen()

    @Slot()
    def stop_session(self) -> None:
        now = time.monotonic()
        self._pomodoro.stop(now)
        self._stopwatch.stop()
        self._finish_recording(now)
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait()

    @Slot()
    def toggle_pause(self) -> None:
        """Ручна пауза: таймер стоїть, трекінг вимкнено; «Продовжити» — знову «у фокусі»."""
        now = time.monotonic()
        if self._stopwatch.started:
            if self._stopwatch.paused:
                self._stopwatch.resume(now)
                self._record(PomodoroEvent.MANUAL_RESUME, PomodoroPhase.WORK, now)
            else:
                self._stopwatch.pause(now)
                self._record(PomodoroEvent.MANUAL_PAUSE, PomodoroPhase.WORK, now)
        elif self._pomodoro.paused_manual:
            self._pomodoro.resume_manual(now)
        else:
            self._pomodoro.pause_manual(now)
        self._apply_phase_view()

    @Slot(object)
    def _on_settings_changed(self, settings: UserSettings) -> None:
        """Зберегти й застосувати одразу; n і y — зокрема в поточній сесії."""
        settings = validate(settings, self._base_cfg)
        camera_changed = (settings.camera_index, settings.camera_resolution) != (
            self._settings.camera_index, self._settings.camera_resolution
        )
        self._settings = settings
        save_settings(self._storage, settings)
        self._cfg = apply_settings(self._base_cfg, settings)
        self._pomodoro.set_config(self._cfg.pomodoro)
        if self._worker is not None:
            self._worker.set_timings(settings.n, settings.y)
        if camera_changed:
            self._camera_id = camera_id_for(settings.camera_index)
            w, h = settings.camera_resolution
            self._storage.set_setting(SETTING_FRAME_SIZE, f"{w}x{h}")  # фактичну перевіримо на старті
            self.refresh_profile()
        self._update_timer()

    @Slot(bool)
    def _on_pomodoro_toggled(self, checked: bool) -> None:
        self._use_pomodoro = checked
        self._storage.set_setting(SETTING_POMODORO_ENABLED, "1" if checked else "0")
        self._update_timer()
        self._update_buttons()

    @Slot()
    def skip_phase(self) -> None:
        self._pomodoro.skip_phase(time.monotonic())

    @Slot(str, int, int)
    def _on_camera_opened(self, camera_id: str, frame_w: int, frame_h: int) -> None:
        """Фактична роздільна здатність могла відрізнитись від профілю."""
        self._storage.set_setting(SETTING_FRAME_SIZE, f"{frame_w}x{frame_h}")
        self.refresh_profile()
        if self._profile_status is not ProfileStatus.VALID:
            self.statusBar().showMessage("Камера або її роздільна здатність змінились — потрібне калібрування")
            self.stop_session()

    @Slot(object)
    def _on_recenter_done(self, result: RecenterResult) -> None:
        self._close_recenter()
        self._hint.clear()
        if result.large_shift:
            self.statusBar().showMessage(_LARGE_SHIFT_HINT)
        now = time.monotonic()  # таймер і запис ідуть з моменту, коли трекінг почався
        self._start_recording(now)
        if self._use_pomodoro:
            self._pomodoro.start(now)
        else:
            self._stopwatch.start(now)
            self._record(PomodoroEvent.PHASE_START, PomodoroPhase.WORK, now)  # сесія без таймера — одна робоча фаза
            self._apply_phase_view()
        self._tick.start()

    @Slot()
    def _on_recenter_retry(self) -> None:
        if self._recenter is not None:
            self._recenter.restart()

    def _close_recenter(self) -> None:
        if self._recenter is not None:
            self._recenter.finish()
            self._recenter = None

    # --- запис сесії ---------------------------------------------------------

    def _start_recording(self, now: float) -> None:
        monitor = self.work_monitor
        s = self._settings
        settings = {
            "n": s.n, "y": s.y, "pomodoro": self._use_pomodoro,
            "work_minutes": s.work_minutes, "short_break_minutes": s.short_break_minutes,
            "long_break_minutes": s.long_break_minutes,
            "pomodoros_before_long_break": s.pomodoros_before_long_break,
        }
        self._recorder = SessionRecorder(
            self._storage,
            self._user_id,
            monitor.monitor_id,
            self._storage.profile_id(self._user_id, monitor.monitor_id, self._camera_id),
            settings,
            datetime.now().isoformat(timespec="seconds"),
            now,
        )

    def _record(self, event: PomodoroEvent, phase: PomodoroPhase, now: float) -> None:
        if self._recorder is not None:
            self._recorder.on_pomodoro_event(event, phase, now)

    def _finish_recording(self, now: float) -> None:
        if self._recorder is not None and not self._recorder.finished:
            self._recorder.finish(now, datetime.now().isoformat(timespec="seconds"))
        self._recorder = None

    @Slot()
    def _on_worker_finished(self) -> None:
        self._close_recenter()
        now = time.monotonic()
        self._pomodoro.stop(now)
        self._stopwatch.stop()
        self._finish_recording(now)
        self._tick.stop()
        self._debug.detach()
        self._worker = None
        self._hint.clear()
        self._render_state(None)
        self._update_timer()
        self._update_buttons()

    # --- Pomodoro -----------------------------------------------------------

    @Slot()
    def _on_tick(self) -> None:
        self._pomodoro.tick(time.monotonic())
        self._update_timer()

    def _on_pomodoro_event(self, info: PomodoroEventInfo) -> None:
        self._record(info.event, info.phase, info.ts)
        if info.event is PomodoroEvent.PHASE_END:
            if self._cfg.pomodoro.sound:
                QApplication.beep()
        if info.event in (PomodoroEvent.PHASE_START, PomodoroEvent.STOP):
            self._apply_phase_view()

    def _session_started(self) -> bool:
        """Трекінг сесії почався (після перецентрування) — з Pomodoro чи без."""
        return self._pomodoro.phase is not PomodoroPhase.IDLE or self._stopwatch.started

    def _paused_manual(self) -> bool:
        return self._pomodoro.paused_manual or self._stopwatch.paused

    def _apply_phase_view(self) -> None:
        """Трекінг, індикатор і кнопки відповідно до фази та ручної паузи."""
        tracking = self._tracking_active()
        if self._worker is not None and self._session_started():
            self._worker.set_tracking(tracking)
        if tracking:
            self._render_state(FocusState.FOCUSED)
        elif self._paused_manual():
            self._render_style(PAUSE_STYLE)
        elif self._pomodoro.phase in (PomodoroPhase.SHORT_BREAK, PomodoroPhase.LONG_BREAK):
            self._render_style(BREAK_STYLE)
        self._update_timer()
        self._update_buttons()

    def _update_timer(self) -> None:
        now = time.monotonic()
        if self._stopwatch.started:
            self._timer_label.setText(format_elapsed(self._stopwatch.elapsed(now)))
            paused = "  ·  пауза" if self._stopwatch.paused else ""
            self._phase_label.setText(_FREE_SESSION_NAME + paused)
            return
        phase = self._pomodoro.phase
        if phase is PomodoroPhase.IDLE:
            idle = format_remaining(self._cfg.pomodoro.work) if self._use_pomodoro else format_elapsed(0.0)
            self._timer_label.setText(idle)
            self._phase_label.setText("")
            return
        self._timer_label.setText(format_remaining(self._pomodoro.remaining(now)))
        text = f"{PHASE_NAMES[phase]}  ·  помодоро: {self._pomodoro.completed_pomodoros}"
        if self._pomodoro.paused_manual:
            text += "  ·  пауза"
        elif self._pomodoro.paused_auto:
            text += "  ·  автопауза"
        self._phase_label.setText(text)

    # --- сигнали worker-а -----------------------------------------------

    @Slot(object)
    def _on_frame_processed(self, result: FrameResult) -> None:
        if result.state is FocusState.RECOVERING:
            self._overlay.show_state(result.state, result.recovering_progress)

    @Slot(object)
    def _on_state_changed(self, change: StateChange) -> None:
        if not self._tracking_active():
            return  # кадр, оброблений до вимкнення трекінгу
        if self._recorder is not None:
            self._recorder.on_state_change(change)
        now = time.monotonic()
        if change.new is FocusState.LOST_FOCUS:
            self._pomodoro.pause_auto(now)
        elif change.new is FocusState.FOCUSED and change.old is FocusState.RECOVERING:
            self._pomodoro.resume_auto(now)
        self._render_state(change.new)
        self._update_timer()

    def _tracking_active(self) -> bool:
        if self._stopwatch.started:
            return not self._stopwatch.paused
        return self._pomodoro.phase is PomodoroPhase.WORK and not self._pomodoro.paused_manual

    @Slot(str)
    def _on_camera_error(self, message: str) -> None:
        self.statusBar().showMessage(message)
        self._debug.show_error(message)

    # --- відображення -------------------------------------------------------

    def _render_state(self, state: FocusState | None) -> None:
        self._state = state
        self._render_style(style_for(state))
        self._overlay.show_state(state, None)

    def _render_style(self, style: StateStyle) -> None:
        self._indicator.setStyleSheet(
            f"background-color: {style.color}; border-radius: {_INDICATOR_SIZE // 2}px;"
        )
        glow = QColor(style.color)
        glow.setAlpha(_GLOW_ALPHA)
        self._glow.setColor(glow)
        self._state_label.setText(style.label)
        if style.overlay_text is None:
            self._overlay.hide_overlay()

    def _update_buttons(self) -> None:
        running = self._worker is not None
        calibrating = self._calibration is not None
        self._start.setEnabled(not running and not calibrating and self._profile_status is ProfileStatus.VALID)
        self._pause.setEnabled(self._session_started())
        self._pause.setText("Продовжити" if self._paused_manual() else "Пауза")
        self._skip.setEnabled(self._pomodoro.phase is not PomodoroPhase.IDLE)
        self._stop.setEnabled(running)
        self._calibrate.setEnabled(not running and not calibrating)
        self._monitor_box.setEnabled(not running and not calibrating)
        self._settings_view.set_camera_editable(not running and not calibrating)
        self._pomodoro_box.setEnabled(not running and not calibrating)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        self.stop_session()
        if self._calibration is not None:
            self._calibration.close()
        self._overlay.close()
        super().closeEvent(event)
