"""Повноекранний майстер калібрування на робочому моніторі.

Покроково: інструкція → відлік → ціль, що стискається за час збору. Для
поворотів голови — індикатор кута (сірий → зелений). Для кроку з телефоном —
звуковий сигнал початку й кінця. У кінці — точність і кнопки «Зберегти» /
«Повторити». Esc — скасувати. Логіка кроків — у CalibrationProcedure.
"""

from __future__ import annotations

import random
import time
from datetime import datetime
from enum import Enum

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QFont, QGuiApplication, QKeyEvent, QPainter, QScreen
from PySide6.QtWidgets import QApplication, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from attention_tracker.calibration.fitting import Quality
from attention_tracker.calibration.procedure import (
    CalibrationProcedure,
    CalibrationStep,
    HeadPose,
    RetryReason,
    StepKind,
    StepState,
)
from attention_tracker.calibration.profile import CalibrationProfile
from attention_tracker.config import AppConfig
from attention_tracker.models import FrameFeatures
from attention_tracker.platform.monitors import MonitorInfo
from attention_tracker.workers.pipeline_worker import PipelineWorker, WorkerMode

# Оформлення
_BG = QColor("#111418")
_TEXT = QColor("#e8eaed")
_HINT = QColor("#ffb74d")
_TARGET = QColor("#ffffff")
_POSE_OK = QColor("#43a047")
_POSE_WAIT = QColor("#757575")
_TARGET_RADIUS = 28.0
_TARGET_MIN_FRACTION = 0.3  # ціль стискається до 30 % за час збору
_DOT_RADIUS = 4.0
_POSE_RADIUS = 18.0
_TEXT_TOP = 0.12  # частка висоти екрана над текстом інструкції
_MS_PER_S = 1000
_TICK_MS = 100  # оновлення відліку

_INSTRUCTIONS: dict[StepKind, str] = {
    StepKind.PRECHECK: "Сядьте зручно, як зазвичай працюєте, і дивіться на екран",
    StepKind.CENTER: "Дивіться на точку в центрі. Голову тримайте прямо",
    StepKind.GRID: "Стежте за точкою тільки очима — голова нерухома",
    StepKind.BOTTOM: "Дивіться на точку внизу екрана природно, не моргайте навмисно",
    StepKind.PHONE: (
        "Подивіться на телефон у руках або на стіл перед клавіатурою, як зазвичай.\n"
        "Звуковий сигнал — початок і кінець"
    ),
    StepKind.CHECK: "Перевірка точності: дивіться на точку",
}
_POSE_INSTRUCTIONS: dict[HeadPose, str] = {
    HeadPose.LEFT: "Поверніть голову вліво",
    HeadPose.RIGHT: "Поверніть голову вправо",
    HeadPose.UP: "Підніміть голову вгору",
    HeadPose.DOWN: "Опустіть голову вниз",
}
_RETRY_HINTS: dict[RetryReason, str] = {
    RetryReason.LOW_VALID: "Не вдалося зібрати дані: перевірте світло, відстань і не рухайтесь",
    RetryReason.FACE_RATIO: "Обличчя погано видно: перевірте світло і положення перед камерою",
    RetryReason.TOO_FAR: "Сядьте ближче до камери",
    RetryReason.TOO_CLOSE: "Сядьте далі від камери",
    RetryReason.LOW_FPS: "Камера працює надто повільно: закрийте інші програми з камерою",
    RetryReason.PITCH_TOO_SMALL: "Опустіть голову помітніше",
}
_QUALITY_TEXT: dict[Quality, str] = {
    Quality.OK: "Калібрування успішне",
    Quality.WARN: "Точність знижена — профіль можна зберегти, але краще повторити",
    Quality.RETRY: "Низька точність — рекомендуємо повторити калібрування",
}


class _Phase(Enum):
    STARTING = "STARTING"  # камера відкривається
    COUNTDOWN = "COUNTDOWN"  # інструкція + відлік, ціль вже видно
    RUNNING = "RUNNING"  # збір даних
    RESULT = "RESULT"


def instruction_for(step: CalibrationStep) -> str:
    if step.kind is StepKind.HEAD:
        return f"{_POSE_INSTRUCTIONS[step.pose]} — індикатор стане зеленим — і дивіться на точку"
    return _INSTRUCTIONS[step.kind]


def _same_instruction(a: CalibrationStep | None, b: CalibrationStep) -> bool:
    return a is not None and a.kind is b.kind and a.pose is b.pose


class CalibrationWindow(QWidget):
    profile_ready = Signal(object)  # CalibrationProfile
    closed = Signal()

    def __init__(
        self,
        cfg: AppConfig,
        user_id: int,
        monitor: MonitorInfo,
        screen: QScreen | None,
        rng: random.Random | None = None,
    ) -> None:
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self._cfg = cfg
        self._user_id = user_id
        self._monitor = monitor
        self._screen = screen
        self._rng = rng
        self._worker: PipelineWorker | None = None
        self._proc: CalibrationProcedure | None = None
        self._camera_id: str | None = None
        self._frame_size: tuple[int, int] | None = None
        self._phase = _Phase.STARTING
        self._prev_step: CalibrationStep | None = None
        self._countdown_end = 0.0
        self._hint = ""
        self._last_raw: FrameFeatures | None = None

        self.setWindowTitle("Калібрування")
        self.setCursor(Qt.CursorShape.BlankCursor)
        self._save = QPushButton("Зберегти")
        self._repeat = QPushButton("Повторити")
        self._save.clicked.connect(self._on_save)
        self._repeat.clicked.connect(self.restart)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self._repeat)
        buttons.addWidget(self._save)
        buttons.addStretch(1)
        layout = QVBoxLayout(self)
        layout.addStretch(1)
        layout.addLayout(buttons)
        layout.setContentsMargins(0, 0, 0, 80)
        self._show_buttons(False)

        self._tick = QTimer(self)
        self._tick.setInterval(_TICK_MS)
        self._tick.timeout.connect(self._on_tick)

    # --- життєвий цикл -------------------------------------------------------

    def open_on_screen(self) -> None:
        """Показати на весь екран робочого монітора і ввімкнути камеру."""
        screen = self._screen or QGuiApplication.primaryScreen()
        if screen is not None:
            self.setGeometry(screen.geometry())
        self.showFullScreen()
        self.activateWindow()
        self.start_camera()

    def start_camera(self) -> None:
        worker = PipelineWorker(self._cfg, WorkerMode.CALIBRATION)
        worker.camera_opened.connect(self.on_camera_opened)
        worker.calibration_sample.connect(self.on_sample)
        worker.camera_error.connect(self._on_camera_error)
        self._worker = worker
        worker.start()

    def _stop_camera(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait()
            self._worker = None

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        self._tick.stop()
        self._stop_camera()
        self.closed.emit()
        super().closeEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 (Qt API)
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    # --- кроки -----------------------------------------------------------------

    @Slot(str, int, int)
    def on_camera_opened(self, camera_id: str, frame_w: int, frame_h: int) -> None:
        self._camera_id = camera_id
        self._frame_size = (frame_w, frame_h)
        self.restart()

    @Slot()
    def restart(self) -> None:
        """Почати калібрування з першого кроку."""
        w, h = self._frame_size
        self._proc = CalibrationProcedure(
            self._cfg.calibration, self._cfg.distraction.ear_freeze_factor, w, h, self._rng
        )
        self._prev_step = None
        self._hint = ""
        self._show_buttons(False)
        self.setCursor(Qt.CursorShape.BlankCursor)
        self._prepare_step()

    def _prepare_step(self) -> None:
        step = self._proc.current
        same = _same_instruction(self._prev_step, step) and not self._hint
        delay = self._cfg.ui.calibration_lead_in if same else self._cfg.ui.calibration_countdown
        self._phase = _Phase.COUNTDOWN
        self._countdown_end = time.monotonic() + delay
        self._tick.start()
        self.update()

    def _on_tick(self) -> None:
        if self._phase is _Phase.COUNTDOWN and time.monotonic() >= self._countdown_end:
            self._start_collecting()
        self.update()

    def _start_collecting(self) -> None:
        step = self._proc.current
        if step.kind is StepKind.PHONE:
            QApplication.beep()
        self._proc.start_step(time.monotonic())
        self._phase = _Phase.RUNNING

    @Slot(object)
    def on_sample(self, raw: FrameFeatures) -> None:
        self._last_raw = raw
        if self._phase is not _Phase.RUNNING:
            self.update()
            return
        state = self._proc.feed(raw)
        if state is StepState.DONE:
            step = self._proc.current
            if step.kind is StepKind.PHONE:
                QApplication.beep()
            self._hint = ""
            self._prev_step = step
            self._proc.next()
            if self._proc.finished:
                self._show_result()
            else:
                self._prepare_step()
        elif state is StepState.RETRY:
            self._hint = _RETRY_HINTS[self._proc.retry_reason]
            self._prepare_step()
        self.update()

    def _show_result(self) -> None:
        self._phase = _Phase.RESULT
        self._tick.stop()
        self.unsetCursor()
        self._show_buttons(True)
        self._save.setDefault(self._proc.quality is not Quality.RETRY)

    def _show_buttons(self, visible: bool) -> None:
        self._save.setVisible(visible)
        self._repeat.setVisible(visible)

    @Slot()
    def _on_save(self) -> None:
        profile = self.result_profile()
        self.profile_ready.emit(profile)
        self.close()

    def result_profile(self) -> CalibrationProfile:
        created_at = datetime.now().isoformat(timespec="seconds")
        return self._proc.result(self._user_id, self._monitor.monitor_id, self._camera_id, created_at)

    @Slot(str)
    def _on_camera_error(self, message: str) -> None:
        self._hint = message
        self.update()

    # --- стан для UI і тестів ------------------------------------------------

    @property
    def phase_name(self) -> str:
        return self._phase.value

    @property
    def procedure(self) -> CalibrationProcedure | None:
        return self._proc

    # --- малювання ---------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), _BG)
        w, h = self.width(), self.height()

        text, hint = self._texts()
        p.setPen(_TEXT)
        p.setFont(QFont(self.font().family(), 20))
        text_rect = QRectF(0, h * _TEXT_TOP, w, h * 0.2)
        p.drawText(text_rect, Qt.AlignmentFlag.AlignHCenter | Qt.TextFlag.TextWordWrap, text)
        if hint:
            p.setPen(_HINT)
            p.setFont(QFont(self.font().family(), 16))
            p.drawText(QRectF(0, h * (_TEXT_TOP + 0.2), w, h * 0.1),
                       Qt.AlignmentFlag.AlignHCenter | Qt.TextFlag.TextWordWrap, hint)

        step = self._proc.current if self._proc is not None else None
        if self._phase in (_Phase.COUNTDOWN, _Phase.RUNNING) and step is not None:
            if step.target is not None:
                self._draw_target(p, step, w, h)
            if step.kind is StepKind.HEAD:
                self._draw_pose_indicator(p, w, h)
        p.end()

    def _texts(self) -> tuple[str, str]:
        if self._phase is _Phase.STARTING:
            return "Вмикаємо камеру…", self._hint
        if self._phase is _Phase.RESULT:
            error = self._proc.calib_error
            return f"{_QUALITY_TEXT[self._proc.quality]}\nСередня похибка: {error:.3f}", ""
        step = self._proc.current
        text = instruction_for(step)
        if self._phase is _Phase.COUNTDOWN:
            left = max(0.0, self._countdown_end - time.monotonic())
            text += f"\n{left:.0f}" if left >= 1.0 else ""
        progress = f"Крок {self._proc.index + 1} з {len(self._proc.steps)}  ·  Esc — скасувати"
        return text, self._hint or progress

    def _draw_target(self, p: QPainter, step: CalibrationStep, w: int, h: int) -> None:
        progress = self._proc.progress(time.monotonic()) or 0.0
        radius = _TARGET_RADIUS * (1.0 - (1.0 - _TARGET_MIN_FRACTION) * progress)
        center = QPointF(step.target[0] * w, step.target[1] * h)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(_TARGET)
        p.drawEllipse(center, radius, radius)
        p.setBrush(_BG)
        p.drawEllipse(center, _DOT_RADIUS, _DOT_RADIUS)

    def _draw_pose_indicator(self, p: QPainter, w: int, h: int) -> None:
        ok = self._last_raw is not None and self._proc.pose_ok(self._last_raw)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(_POSE_OK if ok else _POSE_WAIT)
        p.drawEllipse(QPointF(w / 2, h * (_TEXT_TOP - 0.05)), _POSE_RADIUS, _POSE_RADIUS)
