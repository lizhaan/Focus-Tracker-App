"""Вкладка «Налаштування»: таймери фокусу, Pomodoro, камера, звук.

Кожна зміна одразу видається сигналом `settings_changed`; головне вікно
зберігає її в БД і застосовує. «Скинути до стандартних» повертає значення
з AppConfig.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from attention_tracker.config import AppConfig
from attention_tracker.storage.user_settings import UserSettings
from attention_tracker.ui.theme import camera_icon, eye_icon, timer_icon

# Оформлення
_FIELD_WIDTH = 240
_FIELD_TEXT_MARGIN = 8
_HINT_WIDTH = 260
_CONTENT_MAX_WIDTH = 700
_CARD_MARGINS = (24, 20, 24, 22)
_ROW_SPACING = 14
_CARD_SPACING = 20
_SECONDS_SUFFIX = " с"
_MINUTES_SUFFIX = " хв"

HINT_N = "Скільки секунд можна дивитись убік, поки не зʼявиться плашка"
HINT_Y = "Скільки секунд треба дивитись на екран, щоб плашка зникла і таймер продовжився"
HINT_CAMERA = "Зміна камери чи роздільної здатності діє з наступної сесії і потребує нового калібрування"


class _Card(QFrame):
    """Картка з іконкою і заголовком; рядки «назва — поле — підказка»."""

    def __init__(self, title: str, icon: QPixmap) -> None:
        super().__init__(objectName="card")
        icon_label = QLabel()
        icon_label.setPixmap(icon)
        title_label = QLabel(title, objectName="cardTitle")
        header = QHBoxLayout()
        header.setSpacing(10)
        header.addWidget(icon_label)
        header.addWidget(title_label)
        header.addStretch(1)

        self._grid = QGridLayout()
        self._grid.setHorizontalSpacing(24)
        self._grid.setVerticalSpacing(_ROW_SPACING)
        self._grid.setColumnStretch(0, 1)
        self._row = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(*_CARD_MARGINS)
        layout.setSpacing(18)
        layout.addLayout(header)
        layout.addLayout(self._grid)

    def add_field(self, label: str, field: QWidget, hint: str | None = None) -> None:
        field.setFixedWidth(_FIELD_WIDTH)
        inner = field.findChild(QLineEdit)
        if inner is not None:  # у Fusion padding поля спін-боксу не діє на текст
            inner.setTextMargins(_FIELD_TEXT_MARGIN, 0, 0, 0)
        self._grid.addWidget(QLabel(label, objectName="fieldLabel"), self._row, 0, Qt.AlignmentFlag.AlignVCenter)
        self._grid.addWidget(field, self._row, 1, Qt.AlignmentFlag.AlignRight)
        self._row += 1
        if hint:
            hint_label = QLabel(hint, objectName="hint")
            hint_label.setWordWrap(True)
            hint_label.setFixedWidth(_HINT_WIDTH)
            self._grid.addWidget(hint_label, self._row, 1, Qt.AlignmentFlag.AlignRight)
            self._row += 1

    def add_separator(self) -> None:
        line = QFrame(objectName="separator")
        line.setFixedHeight(1)
        self._grid.addWidget(line, self._row, 0, 1, 2)
        self._row += 1

    def add_right(self, widget: QWidget) -> None:
        self._grid.addWidget(widget, self._row, 0, 1, 2, Qt.AlignmentFlag.AlignRight)
        self._row += 1


class SettingsView(QWidget):
    settings_changed = Signal(object)  # UserSettings

    def __init__(self, cfg: AppConfig, settings: UserSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cfg = cfg
        focus, lim = cfg.focus, cfg.limits

        self._n = QDoubleSpinBox(suffix=_SECONDS_SUFFIX, decimals=1, singleStep=focus.ui_step)
        self._n.setRange(focus.n_min, focus.n_max)
        self._y = QDoubleSpinBox(suffix=_SECONDS_SUFFIX, decimals=1, singleStep=focus.ui_step)
        self._y.setRange(focus.y_min, focus.y_max)

        self._work = QSpinBox(suffix=_MINUTES_SUFFIX)
        self._work.setRange(*lim.work_minutes)
        self._short = QSpinBox(suffix=_MINUTES_SUFFIX)
        self._short.setRange(*lim.break_minutes)
        self._long = QSpinBox(suffix=_MINUTES_SUFFIX)
        self._long.setRange(*lim.break_minutes)
        self._before_long = QSpinBox()
        self._before_long.setRange(*lim.pomodoros_before_long_break)
        self._sound = QCheckBox("Звуковий сигнал наприкінці фази")

        self._camera_index = QSpinBox()
        self._camera_index.setRange(*lim.camera_index)
        self._resolution = QComboBox()
        self._resolutions = list(lim.camera_resolutions)
        self._resolution.addItems([f"{w}×{h}" for w, h in self._resolutions])

        focus_card = _Card("Фокус", eye_icon())
        focus_card.add_field("Час до втрати фокусу", self._n, HINT_N)
        focus_card.add_field("Час повернення у фокус", self._y, HINT_Y)

        pomodoro_card = _Card("Pomodoro", timer_icon())
        pomodoro_card.add_field("Робота", self._work)
        pomodoro_card.add_field("Коротка перерва", self._short)
        pomodoro_card.add_field("Довга перерва", self._long)
        pomodoro_card.add_field("Pomodoro до довгої перерви", self._before_long)
        pomodoro_card.add_separator()
        pomodoro_card.add_right(self._sound)

        camera_card = _Card("Камера", camera_icon())
        camera_card.add_field("Номер камери", self._camera_index)
        camera_card.add_field("Роздільна здатність", self._resolution, HINT_CAMERA)

        self._reset = QPushButton("Скинути до стандартних")
        self._reset.clicked.connect(self.reset_to_defaults)
        reset_row = QHBoxLayout()
        reset_row.addStretch(1)
        reset_row.addWidget(self._reset)

        content = QWidget()
        content.setMaximumWidth(_CONTENT_MAX_WIDTH)
        column = QVBoxLayout(content)
        column.setContentsMargins(24, 24, 24, 24)
        column.setSpacing(_CARD_SPACING)
        for card in (focus_card, pomodoro_card, camera_card):
            column.addWidget(card)
        column.addLayout(reset_row)
        column.addStretch(1)

        centered = QWidget()
        row = QHBoxLayout(centered)
        row.setContentsMargins(0, 0, 0, 0)
        row.addStretch(1)
        row.addWidget(content, 3)
        row.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(centered)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)

        self.set_settings(settings)
        for spin in (self._n, self._y, self._work, self._short, self._long, self._before_long, self._camera_index):
            spin.valueChanged.connect(self._emit)
        self._sound.toggled.connect(self._emit)
        self._resolution.currentIndexChanged.connect(self._emit)

    # --- значення ---------------------------------------------------------

    def settings(self) -> UserSettings:
        return UserSettings(
            n=self._n.value(),
            y=self._y.value(),
            work_minutes=self._work.value(),
            short_break_minutes=self._short.value(),
            long_break_minutes=self._long.value(),
            pomodoros_before_long_break=self._before_long.value(),
            sound=self._sound.isChecked(),
            camera_index=self._camera_index.value(),
            camera_resolution=self._resolutions[max(0, self._resolution.currentIndex())],
        )

    def set_settings(self, s: UserSettings) -> None:
        """Показати значення без сигналу про зміну."""
        widgets = (self._n, self._y, self._work, self._short, self._long, self._before_long,
                   self._sound, self._camera_index, self._resolution)
        for w in widgets:
            w.blockSignals(True)
        self._n.setValue(s.n)
        self._y.setValue(s.y)
        self._work.setValue(s.work_minutes)
        self._short.setValue(s.short_break_minutes)
        self._long.setValue(s.long_break_minutes)
        self._before_long.setValue(s.pomodoros_before_long_break)
        self._sound.setChecked(s.sound)
        self._camera_index.setValue(s.camera_index)
        if s.camera_resolution in self._resolutions:
            self._resolution.setCurrentIndex(self._resolutions.index(s.camera_resolution))
        for w in widgets:
            w.blockSignals(False)

    def set_camera_editable(self, editable: bool) -> None:
        """Камеру не змінюють під час сесії чи калібрування."""
        self._camera_index.setEnabled(editable)
        self._resolution.setEnabled(editable)

    @Slot()
    def reset_to_defaults(self) -> None:
        self.set_settings(UserSettings.defaults(self._cfg))
        self._emit()

    @Slot()
    def _emit(self, *_args: object) -> None:
        self.settings_changed.emit(self.settings())
