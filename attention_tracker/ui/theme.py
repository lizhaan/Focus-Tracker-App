"""Темна тема застосунку: кольори, таблиця стилів Qt, іконки карток налаштувань."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QApplication

from attention_tracker.config import ASSETS_DIR

BG = "#111218"
SURFACE = "#1a1b23"
FIELD = "#0e0f14"
BORDER = "#2b2c37"
TEXT = "#e8e9ef"
MUTED = "#8b8fa3"
HINT = "#6b7194"
ACCENT = "#8b5cf6"
ACCENT_HOVER = "#a07cf8"
OK = "#4caf50"
BAD = "#ef5350"

_ICON_SIZE = 20


def _asset(name: str) -> str:
    return (ASSETS_DIR / name).as_posix()


STYLESHEET = f"""
QWidget {{
    background-color: {BG};
    color: {TEXT};
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 14px;
}}
QToolTip {{ background-color: {SURFACE}; color: {TEXT}; border: 1px solid {BORDER}; }}

QTabWidget::pane {{ border: none; border-top: 1px solid {BORDER}; }}
QTabBar {{ background: transparent; }}
QTabBar::tab {{
    background: transparent; color: {MUTED};
    padding: 10px 22px; margin-right: 4px; border: none;
    border-top-left-radius: 8px; border-top-right-radius: 8px;
}}
QTabBar::tab:selected {{ background: {SURFACE}; color: {TEXT}; font-weight: 600;
    border-bottom: 2px solid {MUTED}; }}
QTabBar::tab:hover:!selected {{ color: {TEXT}; }}

QPushButton {{
    background-color: {SURFACE}; color: {TEXT};
    border: 1px solid {BORDER}; border-radius: 8px; padding: 8px 20px;
}}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:pressed {{ background-color: {FIELD}; }}
QPushButton:disabled {{ color: {HINT}; border-color: {SURFACE}; }}

QComboBox, QSpinBox, QDoubleSpinBox {{
    background-color: {FIELD}; color: {TEXT};
    border: 1px solid {BORDER}; border-radius: 8px;
    padding: 8px 12px; min-height: 20px;
    selection-background-color: {ACCENT};
}}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{ border-color: {ACCENT}; }}
QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{ color: {HINT}; }}
QComboBox::drop-down {{ border: none; width: 30px; }}
QComboBox::down-arrow {{ image: url({_asset("chevron_down.svg")}); width: 12px; height: 12px; }}
QComboBox QAbstractItemView {{
    background-color: {SURFACE}; color: {TEXT}; border: 1px solid {BORDER};
    selection-background-color: {ACCENT}; outline: none;
}}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{
    border: none; width: 26px; background: transparent;
}}
QAbstractSpinBox::up-arrow {{ image: url({_asset("chevron_up.svg")}); width: 10px; height: 10px; }}
QAbstractSpinBox::down-arrow {{ image: url({_asset("chevron_down.svg")}); width: 10px; height: 10px; }}

QCheckBox {{ spacing: 10px; color: {TEXT}; background: transparent; }}
QCheckBox::indicator {{
    width: 18px; height: 18px; border-radius: 4px;
    border: 1px solid {BORDER}; background-color: {FIELD};
}}
QCheckBox::indicator:checked {{
    background-color: {ACCENT}; border-color: {ACCENT}; image: url({_asset("check.svg")});
}}
QCheckBox::indicator:hover {{ border-color: {ACCENT}; }}

QFrame#card {{ background-color: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px; }}
QFrame#card QLabel, QFrame#card QCheckBox {{ background: transparent; }}
QFrame#separator {{ background-color: {BORDER}; max-height: 1px; border: none; }}
QLabel#cardTitle {{ font-size: 17px; font-weight: 600; }}
QLabel#windowTitle {{ font-size: 24px; font-weight: 700; }}
QLabel#subtitle {{ color: {MUTED}; font-size: 13px; }}
QLabel#tileValue {{ font-size: 28px; font-weight: 700; }}
QLabel#tileLabel {{ color: {MUTED}; font-size: 12px; }}
QLabel#fieldLabel {{ font-weight: 600; }}
QLabel#hint {{ color: {HINT}; font-size: 12px; }}
QLabel#timer {{ font-size: 92px; font-weight: 700; }}
QLabel#phase {{ color: {TEXT}; font-size: 17px; }}
QLabel#stateLabel {{ font-size: 28px; font-weight: 700; }}
QLabel#sessionHint {{ color: {MUTED}; font-size: 13px; }}
QLabel#profileOk {{ color: {OK}; font-size: 12px; }}
QLabel#profileBad {{ color: {BAD}; font-size: 12px; }}

QTableWidget {{
    background-color: {SURFACE}; color: {TEXT};
    border: 1px solid {BORDER}; border-radius: 12px;
    selection-background-color: {ACCENT}; selection-color: #ffffff; outline: none;
}}
QTableWidget::item {{ padding: 6px 8px; border-bottom: 1px solid {BG}; }}
QTableWidget::item:selected {{ background-color: {ACCENT}; color: #ffffff; }}
QHeaderView {{ background: transparent; }}
QHeaderView::section {{
    background-color: {SURFACE}; color: {MUTED}; font-size: 12px; font-weight: 600;
    border: none; border-bottom: 1px solid {BORDER}; padding: 8px;
}}
QTableCornerButton::section {{ background: {SURFACE}; border: none; }}

QScrollArea {{ border: none; }}
QScrollBar:vertical {{ background: {BG}; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QStatusBar {{ background-color: {BG}; color: {MUTED}; }}
"""


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")  # однаковий вигляд віджетів на всіх платформах
    app.setStyleSheet(STYLESHEET)


# --- іконки карток налаштувань --------------------------------------------


def _canvas() -> tuple[QPixmap, QPainter]:
    pixmap = QPixmap(_ICON_SIZE, _ICON_SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    return pixmap, painter


def eye_icon() -> QPixmap:
    pixmap, p = _canvas()
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(ACCENT))
    p.drawEllipse(QRectF(1, 5, 18, 10))
    p.setBrush(QColor(SURFACE))
    p.drawEllipse(QPointF(10, 10), 3.5, 3.5)
    p.setBrush(QColor(ACCENT))
    p.drawEllipse(QPointF(10, 10), 1.6, 1.6)
    p.end()
    return pixmap


def timer_icon() -> QPixmap:
    pixmap, p = _canvas()
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(ACCENT))
    p.drawEllipse(QPointF(10, 11.5), 7.5, 7.5)
    p.drawRoundedRect(QRectF(7.5, 1, 5, 3), 1, 1)
    pen = QPen(QColor(SURFACE), 1.8)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.drawLine(QPointF(10, 11.5), QPointF(10, 7.5))
    p.end()
    return pixmap


def camera_icon() -> QPixmap:
    pixmap, p = _canvas()
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(ACCENT))
    p.drawRoundedRect(QRectF(1.5, 5, 17, 12), 3, 3)
    p.drawRoundedRect(QRectF(6.5, 2.5, 7, 4), 1.5, 1.5)
    p.setBrush(QColor(SURFACE))
    p.drawEllipse(QPointF(10, 11), 3.6, 3.6)
    p.end()
    return pixmap
