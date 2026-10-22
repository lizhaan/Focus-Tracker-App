"""Монітори з QScreen і їхні стабільні ідентифікатори."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QGuiApplication, QScreen


def make_monitor_id(
    manufacturer: str, model: str, serial: str, name: str, width: int, height: int
) -> str:
    """`manufacturer|model|serial`; якщо всі три порожні — `name|WxH`."""
    if manufacturer or model or serial:
        return f"{manufacturer}|{model}|{serial}"
    return f"{name}|{width}x{height}"


@dataclass(frozen=True)
class MonitorInfo:
    monitor_id: str
    name: str  # для показу користувачу
    geometry: tuple[int, int, int, int]  # x, y, ширина, висота (логічні пікселі)
    is_primary: bool

    @classmethod
    def from_qscreen(cls, screen: QScreen, is_primary: bool) -> MonitorInfo:
        g = screen.geometry()
        monitor_id = make_monitor_id(
            screen.manufacturer().strip(),
            screen.model().strip(),
            screen.serialNumber().strip(),
            screen.name(),
            g.width(),
            g.height(),
        )
        label = screen.model().strip() or screen.manufacturer().strip() or screen.name()
        return cls(
            monitor_id=monitor_id,
            name=f"{label} ({g.width()}×{g.height()})",
            geometry=(g.x(), g.y(), g.width(), g.height()),
            is_primary=is_primary,
        )


def list_monitors() -> list[MonitorInfo]:
    """Усі підключені монітори; потрібен створений QGuiApplication."""
    primary = QGuiApplication.primaryScreen()
    return [MonitorInfo.from_qscreen(s, s is primary) for s in QGuiApplication.screens()]


def find_screen(monitor_id: str) -> QScreen | None:
    for screen in QGuiApplication.screens():
        if MonitorInfo.from_qscreen(screen, False).monitor_id == monitor_id:
            return screen
    return None
