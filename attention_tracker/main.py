"""Точка входу: `python -m attention_tracker.main`."""

from __future__ import annotations

import sys

# mediapipe раніше за PySide6 — див. attention_tracker/ui/__init__.py
import mediapipe  # noqa: F401
from PySide6.QtWidgets import QApplication

from attention_tracker.config import AppConfig
from attention_tracker.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow(AppConfig())
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
