"""Точка входу: `python -m attention_tracker.main`.

Етап 2, крок 1: відкриває тільки режим налагодження (FR-11).
"""

from __future__ import annotations

import sys

# mediapipe раніше за PySide6 — див. attention_tracker/ui/__init__.py
import mediapipe  # noqa: F401
from PySide6.QtWidgets import QApplication

from attention_tracker.config import AppConfig
from attention_tracker.ui.debug_view import DebugView


def main() -> int:
    app = QApplication(sys.argv)
    view = DebugView(AppConfig())
    view.setWindowTitle("Attention Tracker — Debug")
    view.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
