"""Точка входу: `python -m attention_tracker.main`."""

from __future__ import annotations

import sys

# mediapipe раніше за PySide6 — див. attention_tracker/ui/__init__.py
import mediapipe  # noqa: F401
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from attention_tracker.config import ASSETS_DIR, AppConfig
from attention_tracker.platform.paths import APP_NAME, default_db_path
from attention_tracker.self_test import SELF_TEST_FLAG, run_self_test
from attention_tracker.storage.storage import Storage
from attention_tracker.ui.main_window import MainWindow
from attention_tracker.ui.theme import apply_theme

APP_ICON = "app_icon.png"


def main() -> int:
    if SELF_TEST_FLAG in sys.argv:
        rest = sys.argv[sys.argv.index(SELF_TEST_FLAG) + 1:]
        return run_self_test(rest[0] if rest else None)
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setWindowIcon(QIcon(str(ASSETS_DIR / APP_ICON)))
    apply_theme(app)
    with Storage(default_db_path()) as storage:
        window = MainWindow(AppConfig(), storage)
        window.show()
        return app.exec()


if __name__ == "__main__":
    sys.exit(main())
