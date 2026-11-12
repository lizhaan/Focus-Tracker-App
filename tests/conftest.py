import os

import pytest

# Qt без вікон на екрані — тести UI нічого не показують користувачу
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qapp():
    import mediapipe  # noqa: F401  (раніше за PySide6, див. attention_tracker/ui/__init__.py)
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def main_window(qapp):
    """Головне вікно на БД у памʼяті (без камери)."""
    from attention_tracker.config import AppConfig
    from attention_tracker.storage.storage import Storage
    from attention_tracker.ui.main_window import MainWindow

    storage = Storage(":memory:")
    window = MainWindow(AppConfig(), storage)
    yield window, storage
    window.close()
    storage.close()
