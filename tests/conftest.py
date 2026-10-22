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
