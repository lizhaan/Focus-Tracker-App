"""Самоперевірка зібраної програми: `AttentionTracker.exe --self-test [звіт.txt]`.

Перевіряє, що в збірці є модель, іконки й схеми БД, MediaPipe і OpenCV
завантажуються, БД створюється, головне вікно будується. Камеру не вмикає і
вікон не показує. Код виходу 0 — усе гаразд.
"""

from __future__ import annotations

import os
import sys
import traceback
from collections.abc import Callable
from pathlib import Path

SELF_TEST_FLAG = "--self-test"


def _check_assets() -> str:
    from attention_tracker.config import ASSETS_DIR, LandmarkerConfig
    from attention_tracker.storage import storage

    required = [LandmarkerConfig().model_path, ASSETS_DIR / "check.svg", ASSETS_DIR / "chevron_down.svg"]
    required += list(storage._MIGRATIONS)
    missing = [str(p) for p in required if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    return f"{len(required)} файлів"


def _check_landmarker() -> str:
    import numpy as np

    from attention_tracker.capture.landmarker import FaceLandmarkerWrapper
    from attention_tracker.config import LandmarkerConfig

    with FaceLandmarkerWrapper(LandmarkerConfig()) as lm:
        result = lm.detect(np.zeros((480, 640, 3), dtype=np.uint8), 0)
    if result is not None:
        raise RuntimeError("на чорному кадрі знайдено обличчя")
    return "модель завантажено, детекція працює"


def _check_opencv() -> str:
    import cv2

    from attention_tracker.vision.features import camera_matrix

    camera_matrix(640, 480)
    return f"OpenCV {cv2.__version__}"


def _check_storage() -> str:
    from attention_tracker.storage.storage import SCHEMA_VERSION, Storage

    with Storage(":memory:") as db:
        if db.schema_version != SCHEMA_VERSION:
            raise RuntimeError(f"схема {db.schema_version}, очікується {SCHEMA_VERSION}")
    return f"схема {SCHEMA_VERSION}"


def _check_main_window() -> str:
    from PySide6.QtWidgets import QApplication

    from attention_tracker.config import AppConfig
    from attention_tracker.storage.storage import Storage
    from attention_tracker.ui.main_window import MainWindow
    from attention_tracker.ui.theme import apply_theme

    app = QApplication.instance() or QApplication([])
    apply_theme(app)
    with Storage(":memory:") as db:
        window = MainWindow(AppConfig(), db)
        tabs = window.centralWidget()
        names = [tabs.tabText(i) for i in range(tabs.count())]
        window.close()
    return ", ".join(names)


CHECKS: list[tuple[str, Callable[[], str]]] = [
    ("файли збірки", _check_assets),
    ("OpenCV", _check_opencv),
    ("MediaPipe FaceLandmarker", _check_landmarker),
    ("SQLite", _check_storage),
    ("головне вікно", _check_main_window),
]


def run_self_test(report_path: str | None = None) -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # без вікон на екрані
    lines, failed = [], 0
    for name, check in CHECKS:
        try:
            lines.append(f"OK    {name}: {check()}")
        except Exception:  # noqa: BLE001 — звіт має зібрати всі помилки
            failed += 1
            lines.append(f"FAIL  {name}:\n{traceback.format_exc()}")
    lines.append("RESULT: " + ("OK" if not failed else f"{failed} FAILED"))
    report = "\n".join(lines)
    if sys.stdout is not None:  # у зібраній програмі без консолі stdout немає
        print(report)
    if report_path:
        Path(report_path).write_text(report, encoding="utf-8")
    return 0 if not failed else 1
