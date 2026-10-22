import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    "module",
    ["attention_tracker.main", "attention_tracker.ui.debug_view",
     "attention_tracker.ui.main_window", "attention_tracker.ui.overlay",
     "attention_tracker.ui.calibration_window", "attention_tracker.ui.recenter_window",
     "attention_tracker.workers.pipeline_worker"],
)
def test_module_imports_in_clean_interpreter(module):
    """PySide6 + mediapipe: порядок імпорту не повинен валити запуск."""
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
