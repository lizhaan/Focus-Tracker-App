import dataclasses
import random
import time

import pytest

from attention_tracker.config import AppConfig, UiConfig
from attention_tracker.models import FrameFeatures
from tests.test_calibration_procedure import FPS, SimUser, frames_for

FAST_UI = UiConfig(calibration_countdown=0.01, calibration_lead_in=0.01)
CFG = dataclasses.replace(AppConfig(), ui=FAST_UI)


@pytest.fixture
def window(qapp):
    from attention_tracker.platform.monitors import MonitorInfo
    from attention_tracker.ui.calibration_window import CalibrationWindow

    monitor = MonitorInfo("mon", "Монітор", (0, 0, 1920, 1080), True)
    w = CalibrationWindow(CFG, user_id=1, monitor=monitor, screen=None, rng=random.Random(1))
    w.resize(800, 450)
    yield w
    w.close()


def wait_running(window, timeout=2.0):
    from PySide6.QtTest import QTest

    end = time.monotonic() + timeout
    while window.phase_name != "RUNNING" and time.monotonic() < end:
        QTest.qWait(15)
    assert window.phase_name == "RUNNING"


def run_current_step(window, user):
    """Подає кадри поточного кроку, поки він не завершиться."""
    proc = window.procedure
    index = proc.index
    t0 = time.monotonic()
    for frame in frames_for(proc.current, user, t0, 3.0):
        window.on_sample(frame)
        if proc.index != index or window.phase_name != "RUNNING":
            return


def test_full_wizard_produces_profile(window):
    user = SimUser()
    window.on_camera_opened("index:0", 1280, 720)
    assert window.phase_name == "COUNTDOWN"
    while window.phase_name != "RESULT":
        wait_running(window)
        run_current_step(window, user)
        window.grab()  # малювання кожного кроку не падає

    received = []
    window.profile_ready.connect(received.append)
    window._save.click()
    assert len(received) == 1
    profile = received[0]
    assert profile.monitor_id == "mon" and profile.camera_id == "index:0"
    assert (profile.frame_w, profile.frame_h) == (1280, 720)
    assert profile.calib_error < 0.05


def test_retry_shows_hint_and_restarts_step(window):
    window.on_camera_opened("index:0", 1280, 720)
    wait_running(window)
    t0 = time.monotonic()
    for i in range(int(2.5 * FPS)):  # перевірка перед стартом без обличчя
        window.on_sample(FrameFeatures.invalid(t0 + i / FPS))
    assert window.phase_name == "COUNTDOWN"
    assert window._hint
    assert window.procedure.index == 0
    wait_running(window)  # крок запускається знову


def test_waiting_for_camera_paints(window):
    assert window.phase_name == "STARTING"
    window.grab()


def test_escape_closes(window, qapp):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    closed = []
    window.closed.connect(lambda: closed.append(True))
    window.show()
    QTest.keyClick(window, Qt.Key.Key_Escape)
    assert closed == [True]
