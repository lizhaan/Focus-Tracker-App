import dataclasses

import pytest

from attention_tracker.config import AppConfig
from attention_tracker.storage.storage import Storage
from attention_tracker.storage.user_settings import (
    KEY_CAMERA_RESOLUTION,
    KEY_N,
    KEY_WORK,
    UserSettings,
    apply_settings,
    load_settings,
    save_settings,
    validate,
)

CFG = AppConfig()


@pytest.fixture
def storage():
    with Storage(":memory:") as s:
        yield s


def test_defaults_from_config():
    d = UserSettings.defaults(CFG)
    assert (d.n, d.y) == (3.0, 2.0)
    assert (d.work_minutes, d.short_break_minutes, d.long_break_minutes) == (25, 5, 15)
    assert d.pomodoros_before_long_break == 4
    assert d.sound is True
    assert (d.camera_index, d.camera_resolution) == (0, (1280, 720))


def test_empty_db_gives_defaults(storage):
    assert load_settings(storage, CFG) == UserSettings.defaults(CFG)


def test_save_and_load_roundtrip(storage):
    s = dataclasses.replace(
        UserSettings.defaults(CFG), n=4.5, y=1.5, work_minutes=50, short_break_minutes=10,
        long_break_minutes=30, pomodoros_before_long_break=2, sound=False,
        camera_index=1, camera_resolution=(640, 480),
    )
    save_settings(storage, s)
    assert load_settings(storage, CFG) == s


def test_bad_values_fall_back_or_clamp(storage):
    storage.set_setting(KEY_N, "abc")
    storage.set_setting(KEY_WORK, "500")
    storage.set_setting(KEY_CAMERA_RESOLUTION, "333x222")
    s = load_settings(storage, CFG)
    assert s.n == 3.0  # не число → стандартне
    assert s.work_minutes == 120  # обмежено межею
    assert s.camera_resolution == (1280, 720)  # невідома роздільна здатність


@pytest.mark.parametrize("n, y, expected", [(0.2, 0.1, (1.0, 0.5)), (99.0, 99.0, (30.0, 15.0))])
def test_validate_focus_limits(n, y, expected):
    s = validate(dataclasses.replace(UserSettings.defaults(CFG), n=n, y=y), CFG)
    assert (s.n, s.y) == expected


def test_apply_settings_to_config():
    s = dataclasses.replace(
        UserSettings.defaults(CFG), n=5.0, y=1.0, work_minutes=50, sound=False,
        camera_index=2, camera_resolution=(640, 480),
    )
    cfg = apply_settings(CFG, s)
    assert (cfg.focus.n, cfg.focus.y) == (5.0, 1.0)
    assert cfg.pomodoro.work == 3000.0 and cfg.pomodoro.sound is False
    assert (cfg.camera.index, cfg.camera.width, cfg.camera.height) == (2, 640, 480)
    assert cfg.focus.tau == CFG.focus.tau  # решта не змінюється
