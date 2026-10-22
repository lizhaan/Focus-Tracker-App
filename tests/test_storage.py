import dataclasses
import sqlite3

import numpy as np
import pytest

from attention_tracker.calibration.profile import CalibrationProfile
from attention_tracker.storage.storage import DEFAULT_USER, SCHEMA_VERSION, Storage
from attention_tracker.vision.gaze_model import GazeRegressor

NOW = "2026-10-22T10:00:00"


def make_profile(user_id: int, monitor_id="DEL|U2419|ABC", camera_id="index:0",
                 calib_error=0.05, created_at=NOW) -> CalibrationProfile:
    reg = GazeRegressor(0.5, np.array([0.1, 0.2, 0.0, 0.0]), np.zeros(4), np.ones(4))
    return CalibrationProfile(
        user_id=user_id, monitor_id=monitor_id, camera_id=camera_id, frame_w=1280, frame_h=720,
        yaw_baseline=4.0, pitch_baseline=-6.0, ear_center=0.3, s_pitch=1,
        ear_bottom=0.26, pd_bottom=3.0, ear_phone=0.18, pd_phone=20.0,
        ear_enabled=True, ear_threshold=0.22, ear_freeze=0.182, pitch_threshold=11.5,
        regressor_x=reg, regressor_y=reg, calib_error=calib_error, created_at=created_at,
    )


@pytest.fixture
def db():
    with Storage(":memory:") as storage:
        yield storage


@pytest.fixture
def user(db) -> int:
    return db.ensure_user(DEFAULT_USER, NOW)


def test_schema_version_set(db):
    assert db.schema_version == SCHEMA_VERSION == 1


def test_ensure_user_is_idempotent(db):
    first = db.ensure_user(DEFAULT_USER, NOW)
    assert db.ensure_user(DEFAULT_USER, "2030-01-01T00:00:00") == first
    assert db.ensure_user("Other", NOW) != first


def test_save_and_load_profile(db, user):
    profile = make_profile(user)
    db.save_profile(profile, monitor_name="U2419 (1920×1080)")
    loaded = db.load_profile(user, profile.monitor_id, profile.camera_id)
    assert loaded.pitch_threshold == profile.pitch_threshold
    assert loaded.regressor_x.predict([0.5, 1.0, 0.25, 0.5]) == profile.regressor_x.predict([0.5, 1.0, 0.25, 0.5])
    assert db.load_profile(user, profile.monitor_id, "index:1") is None


def test_new_calibration_replaces_old(db, user):
    first_id = db.save_profile(make_profile(user, calib_error=0.12), "mon")
    second_id = db.save_profile(make_profile(user, calib_error=0.04, created_at="2026-10-23T09:00:00"), "mon")
    assert first_id == second_id
    profiles = db.list_profiles(user)
    assert len(profiles) == 1
    assert profiles[0].calib_error == pytest.approx(0.04)
    assert profiles[0].created_at == "2026-10-23T09:00:00"


def test_profiles_per_monitor_and_camera(db, user):
    db.save_profile(make_profile(user, monitor_id="A"), "Монітор A")
    db.save_profile(make_profile(user, monitor_id="B", created_at="2026-10-23T09:00:00"), "Монітор B")
    db.save_profile(make_profile(user, monitor_id="A", camera_id="index:1"), "Монітор A")
    infos = db.list_profiles(user)
    assert len(infos) == 3
    assert infos[0].monitor_id == "B"  # найновіший першим
    assert {i.monitor_name for i in infos} == {"Монітор A", "Монітор B"}


def test_delete_profile(db, user):
    profile_id = db.save_profile(make_profile(user), "mon")
    db.delete_profile(profile_id)
    assert db.list_profiles(user) == []


def test_profile_requires_existing_user(db):
    with pytest.raises(sqlite3.IntegrityError):
        db.save_profile(make_profile(user_id=999), "mon")


def test_settings(db):
    assert db.get_setting("focus.n") is None
    assert db.get_setting("focus.n", "3.0") == "3.0"
    db.set_setting("focus.n", "4.5")
    db.set_setting("focus.n", "5.0")
    assert db.get_setting("focus.n") == "5.0"


def test_file_database_persists(tmp_path):
    path = tmp_path / "sub" / "app.db"
    with Storage(path) as db:
        user_id = db.ensure_user(DEFAULT_USER, NOW)
        db.save_profile(make_profile(user_id), "mon")
        db.set_setting("work_monitor", "DEL|U2419|ABC")
    with Storage(path) as db:  # повторне відкриття не повторює міграцію
        assert db.schema_version == 1
        assert len(db.list_profiles(user_id)) == 1
        assert db.get_setting("work_monitor") == "DEL|U2419|ABC"


def test_profile_from_newer_version_loads_as_invalid(db, user):
    import json

    data = json.loads(make_profile(user).to_json())
    data["future_field"] = 1  # поле з новішої версії програми
    data["schema_version"] = 99
    newer = CalibrationProfile.from_json(json.dumps(data))
    assert not newer.is_valid_for(newer.monitor_id, newer.camera_id, newer.frame_w, newer.frame_h)


def test_old_schema_profile_loads_as_invalid(db, user):
    import json

    data = json.loads(make_profile(user).to_json())
    del data["pitch_enabled"]
    data["schema_version"] = 1  # профіль, збережений до появи pitch_enabled
    old = CalibrationProfile.from_json(json.dumps(data))
    assert not old.is_valid_for(old.monitor_id, old.camera_id, old.frame_w, old.frame_h)


def test_profile_from_other_schema_version_is_invalid(db, user):
    profile = dataclasses.replace(make_profile(user), schema_version=99)
    assert not profile.is_valid_for(profile.monitor_id, profile.camera_id, 1280, 720)
