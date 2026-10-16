import dataclasses

import pytest

from attention_tracker.config import AppConfig
from attention_tracker.models import FocusState, FrameFeatures, Reason


def test_defaults_match_math_section_9():
    cfg = AppConfig()
    assert cfg.focus.n == 3.0
    assert cfg.focus.y == 2.0
    assert cfg.focus.tau == 0.3
    assert cfg.distraction.t_blink == 0.5
    assert cfg.distraction.t_noface == 0.5
    assert cfg.filters.median_window == 5
    assert cfg.filters.ema_alpha == 0.35
    assert (cfg.distraction.eps_out, cfg.distraction.eps_in) == (0.05, 0.02)
    assert (cfg.calibration.t_calib, cfg.calibration.t_settle) == (2.0, 0.6)
    assert cfg.calibration.ridge_lambda == 0.05
    assert (cfg.distraction.invalid_yaw_deg, cfg.distraction.invalid_pitch_deg) == (45.0, 40.0)


def test_config_is_immutable_and_replaceable():
    cfg = AppConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.focus.n = 5.0  # type: ignore[misc]
    updated = dataclasses.replace(cfg, focus=dataclasses.replace(cfg.focus, n=5.0))
    assert updated.focus.n == 5.0
    assert cfg.focus.n == 3.0


def test_invalid_frame_features():
    f = FrameFeatures.invalid(ts=1.5)
    assert f.ts == 1.5
    assert not f.valid
    assert f.ear is None and f.yaw is None and f.gaze_h is None


def test_enums_have_all_members():
    assert {s.name for s in FocusState} == {"FOCUSED", "NOT_FOCUSED", "LOST_FOCUS", "RECOVERING"}
    assert {r.name for r in Reason} == {
        "NONE", "GAZE_OUT", "PHONE_EYES", "PHONE_HEAD", "FACE_ABSENT", "HEAD_AWAY",
    }
