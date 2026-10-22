import pytest

from attention_tracker.calibration.fitting import (
    Quality,
    compute_thresholds,
    quality_of,
    robust_mean,
    s_pitch_from,
)
from attention_tracker.config import CalibrationConfig

CFG = CalibrationConfig()
FREEZE = 0.7


def test_robust_mean_drops_outliers():
    assert robust_mean([1.0, 1.1, 0.9, 1.0, 1.05, 50.0], mad_k=3.0) == pytest.approx(1.01)


def test_robust_mean_of_identical_values():
    assert robust_mean([2.0, 2.0, 2.0], mad_k=3.0) == pytest.approx(2.0)


def test_robust_mean_empty_raises():
    with pytest.raises(ValueError):
        robust_mean([], mad_k=3.0)


def test_thresholds_ear_enabled_and_pitch_midpoint():
    t = compute_thresholds(ear_bottom=0.26, ear_phone=0.18, pd_bottom=3.0, pd_phone=20.0,
                           cfg=CFG, ear_freeze_factor=FREEZE)
    assert t.ear_enabled is True
    assert t.ear_threshold == pytest.approx(0.22)
    assert t.pitch_threshold == pytest.approx(11.5)
    assert t.ear_freeze == pytest.approx(0.7 * 0.26)


def test_thresholds_ear_disabled_and_pitch_fallback():
    t = compute_thresholds(ear_bottom=0.25, ear_phone=0.24, pd_bottom=3.0, pd_phone=6.0,
                           cfg=CFG, ear_freeze_factor=FREEZE)
    assert t.ear_enabled is False
    assert t.ear_threshold is None
    assert t.pitch_threshold == pytest.approx(3.0 + 8.0)


def test_pitch_threshold_at_least_bottom_plus_margin():
    # середина = (3 + 9) / 2 = 6, але не менше PD_bottom + 4 = 7
    t = compute_thresholds(ear_bottom=0.26, ear_phone=0.18, pd_bottom=3.0, pd_phone=9.0,
                           cfg=CFG, ear_freeze_factor=FREEZE)
    assert t.pitch_threshold == pytest.approx(7.0)


@pytest.mark.parametrize("mean, expected", [(12.0, 1), (-12.0, -1), (4.0, None), (-4.9, None)])
def test_s_pitch_sign(mean, expected):
    assert s_pitch_from(mean, min_abs_deg=5.0) == expected


@pytest.mark.parametrize(
    "error, quality",
    [(0.05, Quality.OK), (0.08, Quality.OK), (0.1, Quality.WARN), (0.15, Quality.WARN), (0.2, Quality.RETRY)],
)
def test_quality_bands(error, quality):
    assert quality_of(error, CFG) is quality
