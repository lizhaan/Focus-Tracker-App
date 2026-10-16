import pytest

from attention_tracker.config import FilterConfig
from attention_tracker.models import FrameFeatures
from attention_tracker.vision.filters import MedianEmaFilter, SignalFilter


def frame(ts: float, value: float) -> FrameFeatures:
    return FrameFeatures(
        ts=ts, valid=True, ear=value, yaw=value, pitch=value, roll=value,
        gaze_h=value, gaze_v=value, iris_distance_px=100.0,
    )


@pytest.mark.parametrize("window, alpha", [(0, 0.35), (5, 0.0), (5, 1.5)])
def test_invalid_parameters_rejected(window, alpha):
    with pytest.raises(ValueError):
        MedianEmaFilter(window, alpha)


def test_first_output_equals_first_input():
    assert MedianEmaFilter(5, 0.35).update(0.42) == pytest.approx(0.42)


def test_single_outlier_removed_by_median():
    flt = MedianEmaFilter(5, 0.35)
    outputs = [flt.update(v) for v in [1.0, 1.0, 1.0, 100.0, 1.0, 1.0]]
    assert max(outputs) == pytest.approx(1.0)


def test_ema_converges_to_constant():
    flt = MedianEmaFilter(5, 0.35)
    flt.update(0.0)
    for _ in range(50):
        out = flt.update(10.0)
    assert out == pytest.approx(10.0, abs=1e-3)


def test_reset_forgets_history():
    flt = MedianEmaFilter(5, 0.35)
    for _ in range(5):
        flt.update(0.9)
    flt.reset()
    assert flt.update(0.1) == pytest.approx(0.1)


def test_signal_filter_smooths_listed_fields_only():
    sf = SignalFilter(FilterConfig())
    sf.update(frame(0.0, 0.0))
    out = sf.update(frame(0.1, 10.0))
    # median([0, 10]) = 5, EMA: 0.35·5 + 0.65·0 = 1.75
    for name in SignalFilter.FIELDS:
        assert getattr(out, name) == pytest.approx(1.75)
    assert out.roll == 10.0
    assert out.iris_distance_px == 100.0
    assert out.ts == 0.1


def test_signal_filter_passes_invalid_frame_without_update():
    sf = SignalFilter(FilterConfig())
    sf.update(frame(0.0, 1.0))
    invalid = FrameFeatures.invalid(0.1)
    assert sf.update(invalid) is invalid
    assert sf.update(frame(0.2, 1.0)).ear == pytest.approx(1.0)


def test_signal_filter_reset():
    sf = SignalFilter(FilterConfig())
    sf.update(frame(0.0, 5.0))
    sf.reset()
    assert sf.update(frame(0.1, 2.0)).yaw == pytest.approx(2.0)
