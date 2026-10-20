import pytest

from attention_tracker.config import AlphaThresholdsConfig, DistractionConfig
from attention_tracker.models import FrameFeatures, Reason
from attention_tracker.vision.distraction import DistractionDetector

DT = 0.1


def ff(ts: float, yaw: float = 0.0, pitch: float = 0.0, gaze_h: float | None = 0.5,
       valid: bool = True) -> FrameFeatures:
    if not valid:
        return FrameFeatures.invalid(ts)
    return FrameFeatures(ts=ts, valid=True, ear=0.3, yaw=yaw, pitch=pitch, roll=0.0,
                         gaze_h=gaze_h, gaze_v=0.0, iris_distance_px=120.0)


def step(det, frame: FrameFeatures):
    return det.update(frame, frame)


@pytest.fixture
def det() -> DistractionDetector:
    d = DistractionDetector(DistractionConfig(), AlphaThresholdsConfig())
    d.reset(0.0)
    return d


def with_baseline(det, yaw=10.0, pitch=-5.0, gaze_h=0.45) -> float:
    """Кадри з постійними значеннями на [0, 2.0]; повертає час останнього кадру."""
    for i in range(21):
        step(det, ff(round(i * DT, 6), yaw, pitch, gaze_h))
    assert det.baseline_ready
    return 2.0


# --- baseline -----------------------------------------------------------------


def test_not_distracted_while_collecting_baseline(det):
    for i in range(20):  # 0.0 … 1.9 — ще не 2 с
        result = step(det, ff(i * DT, yaw=40.0, gaze_h=0.9))
        assert not result.distracted and result.reason is Reason.NONE
    assert not det.baseline_ready


def test_baseline_is_mean_of_first_two_seconds(det):
    for i in range(21):
        step(det, ff(round(i * DT, 6), yaw=10.0 + (i % 2), pitch=-5.0, gaze_h=0.45))
    yaw, pitch, gaze_h = det.baseline
    assert yaw == pytest.approx(10.0 + 10 / 21)
    assert pitch == pytest.approx(-5.0)
    assert gaze_h == pytest.approx(0.45)


def test_baseline_ignores_invalid_frames(det):
    step(det, ff(0.0, yaw=10.0))
    step(det, ff(1.0, valid=False))
    step(det, ff(2.0, valid=False))
    assert not det.baseline_ready  # 2 с минуло, але останній кадр невалідний
    step(det, ff(2.1, yaw=20.0))
    assert det.baseline[0] == pytest.approx(15.0)


def test_baseline_mean_across_180_boundary(det):
    for i in range(21):
        step(det, ff(round(i * DT, 6), pitch=179.0 if i % 2 else -179.0))
    assert abs(det.baseline[1]) == pytest.approx(180.0, abs=0.2)


def test_face_absent_during_baseline_is_reported(det):
    step(det, ff(0.0))
    assert step(det, ff(0.4, valid=False)).distracted is False
    result = step(det, ff(0.6, valid=False))
    assert result.distracted and result.reason is Reason.FACE_ABSENT


# --- пороги погляду --------------------------------------------------------------


@pytest.mark.parametrize(
    "gaze_h, distracted",
    [(0.45, False), (0.59, False), (0.61, True), (0.31, False), (0.29, True)],
)
def test_gaze_h_threshold_relative_to_baseline(det, gaze_h, distracted):
    t = with_baseline(det, gaze_h=0.45)
    result = step(det, ff(t + DT, yaw=10.0, pitch=-5.0, gaze_h=gaze_h))
    assert result.distracted is distracted
    assert result.reason is (Reason.GAZE_OUT if distracted else Reason.NONE)


@pytest.mark.parametrize(
    "yaw, distracted",
    [(34.0, False), (36.0, True), (-14.0, False), (-16.0, True)],
)
def test_yaw_threshold_relative_to_baseline(det, yaw, distracted):
    t = with_baseline(det, yaw=10.0)
    assert step(det, ff(t + DT, yaw=yaw, pitch=-5.0, gaze_h=0.45)).distracted is distracted


def test_yaw_delta_uses_wrap(det):
    t = with_baseline(det, yaw=170.0, gaze_h=0.5)
    assert not step(det, ff(t + DT, yaw=-175.0, pitch=-5.0)).distracted  # Δ = 15°
    assert step(det, ff(t + 2 * DT, yaw=-160.0, pitch=-5.0)).distracted  # Δ = 30°


def test_missing_gaze_does_not_trigger(det):
    t = with_baseline(det)
    assert not step(det, ff(t + DT, yaw=10.0, pitch=-5.0, gaze_h=None)).distracted


def test_decision_uses_smoothed_validity_uses_raw(det):
    t = with_baseline(det, yaw=0.0, pitch=0.0, gaze_h=0.5)
    raw = ff(t + DT, yaw=40.0)  # валідний (≤ 45°), але сам по собі > 25°
    smoothed = ff(t + DT, yaw=20.0)
    assert not det.update(raw, smoothed).distracted


# --- невалідні кадри ---------------------------------------------------------


def test_short_invalid_holds_previous_decision(det):
    t = with_baseline(det)
    gaze_out = step(det, ff(t + DT, yaw=10.0, pitch=-5.0, gaze_h=0.9))
    assert gaze_out.reason is Reason.GAZE_OUT
    held = step(det, ff(t + DT + 0.45, valid=False))  # < T_noface від валідного
    assert held.distracted and held.reason is Reason.GAZE_OUT


def test_short_invalid_after_focused_frame_stays_not_distracted(det):
    t = with_baseline(det)
    step(det, ff(t + DT, yaw=10.0, pitch=-5.0, gaze_h=0.45))
    assert not step(det, ff(t + 0.4, valid=False)).distracted


def test_long_invalid_is_face_absent_and_requests_filter_reset_once(det):
    t = with_baseline(det)
    step(det, ff(t + 0.3, valid=False))
    assert not det.consume_filter_reset()
    result = step(det, ff(t + 0.6, valid=False))
    assert result.distracted and result.reason is Reason.FACE_ABSENT
    assert det.consume_filter_reset()
    step(det, ff(t + 0.9, valid=False))
    assert not det.consume_filter_reset()


@pytest.mark.parametrize("yaw, pitch", [(60.0, -5.0), (-40.0, -5.0), (10.0, 40.0), (10.0, -50.0)])
def test_large_angles_are_head_away(det, yaw, pitch):
    t = with_baseline(det, yaw=10.0, pitch=-5.0)
    step(det, ff(t + DT, yaw=10.0, pitch=-5.0, gaze_h=0.45))
    assert not step(det, ff(t + 0.3, yaw=yaw, pitch=pitch)).distracted  # ще ≤ T_noface
    result = step(det, ff(t + 0.7, yaw=yaw, pitch=pitch))
    assert result.distracted and result.reason is Reason.HEAD_AWAY


def test_face_returns_after_long_absence(det):
    t = with_baseline(det)
    step(det, ff(t + 1.0, valid=False))
    assert det.consume_filter_reset()
    assert not step(det, ff(t + 1.1, yaw=10.0, pitch=-5.0, gaze_h=0.45)).distracted
    step(det, ff(t + 2.0, valid=False))
    assert det.consume_filter_reset()  # новий епізод — новий запит на скидання


# --- reset ---------------------------------------------------------------------


def test_reset_starts_new_baseline(det):
    t = with_baseline(det, yaw=10.0)
    det.reset(t + 1.0)
    assert not det.baseline_ready
    for i in range(21):
        step(det, ff(round(t + 1.0 + i * DT, 6), yaw=-20.0))
    assert det.baseline[0] == pytest.approx(-20.0)


def test_screen_point_is_none_in_alpha(det):
    t = with_baseline(det)
    assert step(det, ff(t + DT, yaw=10.0, pitch=-5.0, gaze_h=0.45)).screen_point is None
