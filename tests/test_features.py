import math

import cv2
import numpy as np
import pytest

from attention_tracker.vision.features import (
    FeatureExtractor,
    camera_matrix,
    ear,
    eye_gaze,
    gaze_raw,
    head_pose,
    iris_distance_px,
    to_pixels,
    wrap,
)
from attention_tracker.vision.landmarks import (
    FACE_MODEL_3D,
    HEAD_POSE_INDICES,
    LEFT_EYE_EAR,
    LEFT_EYE_GAZE,
    N_LANDMARKS,
    RIGHT_EYE_EAR,
    RIGHT_EYE_GAZE,
)

W, H = 640, 480  # W ≠ H — перевіряємо, що все рахується в пікселях


def landmarks_from_px(points: dict[int, tuple[float, float]]) -> np.ndarray:
    """Нормалізовані landmarks (478, 3) з точок у пікселях."""
    lm = np.zeros((N_LANDMARKS, 3), dtype=np.float64)
    for idx, (u, v) in points.items():
        lm[idx] = (u / W, v / H, 0.0)
    return lm


def project_face(rvec, tvec) -> dict[int, tuple[float, float]]:
    pts, _ = cv2.projectPoints(
        FACE_MODEL_3D, np.asarray(rvec, float), np.asarray(tvec, float),
        camera_matrix(W, H), np.zeros((4, 1)),
    )
    return {idx: tuple(p) for idx, p in zip(HEAD_POSE_INDICES, pts.reshape(-1, 2))}


def eye_points(eye, x0: float, y0: float, width: float, gap: float) -> dict:
    """Око P1..P6 у пікселях: кути на (x0, y0) і (x0+width, y0), повіки ±gap/2."""
    p1, p2, p3, p4, p5, p6 = eye
    return {
        p1: (x0, y0), p4: (x0 + width, y0),
        p2: (x0 + 0.3 * width, y0 - gap / 2), p3: (x0 + 0.7 * width, y0 - gap / 2),
        p5: (x0 + 0.7 * width, y0 + gap / 2), p6: (x0 + 0.3 * width, y0 + gap / 2),
    }


# --- wrap -------------------------------------------------------------


@pytest.mark.parametrize(
    "theta, expected",
    [(0.0, 0.0), (179.0, 179.0), (181.0, -179.0), (-181.0, 179.0),
     (360.0, 0.0), (190.0, -170.0), (-190.0, 170.0), (540.0, -180.0)],
)
def test_wrap(theta, expected):
    assert wrap(theta) == pytest.approx(expected)


def test_wrap_at_180_boundary_stays_on_boundary():
    assert abs(wrap(180.0)) == pytest.approx(180.0)
    assert abs(wrap(-180.0)) == pytest.approx(180.0)


# --- EAR ----------------------------------------------------------------


def test_ear_known_geometry_in_pixels():
    # ширина 60 px, щілина 12 px → EAR = (12 + 12) / (2·60) = 0.2
    pts = {**eye_points(RIGHT_EYE_EAR, 100, 200, 60, 12),
           **eye_points(LEFT_EYE_EAR, 300, 200, 60, 12)}
    px = to_pixels(landmarks_from_px(pts), W, H)
    assert ear(px) == pytest.approx(0.2)


def test_ear_closed_lower_than_open():
    open_px = to_pixels(landmarks_from_px({**eye_points(RIGHT_EYE_EAR, 100, 200, 60, 12),
                                           **eye_points(LEFT_EYE_EAR, 300, 200, 60, 12)}), W, H)
    closed_px = to_pixels(landmarks_from_px({**eye_points(RIGHT_EYE_EAR, 100, 200, 60, 1),
                                             **eye_points(LEFT_EYE_EAR, 300, 200, 60, 1)}), W, H)
    assert ear(closed_px) < ear(open_px)


def test_ear_degenerate_eye_is_finite():
    pts = {**eye_points(RIGHT_EYE_EAR, 100, 200, 0, 12),
           **eye_points(LEFT_EYE_EAR, 300, 200, 60, 12)}
    assert math.isfinite(ear(to_pixels(landmarks_from_px(pts), W, H)))


# --- head pose ---------------------------------------------------------


def test_frontal_face_gives_zero_angles():
    px = to_pixels(landmarks_from_px(project_face([0, 0, 0], [0, 0, 500])), W, H)
    pose = head_pose(px, W, H)
    assert pose is not None
    assert pose.yaw == pytest.approx(0.0, abs=1.0)
    assert pose.pitch == pytest.approx(0.0, abs=1.0)
    assert pose.roll == pytest.approx(0.0, abs=1.0)


@pytest.mark.parametrize("axis, field", [(0, "pitch"), (1, "yaw"), (2, "roll")])
def test_single_axis_rotation_is_recovered(axis, field):
    angle = 15.0
    rvec = [0.0, 0.0, 0.0]
    rvec[axis] = math.radians(angle)
    px = to_pixels(landmarks_from_px(project_face(rvec, [0, 0, 500])), W, H)
    pose = head_pose(px, W, H)
    assert pose is not None
    for name in ("yaw", "pitch", "roll"):
        expected = angle if name == field else 0.0
        assert getattr(pose, name) == pytest.approx(expected, abs=1.5)


def test_head_pose_with_extrinsic_guess_matches():
    px = to_pixels(landmarks_from_px(project_face([0, math.radians(10), 0], [0, 0, 500])), W, H)
    first = head_pose(px, W, H)
    second = head_pose(px, W, H, prev=(first.rvec, first.tvec))
    assert second is not None
    assert second.yaw == pytest.approx(first.yaw, abs=0.1)


# --- gaze ------------------------------------------------------------


def gaze_face(iris_shift: tuple[float, float] = (0.0, 0.0)) -> dict:
    """Обидва ока шириною 60 px; райдужки по центру + зсув."""
    dx, dy = iris_shift
    r_cl, r_cr, r_i = RIGHT_EYE_GAZE
    l_cl, l_cr, l_i = LEFT_EYE_GAZE
    return {
        r_cl: (100, 200), r_cr: (160, 200), r_i: (130 + dx, 200 + dy),
        l_cl: (220, 200), l_cr: (280, 200), l_i: (250 + dx, 200 + dy),
    }


def per_eye(px):
    return (eye_gaze(*(px[i] for i in RIGHT_EYE_GAZE)),
            eye_gaze(*(px[i] for i in LEFT_EYE_GAZE)))


def test_centered_iris_gives_half_and_zero():
    px = to_pixels(landmarks_from_px(gaze_face()), W, H)
    assert gaze_raw(px) == pytest.approx((0.5, 0.0))


def test_iris_shift_right_increases_both_eyes():
    base_r, base_l = per_eye(to_pixels(landmarks_from_px(gaze_face()), W, H))
    shift_r, shift_l = per_eye(to_pixels(landmarks_from_px(gaze_face((6, 0))), W, H))
    assert shift_r[0] > base_r[0]
    assert shift_l[0] > base_l[0]
    assert shift_r[0] == pytest.approx(0.6)
    assert shift_l[0] == pytest.approx(0.6)


def test_iris_below_axis_gives_positive_v():
    px = to_pixels(landmarks_from_px(gaze_face((0, 3))), W, H)
    _, gaze_v = gaze_raw(px)
    assert gaze_v == pytest.approx(3 / 60)


def test_gaze_invariant_to_roll_and_scale():
    c_l, c_r, iris = np.array([0.0, 0.0]), np.array([60.0, 0.0]), np.array([36.0, 3.0])
    expected = eye_gaze(c_l, c_r, iris)
    a = math.radians(30)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    offset = np.array([200.0, 150.0])
    transformed = [2.0 * rot @ p + offset for p in (c_l, c_r, iris)]
    assert eye_gaze(*transformed) == pytest.approx(expected)


def test_degenerate_eye_axis_gives_none():
    p = np.array([10.0, 10.0])
    assert eye_gaze(p, p, p) is None


# --- FeatureExtractor ----------------------------------------------------------


def full_face_landmarks() -> np.ndarray:
    """Проєкція 3D-моделі обличчя + внутрішні кути й райдужки біля спроєктованих 33/263."""
    pts = project_face([0, 0, 0], [0, 0, 500])
    r_cl, r_cr, r_i = RIGHT_EYE_GAZE
    l_cl, l_cr, l_i = LEFT_EYE_GAZE
    r_out, l_out = np.array(pts[r_cl]), np.array(pts[l_cr])
    pts[r_cr] = tuple(r_out + (20, 0))
    pts[r_i] = tuple(r_out + (10, 0))
    pts[l_cl] = tuple(l_out - (20, 0))
    pts[l_i] = tuple(l_out - (10, 0))
    return landmarks_from_px(pts)


def test_iris_distance_px():
    px = to_pixels(landmarks_from_px(gaze_face()), W, H)
    assert iris_distance_px(px) == pytest.approx(120.0)


def test_extractor_valid_frame():
    f = FeatureExtractor().extract(full_face_landmarks(), ts=1.0, frame_w=W, frame_h=H)
    assert f.valid
    assert f.ts == 1.0
    assert f.yaw == pytest.approx(0.0, abs=1.0)
    assert f.gaze_h is not None and f.iris_distance_px is not None


def test_extractor_no_face_is_invalid_and_resets_pose():
    ex = FeatureExtractor()
    ex.extract(full_face_landmarks(), ts=1.0, frame_w=W, frame_h=H)
    assert ex._prev_pose is not None
    f = ex.extract(None, ts=2.0, frame_w=W, frame_h=H)
    assert not f.valid
    assert f.ts == 2.0
    assert ex._prev_pose is None
