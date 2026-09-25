import math
import unittest

import cv2
import numpy as np

from core.features import (
    HEAD_POSE_LANDMARKS,
    LEFT_EYE,
    MODEL_POINTS_3D,
    RIGHT_EYE,
    IRIS_CENTER,
    MetricsSmoother,
    _camera_matrix,
    compute_ear,
    compute_gaze_raw,
    compute_head_pose,
)

N_LANDMARKS = 478


def make_landmarks(overrides: dict) -> np.ndarray:
    landmarks = np.zeros((N_LANDMARKS, 3), dtype=np.float64)
    for idx, (x, y, z) in overrides.items():
        landmarks[idx] = (x, y, z)
    return landmarks


# 1. EAR
class TestEAR(unittest.TestCase):
    def _eye_landmarks(self, eye: dict, horizontal: float, vertical_gap: float) -> dict:
        return {
            eye["P1"]: (0.0, 0.0, 0.0),  # зовнішній кут
            eye["P4"]: (horizontal, 0.0, 0.0),  # внутрішній кут
            eye["P2"]: (horizontal * 0.3, -vertical_gap / 2, 0.0),  # верхня повіка
            eye["P3"]: (horizontal * 0.7, -vertical_gap / 2, 0.0),
            eye["P5"]: (horizontal * 0.7, vertical_gap / 2, 0.0),  # нижня повіка
            eye["P6"]: (horizontal * 0.3, vertical_gap / 2, 0.0),
        }

    def _make_face(self, vertical_gap_right: float, vertical_gap_left: float) -> np.ndarray:
        overrides = {}
        overrides.update(self._eye_landmarks(RIGHT_EYE, horizontal=0.1, vertical_gap=vertical_gap_right))
        overrides.update(self._eye_landmarks(LEFT_EYE, horizontal=0.1, vertical_gap=vertical_gap_left))
        return make_landmarks(overrides)

    def test_open_eyes_give_expected_ear(self):
        landmarks = self._make_face(vertical_gap_right=0.06, vertical_gap_left=0.06)
        ear = compute_ear(landmarks)
        self.assertAlmostEqual(ear, 0.6, places=6)
        self.assertGreater(ear, 0.15)  # явно "відкрите" око

    def test_closed_eyes_give_low_ear(self):
        landmarks = self._make_face(vertical_gap_right=0.002, vertical_gap_left=0.002)
        ear = compute_ear(landmarks)
        self.assertLess(ear, 0.05)  # явно "закрите" око

    def test_open_ear_greater_than_closed(self):
        open_landmarks = self._make_face(vertical_gap_right=0.06, vertical_gap_left=0.06)
        closed_landmarks = self._make_face(vertical_gap_right=0.002, vertical_gap_left=0.002)
        self.assertGreater(compute_ear(open_landmarks), compute_ear(closed_landmarks))

    def test_ear_handles_degenerate_horizontal_distance(self):
        overrides = {
            RIGHT_EYE["P1"]: (0.0, 0.0, 0.0),
            RIGHT_EYE["P4"]: (0.0, 0.0, 0.0),
            RIGHT_EYE["P2"]: (0.0, -0.01, 0.0),
            RIGHT_EYE["P3"]: (0.0, -0.01, 0.0),
            RIGHT_EYE["P5"]: (0.0, 0.01, 0.0),
            RIGHT_EYE["P6"]: (0.0, 0.01, 0.0),
        }
        overrides.update(self._eye_landmarks(LEFT_EYE, horizontal=0.1, vertical_gap=0.05))
        landmarks = make_landmarks(overrides)
        ear = compute_ear(landmarks)
        self.assertTrue(math.isfinite(ear))


# Head Pose
class TestHeadPose(unittest.TestCase):
    WIDTH, HEIGHT = 640, 480

    def _project_and_build_landmarks(self, rvec, tvec) -> np.ndarray:
        camera_matrix = _camera_matrix(self.WIDTH, self.HEIGHT)
        dist_coeffs = np.zeros((4, 1))

        image_points_px, _ = cv2.projectPoints(
            MODEL_POINTS_3D, rvec, tvec, camera_matrix, dist_coeffs
        )
        image_points_px = image_points_px.reshape(-1, 2)

        overrides = {}
        for (idx), (px, py) in zip(HEAD_POSE_LANDMARKS.values(), image_points_px):
            overrides[idx] = (px / self.WIDTH, py / self.HEIGHT, 0.0)
        return make_landmarks(overrides)

    def test_frontal_face_gives_near_zero_yaw_pitch(self):
        rvec = np.array([[0.0], [0.0], [0.0]])
        tvec = np.array([[0.0], [0.0], [500.0]])
        landmarks = self._project_and_build_landmarks(rvec, tvec)

        pose = compute_head_pose(landmarks, self.WIDTH, self.HEIGHT)

        self.assertIsNotNone(pose)
        self.assertAlmostEqual(pose.yaw, 0.0, delta=1.0)
        self.assertAlmostEqual(pose.pitch, 0.0, delta=1.0)
        self.assertAlmostEqual(pose.roll, 0.0, delta=1.0)

    def test_yaw_rotation_is_recovered_with_correct_sign_and_magnitude(self):
        yaw_deg = 20.0
        rvec = np.array([[0.0], [math.radians(yaw_deg)], [0.0]])
        tvec = np.array([[0.0], [0.0], [500.0]])
        landmarks = self._project_and_build_landmarks(rvec, tvec)

        pose = compute_head_pose(landmarks, self.WIDTH, self.HEIGHT)

        self.assertIsNotNone(pose)
        self.assertAlmostEqual(pose.yaw, yaw_deg, delta=1.5)
        self.assertAlmostEqual(pose.pitch, 0.0, delta=1.5)

    def test_pitch_rotation_is_recovered_with_correct_sign_and_magnitude(self):
        pitch_deg = -15.0  # нахил голови вниз
        rvec = np.array([[math.radians(pitch_deg)], [0.0], [0.0]])
        tvec = np.array([[0.0], [0.0], [500.0]])
        landmarks = self._project_and_build_landmarks(rvec, tvec)

        pose = compute_head_pose(landmarks, self.WIDTH, self.HEIGHT)

        self.assertIsNotNone(pose)
        self.assertAlmostEqual(pose.pitch, pitch_deg, delta=1.5)
        self.assertAlmostEqual(pose.yaw, 0.0, delta=1.5)


# Gaze_raw — граничні значення

class TestGazeRaw(unittest.TestCase):
    def _base_eye_overrides(self, eye: dict, iris_idx: int, iris_pos) -> dict:
        return {
            eye["P1"]: (0.0, 0.0, 0.0),
            eye["P4"]: (0.1, 0.0, 0.0),
            eye["P2"]: (0.03, -0.02, 0.0),
            eye["P3"]: (0.07, -0.02, 0.0),
            eye["P5"]: (0.07, 0.02, 0.0),
            eye["P6"]: (0.03, 0.02, 0.0),
            iris_idx: iris_pos,
        }

    def _both_eyes(self, iris_offset_from_outer: float) -> np.ndarray:
        overrides = {}
        overrides.update(
            self._base_eye_overrides(
                RIGHT_EYE, IRIS_CENTER["right"], (iris_offset_from_outer, 0.0, 0.0)
            )
        )
        overrides.update(
            self._base_eye_overrides(
                LEFT_EYE, IRIS_CENTER["left"], (iris_offset_from_outer, 0.0, 0.0)
            )
        )
        return make_landmarks(overrides)

    def test_iris_at_outer_corner_gives_zero_horizontal_ratio(self):
        # iris = outer corner (x=0) => D_out = 0 => Gaze_raw = 0/D_in = 0
        landmarks = self._both_eyes(iris_offset_from_outer=0.0)
        gaze_raw, gaze_vertical_raw = compute_gaze_raw(landmarks)
        self.assertAlmostEqual(gaze_raw, 0.0, places=6)
        self.assertTrue(math.isfinite(gaze_vertical_raw))

    def test_iris_at_inner_corner_does_not_raise_and_is_finite(self):
        landmarks = self._both_eyes(iris_offset_from_outer=0.1)
        gaze_raw, gaze_vertical_raw = compute_gaze_raw(landmarks)
        self.assertTrue(math.isfinite(gaze_raw))
        self.assertTrue(math.isfinite(gaze_vertical_raw))
        self.assertEqual(gaze_raw, 0.0)

    def test_iris_in_the_middle_gives_ratio_around_one(self):
        landmarks = self._both_eyes(iris_offset_from_outer=0.05)
        gaze_raw, _ = compute_gaze_raw(landmarks)
        self.assertAlmostEqual(gaze_raw, 1.0, places=6)


# 5. MetricsSmoother


class TestMetricsSmoother(unittest.TestCase):
    def test_rejects_invalid_window_size(self):
        with self.assertRaises(ValueError):
            MetricsSmoother(window_size=2, alpha=0.3)
        with self.assertRaises(ValueError):
            MetricsSmoother(window_size=6, alpha=0.3)

    def test_rejects_invalid_alpha(self):
        with self.assertRaises(ValueError):
            MetricsSmoother(window_size=3, alpha=0.0)
        with self.assertRaises(ValueError):
            MetricsSmoother(window_size=3, alpha=1.5)

    def test_median_filter_suppresses_single_frame_outlier(self):
        smoother = MetricsSmoother(window_size=3, alpha=0.3)
        signal = [1.0, 1.0, 1.0, 100.0, 1.0, 1.0, 1.0]

        outputs = [smoother.update("ear", v) for v in signal]
        spike_index = signal.index(100.0)

        self.assertLess(outputs[spike_index], 2.0)

    def test_ema_converges_towards_constant_signal(self):
        smoother = MetricsSmoother(window_size=3, alpha=0.4)
        outputs = [smoother.update("yaw", 5.0) for _ in range(20)]
        self.assertAlmostEqual(outputs[-1], 5.0, places=3)

    def test_independent_signals_do_not_interfere(self):
        smoother = MetricsSmoother(window_size=3, alpha=0.3)
        for _ in range(5):
            smoother.update("ear", 0.3)
        for _ in range(5):
            smoother.update("pitch", -10.0)

        self.assertAlmostEqual(smoother.update("ear", 0.3), 0.3, places=3)
        self.assertAlmostEqual(smoother.update("pitch", -10.0), -10.0, places=3)

    def test_reset_clears_state(self):
        smoother = MetricsSmoother(window_size=3, alpha=0.3)
        smoother.update("ear", 0.9)
        smoother.reset("ear")
        first_value_after_reset = smoother.update("ear", 0.1)
        self.assertAlmostEqual(first_value_after_reset, 0.1, places=6)


if __name__ == "__main__":
    unittest.main()