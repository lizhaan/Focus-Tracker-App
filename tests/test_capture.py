import numpy as np
import pytest

from attention_tracker.capture.camera import Camera
from attention_tracker.config import CameraConfig, LandmarkerConfig

MODEL_AVAILABLE = LandmarkerConfig().model_path.is_file()


def test_camera_read_without_open_returns_none():
    cam = Camera(CameraConfig())
    assert not cam.is_open
    assert cam.read() is None
    assert cam.camera_id is None
    assert cam.frame_size is None


def test_landmarker_missing_model_raises(tmp_path):
    from attention_tracker.capture.landmarker import FaceLandmarkerWrapper

    with pytest.raises(FileNotFoundError):
        FaceLandmarkerWrapper(LandmarkerConfig(model_path=tmp_path / "missing.task"))


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="немає assets/face_landmarker.task")
def test_landmarker_no_face_returns_none_and_accepts_repeated_ts():
    from attention_tracker.capture.landmarker import FaceLandmarkerWrapper

    black = np.zeros((480, 640, 3), dtype=np.uint8)
    with FaceLandmarkerWrapper(LandmarkerConfig()) as lm:
        assert lm.detect(black, 100) is None
        # однакова або менша мітка не валить VIDEO-режим
        assert lm.detect(black, 100) is None
        assert lm.detect(black, 50) is None


def test_draw_landmarks_does_not_modify_frame():
    from attention_tracker.ui.debug_view import draw_landmarks
    from attention_tracker.vision.landmarks import N_LANDMARKS

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    landmarks = np.full((N_LANDMARKS, 3), 0.5)
    out = draw_landmarks(frame, landmarks)
    assert out.shape == frame.shape
    assert out.any()
    assert not frame.any()
    assert draw_landmarks(frame, None).shape == frame.shape
