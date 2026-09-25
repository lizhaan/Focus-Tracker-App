from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

import numpy as np
import cv2


RIGHT_EYE = {"P1": 33, "P2": 160, "P3": 158, "P4": 133, "P5": 153, "P6": 144}
LEFT_EYE = {"P1": 362, "P2": 385, "P3": 387, "P4": 263, "P5": 373, "P6": 380}

IRIS_CENTER = {"right": 473, "left": 468}

HEAD_POSE_LANDMARKS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 362,
    "right_eye_outer": 33,
    "left_mouth_corner": 291,
    "right_mouth_corner": 61,
}

MODEL_POINTS_3D = np.array(
    [
        (0.0, 0.0, 0.0),  # Кінчик носа
        (0.0, -63.6, -12.5),  # Підборіддя
        (-43.3, 32.7, -26.0),  # Лівий зовн. кут ока
        (43.3, 32.7, -26.0),  # Правий зовн. кут ока
        (-28.9, -28.9, -24.1),  # Лівий кут рота
        (28.9, -28.9, -24.1),  # Правий кут рота
    ],
    dtype=np.float64,
)

_EPS = 1e-9


def _dist(a: np.ndarray, b: np.ndarray, use_z: bool = True) -> float:
    """Евклідова відстань між двома точками"""
    if use_z:
        diff = a - b
    else:
        diff = a[:2] - b[:2]
    return float(np.linalg.norm(diff))