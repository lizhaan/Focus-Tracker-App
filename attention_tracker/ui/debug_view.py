"""Режим налагодження: превʼю камери з landmark-ами та значеннями сигналів.

Кадр дзеркалиться тільки тут, для показу (обробляється недзеркальний).
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from attention_tracker.config import AppConfig
from attention_tracker.models import FrameFeatures, FrameResult
from attention_tracker.vision.landmarks import (
    HEAD_POSE_INDICES,
    LEFT_EYE_EAR,
    LEFT_IRIS,
    RIGHT_EYE_EAR,
    RIGHT_IRIS,
)
from attention_tracker.workers.pipeline_worker import DebugFrame, PipelineWorker

# Оформлення превʼю (BGR)
_MESH_COLOR = (90, 90, 90)
_EYE_COLOR = (0, 200, 255)
_IRIS_COLOR = (0, 255, 0)
_POSE_COLOR = (255, 120, 0)
_MESH_RADIUS = 1
_KEY_RADIUS = 3
_PREVIEW_MIN_SIZE = (640, 360)

_EYE_POINTS = RIGHT_EYE_EAR + LEFT_EYE_EAR


def _fmt(value: float | None, digits: int = 3, signed: bool = True) -> str:
    if value is None:
        return "—"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def draw_landmarks(frame_bgr: np.ndarray, landmarks: np.ndarray | None) -> np.ndarray:
    """Копія кадру з намальованими точками (без дзеркалення)."""
    out = frame_bgr.copy()
    if landmarks is None:
        return out
    h, w = out.shape[:2]
    px = (landmarks[:, :2] * (w, h)).astype(int)
    for u, v in px:
        cv2.circle(out, (int(u), int(v)), _MESH_RADIUS, _MESH_COLOR, -1)
    for idx, color in (
        *((i, _EYE_COLOR) for i in _EYE_POINTS),
        *((i, _POSE_COLOR) for i in HEAD_POSE_INDICES),
        (RIGHT_IRIS, _IRIS_COLOR),
        (LEFT_IRIS, _IRIS_COLOR),
    ):
        cv2.circle(out, (int(px[idx][0]), int(px[idx][1])), _KEY_RADIUS, color, -1)
    return out


class DebugView(QWidget):
    def __init__(self, cfg: AppConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cfg = cfg
        self._worker: PipelineWorker | None = None
        self._smoothed: FrameFeatures | None = None
        self._fps: float | None = None

        self._preview = QLabel("Камера вимкнена")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumSize(*_PREVIEW_MIN_SIZE)

        self._values = QLabel()
        self._values.setFont(QFont("Consolas"))
        self._values.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._values.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self._button = QPushButton("Старт камери")
        self._button.clicked.connect(self._toggle)

        side = QVBoxLayout()
        side.addWidget(self._button)
        side.addWidget(self._values, 1)
        layout = QHBoxLayout(self)
        layout.addWidget(self._preview, 1)
        layout.addLayout(side)
        self._render_values(None)

    # --- керування worker-ом --------------------------------------------------

    def _toggle(self) -> None:
        if self._worker is None:
            self.start()
        else:
            self.stop()

    def start(self) -> None:
        if self._worker is not None:
            return
        worker = PipelineWorker(self._cfg)
        worker.debug_frame.connect(self._on_debug_frame)
        worker.frame_processed.connect(self._on_frame_processed)
        worker.fps_updated.connect(self._on_fps)
        worker.camera_error.connect(self._on_error)
        worker.finished.connect(self._on_finished)
        self._worker = worker
        self._button.setText("Стоп камери")
        worker.start()

    def stop(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait()

    @Slot()
    def _on_finished(self) -> None:
        self._worker = None
        self._fps = None
        self._button.setText("Старт камери")
        if self._preview.pixmap() is not None and not self._preview.pixmap().isNull():
            self._preview.setText("Камера вимкнена")  # setText прибирає pixmap; помилку не перетираємо

    # --- слоти даних -------------------------------------------------------

    @Slot(object)
    def _on_frame_processed(self, result: FrameResult) -> None:
        self._smoothed = result.features

    @Slot(float)
    def _on_fps(self, fps: float) -> None:
        self._fps = fps

    @Slot(str)
    def _on_error(self, message: str) -> None:
        self._preview.setText(message)

    @Slot(object)
    def _on_debug_frame(self, item: DebugFrame) -> None:
        shown = cv2.flip(draw_landmarks(item.frame_bgr, item.landmarks), 1)
        rgb = cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        image = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self._preview.setPixmap(
            QPixmap.fromImage(image).scaled(
                self._preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self._render_values(item.raw)

    def _render_values(self, raw: FrameFeatures | None) -> None:
        s = self._smoothed
        rows = [
            ("EAR", "ear", 3, False),
            ("Yaw °", "yaw", 1, True),
            ("Pitch °", "pitch", 1, True),
            ("Roll °", "roll", 1, True),
            ("Gaze_h", "gaze_h", 3, False),
            ("Gaze_v", "gaze_v", 3, True),
        ]
        lines = [f"{'':8} {'сирий':>8} {'згладж.':>8}"]
        for label, name, digits, signed in rows:
            r = getattr(raw, name) if raw is not None else None
            v = getattr(s, name) if s is not None and s.valid else None
            lines.append(
                f"{label:8} {_fmt(r, digits, signed):>8} {_fmt(v, digits, signed):>8}"
            )
        iris = raw.iris_distance_px if raw is not None else None
        lines.append("")
        lines.append(f"iris px  {_fmt(iris, 1, signed=False):>8}")
        lines.append(f"valid    {('так' if raw is not None and raw.valid else 'ні'):>8}")
        lines.append(f"FPS      {_fmt(self._fps, 1, signed=False):>8}")
        self._values.setText("\n".join(lines))

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        self.stop()
        super().closeEvent(event)
