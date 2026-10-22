"""Режим налагодження: превʼю камери з landmark-ами та значеннями сигналів.

Кадр дзеркалиться тільки тут, для показу (обробляється недзеркальний).
Камеру не запускає сам — підключається до worker-а поточної сесії.
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from attention_tracker.calibration.profile import CalibrationProfile
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
_IDLE_TEXT = "Камера вимкнена — запустіть сесію на вкладці «Сесія»"

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
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._worker: PipelineWorker | None = None
        self._profile: CalibrationProfile | None = None
        self._result: FrameResult | None = None
        self._fps: float | None = None

        self._preview = QLabel(_IDLE_TEXT)
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumSize(*_PREVIEW_MIN_SIZE)

        self._values = QLabel()
        self._values.setFont(QFont("Consolas"))
        self._values.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._values.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        layout = QHBoxLayout(self)
        layout.addWidget(self._preview, 1)
        layout.addWidget(self._values)
        self._render_values(None)

    # --- підключення до worker-а --------------------------------------------

    def attach(self, worker: PipelineWorker, profile: CalibrationProfile | None = None) -> None:
        self.detach()
        self._worker = worker
        self._profile = profile
        worker.debug_frame.connect(self._on_debug_frame)
        worker.frame_processed.connect(self._on_frame_processed)
        worker.fps_updated.connect(self._on_fps)

    def detach(self) -> None:
        if self._worker is not None:
            self._worker.debug_frame.disconnect(self._on_debug_frame)
            self._worker.frame_processed.disconnect(self._on_frame_processed)
            self._worker.fps_updated.disconnect(self._on_fps)
            self._worker = None
        self._result = None
        self._fps = None
        self._preview.setText(_IDLE_TEXT)
        self._render_values(None)

    def show_error(self, message: str) -> None:
        self._preview.setText(message)

    # --- слоти даних -------------------------------------------------------

    @Slot(object)
    def _on_frame_processed(self, result: FrameResult) -> None:
        self._result = result

    @Slot(float)
    def _on_fps(self, fps: float) -> None:
        self._fps = fps

    @Slot(object)
    def _on_debug_frame(self, item: DebugFrame) -> None:
        if not self.isVisible():
            return  # вкладка прихована — не витрачаємо час на малювання
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
        r_ = self._result
        s = r_.features if r_ is not None else None
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
        d = r_.distraction if r_ is not None else None
        lines.append("")
        lines.append(f"iris px  {_fmt(iris, 1, signed=False):>8}")
        lines.append(f"valid    {('так' if raw is not None and raw.valid else 'ні'):>8}")
        lines.append(f"FPS      {_fmt(self._fps, 1, signed=False):>8}")
        lines.append("")
        lines.append(f"distr.   {('так' if d is not None and d.distracted else 'ні'):>8}")
        lines.append(f"reason   {(d.reason.name if d is not None else '—'):>8}")
        point = d.screen_point if d is not None else None
        lines.append(f"sx       {_fmt(point[0] if point else None, 2):>8}")
        lines.append(f"sy       {_fmt(point[1] if point else None, 2):>8}")
        lines.append(f"ΔYaw °   {_fmt(d.d_yaw if d else None, 1):>8}")
        lines.append(f"ΔPitch ° {_fmt(d.d_pitch if d else None, 1):>8}")
        p = self._profile
        threshold = "вимк." if p is not None and not p.pitch_enabled else _fmt(p.pitch_threshold if p else None, 1)
        lines.append(f"вниз °   {_fmt(d.pitch_down if d else None, 1):>8}  поріг {threshold}")
        if p is not None and p.ear_enabled:
            lines.append(f"EAR поріг {_fmt(p.ear_threshold, 3, signed=False):>7}")
        lines.append(f"state    {(r_.state.name if r_ is not None and r_.state else '—')}")
        self._values.setText("\n".join(lines))
