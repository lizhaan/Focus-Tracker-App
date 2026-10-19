"""Доступ до вебкамери.

Кадр не дзеркалиться і нікуди не зберігається. Мітка часу береться
`time.monotonic()` одразу після зчитування кадру (час — у секундах).
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from attention_tracker.config import CameraConfig


class CameraError(RuntimeError):
    """Камеру не вдалося відкрити."""


class Camera:
    def __init__(self, cfg: CameraConfig) -> None:
        self._cfg = cfg
        self._cap: cv2.VideoCapture | None = None
        self._index: int | None = None

    @property
    def is_open(self) -> bool:
        return self._cap is not None

    @property
    def camera_id(self) -> str | None:
        """Ідентифікатор камери для профілю (поки що — індекс пристрою)."""
        return None if self._index is None else f"index:{self._index}"

    @property
    def frame_size(self) -> tuple[int, int] | None:
        if self._cap is None:
            return None
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def open(
        self, index: int | None = None, width: int | None = None, height: int | None = None
    ) -> None:
        """Відкриває камеру; якщо бажана роздільна здатність недоступна — запасна."""
        self.release()
        index = self._cfg.index if index is None else index
        width = self._cfg.width if width is None else width
        height = self._cfg.height if height is None else height

        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            raise CameraError(f"Не вдалося відкрити камеру #{index}")

        if not self._try_size(cap, width, height):
            self._try_size(cap, self._cfg.fallback_width, self._cfg.fallback_height)

        self._cap = cap
        self._index = index

    @staticmethod
    def _try_size(cap: cv2.VideoCapture, width: int, height: int) -> bool:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == width
            and int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == height
        )

    def read(self) -> tuple[np.ndarray, float] | None:
        """(кадр BGR, ts monotonic) або `None`, якщо кадр не отримано."""
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        ts = time.monotonic()
        if not ok or frame is None:
            return None
        return frame, ts

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap = None
        self._index = None
