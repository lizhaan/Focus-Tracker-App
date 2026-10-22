"""Профіль калібрування: прив'язаний до (user_id, monitor_id, camera_id)."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields

from attention_tracker.vision.gaze_model import GazeRegressor

PROFILE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CalibrationProfile:
    user_id: int
    monitor_id: str
    camera_id: str
    frame_w: int
    frame_h: int

    # крок 1 — центр
    yaw_baseline: float
    pitch_baseline: float
    ear_center: float
    # крок 3 — знак нахилу вниз
    s_pitch: int
    # кроки 4, 5 — низ екрана і телефон
    ear_bottom: float
    pd_bottom: float
    ear_phone: float
    pd_phone: float
    # пороги
    ear_enabled: bool
    ear_threshold: float | None
    ear_freeze: float
    pitch_threshold: float
    # регресія
    regressor_x: GazeRegressor
    regressor_y: GazeRegressor
    # перевірка якості
    calib_error: float
    # службові
    created_at: str  # ISO 8601, локальний час
    schema_version: int = PROFILE_SCHEMA_VERSION

    def is_valid_for(self, monitor_id: str, camera_id: str, frame_w: int, frame_h: int) -> bool:
        """Недійсний, якщо змінилися монітор, камера або роздільна здатність кадру."""
        return (
            self.schema_version == PROFILE_SCHEMA_VERSION
            and self.monitor_id == monitor_id
            and self.camera_id == camera_id
            and (self.frame_w, self.frame_h) == (frame_w, frame_h)
        )

    def to_json(self) -> str:
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data["regressor_x"] = self.regressor_x.to_dict()
        data["regressor_y"] = self.regressor_y.to_dict()
        return json.dumps(data, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> CalibrationProfile:
        """Невідомі поля (профіль з іншої версії програми) ігноруються — такий
        профіль читається, а `is_valid_for` за schema_version вважає його недійсним."""
        data = json.loads(text)
        known = {f.name for f in fields(cls)}
        data = {k: v for k, v in data.items() if k in known}
        data["regressor_x"] = GazeRegressor.from_dict(data["regressor_x"])
        data["regressor_y"] = GazeRegressor.from_dict(data["regressor_y"])
        return cls(**data)
