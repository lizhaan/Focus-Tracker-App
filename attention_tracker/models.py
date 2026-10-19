"""Моделі даних та енуми"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FocusState(Enum):

    FOCUSED = "FOCUSED"
    NOT_FOCUSED = "NOT_FOCUSED"
    LOST_FOCUS = "LOST_FOCUS"
    RECOVERING = "RECOVERING"


class Reason(Enum):

    NONE = "NONE"
    GAZE_OUT = "GAZE_OUT"
    PHONE_EYES = "PHONE_EYES"
    PHONE_HEAD = "PHONE_HEAD"
    FACE_ABSENT = "FACE_ABSENT"
    HEAD_AWAY = "HEAD_AWAY"


class PomodoroPhase(Enum):

    IDLE = "IDLE"
    WORK = "WORK"
    SHORT_BREAK = "SHORT_BREAK"
    LONG_BREAK = "LONG_BREAK"


@dataclass(frozen=True)
class FrameFeatures:

    ts: float  # time.monotonic()
    valid: bool  # обличчя знайдено і solvePnP ok
    ear: float | None
    yaw: float | None
    pitch: float | None
    roll: float | None
    gaze_h: float | None
    gaze_v: float | None
    iris_distance_px: float | None

    @classmethod
    def invalid(cls, ts: float) -> FrameFeatures:
        return cls(
            ts=ts,
            valid=False,
            ear=None,
            yaw=None,
            pitch=None,
            roll=None,
            gaze_h=None,
            gaze_v=None,
            iris_distance_px=None,
        )


@dataclass(frozen=True)
class DistractionResult:
    """Вихід детектора."""

    ts: float
    distracted: bool
    reason: Reason
    screen_point: tuple[float, float] | None  # (sx, sy) на робочому моніторі


@dataclass(frozen=True)
class StateChange:

    ts: float
    old: FocusState
    new: FocusState
    reason: Reason


@dataclass(frozen=True)
class FrameResult:

    features: FrameFeatures
    distraction: DistractionResult | None
    state: FocusState | None
    recovering_progress: float | None  # 0..1 у RECOVERING
