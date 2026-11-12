"""Підсумки сесії для показу: метрики, графік по бінах, відрізки станів (без Qt)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from attention_tracker.config import MetricsConfig
from attention_tracker.focus.metrics import (
    FocusEventRow,
    PomodoroEventRow,
    Segment,
    SessionMetrics,
    compute_metrics,
    focus_timeline,
    state_segments,
    tracking_windows,
)
from attention_tracker.storage.storage import SessionInfo, Storage


@dataclass(frozen=True)
class SessionSummary:
    started_at: str  # ISO 8601
    duration: float  # уся сесія, с (разом з перервами й паузами)
    metrics: SessionMetrics
    shares: list[float | None]  # частка фокусу по бінах
    bin_s: float
    segments: list[Segment]


def summary_bin(duration: float, cfg: MetricsConfig) -> float:
    """Короткі сесії — дрібні біни, щоб графік не складався з 2–3 стовпчиків."""
    return cfg.live_bin if duration <= cfg.short_session_max else cfg.history_bin


def build_summary(
    started_at: str,
    focus_events: list[FocusEventRow],
    pomodoro_events: list[PomodoroEventRow],
    end: float,
    cfg: MetricsConfig,
    metrics: SessionMetrics | None = None,
) -> SessionSummary:
    """`metrics` можна передати готові (з БД), інакше вони рахуються з подій."""
    bin_s = summary_bin(end, cfg)
    return SessionSummary(
        started_at=started_at,
        duration=end,
        metrics=metrics if metrics is not None else compute_metrics(focus_events, pomodoro_events, end),
        shares=focus_timeline(focus_events, pomodoro_events, end, bin_s),
        bin_s=bin_s,
        segments=state_segments(focus_events, tracking_windows(pomodoro_events, end)),
    )


def session_duration(storage: Storage, info: SessionInfo) -> float:
    """Тривалість збереженої сесії: за годинником або за останньою подією, що більше."""
    wall = 0.0
    if info.ended_at:
        wall = (datetime.fromisoformat(info.ended_at) - datetime.fromisoformat(info.started_at)).total_seconds()
    return max(wall, storage.last_event_offset(info.id))


def summary_from_storage(storage: Storage, info: SessionInfo, cfg: MetricsConfig) -> SessionSummary:
    """Підсумки збереженої сесії (вкладка «Історія»)."""
    return build_summary(
        info.started_at,
        storage.focus_events(info.id),
        storage.pomodoro_events(info.id),
        session_duration(storage, info),
        cfg,
        info.metrics,
    )
