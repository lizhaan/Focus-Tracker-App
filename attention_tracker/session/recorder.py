"""Запис сесії в Storage: переходи станів, події Pomodoro, метрики при завершенні.

Викликається тільки з головного потоку (SQLite). Час — monotonic секунди;
у БД пишеться зсув від початку сесії.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from attention_tracker.focus.metrics import (
    FocusEventRow,
    PomodoroEventRow,
    SessionMetrics,
    compute_metrics,
)
from attention_tracker.focus.pomodoro import PomodoroEvent
from attention_tracker.models import PomodoroPhase, StateChange
from attention_tracker.storage.storage import Storage


class SessionRecorder:
    def __init__(
        self,
        storage: Storage,
        user_id: int,
        monitor_id: str,
        profile_id: int | None,
        settings: dict,
        started_at: str,
        t0: float,
    ) -> None:
        self._storage = storage
        self._t0 = t0
        self._finished = False
        self._last_phase = PomodoroPhase.IDLE
        self.session_id = storage.create_session(user_id, started_at, settings, monitor_id, profile_id)

    @property
    def finished(self) -> bool:
        return self._finished

    def offset(self, now: float) -> float:
        return max(0.0, now - self._t0)

    def on_state_change(self, change: StateChange) -> None:
        if self._finished:
            return
        row = FocusEventRow(self.offset(change.ts), change.old, change.new, change.reason)
        self._storage.add_focus_event(self.session_id, row)

    def on_pomodoro_event(self, event: PomodoroEvent, phase: PomodoroPhase, now: float) -> None:
        if self._finished:
            return
        self._storage.add_pomodoro_event(self.session_id, PomodoroEventRow(self.offset(now), event, phase))
        if event is PomodoroEvent.PHASE_START:
            self._last_phase = phase
        elif event is PomodoroEvent.STOP:
            self._last_phase = PomodoroPhase.IDLE

    def finish(self, now: float, ended_at: str) -> SessionMetrics:
        """Закрити сесію: подія stop (якщо ще не було), метрики в БД."""
        if self._finished:
            raise RuntimeError("сесію вже завершено")
        if self._last_phase is not PomodoroPhase.IDLE:
            self.on_pomodoro_event(PomodoroEvent.STOP, self._last_phase, now)
        metrics = _metrics_for(self._storage, self.session_id, self.offset(now))
        self._storage.finish_session(self.session_id, ended_at, metrics)
        self._finished = True
        return metrics


def _metrics_for(storage: Storage, session_id: int, end: float) -> SessionMetrics:
    return compute_metrics(storage.focus_events(session_id), storage.pomodoro_events(session_id), end)


def recover_unfinished(storage: Storage) -> int:
    """Закрити сесії, що обірвались (ended_at IS NULL), за часом останньої події.

    Повертає кількість закритих сесій.
    """
    sessions = storage.unfinished_sessions()
    for session_id, started_at in sessions:
        end = storage.last_event_offset(session_id)
        ended_at = (datetime.fromisoformat(started_at) + timedelta(seconds=end)).isoformat(timespec="seconds")
        storage.finish_session(session_id, ended_at, _metrics_for(storage, session_id, end))
    return len(sessions)
