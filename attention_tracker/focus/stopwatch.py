"""Секундомір сесії без Pomodoro: час сесії без урахування ручних пауз.

Час передається явно (`now`, секунди monotonic).
"""

from __future__ import annotations


class Stopwatch:
    def __init__(self) -> None:
        self._elapsed = 0.0
        self._running_since: float | None = None
        self._started = False

    @property
    def started(self) -> bool:
        return self._started

    @property
    def paused(self) -> bool:
        return self._started and self._running_since is None

    def start(self, now: float) -> None:
        self._elapsed = 0.0
        self._running_since = now
        self._started = True

    def stop(self) -> None:
        self._elapsed = 0.0
        self._running_since = None
        self._started = False

    def pause(self, now: float) -> None:
        if self._running_since is not None:
            self._elapsed += now - self._running_since
            self._running_since = None

    def resume(self, now: float) -> None:
        if self._started and self._running_since is None:
            self._running_since = now

    def elapsed(self, now: float) -> float:
        running = now - self._running_since if self._running_since is not None else 0.0
        return self._elapsed + running
