"""Pomodoro-таймер без Qt: фази, ручна й автоматична пауза.

IDLE → WORK → SHORT_BREAK / LONG_BREAK (кожна N-та) → WORK → …; Стоп → IDLE.
Час на паузі не рахується. Автопауза діє тільки в робочій фазі; автоматичне
продовження не знімає ручну паузу. Трекінг на ручній паузі вимкнений, тож
після «Продовжити» стан знову «у фокусі» і автопауза теж знімається.
Час передається явно (`now`, секунди monotonic); події — через callback.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from attention_tracker.config import PomodoroConfig
from attention_tracker.models import PomodoroPhase


class PomodoroEvent(Enum):
    PHASE_START = "phase_start"
    PHASE_END = "phase_end"
    AUTO_PAUSE = "auto_pause"
    AUTO_RESUME = "auto_resume"
    MANUAL_PAUSE = "manual_pause"
    MANUAL_RESUME = "manual_resume"
    STOP = "stop"


@dataclass(frozen=True)
class PomodoroEventInfo:
    ts: float
    event: PomodoroEvent
    phase: PomodoroPhase


EventCallback = Callable[[PomodoroEventInfo], None]


class PomodoroTimer:
    def __init__(self, cfg: PomodoroConfig, on_event: EventCallback | None = None) -> None:
        self._cfg = cfg
        self._on_event = on_event
        self._phase = PomodoroPhase.IDLE
        self._completed = 0  # завершені (або пропущені) робочі фази
        self._elapsed = 0.0  # час фази без пауз, до останнього продовження
        self._running_since: float | None = None  # None — годинник стоїть
        self._paused_auto = False
        self._paused_manual = False

    def set_config(self, cfg: PomodoroConfig) -> None:
        """Нові тривалості діють одразу, зокрема для поточної фази."""
        self._cfg = cfg

    # --- стан ------------------------------------------------------------------

    @property
    def phase(self) -> PomodoroPhase:
        return self._phase

    @property
    def completed_pomodoros(self) -> int:
        return self._completed

    @property
    def paused_auto(self) -> bool:
        return self._paused_auto

    @property
    def paused_manual(self) -> bool:
        return self._paused_manual

    @property
    def paused(self) -> bool:
        return self._paused_auto or self._paused_manual

    def duration(self, phase: PomodoroPhase | None = None) -> float:
        phase = phase or self._phase
        return {
            PomodoroPhase.WORK: self._cfg.work,
            PomodoroPhase.SHORT_BREAK: self._cfg.short_break,
            PomodoroPhase.LONG_BREAK: self._cfg.long_break,
        }.get(phase, 0.0)

    def elapsed(self, now: float) -> float:
        running = now - self._running_since if self._running_since is not None else 0.0
        return self._elapsed + running

    def remaining(self, now: float) -> float:
        if self._phase is PomodoroPhase.IDLE:
            return 0.0
        return max(0.0, self.duration() - self.elapsed(now))

    # --- керування -------------------------------------------------------------

    def start(self, now: float) -> None:
        if self._phase is not PomodoroPhase.IDLE:
            return
        self._completed = 0
        self._begin_phase(PomodoroPhase.WORK, now)

    def tick(self, now: float) -> None:
        """Перехід до наступної фази, коли час поточної вийшов."""
        if self._phase is not PomodoroPhase.IDLE and not self.paused and self.remaining(now) <= 0.0:
            self._advance(now)

    def skip_phase(self, now: float) -> None:
        if self._phase is not PomodoroPhase.IDLE:
            self._advance(now)

    def stop(self, now: float) -> None:
        if self._phase is PomodoroPhase.IDLE:
            return
        phase = self._phase
        self._phase = PomodoroPhase.IDLE
        self._reset_clock()
        self._emit(now, PomodoroEvent.STOP, phase)

    def pause_auto(self, now: float) -> None:
        if self._phase is not PomodoroPhase.WORK or self._paused_auto:
            return
        self._paused_auto = True
        self._update_clock(now)
        self._emit(now, PomodoroEvent.AUTO_PAUSE)

    def resume_auto(self, now: float) -> None:
        if not self._paused_auto:
            return
        self._paused_auto = False
        self._update_clock(now)
        self._emit(now, PomodoroEvent.AUTO_RESUME)

    def pause_manual(self, now: float) -> None:
        if self._phase is PomodoroPhase.IDLE or self._paused_manual:
            return
        self._paused_manual = True
        self._update_clock(now)
        self._emit(now, PomodoroEvent.MANUAL_PAUSE)

    def resume_manual(self, now: float) -> None:
        if not self._paused_manual:
            return
        self._paused_manual = False
        self._paused_auto = False  # трекінг на ручній паузі вимкнений — стан знову «у фокусі»
        self._update_clock(now)
        self._emit(now, PomodoroEvent.MANUAL_RESUME)

    # --- внутрішнє ---------------------------------------------------------

    def _advance(self, now: float) -> None:
        ended = self._phase
        self._emit(now, PomodoroEvent.PHASE_END, ended)
        if ended is PomodoroPhase.WORK:
            self._completed += 1
            long = self._completed % self._cfg.pomodoros_before_long_break == 0
            self._begin_phase(PomodoroPhase.LONG_BREAK if long else PomodoroPhase.SHORT_BREAK, now)
        else:
            self._begin_phase(PomodoroPhase.WORK, now)

    def _begin_phase(self, phase: PomodoroPhase, now: float) -> None:
        self._phase = phase
        self._reset_clock()
        self._running_since = now
        self._emit(now, PomodoroEvent.PHASE_START)

    def _reset_clock(self) -> None:
        self._elapsed = 0.0
        self._running_since = None
        self._paused_auto = False
        self._paused_manual = False

    def _update_clock(self, now: float) -> None:
        """Зупинити або запустити годинник відповідно до пауз."""
        if self.paused and self._running_since is not None:
            self._elapsed += now - self._running_since
            self._running_since = None
        elif not self.paused and self._running_since is None:
            self._running_since = now

    def _emit(self, now: float, event: PomodoroEvent, phase: PomodoroPhase | None = None) -> None:
        if self._on_event is not None:
            self._on_event(PomodoroEventInfo(now, event, phase or self._phase))
