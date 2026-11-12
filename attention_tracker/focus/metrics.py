"""Метрики сесії і графік концентрації з записаних подій.

Враховується тільки час, коли трекінг увімкнений: робочі фази Pomodoro без
ручних пауз (сесія без таймера записується як одна робоча фаза). На початку
кожного такого вікна стан — FOCUSED, бо машина станів тоді скидається.

    T_tracked     = T_focused + T_not_focused + T_lost   (T_lost = LOST_FOCUS + RECOVERING)
    focus_ratio   = (T_focused + T_not_focused) / T_tracked
    n_lost_episodes = кількість переходів NOT_FOCUSED → LOST_FOCUS
    mean_lost_duration = T_lost / n_lost_episodes   (0, якщо епізодів немає)
    longest_focus = найдовший неперервний відрізок без LOST_FOCUS / RECOVERING
    lost_by_reason = розподіл n_lost_episodes за причиною в момент переходу
    focus_share(bin) = час у {FOCUSED, NOT_FOCUSED} / відстежуваний час у біні
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from attention_tracker.focus.pomodoro import PomodoroEvent
from attention_tracker.models import FocusState, PomodoroPhase, Reason

_FOCUS_STATES = frozenset({FocusState.FOCUSED, FocusState.NOT_FOCUSED})


@dataclass(frozen=True)
class FocusEventRow:
    t: float  # секунди від початку сесії
    old: FocusState
    new: FocusState
    reason: Reason


@dataclass(frozen=True)
class PomodoroEventRow:
    t: float
    event: PomodoroEvent
    phase: PomodoroPhase


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    state: FocusState

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class SessionMetrics:
    t_focused: float = 0.0
    t_not_focused: float = 0.0
    t_lost: float = 0.0
    focus_ratio: float | None = None  # None, якщо трекінгу не було
    n_lost_episodes: int = 0
    mean_lost_duration: float = 0.0
    longest_focus: float = 0.0
    n_pomodoros: int = 0
    lost_by_reason: dict[str, int] = field(default_factory=dict)

    @property
    def t_tracked(self) -> float:
        return self.t_focused + self.t_not_focused + self.t_lost


def tracking_windows(pomodoro_events: list[PomodoroEventRow], end: float) -> list[tuple[float, float]]:
    """Відрізки часу, коли трекінг увімкнений: робоча фаза без ручної паузи."""
    windows: list[tuple[float, float]] = []
    in_work = paused = False
    since: float | None = None

    def update(t: float) -> None:
        nonlocal since
        active = in_work and not paused
        if active and since is None:
            since = t
        elif not active and since is not None:
            if t > since:
                windows.append((since, t))
            since = None

    for e in sorted(pomodoro_events, key=lambda e: e.t):
        if e.event is PomodoroEvent.PHASE_START:
            in_work, paused = e.phase is PomodoroPhase.WORK, False
        elif e.event in (PomodoroEvent.PHASE_END, PomodoroEvent.STOP):
            in_work = paused = False
        elif e.event is PomodoroEvent.MANUAL_PAUSE:
            paused = True
        elif e.event is PomodoroEvent.MANUAL_RESUME:
            paused = False
        update(e.t)
    in_work = False
    update(end)
    return windows


def state_segments(
    focus_events: list[FocusEventRow], windows: list[tuple[float, float]]
) -> list[Segment]:
    """Відрізки станів у межах вікон трекінгу; кожне вікно починається з FOCUSED."""
    events = sorted(focus_events, key=lambda e: e.t)
    segments: list[Segment] = []
    for start, end in windows:
        state, t = FocusState.FOCUSED, start
        for e in events:
            if start <= e.t < end:
                if e.t > t:
                    segments.append(Segment(t, e.t, state))
                state, t = e.new, e.t
        if end > t:
            segments.append(Segment(t, end, state))
    return segments


def compute_metrics(
    focus_events: list[FocusEventRow], pomodoro_events: list[PomodoroEventRow], end: float
) -> SessionMetrics:
    windows = tracking_windows(pomodoro_events, end)
    segments = state_segments(focus_events, windows)

    totals: Counter[FocusState] = Counter()
    for s in segments:
        totals[s.state] += s.duration
    t_focused = totals[FocusState.FOCUSED]
    t_not_focused = totals[FocusState.NOT_FOCUSED]
    t_lost = totals[FocusState.LOST_FOCUS] + totals[FocusState.RECOVERING]
    t_tracked = t_focused + t_not_focused + t_lost

    in_window = [e for e in focus_events if any(a <= e.t < b for a, b in windows)]
    lost = [e for e in in_window if e.old is FocusState.NOT_FOCUSED and e.new is FocusState.LOST_FOCUS]
    n_lost = len(lost)

    return SessionMetrics(
        t_focused=t_focused,
        t_not_focused=t_not_focused,
        t_lost=t_lost,
        focus_ratio=(t_focused + t_not_focused) / t_tracked if t_tracked > 0 else None,
        n_lost_episodes=n_lost,
        mean_lost_duration=t_lost / n_lost if n_lost else 0.0,
        longest_focus=_longest_focus(segments),
        n_pomodoros=sum(
            1 for e in pomodoro_events
            if e.event is PomodoroEvent.PHASE_END and e.phase is PomodoroPhase.WORK
        ),
        lost_by_reason=dict(Counter(e.reason.value for e in lost)),
    )


def _longest_focus(segments: list[Segment]) -> float:
    """Найдовший відрізок у FOCUSED / NOT_FOCUSED; розрив вікна теж перериває відрізок."""
    best = current = 0.0
    prev_end: float | None = None
    for s in segments:
        if s.state in _FOCUS_STATES:
            current = current + s.duration if prev_end == s.start else s.duration
            best = max(best, current)
        else:
            current = 0.0
        prev_end = s.end if s.state in _FOCUS_STATES else None
    return best


def focus_timeline(
    focus_events: list[FocusEventRow],
    pomodoro_events: list[PomodoroEventRow],
    end: float,
    bin_s: float,
) -> list[float | None]:
    """focus_share для бінів [0, bin), [bin, 2·bin), …; None — у біні трекінгу не було."""
    windows = tracking_windows(pomodoro_events, end)
    segments = state_segments(focus_events, windows)
    n_bins = int(-(-end // bin_s)) if end > 0 else 0
    shares: list[float | None] = []
    for i in range(n_bins):
        a, b = i * bin_s, min((i + 1) * bin_s, end)
        tracked = sum(_overlap(a, b, s.start, s.end) for s in segments)
        focused = sum(_overlap(a, b, s.start, s.end) for s in segments if s.state in _FOCUS_STATES)
        shares.append(focused / tracked if tracked > 0 else None)
    return shares


def _overlap(a: float, b: float, c: float, d: float) -> float:
    return max(0.0, min(b, d) - max(a, c))
