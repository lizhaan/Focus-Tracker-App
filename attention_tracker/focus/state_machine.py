"""Машина станів фокусу: FOCUSED, NOT_FOCUSED, LOST_FOCUS, RECOVERING.

Вхід — `distracted` і час `now` (секунди, time.monotonic()); час не рахується
кадрами. Параметри: `n` — скільки відволікання до втрати фокусу, `y` — скільки
стабільного повернення до відновлення, `τ` — допуск на шум.
"""

from __future__ import annotations

from attention_tracker.config import FocusConfig
from attention_tracker.models import FocusState, Reason, StateChange


class FocusStateMachine:
    """Стан змінюється тільки в `update()`; переходи повертаються як `StateChange`.

    Неоднозначності специфікації вирішено так:
    - у RECOVERING, якщо в одному кадрі минуло `y` і водночас відволікання
      триває ≥ τ, перемагає LOST_FOCUS;
    - причина в `StateChange` — з поточного кадру для переходів у NOT_FOCUSED
      і LOST_FOCUS, і `Reason.NONE` для переходів у FOCUSED і RECOVERING.
    """

    def __init__(self, cfg: FocusConfig) -> None:
        self._cfg = cfg
        self._n = cfg.n
        self._y = cfg.y
        self._tau = cfg.tau
        self._state = FocusState.FOCUSED
        self._t_start: float | None = None  # початок відволікання (NOT_FOCUSED)
        self._t_return: float | None = None  # початок повернення (RECOVERING)
        self._clear_since: float | None = None  # ¬distracted безперервно з (NOT_FOCUSED)
        self._distracted_since: float | None = None  # distracted безперервно з (RECOVERING)

    @property
    def state(self) -> FocusState:
        return self._state

    @property
    def n(self) -> float:
        return self._n

    @property
    def y(self) -> float:
        return self._y

    def reset(self, now: float) -> None:
        """Старт робочої фази Pomodoro → FOCUSED, усі таймери скинуто."""
        del now  # стан після скидання від часу не залежить
        self._state = FocusState.FOCUSED
        self._t_start = None
        self._t_return = None
        self._clear_since = None
        self._distracted_since = None

    def set_timings(self, n: float, y: float) -> None:
        """Нові n, y діють з наступного кадру; накопичений час не скидається."""
        cfg = self._cfg
        if not cfg.n_min <= n <= cfg.n_max:
            raise ValueError(f"n має бути в [{cfg.n_min}, {cfg.n_max}] с, отримано {n}")
        if not cfg.y_min <= y <= cfg.y_max:
            raise ValueError(f"y має бути в [{cfg.y_min}, {cfg.y_max}] с, отримано {y}")
        self._n = n
        self._y = y

    def recovering_progress(self, now: float) -> float | None:
        """Частка `y`, яку вже пройдено в RECOVERING (0..1); інакше `None`."""
        if self._state is not FocusState.RECOVERING or self._t_return is None:
            return None
        return min(1.0, max(0.0, (now - self._t_return) / self._y))

    def update(self, distracted: bool, reason: Reason, now: float) -> StateChange | None:
        state = self._state

        if state is FocusState.FOCUSED:
            if distracted:
                self._t_start = now
                self._clear_since = None
                return self._go(FocusState.NOT_FOCUSED, reason, now)
            return None

        if state is FocusState.NOT_FOCUSED:
            if distracted:
                self._clear_since = None  # коротке повернення < τ таймер n не скидає
                if now - self._t_start >= self._n:
                    return self._go(FocusState.LOST_FOCUS, reason, now)
                return None
            if self._clear_since is None:
                self._clear_since = now
            if now - self._clear_since >= self._tau:
                return self._go(FocusState.FOCUSED, Reason.NONE, now)
            return None

        if state is FocusState.LOST_FOCUS:
            if not distracted:
                self._t_return = now
                self._distracted_since = None
                return self._go(FocusState.RECOVERING, Reason.NONE, now)
            return None

        # RECOVERING
        if distracted:
            if self._distracted_since is None:
                self._distracted_since = now
            if now - self._distracted_since >= self._tau:
                return self._go(FocusState.LOST_FOCUS, reason, now)
        else:
            self._distracted_since = None  # одиничні кадри < τ таймер y не скидають
        if now - self._t_return >= self._y:
            return self._go(FocusState.FOCUSED, Reason.NONE, now)
        return None

    def _go(self, new: FocusState, reason: Reason, now: float) -> StateChange:
        change = StateChange(ts=now, old=self._state, new=new, reason=reason)
        self._state = new
        if new is FocusState.FOCUSED:
            self._t_start = self._t_return = None
            self._clear_since = self._distracted_since = None
        return change
