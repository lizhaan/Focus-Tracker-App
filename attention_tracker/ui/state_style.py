"""Як показувати стани фокусу: колір і підпис індикатора, текст плашки.

Плашка не показує причину відволікання — вона пишеться тільки в історію.
"""

from __future__ import annotations

from dataclasses import dataclass

from attention_tracker.models import FocusState


@dataclass(frozen=True)
class StateStyle:
    color: str  # колір індикатора
    label: str  # підпис у головному вікні
    overlay_text: str | None  # текст плашки; None — плашка прихована
    overlay_progress: bool  # чи показувати прогрес повернення


IDLE_STYLE = StateStyle("#9e9e9e", "Сесію не запущено", None, False)

STATE_STYLES: dict[FocusState, StateStyle] = {
    FocusState.FOCUSED: StateStyle("#2e7d32", "У фокусі", None, False),
    FocusState.NOT_FOCUSED: StateStyle("#f9a825", "Відволіклись", None, False),
    FocusState.LOST_FOCUS: StateStyle("#c62828", "Фокус втрачено", "Фокус втрачено", False),
    FocusState.RECOVERING: StateStyle(
        "#ef6c00", "Повертаємось у фокус", "Повертаємось у фокус…", True
    ),
}


def style_for(state: FocusState | None) -> StateStyle:
    return IDLE_STYLE if state is None else STATE_STYLES[state]
