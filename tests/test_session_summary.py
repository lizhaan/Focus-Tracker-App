import pytest

from attention_tracker.config import MetricsConfig
from attention_tracker.focus.metrics import FocusEventRow, PomodoroEventRow
from attention_tracker.focus.pomodoro import PomodoroEvent as PE
from attention_tracker.models import FocusState as S, PomodoroPhase, Reason
from attention_tracker.session.summary import build_summary, summary_bin

CFG = MetricsConfig()
FOCUS = [
    FocusEventRow(10, S.FOCUSED, S.NOT_FOCUSED, Reason.GAZE_OUT),
    FocusEventRow(13, S.NOT_FOCUSED, S.LOST_FOCUS, Reason.PHONE_EYES),
    FocusEventRow(15, S.LOST_FOCUS, S.RECOVERING, Reason.NONE),
    FocusEventRow(17, S.RECOVERING, S.FOCUSED, Reason.NONE),
]
POMODORO = [PomodoroEventRow(0, PE.PHASE_START, PomodoroPhase.WORK), PomodoroEventRow(100, PE.STOP, PomodoroPhase.WORK)]


@pytest.mark.parametrize("duration, expected", [(60, 10.0), (600, 10.0), (601, 60.0), (3600, 60.0)])
def test_summary_bin_by_duration(duration, expected):
    assert summary_bin(duration, CFG) == expected


def test_build_summary():
    s = build_summary("2026-11-12T14:05:00", FOCUS, POMODORO, end=100, cfg=CFG)
    assert s.bin_s == 10.0 and len(s.shares) == 10
    assert s.metrics.n_lost_episodes == 1
    assert s.metrics.lost_by_reason == {"PHONE_EYES": 1}
    assert [seg.state for seg in s.segments] == [S.FOCUSED, S.NOT_FOCUSED, S.LOST_FOCUS, S.RECOVERING, S.FOCUSED]


@pytest.mark.parametrize(
    "seconds, text",
    [(0, "0 с"), (45, "45 с"), (60, "1 хв"), (750, "12 хв 30 с"), (3900, "1 год 05 хв")],
)
def test_format_duration(seconds, text):
    from attention_tracker.ui.session_summary import format_duration

    assert format_duration(seconds) == text


def test_format_started_at():
    from attention_tracker.ui.session_summary import format_started_at

    assert format_started_at("2026-11-12T14:05:00") == "12 листопада, 14:05"


def test_summary_window(qapp):
    from attention_tracker.ui.session_summary import SessionSummaryWindow

    s = build_summary("2026-11-12T14:05:00", FOCUS, POMODORO, end=100, cfg=CFG)
    w = SessionSummaryWindow(s, pomodoro=True)
    assert w.tile_values[0] == "96%"
    assert w.tile_values[1] == "1"
    assert len(w.tile_values) == 6  # з плиткою «помодоро»
    assert w.reason_lines == ["Телефон (погляд униз) — 1"]
    assert w.chart.bar_count == 10
    w.grab()
    w.close()


def test_summary_window_without_pomodoro_and_losses(qapp):
    from attention_tracker.ui.session_summary import SessionSummaryWindow

    s = build_summary("2026-11-12T14:05:00", [], POMODORO, end=100, cfg=CFG)
    w = SessionSummaryWindow(s, pomodoro=False)
    assert len(w.tile_values) == 5
    assert w.reason_lines == []
    w.close()
