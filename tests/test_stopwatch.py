import pytest

from attention_tracker.focus.stopwatch import Stopwatch


def test_counts_time_since_start():
    sw = Stopwatch()
    assert not sw.started and sw.elapsed(10.0) == 0.0
    sw.start(10.0)
    assert sw.started and not sw.paused
    assert sw.elapsed(25.0) == pytest.approx(15.0)


def test_paused_time_not_counted():
    sw = Stopwatch()
    sw.start(0.0)
    sw.pause(10.0)
    assert sw.paused
    assert sw.elapsed(100.0) == pytest.approx(10.0)
    sw.resume(100.0)
    assert sw.elapsed(105.0) == pytest.approx(15.0)


def test_repeated_pause_and_resume_are_ignored():
    sw = Stopwatch()
    sw.start(0.0)
    sw.pause(10.0)
    sw.pause(20.0)
    sw.resume(30.0)
    sw.resume(40.0)
    assert sw.elapsed(50.0) == pytest.approx(30.0)


def test_stop_resets():
    sw = Stopwatch()
    sw.start(0.0)
    sw.stop()
    assert not sw.started and not sw.paused
    assert sw.elapsed(10.0) == 0.0
    sw.resume(10.0)  # без старту нічого не відбувається
    assert not sw.started
