import pytest

from attention_tracker.platform.monitors import make_monitor_id


@pytest.mark.parametrize(
    "args, expected",
    [
        (("DEL", "U2419", "ABC123", r"\\.\DISPLAY1", 1920, 1080), "DEL|U2419|ABC123"),
        (("", "Generic PnP", "", r"\\.\DISPLAY2", 1920, 1080), "|Generic PnP|"),
        (("", "", "", r"\\.\DISPLAY1", 1280, 800), r"\\.\DISPLAY1|1280x800"),
    ],
)
def test_make_monitor_id(args, expected):
    assert make_monitor_id(*args) == expected


def test_list_monitors_has_primary(qapp):
    from attention_tracker.platform.monitors import find_screen, list_monitors

    monitors = list_monitors()
    assert monitors
    assert sum(m.is_primary for m in monitors) == 1
    first = monitors[0]
    assert first.geometry[2] > 0 and first.geometry[3] > 0
    assert find_screen(first.monitor_id) is not None
    assert find_screen("no-such-monitor") is None


def test_default_db_path(qapp):
    from attention_tracker.platform.paths import DB_FILENAME, default_db_path

    assert default_db_path().name == DB_FILENAME
