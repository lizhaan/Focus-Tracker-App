def test_self_test_passes_from_source(qapp, tmp_path):
    from attention_tracker.self_test import run_self_test

    report = tmp_path / "report.txt"
    assert run_self_test(str(report)) == 0
    text = report.read_text(encoding="utf-8")
    assert text.splitlines()[-1] == "RESULT: OK"
    assert "FAIL" not in text


def test_self_test_reports_failure(qapp, tmp_path, monkeypatch):
    from attention_tracker import self_test

    def broken() -> str:
        raise RuntimeError("зламано")

    monkeypatch.setattr(self_test, "CHECKS", [("тест", broken)])
    report = tmp_path / "report.txt"
    assert self_test.run_self_test(str(report)) == 1
    assert "FAIL  тест" in report.read_text(encoding="utf-8")


def test_app_icon_exists():
    from attention_tracker.config import ASSETS_DIR
    from attention_tracker.main import APP_ICON

    assert (ASSETS_DIR / APP_ICON).is_file()
