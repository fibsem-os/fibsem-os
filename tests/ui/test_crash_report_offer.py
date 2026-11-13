"""The next launch asks once about the errors the last run recorded."""

import json

import pytest

pytest.importorskip("qtpy")

from qtpy.QtWidgets import QMessageBox  # noqa: E402

from fibsem import diagnostics  # noqa: E402
from fibsem.ui.widgets import bug_report_widget  # noqa: E402


@pytest.fixture
def crash_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics.cfg, "LOG_PATH", str(tmp_path))
    folder = diagnostics.crash_folder()
    folder.mkdir(parents=True)
    return folder


def _earlier_record(folder, name="2026-01-01-00-00-00-1-01", **changes):
    data = {
        "app": "AutoLamella",
        "version": "0.6.0",
        "at": "2026-01-01T00:00:00+00:00",
        "pid": 0,
        "thread": "MainThread",
        "error": "ValueError",
        "traceback": "Traceback (most recent call last):\nValueError: no stage",
        "seen": False,
    }
    data.update(changes)
    (folder / f"{name}.json").write_text(json.dumps(data))


def _answer(monkeypatch, label):
    """Click the button called *label* instead of showing the question."""
    shown = []

    def exec_(box):
        shown.append(box)
        for button in box.buttons():
            if button.text() == label:
                box.clickedButton = lambda button=button: button
        return 0

    monkeypatch.setattr(QMessageBox, "exec", exec_)
    return shown


def test_nothing_recorded_asks_nothing(qapp, crash_folder, monkeypatch):
    shown = _answer(monkeypatch, "Not Now")

    assert bug_report_widget.offer_crash_reports() is False
    assert shown == []


def test_reporting_opens_the_dialog_on_the_latest_error(
    qapp, crash_folder, monkeypatch
):
    _earlier_record(crash_folder)
    _earlier_record(
        crash_folder,
        name="2026-01-02-00-00-00-1-01",
        traceback="Traceback (most recent call last):\nKeyError: 'lamella'",
    )
    shown = _answer(monkeypatch, "Report an Issue...")
    opened = {}
    monkeypatch.setattr(
        bug_report_widget, "open_bug_report_dialog", lambda **kw: opened.update(kw)
    )

    assert bug_report_widget.offer_crash_reports() is True

    assert "2 in all" in shown[0].text()
    assert "KeyError: 'lamella'" in shown[0].informativeText()
    assert "KeyError: 'lamella'" in opened["traceback_text"]
    assert diagnostics.unseen_from_earlier_runs() == []


def test_not_now_asks_once_and_keeps_the_record(qapp, crash_folder, monkeypatch):
    _earlier_record(crash_folder)
    shown = _answer(monkeypatch, "Not Now")
    monkeypatch.setattr(
        bug_report_widget,
        "open_bug_report_dialog",
        lambda **kw: pytest.fail("the dialog opened"),
    )

    assert bug_report_widget.offer_crash_reports() is True
    assert bug_report_widget.offer_crash_reports() is False

    assert len(shown) == 1
    assert [record.seen for record in diagnostics.records()] == [True]
