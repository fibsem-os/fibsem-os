"""Report an Issue builds its bundle off the GUI thread."""

import threading
import time
import zipfile

import pytest

pytest.importorskip("qtpy")

from qtpy.QtWidgets import QMessageBox  # noqa: E402

from fibsem.tools import bug_report  # noqa: E402
from fibsem.ui.widgets import bug_report_widget  # noqa: E402
from fibsem.ui.widgets.bug_report_widget import BugReportDialog  # noqa: E402


def _wait_for(qapp, done, seconds=10.0):
    deadline = time.monotonic() + seconds
    while not done() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert done()


@pytest.fixture
def experiment(tmp_path):
    folder = tmp_path / "exp"
    folder.mkdir()
    (folder / "logfile.log").write_text("it stopped")
    (folder / "events.jsonl").write_text('{"kind": "stop"}\n')
    return str(folder)


def test_create_bundle_runs_on_a_worker_and_shows_where_it_is(
    qapp, experiment, monkeypatch
):
    shown = []
    built_on = []
    build = bug_report.build_bug_report_bundle

    def recording_build(*args, **kwargs):
        built_on.append(threading.current_thread())
        return build(*args, **kwargs)

    monkeypatch.setattr(bug_report, "build_bug_report_bundle", recording_build)
    monkeypatch.setattr(
        bug_report_widget.BundleCreatedDialog,
        "exec",
        lambda self: shown.append(self.zip_path),
    )
    dialog = BugReportDialog(experiment_path=experiment)
    dialog.lineEdit_title.setText("it stopped")
    dialog.checkbox_settings.setChecked(False)

    dialog.pushButton_bundle.click()

    assert not dialog.pushButton_bundle.isEnabled()
    _wait_for(qapp, lambda: shown)
    assert built_on[0] is not threading.main_thread()
    with zipfile.ZipFile(shown[0]) as zf:
        names = zf.namelist()
    assert "exp/events.jsonl" in names
    assert not any(name.startswith("settings/") for name in names)


def test_a_bundle_that_fails_says_so_and_can_be_tried_again(
    qapp, experiment, monkeypatch
):
    said = []

    def failing_build(*args, **kwargs):
        raise OSError("the disk is full")

    monkeypatch.setattr(bug_report, "build_bug_report_bundle", failing_build)
    monkeypatch.setattr(
        QMessageBox, "critical", lambda parent, title, text: said.append(text)
    )
    dialog = BugReportDialog(experiment_path=experiment)
    dialog.lineEdit_title.setText("it stopped")

    dialog.pushButton_bundle.click()

    _wait_for(qapp, lambda: said)
    assert "the disk is full" in said[0]
    assert dialog.pushButton_bundle.isEnabled()
    assert dialog.pushButton_bundle.text() == "Create Bundle"
