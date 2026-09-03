"""A user Stop is logged as a cancellation, not as an error (FIB-859).

``_run_tasks_worker``'s catch-all treated the cancellation unwind the same as a
real failure, so every Stop press put ``ERROR — Error during running tasks:
Workflow aborted by user.`` in the log — a false positive for anyone scanning a
session log for problems, once per Stop. Both cancellation types
(``InterruptedError`` from the abort checks, ``OperationCancelledError`` from
milling/autofocus) log at INFO now; genuine failures still log at ERROR.
"""

import logging
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.structures import Experiment
from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI
from fibsem.cancellation import OperationCancelledError


@pytest.fixture
def ui(qapp, tmp_path, monkeypatch):
    """A real AutoLamellaUI, connected (Demo), ready to run the worker inline."""
    widget = AutoLamellaUI(parent_ui=None)
    widget.system_widget.connect_to_microscope()
    widget.experiment = Experiment(path=tmp_path, name="test-exp")
    # The hook set is irrelevant here and pulls in user preferences.
    monkeypatch.setattr(widget, "setup_hooks", lambda: None)
    yield widget
    if widget.microscope is not None:
        widget.microscope.disconnect()
    widget.close()


def _run_with_manager_raising(ui, monkeypatch, exc):
    from fibsem.applications.autolamella.ui import AutoLamellaUI as module

    class _RaisingManager:
        is_stopped = True

        def __init__(self, **kwargs):
            pass

        def stop(self):
            pass

        def run(self, **kwargs):
            raise exc

        def build_run_summary_dataframe(self):
            return None

    monkeypatch.setattr(module, "TaskManager", _RaisingManager)
    ui._run_tasks_worker(["Rough Milling"], ["01-test"])


@pytest.mark.parametrize(
    "exc",
    [
        InterruptedError("Workflow aborted by user."),
        OperationCancelledError("Milling cancelled"),
    ],
    ids=["interrupted", "operation-cancelled"],
)
def test_a_user_stop_logs_no_error(ui, monkeypatch, caplog, exc):
    with caplog.at_level(logging.INFO):
        _run_with_manager_raising(ui, monkeypatch, exc)

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors == [], [r.message for r in errors]
    assert any("cancelled" in r.message.lower() for r in caplog.records), (
        "the cancellation must still leave a trace, just not an ERROR"
    )


def test_a_genuine_failure_still_logs_an_error(ui, monkeypatch, caplog):
    with caplog.at_level(logging.INFO):
        _run_with_manager_raising(ui, monkeypatch, RuntimeError("stage fell over"))

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("stage fell over" in r.message for r in errors), (
        "downgrading cancellation must not swallow real failures"
    )


def test_the_closing_line_gets_out_after_a_stop(ui):
    """After Stop the abort predicate is true, and a status point raises on it:
    that is what kept "Workflow cancelled by user." off the label. The manager's closing
    line asks for no abort check, and reaches the signal."""
    from fibsem.applications.autolamella.workflows.ui import update_status_ui

    seen = []
    ui.workflow_status_signal.connect(lambda event: seen.append(event.workflow_info))
    ui._workflow_stop_event.set()
    with pytest.raises(InterruptedError):
        update_status_ui(ui, "", workflow_info="Workflow cancelled by user.")
    update_status_ui(ui, "", workflow_info="Workflow cancelled by user.", check_abort=False)
    assert seen == ["Workflow cancelled by user."]
