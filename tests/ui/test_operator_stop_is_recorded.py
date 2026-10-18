"""The operator's stop goes on the session's record, saying what was stopped and
from where (FIB-1397): without it a stopped run leaves only ``task_cancelled``,
which does not say who stopped it.

Uses the shared offscreen ``qapp`` fixture from tests/conftest.py.
"""

import os

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QMessageBox  # noqa: E402

from tests.run_state import StubRunner


@pytest.fixture
def main_ui(qapp):
    from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module

    window = module.AutoLamellaSingleWindowUI()
    yield window
    original_quit = qapp.quit
    qapp.quit = lambda: None
    try:
        window.close()
    finally:
        qapp.quit = original_quit


@pytest.fixture
def recorder(main_ui):
    from fibsem import utils
    from fibsem.workflows.event_recording import EventRecorder

    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    ui = main_ui.autolamella_ui
    previous = ui.session.recorder
    recorder = EventRecorder(microscope, default_actor="operator")
    ui.session.recorder = recorder
    yield recorder
    ui.session.recorder = previous
    recorder.close()
    microscope.disconnect()


def _stops(recorder):
    events = recorder.buffer.events_since(0)["events"]
    return [(e["payload"], e["actor"]) for e in events if e["kind"] == "workflow_stop"]


def test_the_stop_button_records_the_operator_s_stop(main_ui, recorder, monkeypatch):
    stopped = []
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes)
    )
    monkeypatch.setattr(
        main_ui.autolamella_ui, "stop_task_workflow", lambda: stopped.append(True)
    )
    main_ui._on_stop_workflow_clicked()
    assert stopped == [True]
    assert _stops(recorder) == [({"stopped": "workflow", "by": "Stop"}, "operator")]


def test_a_stop_not_confirmed_records_nothing(main_ui, recorder, monkeypatch):
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No)
    )
    monkeypatch.setattr(
        main_ui.autolamella_ui,
        "stop_task_workflow",
        lambda: pytest.fail("stopped without a yes"),
    )
    main_ui._on_stop_workflow_clicked()
    assert _stops(recorder) == []


def test_closing_the_window_mid_run_records_the_stop(main_ui, recorder, monkeypatch):
    ui = main_ui.autolamella_ui
    monkeypatch.setattr(ui.session, "runner", StubRunner(is_running=True))
    monkeypatch.setattr(ui, "stop_task_workflow", lambda: None)
    monkeypatch.setattr(QMessageBox, "exec_", lambda self: QMessageBox.Yes)
    assert main_ui._confirm_close_during_workflow() is True
    assert _stops(recorder) == [
        ({"stopped": "workflow", "by": "closing the window"}, "operator")
    ]
