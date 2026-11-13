"""The operator's stop goes on the session's record, saying what was stopped and
from where (FIB-1397): without it a stopped run leaves only ``task_cancelled``,
which does not say who stopped it.

Uses the shared offscreen ``qapp`` fixture from tests/conftest.py.
"""

import os
import time

import pytest

pytest.importorskip("qtpy.QtWidgets")

from qtpy.QtCore import QTimer  # noqa: E402
from qtpy.QtWidgets import QMessageBox  # noqa: E402

from fibsem.ui.run import view  # noqa: E402
from tests.run_state import StubRunner


@pytest.fixture
def recorder(main_ui):
    from fibsem import utils
    from fibsem.workflows.event_recording import EventRecorder

    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    ui = main_ui
    previous = ui.session.recorder
    recorder = EventRecorder(microscope, default_actor="operator")
    ui.session.recorder = recorder
    yield recorder
    ui.session.recorder = previous
    recorder.close()
    microscope.disconnect()


def _stops(recorder):
    events = recorder.buffer.events_since(0)["events"]
    return [
        ({"stopped": e["payload"]["stops"], "via": e["payload"]["via"]}, e["actor"])
        for e in events
        if e["kind"] == "action" and e["payload"].get("stops")
    ]


@pytest.fixture
def runner(main_ui, monkeypatch):
    """A run going, for the runner to record its stop against."""
    ui = main_ui
    stub = StubRunner(ui.session, is_running=True)
    monkeypatch.setattr(ui.session, "runner", stub)
    yield stub
    stub.is_running = False  # so closing the window asks nothing


def test_the_stop_button_records_the_operator_s_stop(
    main_ui, recorder, runner, monkeypatch
):
    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes),
    )
    main_ui.run_view._on_stop_workflow_clicked()
    assert runner.stops == ["run"]
    assert _stops(recorder) == [({"stopped": "workflow", "via": "Stop"}, "operator")]


def test_a_stop_not_confirmed_records_nothing(main_ui, recorder, runner, monkeypatch):
    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.No),
    )
    main_ui.run_view._on_stop_workflow_clicked()
    assert runner.stops == [], "stopped without a yes"
    assert _stops(recorder) == []


def test_closing_the_window_mid_run_records_the_stop(
    main_ui, recorder, runner, monkeypatch
):
    monkeypatch.setattr(
        QMessageBox, "exec", lambda self: QMessageBox.StandardButton.Yes
    )
    monkeypatch.setattr(view, "CLOSE_WAIT_S", 0.2)  # the stub never ends
    assert main_ui.run_view.confirm_close() is True
    assert _stops(recorder) == [
        ({"stopped": "workflow", "via": "closing the window"}, "operator")
    ]


def test_closing_waits_for_the_stopped_run_to_end(main_ui, runner, monkeypatch):
    """The run's thread is a daemon: a close that does not wait can end it
    before it has turned the beams off (engineering review R2)."""
    monkeypatch.setattr(
        QMessageBox, "exec", lambda self: QMessageBox.StandardButton.Yes
    )
    monkeypatch.setattr(view, "CLOSE_WAIT_S", 30.0)
    QTimer.singleShot(300, lambda: setattr(runner, "is_running", False))
    started = time.monotonic()
    assert main_ui.run_view.confirm_close() is True
    assert runner.is_running is False, "closed before the run ended"
    assert 0.25 < time.monotonic() - started < 10


def test_closing_goes_ahead_when_the_run_does_not_end(
    main_ui, runner, monkeypatch, caplog
):
    monkeypatch.setattr(
        QMessageBox, "exec", lambda self: QMessageBox.StandardButton.Yes
    )
    monkeypatch.setattr(view, "CLOSE_WAIT_S", 0.3)
    assert main_ui.run_view.confirm_close() is True
    assert runner.is_running is True
    assert "did not stop within" in caplog.text
