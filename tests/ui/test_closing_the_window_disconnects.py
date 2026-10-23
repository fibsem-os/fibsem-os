"""Closing either main window lets the microscope go, rather than leaving the
client open until the process ends -- and a client that will not close does not
stop the window closing.

The real windows on Demo.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from tests.protocols import add_planned, planned

pytest.importorskip("PyQt5")


def _refuse_to_disconnect():
    raise RuntimeError("client already gone")


def test_closing_autolamella_disconnects_the_microscope(no_quit):
    from fibsem.applications.autolamella.ui import window as module

    window = module.AutoLamellaWindow()
    window.system_widget.connect_to_microscope()
    microscope = window.microscope
    assert microscope._connection.connected

    window.close()

    assert not microscope._connection.connected


def test_closing_fibsem_disconnects_the_microscope(qapp):
    from fibsem.ui.FibsemUI import FibsemUI

    window = FibsemUI()
    window.system_widget.connect_to_microscope()
    microscope = window.microscope
    assert microscope._connection.connected

    window.close()

    assert not microscope._connection.connected
    window.deleteLater()
    qapp.processEvents()


def test_a_disconnect_that_fails_does_not_stop_the_close(qapp):
    from fibsem.ui.FibsemUI import FibsemUI

    window = FibsemUI()
    window.system_widget.connect_to_microscope()
    window.show()
    microscope = window.microscope
    microscope.disconnect = _refuse_to_disconnect

    window.close()

    assert not window.isVisible()
    assert microscope.try_disconnect() is False
    window.deleteLater()
    qapp.processEvents()


@pytest.fixture
def running(no_quit, qapp, monkeypatch, tmp_path):
    """AutoLamella on Demo with a workflow running: a run started through the
    runner that holds until Stop Workflow, as a run between steps does."""
    from fibsem.applications.autolamella.structures import (
        AutoLamellaTaskProtocol,
        Experiment,
    )
    from fibsem.applications.autolamella.ui import window as module
    from fibsem.workflows.runner import ITEMS, RunRequest
    from tests.run_state import hold_runs

    window = module.AutoLamellaWindow()
    ui = window
    ui.system_widget.connect_to_microscope()
    microscope = ui.microscope
    experiment = Experiment(path=tmp_path, name="closing")
    experiment.task_protocol = AutoLamellaTaskProtocol()
    add_planned(experiment.task_protocol, planned(name="Trench", required=True))
    window.held = hold_runs(monkeypatch, ITEMS)
    started = ui.session.runner.start(
        experiment, RunRequest(kind=ITEMS, task_names=("Trench",))
    )
    assert started.started, started.reason
    window.show()
    try:
        yield window
    finally:
        window.held.end(ui.session.runner)
        if window.isVisible():  # a second close re-runs the whole closeEvent
            window.close()
        microscope.disconnect()
        window.deleteLater()
        qapp.processEvents()


def _answer(button):
    """Click *button* on the dialog the close is about to put up."""
    from PyQt5.QtCore import QTimer
    from PyQt5.QtWidgets import QApplication

    def click():
        box = QApplication.activeModalWidget()
        if box is not None:
            box.button(button).click()

    QTimer.singleShot(0, click)


def test_closing_during_a_workflow_can_be_cancelled(running):
    from PyQt5.QtWidgets import QMessageBox

    ui = running
    _answer(QMessageBox.No)

    running.close()

    assert running.isVisible()
    assert ui.session.runner.is_running
    assert not running.held.managers[0].is_stopped
    assert ui.microscope._connection.connected
    assert ui.session.recorder is not None


def test_confirming_the_close_stops_the_workflow_and_disconnects(running):
    from PyQt5.QtWidgets import QMessageBox

    ui = running
    microscope = ui.microscope
    _answer(QMessageBox.Yes)

    running.close()

    assert not running.isVisible()
    assert running.held.managers[0].is_stopped
    assert not microscope._connection.connected
