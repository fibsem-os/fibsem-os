"""Closing either main window lets the microscope go, rather than leaving the
client open until the process ends -- and a client that will not close does not
stop the window closing.

The real windows on Demo.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")


@pytest.fixture
def no_quit(qapp):
    # AutoLamella's closeEvent ends in app.quit(), which latches on the shared
    # test QApplication (see test_mainui_workflow_status.py).
    original_quit = qapp.quit
    qapp.quit = lambda: None
    try:
        yield
    finally:
        qapp.quit = original_quit


def _refuse_to_disconnect():
    raise RuntimeError("client already gone")


def test_closing_autolamella_disconnects_the_microscope(no_quit):
    from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module

    window = module.AutoLamellaSingleWindowUI()
    window.autolamella_ui.system_widget.connect_to_microscope()
    microscope = window.autolamella_ui.microscope
    assert microscope.connection.connected

    window.close()

    assert not microscope.connection.connected


def test_closing_fibsem_disconnects_the_microscope(qapp):
    from fibsem.ui.FibsemUI import FibsemUI

    window = FibsemUI()
    window.system_widget.connect_to_microscope()
    microscope = window.microscope
    assert microscope.connection.connected

    window.close()

    assert not microscope.connection.connected
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
def running(no_quit, qapp):
    """AutoLamella on Demo with a workflow running: a real worker that runs
    until Stop Workflow's signal, as a run between steps does."""
    from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module
    from fibsem.ui.qt.threading import FunctionWorker

    window = module.AutoLamellaSingleWindowUI()
    ui = window.autolamella_ui
    ui.system_widget.connect_to_microscope()
    microscope = ui.microscope
    ui._task_worker_thread = FunctionWorker(ui._workflow_stop_event.wait, 10)
    ui._task_worker_thread.start()
    window.show()
    try:
        yield window
    finally:
        ui._workflow_stop_event.set()
        ui._task_worker_thread.join(5)
        ui._task_worker_thread = None
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

    ui = running.autolamella_ui
    _answer(QMessageBox.No)

    running.close()

    assert running.isVisible()
    assert ui.is_workflow_running
    assert not ui._workflow_stop_event.is_set()
    assert ui.microscope.connection.connected
    assert ui._event_recorder is not None


def test_confirming_the_close_stops_the_workflow_and_disconnects(running):
    from PyQt5.QtWidgets import QMessageBox

    ui = running.autolamella_ui
    microscope = ui.microscope
    _answer(QMessageBox.Yes)

    running.close()

    assert not running.isVisible()
    assert ui._workflow_stop_event.is_set()
    assert not microscope.connection.connected
