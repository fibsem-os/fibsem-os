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
