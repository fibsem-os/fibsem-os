"""Closing the main window a second time is harmless. QWidget.close() sends the
close event again even when the window is already hidden, so everything in
closeEvent runs twice; an exception escaping it aborts the process under PyQt5.

The real main window on Demo.
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


def test_closing_the_window_twice(no_quit, qapp):
    from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module

    window = module.AutoLamellaSingleWindowUI()
    window.autolamella_ui.system_widget.connect_to_microscope()
    window.autolamella_ui.system_widget.wait_for_connection()
    microscope = window.autolamella_ui.microscope
    window.show()
    try:
        window.close()
        assert not window.isVisible()

        window.close()

        assert not window.isVisible()
        assert window.autolamella_ui._event_recorder is None
    finally:
        microscope.disconnect()
        window.deleteLater()
        qapp.processEvents()
