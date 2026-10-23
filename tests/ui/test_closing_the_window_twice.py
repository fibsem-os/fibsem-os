"""Closing the main window a second time is harmless. QWidget.close() sends the
close event again even when the window is already hidden, so everything in
closeEvent runs twice; an exception escaping it aborts the process under PyQt5.

The real main window on Demo.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")


def test_closing_the_window_twice(no_quit, qapp):
    from fibsem.applications.autolamella.ui import window as module

    window = module.AutoLamellaWindow()
    window.system_widget.connect_to_microscope()
    microscope = window.microscope
    window.show()
    try:
        window.close()
        assert not window.isVisible()

        window.close()

        assert not window.isVisible()
        assert window.session.recorder is None
    finally:
        microscope.disconnect()
        window.deleteLater()
        qapp.processEvents()
