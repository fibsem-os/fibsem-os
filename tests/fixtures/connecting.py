"""Connecting from a test: the Connection tab and the dialog connect on a worker
thread, so a test waits for the connect, and the slots it queued, to finish."""

from __future__ import annotations

import time

from qtpy.QtWidgets import QApplication


def wait_connected(widget, timeout: float = 60.0) -> None:
    """Pump the event loop until *widget* (the tab or the dialog) has no
    connect under way and its finished slots have run."""
    deadline = time.monotonic() + timeout
    while widget.is_connecting:
        if time.monotonic() > deadline:
            raise TimeoutError("the connect did not finish")
        QApplication.processEvents()
        time.sleep(0.002)
    QApplication.processEvents()


def connect_and_wait(widget, timeout: float = 60.0):
    """Press Connect (or Disconnect, when connected) and wait for it to finish."""
    worker = widget.connect_to_microscope()
    wait_connected(widget, timeout)
    return worker
