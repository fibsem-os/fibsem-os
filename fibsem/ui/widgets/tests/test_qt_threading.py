"""Unit test for the napari-free ``thread_worker`` replacement (``fibsem.ui.qt.threading``).

Covers the properties the migration depends on:
  * success   — ``returned`` carries the value and its slot runs on the *main* thread,
                while the body itself runs off it; ``finished`` fires; method binding works.
  * error     — a raising body emits ``errored`` (with the exception) and still ``finished``,
                but not ``returned``.
  * liveness  — a worker started with no surviving local reference is not garbage-collected
                mid-run (the module keeps it alive until ``finished``).
  * lifecycle — ``is_alive()`` / ``join(timeout)`` mirror ``threading.Thread`` for the stored
                -thread sites (``is_acquiring`` / cancel / ``closeEvent``).

Run directly (headless):
    QT_QPA_PLATFORM=offscreen python fibsem/ui/widgets/tests/test_qt_threading.py
"""
from __future__ import annotations

import gc
import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets
from PyQt5.QtCore import QEventLoop, QObject, QTimer

_QAPP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

from fibsem.ui.qt import threading as qtthreading  # aliased: stdlib ``threading`` above stays intact
from fibsem.ui.qt.threading import FunctionWorker, thread_worker

_MAIN_THREAD = threading.current_thread()


def _spin(signal, timeout_ms: int = 5000) -> None:
    """Run the event loop until ``signal`` fires (or the timeout elapses)."""
    loop = QEventLoop()
    signal.connect(loop.quit)
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec_()
    _QAPP.processEvents()


class _Receiver(QObject):
    """A GUI-thread QObject: bound-method slots are the guaranteed cross-thread-queued case."""

    def __init__(self):
        super().__init__()
        self.returned_value = None
        self.returned_on_main = None
        self.finished_on_main = None
        self.errored_value = "unset"

    def on_returned(self, value):
        self.returned_value = value
        self.returned_on_main = threading.current_thread() is _MAIN_THREAD

    def on_finished(self):
        self.finished_on_main = threading.current_thread() is _MAIN_THREAD

    def on_errored(self, exc):
        self.errored_value = exc


class _Host:
    """The decorator must bind as a method: ``self`` is passed through as arg 0."""

    @thread_worker
    def double(self, x: int):
        # runs on the worker thread; return the value + the thread it ran on
        return {"result": x * 2, "thread": threading.current_thread()}


def test_success_delivers_on_main_thread():
    host = _Host()
    recv = _Receiver()

    worker = host.double(21)
    assert isinstance(worker, FunctionWorker)
    worker.returned.connect(recv.on_returned)
    worker.finished.connect(recv.on_finished)
    worker.start()
    _spin(worker.finished)

    assert recv.returned_value["result"] == 42          # method binding worked (self + x)
    assert recv.returned_value["thread"] is not _MAIN_THREAD  # body ran off the main thread
    assert recv.returned_on_main is True                # ...but the slot ran ON the main thread
    assert recv.finished_on_main is True


def test_error_still_finishes():
    @thread_worker
    def boom():
        raise ValueError("nope")

    recv = _Receiver()
    fired = []
    worker = boom()
    worker.errored.connect(recv.on_errored)
    worker.returned.connect(lambda r: fired.append(("returned", r)))  # must NOT fire
    worker.finished.connect(lambda: fired.append(("finished", None)))
    worker.start()
    _spin(worker.finished)

    assert isinstance(recv.errored_value, ValueError)
    assert str(recv.errored_value) == "nope"
    assert not any(k == "returned" for k, _ in fired)  # returned skipped on error
    assert ("finished", None) in fired                 # finished still fired


def test_worker_survives_dropped_local_ref():
    done = []

    @thread_worker
    def slow():
        time.sleep(0.05)  # still running during the gc.collect() below
        return 7

    def launch():
        worker = slow()
        worker.returned.connect(lambda r: done.append(r))
        worker.start()
        # local `worker` goes out of scope here — only _ACTIVE_WORKERS keeps it alive

    launch()
    assert len(qtthreading._ACTIVE_WORKERS) == 1
    gc.collect()  # would collect the worker if it were not self-registered

    deadline = 3.0
    while deadline > 0 and not done:
        _QAPP.processEvents()
        time.sleep(0.02)
        deadline -= 0.02

    assert done == [7]
    _QAPP.processEvents()
    assert len(qtthreading._ACTIVE_WORKERS) == 0  # released once finished fired


def test_is_alive_and_join_mirror_thread():
    """The stored-thread sites poll ``is_alive()`` and ``join(timeout=)`` on a FunctionWorker
    held in place of a raw ``threading.Thread`` — the same shape as ``is_acquiring`` + cancel.
    A widget-owned Event stops the body; ``join`` waits it out."""
    stop_event = threading.Event()

    @thread_worker
    def acquire():
        # a bounded acquisition loop, cancelled by the widget-owned stop_event
        while not stop_event.is_set():
            time.sleep(0.01)

    worker = acquire()
    assert worker.is_alive() is False          # not started yet
    worker.start()
    # give the daemon thread a moment to actually enter the loop
    for _ in range(50):
        if worker.is_alive():
            break
        time.sleep(0.01)
    assert worker.is_alive() is True           # running

    # cancel: set the stop event (widget-owned), then join with a timeout (as cancel does)
    stop_event.set()
    worker.join(timeout=5)
    assert worker.is_alive() is False          # joined cleanly within the timeout

    _spin(worker.finished, timeout_ms=1000)    # drain the queued finished signal


def test_join_before_start_is_noop():
    @thread_worker
    def noop():
        return None

    worker = noop()
    worker.join(timeout=1)                      # must not raise when never started
    assert worker.is_alive() is False


if __name__ == "__main__":
    test_success_delivers_on_main_thread()
    test_error_still_finishes()
    test_worker_survives_dropped_local_ref()
    test_is_alive_and_join_mirror_thread()
    test_join_before_start_is_noop()
    print("PASS")
