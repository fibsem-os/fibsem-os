"""`ensure_main_thread` runs its function on the GUI thread, called from any thread.

superqt's version crashed PySide6 when a worker thread called it and dropped the
future: the GUI thread then wrote the result into a freed object.
"""

import gc
import threading

import pytest

pytest.importorskip("qtpy.QtWidgets")

from qtpy.QtWidgets import QWidget

from fibsem.ui.qt import sip
from fibsem.ui.qt.main_thread import ensure_main_thread


def _drain(qapp, thread):
    while thread.is_alive():
        qapp.processEvents()
    for _ in range(20):
        qapp.processEvents()


def test_calls_from_a_worker_run_on_the_gui_thread(qapp):
    gui = threading.get_ident()
    seen = []

    @ensure_main_thread
    def record(i):
        seen.append((i, threading.get_ident()))

    def worker():
        for i in range(2000):
            record(i)  # the future is dropped at once
            if i % 100 == 0:
                gc.collect()

    thread = threading.Thread(target=worker)
    thread.start()
    _drain(qapp, thread)

    assert [i for i, _ in seen] == list(range(2000))
    assert {ident for _, ident in seen} == {gui}


def test_await_return_hands_back_the_result(qapp):
    @ensure_main_thread(await_return=True, timeout=5000)
    def double(x):
        return x * 2

    assert double(3) == 6  # on the GUI thread: at once
    results = []
    thread = threading.Thread(target=lambda: results.append(double(5)))
    thread.start()
    _drain(qapp, thread)
    assert results == [10]


def test_extra_signal_arguments_are_dropped(qapp):
    @ensure_main_thread
    def one(x):
        return x

    assert one(1, 2, 3).result() == 1


def test_a_deleted_widget_is_not_updated(qapp):
    """A widget Qt deleted can still be subscribed to a microscope signal."""
    calls = []

    class Panel(QWidget):
        @ensure_main_thread
        def refresh(self):
            calls.append(self)

    panel = Panel()
    sip.delete(panel)

    assert panel.refresh().result() is None
    assert calls == []
