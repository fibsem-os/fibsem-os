"""Headless tests for the main-thread garbage collector.

Guards the contract behind the Windows vispy/gloo crash fix: cyclic garbage
that is *created and dropped in a worker thread* must never be finalized on
that worker thread while the collector is installed — finalizers (QObject
destructors, vispy ``GLObject.__del__``) must run on the Qt main thread.

Two sides are tested:

1. **The hazard is real** — with automatic GC enabled (stock CPython), an
   allocation storm in a worker thread collects the cycle *in that thread*.
   This keeps the fix honest: if this test ever fails, CPython's GC behaviour
   changed and the fix may be obsolete.
2. **The fix works** — with :class:`MainThreadGarbageCollector` started,
   the same storm finalizes nothing off-thread; the cycle is collected on the
   main thread by the timer-driven ``check()``.

Uses PyQt5 directly with the offscreen platform (no pytest-qt dependency).
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import gc
import threading

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

from fibsem.ui.qt.gc import MainThreadGarbageCollector, install_main_thread_gc


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _restore_gc_state():
    """Every test leaves the interpreter's GC exactly as it found it."""
    was_enabled = gc.isenabled()
    try:
        yield
    finally:
        gc.collect()
        if was_enabled:
            gc.enable()
        else:
            gc.disable()


class _Finalizable:
    """Self-referential (cyclic) object that records which thread ran __del__."""

    def __init__(self, record: list):
        self._record = record
        self._cycle = self  # guarantees the GC, not refcounting, frees it

    def __del__(self):
        self._record.append(threading.current_thread())


def _worker_drop_and_storm(record: list, n_allocations: int = 30_000):
    """Create + drop a cyclic finalizable, then allocate heavily.

    The storm keeps its allocations alive so the gen-0 counter (a net
    live-allocation count) crosses the threshold many times inside this
    thread. With automatic GC enabled the collector therefore runs here and
    the cycle's finalizer executes on this (worker) thread — the exact
    off-main-thread finalization that corrupts vispy's GLIR queue in the
    real app.
    """
    obj = _Finalizable(record)
    del obj
    sink = []
    for i in range(n_allocations):
        sink.append([i])  # net container allocations feed the gen-0 counter


def test_hazard_worker_thread_gc_finalizes_off_main_thread(qapp):
    """Control: stock CPython runs the finalizer on the worker thread."""
    gc.enable()
    gc.collect()  # start from empty generations

    record: list = []
    worker = threading.Thread(target=_worker_drop_and_storm, args=(record,))
    worker.start()
    worker.join()

    assert record, "allocation storm never triggered a collection — increase n_allocations"
    assert record[0] is not threading.main_thread(), (
        "expected the control finalizer to run on the worker thread; if CPython's "
        "GC threading behaviour changed, the main-thread GC fix may be obsolete"
    )


def test_fix_finalizers_run_only_on_main_thread(qapp):
    """With the collector started, the worker storm finalizes nothing;
    the main-thread check() collects the cycle on the main thread."""
    collector = MainThreadGarbageCollector(interval_ms=50)
    collector.start()
    try:
        gc.collect()  # start from empty generations

        record: list = []
        worker = threading.Thread(target=_worker_drop_and_storm, args=(record,))
        worker.start()
        worker.join()

        assert record == [], "automatic GC ran in the worker thread despite gc.disable()"

        # simulate timer ticks on the main thread until the cycle is collected
        for _ in range(10):
            collector.check()
            if record:
                break
        if not record:
            collector.collect_now()

        assert record, "cycle was never collected by the main-thread collector"
        assert all(t is threading.main_thread() for t in record)
    finally:
        collector.stop()


def test_install_requires_qapp_and_disables_automatic_gc(qapp):
    collector = install_main_thread_gc(interval_ms=1000)
    try:
        assert not gc.isenabled()
    finally:
        collector.stop()
    assert gc.isenabled()


def test_stop_restores_automatic_gc(qapp):
    collector = MainThreadGarbageCollector(interval_ms=1000)
    collector.start()
    assert not gc.isenabled()
    collector.stop()
    assert gc.isenabled()
