"""Keep Python's cyclic garbage collector on the Qt main thread.

CPython's automatic GC runs in whichever thread happens to allocate when a
generation threshold is crossed — during workflows that is usually a background
thread. Any cyclic garbage that holds Qt or vispy objects then has its
finalizers run *off* the GUI thread:

* QObject destructors from the wrong thread — the log signature is
  ``QObject::~QObject: Timers cannot be stopped from another thread``;
* vispy ``GLObject.__del__``, which appends a GLIR ``DELETE`` command to a
  plain (unlocked) Python list that the GUI thread's ``paintGL`` is
  concurrently flushing. On Windows GL drivers this intermittently ends in a
  native access violation inside ``glDrawArrays`` and takes the whole process
  down (observed repeatedly on a Tescan support PC, 2026-07-22).

The fix — as shipped by ilastik and other large PyQt apps — is to disable
automatic collection and run the collector from a QTimer on the main thread,
mirroring CPython's own generation thresholds so collection stays incremental
rather than a periodic stop-the-world full sweep.

Scope: this moves *cyclic* garbage finalization to the main thread. Plain
refcount-zero deallocation still runs on whichever thread drops the last
reference; cycles (signal connections, parent links, closures) are the
dominant path for Qt/napari/vispy objects and the one observed in the crash
logs.
"""
from __future__ import annotations

import gc
import logging
from typing import Optional

from PyQt5.QtCore import QCoreApplication, QObject, QThread, QTimer

__all__ = ["MainThreadGarbageCollector", "install_main_thread_gc"]


class MainThreadGarbageCollector(QObject):
    """Runs generational garbage collection from a timer on the main thread.

    ``start()`` disables automatic GC and starts the timer; ``stop()`` restores
    automatic GC. Each tick re-implements CPython's escalation rule using the
    interpreter's own thresholds: collect gen 0 when its allocation count is
    exceeded, escalating to gen 1 / gen 2 only when their counters are also
    over threshold — so most ticks are no-ops or cheap young-generation passes.
    """

    DEFAULT_INTERVAL_MS = 1000

    def __init__(
        self,
        parent: Optional[QObject] = None,
        interval_ms: int = DEFAULT_INTERVAL_MS,
    ):
        super().__init__(parent)
        self._threshold = gc.get_threshold()
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self.check)

    def start(self) -> None:
        gc.disable()
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        gc.enable()

    def check(self) -> None:
        """One timer tick: collect whichever generations are over threshold."""
        count0, count1, count2 = gc.get_count()
        if count0 <= self._threshold[0]:
            return
        if count1 <= self._threshold[1]:
            gc.collect(0)
        elif count2 <= self._threshold[2]:
            gc.collect(1)
        else:
            gc.collect(2)

    def collect_now(self) -> int:
        """Force an immediate full collection (for tests and shutdown paths)."""
        return gc.collect()


def install_main_thread_gc(
    parent: Optional[QObject] = None,
    interval_ms: int = MainThreadGarbageCollector.DEFAULT_INTERVAL_MS,
) -> MainThreadGarbageCollector:
    """Create and start a :class:`MainThreadGarbageCollector`.

    Must be called from the Qt main thread after the ``QApplication`` exists —
    the timer needs the main event loop, and creating it elsewhere would make
    collections run on the wrong thread, which is the exact bug this prevents.
    """
    app = QCoreApplication.instance()
    if app is None:
        raise RuntimeError(
            "install_main_thread_gc requires a QApplication to exist"
        )
    if QThread.currentThread() is not app.thread():
        raise RuntimeError(
            "install_main_thread_gc must be called from the Qt main thread"
        )
    collector = MainThreadGarbageCollector(parent=parent, interval_ms=interval_ms)
    collector.start()
    logging.debug(
        "Main-thread GC installed (interval=%d ms); automatic GC disabled",
        interval_ms,
    )
    return collector
