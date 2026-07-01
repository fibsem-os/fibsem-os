"""A tiny, napari-free replacement for the subset of ``napari.qt.threading`` we use.

Two threading patterns exist in the GUI, and this wrapper covers both:

* **napari ``@thread_worker`` sites** — a decorated *plain function* (never a generator —
  nothing uses ``yield``), started with ``.start()`` and observed via ``.finished`` (and, in a
  couple of places, ``.returned`` / ``.errored``). Progress is reported through each widget's
  own ``pyqtSignal``s, not napari's ``.yielded``; no site uses pause/resume/abort.
* **manual ``threading.Thread`` sites** — some fire-and-forget, others stored on the widget
  (``self._acquisition_thread``) so the widget can poll ``.is_alive()`` and ``.join(timeout=)``
  on cancel / close. Cancellation there is widget-owned: the widget holds a ``threading.Event``
  and passes it into the blocking call; the worker only needs to be *observable*, not to own
  the stop signal.

So :class:`FunctionWorker` both re-emits its outcome as Qt signals *and* mirrors the slice of
the ``threading.Thread`` API those sites use (``start`` / ``is_alive`` / ``join``), making it a
drop-in for either pattern.

Because the worker is a ``QObject`` constructed on the calling (GUI) thread, emitting its
signals from the worker thread makes Qt deliver them through the GUI event loop — exactly the
mechanism napari uses, so signal-delivery threading is unchanged at every call site.

Migration is a one-line import swap::

    # from napari.qt.threading import thread_worker
    from fibsem.ui.qt.threading import thread_worker

This intentionally implements only the non-generator, bare-decorator subset. A generator body
or ``@thread_worker(connect=...)`` will fail loudly rather than silently misbehave.
"""
from __future__ import annotations

import functools
import logging
import threading
from typing import Callable, Optional, Set

from PyQt5.QtCore import QObject, pyqtSignal

__all__ = ["FunctionWorker", "thread_worker"]

# Call sites keep only a local ``worker = self.x_worker(...)`` reference, which would be
# garbage-collected the moment the method returns — before the thread finishes and its
# signals are delivered. Hold a strong reference here for the worker's lifetime (napari
# keeps its own registry for the same reason); it is released when ``finished`` fires.
_ACTIVE_WORKERS: Set["FunctionWorker"] = set()


class FunctionWorker(QObject):
    """Runs ``func(*args, **kwargs)`` on a daemon thread and re-emits the result as signals.

    Signals mirror the napari names we rely on and are delivered on the thread that created
    the worker (the GUI thread), so connected slots may safely touch widgets:

    * ``started``  — emitted just before the body runs.
    * ``returned`` — the return value, on success only.
    * ``errored``  — the exception, on failure only (also logged with a traceback).
    * ``finished`` — always, after ``returned``/``errored`` (matching napari).

    It also exposes the slice of the :class:`threading.Thread` API the manual-thread sites
    use — :meth:`is_alive` and :meth:`join` — so a widget that stored a raw ``Thread`` for
    lifecycle (``is_acquiring`` / ``cancel`` / ``closeEvent``) can hold a ``FunctionWorker``
    instead with no change to those call sites.
    """

    started = pyqtSignal()
    returned = pyqtSignal(object)
    errored = pyqtSignal(object)
    finished = pyqtSignal()

    def __init__(self, func: Callable, *args, **kwargs):
        super().__init__()
        self._func = func
        self._args = args
        self._kwargs = kwargs
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Launch the worker on a daemon thread."""
        _ACTIVE_WORKERS.add(self)
        # Released on the GUI thread once the worker is done (self lives there).
        self.finished.connect(lambda: _ACTIVE_WORKERS.discard(self))
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def is_alive(self) -> bool:
        """Whether the worker thread has been started and has not yet finished.

        Drop-in for ``threading.Thread.is_alive`` at the stored-thread lifecycle sites
        (``is_acquiring`` / ``is_milling`` / ``is_workflow_running``). ``False`` before
        :meth:`start`.
        """
        return self._thread is not None and self._thread.is_alive()

    def join(self, timeout: Optional[float] = None) -> None:
        """Block until the worker thread finishes (or ``timeout`` elapses).

        Drop-in for ``threading.Thread.join`` at the cancel / ``closeEvent`` sites. A no-op
        if the worker was never started. Cancellation itself is widget-owned (the widget sets
        its ``stop_event`` first, then joins to wait the body out).
        """
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self) -> None:
        self.started.emit()
        try:
            result = self._func(*self._args, **self._kwargs)
        except Exception as exc:  # noqa: BLE001 - report every failure, never swallow it
            logging.exception(
                "worker %r failed", getattr(self._func, "__name__", self._func)
            )
            self.errored.emit(exc)
        else:
            self.returned.emit(result)
        finally:
            self.finished.emit()


def thread_worker(func: Callable) -> Callable:
    """Turn ``func`` into a factory that returns a :class:`FunctionWorker`.

    Drop-in for the subset of :func:`napari.qt.threading.thread_worker` this codebase uses.
    The factory is a plain function, so it binds as a method — ``self.some_worker(...)``
    passes ``self`` through as the first positional argument, just like the original.
    """

    @functools.wraps(func)
    def factory(*args, **kwargs) -> FunctionWorker:
        return FunctionWorker(func, *args, **kwargs)

    return factory
