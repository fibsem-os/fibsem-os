"""``thread_worker``: run a function off the GUI thread and hear how it ended.

:class:`FunctionWorker` runs a plain function (not a generator) on a daemon thread and
re-emits its outcome as Qt signals, delivered on the GUI thread because the worker is a
``QObject`` made there. It also has the slice of the ``threading.Thread`` API widgets
use (``start`` / ``is_alive`` / ``join``). A generator body or
``@thread_worker(connect=...)`` fails loudly.

What belongs in a worker body
-----------------------------

**The operation, and nothing else.** The body runs on a plain thread, so the widget
that started it is off limits there. Either return a value and do every GUI update in
a slot on ``returned`` / ``errored`` (``ObjectiveControlWidget`` is the model; prefer
this one), or, for a worker that reports while it runs, emit one of the widget's own
``pyqtSignal``s: emitting is safe from any thread, calling is not.

Either way, **let the exception out**: ``FunctionWorker`` logs it with a traceback and
re-emits it as ``errored``, the only way the widget can tell a failed run from a
finished one.

The one call allowed from a body: ``notification_service.show_toast`` (and ``show``),
a queued ``pyqtSignal`` behind a function. Qt picks the connection type from the
receiver's thread, and every receiver is a widget on the GUI thread.

``@ensure_main_thread`` would make a widget call from a body correctly synchronised.
The reason not to is coupling: the sequence ends up expressed only as a widget method,
which cannot be run, tested or reused without the widget.
"""

from __future__ import annotations

import contextvars
import functools
import logging
import threading
from typing import TYPE_CHECKING, Callable, Optional, Set

from PyQt5.QtCore import QObject, pyqtSignal

from fibsem.acting import OPERATOR
from fibsem.control import MicroscopeBusy

if TYPE_CHECKING:
    from fibsem.control import Control

__all__ = ["FunctionWorker", "thread_worker"]

# Call sites keep only a local ``worker = self.x_worker(...)`` reference, which would be
# garbage-collected the moment the method returns — before the thread finishes and its
# signals are delivered. Hold a strong reference here for the worker's lifetime; it is
# released when ``finished`` fires.
_ACTIVE_WORKERS: Set["FunctionWorker"] = set()


class FunctionWorker(QObject):
    """Runs ``func(*args, **kwargs)`` on a daemon thread and re-emits the result as signals.

    Signals are delivered on the thread that created the worker (the GUI thread), so
    connected slots may safely touch widgets:

    * ``started``  — emitted just before the body runs.
    * ``returned`` — the return value, on success only.
    * ``errored``  — the exception, on failure only (also logged with a traceback).
    * ``finished`` — always, after ``returned``/``errored``.

    It also has :meth:`is_alive` and :meth:`join`, for a widget that checks on its worker
    (``is_acquiring`` / ``cancel`` / ``closeEvent``).

    The body runs with a copy of the context the worker was started from, which a raw
    ``Thread`` does not do. So work started on a task's behalf is still the task's in the
    experiment's record (``fibsem.acting``), and work started from the GUI is
    unmarked.
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
        self._context: Optional[contextvars.Context] = None
        self._release: Optional[Callable[[], None]] = None

    def start(
        self,
        control: Optional["Control"] = None,
        action: str = "",
    ) -> bool:
        """Launch the worker on a daemon thread. False when it was refused.

        With ``control``, the body is the operator's action on the microscope and
        holds control while it runs. It is taken here, so a refusal
        starts nothing: the worker reports ``errored`` (the ``MicroscopeBusy``) and
        ``finished``, as for a failed body. The operator's own clicks never refuse
        each other: an image during their mill goes ahead.
        """
        self._context = contextvars.copy_context()
        if control is not None and not self._take(control, action):
            return False
        _ACTIVE_WORKERS.add(self)
        # Released on the GUI thread once the worker is done (self lives there).
        self.finished.connect(lambda: _ACTIVE_WORKERS.discard(self))
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def is_alive(self) -> bool:
        """Whether the worker thread has been started and has not yet finished.

        Drop-in for ``threading.Thread.is_alive`` at the stored-thread lifecycle sites
        (``is_acquiring`` / ``is_milling``). ``False`` before
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

    def _take(self, control: "Control", action: str) -> bool:
        # Taken in the body's context, so what the take sets there (a remote
        # microscope's lease) is the body's and goes with it.
        taking = control.take(OPERATOR, action, join=True)
        try:
            self._context.run(taking.__enter__)
        except Exception as error:  # noqa: BLE001 - refused or unreachable: said
            from fibsem.ui.control import refused

            if isinstance(error, MicroscopeBusy) and not self.receivers(self.errored):
                refused(error)
            elif not isinstance(error, MicroscopeBusy):
                logging.warning("could not take control for %s: %s", action, error)
            self.errored.emit(error)
            self.finished.emit()
            return False
        context = self._context
        self._release = lambda: context.run(taking.__exit__, None, None, None)
        return True

    def _run(self) -> None:
        self.started.emit()
        try:
            try:
                result = self._context.run(self._func, *self._args, **self._kwargs)
            finally:
                # Given up before any slot runs, so a slot that acts finds it free.
                if self._release is not None:
                    self._release()
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

    The factory is a plain function, so it binds as a method: ``self.some_worker(...)``
    passes ``self`` through as the first positional argument.
    """

    @functools.wraps(func)
    def factory(*args, **kwargs) -> FunctionWorker:
        return FunctionWorker(func, *args, **kwargs)

    return factory
