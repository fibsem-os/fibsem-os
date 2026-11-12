"""`ensure_main_thread`: run a function on the GUI thread, whichever thread calls it.

A drop-in for superqt's decorator, which crashes PySide6 when it is called from another
thread: it connects `future.set_result` as a bound method, PySide does not keep the
future alive, and the GUI thread then writes into a freed object. Here the call travels
as a Python closure on a queued signal, which both bindings hold until it has run.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future
from functools import wraps
from inspect import signature
from typing import Callable, Optional

from qtpy.QtCore import QCoreApplication, QObject, Qt, QThread, Signal, Slot

__all__ = ["ensure_main_thread"]


class _Dispatcher(QObject):
    call = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        # A slot of this object runs on this object's thread, wherever it was made.
        self.call.connect(self._run, Qt.ConnectionType.QueuedConnection)

    @Slot(object)
    def _run(self, run: Callable[[], None]) -> None:
        run()


_lock = threading.Lock()
_dispatcher: Optional[_Dispatcher] = None
_dispatcher_app: Optional[QCoreApplication] = None


def _gui_dispatcher(app: QCoreApplication) -> _Dispatcher:
    """The dispatcher for *app*, living on its thread. A new QApplication (tests make
    more than one) gets a new dispatcher."""
    global _dispatcher, _dispatcher_app
    with _lock:
        if _dispatcher is None or _dispatcher_app is not app:
            dispatcher = _Dispatcher()
            dispatcher.moveToThread(app.thread())
            _dispatcher, _dispatcher_app = dispatcher, app
        return _dispatcher


def _max_args(func: Callable) -> Optional[int]:
    """How many positional arguments *func* takes, None for any. A slot may take
    fewer than its signal sends; the rest are dropped, as superqt does."""
    try:
        params = signature(func).parameters.values()
    except (TypeError, ValueError):
        return None
    count = 0
    for param in params:
        if param.kind == param.VAR_POSITIONAL:
            return None
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
            count += 1
    return count


def ensure_main_thread(
    func: Optional[Callable] = None, await_return: bool = False, timeout: int = 1000
):
    """Run the decorated function on the GUI thread.

    From the GUI thread it runs at once. From any other thread it is queued, and the
    call returns a `Future` for its result, or with *await_return* waits up to
    *timeout* milliseconds and returns the result itself.
    """

    def decorate(func_: Callable) -> Callable:
        max_args = _max_args(func_)

        @wraps(func_)
        def wrapper(*args, **kwargs):
            args = args[:max_args]
            future: Future = Future()

            def run() -> None:
                # An error is raised on the GUI thread, as superqt does, so it is not
                # lost in a future nobody reads.
                try:
                    result = func_(*args, **kwargs)
                except BaseException as error:
                    future.set_exception(error)
                    raise
                future.set_result(result)

            app = QCoreApplication.instance()
            if app is None or QThread.currentThread() == app.thread():
                result = func_(*args, **kwargs)
                if await_return:
                    return result
                future.set_result(result)
                return future
            _gui_dispatcher(app).call.emit(run)
            return future.result(timeout=timeout / 1000) if await_return else future

        return wrapper

    return decorate if func is None else decorate(func)
