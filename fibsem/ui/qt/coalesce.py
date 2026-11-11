"""Hand values from any thread to the GUI thread at a steady rate, dropping the ones the
GUI could not have drawn in time.

Live FM frames arrive on the acquisition thread up to 20 times a second. Queueing each
one onto the GUI thread let a slow redraw build a backlog that froze the window
(buttons stopped answering, Stop Milling among them) and grew until it crashed.
"""

import logging
import threading
from typing import Any, Callable, List

from PyQt5.QtCore import QObject, QTimer

# How often live views redraw: 10 a second. The camera rarely gives more new frames.
LIVE_DRAW_INTERVAL_MS = 100

_EMPTY = object()


class Coalescer(QObject):
    """``put`` from any thread; ``deliver`` runs on the GUI thread at most once per
    interval with the newest value, or with every value since the last delivery when
    ``keep_all`` (a list, for a plot that must not lose points). Create it on the GUI
    thread."""

    def __init__(
        self,
        deliver: Callable[[Any], None],
        interval_ms: int = LIVE_DRAW_INTERVAL_MS,
        keep_all: bool = False,
        parent: Any = None,
    ):
        super().__init__(parent)
        self._deliver = deliver
        self._keep_all = keep_all
        self._lock = threading.Lock()
        self._pending: Any = [] if keep_all else _EMPTY
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self.flush)
        self._timer.start()

    def put(self, value: Any) -> None:
        with self._lock:
            if self._keep_all:
                self._pending.append(value)
            else:
                self._pending = value

    def flush(self) -> None:
        """Deliver what is waiting now, on the calling (GUI) thread."""
        with self._lock:
            if self._keep_all:
                value: Any = self._pending
                self._pending = []
                if not value:
                    return
            else:
                value, self._pending = self._pending, _EMPTY
                if value is _EMPTY:
                    return
        try:
            self._deliver(value)
        except Exception:  # noqa: BLE001 - PyQt aborts the process on a slot's raise
            logging.exception("Could not show a live update")

    def stop(self) -> None:
        self._timer.stop()
        with self._lock:
            self._pending = [] if self._keep_all else _EMPTY


__all__: List[str] = ["Coalescer", "LIVE_DRAW_INTERVAL_MS"]
