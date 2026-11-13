"""The last value of each instrument read, for a caller that would rather have it
than wait while the instrument is busy.

A read can wait behind an action: on a TFS microscope a detector read waits for
the imaging channel an image holds, and the state read waits for all of them. A
caller that sends ``X-Fibsem-Read-Wait: <seconds>`` waits that long for the read.
Past it, the caller gets the last value read, with its age in seconds in
``X-Fibsem-Read-Age``, and the read finishes in the background to become the
next last value. With no last value yet, the caller waits for the read.

Without the header a read is live, as before, and its value is kept too. The
agent's client sends no header, so the agent always reads live.
"""

import contextvars
import threading
import time
from concurrent.futures import Future
from concurrent.futures import wait as wait_for
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple
from weakref import WeakKeyDictionary

from starlette.requests import Request
from starlette.responses import Response

READ_WAIT_HEADER = "X-Fibsem-Read-Wait"
READ_AGE_HEADER = "X-Fibsem-Read-Age"
# The longest a caller may ask to wait before a last value: past this it waits
# for the read anyway.
MAX_READ_WAIT_S = 30.0


@dataclass
class _Slot:
    kept: Optional[Tuple[Any, float]] = None
    running: Optional[Future] = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class KeptReads:
    """Each read's last value, by the instrument or device it read and a name."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        # By the object read, so a reconnected microscope starts with none.
        self._slots: "WeakKeyDictionary[Any, Dict[str, _Slot]]" = WeakKeyDictionary()

    def _slot(self, owner: Any, name: str) -> _Slot:
        with self._lock:
            return self._slots.setdefault(owner, {}).setdefault(name, _Slot())

    def read(
        self, owner: Any, name: str, read: Callable[[], Any], wait: Optional[float]
    ) -> Tuple[Any, Optional[float]]:
        """*read*'s value and None, or, when it takes longer than *wait* seconds
        and a value was read before, that value and its age in seconds. With
        *wait* None, a live read. A read already running for the same name is
        joined, not started again."""
        slot = self._slot(owner, name)
        if wait is None:
            value = read()
            self._keep(slot, value)
            return value, None
        with slot.lock:
            future = slot.running
            if future is None:
                future = slot.running = Future()
                context = contextvars.copy_context()
                threading.Thread(
                    target=context.run,
                    args=(self._run, slot, future, read),
                    name=f"read {name}",
                    daemon=True,
                ).start()
            kept = slot.kept
        if kept is not None and not wait_for([future], timeout=wait).done:
            value, at = kept
            return value, max(self._clock() - at, 0.0)
        return future.result(), None

    def _run(self, slot: _Slot, future: Future, read: Callable[[], Any]) -> None:
        try:
            value = read()
        except BaseException as error:
            with slot.lock:
                slot.running = None
            future.set_exception(error)
            return
        with slot.lock:
            slot.kept = (value, self._clock())
            slot.running = None
        future.set_result(value)

    def _keep(self, slot: _Slot, value: Any) -> None:
        with slot.lock:
            slot.kept = (value, self._clock())


def read_wait(request: Request) -> Optional[float]:
    """The seconds the caller will wait for a read (`READ_WAIT_HEADER`), or None."""
    raw = request.headers.get(READ_WAIT_HEADER)
    if raw is None:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        return None
    if seconds != seconds:  # NaN
        return None
    return min(max(seconds, 0.0), MAX_READ_WAIT_S)


def with_age(response: Response, age: Optional[float]) -> Response:
    """*response*, saying how old its value is when it is a kept one."""
    if age is not None:
        response.headers[READ_AGE_HEADER] = f"{age:.1f}"
    return response
