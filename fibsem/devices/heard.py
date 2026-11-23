"""When a device was last heard from: kept from the calls a run already makes
to it, never from a call of its own.

Each live read, write and command on a device or service (`fibsem.devices.core`)
goes through `heard_call`, which notes the call while it runs, when one last came
back, and the last that failed in the driver or the hardware. A refusal (a value
out of range, a Safety Halt, a command not available now, a busy resource, a
cancel) says nothing about the hardware and is not a failure here. The App
Server serves it in ``/microscope/health``; the gateway reads a call that has run
too long as a hung driver.
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = ["Heard", "HeardCall", "HeardError", "heard_call", "heard_of"]

# Exceptions that refuse a call before or instead of the hardware: by class
# name, so this module imports none of their modules.
REFUSALS = frozenset(
    {
        "SafetyHalted",
        "ParameterOutOfRange",
        "ParameterReadOnly",
        "ParameterUnavailable",
        "CommandUnavailable",
        "ResourceBusy",
        "RoleUnfilled",
        "OperationCancelledError",
        "NothingToStop",
        "ValueError",
        "TypeError",
        "KeyboardInterrupt",
    }
)
MESSAGE_CHARS = 300


@dataclass(frozen=True)
class HeardCall:
    """A call running now: what it is, since when, and whether it is a long
    command (a mill) that may run for minutes."""

    call: str
    since: float
    long: bool = False


@dataclass(frozen=True)
class HeardError:
    call: str
    at: float
    message: str


@dataclass
class Heard:
    """One device's record of its calls."""

    last_ok: Optional[float] = None
    last_error: Optional[HeardError] = None
    _running: Dict[int, HeardCall] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def running(self) -> List[HeardCall]:
        """The calls running now, longest first."""
        with self._lock:
            return sorted(self._running.values(), key=lambda c: c.since)

    def _start(self, key: int, call: HeardCall) -> None:
        with self._lock:
            self._running[key] = call

    def _end(self, key: int, error: Optional[BaseException], now: float) -> None:
        with self._lock:
            call = self._running.pop(key, None)
            if error is None:
                self.last_ok = now
            elif type(error).__name__ not in REFUSALS and call is not None:
                message = f"{type(error).__name__}: {error}"[:MESSAGE_CHARS]
                self.last_error = HeardError(call.call, now, message)


_keys = itertools.count()
_guard = threading.Lock()


def heard_of(controllable: Any) -> Heard:
    """*controllable*'s record, made the first time it is asked for."""
    found = getattr(controllable, "_heard", None)
    if found is None:
        with _guard:
            found = getattr(controllable, "_heard", None)
            if found is None:
                found = Heard()
                try:
                    controllable._heard = found
                except AttributeError:  # no __dict__: kept nowhere, as a stranger
                    pass
    return found


@contextmanager
def heard_call(
    controllable: Any, call: str, long: bool = False, clock=time.time
) -> Iterator[None]:
    """Note *call* on *controllable* while it runs, and how it ended.

    It never raises into the call nor makes it wait on anything but its own
    dict: a failure to note is logged once and the call runs as it would have,
    its own exception, if any, passed on unchanged."""
    heard: Optional[Heard] = None
    key = 0
    try:
        heard = heard_of(controllable)
        key = next(_keys)
        heard._start(key, HeardCall(call, clock(), long))
    except Exception as failure:  # noqa: BLE001 - noting a call never stops it
        _failed(failure)
        heard = None
    try:
        yield
    except BaseException as error:
        _end(heard, key, error, clock)
        raise
    _end(heard, key, None, clock)


def _end(heard: Optional[Heard], key: int, error, clock) -> None:
    if heard is None:
        return
    try:
        heard._end(key, error, clock())
    except Exception as failure:  # noqa: BLE001 - noting a call never stops it
        _failed(failure)


_warned = False


def _failed(failure: Exception) -> None:
    global _warned
    if not _warned:
        _warned = True
        logger.debug("Could not note a device call", exc_info=failure)


def summary(
    controllable: Any,
) -> Tuple[Optional[float], Optional[HeardError], List[HeardCall]]:
    """``(last_ok, last_error, running)`` for *controllable*."""
    heard = heard_of(controllable)
    return heard.last_ok, heard.last_error, heard.running()
