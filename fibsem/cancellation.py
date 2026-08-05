"""Cooperative cancellation for long-running microscope operations (milling, autofocus, ...).

A user "Stop"/"Cancel" is signalled by setting a :class:`threading.Event`; the operation polls it
at natural checkpoints (between passes/steps, before an irreversible action) via
:func:`raise_if_cancelled`, which raises :class:`OperationCancelledError` to unwind the call stack
cleanly. Callers distinguish a cancel from a real failure — surface it as a neutral "...cancelled"
(not an error), and in a workflow mark the task ``Cancelled`` rather than ``Failed``.
"""
from __future__ import annotations

import threading
from typing import Optional, Union


class OperationCancelledError(Exception):
    """Raised to abort a long-running operation when its stop event is set.

    A user-requested cancel, distinct from a genuine failure — catch it separately to surface
    "...cancelled" instead of "...failed", and re-raise it to unwind rather than swallow.

    Deliberately NOT named ``CancelledError``: that shadows ``concurrent.futures.CancelledError``
    and ``asyncio.CancelledError`` (the latter subclasses ``BaseException``, so ``except
    Exception`` would not catch it — a footgun if the two are ever confused).
    """


class AnyStopEvent:
    """A stop signal that is set when any of its component events is set.

    ``is_set()`` is the only thing :func:`raise_if_cancelled` and its callers ask of
    a stop event, so this stands in wherever a bare :class:`threading.Event` does.
    That lets several independent cancel signals — "stop the whole run", "abandon
    just this task" — share one token, without every polling site having to know
    there is more than one of them.

    Read-only on purpose: there is no ``set()``, because setting a composite would
    have to guess which underlying signal was meant. Set the one you mean.
    """

    def __init__(self, *events: threading.Event):
        self._events = events

    def is_set(self) -> bool:
        return any(event.is_set() for event in self._events)


def raise_if_cancelled(
    stop_event: Optional[Union[threading.Event, AnyStopEvent]],
    msg: str = "Operation cancelled by user.",
) -> None:
    """Raise :class:`OperationCancelledError` if ``stop_event`` is set.

    A no-op when ``stop_event`` is ``None`` or clear. Call at the natural checkpoints in a
    long-running operation (between passes/steps, before an irreversible action) so a Stop
    requested mid-operation takes effect promptly instead of only at coarse boundaries.
    """
    if stop_event is not None and stop_event.is_set():
        raise OperationCancelledError(msg)
