"""Who is acting on the microscope: a task, the agent, or the operator (FIB-1062).

The code that makes a call marks it, for the thread of control it runs on: a
task for the length of its run, the App Server for each request. The
experiment's record reads the mark when an event is emitted, which is on the
thread that caused it.

A context variable rather than a thread-local. A new thread starts unmarked, as
it would with a thread-local, so a task's mark never reaches a UI worker running
beside it. But a web request's handler runs in a worker thread with the
request's context copied in, so the server's mark reaches the handler.

Work handed to another thread that is already running -- a Qt signal to the GUI
thread -- does not take the mark with it. :func:`carried` wraps the work where
it is handed over, so it runs there as whoever handed it over.

Only the task and the agent mark their calls. In the app, the operator is
everything else, and says so where the record is kept, not here.
"""

import functools
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator, Optional, TypeVar

T = TypeVar("T")

TASK = "task"
AGENT = "agent"
OPERATOR = "operator"

_actor: "ContextVar[Optional[str]]" = ContextVar("fibsem_actor", default=None)


@contextmanager
def acting(actor: Optional[str]) -> Iterator[None]:
    """Mark what runs inside this block, in this thread of control, as *actor*'s.

    None clears the mark: for recording, later, something done unmarked.
    Also a decorator: ``@acting(TASK)`` marks every call of the function.
    """
    token = _actor.set(actor)
    try:
        yield
    finally:
        _actor.reset(token)


def current_actor() -> Optional[str]:
    """Who marked this thread of control, or None if nobody did."""
    return _actor.get()


def carried(fn: Callable[..., T]) -> Callable[..., T]:
    """*fn*, marked as whoever called this, on whatever thread it later runs.

    The mark is read now, not when *fn* runs: hand the result to the other
    thread (FIB-1276).
    """
    actor = current_actor()

    @functools.wraps(fn)
    def run(*args, **kwargs) -> T:
        with acting(actor):
            return fn(*args, **kwargs)

    return run
