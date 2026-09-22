"""Who is acting on the microscope: a task, the agent, or the operator (FIB-1062).

The code that makes a call marks it, for the thread of control it runs on: a
task for the length of its run, the agent server for each request. The
experiment's record reads the mark when an event is emitted, which is on the
thread that caused it.

A context variable rather than a thread-local. A new thread starts unmarked, as
it would with a thread-local, so a task's mark never reaches a UI worker running
beside it. But a web request's handler runs in a worker thread with the
request's context copied in, so the server's mark reaches the handler.

Only the task and the agent mark their calls. In the app, the operator is
everything else, and says so where the record is kept, not here.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

TASK = "task"
AGENT = "agent"
OPERATOR = "operator"

_actor: "ContextVar[Optional[str]]" = ContextVar("fibsem_actor", default=None)


@contextmanager
def acting(actor: str) -> Iterator[None]:
    """Mark what runs inside this block, in this thread of control, as *actor*'s.

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
