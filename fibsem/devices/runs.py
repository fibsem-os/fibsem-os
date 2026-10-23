"""Runs: a long command, from its start to how it ended, kept by its service.

A mill, a spot burn, an alignment or an overview runs for minutes. Whoever
started it (a workflow task, the operator's button, the agent over the App
Server) follows it the same way: the service it runs on keeps its latest run,
``service.runs``, with its id, its state and, once it has ended, its result or
error, and records ``command_started`` and ``command_ended`` events. The App
Server's ``GET .../{service}/run`` reads it.

A command marked ``@command(long=True)`` is one run whenever it is called. A
milling task is one run on the milling service, under the task's own id, which
its ``milling`` events carry as ``milling_run_id``. A long command called inside
a run already going on the same thread of control (a spot burn's mill, a task's
stage) is part of that run, and has no run of its own.

There is no job table. A run holds control of the microscope, and control is
one at a time, so each service has at most one run going; the last
one is kept, for its answer, until the next.
"""

from __future__ import annotations

import contextvars
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, Optional

from fibsem.cancellation import CANCELLED, FAILED, ended_by
from fibsem.util.timestamps import now_iso

__all__ = [
    "Run",
    "Runs",
    "RUNNING",
    "DONE",
    "FAILED",
    "CANCELLED",
    "current_run",
    "current_run_id",
    "ended_as",
]

RUNNING = "running"
DONE = "done"

# The run this thread of control is part of. A context variable, so a thread
# started with this context (the App Server's run thread) is part of it too.
_current: "contextvars.ContextVar[Optional[Run]]" = contextvars.ContextVar(
    "fibsem_run", default=None
)


def current_run() -> Optional["Run"]:
    """The run going on in this thread of control, on any service, or None."""
    return _current.get()


def current_run_id(service: str) -> Optional[str]:
    """The id of the run going on in this thread of control when it is a run
    on *service*, else None."""
    run = _current.get()
    return run.run_id if run is not None and run.service == service else None


@dataclass
class Run:
    """One long command, from its start to how it ended."""

    run_id: str
    service: str
    command: str
    started_at: str
    state: str = RUNNING
    ended_at: Optional[str] = None
    # What the command returned.
    result: Any = None
    # How it failed: error_type and message.
    error: Optional[Dict[str, Any]] = None
    # What it raised, for a reader that answers errors its own way (the App Server).
    exception: Optional[BaseException] = field(default=None, repr=False)
    # What its starter says about it on its events (the agent's action and level).
    labels: Dict[str, Any] = field(default_factory=dict, repr=False)
    # Set by a stop asked of it, which it takes as soon as it can: before it
    # starts milling, a stop has nothing else to set.
    stop: threading.Event = field(default_factory=threading.Event, repr=False)
    # How many callers are in :meth:`Runs.wait` for it.
    waiters: int = field(default=0, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "service": self.service,
            "command": self.command,
            "started_at": self.started_at,
            "state": self.state,
            "ended_at": self.ended_at,
            "result": self.result,
            "error": self.error,
        }


def ended_as(result: Any) -> str:
    """The state a run that returned *result* ended in. A stopped command
    answers ``cancelled`` (the milling service's string, the spot burn
    service's status, a milling task's ``status``) rather than raising,
    and a milling task that failed answers ``failed``."""
    if isinstance(result, dict):
        result = result.get("status")
    status = str(getattr(result, "value", result)).lower()
    if status == CANCELLED:
        return CANCELLED
    if status == FAILED:
        return FAILED
    return DONE


class Runs:
    """The latest run of one service (or device): ``service.runs``."""

    def __init__(self, owner: Any):
        self._owner = owner
        self._latest: Optional[Run] = None
        self._cond = threading.Condition()

    @property
    def latest(self) -> Optional[Run]:
        """The run going on, else the last one that ended; None before the first."""
        with self._cond:
            return self._latest

    @contextmanager
    def tracking(
        self, command: str, run_id: Optional[str] = None
    ) -> Iterator[Optional[Run]]:
        """The block is one run of *command*, under *run_id* or a new id.

        Yields the run, for the block to set its ``result`` (and its
        ``exception``, for a block that catches its own failure). Inside a run
        already going on this thread of control it yields None: the block is
        part of that run (`current_run`), which says how it ended.
        """
        if current_run() is not None:
            yield None
            return
        run = self._begin(command, run_id)
        token = _current.set(run)
        try:
            yield run
        except BaseException as error:
            run.exception = error
            raise
        finally:
            _current.reset(token)
            self._end(run)

    def start(
        self,
        command: str,
        call: Callable[[], Any],
        before_end: Optional[Callable[[], None]] = None,
        labels: Optional[Dict[str, Any]] = None,
    ) -> Run:
        """Run *call* as a run of *command* in a thread of its own, and answer
        with the run at once.

        *before_end* is called once the command has ended and before the run
        says so: the App Server gives up the control of the microscope there,
        so a caller that reads the end can act at once. The thread runs in this
        context, so the record still says who acted. *labels* go on its
        ``command_started`` and ``command_ended`` events.
        """
        run = self._begin(command, None, labels)
        context = contextvars.copy_context()
        threading.Thread(
            target=context.run,
            args=(self._carry, run, call, before_end),
            name=f"{run.service}.{command}",
            daemon=True,
        ).start()
        return run

    def _carry(self, run: Run, call, before_end) -> None:
        _current.set(run)  # this thread's own context, a copy
        try:
            run.result = call()
        except BaseException as error:  # noqa: BLE001 - the run's end says it
            run.exception = error
        finally:
            if before_end is not None:
                before_end()
            self._end(run)

    def wait(self, run_id: Optional[str], timeout: float) -> Optional[Run]:
        """The latest run, once it has ended or *timeout* seconds have passed.
        With a *run_id*, waits for that run only: a later one answers at once,
        so a caller can tell its run was not the last. A run that ends while
        someone waits for it says so on its ``command_ended`` (``waited``)."""
        with self._cond:
            run = self._latest
            if run is not None and run_id not in (None, run.run_id):
                run = None
            if run is not None:
                run.waiters += 1
            try:
                self._cond.wait_for(
                    lambda: self._ended(run_id), timeout=max(0.0, timeout)
                )
            finally:
                if run is not None:
                    run.waiters -= 1
            return self._latest

    def _ended(self, run_id: Optional[str]) -> bool:
        run = self._latest
        if run is None:
            return True
        if run_id is not None and run.run_id != run_id:
            return True
        return run.state != RUNNING

    def _begin(
        self,
        command: str,
        run_id: Optional[str],
        labels: Optional[Dict[str, Any]] = None,
    ) -> Run:
        run = Run(
            run_id=run_id or str(uuid.uuid4()),
            service=self._owner.name,
            command=command,
            started_at=now_iso(),
            labels=dict(labels or {}),
        )
        with self._cond:
            self._latest = run
            self._cond.notify_all()
        self._tell("command_started", run)
        return run

    def _end(self, run: Run) -> None:
        error: Optional[Dict[str, Any]] = None
        if run.exception is not None:
            # A stop raised is cancelled, and says so as its error, as it raised.
            state = ended_by(run.exception)["status"]
            error = {
                "error_type": type(run.exception).__name__,
                "message": str(run.exception),
            }
        else:
            state = ended_as(run.result)
            text = run.result.get("error") if isinstance(run.result, dict) else None
            if state == FAILED:
                # A milling task that caught its own failure says it in its answer.
                error = {"message": str(text or "failed")}
        with self._cond:
            run.state, run.error = state, error
            run.ended_at = now_iso()
            waited = run.waiters > 0
            self._cond.notify_all()
        self._tell("command_ended", run, waited=waited)

    def _tell(self, kind: str, run: Run, **extra: Any) -> None:
        payload = {
            "run_id": run.run_id,
            "service": run.service,
            "command": run.command,
            "state": run.state,
            **extra,
        }
        if run.error is not None:
            payload["error"] = run.error.get("message")
        payload.update(run.labels)
        session = getattr(self._owner, "session", None)
        if session is not None:
            session.record(kind, payload)
