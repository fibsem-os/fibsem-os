"""Long commands an agent starts: answered at once, run on (map 8.3).

A mill or a spot burn runs for minutes, longer than an agent's tool call can
wait (the sidecar gives up after 120 s), and a dropped call would lose the
result. So the App Server answers a command marked ``long`` with a run id at
once and runs it on in a thread of its own, still in control of the
microscope. How it ended is read back from ``GET .../{service}/run``, which can
wait for the end, and is recorded as ``command_started`` and ``command_ended``
events on ``/app/events``.

There is no job table. A run holds control of the microscope, and control is
one at a time (FIB-1270), so each service has at most one run going; the last
one is kept, for its answer, until the next.
"""

from __future__ import annotations

import contextvars
import threading
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional

from fibsem.util.timestamps import now_iso

__all__ = ["Run", "Runs", "RUNNING", "DONE", "FAILED", "CANCELLED"]

RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"


@dataclass
class Run:
    """One long command, from its start to how it ended."""

    run_id: str
    service: str
    command: str
    started_at: str
    state: str = RUNNING
    ended_at: Optional[str] = None
    # What the command returned, as its answer would have carried it.
    result: Any = None
    # How it failed: error_type and message, as a refusal carries them.
    error: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _cancelled(result: Any) -> bool:
    """A run that was stopped answers ``cancelled`` (the milling service's
    string, the spot burn service's status) rather than raising (FIB-1377)."""
    return str(getattr(result, "value", result)).lower() == CANCELLED


class Runs:
    """The latest run of each service.

    *record* is told ``command_started`` and ``command_ended`` with the run, for
    the event stream.
    """

    def __init__(self, record: Optional[Callable[[str, Dict], None]] = None):
        self._runs: Dict[str, Run] = {}
        self._cond = threading.Condition()
        self._record = record

    def start(
        self,
        service: str,
        command: str,
        call: Callable[[], Any],
        answer: Callable[[Any], Any],
        failure: Callable[[BaseException], Dict[str, Any]],
        release: Optional[Callable[[], None]] = None,
    ) -> Run:
        """Run *call* in a thread and answer with its run at once.

        *answer* turns what it returned into the run's ``result``, *failure* an
        exception into its ``error``. *release* gives up the control of the
        microscope the request took, once the command has ended and before the
        run says so: an agent that reads the end can act at once. The thread
        runs in the request's context, so the record still says who acted.
        """
        run = Run(
            run_id=str(uuid.uuid4()),
            service=service,
            command=command,
            started_at=now_iso(),
        )
        with self._cond:
            self._runs[service] = run
        self._tell("command_started", run)
        context = contextvars.copy_context()
        threading.Thread(
            target=context.run,
            args=(self._run, run, call, answer, failure, release),
            name=f"{service}.{command}",
            daemon=True,
        ).start()
        return run

    def _run(self, run, call, answer, failure, release) -> None:
        state, result, error = DONE, None, None
        try:
            try:
                returned = call()
                result = answer(returned)
                if _cancelled(returned):
                    state = CANCELLED
            except BaseException as exc:  # noqa: BLE001 - the run's answer says it
                error = failure(exc)
                state = CANCELLED if error.get("status") == CANCELLED else FAILED
        finally:
            if release is not None:
                release()
            with self._cond:
                run.state, run.result, run.error = state, result, error
                run.ended_at = now_iso()
                self._cond.notify_all()
            self._tell("command_ended", run)

    def get(self, service: str) -> Optional[Run]:
        with self._cond:
            return self._runs.get(service)

    def wait(
        self, service: str, run_id: Optional[str], timeout: float
    ) -> Optional[Run]:
        """The service's latest run, once it has ended or *timeout* seconds have
        passed. With a *run_id*, waits for that run only: a later one answers at
        once, so a caller can tell its run was not the last."""
        with self._cond:
            self._cond.wait_for(
                lambda: self._ended(service, run_id), timeout=max(0.0, timeout)
            )
            return self._runs.get(service)

    def _ended(self, service: str, run_id: Optional[str]) -> bool:
        run = self._runs.get(service)
        if run is None:
            return True
        if run_id is not None and run.run_id != run_id:
            return True
        return run.state != RUNNING

    def _tell(self, kind: str, run: Run) -> None:
        if self._record is None:
            return
        payload = {
            "run_id": run.run_id,
            "service": run.service,
            "command": run.command,
            "state": run.state,
        }
        if run.error is not None:
            payload["error"] = run.error.get("message")
        self._record(kind, payload)
