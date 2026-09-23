"""The app's event stream for a microscope session, and its record on disk.

The event buffer (``server/events.py``) began as the agent server's: built when
the server started, gone when it stopped, so with the server off nothing -- the
dashboard, a monitor, a replay -- had a stream to read. It is the app's now.
An :class:`EventRecorder` is created when a microscope connects, whether or not
any server runs, and owns:

* the buffer, with every event stamped with where in the run it happened;
* the taps feeding it: the microscope's signals, the prompt events, the
  open experiment's questions and decisions, and the task lifecycle hook
  (which ``setup_hooks`` registers on every run);
* an :class:`EventFileWriter` recording every event to ``events.jsonl`` beside
  the experiment's ``logfile.log``.

A workflow run records through the same recorder: the task manager finds the
microscope's with :func:`recorder_for`, or makes one for the run when nothing
keeps one -- a run without the GUI (FIB-1044).

The agent server, when it runs, reads the same buffer.

Events while no experiment is loaded stay in the buffer but are not written:
the file belongs to an experiment, and there is none yet.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import uuid
import weakref
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from fibsem.acting import AGENT, OPERATOR, TASK, acting, current_actor
from fibsem.applications.autolamella.proposals import _encode_values
from fibsem.applications.autolamella.server.events import (
    EventBuffer,
    attach_microscope_taps,
    make_lifecycle_hook,
)

EVENTS_FILENAME = "events.jsonl"

# How many records may wait for the disk before new ones are dropped. A stalled
# disk must cost records, never the emitting thread -- which is often milling.
_QUEUE_LIMIT = 50_000
_CLOSE_TIMEOUT_S = 2.0

_PATH, _RECORD, _STOP = "path", "record", "stop"

# The open recorder for each microscope, by id: the one a workflow run records
# through. A second recorder on the same microscope would tap the same signals
# and write every event twice. Weak: whatever made a recorder keeps it (the app
# its window, a run its own), and the registry must not keep it, or the window
# its disposers reach, alive after that is gone.
_RECORDERS: "weakref.WeakValueDictionary[int, EventRecorder]" = (
    weakref.WeakValueDictionary()
)


def recorder_for(microscope: Any) -> Optional["EventRecorder"]:
    """The open recorder for *microscope*, if something keeps one: the app does,
    from the moment a microscope connects."""
    recorder = _RECORDERS.get(id(microscope))
    return (
        recorder if recorder is not None and recorder.microscope is microscope else None
    )


class EventFileWriter:
    """Appends records to a JSON Lines file, from a thread of its own.

    :meth:`write` only enqueues, so it is safe on any thread and never blocks.
    A change of file (:meth:`set_path`) travels through the same queue, so a
    record lands in the file that was current when it was written, not the one
    current when the thread got to it.

    One JSON object per line, flushed after each batch. A torn final write is
    detectable -- the line has no newline -- which :func:`read_events` relies on.
    Failures are logged, once per file, and never raised.
    """

    def __init__(self) -> None:
        self._queue: "queue.Queue[Tuple[str, Any]]" = queue.Queue(maxsize=_QUEUE_LIMIT)
        self.dropped = 0
        self._thread = threading.Thread(
            target=self._run, name="fibsem-event-writer", daemon=True
        )
        self._thread.start()

    def set_path(self, path: Optional[Path]) -> None:
        """Write subsequent records to ``path`` (appending), or nowhere for None."""
        self._put((_PATH, Path(path) if path is not None else None))

    def write(self, record: Dict[str, Any]) -> None:
        """Queue ``record``. Never blocks; drops it if the disk is far behind."""
        self._put((_RECORD, record))

    def close(self, timeout: float = _CLOSE_TIMEOUT_S) -> None:
        """Write what is queued, then stop. Waits at most ``timeout`` seconds."""
        try:
            self._queue.put((_STOP, None), timeout=timeout)
        except queue.Full:
            pass
        self._thread.join(timeout)

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def _put(self, item: Tuple[str, Any]) -> None:
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            self.dropped += 1
            if self.dropped == 1:
                logging.warning(
                    "event recording is behind; dropping events until it catches up"
                )

    def _run(self) -> None:
        path: Optional[Path] = None
        handle = None
        failed: Optional[Path] = None
        while True:
            batch = [self._queue.get()]
            while True:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            for kind, value in batch:
                if kind == _PATH:
                    handle = _close(handle)
                    path = value
                elif kind == _STOP:
                    _close(handle)
                    return
                elif path is not None:
                    try:
                        if handle is None:
                            handle = open(path, "a", encoding="utf-8", newline="\n")
                        handle.write(json.dumps(value, default=str) + "\n")
                    except Exception:  # noqa: BLE001 - recording must not matter
                        handle = _close(handle)
                        if failed != path:
                            failed = path
                            logging.exception(f"could not record events to {path}")
            if handle is not None:
                try:
                    handle.flush()
                except Exception:  # noqa: BLE001
                    handle = _close(handle)


def _close(handle) -> None:
    if handle is not None:
        try:
            handle.close()
        except Exception:  # noqa: BLE001
            pass
    return None


def read_events(path: Path) -> Iterator[Dict[str, Any]]:
    """The records in an events file, in order.

    A final line without its newline is a write that was cut off -- the app
    stopped mid-record -- and is skipped rather than half-read. Any other line
    that does not parse is skipped too.
    """
    with open(path, encoding="utf-8") as f:
        text = f.read()
    lines = text.split("\n")
    # The last element is "" after a complete final line; anything else is torn.
    for line in lines[:-1]:
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            yield record


def _lifecycle_events() -> frozenset:
    from fibsem.hooks import HookEvent

    return frozenset(event.value for event in HookEvent)


_LIFECYCLE_EVENTS = _lifecycle_events()


class _HookRef:
    """A lifecycle event's own context, read like ``microscope.experiment``."""

    def __init__(self, payload: Dict[str, Any]) -> None:
        self.id = payload.get("experiment_id")
        self.name = payload.get("experiment_name")
        self.item_id = payload.get("item_id")
        self.item_name = payload.get("item_name")
        self.task_id = payload.get("task_id")
        self.task_name = payload.get("task_name")


class EventRecorder:
    """Records the event stream for one microscope connection: buffer, taps, file.

    Build it when the microscope connects and :meth:`close` it when it
    disconnects. ``responder`` is the UI's ``QtResponder``, the source of the
    prompt events; None where there is no UI. ``default_actor`` is who acted
    when the thread an event came from carries no mark (``fibsem.acting``):
    the operator in the app, where tasks and the agent mark theirs and nothing
    else calls the microscope; None, "not known", anywhere that is not so.
    """

    def __init__(
        self,
        microscope,
        responder=None,
        experiment_path: Optional[Path] = None,
        default_actor: Optional[str] = None,
        experiment: Any = None,
    ) -> None:
        self.session_id = uuid.uuid4().hex
        self._microscope = microscope
        self.default_actor = default_actor
        self.buffer = EventBuffer(stamp=self._stamp)
        self._proposals: Optional[ProposalTap] = None
        self.writer = EventFileWriter()
        self._disposers: List[Callable[[], None]] = [
            self.buffer.subscribe(self.writer.write)
        ]
        try:
            self._disposers += attach_microscope_taps(self.buffer, microscope)
            if responder is not None:
                self._disposers.append(
                    responder.add_question_observer(self.buffer.append)
                )
            # Registered for each run by the task manager that runs it
            # (FIB-1044), so this object is handed over again every time.
            self.lifecycle_hook = make_lifecycle_hook(self.buffer)
            self.set_experiment(experiment_path, experiment)
        except BaseException:
            # A recorder that could not be made must not leave its writer
            # running: nothing holds it to close it.
            self.close()
            raise
        # The first one open for a microscope is the one a run finds.
        if recorder_for(microscope) is None:
            _RECORDERS[id(microscope)] = self

    @property
    def microscope(self):
        return self._microscope

    def set_experiment(
        self, experiment_path: Optional[Path], experiment: Any = None
    ) -> None:
        """Record to this experiment's directory from now on (None: record nothing),
        and the questions and decisions on *experiment*, when given."""
        self.writer.set_path(
            Path(experiment_path) / EVENTS_FILENAME if experiment_path else None
        )
        if self._proposals is not None:
            self._proposals.dispose()
            self._proposals = None
        if experiment is not None:
            try:
                self._proposals = ProposalTap(self.buffer, experiment)
            except Exception:  # noqa: BLE001 - recording must not cost the experiment
                logging.debug(
                    "could not watch the experiment's decisions", exc_info=True
                )

    def close(self) -> None:
        """Detach every tap, write what is queued, and stop the writer."""
        if _RECORDERS.get(id(self._microscope)) is self:
            del _RECORDERS[id(self._microscope)]
        if self._proposals is not None:
            self._proposals.dispose()
            self._proposals = None
        for dispose in self._disposers:
            try:
                dispose()
            except Exception:  # noqa: BLE001
                pass
        self._disposers = []
        self.writer.close()

    def _stamp(self, kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Where in the run an event happened, read when it is emitted.

        From ``microscope.experiment``, the same record that stamps image
        metadata, so the stream and the images cannot disagree about which
        lamella or task they belong to -- except for the task lifecycle events,
        which carry their own: ``task_completed`` and ``task_failed`` fire after
        the task has cleared that record, and would otherwise say they belong to
        nothing.

        ``actor`` is who marked the thread the event was emitted on: a task, for
        everything it does, or the agent, for every request to the agent server.
        Not whatever task happens to be running -- ``microscope.experiment`` is
        the app's, not the thread's, so reading it would make a move the operator
        made during a run the task's. Unmarked, it is ``default_actor``. A task's
        own lifecycle event is the task's whichever thread fired it (a skip fires
        from the task manager), and an answer says who answered.
        """
        ref = getattr(self._microscope, "experiment", None)
        if kind in _LIFECYCLE_EVENTS and isinstance(payload, dict):
            ref = _HookRef(payload)
        experiment = item = task = None
        if ref is not None:
            if getattr(ref, "id", None) or getattr(ref, "name", None):
                experiment = {"id": ref.id, "name": ref.name}
            if getattr(ref, "item_id", None) or getattr(ref, "item_name", None):
                item = {"id": ref.item_id, "name": ref.item_name}
            if getattr(ref, "task_id", None) or getattr(ref, "task_name", None):
                task = {"id": ref.task_id, "name": ref.task_name}
        if kind == "prompt_answered" and isinstance(payload, dict):
            actor = payload.get("answered_by")
        elif kind in _LIFECYCLE_EVENTS and task is not None:
            actor = TASK
        else:
            actor = current_actor() or self.default_actor
        return {
            "session": self.session_id,
            "actor": actor,
            "experiment": experiment,
            "item": item,
            "task": task,
        }


# Who made a decision, as the stream names who acted: a person is the operator,
# a connected agent the agent, and a decision nobody made -- the producer
# confirming its own proposal, the record expiring or withdrawing one -- the
# task's.
_ACTOR_OF_AUTHOR = {"human": OPERATOR, "agent": AGENT, "auto": TASK}


class ProposalTap:
    """The questions a run asks, and every decision on a proposal, from the
    experiment's own ``asked`` and ``decided`` signals (FIB-1034).

    Every way a decision reaches the record fires ``decided``: the Review tab,
    the prompt bar and an agent over the server through ``Experiment.decide``,
    and the record itself expiring, withdrawing or noting an unasked one. So
    this is the one place they are recorded, whoever made them. Each decision
    is one ``proposal_decided`` event, with the values proposed and the values
    decided, whole: the reader compares them. A confirmation whose values the
    task fills in afterwards (a position "as it stands") is recorded again when
    they arrive, marked ``filled_in``.

    The signals name only the item and the task, so what is new is found by
    what has been seen: whatever the experiment holds when it is watched counts
    as seen. Nothing here raises: a decision that cannot be recorded has still
    been made.
    """

    def __init__(self, buffer: EventBuffer, experiment: Any) -> None:
        self._buffer = buffer
        self._experiment = experiment
        self._asked = set()
        # (proposal id, decision index) -> whether it was recorded with values
        self._decided: Dict[Tuple[str, int], bool] = {}
        for item in _items(experiment):
            for proposals in list(item.proposals.values()):
                for proposal in list(proposals):
                    if proposal.asking:
                        self._asked.add(proposal.id)
                    for index, decision in enumerate(list(proposal.decisions)):
                        self._decided[(proposal.id, index)] = bool(decision.values)
        experiment.asked.connect(self._on_asked)
        experiment.decided.connect(self._on_decided)

    def dispose(self) -> None:
        for signal, slot in (
            (self._experiment.asked, self._on_asked),
            (self._experiment.decided, self._on_decided),
        ):
            try:
                signal.disconnect(slot)
            except Exception:  # noqa: BLE001
                pass

    def _on_asked(self, item_id: str, task_name: str) -> None:
        try:
            item = self._experiment.get_item_by_id(item_id)
            for proposal in list(item.proposals.get(task_name) or []):
                if proposal.asking and proposal.id not in self._asked:
                    self._asked.add(proposal.id)
                    payload = _proposal_payload(item, task_name, proposal)
                    payload["message"] = proposal.provenance.get("message", "")
                    with acting(TASK):  # the task asked it
                        self._buffer.append("proposal_asked", payload)
        except Exception:  # noqa: BLE001 - recording must not matter
            logging.debug(f"could not record the {task_name} question", exc_info=True)

    def _on_decided(self, item_id: str, task_name: str) -> None:
        try:
            item = self._experiment.get_item_by_id(item_id)
            for proposal in list(item.proposals.get(task_name) or []):
                for index, decision in enumerate(list(proposal.decisions)):
                    seen = self._decided.get((proposal.id, index))
                    if seen is None or (seen is False and decision.values):
                        self._decided[(proposal.id, index)] = bool(decision.values)
                        self._record(item, task_name, proposal, index, decision, seen)
        except Exception:  # noqa: BLE001 - recording must not matter
            logging.debug(f"could not record a {task_name} decision", exc_info=True)

    def _record(self, item, task_name, proposal, index, decision, seen) -> None:
        record = decision.to_dict()
        payload = _proposal_payload(item, task_name, proposal)
        payload.update(
            decision=index,
            outcome=record["outcome"],
            author=record["author"],
            via=record["via"],
            reason=record["reason"],
            decided=record["values"],
        )
        if seen is False:  # recorded before its values were filled in
            payload["filled_in"] = True
        with acting(_ACTOR_OF_AUTHOR.get(decision.author.kind.value)):
            self._buffer.append("proposal_decided", payload)


def _items(experiment: Any) -> List[Any]:
    """The lamellae and grids: the items a proposal can sit on."""
    return list(experiment.positions) + list(getattr(experiment, "grids", None) or [])


def _proposal_payload(item: Any, task_name: str, proposal: Any) -> Dict[str, Any]:
    return {
        "item": {"id": item.id, "name": item.name},
        "task": task_name,
        "proposal_id": proposal.id,
        "kind": proposal.kind,
        "proposed": _encode_values(proposal.values),
    }
