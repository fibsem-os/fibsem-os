"""AgentContext: the read-side facade an embedded server sees the app through.

Deliberately not the ``AutoLamellaUI`` object, for the same reasons
``ScriptContext`` isn't: the window's ``experiment`` is rebound on load, its
``_task_manager`` is created per run and nulled in the worker's ``finally``,
and the GUI's internals are still moving. This facade resolves those
references **at call time, on every call**, and returns plain JSON-able data —
never live objects, never Qt, never a pointer into a widget.

Thread contract: every method is safe to call from a server thread. Reads are
point-in-time snapshots, not transactions — the workflow worker may be mutating
the experiment concurrently, and a snapshot that is a task older than the
instant it was taken is the accepted cost of never blocking anything. The one
hardware-adjacent method, :meth:`stage_position`, reads the microscope's cached
position only; nothing here ever issues a hardware call (the app must not be
made to poll the instrument on behalf of an observer).

Vocabulary: the surface speaks ``item_name`` throughout. ``lamella_name`` is
the deprecated alias scheduled to drop with the HookContext shims after v0.6
and never appears in these payloads.

The host is anything carrying the four attributes the facade resolves through
(``experiment``, ``microscope``, ``_task_manager``, ``is_workflow_running``) —
the real ``AutoLamellaUI`` in production, a plain holder of real domain objects
in tests.
"""

import math
from datetime import datetime
from typing import Any, Dict, List, Optional

from fibsem.applications.autolamella import task_outputs as _task_outputs
from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus

__all__ = ["AgentContext"]


def _json_safe(value: Any) -> Any:
    """Coerce one cell of a dataframe/record into something JSON can carry."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item") and not isinstance(value, str):
        # numpy scalars → native python
        try:
            return _json_safe(value.item())
        except (AttributeError, ValueError):
            pass
    if hasattr(value, "isoformat"):
        # pandas.Timestamp and friends
        return value.isoformat()
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value
    return str(value)


def _records(df) -> List[Dict[str, Any]]:
    return [
        {key: _json_safe(val) for key, val in record.items()}
        for record in df.to_dict(orient="records")
    ]


class AgentContext:
    """Read-only facade over a running (or resting) AutoLamella session."""

    def __init__(self, host, event_buffer=None):
        self._host = host
        self._event_buffer = event_buffer

    # --- call-time resolution: never cache what the app rebinds ---------------

    @property
    def _experiment(self):
        return getattr(self._host, "experiment", None)

    @property
    def _manager(self):
        return getattr(self._host, "_task_manager", None)

    @property
    def _microscope(self):
        return getattr(self._host, "microscope", None)

    # --- status ---------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """What the app is doing right now, at a glance."""
        experiment = self._experiment
        running = bool(getattr(self._host, "is_workflow_running", False))
        payload: Dict[str, Any] = {
            "microscope_connected": self._microscope is not None,
            "experiment": None,
            "workflow": {
                "running": running,
                "current_task": None,
                "current_item": None,
                "queue_total": None,
                "queue_version": None,
            },
        }
        if experiment is not None:
            payload["experiment"] = {
                "name": experiment.name,
                "path": str(experiment.path),
                "num_items": len(experiment.positions),
            }
        manager = self._manager
        if manager is not None:
            items = manager.queue.items  # thread-safe copy under the queue's lock
            payload["workflow"]["queue_total"] = len(items)
            payload["workflow"]["queue_version"] = manager.queue.version
            active = next(
                (i for i in items if i.status is AutoLamellaTaskStatus.InProgress),
                None,
            )
            if active is not None:
                payload["workflow"]["current_task"] = active.task_name
                payload["workflow"]["current_item"] = active.lamella_name
        return payload

    def queue(self) -> Dict[str, Any]:
        """The live work queue: id-anchored items plus the mutation version."""
        manager = self._manager
        if manager is None:
            return {"available": False, "items": [], "version": None}
        items = manager.queue.items
        return {
            "available": True,
            "version": manager.queue.version,
            "items": [
                {
                    "id": item.id,
                    "item_name": item.lamella_name,
                    "task_name": item.task_name,
                    "status": item.status.name,
                }
                for item in items
            ],
        }

    # --- experiment record ----------------------------------------------------

    def experiment_summary(self) -> Dict[str, Any]:
        experiment = self._experiment
        if experiment is None:
            return {"available": False, "items": []}
        return {
            "available": True,
            "items": _records(experiment.experiment_summary_dataframe()),
        }

    def task_history(self) -> Dict[str, Any]:
        experiment = self._experiment
        if experiment is None:
            return {"available": False, "items": []}
        return {
            "available": True,
            "items": _records(experiment.task_history_dataframe()),
        }

    def run_summary(self) -> Dict[str, Any]:
        """The most recent run's outcome table, if a run has happened."""
        manager = self._manager
        if manager is not None:
            return {
                "available": True,
                "items": _records(manager.build_run_summary_dataframe()),
            }
        last = getattr(self._host, "_last_run_summary", None)
        if last is not None:
            return {"available": True, "items": _records(last)}
        return {"available": False, "items": []}

    def protocol(self) -> Dict[str, Any]:
        """The workflow definition with live supervision flags and schedules."""
        experiment = self._experiment
        task_protocol = (
            getattr(experiment, "task_protocol", None) if experiment else None
        )
        if task_protocol is None:
            return {"available": False, "tasks": []}
        config = task_protocol.workflow_config
        return {
            "available": True,
            "name": config.name,
            "tasks": [
                {
                    "name": task.name,
                    "supervise": task.supervise,
                    "required": task.required,
                    "requires": list(task.requires),
                    "scheduled_at": task.scheduled_at.isoformat()
                    if task.scheduled_at is not None
                    else None,
                }
                for task in config.tasks
            ],
        }

    def task_outputs(self, item_name: str) -> Dict[str, Any]:
        """The files an item's completed tasks produced, existence-checked."""
        experiment = self._experiment
        if experiment is None:
            return {"available": False, "item_name": item_name}
        lamella = experiment.get_lamella_by_name(item_name)
        if lamella is None:
            return {
                "available": False,
                "item_name": item_name,
                "error": f"No item named {item_name!r} in this experiment.",
            }
        history = list(lamella.task_history)
        return {
            "available": True,
            "item_name": item_name,
            "completed_tasks": [state.name for state in history],
            "final_reference_images": [
                str(p) for p in _task_outputs.final_reference_images(lamella, *history)
            ],
            "fluorescence_images": [
                str(p) for p in _task_outputs.fluorescence_images(lamella, *history)
            ],
        }

    # --- instrument-adjacent (cached only, never a hardware call) --------------

    def stage_position(self) -> Dict[str, Any]:
        """The cached stage position. None until something has read the stage.

        Deliberately reads the cache, not ``get_stage_position()`` — an observer
        must never make the app poll hardware (and on Tescan a concurrent read
        is a socket transaction the workflow may be mid-way through).
        """
        microscope = self._microscope
        cached = getattr(microscope, "_stage_position", None) if microscope else None
        if cached is None:
            return {"available": False, "position": None}
        return {"available": True, "position": cached.to_dict()}

    def display_images(self) -> Dict[str, Any]:
        """The SEM/FIB images the GUI is displaying right now, as previews.

        This is the display cache, not an acquisition: nothing touches
        hardware, and the workflow's post-mill images appear here the moment
        the GUI shows them — long before task completion writes them to disk —
        which is what makes inline inspection possible.

        Thread note: grabs the widget's current image references (replaced
        whole on acquisition, never mutated in place) and encodes previews on
        the server thread; the GUI thread is neither entered nor blocked.
        """
        widget = getattr(self._host, "image_widget", None)
        if widget is None:
            return {"available": False, "sem": None, "fib": None}
        from fibsem.applications.autolamella.server.prompts import _preview_b64

        payload: Dict[str, Any] = {"available": True}
        for key, image in (
            ("sem", getattr(widget, "eb_image", None)),
            ("fib", getattr(widget, "ib_image", None)),
        ):
            entry = _preview_b64(getattr(image, "data", None))
            if entry is not None:
                entry["acquired_at"] = self._acquired_at(image)
            payload[key] = entry
        return payload

    @staticmethod
    def _acquired_at(image) -> Optional[str]:
        try:
            return image.metadata.acquisition_date.isoformat()
        except Exception:
            return None

    # --- workflow control -------------------------------------------------------

    def stop_workflow(self) -> Dict[str, Any]:
        """Stop the running workflow — the same path as the GUI's Stop button.

        Everything on that path is thread-safe from here: the manager's stop is
        an event, and halting the mill/burn is an event plus the same hardware
        stop call the core server's ``stop_milling`` route already issues from
        this thread. Stopping is the safety action, so like ``stop_milling`` it
        rides the read scope and never waits on the command lock.
        """
        host = self._host
        if not hasattr(host, "stop_task_workflow"):
            return {"available": False, "stopped": False}
        if not bool(getattr(host, "is_workflow_running", False)):
            return {
                "available": True,
                "stopped": False,
                "reason": "no workflow is running",
            }
        host.stop_task_workflow()
        return {"available": True, "stopped": True}

    def set_supervision(self, task_name: str, supervise: bool) -> Dict[str, Any]:
        """Set whether ``task_name`` asks for supervision, in the live protocol.

        The workflow reads this at every decision point (never a snapshot), so
        a change takes effect at the next prompt-or-proceed choice — exactly
        the mid-run behaviour the GUI's supervised/automated toggle has. The
        GUI's own indicators refresh on the next task transition rather than
        instantly.
        """
        experiment = self._experiment
        protocol = getattr(experiment, "task_protocol", None) if experiment else None
        config = getattr(protocol, "workflow_config", None) if protocol else None
        if config is None:
            return {"available": False, "applied": False}
        for task in config.tasks:
            if task.name == task_name:
                task.supervise = bool(supervise)
                return {
                    "available": True,
                    "applied": True,
                    "task_name": task_name,
                    "supervise": bool(supervise),
                }
        return {
            "available": True,
            "applied": False,
            "error": f"No task named {task_name!r} in the protocol.",
            "task_names": [t.name for t in config.tasks],
        }

    def requeue_task(
        self, item_name: str, task_name: str, front: bool = False
    ) -> Dict[str, Any]:
        """Queue ``task_name`` for ``item_name`` (again) — the batch-review verb.

        The queue exists only while a workflow runs, so a re-run is queued into
        the live run: "run 03 again" while the batch continues. Adding a
        duplicate of a completed pair is the queue's own re-run mechanism; the
        item lands at the back (or the front with ``front=True``).
        """
        manager = self._manager
        if manager is None:
            return {
                "available": True,
                "queued": False,
                "reason": "no workflow is running — the queue exists only during a run",
            }
        experiment = self._experiment
        if experiment is None or experiment.get_lamella_by_name(item_name) is None:
            return {
                "available": True,
                "queued": False,
                "reason": f"No item named {item_name!r} in this experiment.",
            }
        protocol = getattr(experiment, "task_protocol", None)
        config = getattr(protocol, "workflow_config", None) if protocol else None
        known = [t.name for t in config.tasks] if config is not None else []
        if known and task_name not in known:
            return {
                "available": True,
                "queued": False,
                "reason": f"No task named {task_name!r} in the protocol.",
                "task_names": known,
            }
        item = manager.queue.add(item_name, task_name, front=bool(front))
        if item is None:
            return {"available": True, "queued": False, "reason": "add refused"}
        manager.notify_queue_changed()
        return {
            "available": True,
            "queued": True,
            "item_id": item.id,
            "item_name": item_name,
            "task_name": task_name,
            "queue_version": manager.queue.version,
        }

    # --- supervision prompts (FIB-851) ------------------------------------------

    @property
    def _responder(self):
        return getattr(self._host, "ui_responder", None)

    def pending_prompt(self) -> Dict[str, Any]:
        """The supervision question currently awaiting an answer, if any.

        The serialized request carries everything needed to answer it — the
        FIB-826 contract — including context images as agent-sized previews.
        """
        responder = self._responder
        if responder is None:
            return {"available": False, "pending": None}
        request, nonce = responder.pending_question_and_nonce()
        if request is None:
            return {"available": True, "pending": None}
        from fibsem.applications.autolamella.server.prompts import serialize_request

        payload = serialize_request(request)
        payload["nonce"] = nonce
        current = self._peek_current(responder, payload["type"], nonce)
        if current is not None:
            payload["current"] = current
        return {"available": True, "pending": payload}

    def _peek_current(
        self, responder, request_type: str, nonce: int
    ) -> Optional[Dict[str, Any]]:
        """The live half of the question — where the marker IS, not where it
        started.

        PickPOI and EditAlignmentArea are answered from widget state the
        operator may still be adjusting; the frozen request only shows the
        starting value. The peek marshals to the GUI thread, so it is skipped
        entirely for question types with no live half, and degrades to absent
        (never an error — the question itself must still be visible) if the GUI
        is too busy to answer within 2 s or the prompt changed mid-peek.
        """
        if request_type not in ("PickPOI", "EditAlignmentArea"):
            return None
        from fibsem.applications.autolamella.server.events import to_plain

        try:
            peeked_nonce, value = responder.peek_live_answer().result(timeout=2.0)
        except Exception:
            return None
        if peeked_nonce != nonce or value is None:
            return None
        try:
            return to_plain(value.to_dict())
        except Exception:
            return None

    def answer_prompt(
        self, response: bool, nonce: int, timeout: float = 10.0
    ) -> Dict[str, Any]:
        """Answer the pending question as the matching button click would.

        Routed through the responder's own GUI-thread path, so agent and human
        answers share one first-writer-wins mechanism; ``applied`` is False when
        nothing was pending or a human answered first. The ``nonce`` (from
        :meth:`pending_prompt`) names the question being answered: if that
        posting is gone — answered, withdrawn, or replaced — the result is
        ``stale`` and nothing was clicked.
        """
        responder = self._responder
        if responder is None:
            return {"available": False, "applied": False, "stale": False}
        from fibsem.applications.autolamella.workflows.interaction import (
            StalePromptError,
        )

        outcome = responder.submit_answer(bool(response), nonce=int(nonce))
        try:
            applied = bool(outcome.result(timeout=timeout))
        except StalePromptError:
            return {"available": True, "applied": False, "stale": True}
        return {"available": True, "applied": applied, "stale": False}

    # --- events -----------------------------------------------------------------

    def events(self, since: int = 0, timeout: float = 0.0) -> Dict[str, Any]:
        """Events after ``since``, parking up to ``timeout`` seconds for news.

        Unavailable (rather than empty) when the hosting wired no buffer, so a
        client can tell "no events yet" from "this server has no event stream".
        """
        from fibsem.applications.autolamella.server.events import (
            buffer_or_unavailable,
        )

        return buffer_or_unavailable(self._event_buffer, since=since, timeout=timeout)

    # --- discovery --------------------------------------------------------------

    def recent_experiments(self) -> List[Dict[str, Any]]:
        """The recents list, via the never-raising peek layer."""
        from fibsem.config import get_recent_experiments

        return [
            {
                "name": summary.name,
                "path": summary.path,
                "created_at": summary.created_at,
                "num_items": summary.num_lamella,
                "available": summary.available,
                "instrument_model": summary.instrument_model,
                "software_version": summary.software_version,
            }
            for summary in get_recent_experiments()
        ]
