"""AppContext: what every app's context serves the /app core reads from.

An app's context is the facade an embedded server sees the app through: it
resolves the session's experiment, runner and microscope at call time, on
every call, and returns plain JSON-able data, never live objects or Qt. Reads
are point-in-time snapshots, safe from a server thread. An app subclasses it
and adds what its own experiment and protocol need.
"""

import logging
import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from fibsem.acting import Actor
from fibsem.geometry.saved_images import image_point_to_stage_position, stage_projector
from fibsem.structures import BeamType
from fibsem.util.records import dataframe_records
from fibsem.util.timestamps import to_iso, wall_time_of
from fibsem.workflows.actions import NOTE, note_record
from fibsem.workflows.evidence import EVIDENCE, frame_path
from fibsem.workflows.experiments import (
    default_experiment_directory,
    discover_experiments,
    filter_experiments,
)
from fibsem.workflows.runner import ITEMS, RunRequest
from fibsem.workflows.session import CONFIG, PROTOCOL
from fibsem.workflows.status import TaskStatus

__all__ = ["AppContext", "content_hash", "find_item", "item_pose_positions"]


def content_hash(data) -> str:
    """A short hash naming exactly this content: a document's version."""
    import hashlib
    import json

    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def item_pose_positions(item) -> Dict[str, Any]:
    """Where each of an item's poses puts the stage, by pose name (None where a
    pose has no position): the item document's ``poses``."""
    poses = {}
    for pose_name, pose in dict(item.poses).items():
        position = getattr(pose, "stage_position", None)
        poses[pose_name] = position.to_dict() if position is not None else None
    return poses


def find_item(experiment, item_name: str):
    """The item named *item_name*, and None; or None and why not, in the words
    every route refuses an unknown item with, and the names there are. Each
    route puts the refusal in its own answer's shape."""
    positions = experiment.positions if experiment is not None else []
    item = next((p for p in positions if p.name == item_name), None)
    if item is not None:
        return item, None
    return None, {
        "error": f"No item named {item_name!r} in this experiment.",
        "item_names": [p.name for p in positions],
    }


class AppContext:
    """The reads every app serves, over its ``AppSession``."""

    # The app the operator works in, as the App Server names it to the agent.
    app_name = "fibsemOS"

    def __init__(self, session, event_buffer=None):
        self._session = session
        self._event_buffer = event_buffer

    # --- call-time resolution: never cache what the app rebinds ---------------

    @property
    def _experiment(self):
        return self._session.experiment

    @property
    def experiment_path(self) -> Optional[str]:
        """The open experiment's folder, or None: where an agent may save."""
        experiment = self._experiment
        return str(experiment.path) if experiment is not None else None

    def experiment_root(self, experiment: Optional[str] = None) -> Optional[str]:
        """The folder an experiment's files are read from: the open one's, or
        *experiment*, a past one by its path from the recents or the
        experiments folder (its folder or experiment.yaml). Raises
        ``FileRefusal`` for any other."""
        from fibsem.server.files import FileRefusal

        if experiment is None:
            return self.experiment_path
        wanted = os.path.normcase(os.path.realpath(experiment))
        for entry in self.recent_experiments():
            yaml = os.path.normcase(os.path.realpath(entry["path"]))
            if wanted in (yaml, os.path.dirname(yaml)):
                return os.path.dirname(entry["path"])
        found = self._under_experiments_folder(experiment)
        if found is not None and os.path.basename(found) == "experiment.yaml":
            found = os.path.dirname(found)
        if found is None or not os.path.isfile(os.path.join(found, "experiment.yaml")):
            raise FileRefusal(
                404,
                "not_found",
                f"No experiment at {experiment}: give a path from "
                "experiment_list_recent or experiment_list_all.",
            )
        return found

    def picture_sources(self) -> Optional[Any]:
        """What the agent's pictures read from the experiment (its items'
        images, configs and poses; see ``ai.agent.pictures``), or None."""
        return None

    def recording_experiment(self):
        """The experiment what the agent asks is recorded on: the open one, or
        the session's plain one. Raises OSError when it cannot be made."""
        return self._session.recording_experiment()

    def image_metadata(self, path: str) -> Dict[str, Any]:
        """A saved beam image's full metadata, as its file records it, for a
        path inside the open experiment only (the agent's drill-down behind
        the short block every image answer carries)."""
        from fibsem.structures import FibsemImage

        target, refusal = self._experiment_image(path)
        if refusal:
            return refusal
        image = FibsemImage.load(target)
        if image.metadata is None:
            return {
                "error_type": "not_found",
                "message": f"{path} carries no beam image metadata.",
            }
        return {"path": path, "metadata": image.metadata.to_dict()}

    def _experiment_image(self, path: str):
        """``(real path, None)`` for a file inside the open experiment, else
        ``(None, refusal)``: the agent reads saved images there only."""
        import os

        from fibsem.server.files import FileRefusal, resolve

        try:
            target = resolve(self.experiment_path, path)
        except FileRefusal as refusal:
            return None, refusal.detail
        if not os.path.isfile(target):
            return None, {"error_type": "not_found", "message": f"{path} is a folder."}
        return target, None

    def image_stage_position(self, path: str, px=None, fraction=None) -> Dict[str, Any]:
        """The stage position under a point of any saved image in the
        experiment, beam or fluorescence, through the same projection the
        overview canvases use, built from the image's own metadata: it converts
        as the image was taken, wherever the stage is now. ``px`` is [x, y] in
        source pixels (origin top-left, +y down); ``fraction`` is [x, y] from 0
        to 1 across the image."""
        target, refusal = self._experiment_image(path)
        if refusal:
            return refusal
        try:
            position = image_point_to_stage_position(target, px=px, fraction=fraction)
        except ValueError as exc:
            return {"error_type": "invalid", "message": str(exc)}
        return {"path": path, "stage_position": position.to_dict()}

    def image_pixels(
        self, path: str, stage_positions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Where each stage position falls on a saved image in the experiment,
        in source pixels, with the image's scale block. A position the image
        cannot place (no geometry for it) answers its reason instead."""
        from fibsem.structures import FibsemStagePosition

        target, refusal = self._experiment_image(path)
        if refusal:
            return refusal
        try:
            positions = [FibsemStagePosition.from_dict(p) for p in stage_positions]
        except (TypeError, KeyError, AttributeError) as exc:
            return {"error_type": "invalid", "message": f"Not a stage position: {exc}"}
        block, project = stage_projector(target)
        points = []
        for position in positions:
            try:
                point = project(position)
            except Exception as exc:  # noqa: BLE001 - per-position answer
                points.append({"px": None, "reason": str(exc)})
                continue
            x, y = float(point.x), float(point.y)
            points.append(
                {
                    "px": [x, y],
                    "inside": 0 <= x <= block["width"] and 0 <= y <= block["height"],
                }
            )
        return {"path": path, "image": block, "points": points}

    @property
    def _runner(self):
        return self._session.runner

    @property
    def _manager(self):
        return self._runner.manager

    @property
    def _running(self) -> bool:
        return self._runner.is_running

    @property
    def _microscope(self):
        return self._session.microscope

    # --- status ---------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """What the app is doing right now, at a glance."""
        experiment = self._experiment
        running = self._running
        payload: Dict[str, Any] = {
            "microscope_connected": self._microscope is not None,
            "experiment": None,
            "workflow": {
                "running": running,
                "current_task": None,
                "current_item": None,
                "queue_total": None,
                "queue_version": None,
                "asking": [],
            },
        }
        if experiment is not None:
            payload["experiment"] = {
                "name": experiment.name,
                "path": str(experiment.path),
                "num_items": len(experiment.positions),
            }
            # The proposals a task is stopped on, for an agent that has just
            # connected: their events are behind its cursor.
            payload["workflow"]["asking"] = [
                {
                    "item_id": item.id,
                    "item_name": item.name,
                    "task_name": task_name,
                    "proposal_id": proposal.id,
                }
                for item, task_name, proposal in experiment.pending_proposals()
                if proposal.asking
            ]
        manager = self._manager
        if manager is not None:
            items = manager.queue.items  # thread-safe copy under the queue's lock
            payload["workflow"]["queue_total"] = len(items)
            payload["workflow"]["queue_version"] = manager.queue.version
            active = next(
                (i for i in items if i.status is TaskStatus.InProgress),
                None,
            )
            if active is not None:
                payload["workflow"]["current_task"] = active.task_name
                payload["workflow"]["current_item"] = active.item_name
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
                    "item_name": item.item_name,
                    "task_name": item.task_name,
                    "status": item.status.name,
                }
                for item in items
            ],
        }

    def run_summary(self) -> Dict[str, Any]:
        """The most recent run's outcome table, if a run has happened."""
        manager = self._manager
        if manager is not None:
            return {
                "available": True,
                "items": dataframe_records(manager.build_run_summary_dataframe()),
            }
        runner = self._runner
        last = runner.last_summary if runner is not None else None
        if last is not None:
            return {"available": True, "items": dataframe_records(last)}
        return {"available": False, "items": []}

    def log(
        self,
        since: Optional[str] = None,
        until: Optional[str] = None,
        task_id: Optional[str] = None,
        level: Optional[str] = None,
        lines: int = 200,
    ) -> Dict[str, Any]:
        """The session's log in a time window: ``since``/``until``, or the run
        ``task_id`` (from its start to its end, a few seconds either side), or
        the tail when neither is given. See ``log_tail`` for what is served."""
        from pathlib import Path

        from fibsem.server import log_tail

        experiment = self._experiment
        if experiment is None:
            return {"available": False, "records": []}
        if level is not None:
            level = level.upper()
            if level not in log_tail.LEVELS:
                return {
                    "available": True,
                    "error_type": "invalid",
                    "error": f"No log level {level!r}.",
                    "levels": list(log_tail.LEVELS),
                }
        window = {}
        for name, value in (("since", since), ("until", until)):
            if value is None:
                continue
            window[name] = wall_time_of(value)
            if window[name] is None:
                return {
                    "available": True,
                    "error_type": "invalid",
                    "error": f"{name}={value!r} is not a time.",
                }
        if task_id is not None:
            if window:
                return {
                    "available": True,
                    "error_type": "invalid",
                    "error": "Give task_id or since/until, not both.",
                }
            found = log_tail.run_window(experiment.path, task_id)
            if found is None:
                return {
                    "available": True,
                    "error_type": "not_found",
                    "error": f"No run {task_id!r} in this experiment's record.",
                }
            window = {"since": found[0], "until": found[1]}
            # A run's window at DEBUG is mostly noise, and the cap keeps its
            # end: the start of the run would be the part dropped.
            level = level or "INFO"
        path = Path(experiment.path) / log_tail.LOG_FILENAME
        if not path.is_file():
            return {"available": False, "records": []}
        result = log_tail.read_log(path, level=level, lines=lines, **window)
        start, end = window.get("since"), window.get("until")
        return {
            "available": True,
            "since": start.isoformat() if start is not None else None,
            "until": end.isoformat() if end is not None else None,
            "level": level or "DEBUG",
            **result,
        }

    # --- instrument-adjacent (cached only, never a hardware call) --------------

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
        from fibsem.server.image_payloads import image_payload

        images = self._latest_images
        if images is None:
            return {"available": False, "sem": None, "fib": None}
        return {
            "available": True,
            "sem": image_payload(images.get(BeamType.ELECTRON)),
            "fib": image_payload(images.get(BeamType.ION)),
        }

    @property
    def _latest_images(self):
        """The session's latest image of each beam, window or not; None
        until a microscope connects."""
        if self._microscope is None:
            return None
        return self._session.images

    # --- events -----------------------------------------------------------------

    def events(self, since: int = 0, timeout: float = 0.0) -> Dict[str, Any]:
        """Events after ``since``, parking up to ``timeout`` seconds for news.

        Unavailable (rather than empty) when the hosting wired no buffer, so a
        client can tell "no events yet" from "this server has no event stream".
        """
        from fibsem.workflows.events import (
            buffer_or_unavailable,
        )

        return buffer_or_unavailable(self._event_buffer, since=since, timeout=timeout)

    def event_history(
        self, before: Optional[int] = None, limit: int = 500
    ) -> Dict[str, Any]:
        """The experiment's recorded events from before the live buffer, newest
        page first: everything ``events`` no longer holds, read from the
        experiment's events file, without the live progress records.

        ``start`` is the file offset to pass back as ``before`` for the page
        before this one; 0 when this page reaches the file's start.
        """
        from pathlib import Path

        from fibsem.workflows.event_recording import (
            EVENTS_FILENAME,
            LIVE_KINDS,
            read_events_before,
        )

        experiment = self._experiment
        if experiment is None:
            return {"available": False, "events": [], "start": 0}
        path = Path(experiment.path) / EVENTS_FILENAME
        if not path.is_file():
            return {"available": True, "events": [], "start": 0}
        # What the buffer still holds, `events` serves; leave it out here.
        held = None
        if self._event_buffer is not None:
            first = self._event_buffer.events_since(0).get("events") or []
            if first:
                held = (first[0].get("session"), first[0].get("seq"))

        def keep(record: Dict[str, Any]) -> bool:
            if held is None or record.get("session") != held[0]:
                return True
            seq = record.get("seq")
            return not isinstance(seq, int) or seq < held[1]

        events, start = read_events_before(
            path,
            before=before,
            limit=max(1, min(limit, 2000)),
            skip_kinds=LIVE_KINDS,
            keep=keep,
        )
        return {"available": True, "events": events, "start": start}

    # --- discovery --------------------------------------------------------------

    def _experiments_folder(self) -> str:
        """Where experiments are created and found: the session's folder, else
        the default experiments folder."""
        return self._session.experiments_folder or default_experiment_directory()

    def _under_experiments_folder(self, path: str) -> Optional[str]:
        """*path* (relative to the experiments folder, or absolute) when it is
        inside that folder, links followed; else None."""
        folder = os.path.realpath(self._experiments_folder())
        target = os.path.realpath(os.path.join(folder, path))
        try:
            inside = os.path.commonpath([target, folder]) == folder
        except ValueError:  # another drive
            return None
        return target if inside and target != folder else None

    def experiment_entry(self, summary) -> Dict[str, Any]:
        """One experiment in a listing, from its ``ExperimentSummary``.
        ``relative_path`` is its folder relative to the experiments folder, as
        loads and reads take it; None for one outside that folder."""
        folder = os.path.dirname(summary.path)
        relative = None
        if self._under_experiments_folder(folder) is not None:
            relative = os.path.relpath(
                os.path.realpath(folder), os.path.realpath(self._experiments_folder())
            ).replace(os.sep, "/")
        return {
            "name": summary.name,
            "path": summary.path,
            "relative_path": relative,
            "app": summary.app,
            "created_at": to_iso(summary.created_at),
            "num_items": summary.num_lamella,
            "available": summary.available,
            "instrument_model": summary.instrument_model,
            "instrument_serial": summary.instrument_serial,
            "operator": summary.operator,
            "software_version": summary.software_version,
        }

    def recent_experiments(self) -> List[Dict[str, Any]]:
        """The recents list, via the never-raising peek layer."""
        from fibsem.config import get_recent_experiments

        return [self.experiment_entry(s) for s in get_recent_experiments()]

    def saved_experiments(
        self,
        instrument: Optional[str] = None,
        operator: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Every experiment under the experiments folder, newest first, narrowed
        by `filter_experiments` (each filter left None does not narrow)."""
        folder = self._experiments_folder()
        found = filter_experiments(
            discover_experiments(folder),
            instrument=instrument,
            operator=operator,
            since=since,
            until=until,
        )
        return {"folder": folder, "items": [self.experiment_entry(s) for s in found]}

    def _not_while_running(self, what: str) -> Optional[Dict[str, Any]]:
        """The refusal while a workflow runs (*what* to do when it has
        stopped), or None."""
        if self._running:
            return {
                "error_type": "task_running",
                "error": f"A workflow is running; {what} when it has stopped.",
            }
        return None

    def _busy(self, why: str) -> Optional[Dict[str, Any]]:
        """A run_edits refusal as the routes answer it, or None."""
        return {"error_type": "task_running", "error": why} if why else None

    def _no_microscope(self) -> Optional[Dict[str, Any]]:
        if self._microscope is None:
            return {"error_type": "not_found", "error": "No microscope is connected."}
        return None

    def _session_refusal(self) -> Optional[Dict[str, Any]]:
        """Why the open experiment cannot be replaced now, or None."""
        return self._not_while_running("change experiments") or self._no_microscope()

    def _agent_call(self, call, timeout: float = 30.0):
        """``call()`` where the session edits the experiment."""
        return self._session.run_edit(call).result(timeout=timeout)

    def proposals(self) -> Dict[str, Any]:
        """Every pending proposal, derived from the experiment, with the
        reference image its values sit on as an agent-sized preview."""
        experiment = self._experiment
        if experiment is None:
            return {"available": False, "proposals": []}
        from fibsem.server.image_payloads import image_payload
        from fibsem.workflows.events import to_plain
        from fibsem.workflows.reference_images import (
            load_reference_image,
            preview_of,
        )

        def describe(item, task_name, proposal):
            doc = to_plain(proposal.to_dict())
            doc.update(
                {
                    "item_id": item.id,
                    "item_name": item.name,
                    "task_name": task_name,
                    # what this is; proposal_decide passes it back. The run
                    # is here as a fact about it, and still names it for a
                    # caller that has not learned the id.
                    "proposal_id": proposal.id,
                    "task_id": proposal.task_id,
                }
            )
            image = load_reference_image(experiment, item, proposal)
            # a fluorescence result is shown as its channel composite
            preview = preview_of(image) if image is not None else None
            doc["reference_image"] = (
                image_payload(preview) if preview is not None else None
            )
            return doc

        proposals = [describe(*entry) for entry in experiment.pending_proposals()]
        return {"available": True, "proposals": proposals}

    def decide(
        self,
        item_id: str,
        task_name: str,
        outcome: str,
        values: Optional[Dict[str, Any]] = None,
        reason: str = "",
        actor: str = "",
        task_id: str = "",
        proposal_id: str = "",
        action: str = "",
        patch: Optional[Dict[str, Any]] = None,
        evidence: str = "",
    ) -> Dict[str, Any]:
        """Decide a pending proposal, exactly as the operator's answer is: the same
        Experiment.decide, on the main thread, blocking until applied. The
        actor is recorded as the agent (``actor`` names its model), never as
        the operator.

        ``action`` is what a loop's pass is decided as (run, edit, skip,
        accept). ``patch`` sets fields of its value by dotted path, on the
        value given or else the one proposed: a milling config's
        ``stages.Polish 1.pattern.depth``.

        ``proposal_id`` is the proposal the agent was shown (from
        proposal_list), and what the decision is checked against: a task
        may ask several proposals in one run, and only the id tells the one
        that was shown from the one that replaced it. ``task_id``, the run, is
        accepted in its place from a caller that has not learned the id, and
        is checked the way it always was. One of them is needed; a stale one
        is refused."""
        experiment = self._experiment
        if experiment is None:
            return {"available": False, "applied": False, "reason": "No experiment."}
        from fibsem.workflows.proposals import (
            Decision,
            DecisionOutcome,
            ValueRefused,
            _decode_values,
            patch_values,
        )

        try:
            decided = DecisionOutcome[outcome]
        except KeyError:
            return {
                "available": True,
                "applied": False,
                "invalid_value": f"outcome must be Confirmed or Rejected, not {outcome!r}",
            }
        try:
            decoded = _decode_values(values or {})
        except Exception as exc:  # noqa: BLE001 - any bad document is refused
            return {
                "available": True,
                "applied": False,
                "invalid_value": f"values could not be read: {exc}",
            }
        if patch:
            item = experiment.get_item_by_id(item_id)
            shown = next(
                (
                    p
                    for p in (item.current_proposals(task_name) if item else [])
                    if p.id == proposal_id or (not proposal_id and p.pending)
                ),
                None,
            )
            if shown is None:
                return {
                    "available": True,
                    "applied": False,
                    "invalid_value": "patch needs the proposal_id of a current proposal.",
                }
            try:
                decoded = patch_values(shown.kind, decoded or dict(shown.values), patch)
            except ValueRefused as exc:
                return {"available": True, "applied": False, "invalid_value": str(exc)}
        decision = Decision(
            outcome=decided,
            actor=Actor.agent(actor or "remote"),
            values=decoded,
            reason=reason,
            via="server",
            task_id=str(task_id or ""),
            proposal_id=str(proposal_id or ""),
            action=str(action or ""),
            evidence=evidence,
        )
        result = experiment.decide(item_id, task_name, decision)
        doc = result.to_dict()
        doc["available"] = True
        return doc

    def suggest(
        self,
        item_id: str,
        task_name: str,
        proposal_id: str,
        reason: str,
        outcome: str = "Confirmed",
        values: Optional[Dict[str, Any]] = None,
        patch: Optional[Dict[str, Any]] = None,
        action: str = "",
        evidence: Optional[Dict[str, Any]] = None,
        actor: str = "",
    ) -> Dict[str, Any]:
        """Suggest a decision on a pending proposal, for the operator to
        approve or decline in the window. Nothing is decided. Checked as the
        decision would be, so one that could not be approved is refused now."""
        experiment = self._experiment
        if experiment is None:
            return {"available": False, "applied": False, "reason": "No experiment."}
        from fibsem.workflows.proposals import (
            DecisionOutcome,
            Suggestion,
            _decode_values,
        )

        if outcome not in ("Confirmed", "Rejected"):
            return {
                "available": True,
                "applied": False,
                "error_type": "invalid_value",
                "reason": f"outcome is Confirmed or Rejected, not {outcome!r}",
            }
        try:
            decoded = _decode_values(values or {})
        except Exception as exc:  # noqa: BLE001 - any bad document is refused
            return {
                "available": True,
                "applied": False,
                "error_type": "invalid_value",
                "reason": f"values could not be read: {exc}",
            }
        # The beam image of the cited evidence, to draw the value on.
        image = frame_path(evidence)
        if image:
            root = self.experiment_path
            image = os.path.relpath(image, root) if root else image
        suggestion = Suggestion(
            actor=Actor.agent(actor or "remote"),
            reason=reason,
            outcome=DecisionOutcome[outcome],
            values=decoded,
            patch=dict(patch or {}),
            action=action,
            image=image or "",
            evidence=(evidence or {}).get("evidence_id", ""),
        )
        result = experiment.suggest(item_id, task_name, proposal_id, suggestion)
        if not result.applied:
            return {**result.to_dict(), "available": True}
        # Nothing was decided, so nothing ran or synced.
        doc = {"applied": True, "available": True, "suggestion_id": suggestion.id}
        if result.save_error:
            doc["save_error"] = result.save_error
        return doc

    def add_note(
        self,
        text: str,
        item_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        evidence: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Put an agent observation on the record, or its answer to the
        operator's message (``reply_to``, the message's id, FIB-1423), citing
        the *evidence* (its id) it rests on.

        The agent's judgments otherwise live only in its chat: this writes
        them where the run's story is told — the event stream (live consumers:
        timeline, dashboard, other agents) and the experiment log (durable).
        Notes are observations, not actions: nothing changes state.
        """
        text = str(text).strip()
        if not text:
            return {"available": True, "recorded": False, "error": "empty note"}
        item_id = None
        if item_name is not None:
            item, refusal = find_item(self._experiment, item_name)
            if refusal:
                return {"available": True, "recorded": False, **refusal}
            item_id = item.id
        logging.info(
            "agent note%s: %s",
            f" [{item_name}]" if item_name else "",
            text,
        )
        # With the item's id, so the record files the note under the item
        # it is about, not whichever one is running (FIB-1328).
        self.record_event(
            NOTE,
            note_record(
                text,
                item_id=item_id,
                item_name=item_name,
                reply_to=reply_to,
                evidence=evidence,
            ),
        )
        return {"available": True, "recorded": True}

    def add_evidence(
        self, item_name: Optional[str], save: Callable[[], Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Keep the agent's evidence (*save* keeps it in the experiment and
        answers its record) and put it on the record under the item it names.
        What cites it names its id."""
        item_id = None
        if item_name is not None:
            item, refusal = find_item(self._experiment, item_name)
            if refusal:
                return {"available": True, "recorded": False, **refusal}
            item_id = item.id
        record = save()
        logging.info(
            "agent evidence %s%s: %s",
            record["evidence_id"],
            f" [{item_name}]" if item_name else "",
            record["image"],
        )
        self.record_event(EVIDENCE, {**record, "item_id": item_id})
        return {"available": True, "recorded": True, **record}

    def record_event(
        self, kind: str, payload: Dict[str, Any], actor: Optional[str] = None
    ) -> None:
        """Put an event on the experiment's record: everything the server's
        side writes, the agent's requests and their outcomes (FIB-1283)
        among them. *actor* as for ``EventBuffer.append``."""
        if self._event_buffer is not None:
            self._event_buffer.append(kind, payload, actor=actor)

    def stop_workflow(self) -> Dict[str, Any]:
        """Stop the running workflow — the same path as the GUI's Stop button.

        Everything on that path is thread-safe from here: the manager's stop is
        an event plus the milling service's stop, the call the core server's
        ``stop_milling`` route already issues from this thread. Stopping is the
        safety action, so like ``stop_milling`` it never waits on the command
        lock.
        """
        if not self._running:
            return {
                "available": True,
                "stopped": False,
                "reason": "no workflow is running",
            }
        self._runner.stop()
        return {"available": True, "stopped": True}

    def stop_task(self, item_name: str, task_name: str) -> Dict[str, Any]:
        """Stop the task now running and carry on with the rest of the queue,
        as the timeline's Stop Task does. *item_name* and *task_name* name the
        task meant: a stop that lands after it finished is refused rather than
        stopping the next one."""
        manager = self._manager
        if manager is None:
            return {"stopped": False, "reason": "no workflow is running"}
        active = manager.queue.active
        if active is None or (active.item_name, active.task_name) != (
            item_name,
            task_name,
        ):
            return {
                "stopped": False,
                "reason": f"{task_name} for {item_name} is not the task running",
                "running": (
                    {"item_name": active.item_name, "task_name": active.task_name}
                    if active is not None
                    else None
                ),
            }
        self._runner.stop_task()
        return {"stopped": True, "item_name": item_name, "task_name": task_name}

    def _start_run(self, request: RunRequest, timeout: float) -> Dict[str, Any]:
        """Start ``request`` through the session's runner, where the
        experiment is edited, as every start goes."""
        runner = self._runner
        outcome = self._session.run_edit(
            lambda: runner.start(self._experiment, request).as_dict()
        )
        result = outcome.result(timeout=timeout)
        result["available"] = True
        return result

    def start_workflow(
        self,
        task_names: List[str],
        item_names: Optional[List[str]] = None,
        timeout: float = 15.0,
    ) -> Dict[str, Any]:
        """Start a workflow as the Run button would — the batch-review opener.

        Started where the experiment is edited (the GUI thread, with a window,
        which draws its run chrome there); validation refusals come back
        structured with the valid names. ``item_names`` of None means every item in the experiment.
        The agent at Drive is the consent that stands in for the Run click's
        confirm dialog.
        """
        result = self._start_run(
            RunRequest(
                kind=ITEMS,
                task_names=tuple(task_names),
                names=tuple(item_names) if item_names is not None else None,
            ),
            timeout,
        )
        if result.get("started"):
            # Who started it: the run's own workflow_started fires on its
            # worker thread, which carries no one's mark (FIB-1414).
            self.record_event(
                "workflow_start",
                {"task_names": list(task_names), "item_names": item_names},
            )
        return result

    def requeue_task(
        self, item_name: str, task_name: str, front: bool = False
    ) -> Dict[str, Any]:
        """Queue ``task_name`` for ``item_name`` (again) — the batch-review verb.

        The queue exists only while a workflow runs, so a re-run is queued into
        the live run: "run 03 again" while the batch continues. Adding a
        duplicate of a completed pair is the queue's own re-run mechanism; the
        item lands at the back (or the front with ``front=True``). A pair
        already waiting is refused (``already_queued``): it runs once. The
        timeline's Run again queues the same way (``session.runner.requeue``).
        """
        edit = self._agent_call(
            lambda: self._session.runner.requeue(
                item_name, task_name, front=bool(front)
            )
        )
        if not edit.applied:
            out: Dict[str, Any] = {
                "available": True,
                "queued": False,
                "reason": edit.reason,
                **edit.details,
            }
            if edit.error_type in ("already_queued", "fixed"):
                out["error_type"] = edit.error_type
            return out
        (item,) = edit.items
        return {
            "available": True,
            "queued": True,
            "item_id": item.id,
            "item_name": item_name,
            "task_name": task_name,
            "queue_version": edit.version,
        }

    def edit_queue(self, item_name: str, task_name: str, action: str) -> Dict[str, Any]:
        """Remove or move a waiting (item, task) pair in the running workflow's
        queue, as the timeline's menu does (FIB-1419).

        ``action`` is one of :data:`QUEUE_ACTIONS`: ``remove`` (it will not run;
        kept in the run record as Removed), ``move_up``, ``move_down`` or
        ``run_next``. Only a pair still waiting can be edited: the running one
        is refused (``not_pending``) -- stopping it is ``stop_workflow`` -- and
        so is one that already ran.
        """
        edit = self._session.runner.edit(item_name, task_name, action)
        if not edit.applied:
            if edit.error_type == "not_running":
                return {"available": True, "applied": False, "reason": edit.reason}
            return {
                "available": True,
                "applied": False,
                "error_type": edit.error_type,
                "error": edit.reason,
                **edit.details,
            }
        (item,) = edit.items
        return {
            "available": True,
            "applied": True,
            # Already first or last: nothing moved.
            "changed": edit.changed,
            "item_id": item.id,
            "item_name": item_name,
            "task_name": task_name,
            "action": action,
            "queue_version": edit.version,
        }

    # --- the protocol: its tasks, their order, requirements, schedule and
    # supervision, and each task's settings. Saved with the protocol at once.

    def protocol_task_config(self, task_name: str) -> Dict[str, Any]:
        """One task's protocol-level defaults document — what new items copy."""
        experiment = self._experiment
        protocol = getattr(experiment, "task_protocol", None) if experiment else None
        config_map = getattr(protocol, "task_config", None) if protocol else None
        if config_map is None:
            return {"available": False, "task_name": task_name}
        config = dict(config_map).get(task_name)
        if config is None:
            return {
                "available": True,
                "task_name": task_name,
                "error": f"No task named {task_name!r} in the protocol.",
                "task_names": list(config_map.keys()),
            }
        return self._config_document(task_name, config, level="protocol")

    @staticmethod
    def _config_document(task_name: str, config, level: str) -> Dict[str, Any]:
        """One config as a wire document with its version token: the full
        ``to_dict``, the contract for editing (you cannot patch what you
        cannot read)."""
        from fibsem.server.task_configs import config_version
        from fibsem.workflows.events import to_plain

        return {
            "available": True,
            "task_name": task_name,
            "level": level,
            "version": config_version(config),
            "config": to_plain(config.to_dict()),
        }

    def apply_protocol_task_config_patch(
        self,
        task_name: str,
        patch: Dict[str, Any],
        version: str,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Patch a task's protocol-level settings, what each run of it reads,
        when *version* names the state the patch was written against."""
        from fibsem.server.task_configs import patch_config

        session = self._session

        def edit():
            experiment = session.experiment
            protocol = getattr(experiment, "task_protocol", None)
            config_map = getattr(protocol, "task_config", None)
            if config_map is None:
                return {"applied": False, "error": "no protocol is loaded"}
            config = dict(config_map).get(task_name)
            if config is None:
                return {
                    "applied": False,
                    "error": f"No task named {task_name!r} in the protocol.",
                    "task_names": list(config_map.keys()),
                }
            result = patch_config(config, dict(patch), str(version))
            if result["applied"]:
                result["saved"] = self._save_protocol(experiment)
                result["task_name"] = task_name
                session.changed.emit(
                    CONFIG,
                    {
                        "level": "protocol",
                        "item_name": "",
                        "task_name": task_name,
                        "result": result,
                    },
                )
            return result

        result = self._agent_call(edit, timeout)
        result["available"] = True
        return result

    def set_supervision(self, task_name: str, supervise: bool) -> Dict[str, Any]:
        """Set whether ``task_name`` asks for supervision
        (``workflow_edits.set_supervision``), as the window's chips do."""
        from fibsem.workflows.workflow_edits import set_supervision

        experiment = self._experiment
        if experiment is None or experiment.task_protocol is None:
            return {"available": False, "applied": False}
        return {
            "available": True,
            **set_supervision(experiment, task_name, supervise, self.record_event),
        }

    def set_task_schedule(
        self, task_name: str, scheduled_at: Optional[str]
    ) -> Dict[str, Any]:
        """Set (or clear) when ``task_name`` may start, in the live workflow,
        as the Workflow tab's Edit Task does
        (``RunPlan.set_schedule``): up to two days ahead.

        The workflow reads the schedule at each task start (never a snapshot),
        so a change takes effect at the next start, and like ``set_supervision``
        it is saved at once.

        ``scheduled_at`` is ISO-8601 (naive = local time; an offset is
        normalized by the workflow) or ``None`` to clear.
        """
        from fibsem.workflows.workflow_edits import (
            save_workflow_change,
        )

        parsed = None
        if scheduled_at is not None:
            try:
                parsed = datetime.fromisoformat(scheduled_at)
            except ValueError:
                return {
                    "available": True,
                    "applied": False,
                    "invalid_value": f"{scheduled_at!r} is not an ISO-8601 "
                    "timestamp (e.g. '2026-09-04T06:00:00'); null clears "
                    "the schedule.",
                }
        experiment = self._experiment
        if experiment is None or experiment.task_protocol is None:
            return {"available": False, "applied": False}
        config = experiment.task_protocol.workflow_config
        why = config.set_schedule(task_name, parsed)
        if why:
            return {
                "available": True,
                "applied": False,
                "error": why,
                "task_names": config.workflow,
            }
        iso = parsed.isoformat() if parsed else None
        saved = save_workflow_change(
            experiment,
            {"field": "scheduled_at", "task_name": task_name, "scheduled_at": iso},
            self.record_event,
        )
        return {
            "available": True,
            "applied": True,
            "saved": saved,
            "task_name": task_name,
            "scheduled_at": iso,
        }

    def _workflow_edit(self, what: str):
        """The experiment and its workflow config for an edit, or a refusal."""
        experiment = self._experiment
        protocol = getattr(experiment, "task_protocol", None) if experiment else None
        if protocol is None:
            return (
                None,
                None,
                {"error_type": "not_found", "error": "No experiment is loaded."},
            )
        refusal = self._not_while_running(what)
        if refusal:
            return None, None, refusal
        return experiment, protocol.workflow_config, None

    def _protocol_changed(self, experiment, change: Dict[str, Any]) -> bool:
        """Save the protocol after an edit, say it changed, so no editor writes
        back a list it drew before the change, and put it on the record.
        Whether it saved."""
        saved = self._save_protocol(experiment)
        self._session.changed.emit(PROTOCOL)
        self.record_event("workflow_changed", change)
        return saved

    @staticmethod
    def _save_protocol(experiment) -> bool:
        """Save the experiment with its protocol; whether it saved. A failed
        save leaves the change in memory."""
        try:
            experiment.save(save_protocol=True)
        except Exception:
            logging.exception("save after a protocol change failed; applied in memory")
            return False
        return True

    def remove_task(self, task_name: str) -> Dict[str, Any]:
        """Remove a task from the protocol and the workflow
        (``experiment.remove_task``), as the protocol editor's Remove does.
        Task history stays: it is the record of what ran."""
        experiment, config, refusal = self._workflow_edit("remove tasks")
        if refusal:
            return refusal
        if task_name not in experiment.task_protocol.task_config:
            return {
                "error_type": "not_found",
                "error": f"No task named {task_name!r} in the protocol.",
                "task_names": list(experiment.task_protocol.task_config),
            }

        def remove():
            experiment.remove_task(task_name)
            return self._protocol_changed(
                experiment, {"field": "tasks", "removed": task_name}
            )

        saved = self._agent_call(remove)
        return {"applied": True, "saved": saved, "task_names": config.workflow}

    def reorder_tasks(self, order: List[str]) -> Dict[str, Any]:
        """Put the workflow's tasks in *order*, as dragging them on the Workflow
        tab does: every task, each once. Refused when a task would come before
        a task it requires."""
        experiment, config, refusal = self._workflow_edit("reorder tasks")
        if refusal:
            return refusal

        def reorder():
            why = experiment.task_protocol.reorder(list(order))
            if why:
                return why
            return self._protocol_changed(
                experiment, {"field": "order", "order": order}
            )

        saved = self._agent_call(reorder)
        if isinstance(saved, str):
            return {
                "error_type": "invalid",
                "error": saved,
                "task_names": config.workflow,
            }
        return {"applied": True, "saved": saved, "task_names": config.workflow}

    def set_task_requirements(
        self,
        task_name: str,
        requires: Optional[List[str]] = None,
        required: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Set what a workflow task requires (the tasks an item must have
        completed first) and whether it is required or optional, as the
        Workflow tab's Edit Task does. Refused when a requirement is not an
        earlier task."""
        experiment, config, refusal = self._workflow_edit("edit tasks")
        if refusal:
            return refusal
        protocol = experiment.task_protocol
        task = config.get(task_name)
        if task is None or task_name not in protocol.task_config:
            return {
                "error_type": "not_found",
                "error": f"No task named {task_name!r} in the workflow.",
                "task_names": config.workflow,
            }

        def edit():
            why = protocol.set_requirements(task_name, requires, required)
            if why:
                return why
            return self._protocol_changed(
                experiment,
                {
                    "field": "requirements",
                    "task_name": task_name,
                    "requires": protocol.requirements(task_name),
                    "required": task.required,
                },
            )

        saved = self._agent_call(edit)
        if isinstance(saved, str):
            return {"error_type": "invalid", "error": saved}
        return {
            "applied": True,
            "saved": saved,
            "task_name": task_name,
            "requires": protocol.requirements(task_name),
            "required": task.required,
        }
