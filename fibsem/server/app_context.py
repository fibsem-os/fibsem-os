"""AppContext: what every app's context serves the /app core reads from.

An app's context is the facade an embedded server sees the app through: it
resolves the session's experiment, runner and microscope at call time, on
every call, and returns plain JSON-able data, never live objects or Qt. Reads
are point-in-time snapshots, safe from a server thread. An app subclasses it
and adds what its own experiment and protocol need.
"""

from typing import Any, Dict, List, Optional

from fibsem.geometry.saved_images import image_point_to_stage_position, stage_projector
from fibsem.structures import BeamType
from fibsem.util.records import dataframe_records
from fibsem.util.timestamps import to_iso, wall_time_of
from fibsem.workflows.status import TaskStatus

__all__ = ["AppContext"]


class AppContext:
    """The reads every app serves, over its ``AppSession``."""

    # The app the operator works in, as the App Server names it to the agent.
    app_name = "fibsemOS"

    def __init__(self, session, window: Any = None, event_buffer=None):
        self._session = session
        self._window = window
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

    def recent_experiments(self) -> List[Dict[str, Any]]:
        """The recents list, via the never-raising peek layer."""
        from fibsem.config import get_recent_experiments

        return [
            {
                "name": summary.name,
                "path": summary.path,
                "app": summary.app,
                "created_at": to_iso(summary.created_at),
                "num_items": summary.num_lamella,
                "available": summary.available,
                "instrument_model": summary.instrument_model,
                "software_version": summary.software_version,
            }
            for summary in get_recent_experiments()
        ]
