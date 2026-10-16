"""AutoLamellaAppContext: the read-side facade an embedded server sees the app through.

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
made to poll the instrument on behalf of an observer). The one exception is an
action's own gate run through the host: requeueing reads the grid inventory, as
the Add to queue click does.

Vocabulary: the surface speaks ``item_name`` throughout. ``lamella_name`` is
the deprecated alias scheduled to drop with the HookContext shims after v0.6
and never appears in these payloads.

The host is anything carrying the four attributes the facade resolves through
(``experiment``, ``microscope``, ``_task_manager``, ``is_workflow_running``) —
the real ``AutoLamellaUI`` in production, a plain holder of real domain objects
in tests.
"""

import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

import fibsem.config as fibsem_cfg
from fibsem.applications.autolamella import task_outputs as _task_outputs
from fibsem.applications.autolamella.structures import Attention, AutoLamellaTaskStatus
from fibsem.server.files import jpeg_preview
from fibsem.util.timestamps import to_iso, wall_time_of

__all__ = [
    "AutoLamellaAppContext",
    "ITEM_PATCH_FIELDS",
    "config_version",
    "item_fields_version",
]

# The item-document fields an agent may patch: what a lamella IS — geometry,
# verdict, notes. Everything else on the object (poses, ids, paths, history)
# is either derived, hardware-adjacent, an *outcome* rather than an input
# (milling_angle is recorded from where Setup put the stage — writing it
# would move nothing), or the record itself.
ITEM_PATCH_FIELDS = ("poi", "alignment_area", "description", "defect")


def _content_hash(data) -> str:
    import hashlib
    import json

    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def item_pose_positions(lamella) -> Dict[str, Any]:
    """Where each of an item's poses puts the stage, by pose name (None where a
    pose has no position): the item document's ``poses``."""
    poses = {}
    for pose_name, pose in dict(lamella.poses).items():
        position = getattr(pose, "stage_position", None)
        poses[pose_name] = position.to_dict() if position is not None else None
    return poses


def item_fields_version(lamella) -> str:
    """A content hash naming the state of an item's patchable fields and poses.

    The item document's version, hashing the ITEM_PATCH_FIELDS and where the
    poses put the stage — so a task re-recording geometry or moving the
    milling pose rotates it, and a pending patch goes stale (FIB-1420): a POI
    is in the milling frame, so one chosen before the pose moved names a
    different place after. History growing does not rotate it.
    """
    from fibsem.applications.autolamella.server.events import to_plain

    fields = {
        name: to_plain(getattr(lamella, name, None)) for name in ITEM_PATCH_FIELDS
    }
    fields["poses"] = item_pose_positions(lamella)
    return _content_hash(fields)


def config_version(config) -> str:
    """A content hash naming exactly this state of a task config.

    The config's version (FIB-864): reads serve it, writes echo it, and a
    mismatch is refused as stale — a patch can never apply against a state
    its author did not see. Computed from the serialized document so the
    same content hashes identically wherever it is held.
    """
    import hashlib
    import json

    from fibsem.applications.autolamella.server.events import to_plain

    return _content_hash(to_plain(config.to_dict()))


def _json_safe(value: Any) -> Any:
    """Coerce one cell of a dataframe/record into something JSON can carry."""
    if isinstance(value, datetime):
        # pandas' NaT is a datetime too: a time not yet known, as NaN is for a number
        return None if value != value else value.isoformat()
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


# The overview roles the grid tasks record (imaging.py ROLE_BY_BEAM, the FM
# task). Listed here so a summary can say which overviews a grid has without
# a client knowing the role vocabulary.
GRID_OVERVIEW_ROLES = ("overview_sem", "overview_fib", "overview_fm")


def _trench_window(lamella):
    """The window to measure a lamella's band in, from its own trench patterns
    (those that follow its POI): along the lamella, nine tenths of the
    narrowest pattern, so the trench ends stay out; across it, the widest
    gap between trenches (the rough slab) and half a micron into each trench.
    None when it has no trench pattern."""
    widths, gaps = [], []
    for config in lamella.task_config.values():
        if not getattr(config, "sync_to_poi", False):
            continue
        for milling in (getattr(config, "milling", None) or {}).values():
            for stage in milling.stages:
                pattern = stage.pattern
                if hasattr(pattern, "spacing") and hasattr(pattern, "width"):
                    widths.append(pattern.width)
                    gaps.append(pattern.spacing)
    if not widths:
        return None
    return 0.9 * min(widths), max(gaps) + 1e-6


def find_item(experiment, item_name: str):
    """The item named *item_name*, and None; or None and why not, in the words
    every route refuses an unknown item with, and the names there are. Each
    route puts the refusal in its own answer's shape."""
    lamella = (
        experiment.get_lamella_by_name(item_name) if experiment is not None else None
    )
    if lamella is not None:
        return lamella, None
    return None, {
        "error": f"No item named {item_name!r} in this experiment.",
        "item_names": [p.name for p in experiment.positions]
        if experiment is not None
        else [],
    }


def _overview_point_to_stage(path: str, px=None, fraction=None):
    """The stage position under a point of a recorded overview, as the stored
    overview canvas resolves a click (`fibsem.geometry.projection.image_point_to_stage`)."""
    from fibsem.fm.preview import is_fluorescence_image
    from fibsem.geometry.projection import image_point_to_stage

    if (px is None) == (fraction is None):
        raise ValueError("Give exactly one of px ([x, y] source pixels) or fraction.")
    point = px if px is not None else fraction
    if not (isinstance(point, (list, tuple)) and len(point) == 2):
        raise ValueError("The point is [x, y].")
    x, y = float(point[0]), float(point[1])

    if is_fluorescence_image(path):
        from fibsem.fm.structures import FluorescenceImage

        image = FluorescenceImage.load(path)
    else:
        from fibsem.structures import FibsemImage

        image = FibsemImage.load(path)
    height, width = image.data.shape[-2:]
    if fraction is not None:
        x, y = x * width, y * height
    if not (0 <= x <= width and 0 <= y <= height):
        raise ValueError(f"The point is outside the {width} x {height} image.")
    try:
        return image_point_to_stage(image, x, y)
    except ValueError:
        raise ValueError(
            "The overview has no stage geometry to convert from."
        ) from None


def _overview_projector(path: str):
    """Open a recorded overview and return its scale block plus a projector.

    The projector maps one stage position to a source-pixel Point through the
    canvas's own reprojection for that kind of image: beam overviews
    (FibsemImage) through ``reproject_stage_positions_onto_image2``,
    fluorescence overviews (OME-TIFF) through
    ``reproject_stage_positions_onto_fm_image``. Both raise, per position, when
    the image lacks the metadata a projection needs; the caller records the
    reason against the item rather than guessing. The scale block is what a
    client needs to place the points on its rendition: source size, pixel size
    (m), field width (m), the acquisition's stage position.
    """
    from fibsem.fm.preview import is_fluorescence_image

    if is_fluorescence_image(path):
        from fibsem.fm.reprojection import reproject_stage_positions_onto_fm_image
        from fibsem.fm.structures import FluorescenceImage

        image = FluorescenceImage.load(path)
        height, width = image.data.shape[-2:]
        metadata = image.metadata
        stage = getattr(metadata, "stage_position", None)
        block = {
            "width": int(width),
            "height": int(height),
            "pixel_size": metadata.pixel_size_x if metadata else None,
            "hfw": metadata.pixel_size_x * width if metadata else None,
            "beam_type": "FM",
            "stage_position": stage.to_dict() if stage is not None else None,
        }

        def project(position):
            (point,) = reproject_stage_positions_onto_fm_image(image, [position])
            return point

        return block, project

    from fibsem.imaging.tiling.reprojection import (
        reproject_stage_positions_onto_image2,
    )
    from fibsem.structures import FibsemImage

    image = FibsemImage.load(path)
    height, width = image.data.shape[:2]
    metadata = image.metadata
    state = getattr(metadata, "microscope_state", None) if metadata else None
    stage = state.stage_position if state is not None else None
    block = {
        "width": int(width),
        "height": int(height),
        "pixel_size": metadata.pixel_size.x if metadata else None,
        "hfw": metadata.image_settings.hfw if metadata else None,
        "beam_type": metadata.image_settings.beam_type.name if metadata else None,
        "stage_position": stage.to_dict() if stage is not None else None,
    }

    def project(position):
        (point,) = reproject_stage_positions_onto_image2(image, [position])
        return point

    return block, project


def _attendance_line(task_protocol: Any, task_name: str) -> str:
    """One line on what a task needs from a person, or "" when its type is
    not one this build knows."""
    from fibsem.applications.autolamella.workflows.tasks.attendance import (
        attendance_for,
    )

    try:
        review_on = bool(
            fibsem_cfg.load_user_preferences().features.proposer_reviewer_workflow_enabled
        )
    except Exception:
        review_on = False
    att = attendance_for(task_protocol, task_name, review_on)
    return att.line if att is not None else ""


def _response_for(request, parsed, response: bool) -> bool:
    """The yes/no an answer stands for: the value's, when its action says (the
    milling question), else the answer's own."""
    if parsed is None:
        return bool(response)
    said = request.response_for(parsed)
    return bool(response) if said is None else said


class AutoLamellaAppContext:
    """Read-only facade over a running (or resting) AutoLamella session."""

    def __init__(self, host, event_buffer=None):
        self._host = host
        self._event_buffer = event_buffer

    # --- call-time resolution: never cache what the app rebinds ---------------

    @property
    def _experiment(self):
        return getattr(self._host, "experiment", None)

    @property
    def experiment_path(self) -> Optional[str]:
        """The open experiment's folder, or None: where an agent may save."""
        experiment = self._experiment
        return str(experiment.path) if experiment is not None else None

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
            position = _overview_point_to_stage(target, px=px, fraction=fraction)
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
        block, project = _overview_projector(target)
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
        from fibsem.util.timestamps import to_datetime

        def started(record):
            # Each item's runs in turn is not the order they ran in. Through
            # POSIX time, so old naive stamps and aware ones still compare.
            start = to_datetime(record.get("start_timestamp"))
            return start.timestamp() if start is not None else float("inf")

        items = sorted(_records(experiment.task_history_dataframe()), key=started)
        return {"available": True, "items": items}

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

        from fibsem.applications.autolamella.server import log_tail

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

    def protocol(self) -> Dict[str, Any]:
        """The workflow definition with live supervision flags, schedules and
        what each task needs from a person."""
        experiment = self._experiment
        task_protocol = (
            getattr(experiment, "task_protocol", None) if experiment else None
        )
        if task_protocol is None:
            return {"available": False, "tasks": []}
        config = task_protocol.workflow_config
        grid_protocol = experiment.grid_protocol
        return {
            "available": True,
            "name": config.name,
            # The grid protocol beside the lamella workflow: what a grid run
            # can name. Kind is the registered task type (beam vs FM overview).
            "grid_tasks": [
                {
                    "name": name,
                    "type": getattr(grid_protocol.task_config[name], "task_type", None),
                }
                for name in grid_protocol.ordered_task_names
            ],
            "tasks": [
                {
                    "name": task.name,
                    "attention": task.attention.value,
                    # what the task needs from a person under that attention:
                    # whether someone must be at the microscope while it
                    # runs, and what waits for a decision afterwards
                    "attendance": _attendance_line(task_protocol, task.name),
                    # the boolean the /supervision verb takes, for the agents
                    # that read it back
                    "supervise": task.attention is Attention.supervised,
                    "required": task.required,
                    "requires": list(task.requires),
                    "scheduled_at": task.scheduled_at.isoformat()
                    if task.scheduled_at is not None
                    else None,
                }
                for task in config.tasks
            ],
            # the run options workflow_set_options changes
            "options": {
                "turn_beams_off": task_protocol.options.turn_beams_off,
                "show_run_summary": task_protocol.options.show_run_summary,
            },
        }

    def task_outputs(self, item_name: str) -> Dict[str, Any]:
        """The files an item's completed tasks produced, existence-checked.

        ``tasks`` keeps the per-run grouping the flat lists erase — one entry
        per history entry, in run order, each with its recorded files by role
        (basenames, servable through :meth:`output_image`). That is the review
        story: which task produced which image.
        """
        import os

        experiment = self._experiment
        if experiment is None:
            return {"available": False, "item_name": item_name}
        lamella, refusal = find_item(experiment, item_name)
        if refusal:
            return {"available": False, "item_name": item_name, **refusal}
        history = list(lamella.task_history)
        tasks = []
        for state in history:
            files: Dict[str, List[str]] = {}
            for role, relpaths in dict(state.outputs).items():
                names = [
                    os.path.basename(rp)
                    for rp in relpaths
                    if os.path.isfile(os.path.join(lamella.path, rp))
                ]
                if names:
                    files[role] = names
            tasks.append(
                {"name": state.name, "status": state.status.name, "files": files}
            )
        return {
            "available": True,
            "item_name": item_name,
            "completed_tasks": [
                state.name
                for state in history
                if state.status is AutoLamellaTaskStatus.Completed
            ],
            "tasks": tasks,
            "final_reference_images": [
                str(p) for p in _task_outputs.final_reference_images(lamella, *history)
            ],
            "fluorescence_images": [
                str(p) for p in _task_outputs.fluorescence_images(lamella, *history)
            ],
        }

    @staticmethod
    def _recorded_item_paths(lamella) -> List[str]:
        """Every file :meth:`task_outputs` lists for ``lamella``, as paths."""
        import os

        history = list(lamella.task_history)
        paths = [
            os.path.join(lamella.path, rp)
            for state in history
            for relpaths in dict(state.outputs).values()
            for rp in relpaths
            if os.path.isfile(os.path.join(lamella.path, rp))
        ]
        for path in _task_outputs.final_reference_images(
            lamella, *history
        ) + _task_outputs.fluorescence_images(lamella, *history):
            if str(path) not in paths:
                paths.append(str(path))
        return paths

    def output_image(
        self, item_name: str, filename: str, max_width: int = 768
    ) -> Dict[str, Any]:
        """One recorded output image, rendered as JPEG bytes.

        The read side of :meth:`task_outputs` for clients that cannot touch the
        disk — the dashboard's thumbnails. ``filename`` must be the basename of
        a path :meth:`task_outputs` itself listed; anything else is refused
        with the valid names, so a client can never reach outside the item's
        own recorded outputs. Stacks and multi-channel images are max-projected
        to something viewable; this is a preview, not the data.
        """
        experiment = self._experiment
        if experiment is None:
            return {"available": False, "jpeg": None}
        lamella, refusal = find_item(experiment, item_name)
        if refusal:
            return {"available": False, "jpeg": None, **refusal}
        match, refusal = self._recorded_item_file(lamella, filename)
        if refusal:
            return {"available": True, "jpeg": None, **refusal}
        return jpeg_preview(match, filename, max_width)

    @classmethod
    def _recorded_item_file(cls, lamella, filename: str):
        """The path of the output :meth:`task_outputs` lists as *filename*, and
        None; or None and the refusal naming the valid ones. Only a listed
        basename resolves, so no client reaches outside the item's outputs."""
        import os

        paths = cls._recorded_item_paths(lamella)
        match = next((p for p in paths if os.path.basename(p) == filename), None)
        if match is not None:
            return match, None
        return None, {
            "error": f"No output named {filename!r} for {lamella.name!r}.",
            "filenames": [os.path.basename(p) for p in paths],
        }

    def measure_thickness(
        self,
        item_name: str,
        filename: str,
        width_um: Optional[float] = None,
        height_um: Optional[float] = None,
    ) -> Dict[str, Any]:
        """The lamella's thickness, read off one of the item's saved FIB images.

        The bright band between the trenches, measured column by column at half
        its peak brightness (:func:`fibsem.imaging.thickness.measure_lamella_band`)
        in a window around the item's POI: ``width_um`` along the lamella,
        ``height_um`` across it, each by default from the item's own trench
        patterns (:func:`_trench_window`). Nothing moves; the result is saved
        in a ``thickness`` folder beside the image. The answer carries the
        result's diagnostic plot.
        """
        import base64
        import os

        from fibsem.imaging.thickness import DEFAULT_WINDOW, measure_lamella_band
        from fibsem.structures import BeamType, FibsemImage, Point

        experiment = self._experiment
        if experiment is None:
            return {"error_type": "not_found", "error": "No experiment is open."}
        lamella, refusal = find_item(experiment, item_name)
        if refusal:
            return {"error_type": "not_found", **refusal}
        path, refusal = self._recorded_item_file(lamella, filename)
        if refusal:
            return {"error_type": "not_found", **refusal}
        try:
            image = FibsemImage.load(path)
        except Exception as exc:  # noqa: BLE001 - an unreadable file is a data fact
            return {"error_type": "invalid", "error": f"{filename!r}: {exc}"}
        settings = getattr(image.metadata, "image_settings", None)
        if getattr(settings, "beam_type", None) is BeamType.ELECTRON:
            return {
                "error_type": "invalid",
                "error": f"{filename!r} is an SEM image; the band is read off a FIB image.",
            }
        default, source = _trench_window(lamella), "patterns"
        if default is None:
            default, source = DEFAULT_WINDOW, "default"
        window = (
            default[0] if width_um is None else width_um * 1e-6,
            default[1] if height_um is None else height_um * 1e-6,
        )
        if width_um is not None or height_um is not None:
            source = "given"
        poi = lamella.poi if lamella.poi is not None else Point(0, 0)
        try:
            result = measure_lamella_band(image, poi, window)
        except ValueError as exc:
            return {"error_type": "invalid", "error": str(exc)}
        # Kept beside the image, as an autofocus run keeps its own: one folder
        # per image, replaced when the same image is measured again.
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            saved_to = str(
                result.save(os.path.join(os.path.dirname(path), "thickness"), stem)
            )
        except OSError:
            logging.exception(f"Could not save the thickness result for {filename!r}")
            saved_to = None
        return {
            "item_name": item_name,
            "filename": filename,
            "poi": {"x": poi.x, "y": poi.y},
            "window_um": [round(window[0] * 1e6, 3), round(window[1] * 1e6, 3)],
            "window_from": source,
            "saved_to": saved_to,
            **result.to_dict(),
            "plot": {"image_b64_jpeg": base64.b64encode(result.plot_jpeg()).decode()},
        }

    # --- grids: the screening read model (FIB-876) ----------------------------

    def grids(self) -> Dict[str, Any]:
        """Every grid the experiment knows, with its latest recorded overviews.

        Records only — never the hardware inventory. Which slot a grid sits in
        and whether it is loaded are live stage facts, and this facade
        does not ask the instrument on an observer's behalf.
        """
        experiment = self._experiment
        if experiment is None:
            return {"available": False, "items": []}
        return {
            "available": True,
            "items": [
                self._grid_summary(experiment, grid) for grid in experiment.grids
            ],
        }

    def grid_detail(self, grid_name: str) -> Dict[str, Any]:
        """One grid: its summary plus every task run and the files each recorded.

        ``tasks`` mirrors :meth:`task_outputs` — one entry per history entry in
        run order, files by role as basenames servable through
        :meth:`grid_output_image`. ``items`` names the lamellae linked to the
        grid (``Lamella.grid_id``); an experiment made before grids existed has
        none, which is a fact about the data, not a failure.
        """
        import os

        experiment = self._experiment
        if experiment is None:
            return {"available": False, "grid_name": grid_name}
        grid = experiment.get_grid_by_name(grid_name)
        if grid is None:
            return self._no_such_grid(experiment, grid_name)
        root = experiment.grid_path(grid)
        tasks = []
        for state in grid.task_history:
            files: Dict[str, List[str]] = {}
            for role, relpaths in dict(state.outputs).items():
                names = [
                    os.path.basename(rp)
                    for rp in relpaths
                    if os.path.isfile(os.path.join(root, rp))
                ]
                if names:
                    files[role] = names
            tasks.append(
                {
                    "name": state.name,
                    "status": state.status.name,
                    "files": files,
                }
            )
        detail = self._grid_summary(experiment, grid)
        detail["tasks"] = tasks
        detail["items"] = [p.name for p in experiment.get_lamellae_for_grid(grid)]
        return detail

    def grid_output_image(
        self, grid_name: str, filename: str, max_width: int = 768
    ) -> Dict[str, Any]:
        """One recorded grid output (overview or thumbnail), rendered as JPEG.

        Same contract as :meth:`output_image`: ``filename`` must be a basename
        the grid's own history recorded, anything else is refused with the
        valid names. Overviews are large; this is the downscaled preview.
        """
        experiment = self._experiment
        if experiment is None:
            return {"available": False, "jpeg": None}
        grid = experiment.get_grid_by_name(grid_name)
        if grid is None:
            return dict(self._no_such_grid(experiment, grid_name), jpeg=None)
        match = self.recorded_grid_file(experiment, grid, filename)
        if match is None:
            return {
                "available": True,
                "jpeg": None,
                "error": f"No output named {filename!r} for grid {grid_name!r}.",
                "filenames": self._recorded_grid_basenames(experiment, grid),
            }
        return jpeg_preview(match, filename, max_width)

    def grid_markers(self, grid_name: str, filename: str) -> Dict[str, Any]:
        """Where the experiment's items fall on one recorded overview.

        Computed here, in source-image pixels, through the same reprojections
        the overview canvases use — the beam one for SEM/FIB overviews, the FM
        one for fluorescence overviews — so no client ever re-derives stage
        geometry (the compucentric offset and canvas-origin traps live in one
        place). ``image`` carries the scale contract: source size, pixel size,
        field width, the stage position the overview was taken from. Items
        that cannot be placed (no pose, or an image without the metadata its
        projection needs) are listed under ``unplaced`` with the reason, never
        dropped silently.

        Which items: the lamellae linked to the grid when any are. With none
        linked, an experiment that records a single grid falls back to every
        posed lamella, flagged ``linked: false`` — an experiment from before
        grids were recorded still gets its glance view; with several grids
        that fallback would paint every grid with the same unrelated items, so
        nothing is placed and the payload says why.
        """
        experiment = self._experiment
        if experiment is None:
            return {"available": False, "grid_name": grid_name, "markers": []}
        grid = experiment.get_grid_by_name(grid_name)
        if grid is None:
            return dict(self._no_such_grid(experiment, grid_name), markers=[])
        match = self.recorded_grid_file(experiment, grid, filename)
        if match is None:
            return {
                "available": True,
                "grid_name": grid_name,
                "markers": [],
                "error": f"No output named {filename!r} for grid {grid_name!r}.",
                "filenames": self._recorded_grid_basenames(experiment, grid),
            }
        try:
            image_block, project = _overview_projector(match)
        except Exception as exc:  # noqa: BLE001 - a broken file is a data fact
            return {
                "available": True,
                "grid_name": grid_name,
                "filename": filename,
                "markers": [],
                "error": f"Could not read {filename!r}: {exc}",
            }
        payload: Dict[str, Any] = {
            "available": True,
            "grid_name": grid_name,
            "filename": filename,
            "image": image_block,
            "convention": "source-image pixels, origin top-left, +y down",
            "markers": [],
            "unplaced": [],
        }
        linked = experiment.get_lamellae_for_grid(grid)
        payload["linked"] = bool(linked)
        if linked:
            lamellae = linked
        elif len(experiment.grids) <= 1:
            lamellae = list(experiment.positions)
        else:
            lamellae = []
            payload["reason"] = (
                "no items are linked to this grid, and with several grids "
                "recorded the experiment's items are not assumed to be on it"
            )
        width, height = image_block["width"], image_block["height"]
        workflow = experiment.task_protocol.workflow_config
        for lamella in lamellae:
            pose = lamella.milling_pose
            position = pose.stage_position if pose is not None else None
            if position is None:
                payload["unplaced"].append(
                    {"item_name": lamella.name, "reason": "item has no pose"}
                )
                continue
            try:
                point = project(position)
            except Exception as exc:  # noqa: BLE001 - one bad pose must not hide the rest
                payload["unplaced"].append(
                    {"item_name": lamella.name, "reason": str(exc)}
                )
                continue
            last = lamella.last_completed_task
            payload["markers"].append(
                {
                    "item_name": lamella.name,
                    "x": float(point.x),
                    "y": float(point.y),
                    "inside": bool(0 <= point.x < width and 0 <= point.y < height),
                    "last_completed_task": last.name if last is not None else None,
                    "is_completed": workflow.is_completed(lamella),
                    "is_failure": lamella.is_failure,
                }
            )
        return payload

    @staticmethod
    def _grid_summary(experiment, grid) -> Dict[str, Any]:
        import os

        history = list(grid.task_history)
        completed = [
            t.name for t in history if t.status is AutoLamellaTaskStatus.Completed
        ]
        overviews = {}
        for role in GRID_OVERVIEW_ROLES:
            path = _task_outputs.latest_grid_output(experiment, grid, role)
            if path is not None:
                overviews[role] = os.path.basename(path)
        return {
            "name": grid.name,
            "id": grid.id,
            "description": grid.description,
            "quality": grid.quality.verdict.name,
            "status": grid.task_state.status.name,
            "current_task": grid.task_state.name or None,
            "is_failure": grid.is_failure,
            "completed_tasks": completed,
            "last_completed_task": completed[-1] if completed else None,
            "num_items": len(experiment.get_lamellae_for_grid(grid)),
            "overviews": overviews,
        }

    @staticmethod
    def _no_such_grid(experiment, grid_name: str) -> Dict[str, Any]:
        return {
            "available": False,
            "grid_name": grid_name,
            "error": f"No grid named {grid_name!r} in this experiment.",
            "grid_names": [g.name for g in experiment.grids],
        }

    @staticmethod
    def _recorded_grid_paths(experiment, grid) -> List[str]:
        roles = {role for state in grid.task_history for role in state.outputs}
        return _task_outputs.grid_outputs(experiment, grid, *sorted(roles))

    @classmethod
    def _recorded_grid_basenames(cls, experiment, grid) -> List[str]:
        import os

        return [os.path.basename(p) for p in cls._recorded_grid_paths(experiment, grid)]

    @classmethod
    def recorded_grid_file(cls, experiment, grid, filename: str) -> Optional[str]:
        import os

        return next(
            (
                p
                for p in cls._recorded_grid_paths(experiment, grid)
                if os.path.basename(p) == filename
            ),
            None,
        )

    # --- experiments and protocols: the File menu, for an agent at Drive ---------
    #
    # Kept to known places (Patrick's call, 2026-10-14): a new experiment goes in
    # the default experiments folder, a load is one of the recent experiments, and
    # a protocol is one of the files in the protocol folder, by name.

    def protocols(self) -> Dict[str, Any]:
        """The protocol files an agent may load or create an experiment with."""
        from fibsem.applications.autolamella import experiment_setup

        return {
            "folder": experiment_setup.protocol_directory(),
            "protocols": experiment_setup.list_protocols(),
        }

    def _not_while_running(self, what: str) -> Optional[Dict[str, Any]]:
        """The refusal while a workflow runs (*what* to do when it has
        stopped), or None."""
        if getattr(self._host, "is_workflow_running", False):
            return {
                "error_type": "task_running",
                "error": f"A workflow is running; {what} when it has stopped.",
            }
        return None

    def _no_microscope(self) -> Optional[Dict[str, Any]]:
        if self._microscope is None:
            return {"error_type": "not_found", "error": "No microscope is connected."}
        return None

    def _session_refusal(self) -> Optional[Dict[str, Any]]:
        """Why the open experiment cannot be replaced now, or None."""
        return self._not_while_running("change experiments") or self._no_microscope()

    def _protocol_or_refusal(self, name: Optional[str]):
        from fibsem.applications.autolamella import experiment_setup
        from fibsem.applications.autolamella.structures import AutoLamellaTaskProtocol

        path = (
            experiment_setup.default_protocol_path()
            if name is None
            else experiment_setup.protocol_by_name(name)
        )
        if path is None:
            return None, {
                "error_type": "not_found",
                "error": f"No protocol named {name!r} in the protocol folder.",
                "protocol_names": [
                    p["name"] for p in experiment_setup.list_protocols()
                ],
            }
        try:
            return AutoLamellaTaskProtocol.load(path), None
        except Exception as exc:  # noqa: BLE001 - a broken file is a data fact
            return None, {
                "error_type": "invalid",
                "error": f"Could not load the protocol {path}: {exc}",
            }

    def _adopt(self, experiment) -> None:
        """Make *experiment* the open one, as the dialogs' OK does."""

        def adopt():
            adopt_experiment = getattr(self._host, "_adopt_experiment", None)
            if adopt_experiment is not None:
                adopt_experiment(experiment)
            else:  # a host with no window: just open it
                self._host.experiment = experiment

        self._agent_call(adopt)

    def create_experiment(
        self,
        name: str,
        protocol: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create an experiment in the default experiments folder and open it,
        with *protocol* from the protocol folder (the default protocol when
        omitted). An existing experiment of that name is refused, never
        overwritten."""
        import os

        from fibsem.applications.autolamella import experiment_setup

        refusal = self._session_refusal()
        if refusal:
            return refusal
        if not name or os.path.basename(name) != name or name in (".", ".."):
            return {"error_type": "invalid", "error": "name is a plain folder name."}
        directory = experiment_setup.default_experiment_directory()
        if os.path.exists(os.path.join(directory, name)):
            return {
                "error_type": "invalid",
                "error": f"An experiment named {name!r} exists in {directory}.",
            }
        task_protocol, refusal = self._protocol_or_refusal(protocol)
        if refusal:
            return refusal
        metadata = {"description": description} if description else None
        experiment = experiment_setup.create_experiment(
            directory, name, task_protocol, metadata=metadata
        )
        self._adopt(experiment)
        return {"applied": True, "experiment": self.experiment_summary()}

    def load_experiment(self, path: str) -> Dict[str, Any]:
        """Open one of the recent experiments (``path`` as the recent list gives
        it), as the Load dialog does. One with no protocol is refused."""
        from fibsem.applications.autolamella import experiment_setup

        refusal = self._session_refusal()
        if refusal:
            return refusal
        recent = [entry["path"] for entry in self.recent_experiments()]
        if path not in recent:
            return {
                "error_type": "not_found",
                "error": "Only a recent experiment can be loaded; the operator "
                "opens any other from the Load dialog.",
                "recent_paths": recent,
            }
        experiment, why = experiment_setup.load_for_adoption(path)
        if experiment is None:
            return {"error_type": "invalid", "error": f"Not loaded: {why}."}
        self._adopt(experiment)
        experiment_setup.record_recent(experiment)
        return {"applied": True, "experiment": self.experiment_summary()}

    def load_protocol(self, protocol: str) -> Dict[str, Any]:
        """Load a protocol from the protocol folder into the open experiment, as
        the Load Protocol dialog does: it replaces the experiment's protocol and
        is saved beside it. Items keep the task configs they have (use
        apply_protocol_to_item to bring one up to date)."""
        refusal = self._session_refusal()
        if refusal:
            return refusal
        experiment = self._experiment
        if experiment is None:
            return {"error_type": "not_found", "error": "No experiment is loaded."}
        task_protocol, refusal = self._protocol_or_refusal(protocol)
        if refusal:
            return refusal

        def attach():
            adopt_protocol = getattr(self._host, "_adopt_protocol", None)
            if adopt_protocol is not None:
                adopt_protocol(task_protocol)
            else:  # a host with no window: just set and save it
                experiment.task_protocol = task_protocol
                experiment.save_protocol()

        self._agent_call(attach)
        return {"applied": True, "protocol": self.protocol()}

    # --- lamella positions: the Lamella tab's buttons, for an agent at Drive ------

    def _agent_call(self, call, select_latest: bool = False, timeout: float = 30.0):
        """``call()`` on the GUI thread through the host, or inline where the host
        has no GUI thread to offer (a test host, a headless session)."""
        request = getattr(self._host, "request_agent_call", None)
        if request is None:
            return call()
        return request(call, select_latest=select_latest).result(timeout=timeout)

    def _editable(self):
        """The experiment and microscope an edit needs, or a refusal."""
        experiment = self._experiment
        if experiment is None or experiment.task_protocol is None:
            return (
                None,
                None,
                {"error_type": "not_found", "error": "No experiment is loaded."},
            )
        refusal = self._no_microscope()
        if refusal:
            return None, None, refusal
        return experiment, self._microscope, None

    def overview_stage_position(
        self, grid_name: str, filename: str, px=None, fraction=None
    ) -> Dict[str, Any]:
        """The stage position under a point of a recorded grid overview.

        The inverse of :meth:`grid_markers`, through the same projection the
        overview canvases use, built from the image's own metadata: a saved
        overview converts as it was taken, wherever the stage is now. ``px`` is
        [x, y] in source-image pixels (origin top-left, +y down, as the markers
        give them); ``fraction`` is [x, y] from 0 to 1 across the image, for a
        point read off a preview.
        """
        experiment = self._experiment
        if experiment is None:
            return {"error_type": "not_found", "error": "No experiment is loaded."}
        grid = experiment.get_grid_by_name(grid_name)
        if grid is None:
            refusal = self._no_such_grid(experiment, grid_name)
            return {"error_type": "not_found", **refusal}
        path = self.recorded_grid_file(experiment, grid, filename)
        if path is None:
            return {
                "error_type": "not_found",
                "error": f"No output named {filename!r} for grid {grid_name!r}.",
                "filenames": self._recorded_grid_basenames(experiment, grid),
            }
        try:
            position = _overview_point_to_stage(path, px=px, fraction=fraction)
        except ValueError as exc:
            return {"error_type": "invalid", "error": str(exc)}
        return {
            "grid_name": grid_name,
            "grid_id": grid.id,
            "filename": filename,
            "stage_position": position.to_dict(),
        }

    def add_item(
        self,
        name: Optional[str] = None,
        stage_position: Optional[Dict[str, Any]] = None,
        overview: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Add a lamella as the Lamella tab's Add does: at the stage position now,
        at a given stage position, or under a point of a grid overview (which
        also puts it on that grid). Its poses, milling angle and grid are worked
        out as for a click; it is saved. At the stage position now is refused
        while a workflow runs: the stage is wherever the run has moved it."""
        from fibsem.applications.autolamella.poses import add_lamella
        from fibsem.structures import FibsemStagePosition

        experiment, microscope, refusal = self._editable()
        if refusal:
            return refusal
        if stage_position is None and overview is None:
            refusal = self._not_while_running("add an item where the stage is")
            if refusal:
                refusal["error"] += " Or give its stage_position."
                return refusal
        if name is not None and experiment.get_lamella_by_name(name) is not None:
            return {"error_type": "invalid", "error": f"An item named {name!r} exists."}
        position, grid_id = None, None
        if overview is not None:
            converted = self.overview_stage_position(
                overview.get("grid_name", ""),
                overview.get("filename", ""),
                px=overview.get("px"),
                fraction=overview.get("fraction"),
            )
            if "error_type" in converted:
                return converted
            position = FibsemStagePosition.from_dict(converted["stage_position"])
            grid_id = converted["grid_id"]
        elif stage_position is not None:
            # {x, y, z, r, t} as the tool describes it, in the stage's raw frame,
            # as the stage tools give positions; from_dict also wants the frame.
            position = FibsemStagePosition.from_dict(
                {"coordinate_system": "RAW", **stage_position}
            )

        def add():
            lamella = add_lamella(
                experiment,
                microscope,
                stage_position=position,
                name=name,
                grid_id=grid_id,
            )
            return lamella.name

        try:
            added = self._agent_call(add, select_latest=True)
        except ValueError as exc:
            return {"error_type": "invalid", "error": str(exc)}
        return {"applied": True, "item": self.item_detail(added)}

    def remove_item(self, item_name: str) -> Dict[str, Any]:
        """Remove a lamella from the experiment, as its row's Remove does. Its
        files stay on disk. Refused while a workflow runs."""
        experiment, _, refusal = self._editable()
        if refusal:
            return refusal
        lamella, refusal = find_item(experiment, item_name)
        if refusal:
            return {"error_type": "not_found", **refusal}
        refusal = self._not_while_running("remove items")
        if refusal:
            return refusal

        from fibsem.applications.autolamella.poses import remove_lamella

        self._agent_call(lambda: remove_lamella(experiment, lamella))
        return {"applied": True, "item_name": item_name}

    # -- grid records: the Grids tab's inventory read and Rename ----------------

    def _grids_changed(self) -> None:
        """Tell the app's grid views an agent changed the records."""
        signal = getattr(self._host, "grids_changed", None)
        if signal is not None:
            signal.emit()

    def add_grids(self) -> Dict[str, Any]:
        """Record every grid in the inventory that the experiment has no record
        of, as the Grids tab's inventory read does: the loader's last scan,
        read at once; nothing moves. Saved when anything was added."""
        experiment, microscope, refusal = self._editable()
        if refusal:
            return refusal
        stage = getattr(microscope, "sample_handling", None)
        if stage is None:
            return {"error_type": "not_found", "error": "No sample handling here."}
        stage.get_inventory()

        def add():
            added = experiment.sync_grids_from_inventory(stage)
            if added:
                experiment.save()
                self._grids_changed()
            return [grid.name for grid in added]

        added = self._agent_call(add)
        return {
            "applied": True,
            "added": added,
            "grid_names": [grid.name for grid in experiment.grids],
        }

    def rename_grid(self, grid_name: str, name: str) -> Dict[str, Any]:
        """Rename a grid as the Grids tab's Rename does: the slot that holds it
        first, then its record. Refused for a grid that has run, a name another
        grid has, and while a workflow runs (its queue holds grids by name)."""
        from fibsem.applications.autolamella.workflows.tasks.grid.manager import (
            grid_rename_refusal,
            rename_grid,
        )

        name = name.strip()
        if not name:
            return {"error_type": "invalid", "error": "A grid needs a name."}
        experiment, microscope, refusal = self._editable()
        if refusal:
            return refusal
        refusal = self._not_while_running("rename grids")
        if refusal:
            return refusal
        stage = getattr(microscope, "sample_handling", None)
        present = (
            [e.name for e in stage.grid_inventory() if e.present and e.name]
            if stage is not None
            else []
        )
        known = sorted({grid.name for grid in experiment.grids} | set(present))
        if grid_name not in known:
            return {
                "error_type": "not_found",
                "error": f"No grid named {grid_name!r}.",
                "grid_names": known,
            }
        why = grid_rename_refusal(experiment, grid_name, name)
        if why:
            return {"error_type": "invalid", "error": why}

        def rename() -> Optional[str]:
            try:
                rename_grid(experiment, stage, grid_name, name)
            except Exception as exc:  # noqa: BLE001 - the hardware's refusal, said
                return str(exc)
            experiment.save()
            self._grids_changed()
            return None

        failed = self._agent_call(rename)
        if failed:
            return {
                "error_type": "not_renamed",
                "error": f"Could not rename {grid_name}: {failed}",
            }
        return {"applied": True, "grid_name": name, "was": grid_name}

    def _grid_record(self, grid_name: str):
        """The experiment and its record of *grid_name*, or a refusal."""
        experiment = self._experiment
        if experiment is None:
            return (
                None,
                None,
                {"error_type": "not_found", "error": "No experiment is loaded."},
            )
        grid = experiment.get_grid_by_name(grid_name)
        if grid is None:
            return (
                None,
                None,
                {
                    "error_type": "not_found",
                    **self._no_such_grid(experiment, grid_name),
                },
            )
        return experiment, grid, None

    def set_grid_quality(
        self, grid_name: str, verdict: str, reason: str = ""
    ) -> Dict[str, Any]:
        """Judge a grid, as its card's quality menu does: GOOD, REWORK, FAILED,
        or UNASSESSED to clear it, with an optional reason. Signed as the
        agent's; saved."""
        from fibsem.applications.autolamella.proposals import agent_author
        from fibsem.applications.autolamella.structures import Verdict

        experiment, grid, refusal = self._grid_record(grid_name)
        if refusal:
            return refusal
        names = [v.name for v in Verdict]
        if str(verdict).upper() not in names:
            return {
                "error_type": "invalid",
                "error": f"verdict is one of {', '.join(names)}, not {verdict!r}.",
            }
        state = Verdict[str(verdict).upper()]

        def judge():
            grid.quality.set_defect(
                description=reason or "", state=state, author=agent_author("remote")
            )
            experiment.save()
            self._grids_changed()

        self._agent_call(judge)
        return {
            "applied": True,
            "grid_name": grid_name,
            "quality": grid.quality.to_dict(),
        }

    def set_grid_note(self, grid_name: str, note: str) -> Dict[str, Any]:
        """Keep a note on a grid's record, as the Grids tab's Note does; an empty
        note clears it. Apart from the verdict, so judging the grid again keeps
        it. Saved."""
        experiment, grid, refusal = self._grid_record(grid_name)
        if refusal:
            return refusal

        def keep():
            grid.description = note
            experiment.save()
            self._grids_changed()

        self._agent_call(keep)
        return {"applied": True, "grid_name": grid_name, "note": note}

    def remove_grid(self, grid_name: str) -> Dict[str, Any]:
        """Stop tracking a grid, as the Grids tab's Remove does: its files stay,
        its lamellae stay with no grid. Refused while a workflow runs (its queue
        holds grids by name). Saved."""
        experiment, grid, refusal = self._grid_record(grid_name)
        if refusal:
            return refusal
        refusal = self._not_while_running("remove grids")
        if refusal:
            return refusal

        def remove():
            orphaned = [p.name for p in experiment.get_lamellae_for_grid(grid)]
            experiment.remove_grid(grid_name)
            experiment.save()
            self._grids_changed()
            return orphaned

        orphaned = self._agent_call(remove)
        return {"applied": True, "grid_name": grid_name, "items_without_grid": orphaned}

    def move_to_item_pose(self, item_name: str, pose_name: str) -> Dict[str, Any]:
        """Move the stage to one of a lamella's poses, as the tab's Move to does,
        through the safe absolute move (flat before a large rotation). Runs on
        the caller's thread, which holds control of the microscope."""
        experiment, microscope, refusal = self._editable()
        if refusal:
            return refusal
        lamella, refusal = find_item(experiment, item_name)
        if refusal:
            return {"error_type": "not_found", **refusal}
        pose = dict(lamella.poses).get(pose_name)
        if pose is None or pose.stage_position is None:
            return {
                "error_type": "not_found",
                "error": f"{item_name} has no {pose_name!r} pose with a position.",
                "pose_names": [
                    name
                    for name, p in dict(lamella.poses).items()
                    if p is not None and p.stage_position is not None
                ],
            }
        microscope.safe_absolute_stage_movement(pose.stage_position)
        return {
            "applied": True,
            "item_name": item_name,
            "pose": pose_name,
            "stage_position": microscope.get_stage_position().to_dict(),
        }

    def set_item_pose(self, item_name: str, pose_name: str) -> Dict[str, Any]:
        """Set the stage position now as one of a lamella's poses, as the tab's
        Set pose does. Setting MILLING moves the lamella: its milling angle and
        fluorescence pose follow. Saved. Not while a run is on: the run moves
        the stage, so "now" is wherever it happens to be."""
        from fibsem.applications.autolamella.poses import set_lamella_pose

        experiment, microscope, refusal = self._editable()
        if refusal:
            return refusal
        refusal = self._not_while_running("set a pose from the stage")
        if refusal:
            return refusal
        lamella, refusal = find_item(experiment, item_name)
        if refusal:
            return {"error_type": "not_found", **refusal}
        known = list(dict(lamella.poses))
        if pose_name not in known:
            return {
                "error_type": "not_found",
                "error": f"{item_name} has no pose named {pose_name!r}.",
                "pose_names": known,
            }
        state = microscope.get_microscope_state()

        def set_pose():
            moved = set_lamella_pose(microscope, lamella, pose_name, state)
            experiment.save()
            return moved

        followed = self._agent_call(set_pose)
        return {
            "applied": True,
            "item_name": item_name,
            "pose": pose_name,
            "fluorescence_pose_followed": bool(followed),
            "item": self.item_detail(item_name),
        }

    def set_item_objective_position(self, item_name: str) -> Dict[str, Any]:
        """Store the FM objective's position now on a lamella's fluorescence pose,
        as the tab's Use current does: where it is if inserted, else its focus
        position. Saved."""
        experiment, microscope, refusal = self._editable()
        if refusal:
            return refusal
        lamella, refusal = find_item(experiment, item_name)
        if refusal:
            return {"error_type": "not_found", **refusal}
        fm = getattr(microscope, "fm", None)
        if fm is None:
            return {"error_type": "not_found", "error": "This microscope has no FM."}
        if lamella.fluorescence_pose is None:
            return {
                "error_type": "not_found",
                "error": f"{item_name} has no fluorescence pose.",
            }
        from fibsem.applications.autolamella.poses import current_objective_position

        value = current_objective_position(fm)
        if value is None:
            return {
                "error_type": "not_found",
                "error": "The objective position is unavailable.",
            }

        def store():
            lamella.fluorescence_pose.objective_position = value
            experiment.save()

        self._agent_call(store)
        return {"applied": True, "item_name": item_name, "objective_position": value}

    def move_objective_to_item(self, item_name: str) -> Dict[str, Any]:
        """Move the FM objective to the position stored on a lamella's
        fluorescence pose, as the tab's Move to does: the objective only, not
        the stage. Runs on the caller's thread, which holds control."""
        from fibsem.applications.autolamella.poses import stored_objective_position

        experiment, microscope, refusal = self._editable()
        if refusal:
            return refusal
        lamella, refusal = find_item(experiment, item_name)
        if refusal:
            return {"error_type": "not_found", **refusal}
        fm = getattr(microscope, "fm", None)
        if fm is None:
            return {"error_type": "not_found", "error": "This microscope has no FM."}
        try:
            position = stored_objective_position(fm, lamella)
        except ValueError as exc:
            return {"error_type": "invalid", "error": str(exc)}
        fm.objective.move_absolute(position)
        return {
            "applied": True,
            "item_name": item_name,
            "objective_position": fm.objective.position,
        }

    def item_detail(self, item_name: str) -> Dict[str, Any]:
        """The durable facts about one item, as a curated snapshot.

        One read for everything a supervisor judges an item by: status,
        geometry (POI, alignment area, milling angle), and where its poses
        put the stage. Curated rather than a ``to_dict`` dump — this payload
        serves agents and the monitor dashboard, so every field in it is
        wire contract; internals stay internal. Pointer facts live on their
        own surfaces (files: task_outputs; history: task_history).
        """
        experiment = self._experiment
        if experiment is None:
            return {"available": False, "item_name": item_name}
        lamella, refusal = find_item(experiment, item_name)
        if refusal:
            return {"available": False, "item_name": item_name, **refusal}
        from fibsem.applications.autolamella.server.events import to_plain

        return {
            "available": True,
            "item_name": lamella.name,
            "id": lamella.id,
            "is_failure": bool(lamella.is_failure),
            "description": lamella.description,
            # metres, in the milling coordinate system (+y up, origin centre)
            "poi": lamella.poi.to_dict() if lamella.poi is not None else None,
            # fractions of the FIB frame, origin top-left
            "alignment_area": lamella.alignment_area.to_dict()
            if lamella.alignment_area is not None
            else None,
            "milling_angle": lamella.milling_angle,
            "defect": to_plain(lamella.defect.to_dict())
            if getattr(lamella, "defect", None) is not None
            else None,
            "poses": item_pose_positions(lamella),
            # The item document's version: patches echo it (see apply_item_patch).
            "version": item_fields_version(lamella),
        }

    # --- task configs (FIB-864, read side) --------------------------------------

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

    def item_task_config(self, item_name: str, task_name: str) -> Dict[str, Any]:
        """One item's own copy of a task config — what its run executes."""
        experiment = self._experiment
        if experiment is None:
            return {
                "available": False,
                "item_name": item_name,
                "task_name": task_name,
            }
        lamella, refusal = find_item(experiment, item_name)
        if refusal:
            return {
                "available": False,
                "item_name": item_name,
                "task_name": task_name,
                **refusal,
            }
        config = dict(lamella.task_config).get(task_name)
        if config is None:
            return {
                "available": True,
                "item_name": item_name,
                "task_name": task_name,
                "error": f"No task config named {task_name!r} on {item_name!r}.",
                "task_names": list(lamella.task_config.keys()),
            }
        document = self._config_document(task_name, config, level="item")
        document["item_name"] = item_name
        return document

    def describe_action(
        self, tool: str, arguments: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """A config edit in an agent's request, as the config's form names it
        (FIB-1403): each setting in display units, old -> new. None for any
        other action, or a config this experiment doesn't have."""
        from fibsem.server.request_text import describe_patch

        experiment = self._experiment
        if experiment is None:
            return None
        task_name = arguments.get("task_name")
        if tool == "item_update_task_config":
            lamella = experiment.get_lamella_by_name(arguments.get("item_name"))
            configs = getattr(lamella, "task_config", None) if lamella else None
        elif tool == "protocol_update_task_config":
            protocol = getattr(experiment, "task_protocol", None)
            configs = getattr(protocol, "task_config", None) if protocol else None
        else:
            return None
        config = dict(configs).get(task_name) if configs is not None else None
        if config is None:
            return None
        return describe_patch(config, arguments.get("patch"))

    def apply_item_patch(
        self,
        item_name: str,
        patch: Dict[str, Any],
        version: str,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Patch an item's own document — geometry, verdict, notes.

        The item-level counterpart of the task-config patches: same engine,
        same version dance (``version`` comes from :meth:`item_detail`), but
        the document is what the lamella IS rather than how a step runs, and
        only the ITEM_PATCH_FIELDS are editable. Tasks re-record geometry at
        their own moments (a fiducial task rewrites the alignment area at its
        end) — that rotation stales pending patches, which is the correct
        outcome, and it means an edit here is the value the NEXT run starts
        from, not a permanent override.
        """
        host = self._host
        if not hasattr(host, "request_apply_item_patch"):
            return {"available": False, "applied": False}
        outcome = host.request_apply_item_patch(item_name, dict(patch), str(version))
        result = outcome.result(timeout=timeout)
        result["available"] = True
        return result

    # --- propose and review (FIB-950) -----------------------------------------

    def reviews(self) -> Dict[str, Any]:
        """Every proposal waiting for a decision: the Review tab's inbox, derived
        from the experiment the same way the tab derives it, with the reference
        image each proposal's values sit on as an agent-sized preview."""
        experiment = self._experiment
        if experiment is None:
            return {"available": False, "reviews": []}
        from fibsem.applications.autolamella.server.events import to_plain
        from fibsem.applications.autolamella.server.prompts import _preview_payload
        from fibsem.applications.autolamella.ui.review_tab_widget import (
            _load_reference_image,
            is_gated,
            review_preview,
            waiting_on,
        )

        protocol = getattr(experiment, "task_protocol", None)

        def describe(item, task_name, proposal):
            doc = to_plain(proposal.to_dict())
            doc.update(
                {
                    "item_id": item.id,
                    "item_name": item.name,
                    "task_name": task_name,
                    # what this is; review_decide passes it back. The run is
                    # here as a fact about it, and still names it for a caller
                    # that has not learned the id.
                    "proposal_id": proposal.id,
                    "task_id": proposal.task_id,
                    # The task that made it is stopped on the answer: nothing
                    # else runs until it is decided, so it comes first.
                    "holding_the_run": bool(proposal.asking),
                    "gated": is_gated(experiment, task_name, item),
                    "waiting_on": waiting_on(experiment, task_name, item),
                }
            )
            image = _load_reference_image(experiment, item, proposal)
            # a fluorescence result is shown as its channel composite
            preview = review_preview(image) if image is not None else None
            doc["reference_image"] = (
                _preview_payload(preview) if preview is not None else None
            )
            electron = _load_reference_image(
                experiment, item, proposal, "reference_image_eb"
            )
            doc["reference_image_eb"] = (
                _preview_payload(electron) if electron is not None else None
            )
            return doc

        reviews = [describe(*entry) for entry in experiment.pending_proposals()]
        # Applied by the producer (advise mode), nobody has looked: an agent
        # acknowledges one with Confirmed and no values, which writes nothing.
        to_check = [describe(*entry) for entry in experiment.proposals_to_check()]
        return {"available": True, "reviews": reviews, "to_check": to_check}

    def decide(
        self,
        item_id: str,
        task_name: str,
        outcome: str,
        values: Optional[Dict[str, Any]] = None,
        reason: str = "",
        author: str = "",
        task_id: str = "",
        proposal_id: str = "",
    ) -> Dict[str, Any]:
        """Decide a pending proposal, exactly as the Review tab would: the same
        Experiment.decide, on the main thread, blocking until applied. The
        author is recorded as the agent, never as the operator.

        ``proposal_id`` is the proposal the agent was shown (from
        get_pending_reviews), and what the decision is checked against: a task
        may ask several questions in one run, and only the id tells the one
        that was shown from the one that replaced it. ``task_id``, the run, is
        accepted in its place from a caller that has not learned the id, and
        is checked the way it always was. One of them is needed; a stale one
        is refused."""
        experiment = self._experiment
        if experiment is None:
            return {"available": False, "applied": False, "reason": "No experiment."}
        from fibsem.applications.autolamella.proposals import (
            Decision,
            DecisionOutcome,
            _decode_values,  # the wire shape is the stored shape
            agent_author,
        )

        try:
            decided = DecisionOutcome[outcome]
        except KeyError:
            return {
                "available": True,
                "applied": False,
                "invalid_value": f"outcome must be Confirmed or Rejected, not {outcome!r}",
            }
        decision = Decision(
            outcome=decided,
            author=agent_author(author or "remote"),
            values=_decode_values(values or {}),
            reason=reason,
            via="server",
            task_id=str(task_id or ""),
            proposal_id=str(proposal_id or ""),
        )
        result = experiment.decide(item_id, task_name, decision)
        doc = result.to_dict()
        doc["available"] = True
        if result.applied:
            try:
                experiment.save()
            except Exception:
                logging.exception(
                    "saving the experiment after an agent decision failed"
                )
            if self._event_buffer is not None:
                self._event_buffer.append(
                    "review_decided",
                    {
                        "item_id": item_id,
                        "task_name": task_name,
                        "outcome": decided.name,
                        "author": str(decision.author),
                    },
                )
        return doc

    def add_note(
        self,
        text: str,
        item_name: Optional[str] = None,
        reply_to: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Put an agent observation on the record, or its answer to the
        operator's message (``reply_to``, the message's id, FIB-1423).

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
        if self._event_buffer is not None:
            # With the item's id, so the record files the note under the item
            # it is about, not whichever one is running (FIB-1328).
            payload = {"text": text, "item_name": item_name, "item_id": item_id}
            if reply_to:
                payload["reply_to"] = reply_to
            self._event_buffer.append("agent_note", payload)
        return {"available": True, "recorded": True}

    def record_event(self, kind: str, payload: Dict[str, Any]) -> None:
        """Put an event on the experiment's record: the agent's requests and
        their outcomes (FIB-1283)."""
        if self._event_buffer is not None:
            self._event_buffer.append(kind, payload)

    def reorder_milling_stages(
        self,
        level: str,
        item_name: str,
        task_name: str,
        milling_key: str,
        order: List[str],
        version: str,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Reorder one milling config's stages — structure, so a verb.

        Same rules as the config patches: version-guarded on the GUI thread,
        refused while that task runs for that item, recorded on the event
        stream. Same-set-by-name: this can never add, drop, or duplicate a
        stage.
        """
        host = self._host
        if not hasattr(host, "request_reorder_milling_stages"):
            return {"available": False, "applied": False}
        if level == "item":
            manager = self._manager
            if manager is not None:
                running = any(
                    item.item_name == item_name
                    and item.task_name == task_name
                    and item.status is AutoLamellaTaskStatus.InProgress
                    for item in manager.queue.items
                )
                if running:
                    return {
                        "available": True,
                        "applied": False,
                        "error_type": "task_running",
                        "error": f"{task_name!r} is running for {item_name!r} "
                        "and has already copied its config.",
                    }
        outcome = host.request_reorder_milling_stages(
            level, item_name, task_name, milling_key, list(order), str(version)
        )
        result = outcome.result(timeout=timeout)
        result["available"] = True
        return result

    def apply_protocol_to_item(
        self,
        item_name: str,
        task_names: Optional[List[str]] = None,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Re-copy protocol task configs onto an existing item.

        Protocol-level edits only reach items created after them; this verb
        brings an existing item up to date — the agent's form of the editor's
        apply dialog. Wholesale by design (it IS "replace with the defaults"),
        so no version dance; a task currently running for this item is
        refused like any other config write.
        """
        host = self._host
        if not hasattr(host, "request_apply_protocol_to_item"):
            return {"available": False, "applied": False}
        manager = self._manager
        if manager is not None:
            experiment = self._experiment
            protocol = (
                getattr(experiment, "task_protocol", None) if experiment else None
            )
            config_map = getattr(protocol, "task_config", None) if protocol else {}
            guard_names = (
                list(task_names) if task_names else list((config_map or {}).keys())
            )
            running = any(
                item.item_name == item_name
                and item.task_name in guard_names
                and item.status is AutoLamellaTaskStatus.InProgress
                for item in manager.queue.items
            )
            if running:
                return {
                    "available": True,
                    "applied": False,
                    "error_type": "task_running",
                    "error": f"A task in {guard_names!r} is running for "
                    f"{item_name!r} right now and has already copied its "
                    "config. Apply after it finishes.",
                }
        outcome = host.request_apply_protocol_to_item(item_name, task_names)
        result = outcome.result(timeout=timeout)
        result["available"] = True
        return result

    def apply_protocol_task_config_patch(
        self,
        task_name: str,
        patch: Dict[str, Any],
        version: str,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Patch a task's protocol-level defaults — what new items copy.

        Same version-guarded, GUI-thread apply as the per-item patch. No
        running-task guard: a running task holds its item's copy, so a
        protocol edit can never touch it — the edit reaches items created
        (or re-copied) after it lands.
        """
        host = self._host
        if not hasattr(host, "request_apply_protocol_task_config_patch"):
            return {"available": False, "applied": False}
        outcome = host.request_apply_protocol_task_config_patch(
            task_name, dict(patch), str(version)
        )
        result = outcome.result(timeout=timeout)
        result["available"] = True
        return result

    @staticmethod
    def _config_document(task_name: str, config, level: str) -> Dict[str, Any]:
        """One config as a wire document with its version token.

        The full ``to_dict``, not a curated snapshot — the document is the
        contract for editing (you cannot patch what you cannot read). The
        ``version`` hashes the serialized content, so a write names exactly
        the state it read: the config's version, refused as stale on mismatch.
        """
        from fibsem.applications.autolamella.server.events import to_plain

        return {
            "available": True,
            "task_name": task_name,
            "level": level,
            "version": config_version(config),
            "config": to_plain(config.to_dict()),
        }

    def apply_item_task_config_patch(
        self,
        item_name: str,
        task_name: str,
        patch: Dict[str, Any],
        version: str,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Patch one item's task config, as an operator edit would land.

        The version (from :meth:`item_task_config`) names the state the patch
        was written against; the check happens on the GUI thread beside the
        apply, so an edit landing in between is refused as stale, never
        half-merged. A task currently running for this item is refused too —
        it already copied its config, so the edit would take effect never,
        and a refusal is more honest than a silent no-op. Pending tasks pick
        the change up when they start (the ``set_supervision`` semantics).
        """
        host = self._host
        if not hasattr(host, "request_apply_task_config_patch"):
            return {"available": False, "applied": False}
        manager = self._manager
        if manager is not None:
            running = any(
                item.item_name == item_name
                and item.task_name == task_name
                and item.status is AutoLamellaTaskStatus.InProgress
                for item in manager.queue.items
            )
            if running:
                return {
                    "available": True,
                    "applied": False,
                    "error_type": "task_running",
                    "error": f"{task_name!r} is running for {item_name!r} right "
                    "now and has already copied its config — the edit would "
                    "never take effect. Patch it after the task finishes.",
                }
        outcome = host.request_apply_task_config_patch(
            item_name, task_name, dict(patch), str(version)
        )
        result = outcome.result(timeout=timeout)
        result["available"] = True
        return result

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
        if getattr(self._host, "image_widget", None) is None:
            return {"available": False, "sem": None, "fib": None}
        return {
            "available": True,
            "sem": self._display_frame("eb_image"),
            "fib": self._display_frame("ib_image"),
        }

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
        never waits on the command lock.
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

    def stop_task(self, item_name: str, task_name: str) -> Dict[str, Any]:
        """Stop the task now running and carry on with the rest of the queue,
        as the timeline's Stop Task does. *item_name* and *task_name* name the
        task meant: a stop that lands after it finished is refused rather than
        stopping the next one."""
        host = self._host
        manager = self._manager
        if manager is None or not hasattr(host, "stop_current_task"):
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
        host.stop_current_task()
        return {"stopped": True, "item_name": item_name, "task_name": task_name}

    def start_workflow(
        self,
        task_names: List[str],
        item_names: Optional[List[str]] = None,
        timeout: float = 15.0,
    ) -> Dict[str, Any]:
        """Start a workflow as the Run button would — the batch-review opener.

        Marshalled to the GUI thread (which owns worker creation and window
        chrome); validation refusals come back structured with the valid
        names. ``item_names`` of None means every item in the experiment.
        The agent at Drive is the consent that stands in for the Run click's
        confirm dialog.
        """
        host = self._host
        if not hasattr(host, "request_start_workflow"):
            return {"available": False, "started": False}
        outcome = host.request_start_workflow(list(task_names), item_names)
        result = outcome.result(timeout=timeout)
        result["available"] = True
        if result.get("started"):
            # Who started it: the run's own workflow_started fires on its
            # worker thread, which carries no one's mark (FIB-1414).
            self.record_event(
                "workflow_start",
                {"task_names": list(task_names), "item_names": item_names},
            )
        return result

    def grid_workflow_plan(
        self,
        task_names: List[str],
        grid_names: Optional[List[str]] = None,
        screen_all: bool = False,
    ) -> Dict[str, Any]:
        """What a grid run would execute, for a confirmation — no hardware.

        The ``(grid, step)`` sequence the manager builds, grid-outer with each
        grid's load first, validated against the grid protocol and the
        recorded grids exactly as :meth:`start_grid_workflow` will. For
        ``screen_all`` the grids are whatever the inventory finds at run time,
        so the plan names the known grids and says so rather than guessing.
        Exchanges are deliberately not counted: whether a grid is loaded
        is a live stage fact this facade does not read.
        """
        from fibsem.applications.autolamella.workflows.tasks.grid.manager import (
            plan_grid_run,
        )

        experiment = self._experiment
        if experiment is None or experiment.task_protocol is None:
            return {
                "available": False,
                "valid": False,
                "reason": "no experiment is loaded",
            }
        known_tasks = list(experiment.grid_protocol.ordered_task_names)
        unknown = [t for t in task_names if t not in known_tasks]
        if not task_names or unknown:
            return {
                "available": True,
                "valid": False,
                "reason": f"unknown grid tasks: {unknown!r}" if unknown else "no tasks",
                "task_names": known_tasks,
            }
        known_grids = [g.name for g in experiment.grids]
        if screen_all:
            if grid_names is not None:
                return {
                    "available": True,
                    "valid": False,
                    "reason": "screen_all runs over every present grid; "
                    "do not name grids with it",
                }
            return {
                "available": True,
                "valid": True,
                "screen_all": True,
                "task_names": list(task_names),
                "grid_names": known_grids,
                "note": "inventory first; the grids run are those present at run time",
                "steps": [
                    {"grid": grid, "step": step}
                    for grid, step in plan_grid_run(list(task_names), known_grids)
                ],
            }
        if grid_names is None:
            grid_names = known_grids
        missing = [n for n in grid_names if n not in known_grids]
        if missing:
            return {
                "available": True,
                "valid": False,
                "reason": f"unknown grids: {missing!r}",
                "grid_names": known_grids,
            }
        return {
            "available": True,
            "valid": True,
            "screen_all": False,
            "task_names": list(task_names),
            "grid_names": list(grid_names),
            "steps": [
                {"grid": grid, "step": step}
                for grid, step in plan_grid_run(list(task_names), list(grid_names))
            ],
        }

    def start_grid_workflow(
        self,
        task_names: List[str],
        grid_names: Optional[List[str]] = None,
        screen_all: bool = False,
        unload_at_end: bool = False,
        report_at_end: bool = False,
        timeout: float = 15.0,
    ) -> Dict[str, Any]:
        """Start a grid run as the Grids view's Run (or Screen all grids) would,
        with the confirmation's end-of-run options: unload the last grid, write
        the report.

        Marshalled to the GUI thread like :meth:`start_workflow`; shares its
        worker slot, so it is refused while any workflow — lamella or grid —
        is running. The agent at Drive stands in for the preflight dialog's
        Run click; :meth:`grid_workflow_plan` is that dialog's content.
        """
        host = self._host
        if not hasattr(host, "request_start_grid_workflow"):
            return {"available": False, "started": False}
        outcome = host.request_start_grid_workflow(
            list(task_names),
            grid_names,
            inventory_first=screen_all,
            unload_at_end=unload_at_end,
            report_at_end=report_at_end,
        )
        result = outcome.result(timeout=timeout)
        result["available"] = True
        if result.get("started"):
            self.record_event(
                "workflow_start",
                {"task_names": list(task_names), "grid_names": grid_names},
            )
        return result

    def set_supervision(self, task_name: str, supervise: bool) -> Dict[str, Any]:
        """Set whether ``task_name`` asks for supervision, in the live protocol.

        The workflow reads this at every decision point (never a snapshot), so
        a change takes effect at the next prompt-or-proceed choice — exactly
        the mid-run behaviour the GUI's supervised/automated toggle has. The
        GUI's own indicators refresh on the next task transition rather than
        instantly.

        A supervised question goes to the operator and the agent alike, and the
        first answer wins (FIB-1035): there is no saying whose it is.
        """
        experiment = self._experiment
        protocol = getattr(experiment, "task_protocol", None) if experiment else None
        config = getattr(protocol, "workflow_config", None) if protocol else None
        if config is None:
            return {"available": False, "applied": False}
        for task in config.tasks:
            if task.name == task_name:
                task.attention = (
                    Attention.supervised if supervise else Attention.automated
                )
                # Saved with the protocol, as the Workflow tab's chip saves it and
                # as the schedule is: an agent's change survives a reload too.
                saved = True
                try:
                    experiment.save(save_protocol=True)
                except Exception:
                    saved = False
                    logging.exception(
                        "save after supervision change failed; the change is "
                        "applied in memory but not yet on disk"
                    )
                if self._event_buffer is not None:
                    self._event_buffer.append(
                        "workflow_changed",
                        {
                            "field": "attention",
                            "task_name": task_name,
                            "supervise": bool(supervise),
                        },
                    )
                return {
                    "available": True,
                    "applied": True,
                    "saved": saved,
                    "task_name": task_name,
                    "supervise": bool(supervise),
                }
        return {
            "available": True,
            "applied": False,
            "error": f"No task named {task_name!r} in the protocol.",
            "task_names": [t.name for t in config.tasks],
        }

    def set_task_schedule(
        self, task_name: str, scheduled_at: Optional[str]
    ) -> Dict[str, Any]:
        """Set (or clear) when ``task_name`` may start, in the live workflow.

        The workflow reads the schedule at each task start (never a snapshot),
        so a change takes effect at the next start — the ``set_supervision``
        semantics, and like it this mutates the live config directly (an
        atomic field rebind; no GUI marshal needed), and like it the change is
        persisted immediately.

        ``scheduled_at`` is ISO-8601 (naive = local time; an offset is
        normalized by the workflow) or ``None`` to clear.
        """
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
        protocol = getattr(experiment, "task_protocol", None) if experiment else None
        config = getattr(protocol, "workflow_config", None) if protocol else None
        if config is None:
            return {"available": False, "applied": False}
        for task in config.tasks:
            if task.name == task_name:
                task.scheduled_at = parsed
                saved = True
                try:
                    experiment.save(save_protocol=True)
                except Exception:
                    saved = False
                    logging.exception(
                        "save after schedule change failed; the change is "
                        "applied in memory but not yet on disk"
                    )
                if self._event_buffer is not None:
                    self._event_buffer.append(
                        "workflow_changed",
                        {
                            "field": "scheduled_at",
                            "task_name": task_name,
                            "scheduled_at": parsed.isoformat() if parsed else None,
                        },
                    )
                return {
                    "available": True,
                    "applied": True,
                    "saved": saved,
                    "task_name": task_name,
                    "scheduled_at": parsed.isoformat() if parsed else None,
                }
        return {
            "available": True,
            "applied": False,
            "error": f"No task named {task_name!r} in the protocol.",
            "task_names": [t.name for t in config.tasks],
        }

    def set_workflow_options(self, **options: Any) -> Dict[str, Any]:
        """Set the workflow's run options, as its options panel does, and like
        the operator's it persists with the protocol. Any of:

        - ``show_run_summary``: whether the run summary dialog opens when a run
          ends. Off leaves the run-end toast and its Show summary link: an agent
          starting many short runs turns it off so no dialog stands over the
          window (FIB-1376).
        - ``turn_beams_off``: whether the beams are turned off when a run ends.
        - ``review_wait``: how long a run whose remaining work all waits on a
          review keeps waiting, in seconds of inactivity; None waits until it is
          stopped.

        Each is read when it applies (a run's end, a review wait), so a change
        takes effect from then, mid-run included; plain rebinds, so no GUI
        marshal, like ``set_task_schedule``.
        """
        experiment = self._experiment
        protocol = getattr(experiment, "task_protocol", None) if experiment else None
        if protocol is None:
            return {"available": False, "applied": False}
        for name, value in options.items():
            setattr(protocol.options, name, value)
        saved = True
        try:
            experiment.save(save_protocol=True)
        except Exception:
            saved = False
            logging.exception(
                "save after workflow option change failed; the change is "
                "applied in memory but not yet on disk"
            )
        if self._event_buffer is not None:
            for name, value in options.items():
                self._event_buffer.append(
                    "workflow_changed", {"field": name, "value": value}
                )
        return {"available": True, "applied": True, "saved": saved, **options}

    # -- the workflow's tasks: the Workflow tab's task list and the protocol
    # editor's Add and Remove. Refused while a workflow runs: the run reads them.

    def _experiment_changed(self) -> None:
        """Tell the app's views an agent changed the experiment's protocol, so
        no editor writes back a list it drew before the change."""
        signal = getattr(self._host, "experiment_update_signal", None)
        if signal is not None:
            signal.emit()

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
        """Save the protocol after an edit (GUI thread), redraw, and put it on
        the record. Whether it saved."""
        saved = True
        try:
            experiment.save(save_protocol=True)
        except Exception:
            saved = False
            logging.exception("save after a protocol change failed; applied in memory")
        self._experiment_changed()
        if self._event_buffer is not None:
            self._event_buffer.append("workflow_changed", change)
        return saved

    def add_task(self, task_type: str, task_name: str) -> Dict[str, Any]:
        """Add a task of a registered type, as the protocol editor's Add Task
        does: its default settings in the protocol, at the end of the workflow,
        and a copy on every lamella."""
        from fibsem.applications.autolamella.workflows.tasks import get_tasks

        experiment, _, refusal = self._workflow_edit("add tasks")
        if refusal:
            return refusal
        types = sorted(get_tasks())
        if task_type not in types:
            return {
                "error_type": "not_found",
                "error": f"No task type {task_type!r}.",
                "task_type_names": types,
            }

        def add():
            try:
                experiment.add_task(task_type, task_name)
            except ValueError as exc:
                return str(exc)
            return self._protocol_changed(
                experiment, {"field": "tasks", "added": task_name.strip()}
            )

        saved = self._agent_call(add)
        if isinstance(saved, str):
            return {"error_type": "invalid", "error": saved}
        return {
            "applied": True,
            "saved": saved,
            "task_name": task_name.strip(),
            "task_type": task_type,
            "task_names": experiment.task_protocol.workflow_config.workflow,
        }

    def remove_task(self, task_name: str) -> Dict[str, Any]:
        """Remove a task from the protocol, the workflow and every lamella, as
        the protocol editor's Remove does. Task history stays: it is the record
        of what ran."""
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
        names = config.workflow
        if sorted(order) != sorted(names):
            return {
                "error_type": "invalid",
                "error": "Give every task in the workflow, each once.",
                "task_names": names,
            }
        by_name = {task.name: task for task in config.tasks}
        reordered = [by_name[name] for name in order]
        issues = type(config)(tasks=reordered).validate()
        if issues:
            return {"error_type": "invalid", "error": " ".join(issues)}

        def reorder():
            config.tasks[:] = reordered
            return self._protocol_changed(
                experiment, {"field": "order", "order": order}
            )

        saved = self._agent_call(reorder)
        return {"applied": True, "saved": saved, "task_names": config.workflow}

    def set_task_requirements(
        self,
        task_name: str,
        requires: Optional[List[str]] = None,
        required: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Set what a workflow task requires (the tasks a lamella must have
        completed first) and whether it is required or optional, as the
        Workflow tab's Edit Task does. Refused when a requirement is not an
        earlier task."""
        experiment, config, refusal = self._workflow_edit("edit tasks")
        if refusal:
            return refusal
        task = next((t for t in config.tasks if t.name == task_name), None)
        if task is None:
            return {
                "error_type": "not_found",
                "error": f"No task named {task_name!r} in the workflow.",
                "task_names": config.workflow,
            }
        if requires is not None:
            earlier = config.workflow[: config.workflow.index(task_name)]
            wrong = [name for name in requires if name not in earlier]
            if wrong:
                return {
                    "error_type": "invalid",
                    "error": f"{task_name} can only require tasks before it: "
                    f"{', '.join(earlier) or 'none'}; not {', '.join(wrong)}.",
                }

        def edit():
            if requires is not None:
                task.requires = list(requires)
            if required is not None:
                task.required = bool(required)
            return self._protocol_changed(
                experiment,
                {
                    "field": "requirements",
                    "task_name": task_name,
                    "requires": task.requires,
                    "required": task.required,
                },
            )

        saved = self._agent_call(edit)
        return {
            "applied": True,
            "saved": saved,
            "task_name": task_name,
            "requires": list(task.requires),
            "required": task.required,
        }

    def set_lamella_defaults(self, defaults: Dict[str, Any]) -> Dict[str, Any]:
        """Set what every new lamella starts with, as the protocol editor's
        Lamella Template does: ``use_petname``, ``name_prefix``, ``poi`` ({x, y}
        image fraction) and ``alignment_area`` ({left, top, width, height}
        fractions). Only the keys given change; null or {} clears ``poi``
        or ``alignment_area``. Existing lamellae keep theirs."""
        from fibsem.applications.autolamella.structures import LamellaDefaultConfig

        experiment, _, refusal = self._workflow_edit("edit the lamella defaults")
        if refusal:
            return refusal
        protocol = experiment.task_protocol
        known = ("use_petname", "name_prefix", "poi", "alignment_area")
        unknown = [key for key in defaults if key not in known]
        if not defaults or unknown:
            return {
                "error_type": "invalid",
                "error": f"Set any of {', '.join(known)}"
                + (f"; not {', '.join(unknown)}." if unknown else "."),
            }
        # An empty poi or alignment_area clears it, as null does.
        cleared = {k: None if v == {} else v for k, v in defaults.items()}
        merged = {**protocol.lamella_defaults.to_dict(), **cleared}
        try:
            updated = LamellaDefaultConfig.from_dict(merged)
            if not isinstance(updated.use_petname, bool) or not isinstance(
                updated.name_prefix, str
            ):
                raise ValueError("use_petname is true or false; name_prefix is text.")
        except (KeyError, TypeError, ValueError) as exc:
            return {"error_type": "invalid", "error": f"Not lamella defaults: {exc}"}

        def edit():
            protocol.lamella_defaults = updated
            return self._protocol_changed(
                experiment, {"field": "lamella_defaults", "value": updated.to_dict()}
            )

        saved = self._agent_call(edit)
        return {"applied": True, "saved": saved, "lamella_defaults": updated.to_dict()}

    def save_protocol_as(self, name: str, overwrite: bool = False) -> Dict[str, Any]:
        """Save the experiment's protocol to the protocol folder as *name*, as
        Export Protocol does, so a later experiment can be created with it or
        load it. Refused when the file exists, unless *overwrite*."""
        import os

        from fibsem.applications.autolamella import experiment_setup

        experiment = self._experiment
        protocol = getattr(experiment, "task_protocol", None) if experiment else None
        if protocol is None:
            return {"error_type": "not_found", "error": "No experiment is loaded."}
        try:
            path = experiment_setup.protocol_file_for(name)
        except ValueError as exc:
            return {"error_type": "invalid", "error": str(exc)}
        if os.path.exists(path) and not overwrite:
            return {
                "error_type": "invalid",
                "error": f"{os.path.basename(path)} already exists in the protocol "
                "folder; pass overwrite to replace it.",
                "protocol_names": [
                    p["name"] for p in experiment_setup.list_protocols()
                ],
            }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        protocol.save(path)
        if self._event_buffer is not None:
            self._event_buffer.append(
                "workflow_changed", {"field": "protocol_saved", "path": path}
            )
        return {"saved": True, "name": os.path.basename(path), "path": path}

    # -- the grid protocol: the Protocol tab's grid tasks and the Grids view's
    # task list. Refused while a workflow runs: a grid run reads these configs,
    # shared by every grid.

    def _grid_protocol_edit(self, what: str):
        experiment = self._experiment
        if experiment is None or experiment.task_protocol is None:
            return (
                None,
                None,
                {"error_type": "not_found", "error": "No experiment is loaded."},
            )
        refusal = self._not_while_running(what)
        if refusal:
            return None, None, refusal
        return experiment, experiment.grid_protocol, None

    def _grid_task(self, grid_protocol, task_name: str):
        config = grid_protocol.task_config.get(task_name)
        if config is None:
            return None, {
                "error_type": "not_found",
                "error": f"No grid task named {task_name!r} in the protocol.",
                "task_names": grid_protocol.ordered_task_names,
            }
        return config, None

    def grid_task_config(self, task_name: str) -> Dict[str, Any]:
        """A grid task's settings, with the version a patch echoes."""
        experiment = self._experiment
        if experiment is None or experiment.task_protocol is None:
            return {"error_type": "not_found", "error": "No experiment is loaded."}
        config, refusal = self._grid_task(experiment.grid_protocol, task_name)
        if refusal:
            return refusal
        return self._config_document(task_name, config, "grid_protocol")

    def apply_grid_task_config_patch(
        self, task_name: str, patch: Dict[str, Any], version: str
    ) -> Dict[str, Any]:
        """Patch a grid task's settings with dotted-path edits, as the Protocol
        tab's grid task form does; ``attention`` (automated, supervised) is the
        Grids view's Automated/Supervised chip. Version-guarded like the lamella
        task configs."""
        from fibsem.applications.autolamella.server.events import to_plain
        from fibsem.server.config_patch import PatchError, apply_patch

        experiment, grid_protocol, refusal = self._grid_protocol_edit("edit grid tasks")
        if refusal:
            return refusal
        config, refusal = self._grid_task(grid_protocol, task_name)
        if refusal:
            return refusal

        def edit():
            if config_version(config) != version:
                return {"stale": True}
            try:
                changes = apply_patch(config, patch)
            except PatchError as exc:
                return {"invalid_patch": str(exc), "path": exc.path}
            changed = [
                {"path": p, "old": to_plain(old), "new": to_plain(new)}
                for p, old, new in changes
            ]
            saved = self._protocol_changed(
                experiment,
                {
                    "field": "grid_task_config",
                    "task_name": task_name,
                    "changes": changed,
                },
            )
            return {"changes": changed, "saved": saved}

        result = self._agent_call(edit)
        if "changes" not in result:
            return {"applied": False, **result}
        return {
            "applied": True,
            "task_name": task_name,
            "version": config_version(config),
            **result,
        }

    def reorder_grid_tasks(self, order: List[str]) -> Dict[str, Any]:
        """Put the grid tasks in *order*, as dragging them on the Grids view
        does: every task, each once."""
        experiment, grid_protocol, refusal = self._grid_protocol_edit(
            "reorder grid tasks"
        )
        if refusal:
            return refusal
        names = grid_protocol.ordered_task_names
        if sorted(order) != sorted(names):
            return {
                "error_type": "invalid",
                "error": "Give every grid task, each once.",
                "task_names": names,
            }

        def reorder():
            grid_protocol.order = list(order)
            return self._protocol_changed(
                experiment, {"field": "grid_order", "order": order}
            )

        saved = self._agent_call(reorder)
        return {
            "applied": True,
            "saved": saved,
            "task_names": grid_protocol.ordered_task_names,
        }

    def set_grid_task_requirements(
        self, task_name: str, requires: List[str]
    ) -> Dict[str, Any]:
        """Set what a grid task requires on the same grid, as the Grids view's
        Edit Task does: any other grid task."""
        experiment, grid_protocol, refusal = self._grid_protocol_edit("edit grid tasks")
        if refusal:
            return refusal
        config, refusal = self._grid_task(grid_protocol, task_name)
        if refusal:
            return refusal
        others = [n for n in grid_protocol.ordered_task_names if n != task_name]
        wrong = [name for name in requires if name not in others]
        if wrong:
            return {
                "error_type": "invalid",
                "error": f"{task_name} can require {', '.join(others) or 'no task'}; "
                f"not {', '.join(wrong)}.",
            }

        def edit():
            config.requires = list(requires)
            return self._protocol_changed(
                experiment,
                {
                    "field": "grid_requirements",
                    "task_name": task_name,
                    "requires": requires,
                },
            )

        saved = self._agent_call(edit)
        return {
            "applied": True,
            "saved": saved,
            "task_name": task_name,
            "requires": list(requires),
        }

    def requeue_task(
        self, item_name: str, task_name: str, front: bool = False
    ) -> Dict[str, Any]:
        """Queue ``task_name`` for ``item_name`` (again) — the batch-review verb.

        The queue exists only while a workflow runs, so a re-run is queued into
        the live run: "run 03 again" while the batch continues. Adding a
        duplicate of a completed pair is the queue's own re-run mechanism; the
        item lands at the back (or the front with ``front=True``). A pair
        already waiting is refused (``already_queued``): it runs once.
        """
        manager = self._manager
        if manager is None:
            return {
                "available": True,
                "queued": False,
                "reason": "no workflow is running — the queue exists only during a run",
            }
        experiment = self._experiment
        lamella, refusal = find_item(experiment, item_name)
        if refusal:
            return {
                "available": True,
                "queued": False,
                "reason": refusal["error"],
                "item_names": refusal["item_names"],
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
        # The Add to queue click's grid gate, through the host that owns the
        # stage: a lamella whose grid is not on the stage is refused by name.
        refusal = getattr(self._host, "lamella_run_refusal", None)
        if refusal is not None:
            refused = refusal([lamella])
            if refused:
                return {"available": True, "queued": False, "reason": refused}
        if manager.queue.has_pending_pair(item_name, task_name):
            # A re-ask after a partial failure would queue it twice (FIB-1405).
            return {
                "available": True,
                "queued": False,
                "error_type": "already_queued",
                "reason": f"{item_name} / {task_name} is already waiting in the "
                "queue; it runs once.",
            }
        item = manager.queue.add(item_name, task_name, front=bool(front))
        if item is None:
            return {"available": True, "queued": False, "reason": "add refused"}
        manager.notify_queue_changed()
        self._record_queue_change("requeue_front" if front else "requeue", item)
        return {
            "available": True,
            "queued": True,
            "item_id": item.id,
            "item_name": item_name,
            "task_name": task_name,
            "queue_version": manager.queue.version,
        }

    def queue_grid_tasks(
        self, task_names: List[str], grid_names: List[str]
    ) -> Dict[str, Any]:
        """Add grid tasks for grids to the running grid run, at the end of the
        queue, as the Grids view's Add to queue does: each grid's load, then
        its tasks, less what is already waiting."""
        from fibsem.applications.autolamella.workflows.tasks.grid.manager import (
            GridTaskManager,
        )

        manager = self._manager
        if manager is None:
            return {
                "queued": False,
                "reason": "no workflow is running; the queue exists only during a run",
            }
        if not isinstance(manager, GridTaskManager):
            return {
                "queued": False,
                "reason": "a lamella run is going; grid tasks cannot join it",
            }
        experiment = self._experiment
        known_tasks = list(experiment.grid_protocol.ordered_task_names)
        unknown = [t for t in task_names if t not in known_tasks]
        if not task_names or unknown:
            return {
                "queued": False,
                "reason": f"unknown grid tasks: {unknown!r}" if unknown else "no tasks",
                "task_names": known_tasks,
            }
        known_grids = [g.name for g in experiment.grids]
        missing = [n for n in grid_names if n not in known_grids]
        if not grid_names or missing:
            return {
                "queued": False,
                "reason": f"unknown grids: {missing!r}" if missing else "no grids",
                "grid_names": known_grids,
            }
        pairs = manager.pairs_to_queue(list(task_names), list(grid_names))
        if not pairs:
            return {"queued": False, "reason": "everything named is already queued"}
        added = manager.queue_pairs(pairs)
        for item in added:
            self._record_queue_change("requeue", item)
        return {
            "queued": bool(added),
            "added": [
                {"item_id": i.id, "grid_name": i.item_name, "step": i.task_name}
                for i in added
            ],
            "queue_version": manager.queue.version,
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
        from fibsem.applications.autolamella.workflows.tasks.queue import (
            QUEUE_ACTIONS,
            QueueOp,
        )

        if action not in QUEUE_ACTIONS:
            return {
                "available": True,
                "applied": False,
                "error_type": "invalid",
                "error": f"Unknown action {action!r}.",
                "action_names": list(QUEUE_ACTIONS),
            }
        manager = self._manager
        if manager is None:
            return {
                "available": True,
                "applied": False,
                "reason": "no workflow is running — the queue exists only during a run",
            }
        item = manager.queue.find_pair(item_name, task_name)
        if item is None:
            return {
                "available": True,
                "applied": False,
                "error_type": "not_found",
                "error": f"{item_name} / {task_name} is not in the queue.",
            }
        result = manager.queue.apply(item.id, action)
        if result.op is QueueOp.NOT_PENDING:
            if item.status is AutoLamellaTaskStatus.InProgress:
                state = "is running now; stop it instead"
            elif item.status is AutoLamellaTaskStatus.Removed:
                state = "was already removed"
            else:
                state = f"already ran ({item.status.name})"
            return {
                "available": True,
                "applied": False,
                "error_type": "not_pending",
                "error": f"{item_name} / {task_name} {state}.",
            }
        if result.op is QueueOp.NOT_FOUND:
            return {
                "available": True,
                "applied": False,
                "error_type": "not_found",
                "error": f"{item_name} / {task_name} is not in the queue.",
            }
        if result.ok:
            manager.notify_queue_changed()
            self._record_queue_change(action, item)
        return {
            "available": True,
            "applied": True,
            # Already first or last: nothing moved.
            "changed": result.ok,
            "item_id": item.id,
            "item_name": item_name,
            "task_name": task_name,
            "action": action,
            "queue_version": result.version,
        }

    def _record_queue_change(self, action: str, item) -> None:
        """One ``queue_changed`` event; the stamp says who made the edit."""
        if self._event_buffer is not None:
            self._event_buffer.append(
                "queue_changed",
                {
                    "action": action,
                    "item_name": item.item_name,
                    "task_name": item.task_name,
                    "queue_item_id": item.id,
                },
            )

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
        request, question_id = responder.pending_question_and_id()
        if request is None:
            return {"available": True, "pending": None}
        from fibsem.applications.autolamella.server.events import to_plain
        from fibsem.applications.autolamella.server.prompts import serialize_request

        payload = serialize_request(request)
        payload["question_id"] = question_id
        recorded = responder.recorded_question()
        if recorded is not None:
            # On the item's record as a proposal, and answered there: the same
            # decision an agent makes on any other review. Answering the prompt
            # would decide nothing, so it says so up front.
            payload["answer_via"] = "decide"
            payload["decide"] = {
                "item_id": recorded[0],
                "task_name": recorded[1],
                "proposal_id": recorded[2],
            }
        dismissed = getattr(responder, "dismissed_reason", lambda: None)()
        if dismissed is not None:
            # The operator dismissed the agent's suggestion: say so, and why,
            # to an agent reading the question again.
            payload["suggestion_dismissed"] = {"reason": dismissed}
        live = self._peek_live(responder, request, question_id)
        if live is not None:
            try:
                payload["current"] = to_plain(request.value_to_wire(live))
            except Exception:  # noqa: BLE001 - the question must still show
                pass
        view = getattr(request, "view_frame", None)
        if view is not None:
            # The frame the question's widget draws on is display state, not
            # workflow state: attach it here, where the display cache already
            # lives (the same grab as display_images). Where the view may have
            # moved on since the question was asked (an agent's approved moves,
            # FIB-1399), the view's frame wins over the question's own.
            if view == "prefer" or payload.get("image") is None:
                shown = self._view_picture(request, live)
                if shown is not None:
                    payload["image"] = shown
        return {"available": True, "pending": payload}

    def _view_picture(self, request, live) -> Optional[Dict[str, Any]]:
        """The FIB view's frame as a preview, with the question drawn over it
        when it draws (the milling question's patterns), or None."""
        widget = getattr(self._host, "image_widget", None)
        image = getattr(widget, "ib_image", None)
        if image is None:
            return None
        from fibsem.applications.autolamella.server.prompts import _preview_payload

        drawn = None
        try:
            drawn = request.draw(image, live)
        except Exception:  # noqa: BLE001 - the plain frame is still worth showing
            logging.debug("could not draw the question over the frame", exc_info=True)
        entry = _preview_payload(image if drawn is None else drawn)
        if entry is None:
            return None
        if drawn is not None:
            # The drawing has the frame's shape; its scale is the frame's.
            frame = _preview_payload(image) or {}
            for key in ("hfw", "beam_type", "pixelsize", "full_width", "full_height"):
                if key in frame:
                    entry[key] = frame[key]
            entry["drawn"] = True
        entry["acquired_at"] = self._acquired_at(image)
        return entry

    def _display_frame(self, attr: str) -> Optional[Dict[str, Any]]:
        """One display-cache image as the standard preview payload, or None."""
        widget = getattr(self._host, "image_widget", None)
        image = getattr(widget, attr, None)
        if image is None:
            return None
        from fibsem.applications.autolamella.server.prompts import _preview_payload

        entry = _preview_payload(image)
        if entry is not None:
            entry["acquired_at"] = self._acquired_at(image)
        return entry

    @staticmethod
    def _peek_live(responder, request, question_id: int):
        """The live half of the question: what the widget it is asked on holds
        now (where the marker IS, the config as edited), not where it started.

        Only for a question that takes a value: the peek marshals to the GUI
        thread, and degrades to None (never an error: the question itself must
        still be visible) if the GUI is too busy to answer within 2 s or the
        prompt changed mid-peek.
        """
        if getattr(request, "value_shape", None) is None:
            return None
        try:
            peeked_question_id, value = responder.peek_live_answer().result(timeout=2.0)
        except Exception:
            return None
        if peeked_question_id != question_id:
            return None
        return value

    def answer_prompt(
        self,
        response: bool,
        question_id: int,
        value: Optional[Dict[str, Any]] = None,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Answer the pending question as the matching button click would.

        Routed through the responder's own GUI-thread path, so agent and human
        answers share one first-writer-wins mechanism; ``applied`` is False when
        nothing was pending or a human answered first. The ``question_id`` (from
        :meth:`pending_prompt`) names the question being answered: if that
        posting is gone — answered, withdrawn, or replaced — the result is
        ``stale`` and nothing was clicked.

        ``value`` optionally carries the edit the operator could make in the
        widget the question is asked on: an alignment area
        (``{"left", "top", "width", "height"}``, fractions of the frame) for
        ``EditAlignmentArea``; a point (``{"x", "y"}``, metres in microscope
        image coordinates, origin centre, +y up) for ``PickPOI``; moved
        features (``{"features": [{"name", "px": {"x", "y"}}]}``, source-image
        pixels, only the named ones move) for ``ConfirmDetection``; the action
        and any edits (``{"action", "points": [{"x", "y"}], "milling_current",
        "exposure_time"}``, points as fractions of the frame, origin top-left)
        for ``RunSpotBurn``. The value is
        validated here — in bounds, non-degenerate, the right shape for the
        pending question — and refused as ``invalid_value`` without clicking
        anything when it is not. A valid value is placed into the widget as if
        the operator had dragged it there, then the ordinary click path accepts
        it, so attribution and first-writer-wins hold unchanged.
        """
        responder = self._responder
        if responder is None:
            return {"available": False, "applied": False, "stale": False}
        from fibsem.applications.autolamella.workflows.interaction import (
            InvalidAnswerValue,
            StalePromptError,
        )

        recorded = responder.recorded_question()
        _request, pending_question_id = responder.pending_question_and_id()
        if recorded is not None and pending_question_id == int(question_id):
            # Nothing was clicked. This question is decided, not answered: see
            # ``answer_via`` on the pending prompt.
            return {
                "available": True,
                "applied": False,
                "stale": False,
                "answer_via": "decide",
                "item_id": recorded[0],
                "task_name": recorded[1],
                "proposal_id": recorded[2],
            }

        request, parsed, refusal = self._parse_answer_value(
            responder, int(question_id), value
        )
        if refusal is not None:
            return refusal
        response = _response_for(request, parsed, response)
        outcome = responder.submit_answer(
            bool(response), question_id=int(question_id), value=parsed
        )
        try:
            applied = bool(outcome.result(timeout=timeout))
        except StalePromptError:
            return {"available": True, "applied": False, "stale": True}
        except InvalidAnswerValue as exc:
            # The value could not be made on the config as it now stands.
            return {
                "available": True,
                "applied": False,
                "stale": False,
                "invalid_value": str(exc),
            }
        result = {"available": True, "applied": applied, "stale": False}
        moved_from = getattr(outcome, "moved_from", None)
        if moved_from is not None:
            # The value moved the widget: say from where, in the value's own
            # shape. Absent when the widget already showed it (FIB-1420).
            from fibsem.applications.autolamella.server.events import to_plain

            result["moved_from"] = to_plain(request.value_to_wire(moved_from))
        return result

    def suggest_answer(
        self,
        response: bool,
        question_id: int,
        reason: str,
        value: Optional[Dict[str, Any]] = None,
        timeout: float = 10.0,
        image: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Suggest an answer to the pending question (FIB-1283).

        Nothing is answered: the prompt shows "Agent suggests: ..." with the
        reason, and the operator's click on it answers as theirs, recorded with
        the agent as the one who suggested it. Answering any other way ignores
        it, and a new posting of the question drops it. ``value`` is as for
        :meth:`answer_prompt`, validated the same way. A question on the
        record is decided, not answered: at Assist, through a request.
        """
        responder = self._responder
        if responder is None or not hasattr(responder, "submit_suggestion"):
            return {"available": False, "suggested": False, "stale": False}
        from fibsem.applications.autolamella.workflows.interaction import (
            StalePromptError,
        )

        request, parsed, refusal = self._parse_answer_value(
            responder, int(question_id), value
        )
        if refusal is not None:
            return {**refusal, "suggested": False}
        response = _response_for(request, parsed, response)
        outcome = responder.submit_suggestion(
            bool(response),
            int(question_id),
            value=parsed,
            reason=str(reason),
            image=image,
        )
        try:
            outcome.result(timeout=timeout)
        except StalePromptError:
            return {"available": True, "suggested": False, "stale": True}
        except ValueError as exc:
            return {
                "available": True,
                "suggested": False,
                "stale": False,
                "error": str(exc),
            }
        return {"available": True, "suggested": True, "stale": False}

    @staticmethod
    def _parse_answer_value(responder, question_id: int, value: Any):
        """The pending question, and a wire ``value`` as the edit it stands for.

        Returns ``(request, parsed, None)``, or ``(request, None, refusal)`` with
        the refusal already shaped for :meth:`answer_prompt` to return. The
        question checks the value itself (``Request.parse_value``), against the
        frozen pending request; the responder's own question id check on the GUI
        thread still guards the actual apply, so a question swap between here
        and there is refused as stale, never mis-applied. A question whose
        answer must carry a value (the milling question's action) is refused
        without one.
        """
        from fibsem.applications.autolamella.workflows.interaction import (
            InvalidAnswerValue,
        )

        def _refuse(message: str) -> Dict[str, Any]:
            return {
                "available": True,
                "applied": False,
                "stale": False,
                "invalid_value": message,
            }

        request, pending_question_id = responder.pending_question_and_id()
        if request is None or pending_question_id != question_id:
            return request, None, {"available": True, "applied": False, "stale": True}
        shape = getattr(request, "value_shape", None)
        if value is None:
            if getattr(request, "requires_value", False):
                return (
                    request,
                    None,
                    _refuse(
                        f"a {type(request).__name__} answer carries a value: {shape}"
                    ),
                )
            return request, None, None
        if not isinstance(value, dict):
            return request, None, _refuse("value must be a JSON object.")
        try:
            return request, request.parse_value(value), None
        except InvalidAnswerValue as exc:
            return request, None, _refuse(str(exc))

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

        from fibsem.applications.autolamella.event_recording import (
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
                "created_at": to_iso(summary.created_at),
                "num_items": summary.num_lamella,
                "available": summary.available,
                "instrument_model": summary.instrument_model,
                "software_version": summary.software_version,
            }
            for summary in get_recent_experiments()
        ]
