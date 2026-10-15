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


def item_fields_version(lamella) -> str:
    """A content hash naming the state of an item's patchable fields.

    The item document's version, hashing exactly the ITEM_PATCH_FIELDS — so a
    task re-recording geometry rotates it (stales a pending patch, correctly),
    while poses moving or history growing does not (they are not patchable,
    so they cannot invalidate what a patch was written against).
    """
    from fibsem.applications.autolamella.server.events import to_plain

    return _content_hash(
        {name: to_plain(getattr(lamella, name, None)) for name in ITEM_PATCH_FIELDS}
    )


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


def _rendered_preview(path: str, filename: str, max_width: int) -> Dict[str, Any]:
    """One recorded output file as a downscaled JPEG, or the reason it is not.

    Fluorescence stacks (``.ome.tiff``) composite through the FM canvas's own
    projection, so each channel keeps its colour; plain TIFFs (every
    FibsemImage) read through tifffile; anything else (the PNG thumbnails grid
    tasks write beside their overviews) through PIL. Stacks are max-projected
    to something viewable; this is a preview, not the data.
    """
    try:
        import numpy as np

        from fibsem.fm.preview import is_fluorescence_image, load_projection
        from fibsem.server.images import preview_jpeg_bytes

        metadata = None
        if is_fluorescence_image(path):
            data, _ = load_projection(path)
        elif filename.lower().endswith((".tif", ".tiff")):
            import tifffile

            data = np.squeeze(np.asarray(tifffile.imread(path)))
            metadata = _beam_image_metadata(path)
        else:
            from PIL import Image

            with Image.open(path) as im:
                data = np.asarray(im.convert("RGB"))
        # z-stacks / channel stacks project on leading axes; a trailing
        # RGB(A) axis is colour, not a stack, and stays.
        while data.ndim > 2 and data.shape[-1] not in (3, 4):
            data = data.max(axis=0)
        if data.ndim == 3 and data.shape[-1] == 4:
            data = data[..., :3]
        jpeg, width, height = preview_jpeg_bytes(data, max_width=max_width)
        if metadata is not None and metadata["pixel_size"] and width:
            # metres per PREVIEW pixel, as image answers carry it.
            metadata["preview_pixelsize"] = (
                metadata["pixel_size"] * data.shape[1] / width
            )
    except Exception as exc:  # noqa: BLE001 - a broken file is a data fact
        return {
            "available": True,
            "jpeg": None,
            "error": f"Could not render {filename!r}: {exc}",
        }
    return {
        "available": True,
        "jpeg": jpeg,
        "width": width,
        "height": height,
        "filename": filename,
        "metadata": metadata,
    }


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


def _beam_image_metadata(path: str) -> Optional[Dict[str, Any]]:
    """A saved beam image's metadata block (images.image_metadata), or None for a
    TIFF that is not one: a plain array, a stack, a file that will not load."""
    from fibsem.server.images import image_metadata
    from fibsem.structures import FibsemImage

    try:
        image = FibsemImage.load(path)
    except Exception:  # noqa: BLE001 - no metadata is an answer, not a failure
        return None
    if image.metadata is None:
        return None
    return image_metadata(image, path)


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


# The questions an answer may carry a value for (see ``answer_prompt``).
_VALUE_TYPES = ("PickPOI", "EditAlignmentArea", "ConfirmDetection", "RunSpotBurn")


def _parse_detection_value(request, value: Dict[str, Any], refuse):
    """A ConfirmDetection value as ``{name: Point}`` in source-image pixels.

    Only features the detection has may move, each inside the image. Returns
    ``(parsed, None)`` or ``(None, refusal)``.
    """
    from fibsem.structures import Point

    detection = getattr(request, "detection", None)
    names = [f.name for f in getattr(detection, "features", [])]
    features = value.get("features")
    if not isinstance(features, list) or not features:
        return None, refuse(
            'a ConfirmDetection value needs "features": [{"name", "px": {"x", "y"}}], '
            "source-image pixels, origin top-left, +y down."
        )
    image = getattr(detection, "image", None)
    shape = getattr(getattr(image, "data", image), "shape", None)
    moved: Dict[str, Point] = {}
    for entry in features:
        name = entry.get("name") if isinstance(entry, dict) else None
        if name not in names:
            return None, refuse(
                f"no detected feature named {name!r}; the features are {names}."
            )
        if name in moved:
            return None, refuse(f"feature {name!r} is given twice.")
        try:
            px = Point.from_dict(entry["px"])
        except Exception:
            return None, refuse(
                f"feature {name!r} needs px: {{x, y}}, numbers in source-image pixels."
            )
        if shape is not None and len(shape) >= 2:
            height, width = shape[0], shape[1]
            if not (0 <= px.x < width and 0 <= px.y < height):
                return None, refuse(
                    f"feature {name!r} at ({px.x:.1f}, {px.y:.1f}) px is outside "
                    f"the {width} x {height} px image."
                )
        moved[name] = px
    return moved, None


def _parse_spot_burn_value(value: Dict[str, Any], refuse):
    """A RunSpotBurn value as a list of Points, 0-1 fractions of the frame.

    Returns ``(parsed, None)`` or ``(None, refusal)``.
    """
    from fibsem.services.spot_burn import in_frame
    from fibsem.structures import Point

    points = value.get("points")
    if not isinstance(points, list):
        return None, refuse(
            'a RunSpotBurn value needs "points": [{"x", "y"}], fractions of '
            "the frame, origin top-left, +y down."
        )
    parsed: List[Point] = []
    for entry in points:
        try:
            pt = Point.from_dict(entry)
        except Exception:
            return None, refuse(
                "each spot burn point needs numeric x, y: fractions of the frame."
            )
        if not in_frame(pt):
            return None, refuse(
                f"point ({pt.x:.3f}, {pt.y:.3f}) is outside the frame: x and y "
                "are fractions, 0 to 1."
            )
        parsed.append(pt)
    return parsed, None


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
        import os

        from fibsem.structures import FibsemImage

        root = self.experiment_path
        if root is None:
            return {"error_type": "not_found", "message": "No experiment is open."}
        root = os.path.realpath(root)
        target = os.path.realpath(path)
        if os.path.commonpath([root, target]) != root:
            return {
                "error_type": "not_allowed",
                "message": f"Only images inside the open experiment ({root}).",
            }
        if not os.path.isfile(target):
            return {"error_type": "not_found", "message": f"No file at {path}."}
        image = FibsemImage.load(target)
        if image.metadata is None:
            return {
                "error_type": "not_found",
                "message": f"{path} carries no beam image metadata.",
            }
        return {"path": path, "metadata": image.metadata.to_dict()}

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
            # the run options set_workflow_options changes
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
        import os

        experiment = self._experiment
        if experiment is None:
            return {"available": False, "jpeg": None}
        lamella, refusal = find_item(experiment, item_name)
        if refusal:
            return {"available": False, "jpeg": None, **refusal}
        paths = self._recorded_item_paths(lamella)
        match = next((p for p in paths if os.path.basename(p) == filename), None)
        if match is None:
            return {
                "available": True,
                "jpeg": None,
                "error": f"No output named {filename!r} for {item_name!r}.",
                "filenames": [os.path.basename(p) for p in paths],
            }
        return _rendered_preview(match, filename, max_width)

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
        match = self._recorded_grid_file(experiment, grid, filename)
        if match is None:
            return {
                "available": True,
                "jpeg": None,
                "error": f"No output named {filename!r} for grid {grid_name!r}.",
                "filenames": self._recorded_grid_basenames(experiment, grid),
            }
        return _rendered_preview(match, filename, max_width)

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
        match = self._recorded_grid_file(experiment, grid, filename)
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
    def _recorded_grid_file(cls, experiment, grid, filename: str) -> Optional[str]:
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
        path = self._recorded_grid_file(experiment, grid, filename)
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
        out as for a click; it is saved."""
        from fibsem.applications.autolamella.poses import add_lamella
        from fibsem.structures import FibsemStagePosition

        experiment, microscope, refusal = self._editable()
        if refusal:
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
            position = FibsemStagePosition.from_dict(stage_position)

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
        fluorescence pose follow. Saved."""
        from fibsem.applications.autolamella.poses import set_lamella_pose

        experiment, microscope, refusal = self._editable()
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

        poses = {}
        for pose_name, pose in dict(lamella.poses).items():
            position = getattr(pose, "stage_position", None)
            poses[pose_name] = position.to_dict() if position is not None else None
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
            "poses": poses,
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
                    # what this is; decide_review passes it back. The run is
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

    def add_note(self, text: str, item_name: Optional[str] = None) -> Dict[str, Any]:
        """Put an agent observation on the record.

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
            self._event_buffer.append(
                "agent_note",
                {"text": text, "item_name": item_name, "item_id": item_id},
            )
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
        timeout: float = 15.0,
    ) -> Dict[str, Any]:
        """Start a grid run as the Grids view's Run (or Screen all grids) would.

        Marshalled to the GUI thread like :meth:`start_workflow`; shares its
        worker slot, so it is refused while any workflow — lamella or grid —
        is running. The agent at Drive stands in for the preflight dialog's
        Run click; :meth:`grid_workflow_plan` is that dialog's content.
        """
        host = self._host
        if not hasattr(host, "request_start_grid_workflow"):
            return {"available": False, "started": False}
        outcome = host.request_start_grid_workflow(
            list(task_names), grid_names, inventory_first=screen_all
        )
        result = outcome.result(timeout=timeout)
        result["available"] = True
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

    def set_workflow_options(self, show_run_summary: bool) -> Dict[str, Any]:
        """Set whether the run summary dialog opens when a run ends.

        The workflow's "Show summary after completion", and like the operator's
        checkbox it persists with the protocol. Off leaves the run-end toast
        and its Show summary link: an agent starting many short runs turns it
        off so no dialog stands over the window (FIB-1376). Read when a run
        ends, so it applies from the next end, mid-run included; a plain bool
        rebind, so no GUI marshal, like ``set_task_schedule``.
        """
        experiment = self._experiment
        protocol = getattr(experiment, "task_protocol", None) if experiment else None
        if protocol is None:
            return {"available": False, "applied": False}
        protocol.options.show_run_summary = bool(show_run_summary)
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
            self._event_buffer.append(
                "workflow_changed",
                {"field": "show_run_summary", "value": bool(show_run_summary)},
            )
        return {
            "available": True,
            "applied": True,
            "saved": saved,
            "show_run_summary": bool(show_run_summary),
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
        request, question_id = responder.pending_question_and_id()
        if request is None:
            return {"available": True, "pending": None}
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
        current = self._peek_current(responder, payload["type"], question_id)
        if current is not None:
            payload["current"] = current
        if (
            payload["type"] in ("EditAlignmentArea", "RunSpotBurn")
            and payload.get("image") is None
        ):
            # The frame the area sits on is display state, not workflow state:
            # attach it here, where the display cache already lives, rather
            # than having the workflow reach into the widget to build the
            # request. Same whole-reference grab as display_images.
            payload["image"] = self._display_frame("ib_image")
        return {"available": True, "pending": payload}

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

    def _peek_current(
        self, responder, request_type: str, question_id: int
    ) -> Optional[Dict[str, Any]]:
        """The live half of the question — where the marker IS, not where it
        started.

        PickPOI, EditAlignmentArea, ConfirmDetection and RunSpotBurn are
        answered from widget state the operator may still be adjusting; the
        frozen request only shows the starting value. The peek marshals to the GUI thread, so it is skipped
        entirely for question types with no live half, and degrades to absent
        (never an error — the question itself must still be visible) if the GUI
        is too busy to answer within 2 s or the prompt changed mid-peek.
        """
        if request_type not in _VALUE_TYPES:
            return None
        from fibsem.applications.autolamella.proposals import features_to_dict
        from fibsem.applications.autolamella.server.events import to_plain

        try:
            peeked_question_id, value = responder.peek_live_answer().result(timeout=2.0)
        except Exception:
            return None
        if peeked_question_id != question_id or value is None:
            return None
        try:
            if request_type == "ConfirmDetection":
                # The same shape an answer's value takes.
                features = [{"name": n, "px": px} for n, px in value.items()]
                return {"features": to_plain(features_to_dict(features))}
            if request_type == "RunSpotBurn":
                return {"points": [to_plain(pt.to_dict()) for pt in value]}
            return to_plain(value.to_dict())
        except Exception:
            return None

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
        pixels, only the named ones move) for ``ConfirmDetection``; the points
        to burn (``{"points": [{"x", "y"}]}``, fractions of the frame, origin
        top-left; current and exposure stay the protocol's) for
        ``RunSpotBurn``. The value is
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

        parsed = None
        if value is not None:
            parsed, refusal = self._parse_answer_value(
                responder, int(question_id), value
            )
            if refusal is not None:
                return refusal

        outcome = responder.submit_answer(
            bool(response), question_id=int(question_id), value=parsed
        )
        try:
            applied = bool(outcome.result(timeout=timeout))
        except StalePromptError:
            return {"available": True, "applied": False, "stale": True}
        result = {"available": True, "applied": applied, "stale": False}
        if getattr(outcome, "adjusted", parsed is not None):
            result["adjusted"] = True
        return result

    def suggest_answer(
        self,
        response: bool,
        question_id: int,
        reason: str,
        value: Optional[Dict[str, Any]] = None,
        timeout: float = 10.0,
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

        parsed = None
        if value is not None:
            parsed, refusal = self._parse_answer_value(
                responder, int(question_id), value
            )
            if refusal is not None:
                return refusal
        outcome = responder.submit_suggestion(
            bool(response), int(question_id), value=parsed, reason=str(reason)
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
        """Turn a wire ``value`` into the domain object for the pending question.

        Returns ``(parsed, None)`` on success or ``(None, refusal)`` — the
        refusal already shaped for :meth:`answer_prompt` to return. Validation
        happens against the frozen pending request; the responder's own question id
        check on the GUI thread still guards the actual apply, so a question
        swap between here and there is refused as stale, never mis-applied.
        """

        def _refuse(message: str) -> Dict[str, Any]:
            return {
                "available": True,
                "applied": False,
                "stale": False,
                "invalid_value": message,
            }

        request, pending_question_id = responder.pending_question_and_id()
        if request is None or pending_question_id != question_id:
            return None, {"available": True, "applied": False, "stale": True}
        type_name = type(request).__name__
        if not isinstance(value, dict):
            return None, _refuse("value must be a JSON object.")

        if type_name == "EditAlignmentArea":
            from fibsem.structures import FibsemRectangle

            try:
                rect = FibsemRectangle.from_dict(value)
            except Exception:
                return None, _refuse(
                    "an EditAlignmentArea value needs numeric left, top, "
                    "width, height — fractions of the frame, origin top-left."
                )
            if not rect.is_valid_reduced_area:
                return None, _refuse(
                    "alignment area out of bounds: left/top must be >= 0, "
                    "width/height > 0, and the rectangle must fit inside "
                    "the frame (left+width <= 1, top+height <= 1)."
                )
            return rect, None

        if type_name == "PickPOI":
            from fibsem.structures import Point

            try:
                point = Point.from_dict(value)
            except Exception:
                return None, _refuse(
                    "a PickPOI value needs numeric x, y — metres in "
                    "microscope image coordinates, origin centre, +y up."
                )
            image = getattr(request, "image", None)
            try:
                half_width = image.data.shape[1] / 2 * image.metadata.pixel_size.x
                half_height = image.data.shape[0] / 2 * image.metadata.pixel_size.x
            except Exception:
                return None, _refuse(
                    "the pending question's image is missing its scale; "
                    "the point cannot be validated."
                )
            if abs(point.x) > half_width or abs(point.y) > half_height:
                return None, _refuse(
                    f"point ({point.x:.3e}, {point.y:.3e}) m is outside the "
                    f"image: |x| <= {half_width:.3e}, |y| <= {half_height:.3e}."
                )
            return point, None

        if type_name == "ConfirmDetection":
            return _parse_detection_value(request, value, _refuse)

        if type_name == "RunSpotBurn":
            return _parse_spot_burn_value(value, _refuse)

        return None, _refuse(
            f"{type_name} answers cannot carry a value; only EditAlignmentArea, "
            "PickPOI, ConfirmDetection and RunSpotBurn can."
        )

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
