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

import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

from fibsem.applications.autolamella import task_outputs as _task_outputs
from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus

__all__ = ["AgentContext", "ITEM_PATCH_FIELDS", "config_version", "item_fields_version"]

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

    The item document's nonce, hashing exactly the ITEM_PATCH_FIELDS — so a
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

    The config's nonce (FIB-864): reads serve it, writes echo it, and a
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

        if is_fluorescence_image(path):
            data, _ = load_projection(path)
        elif filename.lower().endswith((".tif", ".tiff")):
            import tifffile

            data = np.squeeze(np.asarray(tifffile.imread(path)))
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
    }


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
                    "supervise": task.supervise,
                    "supervisor": getattr(task, "supervisor", "human"),
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
        lamella = experiment.get_lamella_by_name(item_name)
        if lamella is None:
            return {
                "available": False,
                "item_name": item_name,
                "error": f"No item named {item_name!r} in this experiment.",
            }
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
            tasks.append({"name": state.name, "files": files})
        return {
            "available": True,
            "item_name": item_name,
            "completed_tasks": [state.name for state in history],
            "tasks": tasks,
            "final_reference_images": [
                str(p) for p in _task_outputs.final_reference_images(lamella, *history)
            ],
            "fluorescence_images": [
                str(p) for p in _task_outputs.fluorescence_images(lamella, *history)
            ],
        }

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
        lamella = experiment.get_lamella_by_name(item_name)
        if lamella is None:
            return {
                "available": False,
                "jpeg": None,
                "error": f"No item named {item_name!r} in this experiment.",
            }
        history = list(lamella.task_history)
        paths = _task_outputs.final_reference_images(
            lamella, *history
        ) + _task_outputs.fluorescence_images(lamella, *history)
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
        and whether it is in the beam are live stage facts, and this facade
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
            "quality": grid.quality.name,
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
        lamella = experiment.get_lamella_by_name(item_name)
        if lamella is None:
            return {
                "available": False,
                "item_name": item_name,
                "error": f"No item named {item_name!r} in this experiment.",
            }
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
            # The item document's nonce: patches echo it (see apply_item_patch).
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
        lamella = experiment.get_lamella_by_name(item_name)
        if lamella is None:
            return {
                "available": False,
                "item_name": item_name,
                "task_name": task_name,
                "error": f"No item named {item_name!r} in this experiment.",
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
        if result.get("applied") and self._event_buffer is not None:
            self._event_buffer.append(
                "config_edited",
                {
                    "level": "item_fields",
                    "item_name": item_name,
                    "changes": result.get("changes", []),
                },
            )
        return result

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
        if item_name is not None:
            experiment = self._experiment
            if experiment is None or experiment.get_lamella_by_name(item_name) is None:
                names = [p.name for p in experiment.positions] if experiment else []
                return {
                    "available": True,
                    "recorded": False,
                    "error": f"No item named {item_name!r} in this experiment.",
                    "item_names": names,
                }
        logging.info(
            "agent note%s: %s",
            f" [{item_name}]" if item_name else "",
            text,
        )
        if self._event_buffer is not None:
            self._event_buffer.append(
                "agent_note", {"text": text, "item_name": item_name}
            )
        return {"available": True, "recorded": True}

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
        if result.get("applied") and self._event_buffer is not None:
            self._event_buffer.append(
                "config_edited",
                {
                    "level": level,
                    "item_name": item_name or None,
                    "task_name": task_name,
                    "reorder": {"milling_key": milling_key, "order": list(order)},
                },
            )
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
        if result.get("applied") and self._event_buffer is not None:
            self._event_buffer.append(
                "config_edited",
                {
                    "level": "protocol_applied",
                    "item_name": item_name,
                    "task_names": result.get("task_names", []),
                },
            )
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
        if result.get("applied") and self._event_buffer is not None:
            self._event_buffer.append(
                "config_edited",
                {
                    "level": "protocol",
                    "task_name": task_name,
                    "changes": result.get("changes", []),
                },
            )
        return result

    @staticmethod
    def _config_document(task_name: str, config, level: str) -> Dict[str, Any]:
        """One config as a wire document with its version token.

        The full ``to_dict``, not a curated snapshot — the document is the
        contract for editing (you cannot patch what you cannot read). The
        ``version`` hashes the serialized content, so a write names exactly
        the state it read: the config's nonce, refused as stale on mismatch.
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
        if result.get("applied") and self._event_buffer is not None:
            self._event_buffer.append(
                "config_edited",
                {
                    "level": "item",
                    "item_name": item_name,
                    "task_name": task_name,
                    "changes": result.get("changes", []),
                },
            )
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
        Control-scope arming is the consent that stands in for the Run
        click's confirm dialog.
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
        Exchanges are deliberately not counted: whether a grid is in the beam
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
        is running. Control-scope arming stands in for the preflight dialog's
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

    def set_supervision(
        self,
        task_name: str,
        supervise: bool,
        supervisor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Set whether ``task_name`` asks for supervision, in the live protocol.

        The workflow reads this at every decision point (never a snapshot), so
        a change takes effect at the next prompt-or-proceed choice — exactly
        the mid-run behaviour the GUI's supervised/automated toggle has. The
        GUI's own indicators refresh on the next task transition rather than
        instantly.

        ``supervisor`` optionally sets who the task's questions are addressed
        to ("human" or "agent") — display and watchdog semantics only; the
        questions themselves are raised identically, the operator can always
        answer first, and the designation shows nothing in the GUI unless the
        agent server is running.
        """
        if supervisor not in (None, "human", "agent"):
            return {
                "available": True,
                "applied": False,
                "error": f"supervisor must be 'human' or 'agent', not {supervisor!r}",
            }
        experiment = self._experiment
        protocol = getattr(experiment, "task_protocol", None) if experiment else None
        config = getattr(protocol, "workflow_config", None) if protocol else None
        if config is None:
            return {"available": False, "applied": False}
        for task in config.tasks:
            if task.name == task_name:
                task.supervise = bool(supervise)
                if supervisor is not None:
                    task.supervisor = supervisor
                return {
                    "available": True,
                    "applied": True,
                    "task_name": task_name,
                    "supervise": bool(supervise),
                    "supervisor": getattr(task, "supervisor", "human"),
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
        atomic field rebind; no GUI marshal needed). Unlike supervision,
        a schedule is plan data, so it is persisted immediately.

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
        if payload["type"] == "EditAlignmentArea" and payload.get("image") is None:
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
        self,
        response: bool,
        nonce: int,
        value: Optional[Dict[str, Any]] = None,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Answer the pending question as the matching button click would.

        Routed through the responder's own GUI-thread path, so agent and human
        answers share one first-writer-wins mechanism; ``applied`` is False when
        nothing was pending or a human answered first. The ``nonce`` (from
        :meth:`pending_prompt`) names the question being answered: if that
        posting is gone — answered, withdrawn, or replaced — the result is
        ``stale`` and nothing was clicked.

        ``value`` optionally carries adjusted geometry for the two questions
        answered from live widget state: an alignment area
        (``{"left", "top", "width", "height"}``, fractions of the frame) for
        ``EditAlignmentArea``, or a point (``{"x", "y"}``, metres in microscope
        image coordinates, origin centre, +y up) for ``PickPOI``. The value is
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

        parsed = None
        if value is not None:
            parsed, refusal = self._parse_answer_value(responder, int(nonce), value)
            if refusal is not None:
                return refusal

        outcome = responder.submit_answer(
            bool(response), nonce=int(nonce), value=parsed
        )
        try:
            applied = bool(outcome.result(timeout=timeout))
        except StalePromptError:
            return {"available": True, "applied": False, "stale": True}
        result = {"available": True, "applied": applied, "stale": False}
        if parsed is not None:
            result["adjusted"] = True
        return result

    @staticmethod
    def _parse_answer_value(responder, nonce: int, value: Any):
        """Turn a wire ``value`` into the domain object for the pending question.

        Returns ``(parsed, None)`` on success or ``(None, refusal)`` — the
        refusal already shaped for :meth:`answer_prompt` to return. Validation
        happens against the frozen pending request; the responder's own nonce
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

        request, pending_nonce = responder.pending_question_and_nonce()
        if request is None or pending_nonce != nonce:
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

        return None, _refuse(
            f"{type_name} answers cannot carry a value; "
            "only EditAlignmentArea and PickPOI can."
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
