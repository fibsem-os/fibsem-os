"""AutoLamellaAppContext: the read-side facade an embedded server sees the app through.

Deliberately not the window object, for the same reasons
``ScriptContext`` isn't: the window's ``experiment`` is rebound on load, the
runner's task manager is created per run and dropped when it ends, and the
GUI's internals are still moving. This facade resolves those
references **at call time, on every call**, and returns plain JSON-able data —
never live objects, never Qt, never a pointer into a widget.

Thread contract: every method is safe to call from a server thread. Reads are
point-in-time snapshots, not transactions — the workflow worker may be mutating
the experiment concurrently, and a snapshot that is a task older than the
instant it was taken is the accepted cost of never blocking anything.
An observer never makes the app poll the instrument. The few reads that do
reach the instrument are an action's own: :meth:`set_item_pose` reads the microscope
state it stores, and :meth:`add_grids`, :meth:`rename_grid` and requeueing read
the grid inventory, as the window's own buttons do. None of them writes to the
instrument.

Vocabulary: the surface speaks ``item_name`` throughout.

The facade reads the session (``fibsem.workflows.session.AppSession``): the
experiment, the microscope, the runner, the responder and the latest images,
and starts and stops runs through its runner. It never talks to a window: an
edit runs where the session edits the experiment, and a window redraws on the
experiment's events and ``session.changed``.
"""

import dataclasses
import functools
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import fibsem.config as fibsem_cfg
from fibsem.acting import Actor
from fibsem.applications.autolamella import task_outputs as _task_outputs
from fibsem.applications.autolamella.config_edits import (
    ITEM_PATCH_FIELDS,
    apply_correlation,
    item_fields_version,
    item_patch_refusal,
    poi_patch,
)
from fibsem.codec import encode
from fibsem.geometry.saved_images import image_point_to_stage_position, stage_projector
from fibsem.server.app_context import AppContext
from fibsem.server.files import jpeg_preview
from fibsem.server.responses import (
    ExperimentSummary,
    ItemDetail,
    ItemMillingDrawn,
    ProtocolView,
    ReplayTimeline,
    ReplayView,
    SummaryRow,
    TaskOutputs,
)
from fibsem.structures import BeamType
from fibsem.util.records import content_hash
from fibsem.workflows.actions import NOTE
from fibsem.workflows.attention import Attention
from fibsem.workflows.experiment import find_item
from fibsem.workflows.grid.edits import keep_grid_note, remove_grid_record
from fibsem.workflows.item import item_pose_positions
from fibsem.workflows.outputs import (
    all_item_outputs,
    files_by_run,
    item_output_named,
    latest_item_output,
    saved_file,
)
from fibsem.workflows.runner import GRIDS, ITEMS, RunRequest
from fibsem.workflows.session import CONFIG, GRID_RECORDS
from fibsem.workflows.status import TaskStatus
from fibsem.workflows.task_configs import config_version, workflow_field_refusal

__all__ = [
    "AutoLamellaAppContext",
    "ITEM_PATCH_FIELDS",
    "config_version",
    "item_fields_version",
]


# The overview roles the grid tasks record (imaging.py ROLE_BY_BEAM, the FM
# task). Listed here so a summary can say which overviews a grid has without
# a client knowing the role vocabulary.
GRID_OVERVIEW_ROLES = ("overview_sem", "overview_fib", "overview_fm")


@dataclass
class LamellaRow(SummaryRow):
    milling_angle: Optional[float] = None
    is_completed: bool = False


@dataclass
class LamellaSummary(ExperimentSummary):
    items: List[LamellaRow] = field(default_factory=list)


@dataclass
class LamellaDetail(ItemDetail):
    is_failure: bool = False
    milling_angle: Optional[float] = None
    defect: Optional[Dict[str, Any]] = None


@dataclass
class LamellaOutputs(TaskOutputs):
    final_reference_images: List[str] = field(default_factory=list)
    fluorescence_images: List[str] = field(default_factory=list)


@dataclass
class GridTaskRef:
    name: str
    # the registered task type (beam or FM overview)
    type: Optional[str] = None


@dataclass
class RunOptions:
    turn_beams_off: bool = False
    show_run_summary: bool = False


@dataclass
class LamellaProtocol(ProtocolView):
    # the grid protocol beside it: what a grid run can name
    grid_tasks: List[GridTaskRef] = field(default_factory=list)
    options: Optional[RunOptions] = None


def _grid_items(experiment, grid):
    """The items drawn on a grid's overviews, whether they are linked to it,
    and why there are none: the linked lamellae; with none linked, every
    lamella of an experiment that records one grid; else none."""
    linked = experiment.get_lamellae_for_grid(grid)
    if linked:
        return linked, True, ""
    if len(experiment.grids) <= 1:
        return list(experiment.positions), False, ""
    return (
        [],
        False,
        "no items are linked to this grid, and with several grids "
        "recorded the experiment's items are not assumed to be on it",
    )


def _basenames(paths: List[str]) -> List[str]:
    return [os.path.basename(p) for p in paths]


def _recorded_item_file(lamella, filename: str):
    """The path of the output ``task_outputs`` lists as *filename*, and None;
    or None and the refusal naming the valid ones. Only a listed basename
    resolves, so no client reaches outside the item's outputs."""
    paths = _task_outputs.recorded_files(lamella)
    match = next((p for p in paths if os.path.basename(p) == filename), None)
    if match is not None:
        return match, None
    return None, {
        "error": f"No output named {filename!r} for {lamella.name!r}.",
        "filenames": _basenames(paths),
    }


def _grid_file(experiment, grid, filename: str):
    """``(path, None)`` for *filename*: an output the grid's tasks recorded,
    by basename, else a saved image at that path in the experiment (an
    overview the agent took); ``(None, refusal)`` naming the grid's outputs."""
    match = item_output_named(experiment, grid, filename) or saved_file(
        experiment, experiment.path, filename
    )
    if match is not None:
        return match, None
    return None, {
        "error": f"No output named {filename!r} for grid {grid.name!r}, "
        "and no image at that path in the experiment.",
        "filenames": _basenames(all_item_outputs(experiment, grid)),
    }


def _grid_task_version(grid_protocol: Any, task_name: str, config: Any) -> str:
    """The version a grid task's patch echoes: its settings and its attention,
    which the patch also sets."""
    attention = grid_protocol.get_attention(task_name).value
    return content_hash([config_version(config), attention])


class AutoLamellaAppContext(AppContext):
    """Read-only facade over a running (or resting) AutoLamella session."""

    app_name = "AutoLamella"

    def picture_sources(self) -> _task_outputs.PreviewSources:
        return _task_outputs.PreviewSources(lambda: self._experiment)

    # --- experiment record ----------------------------------------------------

    answer_types = {
        ExperimentSummary: LamellaSummary,
        ItemDetail: LamellaDetail,
        TaskOutputs: LamellaOutputs,
        ProtocolView: LamellaProtocol,
    }

    def summary_row(self, experiment, item) -> LamellaRow:
        row = super().summary_row(experiment, item)
        return LamellaRow(
            **vars(row),
            milling_angle=item.milling_angle,
            is_completed=experiment.task_protocol.workflow_config.is_completed(item),
        )

    def protocol(self) -> Dict[str, Any]:
        """The workflow, the grid tasks a grid run can name, and the run
        options."""
        view = super().protocol()
        if not view["available"]:
            return view
        task_protocol = self._experiment.task_protocol
        grid_protocol = self._experiment.grid_protocol
        return {
            **view,
            # Kind is the registered task type (beam vs FM overview).
            "grid_tasks": [
                {
                    "name": name,
                    "type": getattr(grid_protocol.task_config[name], "task_type", None),
                }
                for name in grid_protocol.ordered_task_names
            ],
            # the run options workflow_set_options changes
            "options": {
                "turn_beams_off": task_protocol.options.turn_beams_off,
                "show_run_summary": task_protocol.options.show_run_summary,
            },
        }

    def outputs_of(self, experiment, item) -> LamellaOutputs:
        outputs = super().outputs_of(experiment, item)
        history = list(item.task_history)
        outputs.final_reference_images = [
            str(p) for p in _task_outputs.final_reference_images(item, *history)
        ]
        outputs.fluorescence_images = [
            str(p) for p in _task_outputs.fluorescence_images(item, *history)
        ]
        return outputs

    def history_images(self, experiment, item):
        return _task_outputs.history_images(item)

    def animation_steps(self, item_name: str) -> Dict[str, Any]:
        """A lamella's workflow animation, as the History tab's button makes it:
        a step per task, with its final images and fluorescence stacks."""
        from fibsem.imaging.animation import AnimationStep

        experiment = self._experiment
        if experiment is None:
            return {"error_type": "no_experiment", "error": "No experiment is open."}
        item, refusal = find_item(experiment, item_name)
        if refusal:
            return {"error_type": "not_found", **refusal}
        return {
            "item_name": item.name,
            "root": str(experiment.path),
            "steps": [
                AnimationStep(title=name, paths=finals, fluorescence=stacks)
                for name, finals, stacks in _task_outputs.images_by_task(item)
            ],
        }

    def output_image(
        self, item_name: str, filename: str, max_width: int = 768
    ) -> Dict[str, Any]:
        """One output image of an item, rendered as JPEG bytes.

        The read side of :meth:`task_outputs` for clients that cannot touch the
        disk — the dashboard's thumbnails. ``filename`` is the basename of a
        path :meth:`task_outputs` listed, or a path in the item's own folder
        (an image the agent saved there); anything else is refused with the
        valid names, and nothing outside the experiment is reached. Stacks
        and multi-channel images are max-projected to something viewable; this
        is a preview, not the data.
        """
        experiment = self._experiment
        if experiment is None:
            return {"available": False, "jpeg": None}
        lamella, refusal = find_item(experiment, item_name)
        if refusal:
            return {"available": False, "jpeg": None, **refusal}
        match, refusal = _recorded_item_file(lamella, filename)
        if refusal:
            # else an image saved in the item's folder (one the agent took)
            match = saved_file(experiment, lamella.path, filename)
        if match is None:
            return {"available": True, "jpeg": None, **refusal}
        return jpeg_preview(match, filename, max_width)

    def measure_thickness(
        self,
        item_name: str,
        path: str,
        width_um: Optional[float] = None,
        height_um: Optional[float] = None,
    ) -> Dict[str, Any]:
        """The lamella's thickness, read off a saved FIB image of the item.

        ``path`` is any FIB image inside the experiment: a task's reference or
        one the agent took. The bright band between the trenches, measured
        column by column at half its peak brightness
        (:func:`fibsem.imaging.thickness.measure_lamella_band`) in a window
        around the item's POI: ``width_um`` along the lamella, ``height_um``
        across it, each by default from the item's own trench patterns
        (``Lamella.trench_window``). Nothing moves; the result is saved in a
        ``thickness`` folder beside the image. The answer carries the result's
        diagnostic plot.
        """
        import base64

        from fibsem.fm.preview import is_fluorescence_image
        from fibsem.imaging.thickness import DEFAULT_WINDOW, measure_lamella_band
        from fibsem.structures import BeamType, FibsemImage, Point

        experiment = self._experiment
        if experiment is None:
            return {"error_type": "not_found", "error": "No experiment is open."}
        lamella, refusal = find_item(experiment, item_name)
        if refusal:
            return {"error_type": "not_found", **refusal}
        target, refusal = self._experiment_image(path)
        if refusal:
            return refusal
        if is_fluorescence_image(target):
            return {
                "error_type": "invalid",
                "error": f"{path!r} is a fluorescence image; the band is read off a FIB image.",
            }
        try:
            image = FibsemImage.load(target)
        except Exception as exc:  # noqa: BLE001 - an unreadable file is a data fact
            return {"error_type": "invalid", "error": f"{path!r}: {exc}"}
        settings = getattr(image.metadata, "image_settings", None)
        if getattr(settings, "beam_type", None) is BeamType.ELECTRON:
            return {
                "error_type": "invalid",
                "error": f"{path!r} is an SEM image; the band is read off a FIB image.",
            }
        default, source = lamella.trench_window(), "patterns"
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
        stem = os.path.splitext(os.path.basename(target))[0]
        try:
            saved_to = str(
                result.save(os.path.join(os.path.dirname(target), "thickness"), stem)
            )
        except OSError:
            logging.exception(f"Could not save the thickness result for {path!r}")
            saved_to = None
        return {
            "item_name": item_name,
            "image": path,
            "poi": {"x": poi.x, "y": poi.y},
            "window_um": [round(window[0] * 1e6, 3), round(window[1] * 1e6, 3)],
            "window_from": source,
            "saved_to": saved_to,
            **result.to_dict(),
            "plot": {"image_b64_jpeg": base64.b64encode(result.plot_jpeg()).decode()},
        }

    # --- replay: the recorded actions, stepped through on the web page ---------

    @functools.cached_property
    def _replay_reader(self):
        from fibsem.applications.autolamella.server.replay import ReplayReader

        return ReplayReader()

    def replay(self) -> ReplayTimeline:
        experiment = self._experiment
        if experiment is None:
            return ReplayTimeline(available=False, error="No experiment is open.")
        return self._replay_reader.timeline(Path(experiment.path))

    def replay_scene(self, index: int, item: Optional[str] = None):
        experiment = self._experiment
        if experiment is None:
            return ReplayView(available=False, error="No experiment is open.")
        # Named for the lamella: a saved position's own name is often empty.
        positions = [
            dataclasses.replace(lamella.stage_position, name=lamella.name)
            for lamella in experiment.positions
            if lamella.stage_position is not None
        ]
        return self._replay_reader.view(
            Path(experiment.path), index, item, self._microscope, positions
        )

    # --- grids: the screening read model ----------------------------

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
        experiment = self._experiment
        if experiment is None:
            return {"available": False, "grid_name": grid_name}
        grid = experiment.get_grid_by_name(grid_name)
        if grid is None:
            return self._no_such_grid(experiment, grid_name)
        detail = self._grid_summary(experiment, grid)
        detail["tasks"] = encode(
            files_by_run(experiment.grid_path(grid), grid.task_history)
        )
        detail["items"] = [p.name for p in experiment.get_lamellae_for_grid(grid)]
        return detail

    def grid_output_image(
        self, grid_name: str, filename: str, max_width: int = 768, cells: bool = False
    ) -> Dict[str, Any]:
        """One grid output (overview or thumbnail), rendered as JPEG.

        ``filename`` is a basename the grid's history recorded, or a saved
        image's path in the experiment (`_grid_file`); anything else is
        refused with the valid names. Overviews are large; this is the
        downscaled preview, with *cells* the overview's named cells drawn on
        it and listed in its metadata.
        """
        experiment = self._experiment
        if experiment is None:
            return {"available": False, "jpeg": None}
        grid = experiment.get_grid_by_name(grid_name)
        if grid is None:
            return dict(self._no_such_grid(experiment, grid_name), jpeg=None)
        match, refusal = _grid_file(experiment, grid, filename)
        if refusal:
            return {"available": True, "jpeg": None, **refusal}
        return jpeg_preview(match, filename, max_width, cells=cells)

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
        match, refusal = _grid_file(experiment, grid, filename)
        if refusal:
            return {"available": True, "grid_name": grid_name, "markers": [], **refusal}
        try:
            image_block, project = stage_projector(match)
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
        lamellae, linked, reason = _grid_items(experiment, grid)
        payload["linked"] = linked
        if reason:
            payload["reason"] = reason
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

    def render_markers(
        self,
        path: str,
        grid_name: Optional[str] = None,
        fov: bool = False,
        current: bool = False,
    ) -> Dict[str, Any]:
        """Any beam image in the experiment with the items drawn on it, as the
        overview plot dialog draws them (`overview_figure`): numbers for names,
        green, red for a failed item. *grid_name* draws that grid's items only;
        without it, every item with a pose. Saved as ``markers/<image>.png``
        beside the image (replaced when drawn again), a path the agent can cite
        as evidence; the answer carries a preview. *fov* draws each item's
        field of view, *current* where the stage is now."""
        import base64

        from fibsem.applications.autolamella.overview_figure import (
            save_markers_picture,
        )
        from fibsem.fm.preview import is_fluorescence_image
        from fibsem.structures import FibsemImage

        experiment = self._experiment
        if experiment is None:
            return {"error_type": "not_found", "error": "No experiment is open."}
        target, refusal = self._experiment_image(path)
        if refusal:
            return refusal
        lamellae, linked, reason = list(experiment.positions), None, ""
        if grid_name is not None:
            grid = experiment.get_grid_by_name(grid_name)
            if grid is None:
                return dict(
                    self._no_such_grid(experiment, grid_name), error_type="not_found"
                )
            lamellae, linked, reason = _grid_items(experiment, grid)
        if is_fluorescence_image(target):
            return {
                "error_type": "invalid",
                "error": f"{path!r} is a fluorescence image; the items are drawn on a beam image.",
            }
        try:
            image = FibsemImage.load(target)
        except Exception as exc:  # noqa: BLE001 - an unreadable file is a data fact
            return {"error_type": "invalid", "error": f"{path!r}: {exc}"}
        stem = os.path.splitext(os.path.basename(target))[0]
        saved = os.path.join(os.path.dirname(target), "markers", f"{stem}.png")
        stage = None
        if current and self._microscope is not None:
            stage = self._microscope.get_stage_position()
        jpeg = save_markers_picture(image, lamellae, saved, fov=fov, current=stage)
        answer = {
            "image": path,
            "path": saved,
            "items": [lam.name for lam in lamellae],
            "plot": {"image_b64_jpeg": base64.b64encode(jpeg).decode()},
        }
        if grid_name is not None:
            answer.update(grid_name=grid_name, linked=linked)
        if reason:
            answer["reason"] = reason
        return answer

    def draw_item_milling(
        self,
        item_name: str,
        task_names: List[str],
        image_path: Optional[str] = None,
        background: str = "related",
        scan_direction: bool = False,
        alignment_area: bool = False,
        metadata_bar: bool = True,
    ) -> Any:
        """An item's milling as its own task configs have it, drawn on an ion
        image (the lamella editor's: the latest FIB image, else *image_path*):
        the first of *task_names*' enabled stages in the milling view's
        colours, the others behind them in black, with the tasks related to
        the first (*background* "related") as the editor shows them, and the
        item's point of interest. Saved as ``drawn/<image>-<task>.png`` beside
        the image, a path the agent can cite as evidence."""
        from fibsem.imaging.marks import drawn_path, image_stem, save_drawn
        from fibsem.milling.patterning.picture import milling_picture
        from fibsem.structures import FibsemImage

        experiment = self._experiment
        if experiment is None:
            return {"error_type": "not_found", "error": "No experiment is open."}
        lamella, refusal = find_item(experiment, item_name)
        if refusal:
            return dict(refusal, error_type="not_found")
        configs = dict(lamella.task_config)
        missing = [
            name
            for name in task_names
            if not getattr(configs.get(name), "milling", None)
        ]
        if not task_names or missing:
            return {
                "error_type": "not_found",
                "error": f"{item_name} has no milling task named "
                f"{(missing or [''])[0]!r}.",
                "task_names": [n for n, c in configs.items() if c.milling],
            }
        if image_path:
            target, refusal = self._experiment_image(image_path)
            if refusal:
                return refusal
        else:
            relative = _task_outputs.latest_task_image(lamella, "_ib.tif")
            if not relative:
                return {
                    "error_type": "not_found",
                    "error": f"{item_name} has no FIB image yet: pass image_path.",
                }
            target = os.path.join(str(lamella.path), relative)

        def enabled(name):
            return [
                stage
                for task_milling in configs[name].milling.values()
                for stage in task_milling.enabled_stages
            ]

        stages = enabled(task_names[0])
        behind = [stage for name in task_names[1:] for stage in enabled(name)]
        if background == "related":
            behind += [
                stage
                for stage in lamella.related_milling_stages(task_names[0])
                if all(stage is not other for other in behind)
            ]
        try:
            image = FibsemImage.load(target)
            rgb = milling_picture(
                image,
                stages,
                behind,
                scan_direction=scan_direction,
                alignment_area=lamella.alignment_area if alignment_area else None,
                poi=lamella.poi,
                metadata_bar=metadata_bar,
            )
        except Exception as exc:  # noqa: BLE001 - an image that can't be drawn on
            return {"error_type": "invalid", "error": f"{target}: {exc}"}
        saved = drawn_path(target, f"{image_stem(target)}-{task_names[0]}")
        return ItemMillingDrawn(
            item_name,
            list(task_names),
            target,
            saved,
            [stage.name for stage in stages],
            [stage.name for stage in behind],
            save_drawn(rgb, saved),
        )

    @staticmethod
    def _grid_summary(experiment, grid) -> Dict[str, Any]:
        import os

        history = list(grid.task_history)
        completed = [t.name for t in history if t.status is TaskStatus.Completed]
        overviews = {}
        for role in GRID_OVERVIEW_ROLES:
            path = latest_item_output(experiment, grid, role)
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

    # --- experiments and protocols: the File menu's AutoLamella files ---------

    @property
    def protocol_type(self):
        from fibsem.applications.autolamella.structures import AutoLamellaTaskProtocol

        return AutoLamellaTaskProtocol

    def _protocol_folder(self) -> Optional[str]:
        from fibsem.applications.autolamella import experiment_setup

        return experiment_setup.protocol_directory()

    def _default_protocol_path(self) -> Optional[str]:
        from fibsem.applications.autolamella import experiment_setup

        return experiment_setup.default_protocol_path()

    def _new_experiment(self, directory, name, protocol, metadata):
        from fibsem.applications.autolamella import experiment_setup

        return experiment_setup.create_experiment(directory, name, protocol, metadata)

    def _load_for_adoption(self, path):
        from fibsem.applications.autolamella import experiment_setup

        return experiment_setup.load_for_adoption(path)

    # --- lamella positions: the Lamella tab's buttons, for an agent at Drive ------

    def _config_edit(
        self, level, item_name, task_name, patch, version, timeout: float, shown=None
    ) -> Dict[str, Any]:
        """An agent's config edit (``config_edits``) where the session edits
        the experiment, said on ``session.changed`` when applied; *shown* is the
        level said, when not *level*."""
        from fibsem.applications.autolamella.config_edits import apply_agent_edit

        session = self._session

        def edit():
            result = apply_agent_edit(
                session.experiment,
                lambda: session.microscope,
                level,
                item_name,
                task_name,
                patch,
                version,
            )
            if result.get("applied"):
                session.changed.emit(
                    CONFIG,
                    {
                        "level": shown or level,
                        "item_name": item_name,
                        "task_name": task_name,
                        "result": result,
                    },
                )
            return result

        result = session.run_edit(edit).result(timeout=timeout)
        result["available"] = True
        return result

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
        """The stage position under a point of a grid overview: one its tasks
        recorded, or any saved image in the experiment (`_grid_file`).

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
        path, refusal = _grid_file(experiment, grid, filename)
        if refusal:
            return {"error_type": "not_found", **refusal}
        try:
            position = image_point_to_stage_position(path, px=px, fraction=fraction)
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
        position, grid_id, refusal = self._marked_position(stage_position, overview)
        if refusal:
            return refusal

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
            added = self._agent_call(add)
        except ValueError as exc:
            return {"error_type": "invalid", "error": str(exc)}
        return {"applied": True, "item": self.item_detail(added)}

    def _marked_position(
        self,
        stage_position: Optional[Dict[str, Any]],
        overview: Optional[Dict[str, Any]],
    ):
        """``(position, grid_id, refusal)`` for a point given as a stage position
        or under a point of a grid overview (which also names its grid); all
        None for neither."""
        from fibsem.structures import FibsemStagePosition

        if overview is not None:
            converted = self.overview_stage_position(
                overview.get("grid_name", ""),
                overview.get("filename", ""),
                px=overview.get("px"),
                fraction=overview.get("fraction"),
            )
            if "error_type" in converted:
                return None, None, converted
            position = FibsemStagePosition.from_dict(converted["stage_position"])
            return position, converted["grid_id"], None
        if stage_position is not None:
            # {x, y, z, r, t} as the tool describes it, in the stage's raw frame,
            # as the stage tools give positions; from_dict also wants the frame.
            position = FibsemStagePosition.from_dict(
                {"coordinate_system": "RAW", **stage_position}
            )
            return position, None, None
        return None, None, None

    def remove_item(self, item_name: str) -> Dict[str, Any]:
        """Remove a lamella from the experiment, as its row's Remove does. Its
        files stay on disk. Refused for the lamella a run is working on."""
        from fibsem.applications.autolamella.run_edits import lamella_edit_refusal

        experiment, _, refusal = self._editable()
        if refusal:
            return refusal
        lamella, refusal = find_item(experiment, item_name)
        if refusal:
            return {"error_type": "not_found", **refusal}
        refusal = self._busy(
            lamella_edit_refusal(self._runner, experiment, lamella, "remove it")
        )
        if refusal:
            return refusal

        from fibsem.applications.autolamella.poses import remove_lamella

        self._agent_call(lambda: remove_lamella(experiment, lamella))
        return {"applied": True, "item_name": item_name}

    # -- grid records: the Grids tab's inventory read and Rename ----------------

    def _grids_changed(self) -> None:
        """Tell whoever draws the grid records they changed."""
        self._session.changed.emit(GRID_RECORDS)

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
        from fibsem.workflows.grid.naming import record_inventoried_grids

        stage.get_inventory()

        def add():
            added = record_inventoried_grids(experiment, stage, self.record_event)
            if added:
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
        grid has, and while a run is on it or has it queued (its queue holds
        grids by name)."""
        from fibsem.applications.autolamella.run_edits import grid_edit_refusal
        from fibsem.workflows.grid.naming import (
            grid_rename_refusal,
            rename_and_save_grid,
        )

        name = name.strip()
        if not name:
            return {"error_type": "invalid", "error": "A grid needs a name."}
        experiment, microscope, refusal = self._editable()
        if refusal:
            return refusal
        refusal = self._busy(
            grid_edit_refusal(
                self._runner, experiment, grid_name, "rename it", renaming=True
            )
        )
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
                rename_and_save_grid(
                    experiment, stage, grid_name, name, self.record_event
                )
            except Exception as exc:  # noqa: BLE001 - the hardware's refusal, said
                return str(exc)
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
        from fibsem.workflows.quality import Verdict

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
                description=reason or "", state=state, actor=Actor.agent("remote")
            )
            experiment.save_verdict(grid)
            self._grids_changed()

        self._agent_call(judge)
        return {
            "applied": True,
            "grid_name": grid_name,
            "quality": grid.quality.to_dict(),
        }

    def set_grid_note(
        self, grid_name: str, note: str, evidence: Optional[str] = None
    ) -> Dict[str, Any]:
        """Keep a note on a grid's record, as the Grids tab's Note does; an empty
        note clears it. Apart from the verdict, so judging the grid again keeps
        it. Saved, and a note (not a clearing) put on the record as the
        agent's, citing the *evidence* (its id) it rests on."""
        experiment, grid, refusal = self._grid_record(grid_name)
        if refusal:
            return refusal

        def keep():
            record = keep_grid_note(experiment, grid, note, evidence)
            self._grids_changed()
            return record

        record = self._agent_call(keep)
        if record is not None:
            self.record_event(NOTE, record)
        return {"applied": True, "grid_name": grid_name, "note": note}

    def remove_grid(self, grid_name: str) -> Dict[str, Any]:
        """Stop tracking a grid, as the Grids tab's Remove does: its files stay,
        its lamellae stay with no grid. Refused while a run is on it or one of
        its lamellae. Saved."""
        from fibsem.applications.autolamella.run_edits import grid_edit_refusal

        experiment, grid, refusal = self._grid_record(grid_name)
        if refusal:
            return refusal
        refusal = self._busy(
            grid_edit_refusal(self._runner, experiment, grid_name, "remove it")
        )
        if refusal:
            return refusal

        def remove():
            orphaned = remove_grid_record(experiment, grid)
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

        followed = self._agent_call(
            lambda: set_lamella_pose(experiment, microscope, lamella, pose_name, state)
        )
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
        position. Saved. Not while a run is on: the run moves the objective."""
        experiment, microscope, refusal = self._editable()
        if refusal:
            return refusal
        refusal = self._not_while_running("store the objective position")
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

        from fibsem.applications.autolamella.poses import set_objective_position

        def store():
            set_objective_position(experiment, lamella, value)
            # A stored objective position emits nothing of its own.
            experiment.positions.events.changed.emit()

        self._agent_call(store)
        return {"applied": True, "item_name": item_name, "objective_position": value}

    def move_item(
        self,
        item_name: str,
        stage_position: Optional[Dict[str, Any]] = None,
        overview: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Move a lamella to a point, as moving it on an overview does: to
        stage_position, or under a point of a grid overview. A point on the
        fluorescence side moves its fluorescence pose and the milling pose
        follows; otherwise the other way round. The stage doesn't move. Saved.
        Refused for the lamella a run is working on."""
        from fibsem.applications.autolamella.poses import (
            FLUORESCENCE_ORIENTATION,
            fluorescence_move,
            move_lamella,
        )
        from fibsem.applications.autolamella.run_edits import lamella_edit_refusal

        experiment, microscope, refusal = self._editable()
        if refusal:
            return refusal
        if stage_position is None and overview is None:
            return {
                "error_type": "invalid",
                "error": "Give stage_position or overview: where to move it to.",
            }
        lamella, refusal = find_item(experiment, item_name)
        if refusal:
            return {"error_type": "not_found", **refusal}
        refusal = self._busy(
            lamella_edit_refusal(self._runner, experiment, lamella, "move it")
        )
        if refusal:
            return refusal
        if lamella.milling_pose is None:
            return {
                "error_type": "invalid",
                "error": f"{item_name} has no pose to move.",
            }
        position, _, refusal = self._marked_position(stage_position, overview)
        if refusal:
            return refusal
        poses = None
        if microscope.get_stage_orientation(position) == FLUORESCENCE_ORIENTATION:
            try:
                poses = fluorescence_move(microscope, lamella, position)
            except ValueError as exc:
                return {"error_type": "invalid", "error": str(exc)}
        self._agent_call(
            lambda: move_lamella(experiment, microscope, lamella, position, poses)
        )
        return {"applied": True, "item": self.item_detail(item_name)}

    def set_objective_position_on_all(self, item_name: str) -> Dict[str, Any]:
        """Store a lamella's objective position on every lamella with a
        fluorescence pose, as its Apply to All Lamella does, except the one a
        run is working on. Saved."""
        from fibsem.applications.autolamella.poses import set_objective_position_on_all
        from fibsem.applications.autolamella.run_edits import busy_lamellae

        experiment, _, refusal = self._editable()
        if refusal:
            return refusal
        lamella, refusal = find_item(experiment, item_name)
        if refusal:
            return {"error_type": "not_found", **refusal}
        pose = lamella.fluorescence_pose
        value = pose.objective_position if pose is not None else None
        if value is None:
            return {
                "error_type": "not_found",
                "error": f"{item_name} has no objective position stored.",
            }
        skipped = busy_lamellae(self._runner, experiment)

        def store():
            count = set_objective_position_on_all(experiment, value, skip=skipped)
            # A stored objective position emits nothing of its own.
            experiment.positions.events.changed.emit()
            return count

        count = self._agent_call(store)
        return {
            "applied": True,
            "objective_position": value,
            "items": count,
            "skipped": skipped,
        }

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

    def detail_of(self, experiment, item) -> LamellaDetail:
        from fibsem.workflows.events import to_plain

        detail = super().detail_of(experiment, item)
        detail.is_failure = bool(item.is_failure)
        detail.milling_angle = item.milling_angle
        if getattr(item, "defect", None) is not None:
            detail.defect = to_plain(item.defect.to_dict())
        return detail

    def item_version(self, item) -> str:
        return item_fields_version(item)

    # --- task configs (read side) --------------------------------------

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
        """A config edit in an agent's request, as the config's form names it:
        each setting in display units, old -> new. None for any
        other action, or a config this experiment doesn't have."""
        from fibsem.util.wording import describe_patch

        config = self._edited_task_config(tool, arguments)
        if config is None:
            return None
        return describe_patch(config, arguments.get("patch"))

    def _edited_task_config(self, tool: str, arguments: Dict[str, Any]) -> Any:
        """The task config a task-config edit targets; None for any other tool,
        or a target this experiment doesn't have."""
        experiment = self._experiment
        if experiment is None:
            return None
        if tool == "item_update_task_config":
            owner = experiment.get_lamella_by_name(arguments.get("item_name"))
        elif tool == "protocol_update_task_config":
            owner = getattr(experiment, "task_protocol", None)
        elif tool == "protocol_update_grid_task_config":
            owner = getattr(experiment, "grid_protocol", None)
        else:
            return None
        configs = getattr(owner, "task_config", None) if owner else None
        if configs is None:
            return None
        return dict(configs).get(arguments.get("task_name"))

    def check_action(self, tool: str, arguments: Dict[str, Any]) -> Optional[str]:
        """Why a config edit in an agent's request would be refused when it
        runs, checked at filing so the operator never approves one that can only
        fail. None for any other action, or one whose target the route itself
        refuses (no such item or task)."""
        from fibsem.workflows.config_patch import PatchError, check_patch

        experiment = self._experiment
        patch = arguments.get("patch")
        if experiment is None or not isinstance(patch, dict):
            return None
        if tool == "item_update":
            lamella = experiment.get_lamella_by_name(arguments.get("item_name"))
            return item_patch_refusal(lamella, patch) if lamella else None
        config = self._edited_task_config(tool, arguments)
        if config is None:
            return None
        if tool == "item_update_task_config":
            running = self._running_task_refusal(
                arguments.get("item_name"), arguments.get("task_name")
            )
            if running:
                return running
        if tool == "protocol_update_grid_task_config":
            patch = {k: v for k, v in patch.items() if k != "attention"}
        else:
            refused = workflow_field_refusal(patch)
            if refused:
                return refused["invalid_patch"]
        try:
            check_patch(config, patch)
        except PatchError as exc:
            return str(exc)
        return None

    def _running_task_refusal(self, item_name: str, task_name: str) -> Optional[str]:
        """Why an edit to this item's task config can't take effect: the task is
        running for it now and has already copied its config. None otherwise."""
        manager = self._manager
        if manager is None:
            return None
        running = any(
            item.item_name == item_name
            and item.task_name == task_name
            and item.status is TaskStatus.InProgress
            for item in manager.queue.items
        )
        if not running:
            return None
        return (
            f"{task_name!r} is running for {item_name!r} right now and has "
            "already copied its config — the edit would never take effect. "
            "Patch it after the task finishes."
        )

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
        return self._config_edit(
            "item_fields", item_name, "", dict(patch), str(version), timeout
        )

    def apply_item_correlation(
        self, item_name: str, run_path: str, timeout: float = 10.0
    ) -> Dict[str, Any]:
        """Move an item's point of interest to a saved correlation run's target,
        as the correlation dialog's Continue does, and record the correlation.

        *run_path* is a run folder in the item's own Correlation folder (as the
        correlation service saves it). The move is an item patch of ``poi``, so
        it is version-checked, the patterns that follow the point move with it,
        and it is saved and shown as the agent's edit.
        """
        from fibsem.correlation.history import correlation_dir, load_run

        experiment = self._experiment
        if experiment is None:
            return {"error_type": "not_found", "error": "No experiment is loaded."}
        lamella, refusal = find_item(experiment, item_name)
        if refusal:
            return {"error_type": "not_found", **refusal}
        root = os.path.realpath(correlation_dir(str(lamella.path)))
        folder = os.path.realpath(run_path)
        if os.path.commonpath([root, folder]) != root or folder == root:
            return {
                "error_type": "invalid",
                "error": f"run_path must be a run folder in {item_name}'s "
                f"Correlation folder ({root}), not {run_path}.",
            }
        try:
            result = load_run(folder).result
        except FileNotFoundError as exc:
            return {"error_type": "not_found", "error": str(exc)}
        if result is None or not result.poi:
            return {
                "error_type": "invalid",
                "error": f"the run at {run_path} has no fitted point of interest.",
            }
        version = item_fields_version(lamella)
        applied = apply_correlation(
            result,
            lamella,
            folder,
            experiment.path,
            lambda poi: self.apply_item_patch(
                item_name, poi_patch(poi), version, timeout=timeout
            ),
            self.record_event,
        )
        if applied.get("applied"):
            applied["run_path"] = run_path
        return applied

    # --- proposals and decisions ------------------------------------

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

        Same rules as the config patches: version-guarded beside the apply,
        refused while that task runs for that item, recorded on the event
        stream. Same-set-by-name: this can never add, drop, or duplicate a
        stage.
        """
        if level == "item":
            manager = self._manager
            if manager is not None:
                running = any(
                    item.item_name == item_name
                    and item.task_name == task_name
                    and item.status is TaskStatus.InProgress
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
        payload = {"level": level, "milling_key": milling_key, "order": list(order)}
        return self._config_edit(
            "reorder_stages",
            item_name,
            task_name,
            payload,
            str(version),
            timeout,
            shown=level,
        )

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
                and item.status is TaskStatus.InProgress
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
        return self._config_edit(
            "apply_protocol", item_name, "", {"task_names": task_names}, "", timeout
        )

    def apply_protocol_task_config_patch(
        self,
        task_name: str,
        patch: Dict[str, Any],
        version: str,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Patch a task's protocol-level defaults — what new items copy.

        Same version-guarded apply as the per-item patch. No
        running-task guard: a running task holds its item's copy, so a
        protocol edit can never touch it — the edit reaches items created
        (or re-copied) after it lands.
        """
        return self._config_edit(
            "protocol", "", task_name, dict(patch), str(version), timeout
        )

    @staticmethod
    def _config_document(task_name: str, config, level: str) -> Dict[str, Any]:
        """One config as a wire document with its version token.

        The full ``to_dict``, not a curated snapshot — the document is the
        contract for editing (you cannot patch what you cannot read). The
        ``version`` hashes the serialized settings, so a write names
        exactly the state it read: the config's version, refused as stale on
        mismatch.
        """
        from fibsem.workflows.events import to_plain

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
        was written against; the check happens beside the
        apply, so an edit landing in between is refused as stale, never
        half-merged. A task currently running for this item is refused too —
        it already copied its config, so the edit would take effect never,
        and a refusal is more honest than a silent no-op. Pending tasks pick
        the change up when they start (the ``set_supervision`` semantics).
        """
        why = self._running_task_refusal(item_name, task_name)
        if why:
            return {
                "available": True,
                "applied": False,
                "error_type": "task_running",
                "error": why,
            }
        return self._config_edit(
            "item", item_name, task_name, dict(patch), str(version), timeout
        )

    # --- workflow control -------------------------------------------------------

    def lamella_run_plan(
        self,
        task_names: List[str],
        item_names: Optional[List[str]] = None,
        run_next: bool = False,
        add: bool = False,
    ) -> Dict[str, Any]:
        """What a lamella run would queue and cost, as the Workflow tab's
        confirmation shows it, or with *add* what adding it to the running
        queue would: no hardware, nothing queued (`plan_lamella_run`)."""
        from fibsem.applications.autolamella.workflows.plan import plan_lamella_run

        experiment = self._experiment
        if experiment is None or experiment.task_protocol is None:
            return {
                "available": False,
                "valid": False,
                "reason": "no experiment is loaded",
            }
        request = RunRequest(
            kind=ITEMS,
            task_names=tuple(task_names),
            names=tuple(item_names) if item_names is not None else None,
        )
        plan = plan_lamella_run(self._runner, experiment, request, run_next, add)
        answer: Dict[str, Any] = {
            "available": True,
            "valid": plan.valid,
            "add": add,
            "run_next": plan.run_next,
        }
        if not plan.valid:
            return {**answer, "reason": plan.reason, **plan.details}
        return {
            **answer,
            "steps": [{"item": item, "task": task} for item, task in plan.pairs],
            "already_queued": list(plan.already_queued),
            "estimate": _run_estimate(plan.estimate),
            "addition": _addition_estimate(plan.addition),
        }

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
        is a live stage fact this facade does not read. ``estimate`` is the
        preflight dialog's (`grid_run_estimate`) without them, with each beam
        overview's tiles and size.
        """
        from fibsem.workflows.grid.manager import (
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
                "estimate": _grid_plan_estimate(experiment, task_names, known_grids),
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
            "estimate": _grid_plan_estimate(experiment, task_names, grid_names),
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

        Started like :meth:`start_workflow`; shares its run slot, so it is refused while any workflow — lamella or grid —
        is running. The agent at Drive stands in for the preflight dialog's
        Run click; :meth:`grid_workflow_plan` is that dialog's content.
        """
        result = self._start_run(
            RunRequest(
                kind=GRIDS,
                task_names=tuple(task_names),
                names=tuple(grid_names) if grid_names is not None else None,
                options={
                    "inventory_first": bool(screen_all),
                    "unload_at_end": bool(unload_at_end),
                    "report_at_end": bool(report_at_end),
                },
            ),
            timeout,
        )
        if result.get("started"):
            self.record_event(
                "workflow_start",
                {"task_names": list(task_names), "grid_names": grid_names},
            )
        return result

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
        why = updated.why_invalid()
        if why:
            return {"error_type": "invalid", "error": why}

        def edit():
            protocol.lamella_defaults = updated
            return self._protocol_changed(
                experiment, {"field": "lamella_defaults", "value": updated.to_dict()}
            )

        saved = self._agent_call(edit)
        return {"applied": True, "saved": saved, "lamella_defaults": updated.to_dict()}

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
        grid_protocol = experiment.grid_protocol
        config, refusal = self._grid_task(grid_protocol, task_name)
        if refusal:
            return refusal
        document = self._config_document(task_name, config, "grid_protocol")
        document["config"]["attention"] = grid_protocol.get_attention(task_name).value
        document["version"] = _grid_task_version(grid_protocol, task_name, config)
        return document

    def apply_grid_task_config_patch(
        self, task_name: str, patch: Dict[str, Any], version: str
    ) -> Dict[str, Any]:
        """Patch a grid task's settings with dotted-path edits, as the Protocol
        tab's grid task form does; ``attention`` (automated, supervised) is the
        Grids view's Automated/Supervised chip. Version-guarded like the lamella
        task configs."""
        from fibsem.workflows.config_patch import PatchError, apply_patch
        from fibsem.workflows.events import to_plain

        experiment, grid_protocol, refusal = self._grid_protocol_edit("edit grid tasks")
        if refusal:
            return refusal
        config, refusal = self._grid_task(grid_protocol, task_name)
        if refusal:
            return refusal

        def edit():
            if _grid_task_version(grid_protocol, task_name, config) != version:
                return {"stale": True}
            settings = {k: v for k, v in patch.items() if k != "attention"}
            old = grid_protocol.get_attention(task_name)
            try:
                new = Attention(patch.get("attention", old))
            except ValueError:
                return {
                    "invalid_patch": "attention is automated or supervised.",
                    "path": "attention",
                }
            try:
                changes = apply_patch(config, settings)
            except PatchError as exc:
                return {"invalid_patch": str(exc), "path": exc.path}
            if new is not old:
                grid_protocol.set_attention(task_name, new)
                changes.append(("attention", old.value, new.value))
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
            "version": _grid_task_version(grid_protocol, task_name, config),
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

        def reorder():
            why = grid_protocol.reorder(list(order))
            if why:
                return why
            return self._protocol_changed(
                experiment, {"field": "grid_order", "order": order}
            )

        saved = self._agent_call(reorder)
        if isinstance(saved, str):
            return {
                "error_type": "invalid",
                "error": saved,
                "task_names": grid_protocol.ordered_task_names,
            }
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
        _, refusal = self._grid_task(grid_protocol, task_name)
        if refusal:
            return refusal

        def edit():
            why = grid_protocol.set_requires(task_name, list(requires))
            if why:
                return why
            return self._protocol_changed(
                experiment,
                {
                    "field": "grid_requirements",
                    "task_name": task_name,
                    "requires": grid_protocol.requirements(task_name),
                },
            )

        saved = self._agent_call(edit)
        if isinstance(saved, str):
            return {"error_type": "invalid", "error": saved}
        return {
            "applied": True,
            "saved": saved,
            "task_name": task_name,
            "requires": grid_protocol.requirements(task_name),
        }

    def queue_grid_tasks(
        self, task_names: List[str], grid_names: List[str]
    ) -> Dict[str, Any]:
        """Add grid tasks for grids to the running grid run, at the end of the
        queue, as the Grids view's Add to queue does: each grid's load, then
        its tasks, less what is already waiting."""
        request = RunRequest(
            kind=GRIDS, task_names=tuple(task_names), names=tuple(grid_names)
        )
        edit = self._agent_call(lambda: self._session.runner.add(request))
        if not edit.applied:
            return {"queued": False, "reason": edit.reason, **edit.details}
        return {
            "queued": True,
            "added": [
                {"item_id": i.id, "grid_name": i.item_name, "step": i.task_name}
                for i in edit.items
            ],
            "queue_version": edit.version,
        }


def _when(value) -> Optional[str]:
    return value.isoformat() if value is not None else None


def _run_estimate(estimate) -> Optional[Dict[str, Any]]:
    """A lamella run's estimate as the preflight dialog shows it."""
    if estimate is None:
        return None
    return {
        "work_seconds": estimate.work_seconds,
        "hold_seconds": estimate.hold_seconds,
        "expected_finish": _when(estimate.expected_finish),
        "tasks": [
            {
                "task_name": row.name,
                "count": row.count,
                "seconds": row.seconds,
                "scheduled_at": _when(row.scheduled_at),
                "hold_seconds": row.hold_seconds,
            }
            for row in estimate.tasks
        ],
    }


def _addition_estimate(addition) -> Optional[Dict[str, Any]]:
    """What adding to the running queue costs, as its confirmation shows it."""
    if addition is None:
        return None
    return {
        "work_seconds": addition.work_seconds,
        "delay_seconds": addition.delay_seconds,
        "finish_before": _when(addition.finish_before),
        "finish_after": _when(addition.finish_after),
        "hold_seconds": addition.hold_seconds,
        "priced": addition.is_priced,
    }


def _grid_plan_estimate(experiment, task_names, grid_names) -> Optional[Dict[str, Any]]:
    """A grid run's time, exchanges not counted, and each beam overview's
    tiles and size: what its confirmation says. None when it can't say."""
    from fibsem.applications.autolamella.workflows.estimate import grid_run_estimate
    from fibsem.workflows.grid.imaging import BeamOverviewGridTaskConfig

    estimate = grid_run_estimate(
        experiment, None, list(task_names), list(grid_names), exchanges=0
    )
    if estimate is None:
        return None
    configs = experiment.grid_protocol.task_config
    tasks = []
    for row in estimate.tasks:
        task: Dict[str, Any] = {
            "task_name": row.name,
            "count": row.count,
            "seconds": row.seconds,
        }
        config = configs.get(row.name)
        if isinstance(config, BeamOverviewGridTaskConfig):
            settings = config.settings
            task["overview"] = {
                "nrows": settings.nrows,
                "ncols": settings.ncols,
                "tiles": settings.n_enabled_tiles,
                "tile_hfw": settings.image_settings.hfw,
                "width": float(settings.total_fov_x),
                "height": float(settings.total_fov_y),
            }
        tasks.append(task)
    return {
        "seconds": estimate.work_seconds,
        "exchanges_counted": False,
        "tasks": tasks,
    }
