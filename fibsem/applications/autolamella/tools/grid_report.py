"""What a grid screening report says, read off the experiment (FIB-1057).

The report exists so that someone can be handed a screened magazine and asked
"which of these do you want?" without opening the app. This module is the data
side of it: it walks every grid record, its history and the outputs recorded on
it, and answers with plain dataclasses a renderer lays out. Nothing here touches
the microscope, imports Qt or reportlab, or globs a directory -- every path comes
from a history entry's `outputs`, as the Grids tab reads them.

Images are read for their metadata only. A stitched overview is tens of
megapixels; what a caption and a lamella marker need (pixel size, field of view,
beam, the stage position it was taken at, the geometry to project with) sits in
the tiff's description tag and costs nothing to read. The renderer decides how
much of the pixel data to load.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from math import degrees
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import tifffile as tff

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskState,
    AutoLamellaTaskStatus,
    Experiment,
    GridRecord,
    Lamella,
    QualityRecord,
    Verdict,
)
from fibsem.applications.autolamella.workflows.tasks.grid.manager import (
    LOAD_ENTRY_NAME,
)
from fibsem.fm.preview import is_fluorescence_image
from fibsem.fm.structures import FluorescenceImageMetadata, safe_ome_from_tiff
from fibsem.projection import BeamStageProjection, FMStageProjection
from fibsem.structures import BeamType, FibsemImageMetadata, FibsemStagePosition

logger = logging.getLogger(__name__)

# The output roles a grid task records, and the modality each stands for.
OVERVIEW_ROLES = ("overview_sem", "overview_fib", "overview_fm")
_MODALITY = {"overview_sem": "SEM", "overview_fib": "FIB", "overview_fm": "FM"}

# The box drawn around a marked lamella: the field of view a lamella image
# covers. The same size the Positions view draws
# (`fibsem.ui.widgets.stored_overview_canvas.POSITION_FOV_*`), restated here so
# this module stays free of `fibsem.ui`; keep the two in step.
LAMELLA_BOX_WIDTH = 100e-6
LAMELLA_BOX_HEIGHT = LAMELLA_BOX_WIDTH * (1024 / 1536)


@dataclass
class LamellaMark:
    """Where a lamella falls on an overview, in that image's own pixels."""

    name: str
    x: float
    y: float
    width: float
    height: float


@dataclass
class OverviewEntry:
    """One history entry on a grid: a run of an overview task, with what it left.

    `path` is None when the run recorded no image -- it failed, was cancelled or
    skipped -- and the row then reads as a gap in the sequence, with the status
    message saying why. The geometry fields are None when the image on disk
    carries no metadata to read them from.
    """

    task_name: str
    role: str
    modality: str  # SEM, FIB, FM
    status: AutoLamellaTaskStatus
    status_message: str
    started: float
    ended: Optional[float]
    path: Optional[str] = None
    pose: str = ""  # the stage pose it was taken at, as text
    tiles: Optional[Tuple[int, int]] = None  # (rows, cols)
    fov: Optional[Tuple[float, float]] = None  # (width, height), metres
    pixel_size: Optional[float] = None
    shape: Optional[Tuple[int, int]] = None  # (height, width)
    channels: List[str] = field(default_factory=list)
    marks: List[LamellaMark] = field(default_factory=list)
    # Lamellae on the grid that could not be placed on this image: no pose for
    # its modality, or no geometry to project with. Named so a caption can say so.
    unmarked: List[str] = field(default_factory=list)

    @property
    def when(self) -> float:
        return self.ended or self.started


@dataclass
class LoadEntry:
    status: AutoLamellaTaskStatus
    status_message: str
    when: float


@dataclass
class GridSection:
    """One grid: the operator's verdict, whether it loaded, and its overviews."""

    name: str
    quality: QualityRecord  # the verdict, who set it and why
    description: str  # the free-text note on the record
    slot: Optional[str]
    load: Optional[LoadEntry]
    lamellae: List[str]
    overviews: List[OverviewEntry]

    @property
    def recommended(self) -> bool:
        return self.quality.verdict is Verdict.GOOD

    @property
    def loaded(self) -> Optional[bool]:
        """True or False by the latest load entry; None when nothing tried."""
        if self.load is None:
            return None
        return self.load.status is AutoLamellaTaskStatus.Completed


@dataclass
class TaskOutcome:
    """How a grid's runs of one protocol task ended: the latest, and how many."""

    status: Optional[AutoLamellaTaskStatus]  # None: never ran on this grid
    runs: int


@dataclass
class GridReport:
    experiment_name: str
    experiment_path: str
    created_at: float
    generated_at: float
    microscope: Optional[str]
    protocol: List[str]  # the grid protocol's tasks, in order
    sections: List[GridSection]
    outcomes: Dict[str, Dict[str, TaskOutcome]]  # grid name -> task name -> outcome

    @property
    def screened(self) -> Optional[Tuple[float, float]]:
        """When the first and last recorded entry happened, or None if nothing ran."""
        stamps = [
            stamp
            for section in self.sections
            for stamp in ([section.load.when] if section.load else [])
            + [o.when for o in section.overviews]
        ]
        if not stamps:
            return None
        return min(stamps), max(stamps)


# ---------------------------------------------------------------------------
# Reading an image's metadata without its pixels
# ---------------------------------------------------------------------------


def read_beam_metadata(
    path: str,
) -> Tuple[Optional[FibsemImageMetadata], Optional[Tuple[int, int]]]:
    """A FibsemImage's metadata and (height, width), without loading the array."""
    try:
        with tff.TiffFile(path) as tiff:
            page = tiff.pages[0]
            shape = (int(page.shape[0]), int(page.shape[1]))
            description = page.tags["ImageDescription"].value
        return FibsemImageMetadata.from_dict(json.loads(description)), shape
    except Exception as e:  # noqa: BLE001 - an image with no metadata is still listed
        logger.debug(f"No readable metadata on {path}: {e}")
        return None, None


def read_fm_metadata(path: str) -> Optional[FluorescenceImageMetadata]:
    """A FluorescenceImage's metadata from its OME annotations, without the volume."""
    try:
        ome = safe_ome_from_tiff(path)
        annotations = ome.structured_annotations
        for annotation in annotations.map_annotations if annotations else []:
            if annotation.value and "FluorescenceImageMetadata" in annotation.value:
                return FluorescenceImageMetadata.from_dict(
                    json.loads(annotation.value["FluorescenceImageMetadata"])
                )
    except Exception as e:  # noqa: BLE001
        logger.debug(f"No readable metadata on {path}: {e}")
    return None


def _pose_text(position: Optional[FibsemStagePosition]) -> str:
    if position is None:
        return ""
    r = degrees(float(position.r or 0.0))
    t = degrees(float(position.t or 0.0))
    return f"r={r:.0f}° t={t:.0f}°"


# ---------------------------------------------------------------------------
# Marking lamellae
# ---------------------------------------------------------------------------


def _pose_of(lamella: Lamella, modality: str) -> Optional[FibsemStagePosition]:
    """The pose the given view puts a lamella at: its fluorescence pose on the FM,
    its milling pose on a beam image. The Positions view's rule."""
    if modality == "FM":
        pose = lamella.fluorescence_pose
        return pose.stage_position if pose is not None else None
    return lamella.stage_position if lamella.milling_pose is not None else None


def _mark(
    lamellae: Sequence[Lamella],
    modality: str,
    projection,
    base: Optional[FibsemStagePosition],
    pixel_size: Optional[float],
    shape: Optional[Tuple[int, int]],
) -> Tuple[List[LamellaMark], List[str]]:
    """Project every lamella onto the image; name the ones that cannot be."""
    marks: List[LamellaMark] = []
    unmarked: List[str] = []
    placeable = (
        projection is not None and base is not None and pixel_size and shape is not None
    )
    for lamella in lamellae:
        pose = _pose_of(lamella, modality)
        if pose is None or not placeable:
            unmarked.append(lamella.name)
            continue
        try:
            dx, dy = projection.to_plane(pose, base)
        except Exception as e:  # noqa: BLE001 - one bad pose must not lose the row
            logger.debug(f"Could not place {lamella.name}: {e}")
            unmarked.append(lamella.name)
            continue
        # The plane's y runs down, image-fashion, so no flip here.
        marks.append(
            LamellaMark(
                name=lamella.name,
                x=shape[1] / 2 + dx / pixel_size,
                y=shape[0] / 2 + dy / pixel_size,
                width=LAMELLA_BOX_WIDTH / pixel_size,
                height=LAMELLA_BOX_HEIGHT / pixel_size,
            )
        )
    return marks, unmarked


# ---------------------------------------------------------------------------
# One history entry
# ---------------------------------------------------------------------------


def _recorded_image(root: str, state: AutoLamellaTaskState, role: str) -> Optional[str]:
    """The last existing image this run recorded under `role`, or None."""
    for relpath in reversed(state.outputs.get(role, [])):
        path = os.path.join(root, relpath)
        if os.path.isfile(path):
            return path
    return None


def _role_of(
    state: AutoLamellaTaskState, protocol_roles: Dict[str, str]
) -> Optional[str]:
    """Which overview role a history entry is: from what it recorded, else from
    the protocol's config for its task (a failed run recorded nothing)."""
    for role in OVERVIEW_ROLES:
        if role in state.outputs:
            return role
    return protocol_roles.get(state.name)


def _tiles_of(config) -> Optional[Tuple[int, int]]:
    settings = getattr(config, "settings", None)
    if settings is not None and hasattr(settings, "nrows"):
        return int(settings.nrows), int(settings.ncols)
    overview = getattr(config, "overview", None)
    if overview is not None and hasattr(overview, "rows"):
        return int(overview.rows), int(overview.cols)
    return None


def overview_entry(
    experiment: Experiment,
    grid: GridRecord,
    state: AutoLamellaTaskState,
    lamellae: Sequence[Lamella],
    protocol_roles: Dict[str, str],
) -> Optional[OverviewEntry]:
    """The report's row for one history entry, or None if it is not an overview."""
    role = _role_of(state, protocol_roles)
    if role is None:
        return None
    modality = _MODALITY[role]
    config = experiment.grid_protocol.task_config.get(state.name)
    entry = OverviewEntry(
        task_name=state.name,
        role=role,
        modality=modality,
        status=state.status,
        status_message=state.status_message,
        started=state.start_timestamp,
        ended=state.end_timestamp,
        tiles=_tiles_of(config),
        path=_recorded_image(str(experiment.grid_path(grid)), state, role),
    )
    if entry.path is None:
        entry.unmarked = [p.name for p in lamellae]
        return entry

    if is_fluorescence_image(entry.path):
        metadata = read_fm_metadata(entry.path)
        if metadata is not None:
            width, height = metadata.resolution
            entry.shape = (int(height), int(width))
            entry.pixel_size = float(metadata.pixel_size_x)
            entry.fov = (width * entry.pixel_size, height * entry.pixel_size)
            entry.pose = _pose_text(metadata.stage_position)
            entry.channels = [c.name for c in metadata.channels]
            projection = (
                FMStageProjection(metadata.geometry, entry.pixel_size, entry.shape)
                if metadata.geometry is not None
                else None
            )
            entry.marks, entry.unmarked = _mark(
                lamellae,
                modality,
                projection,
                metadata.stage_position,
                entry.pixel_size,
                entry.shape,
            )
        else:
            entry.unmarked = [p.name for p in lamellae]
        return entry

    metadata, shape = read_beam_metadata(entry.path)
    if metadata is None:
        entry.unmarked = [p.name for p in lamellae]
        return entry
    entry.shape = shape
    pixel = getattr(metadata.pixel_size, "x", None)
    entry.pixel_size = float(pixel) if pixel else None
    if entry.pixel_size and shape is not None:
        entry.fov = (shape[1] * entry.pixel_size, shape[0] * entry.pixel_size)
    beam = getattr(metadata.image_settings, "beam_type", None)
    if beam is BeamType.ION:
        entry.modality = "FIB"
    elif beam is BeamType.ELECTRON:
        entry.modality = "SEM"
    state_ = metadata.microscope_state
    base = getattr(state_, "stage_position", None) if state_ is not None else None
    entry.pose = _pose_text(base)
    projection = BeamStageProjection.from_image(SimpleNamespace(metadata=metadata))
    entry.marks, entry.unmarked = _mark(
        lamellae, entry.modality, projection, base, entry.pixel_size, shape
    )
    return entry


# ---------------------------------------------------------------------------
# The whole report
# ---------------------------------------------------------------------------


def _latest_load(grid: GridRecord) -> Optional[LoadEntry]:
    for state in reversed(grid.task_history):
        if state.name == LOAD_ENTRY_NAME:
            return LoadEntry(
                status=state.status,
                status_message=state.status_message,
                when=state.end_timestamp or state.start_timestamp,
            )
    return None


def _outcomes(grid: GridRecord, protocol: Sequence[str]) -> Dict[str, TaskOutcome]:
    outcomes = {name: TaskOutcome(status=None, runs=0) for name in protocol}
    for state in grid.task_history:
        if state.name in outcomes:
            outcomes[state.name] = TaskOutcome(
                status=state.status, runs=outcomes[state.name].runs + 1
            )
    return outcomes


def _microscope_of(sections: Sequence[GridSection]) -> Optional[str]:
    """Which instrument, from the first overview that says. The experiment
    record itself does not name one; its images do."""
    for section in sections:
        for entry in section.overviews:
            if entry.path is None or is_fluorescence_image(entry.path):
                continue
            metadata, _ = read_beam_metadata(entry.path)
            info = getattr(metadata, "system_info", None)
            if info is None:
                continue
            name = info.name or info.model or info.manufacturer
            serial = f" ({info.serial_number})" if info.serial_number else ""
            return f"{name}{serial}" if name else None
    return None


def collect_grid_report(
    experiment: Experiment,
    inventory: Optional[Iterable] = None,
) -> GridReport:
    """Everything the grid screening report shows, in the order it shows it.

    `inventory` is the stage's `grid_inventory()` rows, if a stage is connected
    when the report is written; a grid's slot is read from it by name. The record
    deliberately holds no slot of its own, so with no inventory the slot is None
    rather than stale.
    """
    slots: Dict[str, str] = {}
    for entry in inventory or []:
        if entry.name:
            slots[entry.name] = entry.slot_name

    protocol = list(experiment.grid_protocol.order)
    protocol_roles = {
        name: config.role
        for name, config in experiment.grid_protocol.task_config.items()
        if getattr(config, "role", None) in OVERVIEW_ROLES
    }

    sections: List[GridSection] = []
    outcomes: Dict[str, Dict[str, TaskOutcome]] = {}
    for grid in experiment.grids:
        lamellae = experiment.get_lamellae_for_grid(grid)
        overviews = [
            entry
            for state in grid.task_history
            for entry in [
                overview_entry(experiment, grid, state, lamellae, protocol_roles)
            ]
            if entry is not None
        ]
        sections.append(
            GridSection(
                name=grid.name,
                quality=grid.quality,
                description=grid.description,
                slot=slots.get(grid.name),
                load=_latest_load(grid),
                lamellae=[p.name for p in lamellae],
                overviews=overviews,
            )
        )
        outcomes[grid.name] = _outcomes(grid, protocol)

    return GridReport(
        experiment_name=experiment.name,
        experiment_path=str(experiment.path),
        created_at=experiment.created_at,
        generated_at=datetime.timestamp(datetime.now()),
        microscope=_microscope_of(sections),
        protocol=protocol,
        sections=sections,
        outcomes=outcomes,
    )
