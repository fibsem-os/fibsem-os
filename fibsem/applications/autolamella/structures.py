from __future__ import annotations

import logging
import os
import shutil
import uuid
from abc import ABC
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Tuple, Union

import pandas as pd
import petname
import yaml
from psygnal import evented
from psygnal.containers import EventedDict, EventedList

from fibsem.applications.autolamella import config as cfg
from fibsem.codec.upgrades import pre_task_protocol
from fibsem.codec.upgrades import protocol as upgrade
from fibsem.constants import TIME_DISPLAY_AMPM_SHORT
from fibsem.correlation.config import CorrelationConfig
from fibsem.structures import (
    DEFAULT_ALIGNMENT_AREA,
    Configurable,
    Display,
    FibsemImage,
    FibsemRectangle,
    FibsemStagePosition,
    MicroscopeState,
    Parameter,
    Point,
    SessionInfo,
    field_specs,
    param,
)
from fibsem.util.durations import format_duration_precise
from fibsem.util.timestamps import format_time, now
from fibsem.workflows.deciding import ProposalDecisions
from fibsem.workflows.experiment import BaseExperiment
from fibsem.workflows.grid.protocol import GridTaskProtocol
from fibsem.workflows.grids import ExperimentGrids, GridRecord
from fibsem.workflows.item import SiteItem
from fibsem.workflows.proposals import (
    Proposal,
    proposals_from_dict,
    proposals_to_dict,
)
from fibsem.workflows.quality import QualityRecord, Verdict
from fibsem.workflows.saved_format import check_app
from fibsem.workflows.site_task import SiteTaskConfig
from fibsem.workflows.status import TaskState
from fibsem.workflows.task_config import (
    RunPlan,
    TaskConfig,
    TaskProtocol,
    WorkflowOptions,
)

if TYPE_CHECKING:
    import numpy as np

    from fibsem.fm.structures import FluorescenceImage
    from fibsem.microscope import FibsemMicroscope


@dataclass
class AutoLamellaTaskConfig(SiteTaskConfig, ABC):
    """Configuration for AutoLamella tasks: a site task's, with the tasks the
    lamella editor draws beside it."""

    related_tasks: ClassVar[list[type["AutoLamellaTaskConfig"]]] = []


@dataclass
class AutoLamellaWorkflowConfig(RunPlan):
    """The plan a lamella run follows, with its name and description."""

    name: str = ""
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description, **super().to_dict()}

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "AutoLamellaWorkflowConfig":
        data = data or {}
        plan = RunPlan.from_dict(data)
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            tasks=plan.tasks,
        )


@evented
@dataclass
class LamellaDefaultConfig(Configurable):
    """Initial state applied to every new Lamella created from this protocol."""

    use_petname: bool = param(
        True,
        doc="Append a random two-word petname (e.g. 'brave-tiger') to each new "
        "lamella's name. Without it, names use the 'Lamella-01' format.",
        display=Display("Use petname"),
    )
    name_prefix: str = param(
        "",
        doc="Optional text put before every new lamella name, with a dash between "
        "(e.g. 'GridA' gives 'GridA-Lamella-01').",
        display=Display("Prefix"),
    )
    alignment_area: Optional[FibsemRectangle] = None
    poi: Optional[Point] = None

    def why_invalid(self) -> str:
        """Why this template can't be used, or "" when it can: the alignment
        area must lie inside the image, as fractions of it."""
        aa = self.alignment_area
        if aa is not None and not aa.is_valid_reduced_area:
            return (
                "The alignment area must lie inside the image: a width and "
                "height above 0, left and top from 0, and left + width and "
                "top + height at most 1."
            )
        return ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "use_petname": self.use_petname,
            "name_prefix": self.name_prefix,
        }
        if self.alignment_area is not None:
            d["alignment_area"] = self.alignment_area.to_dict()
        if self.poi is not None:
            d["poi"] = self.poi.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LamellaDefaultConfig":
        aa = data.get("alignment_area")
        poi = data.get("poi")
        return cls(
            use_petname=data.get("use_petname", True),
            name_prefix=data.get("name_prefix", ""),
            alignment_area=FibsemRectangle.from_dict(aa)
            if isinstance(aa, dict)
            else None,
            poi=Point.from_dict(poi) if isinstance(poi, dict) else None,
        )


@evented
@dataclass
class AutoLamellaTaskProtocol(TaskProtocol):
    """The lamella tasks, their settings and the plan a lamella run follows,
    with the run options, the lamella template, the correlation settings and
    the grid tasks. The plan may list fewer tasks than ``task_config`` has: a
    task it does not list is never asked about (``get_attention``)."""

    name: str = "AutoLamella Task Protocol"
    task_config: EventedDict[str, AutoLamellaTaskConfig] = field(
        default_factory=lambda: EventedDict()
    )
    workflow_config: AutoLamellaWorkflowConfig = field(
        default_factory=AutoLamellaWorkflowConfig
    )
    description: str = "Protocol for AutoLamella"
    version: str = "1.0"
    options: WorkflowOptions = field(default_factory=WorkflowOptions)
    lamella_defaults: LamellaDefaultConfig = field(default_factory=LamellaDefaultConfig)
    # Experiment-global correlation config (FIB-298): a user-step config, not an
    # automated task, so a peer field rather than an entry in task_config.
    correlation: CorrelationConfig = field(default_factory=CorrelationConfig)
    # Grid-level tasks (overviews, and later cleaning and deposition): their own
    # section, since they run on a GridRecord rather than a lamella.
    grid_tasks: GridTaskProtocol = field(default_factory=GridTaskProtocol)

    @classmethod
    def load_task_configs(
        cls, ddict: Optional[Dict[str, Dict[str, Any]]]
    ) -> EventedDict[str, AutoLamellaTaskConfig]:
        from fibsem.applications.autolamella.workflows.tasks import load_task_config

        return load_task_config(ddict or {})

    @classmethod
    def task_classes(cls) -> Dict[str, type]:
        from fibsem.applications.autolamella.workflows.tasks import get_tasks

        return get_tasks()

    def to_dict(self) -> Dict[str, Any]:
        saved = super().to_dict()
        return {
            "id": saved["id"],
            "name": saved["name"],
            "description": self.description,
            "version": self.version,
            "task_config": saved["task_config"],
            "workflow_config": self.workflow_config.to_dict(),
            "options": self.options.to_dict(),
            "lamella_defaults": self.lamella_defaults.to_dict(),
            "correlation": self.correlation.to_dict(),
            "grid_tasks": self.grid_tasks.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AutoLamellaTaskProtocol":
        data = upgrade.autolamella_protocol(data)
        protocol = cls(
            name=data.get("name", "AutoLamella Task Protocol"),
            description=data.get("description", "Protocol for AutoLamella"),
            version=data.get("version", "1.0"),
            task_config=cls.load_task_configs(data.get("task_config")),
            workflow_config=AutoLamellaWorkflowConfig.from_dict(
                data.get("workflow_config")
            ),
            options=WorkflowOptions.from_dict(data.get("options", {})),
            lamella_defaults=LamellaDefaultConfig.from_dict(
                data.get("lamella_defaults", {})
            ),
            # Missing on protocols saved before this field -> a default config.
            correlation=CorrelationConfig.from_dict(data.get("correlation")),
            # Likewise: a protocol written before grid tasks existed offers none.
            grid_tasks=GridTaskProtocol.from_dict(data.get("grid_tasks")),
        )
        if data.get("id"):
            protocol.id = data["id"]
        return protocol

    @classmethod
    def load(cls, filename: str) -> "AutoLamellaTaskProtocol":
        """Read the protocol file at *filename*. One from before task protocols
        is saved back as the task protocol it becomes (`pre_task_protocol`),
        after a copy of it is kept as ``<file>.before-v1``; where it can't be
        written, it is converted each time it opens."""
        with open(filename) as file:
            data = yaml.safe_load(file)
        protocol = cls.from_dict(data)
        if pre_task_protocol.is_pre_task(data):
            _save_converted_protocol(filename, protocol)
        return protocol


def _save_converted_protocol(path: str, protocol: AutoLamellaTaskProtocol) -> None:
    """Save *protocol*, converted from the pre-task protocol at *path*, over it,
    once a copy of the file as it was is kept."""
    backup = Path(f"{path}.before-v1")
    try:
        if not backup.exists():
            shutil.copy2(path, backup)
        protocol.save(path)
    except OSError as e:
        logging.warning(
            f"Could not save {path} as a task protocol ({e}); it is converted "
            "each time it opens."
        )
        return
    logging.info(
        f"Saved {path} as a task protocol; the file as it was is kept as {backup}."
    )


def _make_thumbnail_placeholder():
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (256, 170), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    text = "No Data"
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=20)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((256 - tw) // 2, (170 - th) // 2), text, fill=(100, 100, 100), font=font)
    return np.asarray(img)


_THUMBNAIL_PLACEHOLDER = None


@evented
@dataclass
class Lamella(SiteItem):
    item_type: ClassVar[str] = "lamella"
    path: Path
    number: int  # TODO: deprecate, use petname instead
    petname: str
    alignment_area: FibsemRectangle = field(
        default_factory=lambda: FibsemRectangle.from_dict(DEFAULT_ALIGNMENT_AREA)
    )
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_config: EventedDict[str, "AutoLamellaTaskConfig"] = field(
        default_factory=lambda: EventedDict()
    )
    poses: Dict[str, MicroscopeState] = field(default_factory=dict)
    task_state: TaskState = field(default_factory=TaskState)
    task_history: List["TaskState"] = field(default_factory=list)
    defect: QualityRecord = field(default_factory=QualityRecord)
    milling_angle: Optional[float] = None
    poi: Point = field(
        default_factory=lambda: Point(0.0, 0.0)
    )  # point of interest within lamella area (milling coordinate system)
    description: str = ""  # free-text note about the lamella
    # The grid this lamella sits on: GridRecord.id, or None when the experiment
    # does not track grids (every experiment before grid records existed). A
    # back-reference only; grid -> lamella is derived by filtering on it.
    grid_id: Optional[str] = None
    # What was proposed for this lamella and what was decided, oldest first,
    # each saying who asked it: every proposal a task made (a question asked
    # mid-run, then the run's result; a re-run after them). A task's last one
    # is its current one; ``proposal()`` returns it. A current
    # proposal with no decisions is pending; the consumer that requires that
    # task is deferred until one is appended. See proposals.py and
    # Experiment.decide.
    proposals: List[Proposal] = field(default_factory=list)

    def __post_init__(self):
        # Deliberately does not create ``path``. Constructing a Lamella is not a
        # request to write to disk, and on the load path this runs from
        # ``from_dict`` with the as-created path out of the yaml -- before
        # ``Experiment.load`` calls ``relocate``. Creating the directory here
        # therefore wrote into whatever machine the experiment came from, at a
        # path the caller never named. Directories are created where a lamella is
        # actually created (``Experiment.add_new_lamella``) and where files are
        # actually written (``FibsemImage.save``, ``save_thumbnail``). See FIB-420.
        if self.id is None:
            self.id = str(uuid.uuid4())
        self.task_state.item_id = self.id

        self._sync_imaging_paths()

    def _sync_imaging_paths(self) -> None:
        """Point every milling task's acquisition at this lamella's directory.

        Milling configs carry their own imaging path, so it has to be re-derived
        whenever ``path`` changes or acquisitions are written to the old location.
        """
        for tc in self.task_config.values():
            for milling_task_config in tc.milling.values():
                milling_task_config.acquisition.imaging.path = self.path

    def relocate(self, experiment_path: Path) -> None:
        """Re-point this lamella at ``experiment_path``.

        ``path`` is persisted in experiment.yaml as the absolute path the lamella
        was *created* at, so a lamella loaded from a moved or copied experiment
        still points at the original machine. Worse, when that directory happens
        to still exist locally, reads silently succeed against the wrong data.
        ``Experiment.load`` calls this so paths follow the experiment. See FIB-367.
        """
        self.path = os.path.join(experiment_path, self.name)
        self._sync_imaging_paths()

    @property
    def name(self) -> str:
        return self.petname

    @name.setter
    def name(self, value: str):
        self.petname = value

    @property
    def is_failure(self) -> bool:
        """Whether a human has judged this lamella defective.

        Deliberately not set by a failing task, and it should stay that way. A
        task failing is a fact about one attempt; a defect is a judgement about
        the lamella. TaskManager._should_skip skips a lamella marked failed for
        *every* remaining task, so auto-setting this on a stage timeout or a
        momentary comms drop would permanently abandon a lamella that only
        needed retrying.

        Whether a failed task blocks dependent work is a separate question,
        answered by completed_tasks -- which filters on task status, so a failed
        prerequisite does not license the task that requires it. See FIB-490.
        """
        return self.defect.verdict is Verdict.FAILED

    def trench_window(self) -> Optional[Tuple[float, float]]:
        """The window to measure the lamella's band in, from its own trench
        patterns (those that follow its POI): along the lamella, nine tenths of
        the narrowest pattern, so the trench ends stay out; across it, the
        widest gap between trenches (the rough slab) and half a micron into
        each trench. None when it has no trench pattern."""
        widths, gaps = [], []
        for config in self.task_config.values():
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

    @property
    def quality(self) -> QualityRecord:
        """The same record as ``defect`` under the name ``GridRecord`` uses, so
        code that judges an item need not know which kind it has. The field is
        still ``defect`` because its evented signal is what the lamella widgets
        subscribe to; renaming it is a separate, mechanical change."""
        return self.defect

    @quality.setter
    def quality(self, value: QualityRecord) -> None:
        self.defect = value

    @property
    def stage_position(self) -> FibsemStagePosition:
        return self.milling_pose.stage_position  # type: ignore

    @stage_position.setter
    def stage_position(self, value: FibsemStagePosition):
        self.milling_pose.stage_position = value

    def finishing_task(self, task_name: str, config: Any, image: Any) -> None:
        """Keep the config the task ran with, and its last FIB image as the
        thumbnail. A thumbnail that cannot be saved is logged: a missing picture
        is worth far less than an accurate task state."""
        self.task_config[task_name] = deepcopy(config)
        if image is None:
            return
        try:
            self.save_thumbnail(image)
        except Exception as e:
            logging.warning(f"Could not save the thumbnail for {self.name}: {e}")

    @property
    def fluorescence_pose(self) -> Optional[MicroscopeState]:
        return self.poses.get("FLUORESCENCE", None)

    @fluorescence_pose.setter
    def fluorescence_pose(self, value: MicroscopeState):
        """Set the fluorescence pose for the lamella."""
        if not isinstance(value, MicroscopeState):
            raise TypeError("Fluorescence pose must be a MicroscopeState instance.")
        self.poses["FLUORESCENCE"] = value

    def related_milling_stages(self, task_name: str) -> list:
        """The enabled milling stages of this lamella's tasks related to
        *task_name*'s (its config's ``related_tasks``): what the lamella editor
        and the agent's milling picture draw behind that task's own."""
        config = self.task_config.get(task_name)
        related = tuple(getattr(config, "related_tasks", None) or ())
        stages = []
        if not related:
            return stages
        for other in self.task_config.values():
            if isinstance(other, related) and other.milling:
                for milling in other.milling.values():
                    stages.extend(milling.enabled_stages)
        return stages

    def update_milling_angle(self, microscope: "FibsemMicroscope") -> None:
        """Recompute milling_angle from the milling-pose stage tilt.

        The milling angle is derived from the stage tilt (plus the microscope pretilt /
        column-tilt configuration), so it is kept consistent with the stored milling pose.
        Leaves the existing value unchanged if the stage tilt/rotation is unavailable.
        """
        if (
            microscope is None
            or self.milling_pose is None
            or self.milling_pose.stage_position is None
        ):
            return
        try:
            self.milling_angle = microscope.get_current_milling_angle(
                stage_position=self.milling_pose.stage_position
            )
        except ValueError:
            logging.debug(
                f"Could not compute milling angle for {self.name}: stage tilt/rotation unavailable"
            )

    def to_dict(self):
        return {
            "petname": self.petname,
            "path": str(self.path),
            "alignment_area": self.alignment_area.to_dict(),
            "number": self.number,
            "id": str(self.id),
            "poses": {k: v.to_dict() for k, v in self.poses.items()},
            "task_config": {k: v.to_dict() for k, v in self.task_config.items()},
            "task_state": self.task_state.to_dict(),
            "task_history": [task.to_dict() for task in self.task_history],
            "defect": self.defect.to_dict(),
            "milling_angle": self.milling_angle,
            "poi": self.poi.to_dict(),
            "description": self.description,
            "grid_id": self.grid_id,
            "proposals": proposals_to_dict(self.proposals),
        }

    @property
    def info(self) -> str:
        return self.status_info

    @property
    def status_info(self) -> str:
        return f"Lamella {self.petname} [{self.task_state.name}]"

    @classmethod
    def from_dict(cls, data: dict) -> "Lamella":
        # backwards compatibility
        alignment_area_ddict = data.get("alignment_area", DEFAULT_ALIGNMENT_AREA)
        alignment_area = FibsemRectangle.from_dict(alignment_area_ddict)

        from fibsem.applications.autolamella.workflows.tasks import load_task_config

        poses = {
            k: MicroscopeState.from_dict(v) for k, v in data.get("poses", {}).items()
        }
        # backwards compat: migrate legacy top-level objective_position into fluorescence_pose
        legacy_obj_pos = data.get("objective_position", None)
        if legacy_obj_pos is not None and "FLUORESCENCE" in poses:
            if poses["FLUORESCENCE"].objective_position is None:
                poses["FLUORESCENCE"].objective_position = legacy_obj_pos

        return cls(
            petname=data["petname"],
            path=data["path"],
            alignment_area=alignment_area,
            number=data.get("number", data.get("number", 0)),
            id=data.get("id", ""),
            poses=poses,
            task_config=load_task_config(data.get("task_config", {})),
            task_state=TaskState.from_dict(data.get("task_state", {})),
            task_history=[
                TaskState.from_dict(task) for task in data.get("task_history", [])
            ],
            defect=QualityRecord.from_dict(data.get("defect", {})),
            milling_angle=data.get("milling_angle", None),
            poi=Point.from_dict(data.get("poi", {"x": 0, "y": 0})),
            description=data.get("description", ""),
            grid_id=data.get("grid_id"),
            proposals=proposals_from_dict(data.get("proposals")),
        )

    def get_thumbnail(self) -> "np.ndarray":
        """Load the thumbnail image for this lamella if available.

        Returns:
            np.ndarray (H, W, 3) RGB, or a blank array if no thumbnail exists.
        """
        global _THUMBNAIL_PLACEHOLDER
        thumb_path = os.path.join(self.path, "thumbnail.png")
        import numpy as np
        from PIL import Image

        if not os.path.exists(thumb_path):
            if _THUMBNAIL_PLACEHOLDER is None:
                _THUMBNAIL_PLACEHOLDER = _make_thumbnail_placeholder()
            return _THUMBNAIL_PLACEHOLDER
        try:
            return np.asarray(Image.open(thumb_path).convert("RGB"))
        except OSError as e:
            # A thumbnail that will not open is not a reason to lose the card. `save_thumbnail`
            # writes atomically now, so a torn file should no longer be produced -- but one
            # written before that fix is still on disk, and any other corruption lands here
            # too. This is reached from a Qt slot, where an escaping exception aborts the
            # process under PyQt5 rather than raising (FIB-329).
            logging.warning(f"Could not read the thumbnail at {thumb_path}: {e}")
            if _THUMBNAIL_PLACEHOLDER is None:
                _THUMBNAIL_PLACEHOLDER = _make_thumbnail_placeholder()
            return _THUMBNAIL_PLACEHOLDER

    def save_thumbnail(self, image: Union["FibsemImage", "FluorescenceImage"]) -> None:
        """Save a thumbnail of the given image to disk as thumbnail.png.

        Written to a temporary file and moved into place, so a reader never sees a
        partly-written one. `get_thumbnail` runs from the lamella cards on the GUI
        thread while saving happens on a worker (FIB-563), so the two do overlap: the
        reported failure was `OSError: image file is truncated` out of `Image.open`,
        with the existence check passing because the file was there, just incomplete.

        `os.replace` is atomic on POSIX and on Windows, provided the temporary file is
        on the same filesystem -- hence writing it in the same directory rather than
        somewhere under /tmp. A reader sees either the previous thumbnail or the new
        one, never a partial.

        Written at display size rather than at the acquired resolution -- see
        `fibsem.imaging.thumbnail.THUMBNAIL_MAX_EDGE`. This is a thumbnail, not a copy of the frame; the frame
        itself is saved separately by the task that acquired it. A fluorescence
        stack's thumbnail is its projection, as the FM canvas draws it.
        """
        from fibsem.fm.preview import composite_projection
        from fibsem.fm.structures import FluorescenceImage
        from fibsem.imaging.thumbnail import write_thumbnail

        data = (
            composite_projection(image)
            if isinstance(image, FluorescenceImage)
            else image.filtered_data
        )
        write_thumbnail(data, os.path.join(self.path, "thumbnail.png"))

    def sync_tasks_to_poi(self, point: Optional[Point] = None) -> list[str]:
        """Sync the milling patterns to point of interest"""
        synced_tasks, moves = self.poi_sync_plan(self.poi if point is None else point)
        for pattern, moved in moves:
            pattern.point = moved
        return synced_tasks

    def poi_sync_plan(self, point: Point) -> Tuple[List[str], List[Tuple[Any, Point]]]:
        """What syncing the patterns to ``point`` would do, without doing it:
        the task names that follow the point, and each pattern with the point
        it would move to. Computed in full first, so a caller can refuse
        before anything moves (a decision is all or nothing)."""
        synced_tasks: List[str] = []
        moves: List[Tuple[Any, Point]] = []
        for task_name, task_config in self.task_config.items():
            # check if task has sync_to_poi enabled
            if not getattr(task_config, "sync_to_poi", False):
                continue
            if not task_config.milling:
                continue

            for milling_config in task_config.milling.values():
                if not milling_config.stages:
                    continue
                # calculate offset from the first stage's pattern point
                diff = point - milling_config.stages[0].pattern.point
                for stage in milling_config.stages:
                    moves.append((stage.pattern, stage.pattern.point + diff))

            synced_tasks.append(task_name)
        return synced_tasks, moves


def find_overviews(root: Union[str, Path]) -> List[Path]:
    """The stitched beam overviews an experiment keeps, oldest first.

    Grid overviews live under ``grids/<grid>/<task>/``; older experiments keep
    ``overview-image-*.tif`` at the root. Fluorescence overviews
    (``*.ome.tiff``) are not included.

    Oldest first by modification time, then by name so that files copied with
    the same time still come out in a fixed order. The last is the most recent.
    """
    root = Path(root)
    found = list(root.glob("overview*.tif")) + list(
        root.glob("grids/*/*/overview*.tif")
    )
    return sorted(found, key=lambda p: (p.stat().st_mtime, p.name))


@evented
@dataclass
class Experiment(ExperimentGrids, ProposalDecisions, BaseExperiment):
    app: ClassVar[str] = "autolamella"

    name: str
    id: str
    path: Path
    positions: EventedList[Lamella] = field(default_factory=EventedList)
    grids: EventedList[GridRecord] = field(default_factory=EventedList)
    # Aware, written as ISO 8601 with its offset (FIB-1197); older files hold a
    # POSIX float, which reads as this machine's zone.
    created_at: Optional[datetime] = field(default_factory=now)
    task_protocol: "AutoLamellaTaskProtocol" = field(
        default_factory=lambda: AutoLamellaTaskProtocol()
    )
    metadata: Dict[str, Any] = field(default_factory=dict)
    session: Optional[SessionInfo] = None

    def __init__(
        self,
        path: Path,
        name: str = cfg.EXPERIMENT_NAME,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create a new experiment.

        Args:
            path: The path where the experiment will be created.
            name: The name of the experiment. Defaults to cfg.EXPERIMENT_NAME.
            metadata: Optional dictionary containing experiment metadata (e.g., description, user, project, organisation).
        """
        super().__init__(path=path, name=name, metadata=metadata)
        self.positions: EventedList[Lamella] = EventedList()
        self.grids: EventedList[GridRecord] = EventedList()

        self.task_protocol: AutoLamellaTaskProtocol = None  # must be set externally

    @property
    def turns_beams_off(self) -> bool:
        protocol = self.task_protocol
        return protocol is not None and protocol.options.turn_beams_off

    def to_dict(self, include_protocol: bool = False) -> dict:

        state_dict = {
            **super().to_dict(),
            "positions": [deepcopy(lamella.to_dict()) for lamella in self.positions],
            "grids": [grid.to_dict() for grid in self.grids],
        }

        if include_protocol:
            state_dict["protocol"] = (
                self.task_protocol.to_dict() if self.task_protocol is not None else None
            )

        return state_dict

    @classmethod
    def from_dict(cls, ddict: dict) -> "Experiment":
        ddict = upgrade.experiment(ddict)
        path = os.path.dirname(ddict["path"])
        name = ddict["name"]
        experiment = Experiment(path=path, name=name)
        experiment._read_identity(ddict)

        # load lamella from dict
        for lamella_dict in ddict["positions"]:
            lamella = Lamella.from_dict(data=lamella_dict)
            experiment.positions.append(lamella)

        # Experiments written before grid records existed have no "grids" key and
        # load with none; nothing else about them changes.
        for grid_dict in ddict.get("grids", []):
            experiment.grids.append(GridRecord.from_dict(grid_dict))

        return experiment

    def items(self) -> List[Union["Lamella", GridRecord]]:
        """The lamellae and grids: the items a task runs on and a proposal can
        sit on."""
        return list(self.positions) + list(self.grids or [])

    def sign_verdict(self, item: Union["Lamella", GridRecord]) -> None:
        """Name the operator on *item*'s verdict, when whatever set it could not.

        A verdict set from a lamella's menu is written by a widget that has no
        experiment to ask who is at the instrument; the code that saves the change
        does, and calls this first. A verdict that already names someone -- an agent,
        a review decision -- is left as it is, and so is one nobody has given.
        """
        record = getattr(item, "defect", None) or getattr(item, "quality", None)
        if record is None or record.actor or record.verdict is Verdict.UNASSESSED:
            return
        record.actor = str(self.actor())

    def save_verdict(self, item: Union["Lamella", GridRecord]) -> None:
        """Save *item*'s new verdict, signed (`sign_verdict`): what every
        window that judges a lamella or a grid, and the App Server, call."""
        self.sign_verdict(item)
        self.save()

    def get_lamella_by_name(self, name: str) -> Optional["Lamella"]:
        """Return the Lamella with the given name, or None if not found."""
        return next((p for p in self.positions if p.name == name), None)

    # -- grids ---------------------------------------------------------------

    @property
    def grid_protocol(self) -> GridTaskProtocol:
        """The grid tasks this experiment can run: the protocol's `grid_tasks`.

        One store, not two: the section of the assigned task protocol, which is
        what `protocol.yaml` saves. The app assigns a protocol when it creates or
        loads an experiment; anything else has to do the same before asking.
        """
        if self.task_protocol is None:
            raise ValueError(
                "No task protocol is assigned to this experiment, so it has no grid "
                "protocol. Set `experiment.task_protocol` first."
            )
        return self.task_protocol.grid_tasks

    def write_grid_report(self, inventory: Any) -> Path:
        """Write the grid screening report, the file Tools -> Reporting writes:
        what a grid run's end-of-run report calls. ``inventory`` is the stage's
        own record, for the slot column."""
        from fibsem.applications.autolamella.tools.grid_report_pdf import (
            generate_grid_report,
        )

        return Path(generate_grid_report(self, inventory=inventory))

    def _release_grid(self, grid: GridRecord) -> None:
        """A grid no longer tracked: its lamellae are orphaned, not deleted."""
        for lamella in self.get_lamellae_for_grid(grid):
            lamella.grid_id = None

    def get_lamellae_for_grid(self, grid: GridRecord) -> List["Lamella"]:
        """Derived from `Lamella.grid_id`; nothing stores the reverse."""
        return [p for p in self.positions if p.grid_id == grid.id]

    def get_grid_for_lamella(self, lamella: "Lamella") -> Optional[GridRecord]:
        return self.get_grid_by_id(lamella.grid_id)

    def __repr__(self) -> str:

        return f"""Experiment: 
        Path: {self.path}
        Positions: {len(self.positions)}
        """

    @staticmethod
    def load(fname: Path) -> "Experiment":
        """Load an experiment from disk.

        Automatically attempts to load the task_protocol from protocol.yaml
        in the same directory if it exists.
        """

        # read and open existing yaml file
        path = Path(fname).with_suffix(".yaml")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No file with name {path} found.")
        with open(path) as f:
            ddict = yaml.safe_load(f)
        check_app(ddict, Experiment.app, path)

        # create experiment from dict
        experiment = Experiment.from_dict(ddict)
        experiment.path = os.path.dirname(fname)
        # a question the task was waiting on when the app closed: nothing is
        # waiting on it now
        experiment.withdraw_what_was_asked("the app closed before it was answered")

        # lamella paths are stored as-created, so re-point them at wherever the
        # experiment actually is now. otherwise a moved or copied experiment
        # reads and writes against the original location. (FIB-367)
        for lamella in experiment.positions:
            lamella.relocate(experiment.path)

        # NOTE: deliberately does not configure logging. configure_logging calls
        # basicConfig(force=True), which closes and replaces every root handler --
        # so reading an experiment would reach into the calling process's global
        # logging and redirect its output into this experiment's logfile. Callers
        # that want that ask for it: the app does so when it adopts an experiment.
        # See FIB-421.

        # attempt to load task protocol from the same directory
        protocol_path = os.path.join(experiment.path, "protocol.yaml")
        if os.path.exists(protocol_path):
            try:
                experiment.task_protocol = AutoLamellaTaskProtocol.load(protocol_path)
                logging.info(f"Loaded task protocol from {protocol_path}")
            except Exception as e:
                logging.warning(
                    f"Failed to load task protocol from {protocol_path}: {e}"
                )

        return experiment

    def apply_lamella_config(
        self,
        lamella_names: List[str],
        task_names: List[str],
        source_lamella_name: Optional[str] = None,
        update_base_protocol: bool = False,
    ) -> int:
        """Apply task configurations to lamella, preserving existing milling pattern positions.

        If source_lamella_name is provided, copies from that lamella's config.
        If None, copies from the base protocol.

        Args:
            lamella_names: Names of the target lamella to apply configurations to.
            task_names: The task names to apply.
            source_lamella_name: Name of the source lamella. If None, uses the base protocol.
            update_base_protocol: Whether to also update the base protocol.

        Returns:
            The number of lamella updated.
        """
        # Resolve the source task config
        if source_lamella_name is not None:
            source_lamella = next(
                (p for p in self.positions if p.name == source_lamella_name), None
            )
            if source_lamella is None:
                logging.warning(f"Source lamella '{source_lamella_name}' not found.")
                return 0
            source_task_config = source_lamella.task_config
            source_display_name = source_lamella_name
        else:
            if self.task_protocol is None:
                logging.warning("No base protocol available.")
                return 0
            source_task_config = self.task_protocol.task_config
            source_display_name = "base protocol"

        target_names = set(lamella_names)
        updated_count = 0
        for lamella in self.positions:
            if lamella.name not in target_names:
                continue

            for task_name in task_names:
                source_config = source_task_config.get(task_name)
                if source_config is None:
                    continue

                new_config = deepcopy(source_config)
                existing_config = lamella.task_config.get(task_name)

                # Preserve existing milling pattern positions
                if existing_config is not None and new_config.milling:
                    for milling_name, new_milling_config in new_config.milling.items():
                        existing_milling_config = existing_config.milling.get(
                            milling_name
                        )
                        if existing_milling_config is None:
                            continue

                        existing_stage_lookup = {
                            (stage.num, stage.name): stage
                            for stage in existing_milling_config.stages
                        }

                        for new_stage in new_milling_config.stages:
                            existing_stage = existing_stage_lookup.get(
                                (new_stage.num, new_stage.name)
                            )
                            if existing_stage is None:
                                continue

                            if type(existing_stage.pattern) is type(
                                new_stage.pattern
                            ) and hasattr(existing_stage.pattern, "point"):
                                new_stage.pattern.point = deepcopy(
                                    existing_stage.pattern.point
                                )

                lamella.task_config[task_name] = new_config

            # The copies still point their images at the source lamella's
            # folder, or at none from the protocol.
            lamella._sync_imaging_paths()
            updated_count += 1
            logging.info(
                f"Applied config from '{source_display_name}' to '{lamella.name}' "
                f"for tasks: {task_names}"
            )

        # Update base protocol if requested (skip if source is already the base protocol)
        if (
            update_base_protocol
            and self.task_protocol is not None
            and source_lamella_name is not None
        ):
            for task_name in task_names:
                if task_name in source_task_config:
                    self._set_protocol_config(source_task_config[task_name])
            logging.info(f"Updated base protocol tasks: {task_names}")

        return updated_count

    def _set_protocol_config(self, config: AutoLamellaTaskConfig) -> None:
        """*config*'s settings into the protocol."""
        self.task_protocol.task_config[config.task_name] = deepcopy(config)

    def add_task(self, task_type: str, task_name: str) -> AutoLamellaTaskConfig:
        """Add a new task of a registered *task_type*, named *task_name*: its
        default config in the protocol, at the end of the workflow (required,
        supervised), and a copy on every lamella. The protocol editor's Add
        Task, and an agent's. Raises ValueError for an unknown type or a name
        the protocol already has."""
        config = self.task_protocol.add_new(task_type, task_name)
        for lamella in self.positions:
            lamella.task_config[config.task_name] = deepcopy(config)
            lamella._sync_imaging_paths()
        logging.info(
            f"Added task {config.task_name} ({task_type}) to experiment {self.name}"
        )
        return config

    def remove_task(self, task_name: str) -> None:
        """Remove a task from the experiment: from the protocol, from the
        workflow, and from every lamella's config (FIB-1109).

        All three together, so the workflow never lists a task the protocol no
        longer has, and no lamella runs a stale copy of one. A task that listed
        it as a requirement no longer does. Task history is left alone: it is
        the record of what ran.
        """
        self.task_protocol.remove(task_name)
        for lamella in self.positions:
            lamella.task_config.pop(task_name, None)

        logging.info(f"Removed task {task_name} from experiment {self.name}")

    def at_failure(self) -> List[Lamella]:
        """Return a list of lamellas that have failed"""
        return [lamella for lamella in self.positions if lamella.is_failure]

    def get_milling_positions(self) -> List[FibsemStagePosition]:
        """Get the milling stage positions for all lamellas in the experiment"""
        positions = []
        for p in self.positions:
            pstate = p.milling_pose
            if pstate is None or pstate.stage_position is None:
                continue
            pos = pstate.stage_position
            pos.name = p.name
            positions.append(pos)
        return positions

    def add_lamella(self, lamella: Lamella) -> None:
        """Add a lamella to the experiment."""
        if not isinstance(lamella, Lamella):
            raise TypeError("lamella must be an instance of Lamella")

        # check if lamella already exists
        if lamella in self.positions:
            raise ValueError(
                f"Lamella {lamella.name} already exists in the experiment."
            )

        self.positions.append(deepcopy(lamella))
        logging.info(f"Added lamella {lamella.name} to experiment {self.name}")

    @classmethod
    def create(
        cls,
        path: Path,
        name: str = cfg.EXPERIMENT_NAME,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Experiment":
        """Create a new experiment with the given path and name. Also configures logging.

        Args:
            path: The path where the experiment will be created.
            name: The name of the experiment. Defaults to cfg.EXPERIMENT_NAME.
            metadata: Optional dictionary containing experiment metadata (e.g., description, user, project, organisation).

        Returns:
            Experiment: The created experiment instance.
        """
        # create the experiment
        experiment = Experiment(path=path, name=name, metadata=metadata)

        os.makedirs(experiment.path, exist_ok=True)

        # NOTE: as with load(), creating an experiment does not reconfigure the
        # calling process's logging -- the app does that when it adopts one.
        # See FIB-421.

        # save the experiment
        experiment.save()

        logging.info(f"Created new experiment {experiment.name} at {experiment.path}")

        return experiment

    def save_protocol(self) -> None:
        """Save the task protocol to disk in the experiment directory."""
        self.task_protocol.save(os.path.join(self.path, "protocol.yaml"))

    ###### TASK REFACTORING ##########

    def add_new_lamella(
        self,
        microscope_state: MicroscopeState,
        task_config: EventedDict[str, AutoLamellaTaskConfig],
        name: Optional[str] = None,
        fluorescence_pose: Optional[MicroscopeState] = None,
        grid_id: Optional[str] = None,
    ) -> None:
        """Create a new lamella and add it to the experiment.

        Args:
            fluorescence_pose: where the lamella is under the objective, if the caller
                worked one out. Passed in rather than assigned by the caller afterwards
                because `add_lamella` publishes `positions.events.inserted`, and
                everything drawing the experiment redraws on it -- a pose attached a
                moment later arrives after every listener has already decided this
                lamella has none. That is not hypothetical: it left each newly marked
                lamella missing from the FM overview until something else forced a
                refresh.
            grid_id: the `GridRecord.id` of the grid this lamella is on, when the
                caller knows it. On the constructor for the same reason as the
                pose: anything grouping lamellae by grid reads it on `inserted`.
                None is "no grid known", which every lamella made before grids
                existed also carries.
        """
        template = self.task_protocol.lamella_defaults
        number = max((pos.number for pos in self.positions), default=0) + 1
        if name is None:
            sep = "-" if template.name_prefix else ""
            if template.use_petname:
                name = f"{template.name_prefix}{sep}{number:02d}-{petname.generate(2)}"
            else:
                name = f"{template.name_prefix}{sep}Lamella-{number:02d}"
        path = Path(os.path.join(self.path, name))

        # create the lamella
        lamella = Lamella(
            petname=name,
            path=path,
            number=number,
            task_config=deepcopy(task_config),
            grid_id=grid_id,
        )
        if template.alignment_area is not None:
            lamella.alignment_area = deepcopy(template.alignment_area)
        if template.poi is not None:
            lamella.poi = deepcopy(template.poi)
        lamella.milling_pose = microscope_state
        if fluorescence_pose is not None:
            lamella.fluorescence_pose = fluorescence_pose

        # create the lamella directory
        os.makedirs(lamella.path, exist_ok=True)

        logging.info(f"Created new lamella {lamella.name} at {lamella.path}")

        self.add_lamella(lamella)

    def task_history_dataframe(self) -> pd.DataFrame:
        """Create a dataframe with the history of all tasks."""
        history: List[dict[Any, Any]] = []
        for pos in self.positions:
            name = pos.name

            for task in pos.task_history:
                ddict = {
                    "lamella_name": name,
                    "lamella_id": task.item_id,
                    "task_name": task.name,
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "task_status": task.status.name,
                    "task_status_message": task.status_message,
                    "start_timestamp": task.start_timestamp,
                    "end_timestamp": task.end_timestamp,
                    "completed_at": task.completed_at,
                    "duration": task.duration,
                }
                history.append(deepcopy(ddict))

        df_task_history = pd.DataFrame(history)
        return df_task_history

    def experiment_summary_dataframe(self) -> pd.DataFrame:
        """Create a summary dataframe of the experiment."""
        edict = []
        for p in self.positions:
            ddict = {
                "experiment_name": self.name,
                "experiment_path": self.path,
                "experiment_created_at": self.created_at,
                "experiment_id": self.id,
                "lamella_name": p.name,
                "lamella_id": p.id,
                "last_completed": p.last_completed_task.completed
                if p.last_completed_task
                else None,
                "last_completed_task": p.last_completed_task.name
                if p.last_completed_task
                else None,
                "last_completed_at": p.last_completed_task.completed_at
                if p.last_completed_task
                else None,
                "is_completed": self.task_protocol.workflow_config.is_completed(p),
                "is_failure": p.is_failure,
                "milling_angle": p.milling_angle,
            }
            edict.append(deepcopy(ddict))

        df = pd.DataFrame(edict)

        return df

    def workflow_dataframe(self) -> pd.DataFrame:
        """Create a dataframe with the workflow"""
        wlist: List[Dict] = []
        protocol = self.task_protocol
        for i, t in enumerate(protocol.workflow_config.tasks, 1):
            ddict = {
                "order": i,
                "task_name": t.name,
                "required": t.required,
                "attention": protocol.get_attention(t.name).value,
            }
            wlist.append(deepcopy(ddict))

        return pd.DataFrame(wlist)
