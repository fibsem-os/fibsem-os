from __future__ import annotations

import logging
import os
import threading
import uuid
from abc import ABC
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import pandas as pd
import petname
import yaml
from psygnal import Signal, evented
from psygnal.containers import EventedDict, EventedList

from fibsem import timing
from fibsem.applications.autolamella import config as cfg
from fibsem.applications.autolamella.proposals import (
    Decision,
    DecisionOutcome,
    DecisionResult,
    Proposal,
    has_value_writer,
    human_author,
    known_value_names,
    proposals_from_dict,
    proposals_to_dict,
    write_value,
)
from fibsem.applications.autolamella.protocol.constants import (
    FIDUCIAL_KEY,
    MICROEXPANSION_KEY,
    MILL_POLISHING_KEY,
    MILL_ROUGH_KEY,
    NOTCH_KEY,
    STRESS_RELIEF_KEY,
    TRENCH_KEY,
    UNDERCUT_KEY,
)
from fibsem.constants import TIME_DISPLAY_AMPM_SHORT
from fibsem.correlation.config import CorrelationConfig
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.structures import (
    DEFAULT_ALIGNMENT_AREA,
    FibsemImage,
    FibsemRectangle,
    FibsemStagePosition,
    FibsemUser,
    ImageSettings,
    MicroscopeState,
    Point,
    ReferenceImageParameters,
    SessionInfo,
    get_fields_with_metadata,
)
from fibsem.utils import configure_logging as _configure_logging
from fibsem.utils import format_duration

if TYPE_CHECKING:
    from fibsem.microscope import FibsemMicroscope


class AutoLamellaTaskStatus(Enum):
    NotStarted = auto()
    InProgress = auto()
    Completed = auto()
    Failed = auto()
    Skipped = auto()
    Cancelled = auto()  # aborted by the user (Stop), distinct from a genuine Failure
    Removed = auto()  # pulled from the queue by the user before it ran


# AutoLamellaUser lived here: a richer user identity (role, preferences, is_default)
# with a to_fibsem_user() bridge into image metadata. Nothing ever constructed one --
# not the app, not a script, not a test -- so the bridge was never called and the
# extra fields never had a reader. Removed rather than left dormant, because half a
# user model is worse than either answer. fibsem.structures.FibsemUser is the only
# user model now; see FIB-450 and the ownership rule on FibsemImageMetadata.


@evented
@dataclass
class AutoLamellaTaskState:
    name: str = ""
    step: str = ""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str = ""
    lamella_id: str = ""
    start_timestamp: float = field(
        default_factory=lambda: datetime.timestamp(datetime.now())
    )
    end_timestamp: Optional[float] = None
    status: AutoLamellaTaskStatus = AutoLamellaTaskStatus.NotStarted
    status_message: str = ""
    # files this run produced, keyed by role, as paths relative to lamella.path.
    # roles mirror the naming convention the files already carry: phase x modality,
    # i.e. final_sem / final_fib / start_sem / start_fib, plus fluorescence.
    # paths only -- the experiment is written with yaml.safe_dump, which refuses
    # numpy scalars and enums and would fail the whole save. measured values belong
    # in a separate field, not here.
    outputs: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def completed(self) -> str:
        return f"{self.name} ({self.completed_at})"

    @property
    def completed_at(self) -> str:
        if self.end_timestamp is None:
            return "in progress"
        return datetime.fromtimestamp(self.end_timestamp).strftime(
            TIME_DISPLAY_AMPM_SHORT
        )

    @property
    def started_at(self) -> str:
        return datetime.fromtimestamp(self.start_timestamp).strftime(
            TIME_DISPLAY_AMPM_SHORT
        )

    @property
    def duration(self) -> float:
        if self.end_timestamp is None:
            return 0
        return self.end_timestamp - self.start_timestamp

    @property
    def duration_str(self) -> str:
        return format_duration(self.duration)

    def to_dict(self) -> dict:
        """Convert the task state to a dictionary."""
        ddict = asdict(self)
        ddict["status"] = self.status.name
        return ddict

    @classmethod
    def from_dict(cls, data: dict) -> "AutoLamellaTaskState":
        """Create a task state from a dictionary."""
        if data is None:
            return cls()
        data = data.copy()
        data["status"] = AutoLamellaTaskStatus[data.get("status", "NotStarted")]
        # drop keys this build doesn't know about: an experiment written by a newer
        # version must still load in an older one rather than raising TypeError.
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@evented
@dataclass
class AutoLamellaTaskConfig(ABC):
    """Configuration for AutoLamella tasks."""

    task_type: ClassVar[str]
    display_name: ClassVar[str]
    related_tasks: ClassVar[list[type["AutoLamellaTaskConfig"]]] = []
    task_name: str = ""  # unique name for identifying in multi-task workflows
    milling: Dict[str, FibsemMillingTaskConfig] = field(default_factory=dict)
    reference_imaging: ReferenceImageParameters = field(
        default_factory=ReferenceImageParameters
    )

    @property
    def parameters(self) -> Tuple[str, ...]:
        core_params = [f.name for f in fields(AutoLamellaTaskConfig)]
        return tuple(f.name for f in fields(self) if f.name not in core_params)

    @property
    def field_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Return dataclass fields with metadata, filling any missing keys with defaults.

        Matches the property patterns and milling strategies already expose. The
        task form used to read raw `dataclasses.fields(...).metadata` instead, so
        the whole vocabulary past four keys was unavailable to it -- a task
        config could declare `minimum` and nothing would read it.
        """
        return get_fields_with_metadata(self.__class__)

    def to_dict(self) -> dict:
        """Convert configuration to a dictionary."""
        ddict = {}
        ddict["task_type"] = self.task_type
        # TODO: explicitly not saving imaging until implemented
        # extract all the .parameters into a "parameters subdict"
        ddict["parameters"] = {}
        for k in self.parameters:
            ddict["parameters"][k] = getattr(self, k)
        ddict["milling"] = {k: v.to_dict() for k, v in self.milling.items()}
        if self.reference_imaging is not None:
            ddict["reference_imaging"] = self.reference_imaging.to_dict()
        return ddict

    @classmethod
    def from_dict(cls, ddict: Dict[str, Any]) -> "AutoLamellaTaskConfig":
        kwargs = {}

        for f in fields(cls):
            if f.name in ddict:
                kwargs[f.name] = ddict[f.name]

        # unroll the parameters dictionary
        if "parameters" in ddict and ddict["parameters"] is not None:
            for key, value in ddict["parameters"].items():
                if key in cls.__annotations__:
                    kwargs[key] = value
                else:
                    logging.warning(f"Unknown parameter '{key}' in task configuration.")

        if "milling" in ddict:
            kwargs["milling"] = {
                k: FibsemMillingTaskConfig.from_dict(v)
                for k, v in ddict["milling"].items()
            }
        if "reference_imaging" in ddict:
            kwargs["reference_imaging"] = ReferenceImageParameters.from_dict(
                ddict["reference_imaging"]
            )

        return cls(**kwargs)

    @property
    def estimated_time(self) -> float:
        """Estimate the total time for this task (milling + reference imaging)."""
        milling_time = sum(t.estimated_time for t in self.milling.values())
        imaging_time = self.reference_imaging.estimated_time
        return milling_time + imaging_time

    @property
    def opens_with_stage_move(self) -> bool:
        """Whether ``_run`` begins by moving the stage onto the task's pose.

        True for nearly every task, and the exceptions say so. Charged even when the
        stage is already in place -- a task following another on the same lamella pays
        0.02 s for the move rather than 7.6 s -- because which case applies is runtime
        state the config cannot see, and over-charging errs long.
        """
        return True

    @property
    def opens_with_reference_alignment(self) -> bool:
        """Whether ``_run`` aligns against the stored reference before its own work.

        Off by default: only some tasks do it, and a task that claims it when it does
        not adds 5 s of fiction to every estimate that includes it.
        """
        return False

    @property
    def estimated_duration(self) -> float:
        """Conservative forward estimate of this task's wall-clock, in seconds.

        Unlike :attr:`estimated_time`, which counts only the scan arithmetic and the
        milling, this adds the measured per-operation costs the task actually pays --
        see :mod:`fibsem.timing`. On the run it was calibrated against, that moved
        Setup Lamella Position from 0.22x of its real duration to 0.99x.

        A task whose cost is dominated by something this base cannot see -- spot burn
        exposures, fluorescence channels -- overrides this and adds its own term. The
        estimate lives on the config rather than on the task because the callers that
        need it (the pre-run dialog, the queue) hold configs and have no microscope to
        construct a task with.
        """
        # Reference images at both ends, but not the same number: a task opens with
        # _acquire_reference_image at one field of view and closes with
        # _acquire_set_of_reference_images over all of them.
        total = timing.reference_image_cost(self.reference_imaging, fovs=1)
        total += timing.reference_image_cost(self.reference_imaging)
        # what _run does before its own work
        if self.opens_with_stage_move:
            total += timing.stage_move_cost(1)
        if self.opens_with_reference_alignment:
            total += timing.REFERENCE_ALIGNMENT_S
        for milling_task in self.milling.values():
            total += timing.milling_task_cost(milling_task)
        return total

    @property
    def imaging(self) -> ImageSettings:
        """Get the imaging settings from the reference imaging parameters."""
        return self.reference_imaging.imaging

    @imaging.setter
    def imaging(self, value: ImageSettings):
        """Set the imaging settings in the reference imaging parameters."""
        self.reference_imaging.imaging = value


@evented
@dataclass
class AutoLamellaTaskDescription:
    name: str  # unique_name
    supervise: bool
    required: bool
    requires: List[str] = field(default_factory=list)
    scheduled_at: Optional[datetime] = None
    # Who a supervised task's questions are addressed to: "human" (the
    # operator, today's behaviour and the default) or "agent" (a connected
    # agent answers; the operator can still answer first). Display-and-watchdog
    # semantics only — prompts are raised identically either way.
    supervisor: str = "human"
    # Propose and review: the task completes and leaves its answer as a proposal
    # for someone to confirm or reject later, off the beam, instead of asking
    # inline. Only tasks that know how to propose honour it (Setup Lamella
    # Position, to start with); the consumer that requires this task is
    # deferred until the proposal is decided. Independent of ``supervise``: a
    # task can still block on the questions that genuinely need a person at
    # the beam and propose the rest. Ignored unless the feature flag is on.
    review: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d.get("scheduled_at") is not None:
            d["scheduled_at"] = self.scheduled_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AutoLamellaTaskDescription":
        if data is None:
            return cls(name="", supervise=False, required=False, requires=[])
        # Known fields only: a protocol written by a newer version (with fields
        # this one does not know) must load, not crash on an unexpected kwarg.
        known = {f.name for f in fields(cls)}
        data = {k: v for k, v in data.items() if k in known}
        sa = data.get("scheduled_at")
        if isinstance(sa, str):
            data["scheduled_at"] = datetime.fromisoformat(sa)
        return cls(**data)


@evented
@dataclass
class AutoLamellaWorkflowConfig:
    name: str = ""
    description: str = ""
    tasks: List[AutoLamellaTaskDescription] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        ddict = asdict(self)
        ddict["tasks"] = [task.to_dict() for task in self.tasks]
        return ddict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AutoLamellaWorkflowConfig":
        data["tasks"] = [
            AutoLamellaTaskDescription.from_dict(task) for task in data.get("tasks", [])
        ]
        return cls(**data)

    @property
    def workflow(self) -> List[str]:
        return [task.name for task in self.tasks]

    @property
    def required_tasks(self) -> List[str]:
        """Get the list of required tasks for this workflow."""
        return [task.name for task in self.tasks if task.required]

    def requirements(self, task_name: str) -> List[str]:
        for task in self.tasks:
            if task.name == task_name:
                return task.requires
        return []

    def get_completed_tasks(
        self, lamella: "Lamella", with_timestamps: bool = False
    ) -> List[str]:
        """Get the list of completed tasks for a given lamella.

        Filtered on status for the same reason as Lamella.completed_tasks:
        task_history holds every terminal outcome since FIB-490. Unfiltered, a
        failed required task would make is_completed() true, which drives the
        ITEM_COMPLETED / EXPERIMENT_COMPLETED events and the is_completed column
        in the experiment summary.
        """
        completed_tasks = []
        for task in lamella.task_history:
            if task.status is not AutoLamellaTaskStatus.Completed:
                continue
            if task.name in self.workflow:
                txt = task.name
                if with_timestamps:
                    txt = task.completed
                completed_tasks.append(txt)
        return completed_tasks

    def get_remaining_tasks(self, lamella: "Lamella") -> List[str]:
        """Get the list of remaining tasks for a given lamella."""
        remaining_tasks = []
        completed_tasks = self.get_completed_tasks(lamella)
        for task in self.required_tasks:
            if task not in completed_tasks:
                remaining_tasks.append(task)
        return remaining_tasks

    def is_completed(self, lamella: "Lamella") -> bool:
        """Check if all required tasks for the workflow are completed."""
        completed_tasks = self.get_completed_tasks(lamella)
        for task in self.required_tasks:
            if task not in completed_tasks:
                return False
        return True

    def get_supervision(self, task_name: str) -> bool:
        """Check if a task requires supervision."""
        for task in self.tasks:
            if task.name == task_name:
                return task.supervise
        return False

    def get_review(self, task_name: str) -> bool:
        """Whether a task should propose rather than ask (see
        AutoLamellaTaskDescription.review)."""
        for task in self.tasks:
            if task.name == task_name:
                return task.review
        return False

    def get_supervisor(self, task_name: str) -> str:
        """Who a supervised task's questions are addressed to: human or agent."""
        for task in self.tasks:
            if task.name == task_name:
                return getattr(task, "supervisor", "human") or "human"
        return "human"

    def get_scheduled_at(self, task_name: str) -> Optional[datetime]:
        """Get the scheduled start time for a task, or None if not scheduled."""
        for task in self.tasks:
            if task.name == task_name:
                return task.scheduled_at
        return None

    def add_task(self, task: AutoLamellaTaskConfig) -> None:
        """Add a task to the workflow configuration."""
        self.tasks.append(
            AutoLamellaTaskDescription(
                name=task.task_name, supervise=True, required=True, requires=[]
            )
        )

    @property
    def is_valid(self) -> bool:
        """Check if the workflow configuration is valid."""
        issues = self.validate()
        return not issues

    def validate(self) -> List[str]:
        """Validate the workflow configuration and return a list of issues."""
        issues = []
        task_names = [task.name for task in self.tasks]
        for i, task in enumerate(self.tasks):
            for req in task.requires:
                if req not in task_names:
                    issues.append(f"Task '{task.name}' requires unknown task '{req}'.")
                elif req not in task_names[:i]:
                    issues.append(
                        f"Task '{task.name}' requires '{req}' which comes after it in the workflow."
                    )
        return issues


@evented
@dataclass
class AutoLamellaWorkflowOptions:
    turn_beams_off: bool = False
    # How long a run whose remaining work all waits on a review keeps waiting for
    # one, in seconds of *inactivity* -- every decision resets the clock, so a
    # slow reviewer mid-sitting is never cut off, only an absent one. 0 exits at
    # once (the old behaviour); None waits until stopped, for an unattended run
    # with an agent deciding. Separate from any per-proposal deadline: this is
    # whether the workflow thread stays parked, not who may decide.
    review_wait: Optional[float] = 1800.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AutoLamellaWorkflowOptions":
        return cls(**data)


@evented
@dataclass
class LamellaDefaultConfig:
    """Initial state applied to every new Lamella created from this protocol."""

    use_petname: bool = True
    name_prefix: str = ""
    alignment_area: Optional[FibsemRectangle] = None
    poi: Optional[Point] = None

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


@dataclass
class GridTaskProtocol:
    """The grid tasks a protocol offers, and their settings.

    A section of the task protocol (`protocol.yaml`, under `grid_tasks`), because
    overview settings are tuned once per microscope and reused across experiments
    exactly as the lamella task settings are. Shared by every grid for now:
    screening is expected to be uniform across a magazine, and per-grid tuning is
    a documented later option (seed a per-record config from this, the way a
    lamella's is seeded from the protocol). Which tasks run in a session, and in
    what order, is chosen at run time; `order` is the default a task list shows.
    """

    name: str = "Grid Task Protocol"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_config: Dict[str, "GridTaskConfig"] = field(default_factory=dict)
    order: List[str] = field(default_factory=list)

    def add(self, config: "GridTaskConfig") -> "GridTaskConfig":
        if not config.task_name:
            raise ValueError("A grid task config needs a task_name.")
        self.task_config[config.task_name] = config
        if config.task_name not in self.order:
            self.order.append(config.task_name)
        return config

    def remove(self, task_name: str) -> None:
        self.task_config.pop(task_name, None)
        if task_name in self.order:
            self.order.remove(task_name)

    @property
    def ordered_task_names(self) -> List[str]:
        """`order` first, then anything in `task_config` it forgot to mention."""
        names = [n for n in self.order if n in self.task_config]
        names += [n for n in self.task_config if n not in names]
        return names

    def to_dict(self) -> Dict[str, Any]:
        return {
            "_id": self.id,
            "name": self.name,
            "tasks": {k: v.to_dict() for k, v in self.task_config.items()},
            "order": list(self.ordered_task_names),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "GridTaskProtocol":
        from fibsem.applications.autolamella.workflows.tasks.grid.registry import (
            load_grid_task_configs,
        )

        data = data or {}
        protocol = cls(
            name=data.get("name", "Grid Task Protocol"),
            task_config=load_grid_task_configs(data.get("tasks", {})),
            order=list(data.get("order", [])),
        )
        if data.get("_id"):
            protocol.id = data["_id"]
        return protocol


@evented
@dataclass
class AutoLamellaTaskProtocol:
    name: str = "AutoLamella Task Protocol"
    description: str = "Protocol for AutoLamella"
    version: str = "1.0"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_config: EventedDict[str, AutoLamellaTaskConfig] = field(
        default_factory=lambda: EventedDict()
    )  # unique_name: AutoLamellaTaskConfig
    workflow_config: AutoLamellaWorkflowConfig = field(
        default_factory=AutoLamellaWorkflowConfig
    )
    options: AutoLamellaWorkflowOptions = field(
        default_factory=AutoLamellaWorkflowOptions
    )
    lamella_defaults: LamellaDefaultConfig = field(default_factory=LamellaDefaultConfig)
    # Experiment-global correlation config (FIB-298): a user-step config, not an
    # automated task, so a peer field rather than an entry in task_config.
    correlation: CorrelationConfig = field(default_factory=CorrelationConfig)
    # Grid-level tasks (overviews, and later cleaning and deposition): their own
    # section, since they run on a GridRecord rather than a lamella.
    grid_tasks: GridTaskProtocol = field(default_factory=GridTaskProtocol)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "_id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "tasks": {k: v.to_dict() for k, v in self.task_config.items()},
            "workflow": self.workflow_config.to_dict(),
            "options": self.options.to_dict(),
            "lamella_defaults": self.lamella_defaults.to_dict(),
            "correlation": self.correlation.to_dict(),
            "grid_tasks": self.grid_tasks.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AutoLamellaTaskProtocol":
        from fibsem.applications.autolamella.workflows.tasks import load_task_config

        task_config = load_task_config(data.get("tasks", {}))
        workflow_config = AutoLamellaWorkflowConfig.from_dict(data.get("workflow", {}))

        protocol = cls(
            name=data.get("name", "AutoLamella Task Protocol"),
            description=data.get("description", "Protocol for AutoLamella"),
            version=data.get("version", "1.0"),
            task_config=task_config,
            workflow_config=workflow_config,
            options=AutoLamellaWorkflowOptions.from_dict(data.get("options", {})),
            lamella_defaults=LamellaDefaultConfig.from_dict(
                data.get("lamella_defaults", {})
            ),
            # Missing on protocols saved before this field -> a default config.
            correlation=CorrelationConfig.from_dict(data.get("correlation")),
            # Likewise: a protocol written before grid tasks existed offers none.
            grid_tasks=GridTaskProtocol.from_dict(data.get("grid_tasks")),
        )
        if "_id" in data:
            protocol.id = data["_id"]
        return protocol

    @classmethod
    def load(cls, filename: str) -> "AutoLamellaTaskProtocol":
        with open(filename, "r") as file:
            data = yaml.safe_load(file)
        return cls.from_dict(data)

    def save(self, filename: str) -> None:
        """Save the task protocol to a YAML file."""
        with open(filename, "w") as file:
            yaml.safe_dump(
                self.to_dict(),
                file,
                indent=4,
                default_flow_style=False,
                sort_keys=False,
            )

    def get_supervision(self, task_name: str) -> bool:
        """Check if a task requires supervision."""
        return self.workflow_config.get_supervision(task_name)

    def get_review(self, task_name: str) -> bool:
        return self.workflow_config.get_review(task_name)

    def get_supervisor(self, task_name: str) -> str:
        """Who a supervised task's questions are addressed to: human or agent."""
        return self.workflow_config.get_supervisor(task_name)

    @classmethod
    def load_from_old_protocol(cls, path: Path) -> "AutoLamellaTaskProtocol":
        """Convert an AutoLamellaProtocol to an AutoLamellaTaskProtocol.
        This involves mapping the milling configurations to the new task names.
        Used to converte old protocols to the new task-based protocol format."""

        from fibsem.applications.autolamella.protocol.legacy import (
            AutoLamellaMethod,
            AutoLamellaProtocol,
            AutoLamellaStage,
        )
        from fibsem.applications.autolamella.workflows.tasks.tasks import (
            MillFiducialTaskConfig,
            MillPolishingTaskConfig,
            MillRoughTaskConfig,
            MillTrenchTaskConfig,
            MillUndercutTaskConfig,
            SelectMillingPositionTaskConfig,
        )

        protocol = AutoLamellaProtocol.load(path)

        # we need to map the milling configurations to the new task names
        # mill_rough -> Rough Milling / mill_rough
        # microexpansion -> Rough Milling / stress-relief
        # notch -> Rough Milling / stress-relief
        # trench -> Trench Milling / trench
        # undercut -> Trench Milling / undercut
        # fiducial -> Setup Lamella / fiducial
        # mill_polishing -> Polishing

        if protocol.method not in [
            AutoLamellaMethod.ON_GRID,
            AutoLamellaMethod.TRENCH,
            AutoLamellaMethod.WAFFLE,
        ]:
            raise ValueError(
                f"Protocol method {protocol.method} not supported for conversion to task protocol"
            )

        ROUGH_MILLING_TASK_NAME = "Rough Milling"
        POLISHING_TASK_NAME = "Polishing"
        TRENCH_MILLING_TASK_NAME = "Trench Milling"
        MILL_FIDUCIAL_TASK_NAME = "Mill Fiducial"
        UNDERCUT_TASK_NAME = "Undercut"
        SETUP_LAMELLA_POSITION_TASK_NAME = "Setup Lamella Position"

        workflow_config = AutoLamellaWorkflowConfig()
        task_config = EventedDict({})

        if protocol.method in [AutoLamellaMethod.ON_GRID, AutoLamellaMethod.WAFFLE]:
            rough_milling_task = MillRoughTaskConfig(
                task_name=ROUGH_MILLING_TASK_NAME,
                milling={
                    MILL_ROUGH_KEY: FibsemMillingTaskConfig.from_stages(
                        protocol.milling[MILL_ROUGH_KEY], name="Rough Milling"
                    )
                },
            )

            if protocol.options.use_microexpansion:
                rough_milling_task.milling[MILL_ROUGH_KEY].stages.extend(
                    protocol.milling[MICROEXPANSION_KEY]
                )
            if protocol.options.use_notch:
                rough_milling_task.milling[STRESS_RELIEF_KEY] = (
                    FibsemMillingTaskConfig.from_stages(
                        protocol.milling[NOTCH_KEY], name="Notch"
                    )
                )

            polishing_milling_task = MillPolishingTaskConfig(
                task_name=POLISHING_TASK_NAME,
                milling={
                    MILL_POLISHING_KEY: FibsemMillingTaskConfig.from_stages(
                        protocol.milling[MILL_POLISHING_KEY], name="Polishing"
                    )
                },
            )

            mill_fiducial_task = MillFiducialTaskConfig(
                task_name=MILL_FIDUCIAL_TASK_NAME,
                milling={
                    FIDUCIAL_KEY: FibsemMillingTaskConfig.from_stages(
                        protocol.milling[FIDUCIAL_KEY], name="Fiducial"
                    )
                },
            )
            setup_lamella_task = SelectMillingPositionTaskConfig(
                task_name=SETUP_LAMELLA_POSITION_TASK_NAME,
                milling={},
                milling_angle=protocol.options.milling_angle,
            )

            task_config[ROUGH_MILLING_TASK_NAME] = rough_milling_task
            task_config[POLISHING_TASK_NAME] = polishing_milling_task
            task_config[MILL_FIDUCIAL_TASK_NAME] = mill_fiducial_task
            task_config[SETUP_LAMELLA_POSITION_TASK_NAME] = setup_lamella_task

            workflow_config.tasks = [
                AutoLamellaTaskDescription(
                    name=SETUP_LAMELLA_POSITION_TASK_NAME,
                    supervise=protocol.supervision[AutoLamellaStage.SetupLamella],
                    required=True,
                ),
                AutoLamellaTaskDescription(
                    name=MILL_FIDUCIAL_TASK_NAME,
                    supervise=protocol.supervision[AutoLamellaStage.SetupLamella],
                    required=True,
                ),
                AutoLamellaTaskDescription(
                    name=ROUGH_MILLING_TASK_NAME,
                    supervise=protocol.supervision[AutoLamellaStage.MillRough],
                    required=True,
                    requires=[MILL_FIDUCIAL_TASK_NAME],
                ),
                AutoLamellaTaskDescription(
                    name=POLISHING_TASK_NAME,
                    supervise=protocol.supervision[AutoLamellaStage.MillPolishing],
                    required=True,
                    requires=[ROUGH_MILLING_TASK_NAME],
                ),
            ]

        if protocol.method in [AutoLamellaMethod.TRENCH, AutoLamellaMethod.WAFFLE]:
            trench_milling_task = MillTrenchTaskConfig(
                task_name=TRENCH_MILLING_TASK_NAME,
                milling={
                    TRENCH_KEY: FibsemMillingTaskConfig.from_stages(
                        protocol.milling[TRENCH_KEY], name="Trench"
                    ),
                },
                orientation="FIB",
            )
            task_config[TRENCH_MILLING_TASK_NAME] = trench_milling_task
            workflow_config.tasks.insert(
                0,
                AutoLamellaTaskDescription(
                    name=TRENCH_MILLING_TASK_NAME,
                    supervise=protocol.supervision[AutoLamellaStage.MillTrench],
                    required=True,
                ),
            )

        if protocol.method is AutoLamellaMethod.WAFFLE:
            undercut_task = MillUndercutTaskConfig(
                task_name=UNDERCUT_TASK_NAME,
                milling={
                    UNDERCUT_KEY: FibsemMillingTaskConfig.from_stages(
                        protocol.milling[UNDERCUT_KEY], name="Undercut"
                    )
                },
                orientation="SEM",
            )
            task_config[UNDERCUT_TASK_NAME] = undercut_task
            workflow_config.tasks.insert(
                1,
                AutoLamellaTaskDescription(
                    name=UNDERCUT_TASK_NAME,
                    supervise=protocol.supervision[AutoLamellaStage.MillUndercut],
                    required=True,
                    requires=[TRENCH_MILLING_TASK_NAME],
                ),
            )

        options = AutoLamellaWorkflowOptions(
            turn_beams_off=protocol.options.turn_beams_off,
        )

        workflow_config.name = protocol.name
        workflow_config.description = (
            f"auto-converted protocol from {protocol.name} - {protocol.method.name}"
        )

        task_protocol = AutoLamellaTaskProtocol(
            name=protocol.name,
            description=f"auto-converted protocol from {protocol.name} - {protocol.method.name}",
            task_config=task_config,
            workflow_config=workflow_config,
            options=options,
        )

        return task_protocol

    def get_task_config_by_type(
        self, task_type: Type["AutoLamellaTaskConfig"]
    ) -> EventedDict[str, AutoLamellaTaskConfig]:
        """Get the task configuration by type."""
        task_configs = EventedDict()
        for k, v in self.task_config.items():
            if isinstance(v, task_type):
                task_configs[k] = v
        return task_configs


class Verdict(Enum):
    """A judgement about a lamella or a grid, made by a person (or a reviewer
    acting for one) and never by a task.

    ``UNASSESSED`` and ``GOOD`` are different answers: nobody has looked, versus
    somebody looked and it was fine. The old ``DefectType.NONE`` meant both.
    """

    UNASSESSED = auto()
    GOOD = auto()
    REWORK = auto()
    FAILED = auto()

    # Aliases for the names the old enums used. ``Verdict["NONE"]`` resolves,
    # ``Verdict.NONE is Verdict.UNASSESSED``, and iteration skips them.
    NONE = UNASSESSED
    FAILURE = FAILED
    POOR = FAILED


@evented
@dataclass
class QualityRecord:
    """The current human verdict on an item, attributed.

    One type for lamellae and grids, replacing ``DefectState`` (lamella) and the
    bare ``GridQuality`` enum (grid), which were the same idea under two names
    with neither recording who set it. Holds the *current* verdict only; the
    full trail of what was proposed and decided lives on the item's proposals,
    and ``decision_id`` points at the decision that set this verdict when a
    review did.
    """

    verdict: Verdict = field(default=Verdict.UNASSESSED)
    author: str = ""  # "human:<name>" | "agent:<model>"; "" when unrecorded
    reason: str = ""
    at_task: str = ""  # the task the item was judged at
    updated_at: Optional[float] = None
    decision_id: Optional[Tuple[str, str]] = None  # (item_id, task_name)

    # -- names the old DefectState API used; kept so callers migrate one at a time
    @property
    def state(self) -> Verdict:
        return self.verdict

    @state.setter
    def state(self, value: Verdict) -> None:
        self.verdict = value

    @property
    def description(self) -> str:
        return self.reason

    @description.setter
    def description(self, value: str) -> None:
        self.reason = value

    @property
    def last_completed_task(self) -> str:
        return self.at_task

    @last_completed_task.setter
    def last_completed_task(self, value: str) -> None:
        self.at_task = value

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.name,
            "author": self.author,
            "reason": self.reason,
            "at_task": self.at_task,
            "updated_at": self.updated_at,
            "decision_id": list(self.decision_id) if self.decision_id else None,
        }

    @classmethod
    def from_dict(cls, data: Union[dict, str, None]) -> "QualityRecord":
        """Read the current shape, the ``DefectState`` shape (``state`` /
        ``description`` / ``last_completed_task``), the pre-``DefectState``
        ``has_defect`` / ``requires_rework`` bools, and the bare name string
        ``GridRecord`` used to write. Unknown names read as ``UNASSESSED``."""
        if not data:
            return cls()
        if isinstance(data, str):
            return cls(verdict=_verdict_by_name(data))
        if "has_defect" in data:
            if data.get("has_defect"):
                verdict = (
                    Verdict.REWORK if data.get("requires_rework") else Verdict.FAILED
                )
            else:
                verdict = Verdict.UNASSESSED
            return cls(
                verdict=verdict,
                reason=data.get("description", ""),
                updated_at=data.get("updated_at", None),
            )
        decision_id = data.get("decision_id")
        return cls(
            verdict=_verdict_by_name(data.get("verdict", data.get("state", ""))),
            author=data.get("author", ""),
            reason=data.get("reason", data.get("description", "")),
            at_task=data.get("at_task", data.get("last_completed_task", "")),
            updated_at=data.get("updated_at", None),
            decision_id=tuple(decision_id) if decision_id else None,
        )

    def clear(self):
        self.verdict = Verdict.UNASSESSED
        self.author = ""
        self.reason = ""
        self.at_task = ""
        self.updated_at = None
        self.decision_id = None

    def set_defect(
        self,
        description: str = "",
        state: Verdict = Verdict.FAILED,
        author: str = "",
    ):
        self.verdict = state
        self.reason = description
        self.author = author
        self.updated_at = datetime.timestamp(datetime.now())


_quality_record_init = QualityRecord.__init__


def _quality_record_init_with_old_names(
    self,
    *args,
    state: Optional[Verdict] = None,
    description: Optional[str] = None,
    last_completed_task: Optional[str] = None,
    **kwargs,
):
    """``DefectState(state=..., description=..., last_completed_task=...)`` keeps
    constructing. The dataclass ``__init__`` only knows the new field names, so
    the old keywords are mapped here; a caller passing both forms gets the new
    one."""
    if state is not None:
        kwargs.setdefault("verdict", state)
    if description is not None:
        kwargs.setdefault("reason", description)
    if last_completed_task is not None:
        kwargs.setdefault("at_task", last_completed_task)
    _quality_record_init(self, *args, **kwargs)


QualityRecord.__init__ = _quality_record_init_with_old_names  # type: ignore[method-assign]


def _verdict_by_name(name: str) -> Verdict:
    try:
        return Verdict[name]
    except KeyError:
        return Verdict.UNASSESSED


# The names this type and its enum had before they were one thing. Every reader
# of ``DefectState`` / ``DefectType`` / ``GridQuality`` keeps working; new code
# says ``QualityRecord`` / ``Verdict``.
DefectState = QualityRecord
DefectType = Verdict
GridQuality = Verdict


# The thumbnail is only ever displayed, and its largest reader is the cozy lamella
# card at 280 px wide (the hover tooltip is 256). Storing the acquired frame at full
# resolution meant decoding a 1536x1024, 823 KB PNG per card per refresh for a picture
# thrown away at a fifth of the size -- 26 ms each, and the card strip re-reads every
# lamella whenever one is added (FIB-681).
#
# 512 rather than 280: it leaves the largest card headroom on a HiDPI display, still
# decodes in ~3 ms, and is a tenth of the bytes. Only ever shrinks -- a frame already
# smaller than this is written through untouched.
# The bound lives with the writer; kept here by name for anything that imports it.
from fibsem.imaging.thumbnail import (
    THUMBNAIL_MAX_EDGE as _THUMBNAIL_MAX_EDGE,  # noqa: E402
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
class Lamella:
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
    task_state: AutoLamellaTaskState = field(default_factory=AutoLamellaTaskState)
    task_history: List["AutoLamellaTaskState"] = field(default_factory=list)
    defect: DefectState = field(default_factory=DefectState)
    milling_angle: Optional[float] = None
    poi: Point = field(
        default_factory=lambda: Point(0, 0)
    )  # point of interest within lamella area (milling coordinate system)
    description: str = ""  # free-text note about the lamella
    # The grid this lamella sits on: GridRecord.id, or None when the experiment
    # does not track grids (every experiment before grid records existed). A
    # back-reference only; grid -> lamella is derived by filtering on it.
    grid_id: Optional[str] = None
    # What tasks proposed for this lamella and what was decided, keyed by the
    # producing task's name. A proposal with no decisions is pending; the
    # consumer that requires that task is deferred until one is appended. See
    # proposals.py and Experiment.decide.
    proposals: Dict[str, Proposal] = field(default_factory=dict)

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
        self.task_state.lamella_id = self.id

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

    def has_completed_task(self, task_name: str) -> bool:
        """Check if the lamella has completed a specific task."""
        return task_name in self.completed_tasks

    @property
    def completed_tasks(self) -> List[str]:
        """Return a list of completed task names.

        Filtered on status: task_history records every terminal outcome, not just
        successes (FIB-490), so an unfiltered read would count a failed task as
        done. That matters most in TaskManager._should_skip, where this gates
        prerequisites -- a failed trench would otherwise license an undercut.
        """
        return [
            task.name
            for task in self.task_history
            if task.status is AutoLamellaTaskStatus.Completed
        ]

    @property
    def last_completed_task(self) -> Optional["AutoLamellaTaskState"]:
        """Return the last completed task state.

        The last *completed* one, not the last recorded: a failure landing last
        would otherwise be reported as the lamella's latest progress in the
        experiment summary.
        """
        for task in reversed(self.task_history):
            if task.status is AutoLamellaTaskStatus.Completed:
                return task
        return None

    @property
    def landing_selected(self) -> bool:
        return self.landing_pose is not None

    @property
    def landing_pose(self) -> Optional[MicroscopeState]:
        return self.poses.get("LANDING", None)

    @landing_pose.setter
    def landing_pose(self, value: MicroscopeState):
        """Set the landing pose for the lamella."""
        if not isinstance(value, MicroscopeState):
            raise TypeError("Landing pose must be a MicroscopeState instance.")
        self.poses["LANDING"] = value

    @property
    def milling_pose(self) -> Optional[MicroscopeState]:
        return self.poses.get("MILLING", None)

    @milling_pose.setter
    def milling_pose(self, value: MicroscopeState):
        """Set the milling pose for the lamella."""
        if not isinstance(value, MicroscopeState):
            raise TypeError("Milling pose must be a MicroscopeState instance.")
        self.poses["MILLING"] = value

    @property
    def fluorescence_pose(self) -> Optional[MicroscopeState]:
        return self.poses.get("FLUORESCENCE", None)

    @fluorescence_pose.setter
    def fluorescence_pose(self, value: MicroscopeState):
        """Set the fluorescence pose for the lamella."""
        if not isinstance(value, MicroscopeState):
            raise TypeError("Fluorescence pose must be a MicroscopeState instance.")
        self.poses["FLUORESCENCE"] = value

    @property
    def fluorescence_selected(self) -> bool:
        return (
            self.fluorescence_pose is not None
            and self.fluorescence_pose.objective_position is not None
        )

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

    @property
    def pretty_fm_name(self) -> str:
        """Generate a pretty name for the stage position."""
        obj_pos = (
            self.fluorescence_pose.objective_position
            if self.fluorescence_pose is not None
            else None
        )
        objective_str = f"{obj_pos * 1e3:.3f}mm" if obj_pos is not None else "N/A"
        return f"{self.name} ({self.stage_position.x * 1e6:.1f}μm, {self.stage_position.y * 1e6:.1f}μm, {objective_str})"

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
            task_state=AutoLamellaTaskState.from_dict(data.get("task_state", {})),
            task_history=[
                AutoLamellaTaskState.from_dict(task)
                for task in data.get("task_history", [])
            ],
            defect=DefectState.from_dict(data.get("defect", {})),
            milling_angle=data.get("milling_angle", None),
            poi=Point.from_dict(data.get("poi", {"x": 0, "y": 0})),
            description=data.get("description", ""),
            grid_id=data.get("grid_id"),
            proposals=proposals_from_dict(data.get("proposals")),
        )

    def load_reference_image(self, fname) -> FibsemImage:
        """Load a specific reference image for this lamella from disk
        Args:
            fname: str
                the filename of the reference image to load
        Returns:
            image: FibsemImage
                the reference image loaded as a FibsemImage
        """

        image = FibsemImage.load(os.path.join(self.path, f"{fname}.tif"))

        return image

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

    def save_thumbnail(self, image: "FibsemImage") -> None:
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
        `_THUMBNAIL_MAX_EDGE`. This is a thumbnail, not a copy of the frame; the frame
        itself is saved separately by the task that acquired it.
        """
        from fibsem.imaging.thumbnail import write_thumbnail

        write_thumbnail(image.filtered_data, os.path.join(self.path, "thumbnail.png"))

    # def get_task_config_by_type(self, task_type: Type['AutoLamellaTaskConfig']) -> Dict[str, AutoLamellaTaskConfig]:
    #     """Get the task configuration by type."""
    #     task_configs = {}
    #     for k, v in self.task_config.items():
    #         if isinstance(v, task_type):
    #             task_configs[k] = v
    #     return task_configs

    def sync_tasks_to_poi(self, point: Optional[Point] = None) -> list[str]:
        """Sync the milling patterns to point of interest"""

        if point is None:
            point = self.poi

        synced_tasks = []

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
                    stage.pattern.point = stage.pattern.point + diff

            synced_tasks.append(task_name)
        return synced_tasks


@evented
@dataclass
class GridRecord:
    """A grid as the workflow knows it, distinct from the hardware's `SampleGrid`.

    Linked to the hardware by name and to lamellae by `Lamella.grid_id -> id`.
    Deliberately holds no slot, stage position, loaded state or copy of the
    hardware grid: all of that is resolved live against the stage, so a record
    stays valid when its grid changes slot or leaves the magazine. A record exists
    from the moment a grid is *available* (present in the inventory), so twelve
    grids can be queued before any has been loaded.

    `task_state` and `task_history` are the same types the lamella side uses,
    and the same rule applies: never replace `task_state` wholesale, mutate it.
    """

    name: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    quality: QualityRecord = field(default_factory=QualityRecord)
    task_state: AutoLamellaTaskState = field(default_factory=AutoLamellaTaskState)
    task_history: List[AutoLamellaTaskState] = field(default_factory=list)
    created_at: float = field(
        default_factory=lambda: datetime.timestamp(datetime.now())
    )
    proposals: Dict[str, Proposal] = field(default_factory=dict)  # as on Lamella

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())

    def has_completed_task(self, task_name: str) -> bool:
        return any(
            t.name == task_name and t.status is AutoLamellaTaskStatus.Completed
            for t in self.task_history
        )

    @property
    def is_failure(self) -> bool:
        """Whether the latest run on this grid ended in failure. Not a verdict on
        the grid -- see `quality` for that, which a person sets and a task never
        does, same as a lamella."""
        return self.task_state.status is AutoLamellaTaskStatus.Failed

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "_id": self.id,
            "description": self.description,
            "quality": self.quality.to_dict(),
            "task_state": self.task_state.to_dict(),
            "task_history": [t.to_dict() for t in self.task_history],
            "created_at": self.created_at,
            "proposals": proposals_to_dict(self.proposals),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GridRecord":
        return cls(
            name=data["name"],
            id=data.get("_id") or str(uuid.uuid4()),
            description=data.get("description", ""),
            quality=QualityRecord.from_dict(data.get("quality")),
            task_state=AutoLamellaTaskState.from_dict(data.get("task_state", {})),
            task_history=[
                AutoLamellaTaskState.from_dict(t) for t in data.get("task_history", [])
            ],
            created_at=data.get("created_at", datetime.timestamp(datetime.now())),
            proposals=proposals_from_dict(data.get("proposals")),
        )

    def __repr__(self) -> str:
        return f"GridRecord(name={self.name!r}, quality={self.quality.verdict.name}, tasks={len(self.task_history)})"


# One process, one open experiment: a single lock serialises the writers that
# can run on different threads at once -- the workflow thread saving after a
# task, and a review confirming from the GUI thread or the agent server. Held
# briefly by both, so neither serialises a half-applied write of the other.
EXPERIMENT_WRITE_LOCK = threading.RLock()


def _call_on_main_thread(func, *args, **kwargs):
    """Run ``func`` on the Qt main thread and wait for its result, when there
    is a Qt application to have one. Without one (a script, a headless
    review, the test job that installs no Qt) it is a plain call.

    Lazy on purpose: superqt and PyQt are in the ``ui`` extra, and this module
    is imported everywhere.
    """
    try:
        from PyQt5.QtCore import QCoreApplication, QThread
        from superqt import ensure_main_thread
    except ImportError:
        return func(*args, **kwargs)
    app = QCoreApplication.instance()
    if app is None or QThread.currentThread() is app.thread():
        return func(*args, **kwargs)
    return ensure_main_thread(await_return=True)(func)(*args, **kwargs)


@evented
@dataclass
class Experiment:
    name: str
    id: str
    path: Path
    positions: EventedList[Lamella] = field(default_factory=EventedList)
    grids: EventedList[GridRecord] = field(default_factory=EventedList)
    landing_positions: List[FibsemStagePosition] = field(default_factory=list)
    created_at: float = field(
        default_factory=lambda: datetime.timestamp(datetime.now())
    )
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
        self.name: str = name
        self.id = str(uuid.uuid4())
        self.path: Path = os.path.join(path, name)
        self.created_at: float = datetime.timestamp(datetime.now())

        self.positions: EventedList[Lamella] = EventedList()
        self.grids: EventedList[GridRecord] = EventedList()
        self.landing_positions: List[FibsemStagePosition] = []

        self.task_protocol: AutoLamellaTaskProtocol = None  # must be set externally
        self.metadata: Dict[str, Any] = metadata if metadata is not None else {}
        # The instrument, operator and software that last worked on this. Filled in
        # by register_metadata, when a session adopts the experiment and there is a
        # microscope to ask; None until then, because creating an experiment happens
        # in a dialog that has no microscope. See FIB-451.
        self.session: Optional[SessionInfo] = None

    def to_dict(self, include_protocol: bool = False) -> dict:

        state_dict = {
            "name": self.name,
            "_id": self.id,
            "path": self.path,
            "positions": [deepcopy(lamella.to_dict()) for lamella in self.positions],
            "grids": [grid.to_dict() for grid in self.grids],
            "landing_positions": [pos.to_dict() for pos in self.landing_positions],
            "created_at": self.created_at,
            "metadata": self.metadata,
            "session": self.session.to_dict() if self.session is not None else None,
        }

        if include_protocol:
            state_dict["protocol"] = (
                self.task_protocol.to_dict() if self.task_protocol is not None else None
            )

        return state_dict

    @classmethod
    def from_dict(cls, ddict: dict) -> "Experiment":

        path = os.path.dirname(ddict["path"])
        name = ddict["name"]
        experiment = Experiment(path=path, name=name)
        experiment.created_at = ddict.get("created_at", None)
        experiment.id = ddict.get("_id", "NULL")

        experiment.metadata = ddict.get("metadata", {})

        # Absent from every experiment written before FIB-451, and from any that no
        # session has adopted yet. Both mean the same thing -- nothing is known about
        # what produced this -- and None says that without guessing.
        session = ddict.get("session")
        experiment.session = SessionInfo.from_dict(session) if session else None

        # load lamella from dict
        for lamella_dict in ddict["positions"]:
            lamella = Lamella.from_dict(data=lamella_dict)
            experiment.positions.append(lamella)

        # Experiments written before grid records existed have no "grids" key and
        # load with none; nothing else about them changes.
        for grid_dict in ddict.get("grids", []):
            experiment.grids.append(GridRecord.from_dict(grid_dict))

        # load landing positions
        for landing_dict in ddict.get("landing_positions", []):
            stage_position = FibsemStagePosition.from_dict(landing_dict)
            experiment.landing_positions.append(stage_position)

        return experiment

    @property
    def description(self) -> str:
        """Get the experiment description from metadata."""
        return self.metadata.get("description", "")

    @description.setter
    def description(self, value: str):
        """Set the experiment description in metadata."""
        self.metadata["description"] = value

    @property
    def user(self) -> str:
        """Get the user name from metadata."""
        return self.metadata.get("user", "")

    @user.setter
    def user(self, value: str):
        """Set the user name in metadata."""
        self.metadata["user"] = value

    @property
    def project(self) -> str:
        """Get the project name from metadata."""
        return self.metadata.get("project", "")

    @project.setter
    def project(self, value: str):
        """Set the project name in metadata."""
        self.metadata["project"] = value

    @property
    def organisation(self) -> str:
        """Get the organisation name from metadata."""
        return self.metadata.get("organisation", "")

    @organisation.setter
    def organisation(self, value: str):
        """Set the organisation name in metadata."""
        self.metadata["organisation"] = value

    # Fired after a decision is applied, with (item_id, task_name), on the thread
    # decide() ran on. The work queue listens so a stalled run wakes and rescans.
    decided = Signal(str, str)

    def get_lamella_by_id(self, lamella_id: str) -> Optional["Lamella"]:
        for lamella in self.positions:
            if lamella.id == lamella_id:
                return lamella
        return None

    def get_item_by_id(self, item_id: str) -> Optional[Union["Lamella", GridRecord]]:
        """A lamella or a grid: the two kinds of item a task runs on, and the
        two things a proposal can sit on. By id -- names change."""
        return self.get_lamella_by_id(item_id) or self.get_grid_by_id(item_id)

    def author(self) -> str:
        """Who is deciding, as a proposal decision records it, from the operator
        named on the experiment (or the OS account when nobody was)."""
        user = self._declared_user() or FibsemUser.from_environment()
        return human_author(user.name)

    def decide(
        self, item_id: str, task_name: str, decision: Decision
    ) -> DecisionResult:
        """The one way a decision reaches the experiment.

        Runs on the Qt main thread when there is one and blocks the caller until
        it has: the Review tab and the agent server are the same client of this
        function, and a confirm can end in Qt work (a generative review adds
        lamellae, whose list rebuild must not fire off the GUI thread). Takes
        the same lock as ``save`` so a save never serialises a half-applied
        decision.

        Confirmed: the decision is appended and each decided value is written
        through to the item (``poi`` moves the point and syncs the patterns
        that follow it). The proposed values are left as they were, so the
        delta survives. Rejected on a gating kind: the item is retired --
        ``quality`` set to ``FAILED`` with the reviewer as author -- because a
        consumer was waiting and *nothing further here* is the answer. Rejected
        on a generative kind: nothing is created.

        Refused, without a write, when there is no such proposal, or when the
        item has a task in progress: a decision then is a stop, not a decision.
        """
        return _call_on_main_thread(self._decide, item_id, task_name, decision)

    def _decide(
        self, item_id: str, task_name: str, decision: Decision
    ) -> DecisionResult:
        with EXPERIMENT_WRITE_LOCK:
            item = self.get_item_by_id(item_id)
            if item is None:
                return DecisionResult(
                    applied=False, reason=f"No item with id {item_id!r}."
                )
            proposal = item.proposals.get(task_name)
            if proposal is None:
                return DecisionResult(
                    applied=False,
                    reason=f"{item.name} has no proposal from {task_name!r}.",
                )
            if item.task_state.status is AutoLamellaTaskStatus.InProgress:
                return DecisionResult(
                    applied=False,
                    running=True,
                    reason=f"{item.name} is running {item.task_state.name}; "
                    "stop it rather than deciding under it.",
                )
            if decision.outcome is DecisionOutcome.Rejected and not decision.reason:
                return DecisionResult(applied=False, reason="A reject needs a reason.")
            if decision.outcome is DecisionOutcome.Confirmed:
                # Refuse before appending: a value nothing consumes is a
                # producer bug, and must not leave a half-applied decision.
                unknown = [n for n in decision.values if not has_value_writer(n)]
                if unknown:
                    return DecisionResult(
                        applied=False,
                        reason=f"No consumer writes {unknown}; known values: "
                        f"{known_value_names()}.",
                    )

            proposal.decisions.append(decision)
            result = DecisionResult(applied=True)
            if decision.outcome is DecisionOutcome.Confirmed:
                for name, value in decision.values.items():
                    synced = write_value(item, name, value)
                    if synced:
                        result.synced_tasks.extend(synced)
                result.delta = proposal.delta(decision)
            elif proposal.gating:
                item.quality.set_defect(
                    description=decision.reason,
                    state=Verdict.FAILED,
                    author=decision.author,
                )
                item.quality.at_task = task_name
                item.quality.decision_id = (item_id, task_name)
            logging.info(
                {
                    "msg": "proposal_decided",
                    "item": item.name,
                    "task_name": task_name,
                    "outcome": decision.outcome.name,
                    "author": decision.author,
                    "delta": {
                        k: getattr(v, "to_dict", lambda: v)()
                        for k, v in result.delta.items()
                    },
                }
            )
        self.decided.emit(item_id, task_name)
        return result

    def pending_proposals(
        self,
    ) -> List[Tuple[Union["Lamella", GridRecord], str, Proposal]]:
        """Every undecided proposal, in item order: the review inbox. Derived,
        never stored, so it is the same list from the GUI and the server."""
        pending = []
        for item in list(self.positions) + list(self.grids):
            for task_name, proposal in item.proposals.items():
                if proposal.pending:
                    pending.append((item, task_name, proposal))
        return pending

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

    def get_grid_by_name(self, name: str) -> Optional[GridRecord]:
        return next((g for g in self.grids if g.name == name), None)

    def get_grid_by_id(self, grid_id: Optional[str]) -> Optional[GridRecord]:
        if grid_id is None:
            return None
        return next((g for g in self.grids if g.id == grid_id), None)

    def add_grid(self, grid: GridRecord) -> GridRecord:
        """Track a grid. Names are the link to the hardware, so they are unique."""
        if self.get_grid_by_name(grid.name) is not None:
            raise ValueError(f"Grid '{grid.name}' already exists in the experiment.")
        self.grids.append(grid)
        return grid

    def remove_grid(self, name: str) -> Optional[GridRecord]:
        """Stop tracking a grid. Its lamellae are orphaned, not deleted."""
        grid = self.get_grid_by_name(name)
        if grid is None:
            return None
        for lamella in self.get_lamellae_for_grid(grid):
            lamella.grid_id = None
        self.grids.remove(grid)
        return grid

    def sync_grids_from_inventory(self, stage) -> List[GridRecord]:
        """Create a record for every grid the inventory reports present.

        Idempotent, matched by name: a grid already tracked is left alone, with its
        history; a grid whose hardware has gone keeps its record too, and the UI
        reads "not present" for it from the inventory. Returns the records added.
        """
        added: List[GridRecord] = []
        for entry in stage.grid_inventory():
            if not entry.present or not entry.name:
                continue
            if self.get_grid_by_name(entry.name) is None:
                added.append(self.add_grid(GridRecord(name=entry.name)))
        return added

    def grid_path(self, grid: GridRecord) -> Path:
        """Where a grid's task output goes: `grids/<name>/` under the experiment.

        Its own directory so it can never collide with a lamella's, which sits
        directly under the experiment path. Not created here; a task creates its
        own output directory when it first writes.
        """
        return Path(self.path) / "grids" / grid.name

    def get_lamellae_for_grid(self, grid: GridRecord) -> List["Lamella"]:
        """Derived from `Lamella.grid_id`; nothing stores the reverse."""
        return [p for p in self.positions if p.grid_id == grid.id]

    def get_grid_for_lamella(self, lamella: "Lamella") -> Optional[GridRecord]:
        return self.get_grid_by_id(lamella.grid_id)

    def save(self, save_protocol: bool = False) -> None:
        """Save the sample data to yaml file"""

        with EXPERIMENT_WRITE_LOCK:
            data = self.to_dict()
        with open(os.path.join(self.path, "experiment.yaml"), "w") as f:
            yaml.safe_dump(data, f, indent=4)
        if save_protocol:
            self.save_protocol()

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
        with open(path, "r") as f:
            ddict = yaml.safe_load(f)

        # create experiment from dict
        experiment = Experiment.from_dict(ddict)
        experiment.path = os.path.dirname(fname)

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
                    self.task_protocol.task_config[task_name] = deepcopy(
                        source_task_config[task_name]
                    )
            logging.info(f"Updated base protocol tasks: {task_names}")

        return updated_count

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

    def configure_logging(self) -> str:
        """Send this process's log output to the experiment's logfile.

        Note this reconfigures the *root* logger: it is process-global, not
        scoped to this experiment, and it replaces whatever handlers were set up
        before. Two consequences worth knowing before calling it:

        * a process that logs elsewhere stops doing so
        * a second call points logging at the second experiment

        ``load`` and ``create`` deliberately do not call this. Reading an
        experiment must not reach into the caller's logging -- the load dialog
        reads one on every click to preview it -- so the callers that do want it
        say so. The app calls this when it adopts an experiment; a headless
        script that wants its output in the experiment folder calls it after
        loading or creating one. See FIB-421.

        Returns:
            The path of the logfile being written to.
        """
        return _configure_logging(path=self.path, log_filename="logfile")

    def register_metadata(self, microscope: "FibsemMicroscope") -> None:
        """Stamp this experiment's identity onto the images ``microscope`` acquires.

        Which experiment produced an image is a property of the run, not of the GUI.
        This previously happened only in AutoLamellaUI, so images acquired from a
        script, a headless run, or the standalone FibsemUI carried the microscope's
        default ``FibsemExperimentRef()`` -- id ``None`` -- and could not be associated
        with an experiment afterwards. See FIB-449.

        Mirrors ``configure_logging``: ``load`` and ``create`` deliberately do not
        call it, because reading an experiment must not reach into the caller's
        microscope. Callers that own the run say so. The app calls it when it adopts
        an experiment, TaskManager calls it when it takes one to run, and a script
        driving the microscope directly calls it itself.

        Registration also runs the other way: this is the only moment the experiment
        meets a microscope, so it is where the experiment learns which session is
        working on it -- instrument, operator, software and plugins (FIB-451).
        """
        from fibsem.utils import _register_metadata

        _register_metadata(
            microscope=microscope,
            application_software="autolamella",
            experiment_id=self.id,
            experiment_name=self.name,
        )

        # After _register_metadata, which sets `application` on the very SystemInfo
        # being snapshotted here.
        self.session = SessionInfo.collect(microscope, user=self._declared_user())

        # Written now rather than left to whatever saves next -- relying on someone
        # else's save is how FIB-490 lost every failed task. Only for an experiment
        # that already has a file, though: constructing one deliberately does not
        # touch the disk (FIB-420), and both `create` and `load` guarantee a file
        # exists, so in a real run this always writes. A caller driving an
        # in-memory experiment is not asking this method to give it a directory.
        if os.path.exists(os.path.join(self.path, "experiment.yaml")):
            self.save()

    def _declared_user(self) -> Optional[FibsemUser]:
        """The operator named when the experiment was created, if anyone was.

        The create dialog collects a name and organisation as free text into the
        untyped ``metadata`` dict. Those beat ``FibsemUser.from_environment()``,
        which reads the OS account -- on a shared facility login that names the
        workstation rather than the person, and somebody who typed a name meant it.

        Only the two fields the dialog offers are overridden. ``hostname`` stays as
        the environment reports it, because it answers a different question: which
        machine, not which person.
        """
        metadata = self.metadata or {}
        name = metadata.get("user")
        organization = metadata.get("organisation")
        if not name and not organization:
            return None

        user = FibsemUser.from_environment()
        if name:
            user.name = name
        if organization:
            user.organization = organization
        return user

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
            petname=name, path=path, number=number, task_config=deepcopy(task_config)
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
                    "lamella_id": task.lamella_id,
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
        for i, t in enumerate(self.task_protocol.workflow_config.tasks, 1):
            ddict = {
                "order": i,
                "task_name": t.name,
                "required": t.required,
                "supervised": t.supervise,
            }
            wlist.append(deepcopy(ddict))

        return pd.DataFrame(wlist)
