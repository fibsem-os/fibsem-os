from __future__ import annotations

import logging
import os
import uuid
from abc import ABC
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from enum import Enum
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
from psygnal import evented
from psygnal.containers import EventedDict, EventedList

# Registers AutoLamella's proposal kinds and how their values are stored and
# written: an experiment read from disk decodes its proposals with them.
import fibsem.applications.autolamella.proposals  # noqa: F401
from fibsem import timing
from fibsem.applications.autolamella import config as cfg
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
    Configurable,
    Display,
    FibsemImage,
    FibsemRectangle,
    FibsemStagePosition,
    ImageSettings,
    MicroscopeState,
    Parameter,
    Point,
    ReferenceImageParameters,
    SessionInfo,
    field_specs,
    param,
)
from fibsem.util.durations import format_duration_precise
from fibsem.util.timestamps import format_time, now
from fibsem.workflows.attention import Attention, attention_from
from fibsem.workflows.deciding import ProposalDecisions
from fibsem.workflows.experiment import BaseExperiment
from fibsem.workflows.grid.protocol import GridTaskProtocol
from fibsem.workflows.grids import ExperimentGrids, GridRecord
from fibsem.workflows.item import ExperimentItem
from fibsem.workflows.proposals import (
    Proposal,
    proposals_from_dict,
    proposals_to_dict,
)
from fibsem.workflows.quality import QualityRecord, Verdict
from fibsem.workflows.saved_format import check_app
from fibsem.workflows.status import TaskState, TaskStatus

if TYPE_CHECKING:
    import numpy as np

    from fibsem.microscope import FibsemMicroscope


# AutoLamellaUser lived here: a richer user identity (role, preferences, is_default)
# with a to_fibsem_user() bridge into image metadata. Nothing ever constructed one --
# not the app, not a script, not a test -- so the bridge was never called and the
# extra fields never had a reader. Removed rather than left dormant, because half a
# user model is worse than either answer. fibsem.structures.FibsemUser is the only
# user model now; see FIB-450 and the ownership rule on FibsemImageMetadata.


@evented
@dataclass
class AutoLamellaTaskConfig(Configurable, ABC):
    """Configuration for AutoLamella tasks."""

    task_type: ClassVar[str]
    display_name: ClassVar[str]
    related_tasks: ClassVar[list[type["AutoLamellaTaskConfig"]]] = []
    # Parameters this task type used to have. Saved protocols and experiments
    # still carry them; they are dropped on load without the "Unknown parameter"
    # warning, which is for keys nobody recognises.
    retired_parameters: ClassVar[frozenset] = frozenset()
    task_name: str = param(
        "",
        doc="The task's name, unique within the protocol: the key the workflow knows it by.",
    )  # unique name for identifying in multi-task workflows
    milling: Dict[str, FibsemMillingTaskConfig] = param(
        doc="The task's milling, by key (trench, rough, polishing, ...): each one's patterns, strategy and alignment.",
        default_factory=dict,
    )
    reference_imaging: ReferenceImageParameters = param(
        doc="The reference images the task acquires when it finishes.",
        default_factory=ReferenceImageParameters,
    )

    @property
    def parameters(self) -> Tuple[str, ...]:
        core_params = [f.name for f in fields(AutoLamellaTaskConfig)]
        return tuple(f.name for f in fields(self) if f.name not in core_params)

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
        params = ddict.get("parameters") or {}
        cls._warn_unknown_parameters(params)
        known = {f.name for f in fields(cls)}
        kwargs.update({k: v for k, v in params.items() if k in known})

        kwargs.update(cls._load_core(ddict))
        return cls(**kwargs)

    @classmethod
    def _load_core(cls, ddict: Dict[str, Any]) -> Dict[str, Any]:
        """The fields every task shares, as constructor kwargs; ``parameters`` is untouched.

        For a subclass that reads its own parameters: calling the base ``from_dict``
        instead would check them against the base's fields and warn for each one.
        """
        kwargs: Dict[str, Any] = {}
        if "task_name" in ddict:
            kwargs["task_name"] = ddict["task_name"]
        if "milling" in ddict:
            kwargs["milling"] = {
                k: FibsemMillingTaskConfig.from_dict(v)
                for k, v in ddict["milling"].items()
            }
        if "reference_imaging" in ddict:
            kwargs["reference_imaging"] = ReferenceImageParameters.from_dict(
                ddict["reference_imaging"]
            )
        return kwargs

    @classmethod
    def _warn_unknown_parameters(cls, params: Dict[str, Any]) -> None:
        """Warn for each key in ``params`` that is not a field of this task config."""
        known = {f.name for f in fields(cls)} | cls.retired_parameters
        for key in params:
            if key not in known:
                logging.warning(
                    f"Unknown parameter '{key}' in "
                    f"{getattr(cls, 'task_type', cls.__name__)} task configuration."
                )

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
    required: bool = True
    # Whose decision this task's record carries, and when (see Attention).
    # Whether a task records a proposal at all is a property of the task kind,
    # in code; it records in every mode.
    attention: Attention = Attention.automated
    requires: List[str] = field(default_factory=list)
    scheduled_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        self.attention = Attention(self.attention)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["attention"] = self.attention.value
        if d.get("scheduled_at") is not None:
            d["scheduled_at"] = self.scheduled_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AutoLamellaTaskDescription":
        if data is None:
            return cls(name="", required=False)
        data = dict(data)
        if "attention" not in data:
            # v0.5.2 and earlier wrote a supervise flag in place of attention,
            # and every protocol and experiment saved by one of them still
            # carries it. That form shipped, so this mapping stays; the review
            # flag and the interim mode strings beside it never did.
            data["attention"] = attention_from(bool(data.pop("supervise", False)))
        else:
            # Through the one reader, so a stored value this build spells
            # differently -- or does not know -- loads instead of raising.
            data["attention"] = attention_from(
                data["attention"], f"task {data.get('name', '')!r}"
            )
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
            if task.status is not TaskStatus.Completed:
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

    def get_attention(self, task_name: str) -> Attention:
        """Who decides the task's record, and when; automated for a task the
        workflow does not list."""
        for task in self.tasks:
            if task.name == task_name:
                return task.attention
        return Attention.automated

    def get_scheduled_at(self, task_name: str) -> Optional[datetime]:
        """Get the scheduled start time for a task, or None if not scheduled."""
        for task in self.tasks:
            if task.name == task_name:
                return task.scheduled_at
        return None

    def reorder(self, order: List[str]) -> str:
        """Put the tasks in *order*: every task, each once, none before a task
        it requires. Why not, with nothing changed, or "". The Workflow tab's
        drag and the App Server's reorder."""
        if sorted(order) != sorted(self.workflow):
            return "Give every task in the workflow, each once."
        by_name = {task.name: task for task in self.tasks}
        reordered = [by_name[name] for name in order]
        issues = type(self)(tasks=reordered).validate()
        if issues:
            return " ".join(issues)
        self.tasks[:] = reordered
        return ""

    def set_requirements(
        self,
        task_name: str,
        requires: Optional[List[str]] = None,
        required: Optional[bool] = None,
    ) -> str:
        """Set the tasks *task_name* requires, which must come before it, and
        whether it is required; None leaves either as it is. Why not, with
        nothing changed, or "". The Workflow tab's Edit Task and the App
        Server's."""
        task = next((t for t in self.tasks if t.name == task_name), None)
        if task is None:
            return f"No task named {task_name!r} in the workflow."
        if requires is not None:
            earlier = self.workflow[: self.workflow.index(task_name)]
            wrong = [name for name in requires if name not in earlier]
            if wrong:
                return (
                    f"{task_name} can only require tasks before it: "
                    f"{', '.join(earlier) or 'none'}; not {', '.join(wrong)}."
                )
            task.requires = list(requires)
        if required is not None:
            task.required = bool(required)
        return ""

    def add_task(self, task: AutoLamellaTaskConfig) -> None:
        """Add a task to the workflow configuration."""
        self.tasks.append(
            AutoLamellaTaskDescription(
                name=task.task_name, required=True, attention=Attention.supervised
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
    # Open the run summary dialog when a run ends. The run-end toast, with its
    # Show summary link, comes either way; off is for runs nobody is watching
    # the end of -- an agent starting many short runs, or overnight (FIB-1376).
    show_run_summary: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AutoLamellaWorkflowOptions":
        data = dict(data)
        # How long a run waited on reviews, before nothing waited on one.
        data.pop("review_wait", None)
        return cls(**data)


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
        with open(filename) as file:
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

    def get_attention(self, task_name: str) -> Attention:
        return self.workflow_config.get_attention(task_name)

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
                    attention=attention_from(
                        protocol.supervision[AutoLamellaStage.SetupLamella]
                    ),
                    required=True,
                ),
                AutoLamellaTaskDescription(
                    name=MILL_FIDUCIAL_TASK_NAME,
                    attention=attention_from(
                        protocol.supervision[AutoLamellaStage.SetupLamella]
                    ),
                    required=True,
                ),
                AutoLamellaTaskDescription(
                    name=ROUGH_MILLING_TASK_NAME,
                    attention=attention_from(
                        protocol.supervision[AutoLamellaStage.MillRough]
                    ),
                    required=True,
                    requires=[MILL_FIDUCIAL_TASK_NAME],
                ),
                AutoLamellaTaskDescription(
                    name=POLISHING_TASK_NAME,
                    attention=attention_from(
                        protocol.supervision[AutoLamellaStage.MillPolishing]
                    ),
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
                    attention=attention_from(
                        protocol.supervision[AutoLamellaStage.MillTrench]
                    ),
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
                    attention=attention_from(
                        protocol.supervision[AutoLamellaStage.MillUndercut]
                    ),
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
class Lamella(ExperimentItem):
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

    @property
    def last_completed_task(self) -> Optional["TaskState"]:
        """Return the last completed task state.

        The last *completed* one, not the last recorded: a failure landing last
        would otherwise be reported as the lamella's latest progress in the
        experiment summary.
        """
        for task in reversed(self.task_history):
            if task.status is TaskStatus.Completed:
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
            task_state=TaskState.from_dict(data.get("task_state", {})),
            task_history=[
                TaskState.from_dict(task) for task in data.get("task_history", [])
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
    landing_positions: List[FibsemStagePosition] = field(default_factory=list)
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
        self.landing_positions: List[FibsemStagePosition] = []

        self.task_protocol: AutoLamellaTaskProtocol = None  # must be set externally

    def to_dict(self, include_protocol: bool = False) -> dict:

        state_dict = {
            **super().to_dict(),
            "positions": [deepcopy(lamella.to_dict()) for lamella in self.positions],
            "grids": [grid.to_dict() for grid in self.grids],
            "landing_positions": [pos.to_dict() for pos in self.landing_positions],
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
        experiment._read_identity(ddict)

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

    def get_lamella_by_id(self, lamella_id: str) -> Optional["Lamella"]:
        for lamella in self.positions:
            if lamella.id == lamella_id:
                return lamella
        return None

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
                    self.task_protocol.task_config[task_name] = deepcopy(
                        source_task_config[task_name]
                    )
            logging.info(f"Updated base protocol tasks: {task_names}")

        return updated_count

    def add_task(self, task_type: str, task_name: str) -> AutoLamellaTaskConfig:
        """Add a new task of a registered *task_type*, named *task_name*: its
        default config in the protocol, at the end of the workflow (required,
        supervised), and a copy on every lamella. The protocol editor's Add
        Task, and an agent's. Raises ValueError for an unknown type or a name
        the protocol already has."""
        from fibsem.applications.autolamella.workflows.tasks import get_tasks

        task_name = task_name.strip()
        registry = get_tasks()
        if task_type not in registry:
            raise ValueError(f"No task type {task_type!r}.")
        if not task_name:
            raise ValueError("A task needs a name.")
        if task_name in self.task_protocol.task_config:
            raise ValueError(f"There is already a task named {task_name!r}.")
        config = registry[task_type].config_cls()  # type: ignore[attr-defined]
        config.task_name = task_name
        self.task_protocol.task_config[task_name] = config
        self.task_protocol.workflow_config.add_task(config)
        for lamella in self.positions:
            lamella.task_config[task_name] = deepcopy(config)
            lamella._sync_imaging_paths()
        logging.info(f"Added task {task_name} ({task_type}) to experiment {self.name}")
        return config

    def remove_task(self, task_name: str) -> None:
        """Remove a task from the experiment: from the protocol, from the
        workflow, and from every lamella's config (FIB-1109).

        All three together, so the workflow never lists a task the protocol no
        longer has, and no lamella runs a stale copy of one. A task that listed
        it as a requirement no longer does. Task history is left alone: it is
        the record of what ran.
        """
        self.task_protocol.task_config.pop(task_name, None)

        workflow = self.task_protocol.workflow_config
        workflow.tasks = [t for t in workflow.tasks if t.name != task_name]
        for task in workflow.tasks:
            if task_name in task.requires:
                task.requires = [r for r in task.requires if r != task_name]

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
        for i, t in enumerate(self.task_protocol.workflow_config.tasks, 1):
            ddict = {
                "order": i,
                "task_name": t.name,
                "required": t.required,
                "attention": t.attention.value,
            }
            wlist.append(deepcopy(ddict))

        return pd.DataFrame(wlist)
