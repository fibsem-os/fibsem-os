import logging
import os
import uuid
from abc import ABC
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type

import pandas as pd
import petname
import yaml
from psygnal import evented
from psygnal.containers import EventedDict, EventedList

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

from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.structures import (
    DEFAULT_ALIGNMENT_AREA,
    FibsemImage,
    FibsemRectangle,
    FibsemStagePosition,
    ImageSettings,
    MicroscopeState,
    Point,
    ReferenceImages,
    ReferenceImageParameters,
)
from fibsem.utils import configure_logging, format_duration




class AutoLamellaTaskStatus(Enum):
    NotStarted = auto()
    InProgress = auto()
    Completed = auto()
    Failed = auto()
    Skipped = auto()


@evented
@dataclass
class AutoLamellaTaskState:
    name: str = ""
    step: str = ""
    task_id: str = ""
    task_type: str = ""
    lamella_id: str = ""
    start_timestamp: float = field(default_factory=lambda: datetime.timestamp(datetime.now()))
    end_timestamp: Optional[float] = None
    status: AutoLamellaTaskStatus = AutoLamellaTaskStatus.NotStarted
    status_message: str = ""

    @property
    def completed(self) -> str:
        return f"{self.name} ({self.completed_at})"

    @property
    def completed_at(self) -> str:
        if self.end_timestamp is None:
            return "in progress"
        return datetime.fromtimestamp(self.end_timestamp).strftime('%I:%M%p')

    @property
    def started_at(self) -> str:
        return datetime.fromtimestamp(self.start_timestamp).strftime('%I:%M%p')

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
    def from_dict(cls, data: dict) -> 'AutoLamellaTaskState':
        """Create a task state from a dictionary."""
        if data is None:
            return cls()
        data = data.copy()
        data["status"] = AutoLamellaTaskStatus[data.get("status", "NotStarted")]
        return cls(**data)


@evented
@dataclass
class AutoLamellaTaskConfig(ABC):
    """Configuration for AutoLamella tasks."""
    task_type: ClassVar[str]
    display_name: ClassVar[str]
    related_tasks: ClassVar[list[type['AutoLamellaTaskConfig']]] = []
    task_name: str = "" # unique name for identifying in multi-task workflows
    milling: Dict[str, FibsemMillingTaskConfig] = field(default_factory=dict)
    reference_imaging: ReferenceImageParameters = field(default_factory=ReferenceImageParameters)

    @property
    def parameters(self) -> Tuple[str, ...]:
        core_params = [f.name for f in fields(AutoLamellaTaskConfig)]
        return tuple(
            f.name
            for f in fields(self)
            if f.name not in core_params
        )

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
    def from_dict(cls, ddict: Dict[str, Any]) -> 'AutoLamellaTaskConfig':
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
                k: FibsemMillingTaskConfig.from_dict(v) for k, v in ddict["milling"].items()
            }
        if "reference_imaging" in ddict:
            kwargs["reference_imaging"] = ReferenceImageParameters.from_dict(ddict["reference_imaging"])

        return cls(**kwargs)

    @property
    def estimated_time(self) -> float:
        """Estimate the total milling time for this task configuration."""
        total_time = 0.0
        for milling_task in self.milling.values():
            total_time += milling_task.estimated_time
        return total_time
    
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
    name: str # unique_name
    supervise: bool
    required: bool
    requires: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AutoLamellaTaskDescription':
        if data is None:
            return cls(name="", task_type="", supervise=False, required=False, requires=[])
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
    def from_dict(cls, data: Dict[str, Any]) -> 'AutoLamellaWorkflowConfig':
        data["tasks"] = [AutoLamellaTaskDescription.from_dict(task) for task in data.get("tasks", [])]
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

    def get_completed_tasks(self, lamella: 'Lamella', with_timestamps: bool = False) -> List[str]:
        """Get the list of completed tasks for a given lamella."""
        completed_tasks = []
        for task in lamella.task_history:
            if task.name in self.workflow:
                txt = task.name
                if with_timestamps:
                    txt = task.completed
                completed_tasks.append(txt)
        return completed_tasks

    def get_remaining_tasks(self, lamella: 'Lamella') -> List[str]:
        """Get the list of remaining tasks for a given lamella."""
        remaining_tasks = []
        completed_tasks = self.get_completed_tasks(lamella)
        for task in self.required_tasks:
            if task not in completed_tasks:
                remaining_tasks.append(task)
        return remaining_tasks

    def is_completed(self, lamella: 'Lamella') -> bool:
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

    def add_task(self, task: AutoLamellaTaskConfig) -> None:
        """Add a task to the workflow configuration."""
        self.tasks.append(AutoLamellaTaskDescription(name=task.task_name, 
                                                     supervise=True, 
                                                     required=True, 
                                                     requires=[]))

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
                    issues.append(f"Task '{task.name}' requires '{req}' which comes after it in the workflow.")
        return issues


@evented
@dataclass
class AutoLamellaWorkflowOptions:
    turn_beams_off: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AutoLamellaWorkflowOptions':
        return cls(**data)


@evented
@dataclass
class AutoLamellaTaskProtocol:
    name: str = "AutoLamella Task Protocol"
    description: str = "Protocol for AutoLamella"
    version: str = "1.0"
    task_config: EventedDict[str, AutoLamellaTaskConfig] = field(default_factory=lambda: EventedDict())   # unique_name: AutoLamellaTaskConfig
    workflow_config: AutoLamellaWorkflowConfig = field(default_factory=AutoLamellaWorkflowConfig)
    options: AutoLamellaWorkflowOptions = field(default_factory=AutoLamellaWorkflowOptions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "tasks": {k: v.to_dict() for k, v in self.task_config.items()},
            "workflow": self.workflow_config.to_dict(),
            "options": self.options.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AutoLamellaTaskProtocol':
        from fibsem.applications.autolamella.workflows.tasks.tasks import load_task_config
        task_config = load_task_config(data.get("tasks", {}))
        workflow_config = AutoLamellaWorkflowConfig.from_dict(data.get("workflow", {}))

        return cls(
            name=data.get("name", "AutoLamella Task Protocol"),
            description=data.get("description", "Protocol for AutoLamella"),
            version=data.get("version", "1.0"),
            task_config=task_config,
            workflow_config=workflow_config,
            options=AutoLamellaWorkflowOptions.from_dict(data.get("options", {}))
        )

    @classmethod
    def load(cls, filename: str) -> 'AutoLamellaTaskProtocol':
        with open(filename, 'r') as file:
            data = yaml.safe_load(file)
        return cls.from_dict(data)

    def save(self, filename: str) -> None:
        """Save the task protocol to a YAML file."""
        with open(filename, 'w') as file:
            yaml.safe_dump(self.to_dict(), 
                           file,
                           indent=4, 
                           default_flow_style=False, 
                           sort_keys=False)

    def get_supervision(self, task_name: str) -> bool:
        """Check if a task requires supervision."""
        return self.workflow_config.get_supervision(task_name)

    @classmethod
    def load_from_old_protocol(cls, path: Path) -> 'AutoLamellaTaskProtocol':
        """Convert an AutoLamellaProtocol to an AutoLamellaTaskProtocol.
        This involves mapping the milling configurations to the new task names.
        Used to converte old protocols to the new task-based protocol format."""

        from fibsem.applications.autolamella.protocol.legacy import (
            AutoLamellaMethod,
            AutoLamellaProtocol,
            AutoLamellaStage,
        )
        from fibsem.applications.autolamella.workflows.tasks.tasks import (
            MillPolishingTaskConfig,
            MillRoughTaskConfig,
            MillTrenchTaskConfig,
            MillUndercutTaskConfig,
            MillFiducialTaskConfig,
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
        
        if protocol.method not in [AutoLamellaMethod.ON_GRID, AutoLamellaMethod.TRENCH, AutoLamellaMethod.WAFFLE]:
            raise ValueError(f"Protocol method {protocol.method} not supported for conversion to task protocol")

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
                milling={MILL_ROUGH_KEY: FibsemMillingTaskConfig.from_stages(protocol.milling[MILL_ROUGH_KEY], name="Rough Milling")},
            )

            if protocol.options.use_microexpansion:
                rough_milling_task.milling[MILL_ROUGH_KEY].stages.extend(protocol.milling[MICROEXPANSION_KEY])
            if protocol.options.use_notch:
                rough_milling_task.milling[STRESS_RELIEF_KEY] = FibsemMillingTaskConfig.from_stages(protocol.milling[NOTCH_KEY], name="Notch")

            polishing_milling_task = MillPolishingTaskConfig(
                task_name=POLISHING_TASK_NAME,
                milling={MILL_POLISHING_KEY: FibsemMillingTaskConfig.from_stages(protocol.milling[MILL_POLISHING_KEY], name="Polishing")},
            )

            mill_fiducial_task = MillFiducialTaskConfig(
                task_name=MILL_FIDUCIAL_TASK_NAME,
                milling={FIDUCIAL_KEY: FibsemMillingTaskConfig.from_stages(protocol.milling[FIDUCIAL_KEY], name="Fiducial")},
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
                AutoLamellaTaskDescription(name=SETUP_LAMELLA_POSITION_TASK_NAME, supervise=protocol.supervision[AutoLamellaStage.SetupLamella], required=True),
                AutoLamellaTaskDescription(name=MILL_FIDUCIAL_TASK_NAME, supervise=protocol.supervision[AutoLamellaStage.SetupLamella], required=True),
                AutoLamellaTaskDescription(name=ROUGH_MILLING_TASK_NAME, supervise=protocol.supervision[AutoLamellaStage.MillRough], required=True, requires=[MILL_FIDUCIAL_TASK_NAME]),
                AutoLamellaTaskDescription(name=POLISHING_TASK_NAME, supervise=protocol.supervision[AutoLamellaStage.MillPolishing], required=True, requires=[ROUGH_MILLING_TASK_NAME]),
            ]


        if protocol.method in [AutoLamellaMethod.TRENCH, AutoLamellaMethod.WAFFLE]:
            trench_milling_task = MillTrenchTaskConfig(
                task_name=TRENCH_MILLING_TASK_NAME,
                milling={TRENCH_KEY: FibsemMillingTaskConfig.from_stages(protocol.milling[TRENCH_KEY], name="Trench"),
                },
                orientation="FIB",
            )
            task_config[TRENCH_MILLING_TASK_NAME] = trench_milling_task
            workflow_config.tasks.insert(0, AutoLamellaTaskDescription(name=TRENCH_MILLING_TASK_NAME,
                                                                    supervise=protocol.supervision[AutoLamellaStage.MillTrench], 
                                                                    required=True))

        if protocol.method is AutoLamellaMethod.WAFFLE:

            undercut_task = MillUndercutTaskConfig(
                task_name=UNDERCUT_TASK_NAME,
                milling={UNDERCUT_KEY: FibsemMillingTaskConfig.from_stages(protocol.milling[UNDERCUT_KEY], name="Undercut")},
                orientation="SEM",

            )
            task_config[UNDERCUT_TASK_NAME] = undercut_task
            workflow_config.tasks.insert(1, AutoLamellaTaskDescription(name=UNDERCUT_TASK_NAME,
                                                                    supervise=protocol.supervision[AutoLamellaStage.MillUndercut], 
                                                                    required=True, requires=[TRENCH_MILLING_TASK_NAME]))


        options = AutoLamellaWorkflowOptions(
            turn_beams_off=protocol.options.turn_beams_off,
        )

        workflow_config.name = protocol.name
        workflow_config.description = f"auto-converted protocol from {protocol.name} - {protocol.method.name}"

        task_protocol = AutoLamellaTaskProtocol(
            name=protocol.name,
            description=f"auto-converted protocol from {protocol.name} - {protocol.method.name}",
            task_config=task_config,
            workflow_config=workflow_config,
            options=options
        )

        return task_protocol

    def get_task_config_by_type(self, task_type: Type['AutoLamellaTaskConfig']) -> EventedDict[str, AutoLamellaTaskConfig]:
        """Get the task configuration by type."""
        task_configs = EventedDict()
        for k, v in self.task_config.items():
            if isinstance(v, task_type):
                task_configs[k] = v
        return task_configs


class DefectType(Enum):
    NONE = auto()
    FAILURE = auto()
    REWORK = auto()


@evented
@dataclass
class DefectState:
    has_defect: bool = False
    requires_rework: bool = False
    description: str = ""
    updated_at: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert the defect state to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'DefectState':
        """Create a defect state from a dictionary."""
        return cls(**data)

    def clear(self):
        self.has_defect = False
        self.description = ""
        self.requires_rework = False
        self.updated_at = None

    def set_defect(self, description: str = "", requires_rework: bool = False):
        self.has_defect = True
        self.description = description
        self.requires_rework = requires_rework
        self.updated_at = datetime.timestamp(datetime.now())


@evented
@dataclass
class Lamella:
    path: Path
    number: int                                                             # TODO: deprecate, use petname instead
    petname: str
    alignment_area: FibsemRectangle = field(default_factory=lambda: FibsemRectangle.from_dict(DEFAULT_ALIGNMENT_AREA))
    _id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_config: EventedDict[str, 'AutoLamellaTaskConfig'] = field(default_factory=lambda: EventedDict())
    poses: Dict[str, MicroscopeState] = field(default_factory=dict)
    task_state: AutoLamellaTaskState = field(default_factory=AutoLamellaTaskState)
    task_history: List['AutoLamellaTaskState'] = field(default_factory=list)
    defect: DefectState = field(default_factory=DefectState)
    objective_position: Optional[float] = None  # TODO: deprecate, use poses instead
    milling_angle: Optional[float] = None
    poi: Point = field(default_factory=lambda: Point(0,0))  # point of interest within lamella area (milling coordinate system)

    def __post_init__(self):
        # only make the dir, if the base path is actually set, 
        # prevents creating path on other computer..
        if os.path.exists(os.path.dirname(self.path)):
            os.makedirs(self.path, exist_ok=True)

        if self._id is None:
            self._id = str(uuid.uuid4())
        self.task_state.lamella_id = self._id

        # assign the imaging path to the task config
        for task_name, tc in self.task_config.items():
            for name, milling_task_config in tc.milling.items():
                milling_task_config.acquisition.imaging.path = self.path

    @property
    def name(self) -> str:
        return self.petname

    @name.setter
    def name(self, value: str):
        self.petname = value

    @property
    def is_failure(self) -> bool:
        return self.defect.has_defect

    @property
    def stage_position(self) -> FibsemStagePosition:
        return self.milling_pose.stage_position # type: ignore

    @stage_position.setter
    def stage_position(self, value: FibsemStagePosition):
        self.milling_pose.stage_position = value

    def has_completed_task(self, task_name: str) -> bool:
        """Check if the lamella has completed a specific task."""
        return task_name in self.completed_tasks

    @property
    def completed_tasks(self) -> List[str]:
        """Return a list of completed task names."""
        return [task.name for task in self.task_history]

    @property
    def last_completed_task(self) -> Optional['AutoLamellaTaskState']:
        """Return the last completed task state."""
        if self.task_history:
            return self.task_history[-1]
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
        return self.fluorescence_pose is not None and self.objective_position is not None

    def to_dict(self):
        return {
            "petname": self.petname,
            "path": str(self.path),
            "alignment_area": self.alignment_area.to_dict(),
            "number": self.number,
            "id": str(self._id),
            "poses": {k: v.to_dict() for k, v in self.poses.items()},
            "task_config": {k: v.to_dict() for k, v in self.task_config.items()},
            "task_state": self.task_state.to_dict(),
            "task_history": [task.to_dict() for task in self.task_history],
            "defect": self.defect.to_dict(),
            "objective_position": self.objective_position,
            "milling_angle": self.milling_angle,
            "poi": self.poi.to_dict(),
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
        if self.objective_position is None:
            objective_str = "N/A"
        else:
            objective_str = f"{self.objective_position * 1e3:.3f}mm"

        return f"{self.name} ({self.stage_position.x * 1e6:.1f}μm, {self.stage_position.y * 1e6:.1f}μm, {objective_str})"

    @classmethod
    def from_dict(cls, data: dict) -> 'Lamella':
        # backwards compatibility
        alignment_area_ddict = data.get("alignment_area", DEFAULT_ALIGNMENT_AREA)
        alignment_area = FibsemRectangle.from_dict(alignment_area_ddict)

        from fibsem.applications.autolamella.workflows.tasks.tasks import load_task_config

        return cls(
            petname=data["petname"],
            path=data["path"],
            alignment_area=alignment_area,
            number=data.get("number", data.get("number", 0)),
            _id=data.get("id", ""),
            poses = {k: MicroscopeState.from_dict(v) for k, v in data.get("poses", {}).items()},
            task_config=load_task_config(data.get("task_config", {})),
            task_state=AutoLamellaTaskState.from_dict(data.get("task_state", {})),
            task_history=[AutoLamellaTaskState.from_dict(task) for task in data.get("task_history", [])],
            defect=DefectState.from_dict(data.get("defect", {})),
            objective_position=data.get("objective_position", None),
            milling_angle=data.get("milling_angle", None),
            poi=Point.from_dict(data.get("poi", {"x":0,"y":0})),
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
        thumb_path = os.path.join(self.path, "thumbnail.png")
        import numpy as np
        if not os.path.exists(thumb_path):
            return np.zeros((170, 256, 3), dtype=np.uint8)
        from PIL import Image
        return np.asarray(Image.open(thumb_path).convert("RGB"))

    def save_thumbnail(self, image: "FibsemImage") -> None:
        """Save a thumbnail of the given image to disk as thumbnail.png."""
        from PIL import Image
        import numpy as np
        data = image.data
        if data.ndim == 2:
            data = np.stack([data, data, data], axis=2)
        Image.fromarray(data.astype(np.uint8)).save(os.path.join(self.path, "thumbnail.png"))

    # convert to method
    def get_reference_images(self, filename: str) -> ReferenceImages:
        reference_images = ReferenceImages(
            low_res_eb=self.load_reference_image(f"{filename}_low_res_eb"),
            high_res_eb=self.load_reference_image(f"{filename}_high_res_eb"),
            low_res_ib=self.load_reference_image(f"{filename}_low_res_ib"),
            high_res_ib=self.load_reference_image(f"{filename}_high_res_ib"),
        )

        return reference_images

    # def get_task_config_by_type(self, task_type: Type['AutoLamellaTaskConfig']) -> Dict[str, AutoLamellaTaskConfig]:
    #     """Get the task configuration by type."""
    #     task_configs = {}
    #     for k, v in self.task_config.items():
    #         if isinstance(v, task_type):
    #             task_configs[k] = v
    #     return task_configs


@evented
@dataclass
class Experiment:
    name: str
    _id: str
    path: Path
    positions: EventedList[Lamella] = field(default_factory=EventedList)
    landing_positions: List[FibsemStagePosition] = field(default_factory=list)
    created_at: float = field(default_factory=lambda: datetime.timestamp(datetime.now()))
    task_protocol: 'AutoLamellaTaskProtocol' = field(default_factory=lambda: AutoLamellaTaskProtocol())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, path: Path,
                 name: str = cfg.EXPERIMENT_NAME,
                 metadata: Optional[Dict[str, Any]] = None) -> None:
        """Create a new experiment.

        Args:
            path: The path where the experiment will be created.
            name: The name of the experiment. Defaults to cfg.EXPERIMENT_NAME.
            metadata: Optional dictionary containing experiment metadata (e.g., description, user, project, organisation).
        """
        self.name: str = name
        self._id = str(uuid.uuid4())
        self.path: Path = os.path.join(path, name)
        self.created_at: float = datetime.timestamp(datetime.now())

        self.positions: EventedList[Lamella] = EventedList()
        self.landing_positions: List[FibsemStagePosition] = []

        self.task_protocol: AutoLamellaTaskProtocol = None # must be set externally
        self.metadata: Dict[str, Any] = metadata if metadata is not None else {}

    def to_dict(self) -> dict:

        state_dict = {
            "name": self.name,
            "_id": self._id,
            "path": self.path,
            "positions": [deepcopy(lamella.to_dict()) for lamella in self.positions],
            "landing_positions": [pos.to_dict() for pos in self.landing_positions],
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

        return state_dict

    @classmethod
    def from_dict(cls, ddict: dict) -> 'Experiment':

        path = os.path.dirname(ddict["path"])
        name = ddict["name"]
        experiment = Experiment(path=path, name=name)
        experiment.created_at = ddict.get("created_at", None)
        experiment._id = ddict.get("_id", "NULL")

        experiment.metadata = ddict.get("metadata", {})

        # load lamella from dict
        for lamella_dict in ddict["positions"]:
            lamella = Lamella.from_dict(data=lamella_dict)
            experiment.positions.append(lamella)

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

    def save(self) -> None:
        """Save the sample data to yaml file"""

        with open(os.path.join(self.path, "experiment.yaml"), "w") as f:
            yaml.safe_dump(self.to_dict(), f, indent=4)

    def __repr__(self) -> str:

        return f"""Experiment: 
        Path: {self.path}
        Positions: {len(self.positions)}
        """

    @staticmethod
    def load(fname: Path) -> 'Experiment':
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

        # configure experiment logging
        configure_logging(path=experiment.path, log_filename="logfile")

        # attempt to load task protocol from the same directory
        protocol_path = os.path.join(experiment.path, "protocol.yaml")
        if os.path.exists(protocol_path):
            try:
                experiment.task_protocol = AutoLamellaTaskProtocol.load(protocol_path)
                logging.info(f"Loaded task protocol from {protocol_path}")
            except Exception as e:
                logging.warning(f"Failed to load task protocol from {protocol_path}: {e}")

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
                        existing_milling_config = existing_config.milling.get(milling_name)
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

                            if (
                                type(existing_stage.pattern) is type(new_stage.pattern)
                                and hasattr(existing_stage.pattern, "point")
                            ):
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
        if update_base_protocol and self.task_protocol is not None and source_lamella_name is not None:
            for task_name in task_names:
                if task_name in source_task_config:
                    self.task_protocol.task_config[task_name] = deepcopy(
                        source_task_config[task_name]
                    )
            logging.info(f"Updated base protocol tasks: {task_names}")

        return updated_count

    def at_failure(self) -> List[Lamella]:
        """Return a list of lamellas that have failed"""
        return [lamella for lamella in self.positions if lamella.defect.has_defect]

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

    def estimate_remaining_time(self) -> float:
        """Estimate the remaining time for all lamellas in the experiment"""
        ESTIMATED_SETUP_TIME = 5*60         # 5min
        OVERHEAD_TIME = 2*60                # 2min
        total_remaining_time: float = 0.0
        for p in self.positions:

            # skip failed lamellas
            if p.defect.has_defect:
                continue

            # remaining time for individual lamella
            remaining_tasks = self.task_protocol.workflow_config.get_remaining_tasks(p)
            remaining_time: float = 0
            for rt in remaining_tasks:
                estimated_milling_time = p.task_config[rt].estimated_time
                remaining_time += estimated_milling_time + OVERHEAD_TIME
            logging.debug(f"Total estimated time: {format_duration(remaining_time)}")

            total_remaining_time += remaining_time
        return total_remaining_time

    def add_lamella(self, lamella: Lamella) -> None:
        """Add a lamella to the experiment."""
        if not isinstance(lamella, Lamella):
            raise TypeError("lamella must be an instance of Lamella")

        # check if lamella already exists
        if lamella in self.positions:
            raise ValueError(f"Lamella {lamella.name} already exists in the experiment.")

        self.positions.append(deepcopy(lamella))
        logging.info(f"Added lamella {lamella.name} to experiment {self.name}")

    @classmethod
    def create(cls, path: Path, name: str = cfg.EXPERIMENT_NAME, metadata: Optional[Dict[str, Any]] = None) -> 'Experiment':
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

        # configure experiment logging
        os.makedirs(experiment.path, exist_ok=True)
        configure_logging(path=experiment.path, log_filename="logfile")

        # save the experiment
        experiment.save()

        logging.info(f"Created new experiment {experiment.name} at {experiment.path}")

        return experiment

    def save_protocol(self) -> None:
        """Save the task protocol to disk in the experiment directory."""
        self.task_protocol.save(os.path.join(self.path, "protocol.yaml"))

###### TASK REFACTORING ##########

    def add_new_lamella(self,
                        microscope_state: MicroscopeState,
                        task_config: EventedDict[str, AutoLamellaTaskConfig],
                        name: Optional[str] = None) -> None:
        """Create a new lamella and add it to the experiment."""
        # create the petname and path
        number = len(self.positions) + 1
        if name is None:
            name = f"{number:02d}-{petname.generate(2)}"
        path = Path(os.path.join(self.path, name))

        # create the lamella
        lamella = Lamella(petname=name,
                          path=path,
                          number=number,
                          task_config=deepcopy(task_config))
        lamella.milling_pose = microscope_state

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
                "experiment_id": self._id,
                "lamella_name": p.name,
                "lamella_id": p._id,
                "last_completed": p.last_completed_task.completed if p.last_completed_task else None,
                "last_completed_task": p.last_completed_task.name if p.last_completed_task else None,
                "last_completed_at": p.last_completed_task.completed_at if p.last_completed_task else None,
                "is_completed": self.task_protocol.workflow_config.is_completed(p),
                "is_failure": p.defect.has_defect,
                "milling_angle": p.milling_angle,
            }
            edict.append(deepcopy(ddict))

        df = pd.DataFrame(edict)

        return df
    
    def workflow_dataframe(self) -> pd.DataFrame:
        """Create a dataframe with the workflow """
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
