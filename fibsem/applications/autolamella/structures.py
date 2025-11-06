import logging
import os
import uuid
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import pandas as pd
import petname
import yaml
from psygnal import evented
from psygnal.containers import EventedList

from fibsem.applications.autolamella import config as cfg
from fibsem.applications.autolamella.protocol.validation import (
    LANDING_KEY,
    LIFTOUT_KEY,
    MILL_POLISHING_KEY,
    MILL_ROUGH_KEY,
    SETUP_LAMELLA_KEY,
    TRENCH_KEY,
    UNDERCUT_KEY,
)
from fibsem.milling import (
    FibsemMillingStage,
    estimate_total_milling_time,
    get_milling_stages,
    get_protocol_from_stages,
)
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.structures import (
    FibsemImage,
    FibsemRectangle,
    FibsemStagePosition,
    ImageSettings,
    MicroscopeSettings,
    MicroscopeState,
    ReferenceImages,
)
from fibsem.utils import configure_logging, format_duration


class AutoLamellaStage(Enum):
    Created = auto()
    PositionReady = auto()
    MillTrench = auto()
    MillUndercut = auto()
    LiftoutLamella = auto()
    LandLamella = auto()
    SetupLamella = auto()
    MillRough = auto()
    SetupPolishing = auto()
    MillPolishing = auto()
    Finished = auto()

    def __str__(self) -> str:
        return self.name

@dataclass
class LamellaState:
    microscope_state: MicroscopeState = field(default_factory=MicroscopeState)
    stage: AutoLamellaStage = AutoLamellaStage.Created
    start_timestamp: float = datetime.timestamp(datetime.now())
    end_timestamp: Optional[float] = None

    @property
    def completed(self) -> str:
        return f"{self.stage.name} ({self.completed_at})"

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

    def to_dict(self):
        return {
            "microscope_state": self.microscope_state.to_dict() if self.microscope_state is not None else "not defined",
            "stage": self.stage.name,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
        }

    @classmethod
    def from_dict(cls, data):
        state = MicroscopeState.from_dict(data["microscope_state"])
        
        # backwards compatibility
        stage_name = data["stage"]
        stage_name= stage_name.replace("Cut", "") # MillRoughCut -> MillRough, MillPolishingCut -> MillPolishing
        if stage_name in ["PreSetupLamella", "SetupTrench"]:
            stage_name = "Created"
        if stage_name == "ReadyTrench":
            stage_name = "PositionReady"
        if stage_name == "ReadyLamella":
            stage_name = "SetupLamella"
        workflow_stage = AutoLamellaStage[stage_name]

        return cls(
            microscope_state=state,
            stage=workflow_stage,
            start_timestamp=data["start_timestamp"],
            end_timestamp=data["end_timestamp"]
        )






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
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'AutoLamellaTaskState':
        """Create a task state from a dictionary."""
        if data is None:
            return cls()
        return cls(**data)

@dataclass
class AutoLamellaTaskConfig(ABC):
    """Configuration for AutoLamella tasks."""
    task_type: ClassVar[str]
    display_name: ClassVar[str]
    task_name: str = "" # unique name for identifying in multi-task workflows
    imaging: ImageSettings = field(default_factory=ImageSettings)
    milling: Dict[str, FibsemMillingTaskConfig] = field(default_factory=dict)

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
        ddict["imaging"] = self.imaging.to_dict()
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
                    # QUERY: should we raise an error here or just ignore unknown parameters?
                    raise ValueError(f"Unknown parameter '{key}' in task configuration.")

        if "imaging" in ddict:
            kwargs["imaging"] = ImageSettings.from_dict(ddict["imaging"])
        if "milling" in ddict:
            kwargs["milling"] = {
                k: FibsemMillingTaskConfig.from_dict(v) for k, v in ddict["milling"].items()
            }

        return cls(**kwargs)

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

    def requires(self, task_name: str) -> List[str]:
        for task in self.tasks:
            if task.name == task_name:
                return task.requires
        return []

    def get_completed_tasks(self, lamella: 'Lamella') -> List[str]:
        """Get the list of completed tasks for a given lamella."""
        completed_tasks = []
        for task in lamella.task_history:
            if task.name in self.workflow:
                completed_tasks.append(task.name)
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

@evented
@dataclass
class AutoLamellaTaskProtocol:
    name: str = "AutoLamella Task Protocol"
    description: str = "Protocol for AutoLamella"
    version: str = "1.0"
    task_config: Dict[str, AutoLamellaTaskConfig] = field(default_factory=dict)   # unique_name: AutoLamellaTaskConfig
    workflow_config: AutoLamellaWorkflowConfig = field(default_factory=AutoLamellaWorkflowConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "tasks": {k: v.to_dict() for k, v in self.task_config.items()},
            "workflow": self.workflow_config.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AutoLamellaTaskProtocol':
        from fibsem.applications.autolamella.workflows.tasks import load_task_config
        task_config = load_task_config(data.get("tasks", {}))
        workflow_config = AutoLamellaWorkflowConfig.from_dict(data.get("workflow", {}))
        
        return cls(
            name=data.get("name", "AutoLamella Task Protocol"),
            description=data.get("description", "Protocol for AutoLamella"),
            version=data.get("version", "1.0"),
            task_config=task_config, 
            workflow_config=workflow_config)

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


@dataclass
class DefectState:
    has_defect: bool = False
    description: str = ""
    updated_at: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert the defect state to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'DefectState':
        """Create a defect state from a dictionary."""
        return cls(**data)

@evented
@dataclass
class Lamella:
    path: Path
    state: LamellaState                                                     # TODO: deprecate, use poses instead
    number: int                                                             # TODO: deprecate, use petname instead
    petname: str
    protocol: dict                                                          # TODO: deprecate, use task_config instead
    is_failure: bool = False                                                # TODO: deprecate, use DefectState instead
    failure_note: str = ""                                                  # TODO: deprecate, use DefectState instead
    failure_timestamp: float = None                                         # TODO: deprecate, use DefectState instead
    alignment_area: FibsemRectangle = field(default_factory=FibsemRectangle)
    history: List[LamellaState] = field(default_factory=list)                                      # TODO: deprecate, use task_history instead
    milling_workflows: Dict[str, List[FibsemMillingStage]] = field(default_factory=dict)           # TODO: deprecate, use task_config instead
    states: Dict[AutoLamellaStage, LamellaState] = field(default_factory=dict)                     # TODO: deprecate, use task_history instead
    _id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_config: Dict[str, 'AutoLamellaTaskConfig'] = field(default_factory=dict)
    poses: Dict[str, MicroscopeState] = field(default_factory=dict)
    task_state: AutoLamellaTaskState = field(default_factory=AutoLamellaTaskState)
    task_history: List['AutoLamellaTaskState'] = field(default_factory=list)
    defect: DefectState = field(default_factory=DefectState)
    objective_position: Optional[float] = None  # TODO: deprecate, use poses instead
    milling_angle: Optional[float] = None

    def __post_init__(self):
        # only make the dir, if the base path is actually set, 
        # prevents creating path on other computer..
        if os.path.exists(os.path.dirname(self.path)):
            os.makedirs(self.path, exist_ok=True)

        if self.protocol is None:
            self.protocol = {}

        # set imaging paths to be the lamella path unless already set
        for k, v in self.protocol.items():  # Dict[str, List[Dict]]
            for stage in v:
                if stage.get("imaging", {}).get("path", None) is None:
                    stage["imaging"]["path"] = self.path

        if self.milling_workflows is None:
            self.milling_workflows = {k: get_milling_stages(k, self.protocol) for k in self.protocol}
        if self._id is None:
            self._id = str(uuid.uuid4())
        self.task_state.lamella_id = self._id

        # assign the imaging path to the task config
        for task_name, tc in self.task_config.items():
            for name, milling_task_config in tc.milling.items():
                milling_task_config.acquisition.imaging.path = self.path

    @property
    def finished(self) -> bool:
        return self.state.stage == AutoLamellaStage.Finished

    @property
    def workflow_stages_completed(self) -> List[AutoLamellaStage]:
        return list(self.states.keys())

    @property
    def name(self) -> str:
        return self.petname

    @name.setter
    def name(self, value: str):
        self.petname = value

    @property
    def status(self) -> str:
        return self.state.stage.name

    @property
    def workflow(self) -> AutoLamellaStage:
        return self.state.stage

    @property
    def last_completed(self) -> str:
        return self.state.completed

    @property
    def is_active(self) -> bool:
        return not self.finished and not self.is_failure

    @property
    def stage_position(self) -> FibsemStagePosition:
        return self.state.microscope_state.stage_position # type: ignore

    @stage_position.setter
    def stage_position(self, value: FibsemStagePosition):
        self.state.microscope_state.stage_position = value

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

    def to_dict(self):
        return {
            "petname": self.petname,
            "state": self.state.to_dict() if self.state is not None else None,
            "path": str(self.path),
            "alignment_area": self.alignment_area.to_dict(),
            "protocol": self.protocol,
            "number": self.number,
            "history": [state.to_dict() for state in self.history],
            "is_failure": self.is_failure,
            "failure_note": self.failure_note,
            "failure_timestamp": self.failure_timestamp,
            "id": str(self._id),
            "states": {k.name: v.to_dict() for k, v in self.states.items()},
            "poses": {k: v.to_dict() for k, v in self.poses.items()},
            "task_config": {k: v.to_dict() for k, v in self.task_config.items()},
            "task_state": self.task_state.to_dict() if self.task_state is not None else None,
            "task_history": [task.to_dict() for task in self.task_history],
            "defect": self.defect.to_dict() if self.defect is not None else None,
            "objective_position": self.objective_position,
            "milling_angle": self.milling_angle,
        }

    @property
    def info(self):
        return f"Lamella {self.petname} [{self.status}]"

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
        state = LamellaState.from_dict(data["state"])

        # backwards compatibility
        alignment_area_ddict = data.get("fiducial_area", data.get("alingment_area", None))
        if alignment_area_ddict is not None:
            alignment_area = FibsemRectangle.from_dict(alignment_area_ddict)
        else:
            alignment_area = FibsemRectangle() # use default

        history=[LamellaState().from_dict(state) for state in data["history"]]

        # load states:
        states = data.get("states", {})
        if states:
            states = {AutoLamellaStage[k]: LamellaState.from_dict(v) for k, v in states.items()}
        else:
            # load from history
            states = {state.stage: state for state in history}


        # protocol backwards compatibility
        protocol = data.get("protocol", {})

        # convert old milling protocol to new protocol
        if "lamella" in protocol or "MillRoughCut" in protocol:
            from fibsem.applications.autolamella.protocol.validation import (
                convert_old_milling_protocol_to_new_protocol,
            )
            nprotocol = convert_old_milling_protocol_to_new_protocol(protocol)
            if "MillRoughCut" in nprotocol:
                nprotocol[MILL_ROUGH_KEY] = nprotocol.pop("MillRoughCut")

            if "MillPolishingCut" in nprotocol:
                nprotocol[MILL_POLISHING_KEY] = nprotocol.pop("MillPolishingCut")

            if "lamella" in nprotocol:
                del nprotocol["lamella"]

            protocol = deepcopy(nprotocol)

        from fibsem.applications.autolamella.workflows.tasks.tasks import load_task_config

        return cls(
            petname=data["petname"],
            state=state,
            path=data["path"],
            alignment_area=alignment_area,
            protocol=protocol,
            number=data.get("number", data.get("number", 0)),
            history=history,
            is_failure=data.get("is_failure", data.get("is_failure", False)),
            failure_note=data.get("failure_note", ""),
            failure_timestamp=data.get("failure_timestamp", None),
            _id=data.get("id", None),
            states=states,
            poses = {k: MicroscopeState.from_dict(v) for k, v in data.get("poses", {}).items()},
            task_config=load_task_config(data.get("task_config", {})),
            task_state=AutoLamellaTaskState.from_dict(data.get("task_state", {})),
            task_history=[AutoLamellaTaskState.from_dict(task) for task in data.get("task_history", [])],
            defect=DefectState.from_dict(data.get("defect", {})),
            objective_position=data.get("objective_position", None),
            milling_angle=data.get("milling_angle", None),
        )

    def load_reference_image(self, fname) -> FibsemImage:
        """Load a specific reference image for this lamella from disk
        Args:
            fname: str
                the filename of the reference image to load
        Returns:
            adorned_img: AdornedImage
                the reference image loaded as an AdornedImage
        """

        adorned_img = FibsemImage.load(os.path.join(self.path, f"{fname}.tif"))

        return adorned_img

    # convert to method
    def get_reference_images(self, filename: str) -> ReferenceImages:
        reference_images = ReferenceImages(
            low_res_eb=self.load_reference_image(f"{filename}_low_res_eb"),
            high_res_eb=self.load_reference_image(f"{filename}_high_res_eb"),
            low_res_ib=self.load_reference_image(f"{filename}_low_res_ib"),
            high_res_ib=self.load_reference_image(f"{filename}_high_res_ib"),
        )

        return reference_images

    def restore_previous_state(self, method: 'AutoLamellaMethod', stage: AutoLamellaStage):
        """Restore the previous state of the lamella based on the current workflow stage"""
        prev = method.get_previous(stage)
        if prev in self.states:
            self.state = self.states[prev]


def create_new_lamella(experiment_path: str, number: int, state: LamellaState, protocol: Dict, name: Optional[str] = None) -> Lamella:
    """Wrapper function to create a new lamella and configure paths."""

    # create the petname and path
    if name is None:
        name = f"{number:02d}-{petname.generate(2)}"
    path = os.path.join(experiment_path, name)

    # create the lamella
    lamella = Lamella(
        petname=name,
        path=path,
        number=number,
        state=state,
        protocol=deepcopy(protocol), # TODO: replace with milling_workflows
    )

    # create the lamella directory
    os.makedirs(lamella.path, exist_ok=True)

    logging.info(f"Created new lamella {lamella.name} at {lamella.path}")

    return lamella

def create_new_experiment(path: Path, name: str, method: str = "autolamella-on-grid") -> 'Experiment':
    """Wrapper function to create an experiment and configure logging."""

    # create the experiment
    experiment = Experiment(path=path, name=name, method=method)

    # configure experiment logging
    os.makedirs(experiment.path, exist_ok=True)
    configure_logging(path=experiment.path, log_filename="logfile")

    # save the experiment
    experiment.save()

    logging.info(f"Created new experiment {experiment.name} at {experiment.path}")

    return experiment

@evented
@dataclass
class Experiment:
    name: str
    _id: str
    path: Path
    positions: EventedList[Lamella] = field(default_factory=EventedList)
    landing_positions: List[FibsemStagePosition] = field(default_factory=list)
    created_at: float = field(default_factory=lambda: datetime.timestamp(datetime.now()))
    method: 'AutoLamellaMethod' = field(default_factory=lambda: get_autolamella_method("autolamella-on-grid"))
    task_protocol: 'AutoLamellaTaskProtocol' = field(default_factory=lambda: AutoLamellaTaskProtocol())

    def __init__(self, path: Path, 
                 name: str = cfg.EXPERIMENT_NAME, 
                 method: str = "autolamella-on-grid") -> None:
        """Create a new experiment."""
        self.name: str = name
        self._id = str(uuid.uuid4())
        self.path: Path = os.path.join(path, name)
        self.created_at: float = datetime.timestamp(datetime.now())

        self.positions: EventedList[Lamella] = EventedList()
        self.landing_positions: List[FibsemStagePosition] = []

        self.method: AutoLamellaMethod = get_autolamella_method(method)

    def to_dict(self) -> dict:

        state_dict = {
            "name": self.name,
            "_id": self._id,
            "path": self.path,
            "positions": [deepcopy(lamella.to_dict()) for lamella in self.positions],
            "landing_positions": [pos.to_dict() for pos in self.landing_positions],
            "created_at": self.created_at,
            "method": self.method.name,
        }

        return state_dict

    @classmethod
    def from_dict(cls, ddict: dict) -> 'Experiment':

        path = os.path.dirname(ddict["path"])
        name = ddict["name"]
        experiment = Experiment(path=path, name=name)
        experiment.created_at = ddict.get("created_at", None)
        experiment._id = ddict.get("_id", "NULL")
        experiment.method = get_autolamella_method(ddict.get("method", "autoLamella-on-grid"))

        # load lamella from dict
        for lamella_dict in ddict["positions"]:
            lamella = Lamella.from_dict(data=lamella_dict)
            experiment.positions.append(lamella)

        # load landing positions
        for landing_dict in ddict.get("landing_positions", []):
            stage_position = FibsemStagePosition.from_dict(landing_dict)
            experiment.landing_positions.append(stage_position)

        return experiment

    def save(self) -> None:
        """Save the sample data to yaml file"""

        with open(os.path.join(self.path, "experiment.yaml"), "w") as f:
            yaml.safe_dump(self.to_dict(), f, indent=4)

    def __repr__(self) -> str:

        return f"""Experiment: 
        Path: {self.path}
        Positions: {len(self.positions)}
        """

    def __to_dataframe__(self) -> pd.DataFrame:

        exp_data = []
        lamella: Lamella
        for lamella in self.positions:

            # lamella
            ldict = {
                "experiment_name": self.name,
                "experiment_path": self.path,
                "experiment_created_at": self.created_at,
                "experiment_id": self._id,
                "method": self.method.name, 
                "number": lamella.number,
                "petname": lamella.petname,  # what?
                "path": lamella.path,
                "lamella.x": lamella.state.microscope_state.stage_position.x,
                "lamella.y": lamella.state.microscope_state.stage_position.y,
                "lamella.z": lamella.state.microscope_state.stage_position.z,
                "lamella.r": lamella.state.microscope_state.stage_position.r,
                "lamella.t": lamella.state.microscope_state.stage_position.t,
                "last_timestamp": lamella.state.microscope_state.timestamp, # dont know if this is the correct timestamp to use here
                "current_stage": lamella.state.stage.name,
                "failure": lamella.is_failure,
                "failure_note": lamella.failure_note,
                "failure_timestamp": lamella.failure_timestamp,
            }

            exp_data.append(ldict)

        df = pd.DataFrame(exp_data)

        return df

    def to_dataframe_v2(self) -> pd.DataFrame:


        edict = {
            "name": self.name,
            "path": self.path,
            "date": self.created_at,
            "experiment_id": self._id,
            "method": self.method.name, 
            "num_lamella": len(self.positions),
        }

        df = pd.DataFrame([edict])

        return df

    def history_dataframe(self) -> pd.DataFrame:
        """Create a dataframe with the history of all lamellas."""
        history = []
        lam: Lamella
        hist: LamellaState
        for lam in self.positions:

            petname = lam.petname

            for hist in lam.history:
                start, end = hist.start_timestamp, hist.end_timestamp
                stage_name = hist.stage.name

                hist_d = {
                    "petname": petname,
                    "stage": stage_name,
                    "start": start,
                    "end": end,
                }
                history.append(deepcopy(hist_d))

        df_stage_history = pd.DataFrame.from_dict(history)
        df_stage_history["duration"] = df_stage_history["end"] - df_stage_history["start"]

        return df_stage_history

    @staticmethod
    def load(fname: Path) -> 'Experiment':
        """Load an experiment from disk."""

        # read and open existing yaml file
        path = Path(fname).with_suffix(".yaml")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No file with name {path} found.")
        with open(path, "r") as f:
            ddict = yaml.safe_load(f)

        # create experiment from dict
        experiment = Experiment.from_dict(ddict)
        experiment.path = os.path.dirname(fname) # TODO: make sure the paths are correctly re-assigned when loaded on a different machine

        # configure experiment logging
        configure_logging(path=experiment.path, log_filename="logfile")

        return experiment
    
    def to_protocol_dataframe(self) -> pd.DataFrame:
        """Create a dataframe with the protocol of all lamellas."""

        plist: List[Dict] = []
        for p in self.positions:
            # convert lamella milling protocols to dataframes
            for k, v in p.protocol.items():
                for vi in v:
                    ddict = {}
                    ddict["Experiment"] = self.name
                    ddict["Workflow"] = k
                    ddict["Lamella"] = p.name
                    for k2, v2 in vi.items():
                        if isinstance(v2, dict):
                            for k3, v3 in v2.items():
                                ddict[ f"{k2}-{k3}"] = v3
                        else:
                            ddict[f"{k2}"] = v2
                    plist.append(deepcopy(ddict))

        df = pd.DataFrame(plist)
        # drop tescan columns  
        TESCAN_COLUMNS = [
            "milling-dwell_time",
            "milling-preset",
            "milling-rate",
            "milling-spacing",
            "milling-spot_size",
        ]
        df = df.drop(columns=TESCAN_COLUMNS)

        return df

    def _convert_dataframe_to_protocol(self, df: pd.DataFrame) -> None:
        """Convert a dataframe to a protocol."""

        PROTOCOL_KEYS = ["trench", "MillUndercut", "fiducial", "notch", "MillRough", "MillRegularCut", "MillPolishing", "microexpansion"]

        df.sort_values(by=["MillingStage"], inplace=True)

        for lamella in self.positions:
            petname = lamella.petname
            print("-"*50, petname, "-"*50)

            df_petname = df[df["Lamella"]==petname].copy(deep=True)

            for k in PROTOCOL_KEYS:
                # convert data frame back to dict

                # sort by milling_stage
                df_filt = df_petname[df_petname["WorkflowStage"]==k].copy(deep=True)

                if df_filt.empty:
                    continue 

                # drop na columns
                df_filt.dropna(axis=1, how="all", inplace=True)

                # drop milling_stage, lamella, stage
                df_filt.drop(columns=["MillingStage", "Lamella", "WorkflowStage"], inplace=True)

                ddict = deepcopy(df_filt.to_dict(orient="records"))

                lamella.protocol[k]["stages"] = deepcopy(ddict)

                from pprint import pprint
                print("KEY: ", k)
                pprint(lamella.protocol[k]["stages"])
                print('-'*100)

    def at_stage(self, stage: AutoLamellaStage) -> List[Lamella]:
        """Return a list of lamellas at a specific stage"""
        return [p for p in self.positions if (p.workflow is stage and p.is_active)]

    def at_failure(self) -> List[Lamella]:
        """Return a list of lamellas that have failed"""
        return [lamella for lamella in self.positions if lamella.is_failure]

    def to_summary_dataframe(self) -> pd.DataFrame:
        """Convert the experiment to a summary dataframe"""
        dat = []
        for p in self.positions:
            
            is_finished = p.finished
            is_failure = p.is_failure

            status_msg = "Active"
            if is_finished:
                status_msg = "Finished"
            if is_failure:
                if len(p.failure_note) > 5:
                    note = f"{p.failure_note[:3]}..."
                else:
                    note = p.failure_note
                status_msg = f"Defect ({note})"

            # get the last completed workflow stage
            last_label = p.last_completed
            if is_finished:
                # special case for finished lamella
                prev = self.method.get_previous(p.workflow)
                state = p.states[prev]
                last_label = state.completed

            d = {
                "Name": p.name,
                "Status": status_msg,
                "Last Completed": last_label,
                "Milling Angle": round(p.milling_angle, 2) if p.milling_angle is not None else None,
            }

            dat.append(deepcopy(d))

        df = pd.DataFrame(dat)

        return df
    
    def estimate_remaining_time(self) -> float:
        """Estimate the remaining time for all lamellas in the experiment"""
        total_remaining_time: float = 0
        for p in self.positions:
            
            # skip failed lamellas
            if p.is_failure:
                continue
            
            # remaining time for individual lamella
            remaining_time = estimate_remaining_time(p, method=self.method)           
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
    def create(cls, path: Path, name: str = cfg.EXPERIMENT_NAME) -> 'Experiment':
        """Create a new experiment with the given path and name. Also configures logging.
        Args:
            path (Path): The path where the experiment will be created.
            name (str): The name of the experiment. Defaults to cfg.EXPERIMENT_NAME.
        Returns:
                Experiment: The created experiment instance.
        """
        # create the experiment
        experiment = Experiment(path=path, name=name)

        # configure experiment logging
        os.makedirs(experiment.path, exist_ok=True)
        configure_logging(path=experiment.path, log_filename="logfile")

        # save the experiment
        experiment.save()

        logging.info(f"Created new experiment {experiment.name} at {experiment.path}")

        return experiment

###### TASK REFACTORING ##########

    def add_new_lamella(self, state: LamellaState,
                        task_config: Dict[str, AutoLamellaTaskConfig],
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
                          state=state,
                          protocol={},
                          task_config=task_config)

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
                    "start_timestamp": task.start_timestamp,
                    "end_timestamp": task.end_timestamp,
                    "duration": task.duration,
                }
                history.append(deepcopy(ddict))

        df_task_history = pd.DataFrame.from_dict(history)
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
                "last_completed": p.last_completed_task.name if p.last_completed_task else None,
                "last_completed_at": p.last_completed_task.completed_at if p.last_completed_task else None,
                "is_completed": self.task_protocol.workflow_config.is_completed(p),
                "is_failure": p.is_failure,
            }
            edict.append(deepcopy(ddict))

        df = pd.DataFrame(edict)

        return df

########## PROTOCOL V2 ##########

@dataclass
class FibsemProtocol(ABC):
    pass

@dataclass
class MethodConfig:
   name: str
   workflow: List[AutoLamellaStage]

class AutoLamellaMethod(Enum):
    ON_GRID = MethodConfig(
        name="AutoLamella-OnGrid",
        workflow=[
            AutoLamellaStage.SetupLamella,
            AutoLamellaStage.MillRough,
            AutoLamellaStage.SetupPolishing,
            AutoLamellaStage.MillPolishing,
        ]
    )

    TRENCH = MethodConfig(
        name="AutoLamella-Trench",
        workflow=[
            AutoLamellaStage.MillTrench,
        ]
    )

    WAFFLE = MethodConfig(
        name="AutoLamella-Waffle",
        workflow=[
            AutoLamellaStage.MillTrench,
            AutoLamellaStage.MillUndercut,
            AutoLamellaStage.SetupLamella,
            AutoLamellaStage.MillRough,
            AutoLamellaStage.SetupPolishing,
            AutoLamellaStage.MillPolishing,
        ]
    )

    LIFTOUT = MethodConfig(
        name="AutoLamella-Liftout",
        workflow=[
            AutoLamellaStage.MillTrench,
            AutoLamellaStage.MillUndercut,
            AutoLamellaStage.LiftoutLamella,
            AutoLamellaStage.LandLamella,
            AutoLamellaStage.SetupLamella,
            AutoLamellaStage.MillRough,
            AutoLamellaStage.SetupPolishing,
            AutoLamellaStage.MillPolishing,
        ]
    )

    SERIAL_LIFTOUT = MethodConfig(
        name="AutoLamella-Serial-Liftout",
        workflow=[
            AutoLamellaStage.MillTrench,
            AutoLamellaStage.MillUndercut,
            AutoLamellaStage.LiftoutLamella,
            AutoLamellaStage.LandLamella,
            AutoLamellaStage.SetupLamella,
            AutoLamellaStage.MillRough,
            AutoLamellaStage.SetupPolishing,
            AutoLamellaStage.MillPolishing,
        ]
    )

    @property
    def name(self) -> str:
       return self.value.name

    @property
    def workflow(self) -> List[AutoLamellaStage]:
       return self.value.workflow
   
    @property
    def is_on_grid(self) -> bool:
        return self in [AutoLamellaMethod.ON_GRID, 
                        AutoLamellaMethod.WAFFLE]
   
    @property
    def is_trench(self) -> bool:
        return self in [AutoLamellaMethod.TRENCH, 
                        AutoLamellaMethod.WAFFLE, 
                        AutoLamellaMethod.LIFTOUT, 
                        AutoLamellaMethod.SERIAL_LIFTOUT]
   
    @property
    def is_liftout(self) -> bool:
        return self in [AutoLamellaMethod.LIFTOUT, 
                        AutoLamellaMethod.SERIAL_LIFTOUT]

    def get_next(self, current_stage: AutoLamellaStage) -> Optional[AutoLamellaStage]:
        if current_stage is AutoLamellaStage.Finished:
            return AutoLamellaStage.Finished
        if current_stage in [AutoLamellaStage.Created, AutoLamellaStage.PositionReady]:
            return self.workflow[0]
        
        idx = self.workflow.index(current_stage)

        # clip idx to 0
        if idx < len(self.workflow)-1:
            return self.workflow[idx+1]
        else:
            return None

    def get_previous(self, current_stage: AutoLamellaStage) -> Optional[AutoLamellaStage]:
        if current_stage is AutoLamellaStage.Finished:
            return self.workflow[-1]

        if current_stage in [AutoLamellaStage.Created, AutoLamellaStage.PositionReady]:
            return AutoLamellaStage.Created

        idx = self.workflow.index(current_stage)
        if idx == 0:
            return AutoLamellaStage.PositionReady

        # clip idx to 0
        if idx <= len(self.workflow)-1:
            return self.workflow[idx-1]
        else:
            return None

def is_ready_for(lamella: Lamella, method: AutoLamellaMethod, workflow: AutoLamellaStage) -> bool:
    """Is the lamella ready for the workflow step based on the 
    previous step being completed and not failed"""
    if lamella.is_failure:
        return False
    
    if workflow not in method.workflow:
        logging.debug(f"Workflow {workflow} not in method {method.name}")
        return False

    # get previous
    previous = method.get_previous(workflow)
    # check if previous is completed (at any point)
    if previous is not None:
        if previous in lamella.states:
            return True

    return False

DEFAULT_AUTOLAMELLA_METHOD = AutoLamellaMethod.ON_GRID.name

WORKFLOW_STAGE_TO_PROTOCOL_KEY = {
    AutoLamellaStage.MillTrench: TRENCH_KEY,
    AutoLamellaStage.MillUndercut: UNDERCUT_KEY,
    AutoLamellaStage.SetupLamella: SETUP_LAMELLA_KEY,
    AutoLamellaStage.LiftoutLamella: LIFTOUT_KEY,
    AutoLamellaStage.LandLamella: LANDING_KEY,
    AutoLamellaStage.MillRough: MILL_ROUGH_KEY,
    AutoLamellaStage.SetupPolishing: SETUP_LAMELLA_KEY,
    AutoLamellaStage.MillPolishing: MILL_POLISHING_KEY,
}

@dataclass
class AutoLamellaProtocolOptions:
    use_fiducial: bool
    use_microexpansion: bool
    use_notch: bool
    take_final_reference_images: bool
    alignment_attempts: int 
    alignment_at_milling_current: bool
    milling_angle: float
    undercut_tilt_angle: float
    checkpoint: str
    turn_beams_off: bool = False

    def to_dict(self):
        return {
            "use_fiducial": self.use_fiducial,
            "use_notch": self.use_notch,
            "use_microexpansion": self.use_microexpansion,
            "take_final_reference_images": self.take_final_reference_images,
            "alignment_attempts": self.alignment_attempts,
            "alignment_at_milling_current": self.alignment_at_milling_current,
            "milling_angle": self.milling_angle,
            "undercut_tilt_angle": self.undercut_tilt_angle,
            "checkpoint": self.checkpoint,
            "turn_beams_off": self.turn_beams_off,
        }

    @classmethod
    def from_dict(cls, ddict: dict) -> 'AutoLamellaProtocolOptions':        
        return cls(
            use_fiducial=ddict.get("use_fiducial", True),
            use_notch=ddict.get("use_notch", False),
            use_microexpansion=ddict.get("use_microexpansion", True),
            take_final_reference_images=ddict["take_final_reference_images"],
            alignment_attempts=int(ddict.get("alignment_attempts", 3)),
            alignment_at_milling_current=ddict.get("alignment_at_milling_current", False),
            milling_angle=ddict.get("milling_angle",
                                    ddict.get("milling_tilt_angle",
                                              ddict.get("lamella_tilt_angle", 18))),
            undercut_tilt_angle=ddict.get("undercut_tilt_angle", -5),
            checkpoint=ddict.get("checkpoint", "autolamella-mega-20240107.pt"),
            turn_beams_off=ddict.get("turn_beams_off", False),
        )

def get_completed_stages(pos: Lamella, method: AutoLamellaMethod) -> List[AutoLamellaStage]:
    """Get a list of completed worflow stages in a method for a given position"""
    # filter out the states that are not in the method (setups, finishes, etc.)
    workflow_states = sorted(pos.states.keys(), key=lambda x: x.value)
    completed_states = [wf for wf in workflow_states if wf in method.workflow]

    return completed_states

def get_remaining_stages(pos: Lamella, method: AutoLamellaMethod) -> List[AutoLamellaStage]:
    """Get a list of remaining worflow stages in a method for a given position"""
    completed_states = get_completed_stages(pos, method)
    remaining_states = [wf for wf in method.workflow if wf not in completed_states]

    return remaining_states

def estimate_remaining_time(p:Lamella, method: AutoLamellaMethod) -> float:
    """Estimate the remaiing time in the workflows for a given method"""
    ESTIMATED_SETUP_TIME = 5*60
    OVERHEAD_TIME = 2*60

    remaining_stages = get_remaining_stages(p, method=method)

    remaining_time: float = 0
    for rs in remaining_stages:
        mwf = WORKFLOW_STAGE_TO_PROTOCOL_KEY[rs]
        if mwf not in p.milling_workflows:
            logging.debug(f"Estimated time for {rs}: {format_duration(ESTIMATED_SETUP_TIME)}")
            remaining_time += ESTIMATED_SETUP_TIME
            continue
        milling_stages = p.milling_workflows[mwf]
        est_milling_time = estimate_total_milling_time(milling_stages)
        logging.debug(f"Estimated time for {rs}: {format_duration(est_milling_time)}")
        remaining_time += est_milling_time + OVERHEAD_TIME

    logging.debug(f"Total estimated time: {format_duration(remaining_time)}")

    return remaining_time


def get_autolamella_method(name: str) -> AutoLamellaMethod:
    method_aliases = {
        AutoLamellaMethod.ON_GRID: ["autolamella-on-grid", "on-grid", "AutoLamella-OnGrid", "AutoLiftout"],
        AutoLamellaMethod.WAFFLE: ["autolamella-waffle", "waffle", "AutoLamella-Waffle"],
        AutoLamellaMethod.TRENCH: ["autolamella-trench", "trench", "AutoLamella-Trench"],
        AutoLamellaMethod.LIFTOUT: ["autolamella-liftout", "liftout", "AutoLamella-Liftout"],
        AutoLamellaMethod.SERIAL_LIFTOUT: ["autolamella-serial-liftout", "serial-liftout", "AutoLamella-Serial-Liftout"],
    }
    
    # Create a flattened mapping of all aliases to their methods
    name_mapping = {
        alias.lower(): method 
        for method, aliases in method_aliases.items() 
        for alias in aliases
    }

    normalized_name = name.lower()
    if normalized_name not in name_mapping:
        valid_names = sorted(set(alias for aliases in method_aliases.values() for alias in aliases))
        raise ValueError(f"Unknown method: {name}. Valid methods are: {valid_names}")
    
    return name_mapping[normalized_name]

def get_supervision(stage: AutoLamellaStage, protocol: dict) -> bool:
    key = WORKFLOW_STAGE_TO_PROTOCOL_KEY.get(stage, None)
    return protocol.get("supervise", {}).get(key, True)

@dataclass
class AutoLamellaProtocol(FibsemProtocol):
    name: str
    method: AutoLamellaMethod
    supervision: Dict[AutoLamellaStage, bool]
    configuration: MicroscopeSettings               # microscope configuration
    options: AutoLamellaProtocolOptions             # options for the protocol
    milling: Dict[str, List[FibsemMillingStage]]    # milling workflows
    tmp: dict # TODO: remove tmp use something real

    def to_dict(self):
        return {
            "name": self.name,
            "method": self.method.name,
            "supervision": {k.name: v for k, v in self.supervision.items()},
            "configuration": self.configuration.to_dict(),
            "options": self.options.to_dict(),
            "milling": {k: get_protocol_from_stages(v) for k, v in self.milling.items()},
            "tmp": self.tmp,
        }

    @classmethod
    def from_dict(cls, ddict: dict) -> 'AutoLamellaProtocol':

        # backwards compatibility
        if "name" not in ddict:
            ddict["name"] = ddict.get("options", {}).get("name", "Default-AutoLamella-Protocol")
        if "method" not in ddict:
            ddict["method"] = ddict.get("options", {}).get("method", DEFAULT_AUTOLAMELLA_METHOD)

        if "tmp" not in ddict:
            ddict["tmp"] = ddict.get("options", {})

        # get the method
        method = get_autolamella_method(ddict["method"])


        # load the supervision tasks
        if "supervision" in ddict:
            supervision_tasks = {AutoLamellaStage[k]: v for k, v in ddict["supervision"].items()}
        else:
            # backwards compatibility
            supervision_tasks = {k: get_supervision(k, ddict["options"]) for k in WORKFLOW_STAGE_TO_PROTOCOL_KEY.keys()}    
        
        # filter out tasks that arent part of the method
        supervision_tasks = {k: v for k, v in supervision_tasks.items() if k in method.workflow}

        return cls(
            name=ddict["name"],
            method=method,
            supervision=supervision_tasks,
            configuration=ddict.get("configuration", {}),
            options=AutoLamellaProtocolOptions.from_dict(ddict["options"]),
            milling={k: get_milling_stages(k, ddict["milling"]) for k in ddict["milling"]},
            tmp=ddict["tmp"],
        )
    
    def save(self, path: Path) -> None:
        """Save the protocol to disk."""
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, indent=4)
    
    @staticmethod
    def load(path: Path) -> 'AutoLamellaProtocol':
        """Load the protocol from disk."""
        with open(path, "r") as f:
            ddict = yaml.safe_load(f)

        tmp_ddict = deepcopy(ddict)
        try:
            from fibsem.applications.autolamella.protocol.validation import (
                validate_and_convert_protocol,
            )
            ddict = validate_and_convert_protocol(ddict)
        except Exception as e:
            logging.debug(f"Error converting protocol: {e}")
            ddict = tmp_ddict
        
        return AutoLamellaProtocol.from_dict(ddict)