import logging
import os
import uuid
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Tuple

import yaml

from fibsem.milling import (
    FibsemMillingStage,
    get_milling_stages,
    get_protocol_from_stages,
)

from fibsem.applications.autolamella.protocol.constants import (
    LANDING_KEY,
    LIFTOUT_KEY,
    MILL_POLISHING_KEY,
    MILL_ROUGH_KEY,
    SETUP_LAMELLA_KEY,
    TRENCH_KEY,
    UNDERCUT_KEY,
)

if TYPE_CHECKING:
    from fibsem.structures import MicroscopeSettings
    from fibsem.applications.autolamella.structures import Lamella

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
            take_final_reference_images=ddict.get("take_final_reference_images", True),
            alignment_attempts=int(ddict.get("alignment_attempts", 3)),
            alignment_at_milling_current=ddict.get("alignment_at_milling_current", False),
            milling_angle=ddict.get("milling_angle",
                                    ddict.get("milling_tilt_angle",
                                              ddict.get("lamella_tilt_angle", 18))),
            undercut_tilt_angle=ddict.get("undercut_tilt_angle", -5),
            checkpoint=ddict.get("checkpoint", "autolamella-mega-20240107.pt"),
            turn_beams_off=ddict.get("turn_beams_off", False),
        )

def get_completed_stages(pos: 'Lamella', method: 'AutoLamellaMethod') -> List['AutoLamellaStage']:
    """Get a list of completed worflow stages in a method for a given position"""
    # filter out the states that are not in the method (setups, finishes, etc.)
    workflow_states = sorted(pos.states.keys(), key=lambda x: x.value)
    completed_states = [wf for wf in workflow_states if wf in method.workflow]

    return completed_states

def get_remaining_stages(pos: 'Lamella', method: 'AutoLamellaMethod') -> List['AutoLamellaStage']:
    """Get a list of remaining worflow stages in a method for a given position"""
    completed_states = get_completed_stages(pos, method)
    remaining_states = [wf for wf in method.workflow if wf not in completed_states]

    return remaining_states

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
    configuration: 'MicroscopeSettings'               # microscope configuration
    options: AutoLamellaProtocolOptions             # options for the protocol
    milling: Dict[str, List[FibsemMillingStage]]    # milling workflows
    tmp: dict # TODO: remove tmp use something real

    def to_dict(self):
        return {
            "name": self.name,
            "method": self.method.name,
            "supervision": {k.name: v for k, v in self.supervision.items()},
            "configuration": self.configuration.to_dict() if hasattr(self.configuration, "to_dict") else {},
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