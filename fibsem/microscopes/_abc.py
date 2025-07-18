from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

from dataclasses import dataclass, field
from fibsem.structures import FibsemStagePosition, FibsemManipulatorPosition, BeamType, Point, BeamSettings, FibsemDetectorSettings
from fibsem.microscope import FibsemMicroscope, FluorescenceMicroscope

GRID_RADIUS = 1e-3  # 1mm tolerance for grid matching

@dataclass
class Axis:
    """Class representing an axis of the stage."""
    name: str
    position: float = 0.0
    limits: Tuple[float, float] = (0.0, 0.0)
    locked: bool = False

@dataclass
class Stage(ABC):
    """Abstract base class for the stage functionality of a FibsemMicroscope."""
    holder: 'SampleHolder'

    def __init__(self, parent: 'FibsemMicroscope', holder: 'SampleHolder'):
        """Initialize the stage with a parent FibsemMicroscope and a SampleHolder."""
        self.parent = parent
        self.holder = holder

    @property
    def position(self) -> FibsemStagePosition:
        """Get the current stage position."""
        return self.parent.get_stage_position()

    @property
    def orientation(self) -> str:
        """Get the current stage orientation."""
        return self.parent.get_stage_orientation()

    @property
    def milling_angle(self) -> float:
        """Get the current milling angle."""
        return self.parent.get_current_milling_angle()

    def get_named_orientation(self, orientation: str) -> Optional[FibsemStagePosition]:
        """Get the stage orientation."""
        return self.parent.get_orientation(orientation)

    @property
    def axes(self) -> Dict[str, Axis]:
        """Get the current stage axes."""
        raise NotImplementedError("This method should be implemented in the subclass.")

    @property
    def current_grid(self) -> Optional[str]:
        """Get the current grid name."""
        if not self.holder or not self.holder.grids:
            return None

        # loop through the grids and find the one that matches the current position
        for name, position in self.holder.grids.items():
            if position.is_close(self.position, tol=GRID_RADIUS):  # 1mm tolerance
                return name

    def move_absolute(self, position: FibsemStagePosition) -> None:
        """Move the stage to an absolute position."""
        self.parent.move_stage_absolute(position)

    def move_relative(self, position: FibsemStagePosition) -> None:
        """Move the stage to a position relative to the current position."""
        self.parent.move_stage_relative(position)

    def move_vertical(self, dy: float, dx: float = 0) -> None:
        """Move the stage vertically."""
        self.parent.vertical_move(dy, dx)

    def move_stable(self, dx: float, dy: float, beam_type: BeamType) -> FibsemStagePosition:
        """Move the stage to a stable position."""
        return self.parent.stable_move(dx, dy, beam_type)

@dataclass
class SampleHolder(ABC):
    """Abstract base class for the sample holder functionality of a FibsemMicroscope."""
    pre_tilt: float
    grids: Dict[str, FibsemStagePosition] = field(default_factory=dict)

    def add_grid(self, name: str, position: FibsemStagePosition) -> None:
        """Add a grid to the sample holder."""
        if name in self.grids:
            raise ValueError(f"Grid '{name}' already exists in the sample holder.")
        self.grids[name] = position

    def get_grid_position(self, name: str) -> FibsemStagePosition:
        """Get the position of a grid in the sample holder."""
        if name not in self.grids:
            raise ValueError(f"Grid '{name}' does not exist in the sample holder.")
        return self.grids[name]

class Manipulator(ABC):
    """Abstract base class for the manipulator functionality of a FibsemMicroscope."""
    pass

    @property
    def position(self) -> FibsemManipulatorPosition:
        """Get the current manipulator position."""
        raise NotImplementedError("This method should be implemented in the subclass.")

    @abstractmethod
    def move_relative(self, position: FibsemManipulatorPosition) -> None:
        """Move the manipulator to a position relative to the current position."""
        raise NotImplementedError("This method should be implemented in the subclass.")
    
    @abstractmethod
    def move_absolute(self, position: FibsemManipulatorPosition) -> None:
        """Move the manipulator to an absolute position."""
        raise NotImplementedError("This method should be implemented in the subclass.")

class Column(ABC):

    def __init__(self, parent: FibsemMicroscope, config: Optional['ColumnConfig'] = None):
        """Initialize the column with a parent FibsemMicroscope."""
        self.parent = parent
        self.beam = Beam()
        self.detector = Detector()
        self.config = config

class Beam(ABC):
    
    @property
    def working_distance(self) -> float:
        """Get the current working distance."""
        raise NotImplementedError("This method should be implemented in the subclass.")
    
    @working_distance.setter
    def working_distance(self, value: float) -> None:
        """Set the working distance."""
        raise NotImplementedError("This method should be implemented in the subclass.")
    
    @property
    def current(self) -> float:
        """Get the current beam current."""
        raise NotImplementedError("This method should be implemented in the subclass.")
    @current.setter
    def current(self, value: float) -> None:
        """Set the beam current."""
        raise NotImplementedError("This method should be implemented in the subclass.")
    
    @property
    def voltage(self) -> float:
        """Get the current beam voltage."""
        raise NotImplementedError("This method should be implemented in the subclass.")

    @voltage.setter
    def voltage(self, value: float) -> None:
        """Set the beam voltage."""
        raise NotImplementedError("This method should be implemented in the subclass.")
    
    @property
    def field_of_view(self) -> float:
        """Get the current field of view."""
        raise NotImplementedError("This method should be implemented in the subclass.")
    @field_of_view.setter
    def field_of_view(self, value: float) -> None:
        """Set the field of view."""
        raise NotImplementedError("This method should be implemented in the subclass.")
    
    @property
    def resolution(self) -> float:
        """Get the current resolution."""
        raise NotImplementedError("This method should be implemented in the subclass.")
    
    @resolution.setter
    def resolution(self, value: float) -> None:
        """Set the resolution."""
        raise NotImplementedError("This method should be implemented in the subclass.")

    @property
    def stigmation(self) -> Point:
        """Get the current stigmation."""
        raise NotImplementedError("This method should be implemented in the subclass.")
    
    @stigmation.setter
    def stigmation(self, value: Point) -> None:
        """Set the stigmation."""
        raise NotImplementedError("This method should be implemented in the subclass.")
    
    @property
    def shift(self) -> Point:
        """Get the current beam shift."""
        raise NotImplementedError("This method should be implemented in the subclass.")
    @shift.setter
    def shift(self, value: Point) -> None:
        """Set the beam shift."""
        raise NotImplementedError("This method should be implemented in the subclass.")

class Detector(ABC):
    
    @property
    def type(self) -> str:
        """Get the current detector type."""
        raise NotImplementedError("This method should be implemented in the subclass.")
    @type.setter
    def type(self, value: str) -> None:
        """Set the detector type."""
        raise NotImplementedError("This method should be implemented in the subclass.")
    @property
    def mode(self) -> str:
        """Get the current detector mode."""
        raise NotImplementedError("This method should be implemented in the subclass.")
    
    @mode.setter
    def mode(self, value: str) -> None:
        """Set the detector mode."""
        raise NotImplementedError("This method should be implemented in the subclass.")
    @property
    def brightness(self) -> float:
        """Get the current detector brightness."""
        raise NotImplementedError("This method should be implemented in the subclass.")
    
    @brightness.setter
    def brightness(self, value: float) -> None:
        """Set the detector brightness."""
        raise NotImplementedError("This method should be implemented in the subclass.")
    
    @property
    def contrast(self) -> float:
        """Get the current detector contrast."""
        raise NotImplementedError("This method should be implemented in the subclass.")
    @contrast.setter
    def contrast(self, value: float) -> None:
        """Set the detector contrast."""
        raise NotImplementedError("This method should be implemented in the subclass.")

@dataclass
class StageConfig:
    reference_rotation: float
    pre_tilt: float
    grids: Dict[str, FibsemStagePosition] = field(default_factory=dict)

@dataclass
class ColumnConfig:
    beam: BeamSettings
    detector: FibsemDetectorSettings
    column_tilt: float
    eucentric_height: float
    plasma_gas: Optional[str]

class MicroscopeV2(ABC):
    sem: Optional[Column]
    fib: Optional[Column]
    fm: Optional[FluorescenceMicroscope]
    stage: Optional[Stage]
    manipulator: Optional[Manipulator]
