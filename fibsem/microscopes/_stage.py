from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

from psygnal import Signal

from fibsem import constants
from fibsem.structures import BeamType, FibsemStagePosition, Point, RangeLimit

if TYPE_CHECKING:
    from fibsem.microscope import FibsemMicroscope

GRID_RADIUS = 1e-3  # 1mm

@dataclass
class SampleGrid:
    name: str
    index: int
    position: FibsemStagePosition
    radius: float = field(default=GRID_RADIUS, metadata={"units": "mm",
                                                         "tooltip": "Radius of the sample grid", 
                                                         "scale": 1e3})
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "index": self.index,
            "position": self.position.to_dict(),
            "radius": self.radius,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SampleGrid":
        return SampleGrid(
            name=data.get("name", ""),
            index=data.get("index", 0),
            position=FibsemStagePosition(**data.get("position", {})),
            radius=data.get("radius", GRID_RADIUS),
            description=data.get("description", "")
        )

@dataclass
class SampleHolder:
    name: str = field(default="Sample Holder", metadata={"tooltip": "Name of the sample holder"})
    description: str = field(default="", metadata={"tooltip": "Description of the sample holder"})
    pre_tilt: float = field(default=0.0, 
                            metadata={"units": constants.DEGREE_SYMBOL, 
                                      "minimum": 0.0, "maximum": 90.0, "decimals": 0,
                                        "tooltip": "Pre-tilt angle of the sample holder"})
    reference_rotation: float = field(default=0.0, 
                                      metadata={"units": constants.DEGREE_SYMBOL,
                                        "minimum": 0.0, "maximum": 360.0, "decimals": 0,
                                        "tooltip": "Reference rotation angle of the sample holder"})
    grids: dict[str, SampleGrid] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "pre_tilt": self.pre_tilt,
            "reference_rotation": self.reference_rotation,
            "grids": {name: grid.to_dict() for name, grid in self.grids.items()},
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SampleHolder":
        grids = {name: SampleGrid.from_dict(grid_data)
                 for name, grid_data in data.get("grids", {}).items()}
        return SampleHolder(
            name=data.get("name", "Sample Holder"),
            pre_tilt=data.get("pre_tilt", 0.0),
            reference_rotation=data.get("reference_rotation", 0.0),
            grids=grids
        )


@dataclass
class SampleGridLoader:
    parent: "FibsemMicroscope"
    _loaded_grids: Optional[List[SampleGrid]] = None

    def __init__(self, parent: "FibsemMicroscope") -> None:
        self.parent = parent

    @property
    def loaded_grids(self) -> Optional[List[SampleGrid]]:
        return self._loaded_grids

    @property
    def available_grids(self) -> List[SampleGrid]:
        return []

    def load_grid(self, grid_name: str) -> None:
        pass

    def unload_grid(self) -> None:
        pass


class Stage:
    parent: "FibsemMicroscope"
    holder: 'SampleHolder'
    loader: Optional[SampleGridLoader] = None
    _position: Optional[FibsemStagePosition] = None
    position_changed = Signal(FibsemStagePosition)
    limits: dict[str, RangeLimit] = field(default_factory=dict)

    def __init__(self, parent: "FibsemMicroscope",
                 holder: SampleHolder,
                 loader: Optional[SampleGridLoader] = None) -> None:
        self.parent = parent
        self.holder = holder
        self.loader = loader

        # get the limits from the parent microscope
        self.limits = self.parent._get_axis_limits()

    def __repr__(self) -> str:
        return f"<Stage: position={self.position}, holder={self.holder}>"

    @property
    def axes(self) -> Tuple[str, ...]:
        return tuple(self.limits.keys())

    @property
    def position(self) -> FibsemStagePosition:
        return self.parent.get_stage_position()

    @property
    def orientation(self) -> str:
        """Get the current stage orientation."""
        return self.parent.get_stage_orientation()

    @property
    def milling_angle(self) -> float:
        """Get the current milling angle of the stage."""
        return self.parent.get_current_milling_angle()

    @property
    def current_grid(self) -> Optional[SampleGrid]:
        """Get the current sample grid."""
        if self.holder is None:
            return None

        stage_position = self.position
        for name, grid in self.holder.grids.items():
            if stage_position.is_close2(grid.position, tol=GRID_RADIUS, axes=['x', 'y']):
                return grid
        return None

    @property
    def is_homed(self) -> bool:
        """Check if the stage is homed."""
        return self.parent.get("stage_homed") # type: ignore

    def move_absolute(self, position: FibsemStagePosition) -> FibsemStagePosition:
        """Move the stage to an absolute position."""
        return self.parent.move_stage_absolute(position)

    def move_relative(self, position: FibsemStagePosition) -> FibsemStagePosition:
        """Move the stage by a relative delta."""
        return self.parent.move_stage_relative(position)
    
    def stable_move(self, dx: float, dy: float, beam_type: BeamType) -> FibsemStagePosition:
        """Perform a stable move of the stage."""
        return self.parent.stable_move(dx, dy, beam_type)

    def vertical_move(self, dy: float, dx: float = 0.0) -> FibsemStagePosition:
        """Perform a vertical move of the stage."""
        return self.parent.vertical_move(dy, dx)

    def move_to_milling_angle(self, milling_angle: float) -> bool:
        """Move the stage to a specific milling angle."""
        return self.parent.move_to_milling_angle(milling_angle)

    def home(self) -> bool:
        """Home the stage."""
        return self.parent.home()
    
    def project_stable_move(self, dx: float, dy: float, beam_type: BeamType, base_position: FibsemStagePosition) -> FibsemStagePosition:
        """Project a stable move of the stage."""
        return self.parent.project_stable_move(dx, dy, beam_type, base_position)
    
    def move_to_grid(self, grid_name: str) -> FibsemStagePosition:
        """Move the stage to a specific grid."""
        if self.holder is None:
            raise ValueError("No sample holder defined.")
        if grid_name not in self.holder.grids:
            raise ValueError(f"Grid '{grid_name}' not found in sample holder.")
        
        grid = self.holder.grids[grid_name]
        self.move_absolute(grid.position)

        return self.position
