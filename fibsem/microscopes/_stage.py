from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import yaml
from psygnal import Signal

from fibsem import constants
from fibsem.structures import BeamType, FibsemStagePosition, RangeLimit

if TYPE_CHECKING:
    from fibsem.microscope import FibsemMicroscope

GRID_RADIUS = 1e-3  # 1mm


@dataclass
class SampleGrid:
    """A physical TEM grid or sample that can be loaded into a GridSlot."""

    name: str
    description: str = ""
    radius: float = field(
        default=GRID_RADIUS,
        metadata={"units": "mm", "tooltip": "Radius of the sample grid", "scale": 1e3},
    )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "radius": self.radius,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SampleGrid":
        return SampleGrid(
            name=data.get("name", ""),
            description=data.get("description", ""),
            radius=data.get("radius", GRID_RADIUS),
        )


@dataclass
class GridSlot:
    """A fixed physical slot on a SampleHolder. May have a SampleGrid loaded into it."""

    name: str
    index: int
    position: FibsemStagePosition
    loaded_grid: Optional[SampleGrid] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "index": self.index,
            "position": self.position.to_dict(),
            "loaded_grid": self.loaded_grid.to_dict() if self.loaded_grid is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GridSlot":
        loaded_grid_data = data.get("loaded_grid")
        loaded_grid = SampleGrid.from_dict(loaded_grid_data) if loaded_grid_data is not None else None
        slot = GridSlot(
            name=data.get("name", ""),
            index=data.get("index", 0),
            position=FibsemStagePosition(**data.get("position", {})),
            loaded_grid=loaded_grid,
        )
        slot.position.name = slot.name
        return slot


@dataclass
class SampleHolder:
    name: str = field(
        default="Sample Holder", metadata={"tooltip": "Name of the sample holder"}
    )
    description: str = field(
        default="", metadata={"tooltip": "Description of the sample holder"}
    )
    pre_tilt: float = field(
        default=0.0,
        metadata={
            "units": constants.DEGREE_SYMBOL,
            "minimum": 0.0,
            "maximum": 90.0,
            "decimals": 0,
            "tooltip": "Pre-tilt angle of the sample holder",
        },
    )
    reference_rotation: float = field(
        default=0.0,
        metadata={
            "units": constants.DEGREE_SYMBOL,
            "minimum": 0.0,
            "maximum": 360.0,
            "decimals": 0,
            "tooltip": "Reference rotation angle of the sample holder",
        },
    )
    capacity: int = field(
        default=2,
        metadata={
            "minimum": 1,
            "maximum": 12,
            "tooltip": "Number of grid slots on this holder",
        },
    )
    slots: dict[str, GridSlot] = field(default_factory=dict)

    def find_slot_for_grid(self, grid: "SampleGrid") -> Optional["GridSlot"]:
        """Return the slot that has this SampleGrid loaded, or None."""
        for slot in self.slots.values():
            if slot.loaded_grid is grid:
                return slot
        return None

    def find_slot_by_grid_name(self, grid_name: str) -> Optional["GridSlot"]:
        """Return the slot whose loaded grid matches the given name, or None."""
        for slot in self.slots.values():
            if slot.loaded_grid is not None and slot.loaded_grid.name == grid_name:
                return slot
        return None

    def _ensure_slots(self) -> None:
        """Ensure exactly `capacity` slots exist; add empty ones for missing indices."""
        for i in range(self.capacity):
            name = f"Slot-{i + 1:02d}"
            if name not in self.slots:
                self.slots[name] = GridSlot(
                    name=name,
                    index=i,
                    position=FibsemStagePosition(name=name, x=0.0, y=0.0, z=0.0),
                )
        for name in [n for n, s in list(self.slots.items()) if s.index >= self.capacity]:
            del self.slots[name]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "capacity": self.capacity,
            "pre_tilt": self.pre_tilt,
            "reference_rotation": self.reference_rotation,
            "slots": {name: slot.to_dict() for name, slot in self.slots.items()},
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SampleHolder":
        slots = {
            name: GridSlot.from_dict(slot_data)
            for name, slot_data in data.get("slots", {}).items()
        }
        holder = SampleHolder(
            name=data.get("name", "Sample Holder"),
            capacity=data.get("capacity", max(len(slots), 1)),
            pre_tilt=data.get("pre_tilt", 0.0),
            reference_rotation=data.get("reference_rotation", 0.0),
            slots=slots,
            description=data.get("description", ""),
        )
        holder._ensure_slots()
        return holder

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SampleHolder":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Sample holder config not found: {path}")
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


@dataclass
class SampleGridLoader:
    parent: "FibsemMicroscope"

    def __init__(self, parent: "FibsemMicroscope") -> None:
        self.parent = parent

    def load_grid(self, slot_name: str, grid: SampleGrid) -> None:
        """Load a SampleGrid into the named slot."""
        slot = self.parent._stage.holder.slots.get(slot_name)
        if slot is not None:
            slot.loaded_grid = grid

    def unload_grid(self, slot_name: str) -> None:
        """Remove the SampleGrid from the named slot."""
        slot = self.parent._stage.holder.slots.get(slot_name)
        if slot is not None:
            slot.loaded_grid = None

    @property
    def loaded_slots(self) -> List[GridSlot]:
        """Return all slots that currently have a grid loaded."""
        if self.parent._stage.holder is None:
            return []
        return [s for s in self.parent._stage.holder.slots.values() if s.loaded_grid is not None]


class Stage:
    parent: "FibsemMicroscope"
    holder: "SampleHolder"
    loader: Optional[SampleGridLoader] = None
    _position: Optional[FibsemStagePosition] = None
    position_changed = Signal(FibsemStagePosition)
    limits: dict[str, RangeLimit] = field(default_factory=dict)

    def __init__(
        self,
        parent: "FibsemMicroscope",
        holder: SampleHolder,
        loader: Optional[SampleGridLoader] = None,
    ) -> None:
        self.parent = parent
        self.holder = holder
        self.loader = loader
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
        return self.parent.get_stage_orientation()

    @property
    def milling_angle(self) -> float:
        return self.parent.get_current_milling_angle()

    @property
    def current_slot(self) -> Optional[GridSlot]:
        """Get the slot the stage is currently positioned at, if any."""
        if self.holder is None:
            return None
        stage_position = self.parent._stage_position
        for slot in self.holder.slots.values():
            if stage_position.is_close2(slot.position, tol=GRID_RADIUS, axes=["x", "y"]):
                return slot
        return None

    @property
    def current_grid(self) -> Optional[SampleGrid]:
        """Get the loaded SampleGrid at the current slot, if any."""
        slot = self.current_slot
        return slot.loaded_grid if slot is not None else None

    @property
    def is_homed(self) -> bool:
        return self.parent.get("stage_homed")  # type: ignore

    def move_absolute(self, position: FibsemStagePosition) -> FibsemStagePosition:
        return self.parent.move_stage_absolute(position)

    def move_relative(self, position: FibsemStagePosition) -> FibsemStagePosition:
        return self.parent.move_stage_relative(position)

    def stable_move(self, dx: float, dy: float, beam_type: BeamType) -> FibsemStagePosition:
        return self.parent.stable_move(dx, dy, beam_type)

    def vertical_move(self, dy: float, dx: float = 0.0) -> FibsemStagePosition:
        return self.parent.vertical_move(dy, dx)

    def move_to_milling_angle(self, milling_angle: float) -> bool:
        return self.parent.move_to_milling_angle(milling_angle)

    def home(self) -> bool:
        return self.parent.home()

    def project_stable_move(
        self,
        dx: float,
        dy: float,
        beam_type: BeamType,
        base_position: FibsemStagePosition,
    ) -> FibsemStagePosition:
        return self.parent.project_stable_move(dx, dy, beam_type, base_position)

    def move_to_slot(self, slot_name: str) -> FibsemStagePosition:
        """Move the stage to a specific slot."""
        if self.holder is None:
            raise ValueError("No sample holder defined.")
        if slot_name not in self.holder.slots:
            raise ValueError(f"Slot '{slot_name}' not found in sample holder.")
        slot = self.holder.slots[slot_name]
        self.move_absolute(slot.position)
        return self.position

    def move_to_grid(self, grid_name: str) -> FibsemStagePosition:
        """Alias for move_to_slot for backward compatibility."""
        return self.move_to_slot(grid_name)
