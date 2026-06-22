from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
import yaml
from psygnal import Signal

from fibsem.config import (
    DEFAULT_SAMPLE_HOLDER_CONFIGURATION_PATH,
    SAMPLE_HOLDER_CONFIGURATION_PATH,
)
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
    """A numbered slot that may hold a SampleGrid.

    Holder slots carry a stage ``position`` (the beam location, used by
    ``move_to_slot`` / grid tasks). Loader *magazine* slots are storage only and
    leave ``position`` as ``None``.
    """

    name: str
    index: int
    position: Optional[FibsemStagePosition] = None
    loaded_grid: Optional[SampleGrid] = None

    def to_dict(self, include_grids: bool = True) -> dict:
        loaded_grid = (
            self.loaded_grid.to_dict()
            if (include_grids and self.loaded_grid is not None)
            else None
        )
        return {
            "name": self.name,
            "index": self.index,
            "position": self.position.to_dict() if self.position is not None else None,
            "loaded_grid": loaded_grid,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GridSlot":
        loaded_grid_data = data.get("loaded_grid")
        loaded_grid = (
            SampleGrid.from_dict(loaded_grid_data)
            if loaded_grid_data is not None
            else None
        )
        position_data = data.get("position")
        position = (
            FibsemStagePosition(**position_data) if position_data else None
        )
        slot = GridSlot(
            name=data.get("name", ""),
            index=data.get("index", 0),
            position=position,
            loaded_grid=loaded_grid,
        )
        if slot.position is not None:
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
    capacity: int = field(
        default=2,
        metadata={
            "minimum": 1,
            "maximum": 12,
            "tooltip": "Number of grid slots on this holder",
        },
    )
    slots: dict[str, GridSlot] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._parent: Optional["FibsemMicroscope"] = None

    @property
    def pre_tilt(self) -> float:
        if self._parent is not None:
            return self._parent.system.stage.shuttle_pre_tilt
        return 0.0

    @property
    def reference_rotation(self) -> float:
        if self._parent is not None:
            return self._parent.system.stage.rotation_reference
        return 0.0

    def find_slot_for_grid(self, grid: "SampleGrid") -> Optional["GridSlot"]:
        """Return the slot that has this SampleGrid loaded, or None."""
        for slot in self.slots.values():
            if slot.loaded_grid is not None and slot.loaded_grid.name == grid.name:
                return slot
        return None

    def find_slot_by_grid_name(self, grid_name: str) -> Optional["GridSlot"]:
        """Return the slot whose loaded grid matches the given name, or None."""
        for slot in self.slots.values():
            if slot.loaded_grid is not None and slot.loaded_grid.name == grid_name:
                return slot
        return None

    @property
    def occupied_slots(self) -> List["GridSlot"]:
        """The working slots that currently have a grid loaded."""
        return [s for s in self.slots.values() if s.loaded_grid is not None]

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
        for name in [
            n for n, s in list(self.slots.items()) if s.index >= self.capacity
        ]:
            del self.slots[name]

    def to_dict(self, include_grids: bool = True) -> dict:
        return {
            "name": self.name,
            "capacity": self.capacity,
            "slots": {
                name: slot.to_dict(include_grids=include_grids)
                for name, slot in self.slots.items()
            },
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

    def save(self, path: Union[str, Path], include_grids: bool = True) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(
                self.to_dict(include_grids=include_grids),
                f,
                default_flow_style=False,
                sort_keys=False,
            )


LOADER_CAPACITY = 12  # storage slots in a typical autoloader magazine
LOADER_SIM_MOVE_DELAY_S = 5.0  # simulated autoloader exchange time (sim only)
LOADER_SIM_INVENTORY_DELAY_S = 15.0  # simulated magazine inventory scan (sim only)


class SampleGridLoader:
    """Robotic autoloader: a magazine of storage slots that exchange grids into
    the holder's working slot.

    Two distinct slot sets:
    - ``slots`` — the **magazine** (its own ``capacity``, loaded by a human
      operator). The inventory of grids available to load. ``run_inventory``
      scans it; ``assign_grid`` records/names a grid in a magazine slot.
    - the holder's working slot(s) — the beam position. ``load_grid`` /
      ``unload_grid`` insert/retract a grid there (which the holder reports via
      ``SampleHolder.occupied_slots``).
    """

    def __init__(self, parent: "FibsemMicroscope",
                 capacity: int = LOADER_CAPACITY,
                 name: str = "Autoloader Magazine") -> None:
        self.parent = parent
        self.name = name
        self.capacity = capacity
        self.slots: dict[str, GridSlot] = {}
        # simulated timings (seconds); 0 = instant. Set by the simulator setup so
        # demo load/unload + inventory feel like real autoloader hardware.
        self.move_delay_s: float = 0.0
        self.inventory_delay_s: float = 0.0
        self._ensure_slots()

    def _simulate_move(self) -> None:
        if self.move_delay_s > 0:
            time.sleep(self.move_delay_s)

    def _ensure_slots(self) -> None:
        """Materialise exactly ``capacity`` magazine slots (Magazine-01, …).

        Magazine slots are storage only — no stage ``position``.
        """
        for i in range(self.capacity):
            name = f"Magazine-{i + 1:02d}"
            if name not in self.slots:
                self.slots[name] = GridSlot(name=name, index=i)

    # --- magazine (inventory) ---

    def run_inventory(self) -> List[GridSlot]:
        """Scan the magazine and return the slots that hold a grid.

        In a real loader this queries the hardware; here it reports the current
        magazine state (after a simulated scan delay, if configured).
        """
        if self.inventory_delay_s > 0:
            time.sleep(self.inventory_delay_s)
        return self.loaded_magazine_slots

    @property
    def loaded_magazine_slots(self) -> List[GridSlot]:
        return [s for s in self.slots.values() if s.loaded_grid is not None]

    def assign_grid(self, slot_name: str, grid: SampleGrid) -> None:
        """Operator action: place/name a grid in a magazine slot."""
        slot = self.slots.get(slot_name)
        if slot is not None:
            slot.loaded_grid = grid

    def find_grid(self, grid_name: str) -> Optional[GridSlot]:
        """Return the magazine slot holding the named grid, or None."""
        for slot in self.slots.values():
            if slot.loaded_grid is not None and slot.loaded_grid.name == grid_name:
                return slot
        return None

    # --- working slot (holder) insertion ---

    def load_grid(self, slot_name: str, grid: SampleGrid) -> None:
        """Insert a SampleGrid into the named holder working slot."""
        slot = self.parent._stage.holder.slots.get(slot_name)
        if slot is not None:
            self._simulate_move()
            slot.loaded_grid = grid

    def unload_grid(self, slot_name: str) -> None:
        """Retract the SampleGrid from the named holder working slot."""
        slot = self.parent._stage.holder.slots.get(slot_name)
        if slot is not None:
            self._simulate_move()
            slot.loaded_grid = None


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

    def slot_at_position(self, position: FibsemStagePosition) -> Optional[GridSlot]:
        """Get the holder slot a given stage position falls within, if any.

        Generalises ``current_slot`` to an arbitrary position (e.g. a lamella
        placed away from where the stage currently sits). Slots without a
        calibrated position are skipped.
        """
        if self.holder is None:
            return None
        for slot in self.holder.slots.values():
            if slot.position is not None and position.is_close2(
                slot.position, tol=GRID_RADIUS, axes=["x", "y"]
            ):
                return slot
        return None

    def grid_at_position(self, position: FibsemStagePosition) -> Optional[SampleGrid]:
        """Get the loaded SampleGrid at the slot a position falls within, if any."""
        slot = self.slot_at_position(position)
        return slot.loaded_grid if slot is not None else None

    @property
    def current_slot(self) -> Optional[GridSlot]:
        """Get the slot the stage is currently positioned at, if any."""
        return self.slot_at_position(self.parent._stage_position)

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

    def stable_move(
        self, dx: float, dy: float, beam_type: BeamType
    ) -> FibsemStagePosition:
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

    def move_to_orientation(self, orientation: str) -> FibsemStagePosition:
        """Move the stage to a specific orientation."""
        return self.parent.move_to_orientation(orientation)

    def move_to_grid(self, grid_name: str) -> FibsemStagePosition:
        """Alias for move_to_slot for backward compatibility."""
        return self.move_to_slot(grid_name)


def _create_sample_stage(microscope: "FibsemMicroscope") -> "Stage":

    if microscope.stage_is_compustage:
        slot01 = GridSlot(
            name="Slot-01",
            index=0,
            position=FibsemStagePosition(
                name="Slot-01", x=0.0, y=0.0, z=0.0, r=0.0, t=np.radians(0)
            ),
        )
        holder = SampleHolder(
            name="CompuStage Holder", capacity=1, slots={"Slot-01": slot01}
        )
        loader: Optional[SampleGridLoader] = SampleGridLoader(parent=microscope)
        # give the simulated loader realistic exchange + inventory timing in the
        # app, but keep it instant under pytest so the suite stays fast.
        if "PYTEST_CURRENT_TEST" not in os.environ:
            loader.move_delay_s = LOADER_SIM_MOVE_DELAY_S
            loader.inventory_delay_s = LOADER_SIM_INVENTORY_DELAY_S
    else:
        path = Path(SAMPLE_HOLDER_CONFIGURATION_PATH)
        if not path.exists():
            logging.info(f"Sample holder config not found at {path}, using default.")
            path = Path(DEFAULT_SAMPLE_HOLDER_CONFIGURATION_PATH)
        orientation = microscope.get_orientation("SEM")
        holder = SampleHolder.load(path)
        for slot in holder.slots.values():
            slot.position.r = orientation.r
            slot.position.t = orientation.t
        loader = None

    holder._parent = microscope
    return Stage(parent=microscope, holder=holder, loader=loader)
