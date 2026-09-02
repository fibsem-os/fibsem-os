from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Mapping, Optional, Tuple, Union

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
        metadata={"unit": "mm", "tooltip": "Radius of the sample grid", "scale": 1e3},
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
    """A slot that may hold one SampleGrid.

    A holder *working* slot has a calibrated stage ``position``. A loader *magazine*
    slot is storage and has none, so ``position`` is optional.
    """

    name: str
    index: int
    position: Optional[FibsemStagePosition] = None
    loaded_grid: Optional[SampleGrid] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "index": self.index,
            "position": self.position.to_dict() if self.position is not None else None,
            "loaded_grid": self.loaded_grid.to_dict()
            if self.loaded_grid is not None
            else None,
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
            FibsemStagePosition(**position_data) if position_data is not None else None
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
        """The working slots that hold a grid: what is in the beam right now."""
        return [
            s
            for s in sorted(self.slots.values(), key=lambda s: s.index)
            if s.loaded_grid is not None
        ]

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

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "capacity": self.capacity,
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


class GridExchangeError(RuntimeError):
    """A grid could not be moved into, or out of, the holder's working slot."""


@dataclass
class GridInventoryEntry:
    """One row of ``Stage.grid_inventory()``: a slot, and what it holds.

    ``source`` says which hardware answered -- ``"magazine"`` on a system with a
    loader, ``"holder"`` otherwise. ``present`` is whether the slot holds a grid;
    ``in_beam`` whether that grid is in a holder working slot right now. The two
    are kept apart on purpose: *available* is what a run can select from, *in beam*
    is what it can act on without an exchange.
    """

    slot_name: str
    index: int
    source: str
    name: Optional[str]
    present: bool
    in_beam: bool


class SampleGridLoader:
    """The robotic actuator that exchanges grids between its magazine and the beam.

    The loader owns a **magazine** -- its own storage slots, filled by hand -- which
    is distinct from the holder's working slot(s). ``load_grid`` moves a grid from a
    magazine slot into the working slot and ``unload_grid`` retracts it.

    A magazine slot keeps its grid while that grid is in the beam: the slot is the
    grid's home, and the working slot references the *same* ``SampleGrid``. So
    "present" is read from the magazine and "in beam" from the holder, and neither
    is cached anywhere else.

    Subclasses talk to hardware through ``_do_load`` / ``_do_unload`` /
    ``_scan_magazine`` / ``_write_slot_description``; this base class is the in-memory
    model they all share.
    """

    def __init__(self, parent: "FibsemMicroscope", capacity: int = 12) -> None:
        self.parent = parent
        self.capacity = capacity
        self.slots: dict[str, GridSlot] = {}
        self._ensure_slots()

    def _ensure_slots(self) -> None:
        """Ensure exactly ``capacity`` magazine slots exist, named like holder slots."""
        for i in range(self.capacity):
            name = _slot_name(i)
            if name not in self.slots:
                self.slots[name] = GridSlot(name=name, index=i, position=None)
        for name in [
            n for n, s in list(self.slots.items()) if s.index >= self.capacity
        ]:
            del self.slots[name]

    # -- the holder side ---------------------------------------------------

    @property
    def holder(self) -> SampleHolder:
        return self.parent._stage.holder

    @property
    def working_slot(self) -> GridSlot:
        """The holder slot an exchange loads into (the first one; an autoloader has one)."""
        slots = sorted(self.holder.slots.values(), key=lambda s: s.index)
        if not slots:
            raise GridExchangeError("The sample holder has no working slot.")
        return slots[0]

    @property
    def loaded_slots(self) -> List[GridSlot]:
        """The holder slots that hold a grid. Kept for callers of the old loader."""
        return self.holder.occupied_slots

    # -- the magazine side -------------------------------------------------

    @property
    def loaded_magazine_slots(self) -> List[GridSlot]:
        """Magazine slots that hold a grid: the grids available to load."""
        return [
            s
            for s in sorted(self.slots.values(), key=lambda s: s.index)
            if s.loaded_grid is not None
        ]

    def find_grid(self, grid_name: str) -> Optional[GridSlot]:
        """The magazine slot holding the grid of this name, or None."""
        for slot in self.slots.values():
            if slot.loaded_grid is not None and slot.loaded_grid.name == grid_name:
                return slot
        return None

    def assign_grid(self, slot_name: str, grid: Optional[SampleGrid]) -> None:
        """Name (or clear) the grid in a magazine slot, and tell the hardware."""
        slot = self._magazine_slot(slot_name)
        slot.loaded_grid = grid
        self._write_slot_description(slot)

    def run_inventory(self) -> List[GridSlot]:
        """Ask the hardware which magazine slots hold a grid, then report them."""
        self._scan_magazine()
        return self.loaded_magazine_slots

    # -- exchange ----------------------------------------------------------

    def load_grid(self, slot_name: str) -> SampleGrid:
        """Bring the grid in a magazine slot into the working slot.

        A no-op when that grid is already there; otherwise the working slot is
        emptied first. Raises ``GridExchangeError`` when the slot is empty or the
        hardware refuses.
        """
        slot = self._magazine_slot(slot_name)
        grid = slot.loaded_grid
        if grid is None:
            raise GridExchangeError(f"Magazine slot '{slot_name}' holds no grid.")
        working = self.working_slot
        if working.loaded_grid is not None and working.loaded_grid.name == grid.name:
            return grid
        if working.loaded_grid is not None:
            self.unload_grid()
        self._do_load(slot)
        working.loaded_grid = grid
        logging.info(f"Loaded grid '{grid.name}' from {slot_name} into {working.name}.")
        return grid

    def unload_grid(self) -> None:
        """Retract whatever is in the working slot back into the magazine."""
        working = self.working_slot
        if working.loaded_grid is None:
            return
        name = working.loaded_grid.name
        self._do_unload(working)
        working.loaded_grid = None
        logging.info(f"Unloaded grid '{name}' from {working.name}.")

    # -- hardware hooks ----------------------------------------------------

    def _do_load(self, slot: GridSlot) -> None:
        """Physically move the grid in ``slot`` into the working slot."""

    def _do_unload(self, working_slot: GridSlot) -> None:
        """Physically retract the grid in the working slot."""

    def _scan_magazine(self) -> None:
        """Refresh ``self.slots`` from the hardware. Nothing to scan in memory."""

    def _write_slot_description(self, slot: GridSlot) -> None:
        """Persist a slot's grid name where the hardware keeps it."""

    def _magazine_slot(self, slot_name: str) -> GridSlot:
        try:
            return self.slots[slot_name]
        except KeyError:
            raise GridExchangeError(
                f"No magazine slot '{slot_name}' (capacity {self.capacity})."
            ) from None


class DemoSampleLoader(SampleGridLoader):
    """An in-memory autoloader for the simulator.

    ``occupied`` lists the 1-based magazine slot numbers that hold a grid, as printed
    on a real magazine; ``names`` maps a slot number to a grid name, and any occupied
    slot without one gets ``Grid-NN`` (what a scan reports for an unnamed grid).

    Set ``fail_next_exchange`` to make the next load or unload raise
    ``GridExchangeError`` and leave the state untouched, so the run loop's
    load-failure path can be exercised. ``exchange_delay`` is honoured only when
    non-zero; tests leave it at zero.
    """

    def __init__(
        self,
        parent: "FibsemMicroscope",
        capacity: int = 12,
        occupied: Iterable[int] = (),
        names: Optional[Mapping[Union[int, str], str]] = None,
        exchange_delay: float = 0.0,
    ) -> None:
        super().__init__(parent, capacity)
        self.exchange_delay = exchange_delay
        self.fail_next_exchange = False
        names = names or {}
        for number in occupied:
            number = int(number)
            slot = self.slots.get(_slot_name(number - 1))
            if slot is None:
                raise ValueError(
                    f"Magazine slot {number} is outside capacity {capacity}."
                )
            name = names.get(number, names.get(str(number))) or f"Grid-{number:02d}"
            slot.loaded_grid = SampleGrid(name=str(name))

    def _do_load(self, slot: GridSlot) -> None:
        self._exchange()

    def _do_unload(self, working_slot: GridSlot) -> None:
        self._exchange()

    def _exchange(self) -> None:
        if self.fail_next_exchange:
            self.fail_next_exchange = False
            raise GridExchangeError("Simulated autoloader exchange failure.")
        if self.exchange_delay > 0:
            time.sleep(self.exchange_delay)


def _slot_name(index: int) -> str:
    return f"Slot-{index + 1:02d}"


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
            if slot.position is None:
                continue
            if stage_position.is_close2(
                slot.position, tol=GRID_RADIUS, axes=["x", "y"]
            ):
                return slot
        return None

    @property
    def current_grid(self) -> Optional[SampleGrid]:
        """Get the loaded SampleGrid at the current slot, if any."""
        slot = self.current_slot
        return slot.loaded_grid if slot is not None else None

    def grid_inventory(self) -> List[GridInventoryEntry]:
        """Which grids exist this session, and which are in the beam. Derived, not stored.

        With a loader the rows are the magazine slots and "in beam" means the grid is
        in a holder working slot. Without one the rows are the holder slots, and every
        present grid is in the beam, because reaching it is only a stage move. Callers
        never need to know which case they got.
        """
        if self.loader is not None:
            in_beam = {s.loaded_grid.name for s in self.holder.occupied_slots}
            slots = sorted(self.loader.slots.values(), key=lambda s: s.index)
            source = "magazine"
        else:
            in_beam = None
            slots = sorted(self.holder.slots.values(), key=lambda s: s.index)
            source = "holder"

        entries: List[GridInventoryEntry] = []
        for slot in slots:
            grid = slot.loaded_grid
            present = grid is not None
            entries.append(
                GridInventoryEntry(
                    slot_name=slot.name,
                    index=slot.index,
                    source=source,
                    name=grid.name if grid is not None else None,
                    present=present,
                    in_beam=present and (in_beam is None or grid.name in in_beam),  # type: ignore[union-attr]
                )
            )
        return entries

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

    def vertical_move(
        self, dy: float, dx: float = 0.0, beam_type: BeamType = BeamType.ION
    ) -> FibsemStagePosition:
        return self.parent.vertical_move(dy, dx, beam_type)

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
        """Move the stage to the holder slot that holds this grid."""
        slot = self.holder.find_slot_by_grid_name(grid_name)
        if slot is None:
            raise ValueError(f"Grid '{grid_name}' is not in any holder slot.")
        return self.move_to_slot(slot.name)

    # -- the one "make reachable" primitive --------------------------------

    @property
    def loaded_grids(self) -> List[SampleGrid]:
        """The grids in the beam right now, read from the holder every time."""
        return [s.loaded_grid for s in self.holder.occupied_slots]  # type: ignore[misc]

    def ensure_loaded(self, grid_name: str) -> GridSlot:
        """Make a grid reachable, and return the working slot it occupies.

        A no-op when the grid is already in a working slot. With a loader it is a
        magazine exchange (the current grid is retracted first); without one every
        present grid already sits in a holder slot, so there is nothing to do but
        confirm it is there. Raises ``GridExchangeError`` when the grid is not in the
        inventory or the hardware refuses.

        This does not move the stage: how the grid gets under the beam is the
        hardware's business, and where on the grid to go is the caller's. Follow it
        with ``move_to_slot(slot.name)`` (or a task's own positioning).
        """
        working = self.holder.find_slot_by_grid_name(grid_name)
        if working is not None:
            return working
        if self.loader is None:
            raise GridExchangeError(
                f"Grid '{grid_name}' is not in any holder slot. Place it in the "
                "holder and update the sample holder configuration."
            )
        home = self.loader.find_grid(grid_name)
        if home is None:
            raise GridExchangeError(
                f"Grid '{grid_name}' is not in the magazine. Run an inventory."
            )
        self.loader.load_grid(home.name)
        return self.loader.working_slot

    def unload(self) -> None:
        """Retract the working slot. Nothing to do on a fixed holder."""
        if self.loader is None:
            logging.debug("No loader: nothing to unload from a fixed holder.")
            return
        self.loader.unload_grid()


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
        # The compustage is the autoloader stage, so it is also what says "this system
        # has a loader". Which loader is the backend's call: the simulator builds one
        # from its config, a real system wraps its autoloader.
        loader: Optional[SampleGridLoader] = microscope._create_grid_loader()
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
