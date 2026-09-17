from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import yaml
from psygnal import Signal

from fibsem.config import (
    SAMPLE_HOLDER_CONFIGURATION_PATH,
    SAMPLE_HOLDER_OCCUPANCY_PATH,
)
from fibsem.structures import (
    GRID_RADIUS,
    BeamType,
    FibsemStagePosition,
    GridSlot,
    RangeLimit,
    SampleGrid,
    SampleHolder,
    SlotCalibration,
    default_sample_holder,
)

# Re-exported: these four moved into `structures.py` so `SystemSettings` could hold a
# holder without closing an import loop. They are still part of this module's surface
# -- around twenty call sites import them from here.
__all__ = [
    "GRID_RADIUS",
    "SampleGrid",
    "SlotCalibration",
    "GridSlot",
    "SampleHolder",
]

if TYPE_CHECKING:
    from fibsem.microscope import FibsemMicroscope


class GridExchangeError(RuntimeError):
    """A grid could not be moved into, or out of, the holder's working slot."""


class GridSlotState(str, Enum):
    """Where a grid is, as one word: the state of an inventory slot.

    ``UNKNOWN`` -- no inventory has answered for this slot yet; ``EMPTY`` -- the
    slot holds no grid; ``OCCUPIED`` -- a grid is in the slot and not on the
    microscope; ``LOADED`` -- the grid is in the microscope's sample holder. On a
    fixed holder a named slot is LOADED outright: its grid is already in the
    holder, and reaching it is only a stage move.
    """

    UNKNOWN = "unknown"
    EMPTY = "empty"
    OCCUPIED = "occupied"
    LOADED = "loaded"


@dataclass
class GridInventoryEntry:
    """One row of ``Stage.grid_inventory()``: a slot, and what it holds.

    ``source`` says which hardware answered -- ``"magazine"`` on a system with a
    loader, ``"holder"`` otherwise. ``state`` is the one stored fact; ``present``
    (the grid is available to the system) and ``loaded`` (it is in the holder
    right now) are read off it. Kept apart on purpose: *present* is what a run
    can select from, *loaded* is what it can act on without an exchange.
    """

    slot_name: str
    index: int
    source: str
    name: Optional[str]
    state: GridSlotState

    @property
    def present(self) -> bool:
        return self.state in (GridSlotState.OCCUPIED, GridSlotState.LOADED)

    @property
    def loaded(self) -> bool:
        return self.state is GridSlotState.LOADED


class SampleGridLoader:
    """The robotic actuator that exchanges grids between its magazine and the beam.

    The loader owns a **magazine** -- its own storage slots, filled by hand -- which
    is distinct from the holder's working slot(s). ``load_grid`` moves a grid from a
    magazine slot into the working slot and ``unload_grid`` retracts it.

    A magazine slot keeps its grid while that grid is loaded: the slot is the
    grid's home, and the working slot references the *same* ``SampleGrid``. So
    "present" is read from the magazine and "loaded" from the holder, and neither
    is cached anywhere else.

    Subclasses talk to hardware through ``_do_load`` / ``_do_unload`` /
    ``_scan_magazine`` / ``_write_slot_description``; this base class is the in-memory
    model they all share.
    """

    def __init__(self, parent: "FibsemMicroscope", capacity: int = 12) -> None:
        self.parent = parent
        self.capacity = capacity
        self.slots: dict[str, GridSlot] = {}
        # Until the hardware has answered, every slot is UNKNOWN; after a scan the
        # slots it still could not read stay so. An in-memory magazine is known
        # from the start.
        self.scanned = False
        self.unknown_slots: set = set()
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

    def get_inventory(self) -> List[GridSlot]:
        """Read what the autoloader already knows about its magazine: instant,
        nothing moves. Its answer is only as fresh as its own last scan."""
        self._read_magazine()
        self.scanned = True
        return self.loaded_magazine_slots

    def run_inventory(self) -> List[GridSlot]:
        """Have the autoloader scan the magazine, slot by slot: slow, and the
        magazine must stay shut while it runs. Then report what it found."""
        self._scan_magazine()
        self.scanned = True
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

    def _read_magazine(self) -> None:
        """Refresh ``self.slots`` from what the hardware already knows. Nothing to
        read in memory: the slots are the state."""

    def _scan_magazine(self) -> None:
        """Refresh ``self.slots`` from a physical scan. Nothing to scan in memory."""

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
        self.scanned = True  # an in-memory magazine is known from the start

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
        # The cached position, never a hardware read: this is called from UI
        # paint paths. It is None until the first read or move after connecting,
        # and then the answer is "not known to be at any slot", not a crash.
        stage_position = self.parent._stage_position
        if stage_position is None:
            return None
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
        """Which grids exist this session, and which are loaded. Derived, not stored.

        With a loader the rows are the magazine slots and "loaded" means the grid is
        in a holder working slot. Without one the rows are the holder slots, and every
        present grid is loaded, because reaching it is only a stage move. Callers
        never need to know which case they got.
        """
        loader = self.loader
        if loader is not None:
            loaded = {s.loaded_grid.name for s in self.holder.occupied_slots}
            slots = sorted(loader.slots.values(), key=lambda s: s.index)
            source = "magazine"
        else:
            loaded = None
            slots = sorted(self.holder.slots.values(), key=lambda s: s.index)
            source = "holder"

        entries: List[GridInventoryEntry] = []
        for slot in slots:
            grid = slot.loaded_grid
            if loader is not None and (
                not loader.scanned or slot.name in loader.unknown_slots
            ):
                state = GridSlotState.UNKNOWN
            elif grid is None:
                state = GridSlotState.EMPTY
            elif loaded is None or grid.name in loaded:
                state = GridSlotState.LOADED
            else:
                state = GridSlotState.OCCUPIED
            entries.append(
                GridInventoryEntry(
                    slot_name=slot.name,
                    index=slot.index,
                    source=source,
                    name=grid.name if grid is not None else None,
                    state=state,
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
        """Move the stage to a specific slot. Refuses a slot that is not calibrated."""
        if self.holder is None:
            raise ValueError("No sample holder defined.")
        if slot_name not in self.holder.slots:
            raise ValueError(f"Slot '{slot_name}' not found in sample holder.")
        slot = self.holder.slots[slot_name]
        if slot.position is None:
            raise ValueError(uncalibrated_message(slot_name))
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
        """The grids loaded right now, read from the holder every time."""
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
            if working.position is None and self.loader is None:
                # Present, but nothing can be sent to it: on a fixed holder "in the
                # beam" only means anything once the slot has a trusted position.
                raise GridExchangeError(uncalibrated_message(working.name))
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

    # -- naming and refreshing the inventory, on either shape --------------

    def get_inventory(self) -> List[GridInventoryEntry]:
        """Read what the hardware knows and return the inventory: instant.

        With a loader, the autoloader's own record of its magazine, as fresh as
        its last scan (xT's inventory counts). On a fixed holder occupancy is what
        the operator declared, so this is a plain refresh.
        """
        if self.loader is not None:
            self.loader.get_inventory()
        return self.grid_inventory()

    def run_inventory(self) -> List[GridInventoryEntry]:
        """Scan the magazine and return the inventory: the slow one.

        With a loader the autoloader checks every slot physically, which takes a
        while and needs the magazine shut. On a fixed holder there is nothing to
        scan, so it is the same refresh as ``get_inventory``.
        """
        if self.loader is not None:
            self.loader.run_inventory()
        return self.grid_inventory()

    def assign_grid(
        self, slot_name: str, grid: Optional[SampleGrid], persist: bool = True
    ) -> None:
        """Name (or clear) the grid in an inventory slot, and keep it.

        With a loader the slot is a magazine slot and the name goes to the hardware's
        slot description. On a fixed holder the slot is a holder slot and the name is
        saved to the occupancy file, so it is there next session; the calibration file
        is not touched. Pass ``persist=False`` to change only the in-memory holder.
        """
        if self.loader is not None:
            self.loader.assign_grid(slot_name, grid)
            return
        slot = self.holder.slots.get(slot_name)
        if slot is None:
            raise ValueError(f"Slot '{slot_name}' not found in sample holder.")
        slot.loaded_grid = grid
        if persist:
            self.holder.save_occupancy(SAMPLE_HOLDER_OCCUPANCY_PATH)


def uncalibrated_message(slot_name: str) -> str:
    return (
        f"{slot_name} has no calibrated position. Run 'Calibrate slot positions' "
        "in the Sample Holder panel on the Microscope tab."
    )


def _resolve_configured_holder(stage_settings) -> SampleHolder:
    """The holder on the stage: from the configuration, or imported into it.

    Three cases, in order.

    **The configuration names one.** `stage.holders` with an `active_holder` that
    picks one out of it. This is where a system ends up once its configuration has
    been saved, and the only case that involves no files.

    **It does not, and the site has a `sample-holder.yaml`.** The holder moved into the
    microscope configuration, but every calibrated site has its slot positions in the
    old file and those are not reproducible -- someone stood at the microscope and
    captured them. So the file is imported as an entry and selected, and the site keeps
    its calibration without being shown a list it did not ask for.

    Read-only: nothing is written back here. The imported holder is on
    `stage.holders`, so the next save of the configuration carries it, but a session
    that saves nothing leaves both files exactly as it found them. Re-importing every
    session costs nothing and is safer than rewriting a user's configuration on their
    behalf at connect time.

    **Neither.** The shipped default, as before.
    """
    active = stage_settings.active_holder
    configured = stage_settings.holders.get(active) if active else None
    if configured is not None:
        return configured

    configured_pre_tilt = float(stage_settings.shuttle_pre_tilt)
    path = Path(SAMPLE_HOLDER_CONFIGURATION_PATH)

    if path.exists():
        holder = SampleHolder.load(path)
        logging.info(
            f"Imported sample holder '{holder.name}' from {path} for this session. "
            "The file is not written back; it is imported again at every connect."
        )
        # The pre-tilt is the configured one -- always, not just when the file is
        # silent. A holder file may *carry* a `pre_tilt`: they did once, and it has
        # been ignored ever since the value became derived from the stage. Honouring
        # it now would resurrect a number that has not been in effect for however long
        # the file has sat there, and do it silently, in the term every projection is
        # built on. It becomes the holder's own from the moment the configuration is
        # saved, which is the point at which someone has seen it.
        holder.pre_tilt = configured_pre_tilt
    else:
        logging.info("No sample holder configuration found, using the default.")
        holder = default_sample_holder(pre_tilt=configured_pre_tilt)

    # Selected either way, so a session that saves its configuration records which
    # holder it was actually using rather than an empty selection.
    stage_settings.holders[holder.name] = holder
    stage_settings.active_holder = holder.name
    return holder


def _create_sample_stage(microscope: "FibsemMicroscope") -> "Stage":
    if microscope.stage_is_compustage:
        # The working slot is the compustage origin by construction: the loader puts
        # every grid at the same place and the coordinate system is referenced to
        # it. That is a hardware fact, so the slot is calibrated without a capture.
        stage_settings = microscope.system.stage
        slot01 = GridSlot(
            name="Slot-01",
            index=0,
            position=FibsemStagePosition(
                name="Slot-01", x=0.0, y=0.0, z=0.0, r=0.0, t=np.radians(0)
            ),
            calibration=SlotCalibration.builtin(
                float(stage_settings.shuttle_pre_tilt),
                float(stage_settings.rotation_reference),
            ),
        )
        holder = SampleHolder(
            name="CompuStage Holder",
            capacity=1,
            slots={"Slot-01": slot01},
            # Built here rather than resolved from the configuration, so it has to be
            # given its pre-tilt explicitly -- it does not pass through
            # `_resolve_configured_holder`, which is where every other holder gets one.
            pre_tilt=float(stage_settings.shuttle_pre_tilt),
        )
        # The compustage is the autoloader stage, so it is also what says "this system
        # has a loader". Which loader is the backend's call: the simulator builds one
        # from its config, a real system wraps its autoloader.
        loader: Optional[SampleGridLoader] = microscope._create_grid_loader()
    else:
        holder = _resolve_configured_holder(microscope.system.stage)
        # Trust only positions the wizard captured against this stage geometry. The
        # old stamping of SEM r/t onto whatever x/y/z the file held is gone: a
        # calibrated position carries its own r/t, and an uncalibrated one has none.
        stage_settings = microscope.system.stage
        for note in holder.discard_untrusted_positions(
            float(stage_settings.shuttle_pre_tilt),
            float(stage_settings.rotation_reference),
        ):
            logging.warning(f"Sample holder: {note}. Recalibrate it.")
        # The grids in the slots are session state, remembered in their own file so
        # a restart does not forget what is physically still in the shuttle.
        holder.load_occupancy(SAMPLE_HOLDER_OCCUPANCY_PATH)
        loader = None

    holder._parent = microscope
    return Stage(parent=microscope, holder=holder, loader=loader)
