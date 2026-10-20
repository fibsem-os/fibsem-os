"""Sample handling as a service: the holder on the stage, the grids in it and in the
loader's magazine, and the commands that make a grid reachable.

`SampleHandling` is the same on every backend and every kind of system, and callers
never ask which they have:

- **No loader** (a flat stage, or a stage with a built-in shuttle whose grids are
  exchanged by hand): the inventory is the holder's slots, their occupancy is what the
  operator declared (kept in the session state), every present grid is already loaded,
  ``ensure_loaded`` only checks its slot is calibrated and ``unload`` does nothing.
- **A loader** (a stage with a built-in shuttle and an autoloader): the inventory is
  the loader's magazine, ``ensure_loaded`` is an exchange through the ``sample_loader``
  device, and the holder is the built-in shuttle's one working slot.

The holder is configuration and session state, not a device: which holder is on the
stage and where its slots are come from the microscope configuration, and which grid is
in which slot from the session state. The grid model of the magazine is a
`GridMagazine` (``sample_handling.magazine``), over the ``sample_loader`` device in the
``loader`` role.

Every microscope with a stage has one, ``microscope.sample_handling``, built by
`bind_sample_handling`; a microscope without a stage has none.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import yaml
from psygnal import Signal

from fibsem import config as cfg
from fibsem.config import SAMPLE_HOLDER_CONFIGURATION_PATH
from fibsem.devices.core import Role, command, query, session_of
from fibsem.devices.sample_loader import (
    GridExchangeError,
    Magazine,
    MagazineSlotState,
    SampleLoader,
    StageSample,
)
from fibsem.devices.stage import Stage
from fibsem.services.core import Service
from fibsem.session_state import SessionState, session_state_for
from fibsem.structures import (
    GRID_RADIUS,
    FibsemStagePosition,
    GridInventoryEntry,
    GridSlot,
    GridSlotState,
    SampleGrid,
    SampleHolder,
    SlotCalibration,
    default_sample_holder,
)
from fibsem.util import wording

if TYPE_CHECKING:
    from fibsem.microscope import FibsemMicroscope


class GridMagazine:
    """The grids in a loader's magazine, and the exchanges between it and the holder.

    The magazine is the loader's own storage slots, filled by hand, distinct from the
    holder's working slot(s). ``load_grid`` moves a grid from a magazine slot into the
    working slot and ``unload_grid`` retracts it.

    A magazine slot keeps its grid while that grid is loaded: the slot is the grid's
    home, and the working slot references the *same* ``SampleGrid``. So "present" is
    read from the magazine and "loaded" from the holder, and neither is cached anywhere
    else.

    Subclasses talk to hardware through ``_do_load`` / ``_do_unload`` /
    ``_read_magazine`` / ``_scan_magazine`` / ``_write_slot_description``; this base
    class is the in-memory model they all share.
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

    @property
    def exchange_seconds(self) -> float:
        """How long one exchange takes, the unload and the load, in seconds: what a
        run's time estimate charges per grid it brings in. Declared by each loader,
        since it is the hardware's. Nothing for this in-memory one."""
        return 0.0

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
        return self.parent.sample_handling.holder

    @property
    def working_slot(self) -> GridSlot:
        """The holder slot an exchange loads into (the first one; an autoloader has one)."""
        slots = sorted(self.holder.slots.values(), key=lambda s: s.index)
        if not slots:
            raise GridExchangeError("The sample holder has no working slot.")
        return slots[0]

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
        """Name (or clear) the grid in a magazine slot, and tell the hardware.

        A loaded grid is the magazine slot's own record, so the working slot
        follows the rename. A write the hardware refuses puts both back and
        raises: the next read would bring the old name back anyway."""
        slot = self._magazine_slot(slot_name)
        previous = slot.loaded_grid
        working = next(
            (
                s
                for s in self.holder.slots.values()
                if previous is not None and s.loaded_grid is previous
            ),
            None,
        )
        slot.loaded_grid = grid
        if working is not None:
            working.loaded_grid = grid
        try:
            self._write_slot_description(slot)
        except Exception:
            slot.loaded_grid = previous
            if working is not None:
                working.loaded_grid = previous
            raise

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
        # On the record (FIB-1359), with how long the exchange took and why it
        # failed, if it did.
        with session_of(self.parent).recording(
            "grid_loaded", grid=grid.name, slot=slot_name, working_slot=working.name
        ):
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
        home = self.find_grid(name)
        with session_of(self.parent).recording(
            "grid_unloaded",
            grid=name,
            slot=home.name if home is not None else None,
            working_slot=working.name,
        ):
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


class DeviceGridMagazine(GridMagazine):
    """The grid model over a ``SampleLoader`` device (``fibsem.devices.sample_loader``).

    The device reports what the hardware knows -- numbered slots, their states and
    descriptions, what is on the stage -- and does the exchanges. This turns that into
    grids: a slot's description is its grid's name (``Grid-NN`` when blank), a grid
    keeps its identity across reads, and the grid on the stage is the same object in
    its home slot and the working slot.

    Two hardware behaviours are absorbed here, whichever driver reports them: the home
    slot of the grid on the stage reads ``LOADED`` (AutoScript 4.14) or ``EMPTY``
    (up to 4.13, where only memory keeps the grid in its home slot); and the stage may
    hold a grid nobody loaded through us, which ``on_stage`` reports.

    ``read_at_connect`` reads the magazine once when built, for a loader whose read
    is free (the simulator); otherwise nothing is known until ``get_inventory``.
    """

    def __init__(
        self,
        parent: "FibsemMicroscope",
        device,
        read_at_connect: bool = False,
    ) -> None:
        self.device = device
        super().__init__(parent, capacity=int(device.capacity.get_value()))
        if read_at_connect:
            self.get_inventory()

    @property
    def exchange_seconds(self) -> float:
        return float(self.device.exchange_time.cached)

    def _read_magazine(self) -> None:
        self._apply(self.device.magazine.get_value())

    def _scan_magazine(self) -> None:
        self._apply(self.device.scan())

    def _do_load(self, slot: GridSlot) -> None:
        self.device.load(slot.index + 1)

    def _do_unload(self, working_slot: GridSlot) -> None:
        self.device.unload()

    def _write_slot_description(self, slot: GridSlot) -> None:
        description = slot.loaded_grid.name if slot.loaded_grid is not None else ""
        self.device.set_description(slot.index + 1, description)

    def _on_stage(self) -> StageSample:
        if "on_stage" not in self.device.parameters:
            return StageSample()
        return self.device.on_stage.get_value()

    def _apply(self, magazine: Magazine) -> None:
        if magazine.slots:
            self.capacity = len(magazine.slots)
        # A read at connect comes before the service, and so the holder, exists.
        holder = getattr(getattr(self.parent, "sample_handling", None), "holder", None)
        loaded = (
            {s.loaded_grid.name for s in holder.occupied_slots} if holder else set()
        )
        slots: dict = {}
        unknown: set = set()
        on_stage: Optional[SampleGrid] = None
        for row in magazine.slots:
            name = _slot_name(row.number - 1)
            previous = self.slots.get(name)
            grid: Optional[SampleGrid] = None
            if row.state in (MagazineSlotState.OCCUPIED, MagazineSlotState.LOADED):
                grid_name = row.description.strip() or f"Grid-{row.number:02d}"
                if previous is not None and previous.loaded_grid is not None:
                    if previous.loaded_grid.name == grid_name:
                        grid = previous.loaded_grid  # keep identity across reads
                if grid is None:
                    grid = SampleGrid(name=grid_name)
                if row.state is MagazineSlotState.LOADED:
                    on_stage = grid  # its home slot, and it is on the stage
            elif (
                previous is not None
                and previous.loaded_grid is not None
                and previous.loaded_grid.name in loaded
            ):
                grid = (
                    previous.loaded_grid
                )  # <= 4.13: its home reads Empty while loaded
            elif row.state is MagazineSlotState.UNKNOWN:
                logging.warning(
                    f"Sample loader slot {row.number} has not been scanned."
                )
                unknown.add(name)
            slots[name] = GridSlot(name=name, index=row.number - 1, loaded_grid=grid)
        self.slots = slots
        self.unknown_slots = unknown
        if holder is None:
            return

        working = self.working_slot
        if on_stage is not None:
            # The same object in both places, as a load through us leaves it.
            working.loaded_grid = on_stage
        # Something may be on the stage that we did not load: reflect the hardware.
        reported = self._on_stage()
        if working.loaded_grid is None and reported.present is True:
            working.loaded_grid = SampleGrid(
                name=reported.description.strip() or "Grid-on-stage"
            )
        elif working.loaded_grid is not None and reported.present is False:
            working.loaded_grid = None


def _slot_name(index: int) -> str:
    return f"Slot-{index + 1:02d}"


# --- A command as the operator reads it, in an agent's request (wording) ---


def _loaded(inventory: List[Dict[str, Any]]) -> Optional[str]:
    """The grid on the stage now, by name."""
    return next((e.get("name") for e in inventory if e.get("state") == "loaded"), None)


# A load the inventory does not hold: an agent's request for it is refused.
NOT_IN_INVENTORY = ": it is not in the inventory"


def _grid_load(name: str, inventory: Any) -> str:
    """``Load grid 'G2' from Slot-02 (unloads 'G1' first)``: the exchange the
    loader will make, from the inventory as it reads now."""
    change = f"Load grid '{name}'"
    if not isinstance(inventory, list) or not inventory:
        return change
    entry = next((e for e in inventory if e.get("name") == name), None)
    if entry is None:
        return change + NOT_IN_INVENTORY
    if entry.get("source") != "magazine":
        return change + f": it is in {entry.get('slot_name')} already, nothing moves"
    if entry.get("state") == "loaded":
        return change + ": it is on the stage already, nothing moves"
    change += f" from {entry.get('slot_name')}"
    current = _loaded(inventory)
    return change + (f" (unloads '{current}' first)" if current else "")


def _grid_unload(inventory: Any) -> str:
    if not isinstance(inventory, list) or not inventory:
        return "Return the grid on the stage to the magazine"
    if inventory[0].get("source") != "magazine":
        return "Unload: nothing to do on a fixed holder"
    current = _loaded(inventory)
    if current is None:
        return "Unload: no grid is on the stage"
    return f"Return grid '{current}' to the magazine"


def _slot_move(change: str, slot: Optional[Dict[str, Any]], now: Any) -> str:
    """A move to a holder slot, with where that is and where the stage is now."""
    target = slot.get("position") if slot else None
    if target:
        change += f": {wording.position_text(target)}"
    elif slot is not None:
        change += ": the slot is not calibrated, so it will be refused"
    if target and now:
        change += f"\nfrom {wording.position_text(now)}"
    return change


def _queried(ctx: wording.ActionContext, query: str) -> Any:
    return wording.result_of(
        ctx.get(f"/microscope/services/sample_handling/queries/{query}")
    )


def _load_line(kwargs: Dict[str, Any], ctx: wording.ActionContext) -> Dict[str, Any]:
    inventory = _queried(ctx, "get_inventory")
    return {
        "what": "Grids",
        "change": _grid_load(str(kwargs.get("grid_name")), inventory),
    }


def _unload_line(kwargs: Dict[str, Any], ctx: wording.ActionContext) -> Dict[str, Any]:
    return {"what": "Grids", "change": _grid_unload(_queried(ctx, "get_inventory"))}


def _inventory_line(
    kwargs: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    return {
        "what": "Grids",
        "change": "Scan the magazine slot by slot, to read which grids it holds",
    }


def _slots(ctx: wording.ActionContext) -> List[Dict[str, Any]]:
    slots = _queried(ctx, "holder_slots")
    return slots if isinstance(slots, list) else []


def _to_grid_line(kwargs: Dict[str, Any], ctx: wording.ActionContext) -> Dict[str, Any]:
    slots = _slots(ctx)
    name = str(kwargs.get("grid_name"))
    change = f"Move to grid '{name}'"
    slot = next(
        (s for s in slots if (s.get("loaded_grid") or {}).get("name") == name),
        None,
    )
    if slot is not None:
        change += f", in {slot.get('name')}"
    elif slots:
        # As of now: an earlier step of the same request may load it.
        change += " (not on the stage yet)"
    return {"what": "Stage", "change": _slot_move(change, slot, ctx.stage_now())}


def _to_slot_line(kwargs: Dict[str, Any], ctx: wording.ActionContext) -> Dict[str, Any]:
    name = str(kwargs.get("slot_name"))
    slot = next((s for s in _slots(ctx) if s.get("name") == name), None)
    change = _slot_move(f"Move to slot '{name}'", slot, ctx.stage_now())
    return {"what": "Stage", "change": change}


class SampleHandling(Service):
    """Which grid is where, and making a grid reachable.

    The holder (``holder``), the grid model of the loader's magazine when there is a
    loader (``magazine``), and the commands over both. Stage moves are the stage
    device's (``microscope.stage``) and the stage movement service's
    (``microscope.stage_movement``); this only adds the moves that need the holder, to
    a slot or to a grid.
    """

    stage = Role(Stage, doc="The stage the holder sits on.")
    loader = Role(
        SampleLoader,
        required=False,
        doc="The autoloader, where there is one: its magazine is the inventory.",
    )

    # The sample holder changed: slots calibrated, or another holder loaded from the
    # configuration (`load_holder`). Carries the holder. Emitted on the caller's
    # thread.
    holder_changed = Signal(object)

    remote_signals = ("holder_changed",)

    def __init__(self, name: str = "sample_handling", **kwargs: Any):
        super().__init__(name, **kwargs)
        self.holder: Optional[SampleHolder] = None
        self.magazine: Optional[GridMagazine] = None
        # Whether `holder` is the stage's built-in shuttle, which this built, rather
        # than one the configuration describes.
        self.holder_is_builtin = False

    def __repr__(self) -> str:
        return f"<SampleHandling: holder={self.holder}, magazine={self.magazine}>"

    def connect(self) -> "SampleHandling":
        """Bind the service, then build its holder (`load_holder`)."""
        super().connect()
        self.load_holder()
        return self

    def load_holder(self) -> None:
        """Build the holder from the configuration, and the magazine's grid model over
        the loader if there is one; then restore the occupancy the session state
        records, once the configuration is known, and announce the holder on
        ``holder_changed``. Applying a configuration that names a holder calls this
        again, so subscribers stay connected."""
        microscope = self.parent
        stage_settings = microscope.system.stage
        builtin = self.stage.has_builtin_shuttle()
        if builtin:
            working = _builtin_working_slot(stage_settings)
            holder = SampleHolder(
                name=BUILTIN_HOLDER_NAME,
                capacity=1,
                slots={working.name: working},
                # Built here rather than resolved from the configuration, so it has to
                # be given its pre-tilt explicitly -- it does not pass through
                # `_resolve_configured_holder`, which is where every other holder gets
                # one.
                pre_tilt=float(stage_settings.shuttle_pre_tilt),
            )
            # A built-in shuttle is the only kind an autoloader loads, so it is the only
            # stage whose grids come from a magazine; without a loader they are
            # exchanged by hand.
            loader = self._roles.get("loader")
            magazine: Optional[GridMagazine] = None
            if loader is not None:
                # Read when built only where that read is free (the simulator).
                magazine = DeviceGridMagazine(
                    microscope,
                    loader,
                    read_at_connect=bool(
                        getattr(microscope, "_read_magazine_at_connect", False)
                    ),
                )
        else:
            holder = _resolve_configured_holder(stage_settings)
            # Trust only positions the wizard captured against this stage geometry. A
            # calibrated position carries its own r/t, and an uncalibrated one has
            # none.
            for note in holder.discard_untrusted_positions(
                float(stage_settings.shuttle_pre_tilt),
                float(stage_settings.rotation_reference),
            ):
                logging.warning(f"Sample holder: {note}. Recalibrate it.")
            magazine = None

        holder._parent = microscope
        self.holder = holder
        self.holder_is_builtin = builtin
        self.magazine = magazine
        # The grids in the slots are session state, so a restart does not forget what
        # is physically still in the shuttle. Restored here when the service is
        # rebuilt; at the first connect the configuration is not known yet, and
        # `setup_session` restores it once it is.
        if getattr(microscope, "configuration_path", None):
            self.restore_occupancy()
        self.holder_changed.emit(holder)

    # -- where the stage is, against the holder -----------------------------

    def slot_at_position(
        self, position: Optional[FibsemStagePosition]
    ) -> Optional[GridSlot]:
        """The holder slot whose calibrated position *position* falls within, if any.

        Within ``GRID_RADIUS`` in x and y; a slot with no calibrated position is
        never matched. None for no position, so a caller with nothing to ask
        about gets "not known" rather than a guess.
        """
        if self.holder is None or position is None:
            return None
        for slot in self.holder.slots.values():
            if slot.position is None:
                continue
            if position.is_close2(slot.position, tol=GRID_RADIUS, axes=["x", "y"]):
                return slot
        return None

    def grid_at_position(
        self, position: Optional[FibsemStagePosition]
    ) -> Optional[SampleGrid]:
        """The grid in the slot *position* falls within, if any."""
        slot = self.slot_at_position(position)
        return slot.loaded_grid if slot is not None else None

    @property
    def current_slot(self) -> Optional[GridSlot]:
        """Get the slot the stage is currently positioned at, if any."""
        # The stage device's cached position, never a hardware read: this is called
        # from UI paint paths. It is None until the first read or move after
        # connecting, and then the answer is "not known to be at any slot", not a crash.
        stage = self.parent.stage
        return self.slot_at_position(stage.position.last_known if stage else None)

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
        magazine = self.magazine
        if magazine is not None:
            loaded = {s.loaded_grid.name for s in self.holder.occupied_slots}
            slots = sorted(magazine.slots.values(), key=lambda s: s.index)
            source = "magazine"
        else:
            loaded = None
            slots = sorted(self.holder.slots.values(), key=lambda s: s.index)
            source = "holder"

        entries: List[GridInventoryEntry] = []
        for slot in slots:
            grid = slot.loaded_grid
            if magazine is not None and (
                not magazine.scanned or slot.name in magazine.unknown_slots
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

    @command(line=_to_slot_line)
    def move_to_slot(self, slot_name: str) -> FibsemStagePosition:
        """Move the stage to a specific slot. Refuses a slot that is not calibrated."""
        if self.holder is None:
            raise ValueError("No sample holder defined.")
        slot_name = self._holder_slot_name(slot_name)
        if slot_name not in self.holder.slots:
            raise ValueError(f"Slot '{slot_name}' not found in sample holder.")
        slot = self.holder.slots[slot_name]
        if slot.position is None:
            raise ValueError(uncalibrated_message(slot_name))
        return self.parent.safe_absolute_stage_movement(slot.position)

    def _holder_slot_name(self, slot_name: str) -> str:
        """The shuttle's slot under the name a script written before it had its
        own may still use ("Slot-01")."""
        if self.holder_is_builtin and slot_name == _LEGACY_BUILTIN_SLOT_NAME:
            return BUILTIN_SLOT_NAME
        return slot_name

    @command(line=_to_grid_line)
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

    @command(long=True, line=_load_line)
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
            if working.position is None and self.magazine is None:
                # Present, but nothing can be sent to it: on a fixed holder "in the
                # beam" only means anything once the slot has a trusted position.
                raise GridExchangeError(uncalibrated_message(working.name))
            return working
        if self.magazine is None:
            raise GridExchangeError(
                f"Grid '{grid_name}' is not in any holder slot. Place it in the "
                "holder and update the sample holder configuration."
            )
        home = self.magazine.find_grid(grid_name)
        if home is None:
            raise GridExchangeError(
                f"Grid '{grid_name}' is not in the magazine. Run an inventory."
            )
        self.magazine.load_grid(home.name)
        return self.magazine.working_slot

    @command(long=True, line=_unload_line)
    def unload(self) -> None:
        """Retract the working slot. Nothing to do on a fixed holder."""
        if self.magazine is None:
            logging.debug("No loader: nothing to unload from a fixed holder.")
            return
        self.magazine.unload_grid()

    # -- naming and refreshing the inventory, on either shape --------------

    @query
    def get_inventory(self) -> List[GridInventoryEntry]:
        """Read what the hardware knows and return the inventory: instant.

        With a loader, the autoloader's own record of its magazine, as fresh as
        its last scan (xT's inventory counts). On a fixed holder occupancy is what
        the operator declared, so this is a plain refresh.
        """
        if self.magazine is not None:
            self.magazine.get_inventory()
        return self.grid_inventory()

    @query
    def current(self) -> Dict[str, Optional[str]]:
        """Where the stage is now: the holder slot it is at and the grid loaded
        there, by name, each None when it is at none. Reads the stage."""
        slot = self.slot_at_position(self.parent.get_stage_position())
        grid = slot.loaded_grid if slot is not None else None
        return {
            "slot": slot.name if slot is not None else None,
            "grid": grid.name if grid is not None else None,
        }

    @query
    def holder_slots(self) -> List[GridSlot]:
        """The holder's slots in order: each one's stage position (None until it
        is calibrated), its calibration and the grid loaded in it."""
        return sorted(self.holder.slots.values(), key=lambda slot: slot.index)

    @query
    def overview_slots(self) -> List[GridSlot]:
        """The slots a grid overview is centred on. With a loader, its working
        slot. On a fixed holder, the occupied calibrated slots, or every
        calibrated one when none is occupied. Empty before a holder is loaded."""
        if self.holder is None:
            return []
        calibrated = [s for s in self.holder_slots() if s.position is not None]
        if self.magazine is not None:
            return calibrated[:1]
        occupied = [s for s in calibrated if s.loaded_grid is not None]
        return occupied or calibrated

    @command(long=True, line=_inventory_line)
    def run_inventory(self) -> List[GridInventoryEntry]:
        """Scan the magazine and return the inventory: the slow one.

        With a loader the autoloader checks every slot physically, which takes a
        while and needs the magazine shut. On a fixed holder there is nothing to
        scan, so it is the same refresh as ``get_inventory``.
        """
        if self.magazine is not None:
            self.magazine.run_inventory()
        return self.grid_inventory()

    @command
    def assign_grid(
        self, slot_name: str, grid: Optional[SampleGrid], persist: bool = False
    ) -> None:
        """Name (or clear) the grid in an inventory slot.

        With a loader the slot is a magazine slot and the name goes to the hardware's
        slot description. On a fixed holder the slot is a holder slot; with
        ``persist=True`` the occupancy is also saved to the session state, so it is
        there next session. The application persists; a script leaves it off, so it
        cannot rewrite what the operator declared is in the shuttle. The
        calibration is never touched.
        """
        if self.magazine is not None:
            self.magazine.assign_grid(slot_name, grid)
            return
        slot = self.holder.slots.get(slot_name)
        if slot is None:
            raise ValueError(f"Slot '{slot_name}' not found in sample holder.")
        slot.loaded_grid = grid
        if persist:
            save_holder_occupancy(
                self.holder, session_state_for(self.parent, writable=True)
            )

    def restore_occupancy(self) -> bool:
        """Put back the grids the session state records in the holder's slots.

        Only a fixed holder: an autoloader's magazine is read from the hardware.
        Returns whether anything was recorded.
        """
        if self.magazine is not None:
            return False
        return load_holder_occupancy(self.holder, session_state_for(self.parent))


def bind_sample_handling(microscope: Any) -> Optional[SampleHandling]:
    """Build the sample handling service over the stage, and the sample loader if
    there is one; None without a stage device. Every driver builds it once its
    devices exist (`FibsemMicroscope._build_sample_handling`)."""
    if microscope.stage is None:
        return None
    handling = SampleHandling(parent=microscope)
    handling.fill_roles(stage=microscope.stage)
    loader = microscope.devices.get("sample_loader")
    if loader is not None:
        handling.fill_roles(loader=loader)
    return handling.connect()


# ---- which grid is in which slot: session state -------------------------------------

HOLDER_OCCUPANCY = "holder_occupancy"


def _import_occupancy() -> Optional[dict]:
    """The occupancy from the file it lived in before the session state, if any."""
    path = Path(cfg.SAMPLE_HOLDER_OCCUPANCY_PATH)
    if not path.exists():
        return None
    with open(path) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else None


def load_holder_occupancy(holder: SampleHolder, state: SessionState) -> bool:
    """Apply the recorded occupancy to *holder*. Returns whether there was one."""
    data = state.load_section(HOLDER_OCCUPANCY, migrate=_import_occupancy)
    if data is None:
        return False
    if (
        BUILTIN_SLOT_NAME in holder.slots
        and isinstance(data, dict)
        and BUILTIN_SLOT_NAME not in data
        and _LEGACY_BUILTIN_SLOT_NAME in data
    ):
        # Recorded before the shuttle's slot had its own name.
        data = {**data, BUILTIN_SLOT_NAME: data[_LEGACY_BUILTIN_SLOT_NAME]}
    try:
        holder.apply_occupancy(data)
    except Exception as e:
        logging.warning(f"Could not restore the sample holder occupancy: {e}")
        return False
    return True


def save_holder_occupancy(holder: SampleHolder, state: SessionState) -> bool:
    """Record which grid is in which slot. Returns whether anything was written."""
    return state.save_section(HOLDER_OCCUPANCY, holder.occupancy_to_dict())


def uncalibrated_message(slot_name: str) -> str:
    return (
        f"{slot_name} has no calibrated position. Run 'Calibrate slot positions' "
        "in the Sample Holder panel on the Microscope tab."
    )


# ---- which holder is on the stage: configuration ------------------------------------


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


BUILTIN_HOLDER_NAME = "Built-in holder"
# The built-in shuttle's one slot. Its own name, not "Slot-01", which is also
# the loader's first magazine slot: "loaded from Slot-01 into Slot-01" named two
# places as one (FIB-1359).
BUILTIN_SLOT_NAME = "Shuttle"
# What it was called before it had its own name. A calibration saved under it is
# read, and recorded occupancy under it restored.
_LEGACY_BUILTIN_SLOT_NAME = "Slot-01"
# What configurations written before the built-in holder had a neutral name call it.
# Read, then renamed to `BUILTIN_HOLDER_NAME`, so a calibration saved under it is kept.
_LEGACY_BUILTIN_HOLDER_NAMES = ("CompuStage Holder",)


def _builtin_working_slot(stage_settings) -> GridSlot:
    """The built-in shuttle's one slot: where the autoloader puts every grid.

    Nominally the stage origin, and that is the slot until someone calibrates it.
    A loaded grid on a real instrument sits off the origin, in x, y and z, by the same
    amount on every load (FIB-1144), so a position captured with the calibration
    wizard and saved in the configuration replaces it -- once it passes the trust
    check a fixed holder's slots get: captured against this pre-tilt and reference
    rotation. One that does not is dropped with a warning, and the origin is used.
    """
    pre_tilt = float(stage_settings.shuttle_pre_tilt)
    rotation_reference = float(stage_settings.rotation_reference)
    for legacy in _LEGACY_BUILTIN_HOLDER_NAMES:
        if legacy in stage_settings.holders:
            old = stage_settings.holders.pop(legacy)
            if BUILTIN_HOLDER_NAME not in stage_settings.holders:
                old.name = BUILTIN_HOLDER_NAME
                stage_settings.holders[BUILTIN_HOLDER_NAME] = old
    saved = stage_settings.holders.get(BUILTIN_HOLDER_NAME)
    if saved is not None:
        for note in saved.discard_untrusted_positions(pre_tilt, rotation_reference):
            logging.warning(
                f"Built-in shuttle working slot: {note}. Using the stage origin; "
                "recalibrate it from the Sample view."
            )
        captured = saved.slots.get(BUILTIN_SLOT_NAME) or saved.slots.get(
            _LEGACY_BUILTIN_SLOT_NAME
        )
        if captured is not None and captured.is_calibrated:
            return GridSlot(
                name=BUILTIN_SLOT_NAME,
                index=0,
                position=deepcopy(captured.position),
                calibration=deepcopy(captured.calibration),
            )
    return GridSlot(
        name=BUILTIN_SLOT_NAME,
        index=0,
        position=FibsemStagePosition(
            name=BUILTIN_SLOT_NAME, x=0.0, y=0.0, z=0.0, r=0.0, t=np.radians(0)
        ),
        calibration=SlotCalibration.builtin(pre_tilt, rotation_reference),
    )
