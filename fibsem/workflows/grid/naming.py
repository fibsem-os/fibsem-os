"""A grid's name: when it may change, and changing it on the hardware and the
record together (FIB-1137).

A grid is named on its slot, where the hardware keeps it, and on its record in
the experiment, which links to the slot by that name. Once a task has run on
it, the name is fixed: its folder and images carry it.
"""

from __future__ import annotations

from fibsem.structures import SampleGrid
from fibsem.workflows.grid.manager import EXCHANGE_ENTRY_NAMES
from fibsem.workflows.grids import GridRecord


def grid_has_run(grid: GridRecord) -> bool:
    """Whether a task has run on this grid, as against it only being loaded.

    Its name is fixed from then on: the grid's folder is named after it, and every
    image a task wrote carries it. Renaming after that could only lose the files,
    move them, or leave them saying something else, so nothing renames it."""
    return any(t.name not in EXCHANGE_ENTRY_NAMES for t in grid.task_history)


NAME_FIXED_REASON = (
    "Named once it has run: its folder and images carry this name. "
    "Use Edit note… for a correction."
)


def grid_rename_refusal(experiment, old: str, new: str) -> str:
    """Why grid *old* may not be renamed *new*, or "": the Sample view's rules
    (FIB-1137). A grid that has run keeps its name, and no two grids share one."""
    record = experiment.get_grid_by_name(old)
    if record is not None and grid_has_run(record):
        return f"{old} cannot be renamed. {NAME_FIXED_REASON}"
    other = experiment.get_grid_by_name(new)
    if other is not None and other is not record:
        return f"There is already a grid named {new}."
    return ""


def name_slot_grid(stage, slot_name: str, name: str) -> None:
    """Name the grid in an inventory slot, keeping its description and radius:
    the slot description on a loader, the session state on a fixed holder
    (``assign_grid(..., persist=True)``). A write the hardware refuses raises,
    and the slot keeps its old name, as the next inventory read would."""
    slots = stage.magazine.slots if stage.magazine is not None else stage.holder.slots
    slot = slots[slot_name]
    current = slot.loaded_grid
    grid = SampleGrid(name=name, description=current.description, radius=current.radius)
    try:
        stage.assign_grid(slot_name, grid, persist=True)
    except Exception:
        slot.loaded_grid = current
        raise


def rename_grid(experiment, stage, old: str, new: str) -> None:
    """Rename grid *old* to *new*: the slot that holds it first, then its record.

    The record links to the hardware by name, and a slot that kept the old name
    would hand it back at the next inventory read, so a slot the hardware will
    not rename raises and the record keeps its name. A grid with no record yet
    is named on its slot only; the next inventory sync records it under the new
    name. Check :func:`grid_rename_refusal` first; the caller saves.
    """
    inventory = stage.grid_inventory() if stage is not None else []
    entry = next((e for e in inventory if e.name == old and e.present), None)
    if entry is not None:
        name_slot_grid(stage, entry.slot_name, new)
    record = experiment.get_grid_by_name(old)
    if record is not None:
        record.name = new
