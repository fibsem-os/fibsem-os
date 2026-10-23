"""One action: inventory the grids, then run the protocol on every one of them.

The Arctis user's real ask is "inventory all my grids and acquire the overviews".
The run is a grid run with ``inventory_first``, started through the session's
runner as every run is, so the button, a script and the App Server run the same
thing. This is what it reads: the grids present, and the plan a confirmation
shows. On a fixed holder every present grid is already loaded, so the same run
has zero exchanges.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING, Dict, List, Mapping

from fibsem.structures import GridInventoryEntry
from fibsem.workflows.grid.naming import grid_rename_refusal, name_slot_grid

if TYPE_CHECKING:
    from fibsem.microscope import FibsemMicroscope
    from fibsem.workflows.grid.protocol import GridExperiment


def present_grids(
    microscope: "FibsemMicroscope", experiment: "GridExperiment"
) -> List[str]:
    """Refresh the inventory, record every present grid, and return their names
    in slot order. The selection "Screen all grids" runs over."""
    stage = microscope.sample_handling
    # A read, not a scan: the operator will have inventoried the magazine (in
    # xT or from the Sample view), and a physical scan in front of every
    # screening run is the wait this call exists to avoid.
    inventory = stage.get_inventory()
    added = experiment.sync_grids_from_inventory(stage)
    if added:
        logging.info(f"Inventory added {len(added)} grid(s): {[g.name for g in added]}")
    names = [e.name for e in inventory if e.present and e.name]
    if not names:
        logging.warning("Inventory found no grids to screen.")
    return names


class GridNamingError(ValueError):
    """Names "Screen all grids" cannot apply, and why: one line per problem."""


def screen_names(
    inventory: List[GridInventoryEntry], names: Mapping[str, str]
) -> Dict[str, str]:
    """The name each present grid will be screened under, by slot: the one typed
    for its slot, else the one the slot already reads. A name typed for an empty
    or unread slot is not used."""
    final: Dict[str, str] = {}
    for entry in inventory:
        if not entry.present or not entry.name:
            continue
        final[entry.slot_name] = (
            names.get(entry.slot_name) or ""
        ).strip() or entry.name
    return final


def naming_problems(
    experiment: "GridExperiment",
    inventory: List[GridInventoryEntry],
    names: Mapping[str, str],
) -> List[str]:
    """Why these names cannot be applied, or nothing. The Sample view's rules:
    a grid that has run keeps its name, and no two grids share one."""
    final = screen_names(inventory, names)
    problems: List[str] = []
    for name, count in Counter(final.values()).items():
        if count > 1:
            problems.append(f"{count} slots are named {name}.")
    for entry in inventory:
        new = final.get(entry.slot_name)
        if new is None or new == entry.name:
            continue
        why = grid_rename_refusal(experiment, entry.name, new)
        if why:
            problems.append(why)
    return problems


def name_and_record_grids(
    stage, experiment: "GridExperiment", names: Mapping[str, str]
) -> List[str]:
    """Screen all grids' one commit: name the slots, record the grids, and return
    the names to screen, in slot order.

    ``names`` maps a slot to the name the operator settled on; a slot left out,
    or named as it already reads, is not written. A name that differs is written
    to the slot (``assign_grid(..., persist=True)``: the slot description on a
    loader, the session state on a fixed holder), and a record the experiment
    already holds for that grid follows it, as a rename on the Sample view does.
    The grids are then recorded under their final names, so a new grid's first
    name is its last.

    Checks every name before writing any, and raises ``GridNamingError`` with
    every problem. A write the hardware refuses also raises, after the slots
    already written: each of those is named and recorded, and nothing runs.
    """
    inventory = stage.grid_inventory()
    problems = naming_problems(experiment, inventory, names)
    if problems:
        raise GridNamingError("\n".join(problems))
    final = screen_names(inventory, names)
    for entry in inventory:
        new = final.get(entry.slot_name)
        if new is None or new == entry.name:
            continue
        try:
            name_slot_grid(stage, entry.slot_name, new)
        except Exception as e:  # noqa: BLE001 - whatever the hardware raised, said
            raise GridNamingError(
                f"Could not name the grid in {entry.slot_name} {new}: {e}"
            ) from e
        record = experiment.get_grid_by_name(entry.name)
        if record is not None:
            record.name = new
        logging.info(f"Named the grid in {entry.slot_name} {new} (was {entry.name}).")
    added = experiment.sync_grids_from_inventory(stage)
    if added:
        logging.info(f"Recorded {len(added)} grid(s): {[g.name for g in added]}")
    experiment.save()
    return [final[e.slot_name] for e in inventory if e.slot_name in final]
