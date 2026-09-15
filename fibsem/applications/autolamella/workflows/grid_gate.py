"""Whether a lamella run can start: every linked lamella's grid must be on the stage.

A lamella marked on a grid's stored overview belongs to that grid whether or not
the grid is on the stage (FIB-71). Running it while the grid is in the magazine
would drive the stage to that pose on whatever grid *is* there, so a run that
includes one is refused, by name, before anything moves. There is no override:
load the grid from the Grids tab, or leave that lamella out.

The inventory is *read* first (`Stage.get_inventory`: instant, nothing moves) so
the answer is the hardware's, not a stale chip's. A lamella that is not linked to
a grid, or whose grid has no record in this experiment, is never refused: the
rule is about a known grid being somewhere else, not about missing bookkeeping.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Iterable, List

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from fibsem.applications.autolamella.structures import Experiment, Lamella
    from fibsem.microscopes._stage import Stage

logger = logging.getLogger(__name__)


def lamellae_off_the_stage(
    experiment: "Experiment", stage: "Stage", lamellae: Iterable["Lamella"]
) -> Dict[str, List[str]]:
    """Grid name -> the names of the given lamellae on it that cannot be run,
    because that grid is not on the stage. Empty when the run is clear."""
    try:
        stage.get_inventory()
    except Exception as e:  # noqa: BLE001 - answer from the last read instead
        logger.warning(f"Could not read the grid inventory before the run: {e}")
    try:
        loaded = {g.name for g in stage.loaded_grids}
    except Exception as e:  # noqa: BLE001 - nothing known: refuse nothing
        logger.warning(f"Could not read which grids are loaded: {e}")
        return {}

    off: Dict[str, List[str]] = {}
    for lamella in lamellae:
        grid = experiment.get_grid_for_lamella(lamella)
        if grid is None or grid.name in loaded:
            continue
        off.setdefault(grid.name, []).append(lamella.name)
    return off


def run_refusal(
    experiment: "Experiment", stage: "Stage", lamellae: Iterable["Lamella"]
) -> str:
    """Why this run cannot start, naming the lamellae and their grid, or ""."""
    off = lamellae_off_the_stage(experiment, stage, lamellae)
    if not off:
        return ""
    lines = []
    for grid_name, names in off.items():
        lines.append(
            f"{', '.join(names)} {'is' if len(names) == 1 else 'are'} on "
            f"{grid_name}, which is not on the stage."
        )
    grids = list(off)
    which = grids[0] if len(grids) == 1 else "the grid"
    return (
        "Cannot run: "
        + " ".join(lines)
        + f" Load {which} from the Grids tab first, or leave "
        + (
            "that lamella"
            if sum(len(n) for n in off.values()) == 1
            else "those lamellae"
        )
        + " out."
    )
