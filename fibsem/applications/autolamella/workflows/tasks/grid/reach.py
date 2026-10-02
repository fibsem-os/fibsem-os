"""Whether a grid overview fits within the stage's travel, asked before it runs.

A grid overview larger than the stage can reach acquires the tiles in reach and
skips the rest (FIB-1131). That is found out on the first grid; this asks the same
question while the overview is still being sized, on the Protocol tab's Grid page
(FIB-1152), through the same helpers the task uses, so the two cannot disagree.

Asked at each slot a grid overview would be centred on: the autoloader's working
slot, or each occupied, calibrated slot of a fixed holder (every calibrated one when
none is occupied), at the slot's centre in the task's orientation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

from fibsem.applications.autolamella.workflows.tasks.grid.base import GridTaskConfig
from fibsem.applications.autolamella.workflows.tasks.grid.fluorescence import (
    FluorescenceOverviewGridTaskConfig,
    reachable_fluorescence_overview,
)
from fibsem.applications.autolamella.workflows.tasks.grid.imaging import (
    BeamOverviewGridTaskConfig,
    reachable_overview,
)
from fibsem.projection import BeamStageProjection, FMStageProjection

if TYPE_CHECKING:
    from fibsem.microscope import FibsemMicroscope
    from fibsem.structures import GridSlot

# The key a fluorescence projection is kept under, beside the beams'.
FM_PROJECTION = "FM"


@dataclass
class SlotReach:
    """One slot's answer: the tiles an overview centred on it would skip."""

    slot: str
    skipped: List[Tuple[int, int]]
    enabled: int  # tiles the overview would visit, before any are skipped
    nrows: int
    ncols: int


def overview_slots(microscope: "FibsemMicroscope") -> List["GridSlot"]:
    """The slots a grid overview would be centred on. With a loader, its working
    slot. On a fixed holder, the occupied calibrated slots, or every calibrated
    one when none is occupied."""
    stage = getattr(microscope, "_stage", None)
    holder = getattr(stage, "holder", None)
    if holder is None:
        return []
    calibrated = [
        s
        for s in sorted(holder.slots.values(), key=lambda s: s.index)
        if s.position is not None
    ]
    if getattr(stage, "loader", None) is not None:
        return calibrated[:1]
    occupied = [s for s in calibrated if s.loaded_grid is not None]
    return occupied or calibrated


def overview_reach(
    microscope: "FibsemMicroscope",
    config: GridTaskConfig,
    projections: Optional[Dict[object, object]] = None,
) -> List[SlotReach]:
    """Each slot whose overview would skip tiles, and which. Empty when every tile is
    in reach everywhere, for a task that is not an overview, or when it cannot be
    worked out (no stage limits, no projection).

    *projections* keeps a projection per beam (and one for the FM) across calls, so a
    caller asking on every edit reads the instrument once, not per edit.
    """
    projections = {} if projections is None else projections
    reaches: List[SlotReach] = []
    for slot in overview_slots(microscope):
        if isinstance(config, BeamOverviewGridTaskConfig):
            beam = config.settings.image_settings.beam_type
            if beam not in projections:
                projections[beam] = BeamStageProjection.from_microscope(
                    microscope, beam
                )
            if projections[beam] is None:
                return []
            centre = microscope.get_target_position(slot.position, config.orientation)
            settings, skipped = reachable_overview(
                microscope, config.settings, centre, projections[beam]
            )
            enabled = config.settings.n_enabled_tiles
            nrows, ncols = settings.nrows, settings.ncols
        elif isinstance(config, FluorescenceOverviewGridTaskConfig):
            if getattr(microscope, "fm", None) is None:
                return []
            if FM_PROJECTION not in projections:
                projections[FM_PROJECTION] = FMStageProjection.from_microscope(
                    microscope
                )
            if projections[FM_PROJECTION] is None:
                return []
            centre = microscope.to_device(slot.position, "FM")
            overview, skipped = reachable_fluorescence_overview(
                microscope, config.overview, centre, projections[FM_PROJECTION]
            )
            enabled = config.overview.n_enabled_tiles
            nrows, ncols = overview.rows, overview.cols
        else:
            return []
        if skipped:
            reaches.append(SlotReach(slot.name, skipped, enabled, nrows, ncols))
    return reaches


def _listed(items: Sequence[object]) -> str:
    """'0', '0 and 3', '0, 1 and 3'."""
    words = [str(i) for i in items]
    return words[0] if len(words) == 1 else f"{', '.join(words[:-1])} and {words[-1]}"


def _which(reach: SlotReach) -> str:
    """The skipped tiles in the fewest words: whole rows and columns by number,
    a handful of tiles by position, or just how many."""
    skipped = set(reach.skipped)
    rows = [
        r
        for r in range(reach.nrows)
        if all((r, c) in skipped for c in range(reach.ncols))
    ]
    cols = [
        c
        for c in range(reach.ncols)
        if all((r, c) in skipped for r in range(reach.nrows))
    ]
    covered = {(r, c) for r in rows for c in range(reach.ncols)}
    covered |= {(r, c) for c in cols for r in range(reach.nrows)}
    if covered == skipped:
        parts = []
        if rows:
            parts.append(f"row{'s' if len(rows) > 1 else ''} {_listed(rows)}")
        if cols:
            parts.append(f"column{'s' if len(cols) > 1 else ''} {_listed(cols)}")
        return " and ".join(parts)
    if len(reach.skipped) <= 6:
        return ", ".join(f"({r},{c})" for r, c in reach.skipped)
    return ""


def describe_reach(reaches: Sequence[SlotReach], one_slot: bool) -> str:
    """The warning for the Grid page, or "" when nothing is out of reach.
    *one_slot*: the overview has only one place to be (an autoloader's working
    slot), so the slot is not named."""
    lines = []
    for reach in reaches:
        which = _which(reach)
        line = (
            f"{len(reach.skipped)} of {reach.enabled} tiles are past the stage's "
            f"reach and will be skipped" + (f" ({which})." if which else ".")
        )
        lines.append(line if one_slot else f"{reach.slot}: {line}")
    return "\n".join(lines)
