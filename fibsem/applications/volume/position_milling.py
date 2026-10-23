"""Where the protocol's milling lands on one position: what its tasks mill
there, placed as they place it, for the window to draw."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

from fibsem.applications.volume.tasks.milling import (
    VolumeFIBMillingTaskConfig,
    lanes_at,
    slice_at,
    start_slice_at,
    trench_at,
    volume_milling,
)
from fibsem.applications.volume.tasks.setup import VolumeSetupMillingTaskConfig
from fibsem.milling.patterning.shapes import COLOURS
from fibsem.milling.tasks import FibsemMillingTaskConfig

if TYPE_CHECKING:
    from fibsem.applications.volume.experiment import VolumeExperiment
    from fibsem.applications.volume.position import VolumePosition

__all__ = [
    "PlacedMilling",
    "WHOLE_VOLUME",
    "whole_volume",
    "position_millings",
    "setup_millings",
    "slice_depth_text",
    "slice_progress",
]


# The slices draw in the first stage colour; the setup's pieces in the next.
PROGRESS_COLOUR = COLOURS[0]
WHOLE_VOLUME = "Volume milling"


@dataclass
class PlacedMilling:
    label: str
    milling: FibsemMillingTaskConfig
    # each stage's style and colour, when it is not the plain one (see MillingSpec)
    styles: Tuple[str, ...] = ()
    colours: Tuple[str, ...] = ()


def _band(
    config: VolumeFIBMillingTaskConfig,
    position: "VolumePosition",
    first: int,
    last: int,
    name: str,
):
    """One stage covering slices *first* to *last*: the slice's rectangle,
    as tall as all of them."""
    a = slice_at(config, position, first).stages[0]
    b = slice_at(config, position, last).stages[0]
    band = deepcopy(a)
    band.name = name
    band.pattern.height = a.pattern.height + abs(a.pattern.point.y - b.pattern.point.y)
    band.pattern.point.y = (a.pattern.point.y + b.pattern.point.y) / 2
    return band


def slice_progress(
    config: VolumeFIBMillingTaskConfig, position: "VolumePosition", cut: int
) -> PlacedMilling:
    """A position's slices as a run goes: those cut (filled), the next
    (outlined) and those still to cut (dashed), from its first slice against
    the start slice to its last."""
    last = max(position.iterations - 1, 0)
    cut = min(cut, position.iterations)
    milling = deepcopy(config.milling)
    milling.stages, styles = [], []
    if cut:
        milling.stages.append(
            _band(config, position, 0, cut - 1, f"Cut · slices 1 to {cut}")
        )
        styles.append("filled")
    if cut <= last:
        stage = slice_at(config, position, cut).stages[0]
        stage.name = f"Next · slice {cut + 1}"
        milling.stages.append(stage)
        styles.append("outline")
    if cut + 1 <= last:
        milling.stages.append(
            _band(
                config,
                position,
                cut + 1,
                last,
                f"To cut · slices {cut + 2} to {last + 1}",
            )
        )
        styles.append("dashed")
    return PlacedMilling(
        config.task_name,
        milling,
        tuple(styles),
        (PROGRESS_COLOUR,) * len(styles),
    )


def setup_millings(
    config: VolumeFIBMillingTaskConfig, position: "VolumePosition"
) -> List[PlacedMilling]:
    """What the setup mills of the volume milling on *position*: the trench
    when it is milled, the lanes asked for, and the start slice."""
    millings = []
    if config.mill_trench:
        millings.append(
            PlacedMilling("Setup · Clearing trench", trench_at(config, position))
        )
    lanes = lanes_at(config, position)
    if lanes is not None:
        millings.append(PlacedMilling("Setup · Redeposition lanes", lanes))
    millings.append(
        PlacedMilling("Setup · Start slice", start_slice_at(config, position))
    )
    return millings


def whole_volume(progress: PlacedMilling, pieces: List[PlacedMilling]) -> PlacedMilling:
    """The slices' progress and every setup piece, in one drawing."""
    milling = deepcopy(progress.milling)
    milling.name = WHOLE_VOLUME
    colours = list(progress.colours)
    for k, piece in enumerate(pieces):
        milling.stages += piece.milling.stages
        colours += [COLOURS[1 + k % (len(COLOURS) - 1)]] * len(piece.milling.stages)
    return PlacedMilling(WHOLE_VOLUME, milling, progress.styles, tuple(colours))


def position_millings(
    experiment: "VolumeExperiment", position: "VolumePosition"
) -> List[PlacedMilling]:
    """The whole volume milling at once, then each piece: the slices' progress
    (cut, next, still to cut), then what the setup mills (the volume milling's
    trench, lanes and start slice, and each setup's fiducial)."""
    protocol = experiment.task_protocol
    config = volume_milling(protocol)
    if config is None:
        return []
    progress = slice_progress(config, position, experiment.slices_cut(position))
    pieces = setup_millings(config, position)
    for name in protocol.workflow_config.workflow:
        setup = protocol.task_config.get(name)
        if isinstance(setup, VolumeSetupMillingTaskConfig) and setup.mill_fiducial:
            pieces.append(
                PlacedMilling(f"Setup · {setup.fiducial.name}", setup.fiducial)
            )
    return [whole_volume(progress, pieces), progress, *pieces]


def slice_depth_text(
    experiment: "VolumeExperiment", position: "VolumePosition"
) -> Optional[str]:
    """ "2.4 of 8.0 µm (40 × 0.200 µm)": how far into the face the slices have
    cut, of how far they will; None without a slice milling task."""
    config = volume_milling(experiment.task_protocol)
    if config is None:
        return None
    um = config.slice_thickness * 1e6
    cut = experiment.slices_cut(position) * um
    total = position.iterations * um
    return f"{cut:.1f} of {total:.1f} µm ({position.iterations} × {um:.3f} µm)"
