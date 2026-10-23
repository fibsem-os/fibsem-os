"""Where the protocol's milling lands on one position: what its tasks mill
there, placed as they place it, for the window to draw."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

from fibsem.applications.volume.tasks.base import placed
from fibsem.applications.volume.tasks.milling import (
    VolumeFIBMillingTaskConfig,
    slice_milling_at,
)
from fibsem.applications.volume.tasks.setup import VolumeSetupMillingTaskConfig
from fibsem.milling.tasks import FibsemMillingTaskConfig

if TYPE_CHECKING:
    from fibsem.applications.volume.experiment import VolumeExperiment
    from fibsem.applications.volume.position import VolumePosition

__all__ = ["PlacedMilling", "position_millings", "slice_depth_text", "slice_progress"]


@dataclass
class PlacedMilling:
    label: str
    milling: FibsemMillingTaskConfig
    # each stage's style, when it is not the plain one (see MillingSpec)
    styles: Tuple[str, ...] = ()


def _band(
    config: VolumeFIBMillingTaskConfig,
    position: "VolumePosition",
    first: int,
    last: int,
    name: str,
):
    """One stage covering slices *first* to *last*: the slice's rectangle,
    as tall as all of them."""
    top = slice_milling_at(config, position, first).stages[0]
    bottom = slice_milling_at(config, position, last).stages[0]
    band = deepcopy(top)
    band.name = name
    band.pattern.height = top.pattern.height + (
        top.pattern.point.y - bottom.pattern.point.y
    )
    band.pattern.point.y = (top.pattern.point.y + bottom.pattern.point.y) / 2
    return band


def slice_progress(
    config: VolumeFIBMillingTaskConfig, position: "VolumePosition", cut: int
) -> PlacedMilling:
    """A position's slices as a run goes: those cut (filled), the next
    (outlined) and those still to cut (dashed), from its first slice at the
    point of interest to its last."""
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
        stage = slice_milling_at(config, position, cut).stages[0]
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
    return PlacedMilling(config.task_name, milling, tuple(styles))


def position_millings(
    experiment: "VolumeExperiment", position: "VolumePosition"
) -> List[PlacedMilling]:
    """Each slice milling's progress (cut, next, still to cut), then each setup's
    trench, fiducial and face, in protocol order."""
    protocol = experiment.task_protocol
    slices: List[PlacedMilling] = []
    setup: List[PlacedMilling] = []
    for name in protocol.workflow_config.workflow:
        config = protocol.task_config.get(name)
        if isinstance(config, VolumeFIBMillingTaskConfig):
            progress = slice_progress(config, position, experiment.slices_cut(position))
            progress.label = name
            slices.append(progress)
        elif isinstance(config, VolumeSetupMillingTaskConfig):
            for milling, poi in (
                (config.trench, None),
                (config.fiducial, None),
                (config.face, position.poi),
            ):
                setup.append(
                    PlacedMilling(f"Setup · {milling.name}", placed(milling, poi))
                )
    return slices + setup


def slice_depth_text(
    experiment: "VolumeExperiment", position: "VolumePosition"
) -> Optional[str]:
    """ "2.4 of 8.0 µm (40 × 0.200 µm)": how far into the face the slices have
    cut, of how far they will; None without a slice milling task."""
    protocol = experiment.task_protocol
    config = next(
        (
            c
            for c in protocol.task_config.values()
            if isinstance(c, VolumeFIBMillingTaskConfig)
        ),
        None,
    )
    if config is None:
        return None
    um = config.slice_thickness * 1e6
    cut = experiment.slices_cut(position) * um
    total = position.iterations * um
    return f"{cut:.1f} of {total:.1f} µm ({position.iterations} × {um:.3f} µm)"
