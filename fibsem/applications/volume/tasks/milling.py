"""A position's volume milling, in one piece: the clearing trench, the
redeposition lanes and the start slice, milled once by the setup, then one
slice per iteration, each a slice thickness past the one before.

Every piece is placed from the position's point of interest (the start slice's
centre) and the slicing direction: the trench on the side the slices move away
from, a lane down each side of the slices for their whole run.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Optional

from fibsem.applications.volume.position import FIB_REFERENCE, VolumePosition
from fibsem.applications.volume.tasks.base import VolumeTask
from fibsem.milling.base import FibsemMillingStage
from fibsem.milling.patterning.patterns import RectanglePattern
from fibsem.milling.tasks import FibsemMillingTaskConfig, run_milling_task
from fibsem.structures import (
    BeamType,
    CrossSectionPattern,
    Display,
    FibsemMillingSettings,
    Point,
    distance_param,
    param,
)
from fibsem.workflows.task_config import TaskConfig

if TYPE_CHECKING:
    from fibsem.workflows.task_config import TaskProtocol

# The FIB image the slice's milling kept after it, when its acquisition takes one.
SLICE_FIB_ROLE = "slice_fib"
FIELD_OF_VIEW = 50e-6


def _milling(
    name: str, current: float, cross_section=CrossSectionPattern.Rectangle, **size
) -> FibsemMillingTaskConfig:
    return FibsemMillingTaskConfig(
        name=name,
        field_of_view=FIELD_OF_VIEW,
        stages=[
            FibsemMillingStage(
                name=name,
                milling=FibsemMillingSettings(milling_current=current),
                pattern=RectanglePattern(cross_section=cross_section, **size),
            )
        ],
    )


def clearing_trench() -> FibsemMillingTaskConfig:
    """A trench in front of the start slice, so the face can be imaged."""
    return _milling("Clearing trench", 1e-9, width=30e-6, height=10e-6, depth=5e-6)


def redeposition_lanes() -> FibsemMillingTaskConfig:
    """A channel beside the slices, milled before them: somewhere for what
    they redeposit to go. Its height is the slices' run."""
    return _milling("Redeposition lane", 300e-12, width=3e-6, height=20e-6, depth=2e-6)


def start_slice() -> FibsemMillingTaskConfig:
    """A cleaning cross-section at the point of interest: the face the first
    slice is cut from."""
    return _milling(
        "Start slice",
        60e-12,
        CrossSectionPattern.CleaningCrossSection,
        width=20e-6,
        height=2e-6,
        depth=1e-6,
    )


def slice_milling() -> FibsemMillingTaskConfig:
    """One cleaning cross-section: a slice off the face. Its width and depth
    are the start slice's."""
    return _milling(
        "Slice",
        20e-12,
        CrossSectionPattern.CleaningCrossSection,
        width=20e-6,
        height=200e-9,
        depth=1e-6,
    )


@dataclass
class VolumeFIBMillingTaskConfig(TaskConfig):
    task_type: ClassVar[str] = "VOLUME_FIB_MILLING"
    display_name: ClassVar[str] = "Volume FIB Milling"
    task_name: str = "Volume FIB Milling"
    mill_trench: bool = param(
        True,
        doc="Mill the clearing trench in the setup.",
        display=Display("Mill the trench"),
    )
    trench: FibsemMillingTaskConfig = param(
        doc="The clearing trench, in front of the start slice.",
        default_factory=clearing_trench,
    )
    lane_left: bool = param(
        True,
        doc="Mill the redeposition lane left of the slices.",
        display=Display("Left lane"),
    )
    lane_right: bool = param(
        True,
        doc="Mill the redeposition lane right of the slices.",
        display=Display("Right lane"),
    )
    lanes: FibsemMillingTaskConfig = param(
        doc="Each redeposition lane. Its height is set to the slices' run.",
        default_factory=redeposition_lanes,
    )
    start_slice: FibsemMillingTaskConfig = param(
        doc="The start slice, centred on the point of interest.",
        default_factory=start_slice,
    )
    milling: FibsemMillingTaskConfig = param(
        doc="What one slice mills. Its width and depth are the start slice's.",
        default_factory=slice_milling,
    )
    slice_thickness: float = param(
        200e-9,
        distance_param(
            "Slice thickness", "How far each slice moves the face.", decimals=3
        ),
    )
    flipped: bool = param(
        False,
        doc="Slice up the image instead of down: the trench goes below the start slice.",
        display=Display("Flip direction"),
    )


def volume_milling(protocol: "TaskProtocol") -> Optional[VolumeFIBMillingTaskConfig]:
    """The protocol's volume milling: its first Volume FIB Milling task's."""
    return next(
        (
            c
            for c in protocol.task_config.values()
            if isinstance(c, VolumeFIBMillingTaskConfig)
        ),
        None,
    )


def _direction(config: VolumeFIBMillingTaskConfig) -> int:
    """+1 when the slices move up the image, -1 down."""
    return 1 if config.flipped else -1


def _placed(
    config: VolumeFIBMillingTaskConfig,
    milling: FibsemMillingTaskConfig,
    point: Point,
    **pattern,
) -> FibsemMillingTaskConfig:
    """A copy of *milling* with every pattern at *point*, scanned toward where
    the slices go, with *pattern*'s fields set."""
    placed = deepcopy(milling)
    scan = "BottomToTop" if config.flipped else "TopToBottom"
    for stage in placed.stages:
        stage.pattern.point = Point(x=point.x, y=point.y)
        if hasattr(stage.pattern, "scan_direction"):
            stage.pattern.scan_direction = scan
        for name, value in pattern.items():
            setattr(stage.pattern, name, value)
    return placed


def _start(config: VolumeFIBMillingTaskConfig):
    return config.start_slice.stages[0].pattern


def start_slice_at(
    config: VolumeFIBMillingTaskConfig, position: VolumePosition
) -> FibsemMillingTaskConfig:
    """The start slice, at *position*'s point of interest."""
    return _placed(config, config.start_slice, position.poi)


def trench_at(
    config: VolumeFIBMillingTaskConfig, position: VolumePosition
) -> FibsemMillingTaskConfig:
    """The clearing trench, against the start slice on the side the slices
    move away from."""
    poi, d = position.poi, _direction(config)
    height = config.trench.stages[0].pattern.height
    y = poi.y - d * (_start(config).height + height) / 2
    return _placed(config, config.trench, Point(x=poi.x, y=y))


def lanes_at(
    config: VolumeFIBMillingTaskConfig, position: VolumePosition
) -> Optional[FibsemMillingTaskConfig]:
    """A lane against each side asked for of the start slice and the slices,
    from the start slice's back to the last slice; None with neither side."""
    sides = [
        (side, name)
        for side, name, on in (
            (-1, "left", config.lane_left),
            (1, "right", config.lane_right),
        )
        if on
    ]
    if not sides:
        return None
    poi, d, start = position.poi, _direction(config), _start(config)
    run = position.iterations * config.slice_thickness
    y = poi.y + d * run / 2
    lanes = _placed(
        config, config.lanes, Point(x=poi.x, y=y), height=start.height + run
    )
    stages, lanes.stages = lanes.stages, []
    for side, name in sides:
        for stage in deepcopy(stages):
            stage.pattern.point.x = (
                poi.x + side * (start.width + stage.pattern.width) / 2
            )
            stage.name = f"{stage.name} · {name}"
            lanes.stages.append(stage)
    return lanes


def slice_at(
    config: VolumeFIBMillingTaskConfig, position: VolumePosition, slice: int
) -> FibsemMillingTaskConfig:
    """What slice *slice* (from 0) of *position* mills: its cut ends
    ``slice + 1`` thicknesses past the start slice's face."""
    poi, d, start = position.poi, _direction(config), _start(config)
    height = config.milling.stages[0].pattern.height
    reach = start.height / 2 + (slice + 1) * config.slice_thickness - height / 2
    return _placed(
        config,
        config.milling,
        Point(x=poi.x, y=poi.y + d * reach),
        width=start.width,
        depth=start.depth,
    )


class VolumeFIBMillingTask(VolumeTask):
    """Move to the milling pose, align to the position's FIB reference, and mill
    the slice."""

    config_cls = VolumeFIBMillingTaskConfig
    config: VolumeFIBMillingTaskConfig

    def _run(self) -> None:
        self._move_to_pose()
        self._align_reference_image(FIB_REFERENCE, required=True)

        config = slice_at(self.config, self.position, self.slice)
        config.acquisition.imaging.path = self.output_dir

        self.log_status_message(
            "MILL_SLICE",
            f"Milling slice {self.slice + 1} of {self.position.iterations}.",
        )
        task = run_milling_task(
            self.microscope,
            config,
            stop_event=self._stop_event,
            output_dir=self.output_dir,
        )
        self._check_for_abort()
        if task.error is not None:
            raise task.error
        fib_images = [
            i
            for i in task.images
            if i.metadata is not None and i.metadata.beam_type is BeamType.ION
        ]
        if fib_images:
            self.record_output(SLICE_FIB_ROLE, fib_images[-1])
