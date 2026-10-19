"""Mill one slice off a position's face."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import ClassVar

from fibsem.applications.volume.position import FIB_REFERENCE, MILL_POSE
from fibsem.applications.volume.tasks.base import VolumeTask
from fibsem.milling.base import FibsemMillingStage
from fibsem.milling.patterning.patterns import RectanglePattern
from fibsem.milling.tasks import FibsemMillingTaskConfig, run_milling_task
from fibsem.structures import (
    CrossSectionPattern,
    FibsemMillingSettings,
    distance_param,
    param,
)
from fibsem.workflows.task_config import TaskConfig


def slice_milling() -> FibsemMillingTaskConfig:
    """One cleaning cross-section, milled top to bottom: a slice off the face."""
    return FibsemMillingTaskConfig(
        name="Slice",
        field_of_view=50e-6,
        stages=[
            FibsemMillingStage(
                name="Slice",
                milling=FibsemMillingSettings(milling_current=20e-12),
                pattern=RectanglePattern(
                    width=10e-6,
                    height=200e-9,
                    depth=1e-6,
                    scan_direction="TopToBottom",
                    cross_section=CrossSectionPattern.CleaningCrossSection,
                ),
            )
        ],
    )


@dataclass
class VolumeFIBMillingTaskConfig(TaskConfig):
    task_type: ClassVar[str] = "VOLUME_FIB_MILLING"
    display_name: ClassVar[str] = "Volume FIB Milling"
    task_name: str = "Volume FIB Milling"
    milling: FibsemMillingTaskConfig = param(
        doc="What one slice mills. Its patterns are placed at the slice's point of interest.",
        default_factory=slice_milling,
    )
    slice_thickness: float = param(
        200e-9,
        distance_param(
            "Slice thickness",
            "How far each slice moves the point of interest down the face.",
            decimals=3,
        ),
    )


class VolumeFIBMillingTask(VolumeTask):
    """Move to the milling pose, align to the position's FIB reference, and mill
    the slice at its point of interest."""

    config_cls = VolumeFIBMillingTaskConfig
    config: VolumeFIBMillingTaskConfig

    def _run(self) -> None:
        self._move_to_pose(MILL_POSE)
        self._align_to_reference(FIB_REFERENCE)

        config = deepcopy(self.config.milling)
        poi = self.position.slice_poi(self.slice, self.config.slice_thickness)
        for stage in config.stages:
            stage.pattern.point = deepcopy(poi)
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
