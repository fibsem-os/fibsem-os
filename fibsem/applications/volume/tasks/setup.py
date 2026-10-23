"""Set a position up for slicing, once: its point of interest, the volume
milling's trench, lanes and start slice there, the fiducial, and the poses and
references every slice aligns to."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import ClassVar

from fibsem.applications.volume.position import SEM_POSE
from fibsem.applications.volume.tasks.base import VolumeTask
from fibsem.applications.volume.tasks.milling import (
    lanes_at,
    start_slice_at,
    trench_at,
    volume_milling,
)
from fibsem.milling.base import FibsemMillingStage
from fibsem.milling.patterning.patterns import FiducialPattern
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.structures import (
    BeamType,
    FibsemMillingSettings,
    FibsemRectangle,
    ImageSettings,
    Point,
    distance_param,
    param,
)
from fibsem.workflows.proposals import (
    ALIGNMENT_AREA,
    MILLING,
    POINT_OF_INTEREST,
    STATE,
    DecisionOutcome,
)
from fibsem.workflows.task_config import TaskConfig

REFERENCE_FIELD_OF_VIEW = 50e-6
# The SEM reference is the whole frame: the alignment area is checked on the FIB view.
FULL_FRAME = FibsemRectangle(0, 0, 1, 1)


def _milling(name: str, *stages: FibsemMillingStage) -> FibsemMillingTaskConfig:
    return FibsemMillingTaskConfig(
        name=name, field_of_view=REFERENCE_FIELD_OF_VIEW, stages=list(stages)
    )


def fiducial_milling() -> FibsemMillingTaskConfig:
    """A fiducial beside the face, for the FIB alignment to lock on to."""
    return _milling(
        "Fiducial",
        FibsemMillingStage(
            name="Fiducial",
            milling=FibsemMillingSettings(milling_current=1e-9),
            pattern=FiducialPattern(
                width=0.5e-6, height=10e-6, depth=1e-6, point=Point(x=22e-6, y=0)
            ),
        ),
    )


@dataclass
class VolumeSetupMillingTaskConfig(TaskConfig):
    task_type: ClassVar[str] = "VOLUME_SETUP_FIB_MILLING"
    display_name: ClassVar[str] = "Setup FIB Milling Position"
    task_name: str = "Setup FIB Milling Position"
    mill_fiducial: bool = param(True, doc="Mill the fiducial before the start slice.")
    fiducial: FibsemMillingTaskConfig = param(
        doc="The fiducial the FIB alignment locks on to.",
        default_factory=fiducial_milling,
    )
    field_of_view: float = param(
        REFERENCE_FIELD_OF_VIEW,
        distance_param(
            "Field of view", "The field of view of the references.", decimals=1
        ),
    )


class VolumeSetupMillingTask(VolumeTask):
    """At the milling pose: confirm the position, place the point of interest,
    mill the protocol's volume milling setup there (trench, lanes), the fiducial
    and the start slice, check the alignment area and take the FIB reference the
    slices align to. The pose as it ends is the position's milling pose."""

    config_cls = VolumeSetupMillingTaskConfig
    config: VolumeSetupMillingTaskConfig
    asks = (STATE, MILLING, POINT_OF_INTEREST, ALIGNMENT_AREA)
    per_slice = False

    def _run(self) -> None:
        name, fov = self.position.name, self.config.field_of_view
        self._move_to_pose()
        self._acquire_reference(BeamType.ION, "ref_setup_start", fov)
        self._confirm_position(f"Confirm the milling position of {name}.")

        volume = volume_milling(self.experiment.task_protocol)
        if volume is None:
            raise RuntimeError(
                f"{self.task_name} mills the protocol's volume milling, and the "
                "protocol has no Volume FIB Milling task."
            )

        self._place_point_of_interest()
        position = self.position
        if volume.mill_trench:
            self._mill_piece(
                trench_at(volume, position), f"Mill the clearing trench for {name}."
            )
        lanes = lanes_at(volume, position)
        if lanes is not None:
            self._mill_piece(lanes, f"Mill the redeposition lanes for {name}.")
        if self.config.mill_fiducial:
            self._mill_piece(
                deepcopy(self.config.fiducial), f"Mill the fiducial for {name}."
            )
        self._mill_piece(
            start_slice_at(volume, position), f"Mill the start slice of {name}."
        )

        self._acquire_reference(BeamType.ION, "ref_setup_face", fov)
        self._validate_alignment_area()
        self._acquire_alignment_reference_image(
            ImageSettings(), fov, self.position.alignment_area
        )
        self.position.milling_pose = self.microscope.get_microscope_state()

    def _mill_piece(self, config: FibsemMillingTaskConfig, msg: str) -> None:
        config.acquisition.imaging.path = self.output_dir
        self.update_milling_config_ui(config, msg=msg)

    def _place_point_of_interest(self) -> None:
        decision = self.ask(
            POINT_OF_INTEREST,
            {"poi": deepcopy(self.position.poi)},
            image=self._last_fib_image_file(),
            message=f"Place the point of interest of {self.position.name}: "
            "the start slice is centred there.",
        )
        if decision.outcome is DecisionOutcome.Rejected:
            raise RuntimeError(
                f"{self.task_name} was rejected: {decision.reason or 'no reason given'}"
            )
        poi = decision.values.get("poi")
        if isinstance(poi, Point):
            self.position.poi = poi


@dataclass
class VolumeSetupImagingTaskConfig(TaskConfig):
    task_type: ClassVar[str] = "VOLUME_SETUP_SEM_ACQUISITION"
    display_name: ClassVar[str] = "Setup SEM Acquisition"
    task_name: str = "Setup SEM Acquisition"
    field_of_view: float = param(
        REFERENCE_FIELD_OF_VIEW,
        distance_param(
            "Field of view", "The field of view of the SEM reference.", decimals=1
        ),
    )


class VolumeSetupImagingTask(VolumeTask):
    """At the imaging pose: confirm the position and take the SEM reference the
    slices' images align to. The pose as it ends is the position's imaging
    pose."""

    config_cls = VolumeSetupImagingTaskConfig
    config: VolumeSetupImagingTaskConfig
    asks = (STATE,)
    per_slice = False

    def _run(self) -> None:
        fov = self.config.field_of_view
        self._move_to_pose(SEM_POSE)
        self._acquire_reference(BeamType.ELECTRON, "ref_setup_start", fov)
        self._confirm_position(f"Confirm the imaging position of {self.position.name}.")
        self._acquire_alignment_reference_image(
            ImageSettings(), fov, FULL_FRAME, beam_type=BeamType.ELECTRON
        )
        self.position.poses[SEM_POSE] = self.microscope.get_microscope_state()
