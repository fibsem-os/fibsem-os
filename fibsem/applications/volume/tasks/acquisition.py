"""Image the face a slice has just exposed."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import ClassVar, Dict

from fibsem import acquire
from fibsem.applications.volume.position import SEM_POSE, SEM_REFERENCE
from fibsem.applications.volume.tasks.base import VolumeTask
from fibsem.structures import BeamType, FibsemRectangle, ImageSettings, param
from fibsem.workflows.task_config import TaskConfig

# The output role a slice's image is recorded under.
SLICE_ROLE = "slice"


def slice_imaging() -> ImageSettings:
    return ImageSettings(
        beam_type=BeamType.ELECTRON,
        resolution=(3072, 2048),
        dwell_time=100e-9,
        hfw=50e-6,
        autocontrast=False,
        line_integration=16,
    )


@dataclass
class VolumeSEMAcquisitionTaskConfig(TaskConfig):
    task_type: ClassVar[str] = "VOLUME_SEM_ACQUISITION"
    display_name: ClassVar[str] = "Volume SEM Acquisition"
    task_name: str = "Volume SEM Acquisition"
    imaging: ImageSettings = param(
        doc="How each slice's face is imaged, with the electron beam.",
        default_factory=slice_imaging,
    )
    autofocus_every: int = param(
        0, doc="Autofocus before every n-th slice, from the first; 0 never."
    )
    autocontrast_every: int = param(
        0, doc="Run autocontrast before every n-th slice, from the first; 0 never."
    )
    auto_area: FibsemRectangle = param(
        doc="The part of the frame autofocus and autocontrast look at.",
        default_factory=lambda: FibsemRectangle(0.25, 0.25, 0.5, 0.5),
    )


class VolumeSEMAcquisitionTask(VolumeTask):
    """Move to the imaging pose, align to the position's SEM reference, focus
    when due, and save the slice's image."""

    config_cls = VolumeSEMAcquisitionTaskConfig
    config: VolumeSEMAcquisitionTaskConfig
    result_images: ClassVar[Dict[str, str]] = {"reference_image": SLICE_ROLE}

    def _run(self) -> None:
        self._move_to_pose(SEM_POSE)
        self._align_to_reference(SEM_REFERENCE)

        area = self.config.auto_area
        if self._due(self.config.autofocus_every):
            self.log_status_message("AUTOFOCUS", "Focusing.")
            self.microscope.auto_focus(BeamType.ELECTRON, reduced_area=area)
        if self._due(self.config.autocontrast_every):
            self.log_status_message("AUTOCONTRAST", "Adjusting contrast.")
            self.microscope.autocontrast(BeamType.ELECTRON, reduced_area=area)
        self._check_for_abort()

        settings = deepcopy(self.config.imaging)
        settings.beam_type = BeamType.ELECTRON
        settings.path = self.output_dir
        settings.filename = f"slice_{self.slice:04d}"
        settings.save = True
        self.log_status_message(
            "ACQUIRE_SLICE",
            f"Imaging slice {self.slice + 1} of {self.position.iterations}.",
        )
        image = acquire.acquire_image(self.microscope, settings)
        self.record_output(SLICE_ROLE, image)

    def _due(self, every: int) -> bool:
        return every > 0 and self.slice % every == 0
