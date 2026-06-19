"""Imaging grid tasks: tiled overview acquisition + single hi-res image."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import ClassVar, Literal, Optional, Type

from fibsem.applications.autolamella.workflows.tasks.grid.base import (
    GridTask,
    GridTaskConfig,
)
from fibsem.imaging.tiled import tiled_image_acquisition_and_stitch
from fibsem.structures import BeamType, ImageSettings, OverviewAcquisitionSettings


def _default_overview_settings() -> OverviewAcquisitionSettings:
    """Default tiled-overview settings for the overview grid task."""
    return OverviewAcquisitionSettings(
        image_settings=ImageSettings(
            resolution=(1024, 1024),
            hfw=500e-6,
            dwell_time=1e-6,
            beam_type=BeamType.ION,
            path=None,
            filename="overview-image",
        ),
        nrows=1,
        ncols=2,
    )


@dataclass
class AcquireOverviewImageGridTaskConfig(GridTaskConfig):
    """Configuration for acquiring overview image grid task."""
    task_type: ClassVar[str] = "ACQUIRE_OVERVIEW_IMAGE_GRID"
    display_name: ClassVar[str] = "Acquire Overview Image"
    orientation: Optional[Literal["SEM", "FIB", "MILLING"]] = "SEM"
    settings: OverviewAcquisitionSettings = field(default_factory=_default_overview_settings)


class AcquireOverviewImageGridTask(GridTask):
    """Task to acquire an overview image of the sample grid."""
    config_cls: ClassVar[Type[GridTaskConfig]] = AcquireOverviewImageGridTaskConfig
    config: AcquireOverviewImageGridTaskConfig

    def _run(self):
        """Acquire an overview image of the sample grid."""
        slot = self.slot
        if slot is None:
            raise RuntimeError(f"Grid '{self.grid.name}' is not loaded in any slot.")

        test_path = self.task_dir()
        self.config.settings.image_settings.path = test_path

        logging.info(f"Path: {test_path}")
        self.update_status_ui(f"Moving to grid {self.grid.name}...")

        self._move_to_grid_slot_position(self.config.orientation)

        self.update_status_ui("Acquiring overview image...")
        image = tiled_image_acquisition_and_stitch(
            microscope=self.microscope,
            settings=self.config.settings
        )

        # save the stitched overview (full) + a small thumbnail (for grid cards)
        overview_path = os.path.join(test_path, "overview.tif")
        image.save(overview_path)
        thumb_path = self._save_thumbnail(image)
        self.record_result(overview=overview_path, thumbnail=thumb_path)

        logging.info(f"Acquired overview image for grid {self.grid.name}")

    def _save_thumbnail(self, image, width: int = 400) -> str:
        """Save a small resized PNG thumbnail of the overview for grid cards."""
        import numpy as np
        from PIL import Image as PILImage

        data = image.data
        if data.ndim == 3:
            data = data[..., :3]
        else:
            data = np.stack([data, data, data], axis=2)
        thumb = PILImage.fromarray(data.astype(np.uint8))
        thumb.thumbnail((width, width))  # preserves aspect ratio
        thumb_path = os.path.join(self.grid_dir(), "thumbnail.png")
        thumb.save(thumb_path)
        return thumb_path


def _default_acquire_image_settings() -> ImageSettings:
    """Default image settings for the single-image grid task."""
    return ImageSettings(
        resolution=(4096, 4096),
        hfw=2000e-6,
        dwell_time=1e-6,
        beam_type=BeamType.ELECTRON,
    )


@dataclass
class AcquireImageGridTaskConfig(GridTaskConfig):
    """Configuration for acquiring overview image grid task."""
    task_type: ClassVar[str] = "ACQUIRE_IMAGE_GRID"
    display_name: ClassVar[str] = "Acquire Image"
    orientation: Optional[Literal["SEM", "FIB", "MILLING"]] = "SEM"
    voltage: float = field(default=5_000, metadata={"label": "Imaging Voltage" })
    beam_current: float = field(default=1e-9, metadata={"label": "Beam Current"})
    image_settings: ImageSettings = field(default_factory=_default_acquire_image_settings)


class AcquireImageTask(GridTask):
    """Task to acquire an image of the sample grid at a specified voltage."""
    config_cls: ClassVar[Type[GridTaskConfig]] = AcquireImageGridTaskConfig
    config: AcquireImageGridTaskConfig

    def _run(self):
        """Acquire an overview image of the sample grid."""
        slot = self.slot
        if slot is None:
            raise RuntimeError(f"Grid '{self.grid.name}' is not loaded in any slot.")

        test_path = self.task_dir()
        # self.config.settings.image_settings.path = test_path

        logging.info(f"Path: {test_path}")
        self.update_status_ui(f"Moving to grid {self.grid.name}...")

        self._move_to_grid_slot_position(self.config.orientation)

        self.update_status_ui("Acquiring image...")
        image_settings = self.config.image_settings  # per-run copy (run_grid_task deepcopies)
        image_settings.save = False

        # apply voltage/current to the beam the image is acquired on (not always ELECTRON)
        beam_type = image_settings.beam_type
        inital_state = self.microscope.get_microscope_state()
        self.microscope.set_beam_voltage(self.config.voltage, beam_type=beam_type)
        self.microscope.set_beam_current(self.config.beam_current, beam_type=beam_type)

        from fibsem import utils
        image = self.microscope.acquire_image(image_settings=image_settings)
        image_path = os.path.join(test_path, f"grid-image-{utils.current_timestamp_v3()}.tif")
        image.save(image_path)
        self.record_result(image=image_path)

        self.microscope.set_microscope_state(inital_state)

        logging.info(f"Acquired image for grid {self.grid.name}")
