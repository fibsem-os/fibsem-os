import logging
import time
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

import numpy as np

from fibsem.fm.acquisition import acquire_z_stack
from fibsem.fm.structures import ChannelSettings, FluorescenceImage, ZParameters
from fibsem.microscope import FibsemMicroscope
from fibsem.milling.base import (
    FibsemMillingStage,
    MillingStrategy,
    MillingStrategyConfig,
)
from fibsem.milling.core import draw_patterns, finish_milling, setup_milling
from fibsem.structures import BeamType, FibsemRectangle, MillingState
from psygnal import Signal

if TYPE_CHECKING:
    from fibsem.ui.FMCoincidenceMillingWidget import FMCoincidenceMillingWidget

channel_settings = ChannelSettings(
    name="Channel-01",
    excitation_wavelength=550,
    emission_wavelength=None,
    power=0.003,
    exposure_time=0.1,
)


@dataclass
class CoincidenceMillingStrategyConfig(MillingStrategyConfig):
    """Configuration for the Coincidence Milling Strategy."""

    channel_settings: ChannelSettings = channel_settings
    zparams: ZParameters = ZParameters(-5e-6, 5e-6, 0.5e-6)
    timeout: int = 60  # seconds, default timeout for milling
    acquire_z_stack: bool = False  # acquire a z-stack after milling
    acquire_fib_image: bool = False  # acquire a FIB image after milling
    intensity_drop_threshold: float = 0.75  # threshold for intensity drop
    bbox: Optional[FibsemRectangle] = None  # reduced area for monitoring
    # acquire fib image before/after
    # save fm images
    # acquire post z-stack
    # intensity drop threshold
    # oscillation parameters


class CoincidenceMillingStrategy(MillingStrategy[CoincidenceMillingStrategyConfig]):
    """Coincidence Milling Strategy for milling and fluorescence acquisition."""

    name: str = "CoincidenceMilling"
    fullname: str = "fibsem.milling.CoincidenceMillingStrategy"
    config_class = CoincidenceMillingStrategyConfig
    intensity_drop_signal = Signal(dict)

    def __init__(self, config: Optional[CoincidenceMillingStrategyConfig] = None):
        if config is None:
            config = self.config_class()
        self.config = config
        self.microscope: Optional[FibsemMicroscope] = None
        self.stage: Optional[FibsemMillingStage] = None
        self.intensities: List[float] = []

    def run(
        self,
        microscope: FibsemMicroscope,
        stage: FibsemMillingStage,
        asynch: bool = False,
        parent_ui: Optional['FMCoincidenceMillingWidget'] = None,
    ) -> None:
        """Coincidence Milling Strategy"""
        logging.info(f"Running {self.name} Milling Strategy for {stage.name}")

        self.microscope = microscope
        self.stage = stage
        self.parent_ui = parent_ui

        if self.microscope is None:
            raise ValueError(
                "Microscope is not set. Please set the microscope before running the strategy."
            )
        if microscope.fm is None:
            raise ValueError(
                "Coincidence Milling Strategy requires a Fluorescence Module (FM) to be available on the microscope."
            )

        # TODO: enable this by threading the milling loop
        # if parent_ui and hasattr(parent_ui, "bbox_updated_signal"):
        #     logging.info("Connecting to parent UI's bbox_updated_signal")
        #     parent_ui.bbox_updated_signal.connect(self._on_bbox_update)

        if self.parent_ui and hasattr(self.parent_ui, "on_intensity_drop_signal"):
            self.intensity_drop_signal.connect(self.parent_ui.on_intensity_drop_signal)

        # setup milling
        microscope.stop_milling()
        setup_milling(microscope, stage)
        draw_patterns(microscope, patterns=stage.pattern.define())

        # connect the acquisition signal
        microscope.fm.acquisition_signal.disconnect(self.on_fm_acquisition_signal)
        microscope.fm.acquisition_signal.connect(self.on_fm_acquisition_signal)

        # start fm acquisition, start milling
        microscope.fm.start_acquisition(channel_settings=self.config.channel_settings)
        microscope.start_milling()

        estimated_time = microscope.estimate_milling_time()
        remaining_time = estimated_time
        start_time = time.time() + self.config.timeout
        estimated_end_time = start_time + estimated_time
        max_end_time = start_time + self.config.timeout
        SLEEP_DURATION = 1  # seconds
        while True:
            if self.parent_ui and hasattr(self.parent_ui, "get_bounding_box"):
                self.config.bbox = self.parent_ui.get_bounding_box()
            milling_state = microscope.get_milling_state()
            if milling_state == MillingState.RUNNING:
                logging.info(f"Milling is running... {remaining_time:.2f} seconds remaining.")
                remaining_time -= SLEEP_DURATION
            elif milling_state == MillingState.PAUSED:
                # QUERY: should we also stop fm acquisition?
                logging.info("Milling is paused. Waiting for resume...")
            elif milling_state == MillingState.IDLE:
                logging.info("Milling is idle. Finishing...")
                break
            time.sleep(SLEEP_DURATION)

            # update milling progress via signal
            microscope.milling_progress_signal.emit({"progress": {
                    "state": "update", 
                    "start_time": start_time,
                    "milling_state": milling_state,
                    "estimated_time": estimated_time, 
                    "remaining_time": remaining_time}
                    })

            # timeout
            if time.time() >= max_end_time:
                logging.info(
                    f"{self.config.timeout} seconds have passed. Stopping milling."
                )
                break

        # stop fm
        microscope.stop_milling()
        microscope.fm.stop_acquisition()
        logging.info("Milling finished. FM acquisition stopped.")

        # finalize
        microscope.fm.acquisition_signal.disconnect(self.on_fm_acquisition_signal)
        finish_milling(microscope)

        # acquire a z-stack post-milling
        if self.config.acquire_z_stack:
            image = acquire_z_stack(
                microscope.fm,
                channel_settings=self.config.channel_settings,
                zparams=self.config.zparams,
            )

        # acquire a fibsem image
        if self.config.acquire_fib_image:
            image = microscope.acquire_image(beam_type=BeamType.ION)
            if parent_ui and hasattr(parent_ui, "fib_image_acquired_signal"):
                parent_ui.fib_image_acquired_signal.emit(image)

    def on_fm_acquisition_signal(self, image: FluorescenceImage):
        if self.microscope is None or self.microscope.fm is None:
            logging.error(
                "Microscope or FM is not set. Cannot process FM acquisition signal."
            )
            return
        logging.info(
            f"FM acq_date: {image.metadata.acquisition_date}, {image.data.shape}, {self.microscope.get_milling_state()}"
        )

        # crop to bounding box if specified
        data = image.data
        if self.config.bbox is not None:
            x, y, w, h = self.config.bbox.to_pixel_coordinates(
                self.microscope.fm.camera.resolution
            )
            data = image.data[y : y + h, x : x + w]

        # calc mean intensity
        mean_intensity = data.mean()
        if np.isnan(mean_intensity):
            return
        n_rolling = 10
        self.intensities.append(mean_intensity)
        rolling_mean = sum(self.intensities[-n_rolling:]) / min(
            len(self.intensities), n_rolling
        )
        average_mean = (
            sum(self.intensities) / len(self.intensities) if self.intensities else 0
        )
        logging.info(
            f"Mean Intensity: {mean_intensity:.2f}, Rolling Mean ({n_rolling}): {rolling_mean:.2f}, Average Mean: {average_mean:.2f}"
        )

        if self.parent_ui and hasattr(self.parent_ui, "update_line_plot_signal"):
            self.parent_ui.update_line_plot_signal.emit(mean_intensity)

        # check if the rolling mean intensity has dropped by 25%
        if rolling_mean < self.config.intensity_drop_threshold * average_mean:
            logging.warning(
                f"Rolling Mean Intensity has dropped by 25%: {rolling_mean:.2f} < {self.config.intensity_drop_threshold} * {average_mean:.2f}"
            )
            ddict = {
                "mean_intensity": mean_intensity,
                "rolling_mean": rolling_mean,
                "average_mean": average_mean,
                "intensity_drop_threshold": self.config.intensity_drop_threshold,
            }
            self.intensity_drop_signal.emit(ddict)

        logging.info("-" * 80)

    def _on_bbox_update(self, bbox: FibsemRectangle):
        """Handle updates to the bounding box."""
        logging.info(f"Bounding Box Updated: {bbox}")
        self.config.bbox = bbox
