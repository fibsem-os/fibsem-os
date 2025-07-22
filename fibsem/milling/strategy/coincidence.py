from dataclasses import dataclass
import time
from pathlib import Path
from typing import List, Optional
from fibsem.milling.base import FibsemMillingStage, MillingStrategy, MillingStrategyConfig
from fibsem.milling.core import setup_milling, draw_patterns, finish_milling
from fibsem.microscope import FibsemMicroscope
from fibsem.fm.structures import FluorescenceImage, ChannelSettings, ZParameters
from fibsem.fm.acquisition import acquire_z_stack
from fibsem.structures import MillingState
import logging
from fibsem.structures import BeamType

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
    timeout: int = 30  # seconds, default timeout for milling
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

    def __init__(self, config: Optional[CoincidenceMillingStrategyConfig] = None):
        if config is None:
            config = self.config_class()
        self.config = config
        self.microscope: Optional[FibsemMicroscope] = None
        self.stage: Optional[FibsemMillingStage] = None

        self.intensities: List[float] = []

    def run(self, microscope: FibsemMicroscope, stage: FibsemMillingStage, asynch: bool = False, parent_ui = None) -> None:
        """Coincidence Milling Strategy"""
        logging.info(f"Running {self.name} Milling Strategy for {stage.name}")

        if microscope.fm is None:
            raise ValueError("Coincidence Milling Strategy requires a Fluorescence Module (FM) to be available on the microscope.")

        self.microscope = microscope
        self.stage = stage

        # setup milling
        setup_milling(microscope, stage)
        draw_patterns(microscope, patterns=stage.pattern.define())

        # stop milling if it is running
        microscope.stop_milling()

        # connect the acquisition signal
        microscope.fm.acquisition_signal.disconnect(self.on_fm_acquisition_signal)
        microscope.fm.acquisition_signal.connect(self.on_fm_acquisition_signal)

        # start fm
        microscope.fm.start_acquisition(channel_settings=self.config.channel_settings)

        microscope.start_milling()

        estimated_time = microscope.estimate_milling_time()
        start_time = time.time() + self.config.timeout
        estimated_end_time = start_time + estimated_time
        max_end_time = start_time + self.config.timeout
        SLEEP_DURATION = 1  # seconds
        while True:
            if microscope.get_milling_state() == MillingState.RUNNING:
                logging.info(f"Milling is running... {estimated_end_time:.2f} seconds remaining.")
                estimated_time -= SLEEP_DURATION
            elif microscope.get_milling_state() == MillingState.PAUSED: #QUERY: should we also stop fm acquisition?
                logging.info("Milling is paused. Waiting for resume...")
            elif microscope.get_milling_state() == MillingState.IDLE:
                logging.info("Milling is idle. Finishing...")
                break
            time.sleep(SLEEP_DURATION)

            # timeout
            if time.time() >= max_end_time:
                logging.info(f"{self.config.timeout} seconds have passed.")
                break

        # stop fm
        microscope.stop_milling()
        microscope.fm.stop_acquisition()
        logging.info("Milling finished. FM acquisition stopped.")

        # finalize
        microscope.fm.acquisition_signal.disconnect(self.on_fm_acquisition_signal)
        finish_milling(microscope)

        # acquire a z-stack post-milling
        image = acquire_z_stack(microscope.fm, 
                                channel_settings=self.config.channel_settings, 
                                zparams=self.config.zparams)

        # acquire a fibsem image
        image = microscope.acquire_image(beam_type=BeamType.ION)
        if parent_ui and hasattr(parent_ui, "fib_image_acquired_signal"):
            parent_ui.fib_image_acquired_signal.emit(image)

    def on_fm_acquisition_signal(self, image: FluorescenceImage):
        logging.info(f"FM Acquisition Signal Received! Acquisition Date: {image.metadata.acquisition_date}")
        logging.info(f"Image shape: {image.data.shape}, Milling State: {self.microscope.get_milling_state()}")
        # TODO: handle feedback here...

        # calc mean intensity
        mean_intensity = image.data.mean()
        self.intensities.append(mean_intensity)
        logging.info(f"Mean Intensity: {mean_intensity:.2f}, Rolling Mean (10): {sum(self.intensities[-10:]) / min(len(self.intensities), 10):.2f}")
        logging.info("-"*80)