import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

from fibsem.cancellation import raise_if_cancelled
from fibsem.microscope import FibsemMicroscope
from fibsem.milling import setup_milling
from fibsem.milling.base import (
    FibsemMillingStage,
    MillingStrategy,
    MillingStrategyConfig,
)
from fibsem.milling.progress import MillingProgress, MillingProgressStatus


@dataclass
class StandardMillingConfig(MillingStrategyConfig):
    """Configuration for standard milling strategy"""

    pass


class StandardMillingStrategy(MillingStrategy[StandardMillingConfig]):
    """Basic milling strategy that mills continuously until completion"""

    name: str = "Standard"
    fullname: str = "Standard Milling"
    config_class = StandardMillingConfig

    def run(
        self,
        microscope: FibsemMicroscope,
        stage: FibsemMillingStage,
        asynch: bool = False,
        parent_ui=None,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        logging.info(f"Running {self.name} Milling Strategy for {stage.name}")
        setup_milling(microscope, milling_stage=stage, stop_event=stop_event)

        microscope.draw_patterns(stage.define_patterns())

        estimated_time = microscope.estimate_milling_time()
        logging.info(f"Estimated time for {stage.name}: {estimated_time:.2f} seconds")

        # The strategy's own words. This used to carry no `state` at all, so it matched
        # no branch in any of the three consumers and rendered nowhere -- the one
        # message a strategy sends was precisely the one that was dropped. A consumer
        # keeps it standing across the messageless ticks that follow, which is what a
        # *delegating* strategy needs: `run_milling` below hands the loop to a backend
        # that has no idea what this strategy calls itself.
        microscope.milling_progress_signal.emit(
            MillingProgress(
                status=MillingProgressStatus.STAGE_UPDATE,
                message=f"Running {stage.name}...",
                stage_name=stage.name,
                start_time=time.time(),
                estimated_time=estimated_time,
            )
        )

        raise_if_cancelled(stop_event)  # last chance before the beam starts
        microscope.run_milling(
            milling_current=stage.milling.milling_current,
            milling_voltage=stage.milling.milling_voltage,
            asynch=asynch,
        )
