import logging
from dataclasses import dataclass

from fibsem.microscope import FibsemMicroscope
from fibsem.milling import setup_milling
from fibsem.milling.base import (FibsemMillingStage, MillingStrategy,
                                 MillingStrategyConfig)

import time

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
        parent_ui = None,
    ) -> None:
        logging.info(f"Running {self.name} Milling Strategy for {stage.name}")
        setup_milling(microscope, milling_stage=stage)

        if parent_ui and hasattr(parent_ui, "_milling_stop_event"):
            if parent_ui._milling_stop_event.is_set():
                logging.info(f"Stopping {self.name} Milling Strategy for {stage.name}")
                return

        microscope.draw_patterns(stage.define_patterns())

        estimated_time = microscope.estimate_milling_time()
        logging.info(f"Estimated time for {stage.name}: {estimated_time:.2f} seconds")

        if parent_ui and hasattr(parent_ui, "milling_progress_signal"):
            parent_ui.milling_progress_signal.emit({"msg": f"Running {stage.name}...", 
                                                    "progress": 
                                                        {"started": True,
                                                            "start_time": time.time(), 
                                                            "estimated_time": estimated_time,
                                                            "name": stage.name}
                                                        })

        microscope.run_milling(
            milling_current=stage.milling.milling_current,
            milling_voltage=stage.milling.milling_voltage,
            asynch=asynch,
        )
