"""Cryo grid tasks: deposition / sputter (stubs) + cryo-cleaning milling."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import ClassVar, Literal, Optional, Type

from fibsem.applications.autolamella.workflows.tasks.grid.base import (
    GridTask,
    GridTaskConfig,
)
from fibsem.structures import BeamType


@dataclass
class CryoDepositionGridTaskConfig(GridTaskConfig):
    """Configuration for cryo deposition task."""
    task_type: ClassVar[str] = "CRYO_DEPOSITION_GRID"
    display_name: ClassVar[str] = "Cryo Deposition"
    deposition_time: float = 30.0  # seconds


@dataclass
class CryoSputterGridTaskConfig(GridTaskConfig):
    """Configuration for cryo sputter task."""
    task_type: ClassVar[str] = "CRYO_SPUTTER_GRID"
    display_name: ClassVar[str] = "Cryo Sputter"
    sputter_time: float = 60.0  # seconds
    sputter_voltage: float = 5.0  # kV
    sputter_current: float = 0.1  # nA


@dataclass
class CryoCleaningGridTaskConfig(GridTaskConfig):
    """Configuration for cryo cleaning milling task."""
    task_type: ClassVar[str] = "CRYO_CLEANING_GRID"
    display_name: ClassVar[str] = "Cryo Cleaning Milling"
    orientation: Optional[Literal["SEM", "FIB", "MILLING"]] = "SEM"
    milling_angle: float = 38.0 # degrees
    field_of_view: float = 900e-6  # meters
    duration: float = 10.0  # seconds
    current: float = 15e-9  # A


class CryoCleaningGridTask(GridTask):
    """Task to perform cryo cleaning on the sample grid."""
    config_cls: ClassVar[Type[GridTaskConfig]] = CryoCleaningGridTaskConfig
    config: CryoCleaningGridTaskConfig

    # ref: https://www.nature.com/articles/s41467-025-57493-3

    def _run(self):
        """Perform cryo cleaning on the sample grid using FIB."""
        slot = self.slot
        if slot is None:
            raise RuntimeError(f"Grid '{self.grid.name}' is not loaded in any slot.")

        logging.info(f"Starting cryo cleaning for grid {self.grid.name}")

        # move to grid position
        self.update_status_ui(f"Moving to grid {self.grid.name}...")
        self._move_to_grid_slot_position(self.config.orientation)

        # supervised checkpoint before the (destructive) FIB clean
        if not self.ask_user(
            f"Start cryo cleaning on '{self.grid.name}' "
            f"({self.config.duration:.0f}s at {self.config.current * 1e9:.1f} nA)?",
            pos="Start", neg="Skip",
        ):
            self.update_status_ui("Cryo cleaning skipped by user.")
            logging.info(f"Cryo cleaning skipped for grid {self.grid.name}")
            return

        # set beam parameters
        self.microscope.set_beam_current(self.config.current, beam_type=BeamType.ION)
        self.microscope.set_field_of_view(self.config.field_of_view, beam_type=BeamType.ION)

        # start timer for duration
        start_time = time.time()
        self.microscope.start_acquisition(beam_type=BeamType.ION)
        while (time.time() - start_time) < self.config.duration:
            if self._stop_event and self._stop_event.is_set():
                logging.info("Cryo cleaning task stopped.")
                break
            time.sleep(1)  # wait for 1 second before checking again
            remaining_time = self.config.duration - (time.time() - start_time)
            self.update_status_ui(f"Cryo cleaning... {remaining_time:.0f}s remaining")
        self.microscope.stop_acquisition()

        # restore previous settings if needed
        self.microscope.set_beam_current(self.microscope.system.ion.beam.beam_current, beam_type=BeamType.ION)

        # Implement the cryo cleaning logic here
        logging.info(f"Completed cryo cleaning for grid {self.grid.name}")
        # acquire image
        image = self.microscope.acquire_image(beam_type=BeamType.ION)
        path = self.task_dir()
        fib_path = os.path.join(path, "post-grid-cleaining_ib.tif")
        image.save(fib_path)
        self.record_result(fib=fib_path)
