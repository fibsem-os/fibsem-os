"""Cryo grid tasks: deposition / sputter (stubs) + cryo-cleaning milling."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import ClassVar, Literal, Optional, Type

from fibsem.applications.autolamella.workflows.tasks.grid.base import (
    GridTask,
    GridTaskConfig,
)
from fibsem.structures import BeamType


@dataclass
class CryoDepositionGridTaskConfig(GridTaskConfig):
    """Configuration for cryo (GIS) deposition task."""
    task_type: ClassVar[str] = "CRYO_DEPOSITION_GRID"
    display_name: ClassVar[str] = "Cryo GIS Deposition"
    orientation: Optional[Literal["SEM", "FIB", "MILLING"]] = "SEM"
    deposition_time: float = 30.0  # seconds


@dataclass
class CryoSputterGridTaskConfig(GridTaskConfig):
    """Configuration for cryo sputter task."""
    task_type: ClassVar[str] = "CRYO_SPUTTER_GRID"
    display_name: ClassVar[str] = "Cryo Sputter"
    sputter_time: float = 60.0  # seconds
    sputter_voltage: float = 5.0  # kV
    sputter_current: float = 0.1  # nA


class CryoDepositionGridTask(GridTask):
    """Task to deposit a protective layer on the grid via the GIS source."""
    config_cls: ClassVar[Type[GridTaskConfig]] = CryoDepositionGridTaskConfig
    config: CryoDepositionGridTaskConfig

    def _run(self):
        """Deposit a protective layer on the grid using the GIS.

        Move to the grid, drop the stage to a safe GIS insertion height, insert
        and open the GIS port for the configured time, then retract and restore
        the initial microscope state. Honours the stop event during deposition.
        """
        from fibsem.microscopes.autoscript import AutoscriptGISPort

        slot = self.slot
        if slot is None:
            raise RuntimeError(f"Grid '{self.grid.name}' is not loaded in any slot.")

        logging.info(f"Starting GIS deposition for grid {self.grid.name}")

        # move to grid position
        self.update_status_ui(f"Moving to grid {self.grid.name}...")
        self._move_to_grid_slot_position(self.config.orientation)

        # supervised checkpoint before the (destructive) GIS insertion
        if not self.ask_user(
            f"Start GIS deposition on '{self.grid.name}' "
            f"({self.config.deposition_time:.0f}s)?",
            pos="Start", neg="Skip",
        ):
            self.update_status_ui("GIS deposition skipped by user.")
            logging.info(f"GIS deposition skipped for grid {self.grid.name}")
            return

        # capture state to restore once the GIS is retracted
        initial_state = self.microscope.get_microscope_state()

        gis_port = AutoscriptGISPort(self.microscope)

        # drop the stage to a safe insertion height and verify before inserting
        self.update_status_ui("Moving to safe GIS position...")
        gis_port._move_to_safe_gis_position()
        gis_port._run_safety_check()

        # the port owns the insert/open/wait/close/retract cycle; we pass the
        # stop event + a progress callback so it stays abort-aware and on-screen
        gis_port.run_deposition(
            self.config.deposition_time,
            stop_event=self._stop_event,
            on_progress=lambda remaining: self.update_status_ui(
                f"Depositing... {remaining:.0f}s remaining"
            ),
        )

        self.microscope.set_microscope_state(initial_state)
        logging.info(f"Completed GIS deposition for grid {self.grid.name}")
        self.record_result(deposition_time=self.config.deposition_time)


class CryoSputterGridTask(GridTask):
    """Task to sputter-coat the grid (stub: logs only)."""
    config_cls: ClassVar[Type[GridTaskConfig]] = CryoSputterGridTaskConfig
    config: CryoSputterGridTaskConfig

    def _run(self):
        # TODO: implement sputter coating (move to grid, run sputter source)
        logging.info(
            f"Cryo sputter on grid {self.grid.name} "
            f"({self.config.sputter_time:.0f}s at {self.config.sputter_voltage:.1f} kV) "
            f"— not yet implemented."
        )
        self.update_status_ui("Cryo sputter (not implemented)")


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

        # run the FIB clean for the configured duration (abort-aware countdown)
        self.microscope.start_acquisition(beam_type=BeamType.ION)
        try:
            self.wait_with_progress(self.config.duration, "Cryo cleaning")
        finally:
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
