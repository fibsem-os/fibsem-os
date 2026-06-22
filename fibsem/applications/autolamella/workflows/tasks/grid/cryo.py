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
    """Configuration for cryo sputter-coating task.

    The Autoscript sputter coater exposes a run *time* and a *current* (Amps) —
    there is no voltage control, so it isn't configured here.
    """
    task_type: ClassVar[str] = "CRYO_SPUTTER_GRID"
    display_name: ClassVar[str] = "Cryo Sputter"
    orientation: Optional[Literal["SEM", "FIB", "MILLING"]] = "SEM"
    sputter_time: float = 60.0  # seconds
    sputter_current: float = 0.01  # A (10 mA)


class CryoDepositionGridTask(GridTask):
    """Task to deposit a protective layer on the grid via the GIS source."""
    config_cls: ClassVar[Type[GridTaskConfig]] = CryoDepositionGridTaskConfig
    config: CryoDepositionGridTaskConfig

    def _run(self):
        """Deposit a protective layer on the grid using the GIS.

        Move to the grid, then run the GIS deposition. The microscope owns the
        GIS mechanics (safe insertion height, insert/open/wait/close/retract and
        state restore); we pass the stop event + a progress callback so it stays
        abort-aware and on-screen.
        """
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

        self.update_status_ui("Depositing...")
        self.microscope.run_gis_deposition(
            self.config.deposition_time,
            stop_event=self._stop_event,
            on_progress=lambda remaining: self.update_status_ui(
                f"Depositing... {remaining:.0f}s remaining"
            ),
        )

        logging.info(f"Completed GIS deposition for grid {self.grid.name}")
        self.record_result(deposition_time=self.config.deposition_time)


class CryoSputterGridTask(GridTask):
    """Task to sputter-coat the grid via the sputter coater."""
    config_cls: ClassVar[Type[GridTaskConfig]] = CryoSputterGridTaskConfig
    config: CryoSputterGridTaskConfig

    def _run(self):
        """Sputter-coat the grid.

        Move to the grid, then run the sputter coater for the configured time
        and current. ``run_sputter_coater`` is a single blocking hardware call
        (it owns its own prepare/run/recover), so there is no interruptible
        countdown here.
        """
        slot = self.slot
        if slot is None:
            raise RuntimeError(f"Grid '{self.grid.name}' is not loaded in any slot.")

        logging.info(f"Starting sputter coating for grid {self.grid.name}")

        # move to grid position
        self.update_status_ui(f"Moving to grid {self.grid.name}...")
        self._move_to_grid_slot_position(self.config.orientation)

        # supervised checkpoint before sputtering
        if not self.ask_user(
            f"Start sputter coating on '{self.grid.name}' "
            f"({self.config.sputter_time:.0f}s at {self.config.sputter_current * 1e3:.1f} mA)?",
            pos="Start", neg="Skip",
        ):
            self.update_status_ui("Sputter coating skipped by user.")
            logging.info(f"Sputter coating skipped for grid {self.grid.name}")
            return

        # single blocking hardware call (prepare/run/recover live in the driver)
        self.update_status_ui(f"Sputter coating... ({self.config.sputter_time:.0f}s)")
        self.microscope.run_sputter_coater(
            int(self.config.sputter_time), current=self.config.sputter_current
        )

        logging.info(f"Completed sputter coating for grid {self.grid.name}")
        self.record_result(sputter_time=self.config.sputter_time,
                           sputter_current=self.config.sputter_current)


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
