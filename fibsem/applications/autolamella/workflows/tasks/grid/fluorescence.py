"""The fluorescence overview grid task: a thin wrapper over the FM tiled runner.

The operation is `fibsem.fm.acquisition.acquire_fm_tiled_overview`, which the FM
Overview tab and the imaging service call too. The task adds the travel to the
FM, the objective, the grid's own directory, and the record: the mosaic and a
channel-composite thumbnail.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, List, Optional, Tuple, Type

from fibsem import timing
from fibsem.applications.autolamella.workflows.tasks.grid.base import (
    GridTask,
    GridTaskConfig,
)
from fibsem.autofunctions.autofocus import AutoFocusSettings
from fibsem.fm.acquisition import OverviewDestination, acquire_fm_tiled_overview
from fibsem.fm.preview import composite_projection
from fibsem.fm.structures import ChannelSettings, OverviewParameters, ZParameters
from fibsem.fm.timing import estimate_fm_overview_time
from fibsem.imaging.thumbnail import write_thumbnail
from fibsem.imaging.tiled import stamped_overview_name
from fibsem.imaging.tiling.reach import reachable_fluorescence_overview
from fibsem.services.sample_handling import uncalibrated_message
from fibsem.structures import (
    Display,
    FibsemStagePosition,
    InsertableDeviceState,
    Parameter,
    param,
)

if TYPE_CHECKING:
    from fibsem.microscope import FibsemMicroscope


@dataclass
class FluorescenceOverviewGridTaskConfig(GridTaskConfig):
    """A tiled fluorescence overview of the grid: the FM Overview tab's inputs."""

    task_type: ClassVar[str] = "FM_OVERVIEW_GRID"
    display_name: ClassVar[str] = "Fluorescence overview"
    channels: List[ChannelSettings] = param(
        doc="The FM channels each tile is acquired in.",
        default_factory=lambda: [ChannelSettings(name="Channel-01")],
    )
    # `overview`, not `parameters`: the base config's `parameters` is the list
    # of a form's fields, and a field of that name would shadow it.
    overview: OverviewParameters = param(
        doc="The grid of tiles, its order, focus and registration.",
        default_factory=OverviewParameters,
    )
    zparams: ZParameters = param(
        doc="The z-stack each tile takes, when the overview uses one.",
        default_factory=ZParameters,
    )
    autofocus_settings: Optional[AutoFocusSettings] = param(
        None, doc="How to autofocus, when the overview does. Empty uses the defaults."
    )
    filename: str = param(
        "overview",
        doc="The stem of the name the overview is saved under, in the grid's folder; a time stamp is added per run.",
        display=Display("Filename"),
    )

    role: ClassVar[str] = "overview_fm"

    @property
    def estimated_duration(self) -> float:
        """What ``_run`` does, in order: the move to the FM, the objective in, the
        overview (`estimate_fm_overview_time`, which counts the runner's return of
        the stage and objective), the objective out. Inserting and retracting are
        always counted, though the task skips both when the objective is already
        in: which it is, is runtime state. Errs long.
        """
        if self.overview.n_enabled_tiles <= 0:
            return 0.0
        total = estimate_fm_overview_time(
            self.channels,
            self.overview,
            zparams=self.zparams,
            autofocus_settings=self.autofocus_settings,
        )
        total += timing.stage_move_cost(1) + timing.OBJECTIVE_INSERT_S
        total += timing.OBJECTIVE_RETRACT_S
        return total


class FluorescenceOverviewGridTask(GridTask):
    """Move to the FM, put the objective in, acquire, record, and put things back.

    The objective is inserted for the run if it was not already, and returned to
    how it was found afterwards: a grid exchange with the objective in is not a
    thing to leave possible by accident.
    """

    config_cls: ClassVar[Type[GridTaskConfig]] = FluorescenceOverviewGridTaskConfig
    config: FluorescenceOverviewGridTaskConfig

    def grid_centre(self) -> FibsemStagePosition:
        """The grid's calibrated slot position, as the FM sees it.

        One call on either mounting (`to_device`): a compustage gets the flip, an
        offset mount gets the traverse, and the pose is kept where the objective
        images from it and otherwise put into the first orientation it declares.
        """
        slot = self.slot
        if slot is None:
            raise RuntimeError(
                f"Grid '{self.grid.name}' is not in a holder slot. Load it first."
            )
        if slot.position is None:
            raise RuntimeError(uncalibrated_message(slot.name))
        return self.microscope.to_device(slot.position, "FM")

    def _run(self) -> None:
        fm = getattr(self.microscope, "fm", None)
        if fm is None:
            raise RuntimeError("This system has no fluorescence microscope.")
        centre = self.grid_centre()

        # Read before the travel: on a compustage `move_to_device("FM")` inserts the
        # objective itself, and "how it was found" means before any of this ran.
        objective = fm.objective
        was_inserted = objective.state is InsertableDeviceState.INSERTED

        self.log_status_message("MOVE_TO_FM", f"Moving {self.grid.name} to the FM")
        self.microscope.move_to_device("FM")
        self._check_for_abort()

        if objective.state is not InsertableDeviceState.INSERTED:
            self.log_status_message("INSERT_OBJECTIVE", "Inserting the objective")
            objective.insert()
        try:
            overview, skipped = reachable_fluorescence_overview(
                self.microscope, self.config.overview, centre
            )
            enabled = self.config.overview.n_enabled_tiles
            if skipped and overview.n_enabled_tiles == 0:
                raise RuntimeError(
                    f"None of the {enabled} tiles of this overview is within the "
                    f"stage's reach from the centre of {self.grid.name}. Make it "
                    "smaller."
                )
            self.note_skipped_tiles(skipped, enabled)
            out_of_reach = f", {len(skipped)} out of reach" if skipped else ""
            self.log_status_message(
                "ACQUIRE",
                f"Acquiring fluorescence overview: {overview.rows} x "
                f"{overview.cols} tiles{out_of_reach}, "
                f"{len(self.config.channels)} channel(s)",
            )
            self.output_dir.mkdir(parents=True, exist_ok=True)
            overview_result = acquire_fm_tiled_overview(
                self.microscope,
                self.config.channels,
                overview,
                OverviewDestination.create(
                    str(self.output_dir), stamped_overview_name(self.config.filename)
                ),
                zparams=self.config.zparams if overview.use_zstack else None,
                autofocus_settings=self.config.autofocus_settings,
                centre_position=centre,
                stop_event=self._stop_event,
            )
            mosaic, saved = overview_result.image, overview_result.mosaic_path
            if saved is None:
                raise RuntimeError("The fluorescence overview could not be saved.")
            self.record_output(self.config.role, saved)
            try:
                thumbnail = write_thumbnail(
                    composite_projection(mosaic),
                    self.output_dir / f"{Path(saved).name.split('.')[0]}-thumbnail.png",
                )
                self.record_output(f"{self.config.role}_thumbnail", thumbnail)
            except Exception as e:  # noqa: BLE001 - the overview is already recorded
                logging.warning(f"Could not write the fluorescence thumbnail: {e}")
        finally:
            if not was_inserted:
                self.log_status_message("RETRACT_OBJECTIVE", "Retracting the objective")
                objective.retract()
        self.log_status_message("ACQUIRED", "Fluorescence overview recorded")
