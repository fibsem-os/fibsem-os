"""The fluorescence overview grid task: a thin wrapper over the FM tiled runner.

The operation -- `acquire_fluorescence_overview` -- is a plain function over
`FMTiledAcquisitionRunner` and `OverviewDestination`, laying files out as the FM
Overview tab does. The task adds the travel to the FM, the objective, the grid's
own directory, and the record: the mosaic and a channel-composite thumbnail.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, List, Optional, Tuple, Type, Union

from fibsem.applications.autolamella.workflows.tasks.grid.base import (
    GridTask,
    GridTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.grid.registry import (
    register_grid_task,
)
from fibsem.autofunctions.autofocus import AutoFocusSettings
from fibsem.cancellation import OperationCancelledError
from fibsem.fm.acquisition import FMTiledAcquisitionRunner, OverviewDestination
from fibsem.fm.preview import composite_projection
from fibsem.fm.structures import ChannelSettings, OverviewParameters, ZParameters
from fibsem.imaging.thumbnail import write_thumbnail
from fibsem.imaging.tiled import stamped_overview_name
from fibsem.imaging.tiling.progress import (
    MODALITY_FLUORESCENCE,
    TiledProgress,
    TiledStatus,
)
from fibsem.microscopes._stage import uncalibrated_message
from fibsem.structures import FibsemStagePosition

if TYPE_CHECKING:
    from fibsem.fm.structures import FluorescenceImage
    from fibsem.microscope import FibsemMicroscope

# ---------------------------------------------------------------------------


def acquire_fluorescence_overview(
    microscope: "FibsemMicroscope",
    channels: List[ChannelSettings],
    parameters: OverviewParameters,
    centre: FibsemStagePosition,
    directory: Union[str, Path],
    stem: str = "overview",
    zparams: Optional[ZParameters] = None,
    autofocus_settings: Optional[AutoFocusSettings] = None,
    stop_event=None,
) -> Tuple["FluorescenceImage", Optional[str]]:
    """Acquire a fluorescence tileset centred on `centre`, saved under `directory`.

    Returns the stitched mosaic and where it was written (None if the write
    failed; the mosaic is still returned). The tiles land in a directory of the
    run's name beside the mosaic, as the FM Overview tab lays them out.
    """
    Path(directory).mkdir(parents=True, exist_ok=True)
    destination = OverviewDestination.create(
        str(directory), stamped_overview_name(stem)
    )
    runner = FMTiledAcquisitionRunner(
        microscope=microscope,
        channel_settings=list(channels),
        overview_parameters=parameters,
        zparams=zparams if parameters.use_zstack else None,
        autofocus_settings=autofocus_settings,
        save_directory=destination.tiles_directory,
        stop_event=stop_event,
        centre_position=centre,
    )

    # The runner reports up to the stitch and leaves the save and the ending to
    # whoever does the save: on the FM Overview tab that is the widget, here it
    # is this function. Without the terminal report the window's status bar
    # stays on "Stitching tiles" after the run has finished.
    def report(status: TiledStatus, error: Optional[str] = None) -> None:
        microscope.tiled_acquisition_signal.emit(
            TiledProgress(status=status, modality=MODALITY_FLUORESCENCE, error=error)
        )

    try:
        mosaic = runner.run_and_stitch()
        report(TiledStatus.SAVING)
        path = destination.save_mosaic(mosaic)
    except OperationCancelledError:
        report(TiledStatus.CANCELLED)
        raise
    except Exception as e:
        report(TiledStatus.FAILED, error=str(e))
        raise
    report(TiledStatus.FINISHED)
    return mosaic, path


@dataclass
class FluorescenceOverviewGridTaskConfig(GridTaskConfig):
    """A tiled fluorescence overview of the grid: the FM Overview tab's inputs."""

    task_type: ClassVar[str] = "FM_OVERVIEW_GRID"
    display_name: ClassVar[str] = "Fluorescence overview"
    channels: List[ChannelSettings] = field(
        default_factory=lambda: [ChannelSettings(name="Channel-01")]
    )
    # `overview`, not `parameters`: the base config's `parameters` is the list
    # of a form's fields, and a field of that name would shadow it.
    overview: OverviewParameters = field(default_factory=OverviewParameters)
    zparams: ZParameters = field(default_factory=ZParameters)
    autofocus_settings: Optional[AutoFocusSettings] = None
    filename: str = "overview"

    role: ClassVar[str] = "overview_fm"


@register_grid_task
class FluorescenceOverviewGridTask(GridTask):
    """Move to the FM, put the objective in, acquire, record, and put things back.

    The objective is inserted for the run if it was not already, and returned to
    how it was found afterwards: a grid exchange with the objective in is not a
    thing to leave possible by accident.
    """

    config_cls: ClassVar[Type[GridTaskConfig]] = FluorescenceOverviewGridTaskConfig
    config: FluorescenceOverviewGridTaskConfig

    def grid_centre(self) -> FibsemStagePosition:
        """The grid's calibrated slot position, at the FM.

        On a compustage the FM is an orientation (a flip); on an offset mount it is
        a device the stage travels to with its pose kept. `get_target_position`
        spells the two differently, and refuses the wrong one, so the branch is here.
        """
        slot = self.slot
        if slot is None:
            raise RuntimeError(
                f"Grid '{self.grid.name}' is not in a holder slot. Load it first."
            )
        if slot.position is None:
            raise RuntimeError(uncalibrated_message(slot.name))
        if self.microscope.stage_is_compustage:
            return self.microscope.get_target_position(slot.position, "FM")
        return self.microscope.get_target_position(slot.position, target_device="FM")

    def _run(self) -> None:
        fm = getattr(self.microscope, "fm", None)
        if fm is None:
            raise RuntimeError("This system has no fluorescence microscope.")
        centre = self.grid_centre()

        # Read before the travel: on a compustage `move_to_device("FM")` inserts the
        # objective itself, and "how it was found" means before any of this ran.
        objective = fm.objective
        was_inserted = objective.state == "Inserted"

        self.log_status_message("MOVE_TO_FM", f"Moving {self.grid.name} to the FM")
        self.microscope.move_to_device("FM")
        self._check_for_abort()

        if objective.state != "Inserted":
            self.log_status_message("INSERT_OBJECTIVE", "Inserting the objective")
            objective.insert()
        try:
            self.log_status_message(
                "ACQUIRE",
                f"Acquiring fluorescence overview: {self.config.overview.rows} x "
                f"{self.config.overview.cols} tiles, "
                f"{len(self.config.channels)} channel(s)",
            )
            mosaic, saved = acquire_fluorescence_overview(
                self.microscope,
                self.config.channels,
                self.config.overview,
                centre,
                self.output_dir,
                stem=self.config.filename,
                zparams=self.config.zparams,
                autofocus_settings=self.config.autofocus_settings,
                stop_event=self._stop_event,
            )
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
