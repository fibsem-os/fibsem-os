"""The grid task lifecycle, mirroring the lamella one.

`pre_task -> _run -> post_task`, with a failed or cancelled `_run` recorded on the
grid's history before the exception continues. Everything a task writes is
recorded on its history entry under a role, relative to the grid's directory, so
the record survives the experiment being copied off the microscope and nothing
downstream has to glob.
"""

from __future__ import annotations

import logging
import os
from abc import ABC
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

from fibsem.workflows.attention import Attention
from fibsem.workflows.grids import GridRecord
from fibsem.workflows.status import TaskStatus
from fibsem.workflows.task import BaseTask, iteration_folder
from fibsem.workflows.task_config import TaskConfig

if TYPE_CHECKING:
    from fibsem.microscope import FibsemMicroscope
    from fibsem.services.sample_handling import SampleHandling
    from fibsem.structures import GridSlot
    from fibsem.workflows.grid.protocol import GridExperiment


@dataclass
class GridTaskConfig(TaskConfig, ABC):
    """Configuration for one grid task. Lean and flat.

    No `milling` or `reference_imaging` inherited from the lamella side: no grid
    task patterns a mill or aligns to a reference. A task that needs either adds
    its own field. The exchange that brings the grid in is not in a task's
    `estimated_duration`: it is the loader's (``GridMagazine.exchange_seconds``),
    priced once per grid.
    """


class GridTask(BaseTask):
    """Base class for grid tasks: a task on a grid. Subclasses implement `_run`."""

    config_cls: ClassVar[Type[GridTaskConfig]]
    config: GridTaskConfig
    item_type: ClassVar[str] = "grid"

    def __init__(
        self,
        microscope: "FibsemMicroscope",
        config: GridTaskConfig,
        grid: GridRecord,
        experiment: "GridExperiment",
        task_manager=None,
    ) -> None:
        super().__init__(
            microscope,
            config,
            grid,
            experiment=experiment,
            task_manager=task_manager,
        )

    @property
    def grid(self) -> GridRecord:
        """The grid this task runs on: its item."""
        return self.item

    @grid.setter
    def grid(self, grid: GridRecord) -> None:
        self.item = grid

    @property
    def supervised(self) -> bool:
        """Whether this task's config attention is supervised."""
        return self.config.attention is Attention.supervised

    # -- where the grid is, and where its files go -----------------------------

    @property
    def stage(self) -> "SampleHandling":
        return self.microscope.sample_handling

    @property
    def slot(self) -> Optional["GridSlot"]:
        """The holder slot this grid is in right now, resolved live; None if not."""
        return self.stage.holder.find_slot_by_grid_name(self.grid.name)

    @property
    def output_dir(self) -> Path:
        """This task's directory under the grid's, created on first use:
        ``<grid>/<task name>/``, or ``<grid>/<task name>/<iteration:04d>/`` for a
        grid that runs in iterations, one folder per iteration."""
        directory = self.path / self.task_name
        if self.iteration is not None:
            directory = directory / iteration_folder(self.iteration)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def note_skipped_tiles(self, skipped, enabled: int) -> None:
        """Record, on this run's history entry, the tiles an overview skipped as past
        the stage's reach: the image has blanks there, and this says why."""
        if not skipped:
            return
        tiles = ", ".join(f"({row},{col})" for row, col in skipped)
        self.completion_note = (
            f"Finished; {len(skipped)} of {enabled} tiles were past the stage's reach "
            f"and skipped: {tiles}"
        )
        logging.warning(
            f"{self.task_name} on {self.grid.name}: {len(skipped)} of {enabled} tiles "
            f"are past the stage's reach and are skipped: {tiles}"
        )
