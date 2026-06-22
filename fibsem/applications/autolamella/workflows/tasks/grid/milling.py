"""Milling grid tasks (stub)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import ClassVar, Type

from fibsem.applications.autolamella.workflows.tasks.grid.base import (
    GridTask,
    GridTaskConfig,
)


@dataclass
class ParallelTrenchMillingGridTaskConfig(GridTaskConfig):
    """Configuration for parallel trench milling task."""
    task_type: ClassVar[str] = "PARALLEL_TRENCH_MILLING_GRID"
    display_name: ClassVar[str] = "Parallel Trench Milling"


class ParallelTrenchMillingGridTask(GridTask):
    """Task to mill parallel trenches across the grid (stub: logs only)."""
    config_cls: ClassVar[Type[GridTaskConfig]] = ParallelTrenchMillingGridTaskConfig
    config: ParallelTrenchMillingGridTaskConfig

    def _run(self):
        # TODO: implement parallel trench milling
        logging.info(
            f"Parallel trench milling on grid {self.grid.name} — not yet implemented."
        )
        self.update_status_ui("Parallel trench milling (not implemented)")
