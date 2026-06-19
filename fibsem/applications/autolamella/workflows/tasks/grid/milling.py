"""Milling grid tasks (stub)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from fibsem.applications.autolamella.workflows.tasks.grid.base import GridTaskConfig


@dataclass
class ParallelTrenchMillingGridTaskConfig(GridTaskConfig):
    """Configuration for parallel trench milling task."""
    task_type: ClassVar[str] = "PARALLEL_TRENCH_MILLING_GRID"
    display_name: ClassVar[str] = "Parallel Trench Milling"
