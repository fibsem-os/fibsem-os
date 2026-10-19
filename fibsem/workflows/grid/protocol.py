"""The grid tasks a protocol offers, and their settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Protocol

from fibsem.workflows.grid.base import GridTaskConfig
from fibsem.workflows.grid.registry import load_grid_task_configs
from fibsem.workflows.task_config import TaskProtocol

if TYPE_CHECKING:
    from fibsem.workflows.grids import GridRecord


@dataclass
class GridTaskProtocol(TaskProtocol):
    """The grid tasks a protocol offers, and their settings.

    A section of the task protocol (`protocol.yaml`, under `grid_tasks`), because
    overview settings are tuned once per microscope and reused across experiments
    exactly as the lamella task settings are. Shared by every grid for now:
    screening is expected to be uniform across a magazine, and per-grid tuning is
    a documented later option (seed a per-record config from this, the way a
    lamella's is seeded from the protocol).
    """

    NOUN: ClassVar[str] = "grid task"

    name: str = "Grid Task Protocol"

    @classmethod
    def load_task_configs(
        cls, ddict: Optional[Dict[str, Dict[str, Any]]]
    ) -> Dict[str, GridTaskConfig]:
        return load_grid_task_configs(ddict or {})


class GridExperiment(Protocol):
    """What the grid tasks and their manager use of an experiment: a
    ``BaseExperiment`` with ``ExperimentGrids``, plus the app's grid task
    protocol and its end-of-run report."""

    path: Path
    grids: List["GridRecord"]
    decided: Any  # psygnal Signal(proposal_id, item_name)

    @property
    def grid_protocol(self) -> GridTaskProtocol: ...

    def save(self, save_protocol: bool = False) -> None: ...

    def get_grid_by_name(self, name: str) -> Optional["GridRecord"]: ...

    def grid_path(self, grid: "GridRecord") -> Path: ...

    def item_path(self, item: Any) -> Path: ...

    def sync_grids_from_inventory(self, stage: Any) -> List["GridRecord"]: ...

    def write_grid_report(self, inventory: Any) -> Path:
        """Write the grid screening report for this experiment; its path."""
        ...
