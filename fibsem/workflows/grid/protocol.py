"""The grid tasks a protocol offers, and their settings."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol

from fibsem.workflows.grid.base import GridTaskConfig
from fibsem.workflows.grid.registry import load_grid_task_configs

if TYPE_CHECKING:
    from fibsem.workflows.grids import GridRecord


@dataclass
class GridTaskProtocol:
    """The grid tasks a protocol offers, and their settings.

    A section of the task protocol (`protocol.yaml`, under `grid_tasks`), because
    overview settings are tuned once per microscope and reused across experiments
    exactly as the lamella task settings are. Shared by every grid for now:
    screening is expected to be uniform across a magazine, and per-grid tuning is
    a documented later option (seed a per-record config from this, the way a
    lamella's is seeded from the protocol). Which tasks run in a session, and in
    what order, is chosen at run time; `order` is the default a task list shows.
    """

    name: str = "Grid Task Protocol"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_config: Dict[str, GridTaskConfig] = field(default_factory=dict)
    order: List[str] = field(default_factory=list)

    def add(self, config: GridTaskConfig) -> GridTaskConfig:
        if not config.task_name:
            raise ValueError("A grid task config needs a task_name.")
        self.task_config[config.task_name] = config
        if config.task_name not in self.order:
            self.order.append(config.task_name)
        return config

    def remove(self, task_name: str) -> None:
        self.task_config.pop(task_name, None)
        if task_name in self.order:
            self.order.remove(task_name)

    def reorder(self, order: List[str]) -> str:
        """Put the tasks in *order*: every task, each once. Why not, with
        nothing changed, or "". The Grids view's drag and the App Server's
        reorder."""
        if sorted(order) != sorted(self.ordered_task_names):
            return "Give every grid task, each once."
        self.order = list(order)
        return ""

    def set_requires(self, task_name: str, requires: List[str]) -> str:
        """Set the grid tasks *task_name* requires on the same grid, kept in
        run order. Why not, with nothing changed, or "". The Grids view's Edit
        Task and the App Server's."""
        config = self.task_config.get(task_name)
        if config is None:
            return f"No grid task named {task_name!r}."
        others = [n for n in self.ordered_task_names if n != task_name]
        wrong = [name for name in requires if name not in others]
        if wrong:
            return (
                f"{task_name} can require {', '.join(others) or 'no task'}; "
                f"not {', '.join(wrong)}."
            )
        config.requires = [n for n in others if n in requires]
        return ""

    def requirements(self, task_name: str) -> List[str]:
        """The tasks ``task_name`` requires on the same grid; none for a task
        the protocol does not have."""
        config = self.task_config.get(task_name)
        return list(config.requires) if config is not None else []

    @property
    def ordered_task_names(self) -> List[str]:
        """`order` first, then anything in `task_config` it forgot to mention."""
        names = [n for n in self.order if n in self.task_config]
        names += [n for n in self.task_config if n not in names]
        return names

    def to_dict(self) -> Dict[str, Any]:
        return {
            "_id": self.id,
            "name": self.name,
            "tasks": {k: v.to_dict() for k, v in self.task_config.items()},
            "order": list(self.ordered_task_names),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "GridTaskProtocol":
        data = data or {}
        protocol = cls(
            name=data.get("name", "Grid Task Protocol"),
            task_config=load_grid_task_configs(data.get("tasks", {})),
            order=list(data.get("order", [])),
        )
        if data.get("_id"):
            protocol.id = data["_id"]
        return protocol


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

    def sync_grids_from_inventory(self, stage: Any) -> List["GridRecord"]: ...

    def write_grid_report(self, inventory: Any) -> Path:
        """Write the grid screening report for this experiment; its path."""
        ...
