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
from dataclasses import dataclass, field, fields
from datetime import datetime
from enum import Enum
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
    get_type_hints,
)

from fibsem.applications.autolamella.structures import (
    Attention,
    Configurable,
    GridRecord,
    Parameter,
    attention_from,
    field_specs,
)
from fibsem.structures import Display, param
from fibsem.workflows.status import TaskStatus
from fibsem.workflows.task import BaseTask

if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import Experiment
    from fibsem.microscope import FibsemMicroscope
    from fibsem.services.sample_handling import SampleHandling
    from fibsem.structures import GridSlot


@dataclass
class GridTaskConfig(Configurable, ABC):
    """Configuration for one grid task. Lean and flat.

    No `milling` or `reference_imaging` inherited from the lamella side: no grid
    task patterns a mill or aligns to a reference. A task that needs either adds
    its own field. Serialised flat -- every field at the top level, nested
    dataclasses through their own `to_dict` / `from_dict`, enums by name -- so the
    on-disk shape is the dataclass, not a `parameters` sub-dict that has to know
    what it may contain.
    """

    task_type: ClassVar[str]
    display_name: ClassVar[str]
    task_name: str = param(
        "",
        doc="The task's name, unique within the protocol: the key the workflow knows it by.",
    )  # unique within a protocol; the key the workflow uses
    # Who decides the task's record: automated (the task confirms its own) or
    # supervised (the task ends AwaitingDecision and the Review tab decides;
    # a grid task asks nothing while it runs, so that is the whole of it).
    # One per task name, shared by every grid like the rest of the config;
    # read as automated while the review preference is off.
    attention: Attention = param(
        Attention.automated,
        doc="Who decides the task's result: the task itself (automated), or the operator in the Review tab (supervised). Read as automated while reviews are off.",
        display=Display("Attention"),
    )
    # The tasks, by name, whose result this one uses, as on the lamella
    # workflow: it waits while one of them awaits a decision or is still queued
    # for the grid, and is skipped when one of them did not complete. Empty:
    # it runs whatever happened before it on the grid.
    requires: List[str] = param(
        doc="The tasks, by name, whose result this one uses. It waits while one of them awaits a decision or is still queued for the grid, and is skipped when one did not complete. Empty runs whatever happened before it.",
        display=Display("Requires"),
        default_factory=list,
    )

    @property
    def estimated_duration(self) -> float:
        """Conservative forward estimate of one run of this task on one grid, in
        seconds: what the run confirmation, the timeline and Add to queue quote.

        On the config, as on the lamella side, because those callers hold configs
        and no microscope. The exchange that brings the grid in is not the task's:
        it is the loader's (``GridMagazine.exchange_seconds``), priced once per
        grid. Zero here; each task type adds what its run does (see
        :mod:`fibsem.timing`).
        """
        return 0.0

    @property
    def parameters(self) -> Tuple[str, ...]:
        """The task-specific fields, in declaration order: what a form shows."""
        return tuple(f.name for f in fields(self) if f.name not in _WORKFLOW_FIELDS)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "task_type": self.task_type,
            "task_name": self.task_name,
            "attention": self.attention.value,
            "requires": list(self.requires),
        }
        for name in self.parameters:
            data[name] = _serialise(getattr(self, name))
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GridTaskConfig":
        hints = get_type_hints(cls)
        kwargs: Dict[str, Any] = {}
        for f in fields(cls):
            if f.name not in data:
                continue
            if f.name == "attention":
                kwargs[f.name] = attention_from(
                    data[f.name], f"grid task '{data.get('task_name', '')}'"
                )
                continue
            if f.name == "requires":
                kwargs[f.name] = _requires(data[f.name], data.get("task_name", ""))
                continue
            kwargs[f.name] = _deserialise(hints.get(f.name), data[f.name])
        unknown = set(data) - {f.name for f in fields(cls)} - {"task_type"}
        for key in sorted(unknown):
            logging.warning(f"Unknown field '{key}' in {cls.__name__}; ignored.")
        return cls(**kwargs)


# Fields that say how a task takes part in the workflow, not how it runs: not
# form parameters, and serialised by the base.
_WORKFLOW_FIELDS = ("task_name", "attention", "requires")


def _requires(value: Any, task_name: str) -> List[str]:
    """Stored requirements as task names, or none with a warning: a malformed
    value must not drop the task from the protocol."""
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return list(value)
    logging.warning(
        f"Grid task '{task_name}' has requires {value!r}, not a list of task "
        "names; read as none."
    )
    return []


def _serialise(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_serialise(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialise(v) for k, v in value.items()}
    return value


def _deserialise(hint: Any, value: Any) -> Any:
    """Rebuild a field from its serialised form, guided by the annotation."""
    if value is None or hint is None:
        return value
    origin = getattr(hint, "__origin__", None)
    if origin is Union:  # Optional[X]
        inner = [a for a in hint.__args__ if a is not type(None)]
        return _deserialise(inner[0], value) if len(inner) == 1 else value
    if origin in (list, List) and isinstance(value, list):  # List[X]
        (item_hint,) = getattr(hint, "__args__", (None,))
        return [_deserialise(item_hint, v) for v in value]
    if isinstance(hint, type):
        if issubclass(hint, Enum) and isinstance(value, str):
            return hint[value]
        if hasattr(hint, "from_dict") and isinstance(value, dict):
            return hint.from_dict(value)
    return value


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
        experiment: "Experiment",
        parent_ui=None,
        task_manager=None,
    ) -> None:
        super().__init__(
            microscope,
            config,
            grid,
            experiment=experiment,
            parent_ui=parent_ui,
            task_manager=task_manager,
        )
        # What the history entry says on success, when there is more to say than
        # "Finished": an overview that skipped tiles out of the stage's reach.
        self.completion_note = ""

    @property
    def grid(self) -> GridRecord:
        """The grid this task runs on: its item."""
        return self.item

    @grid.setter
    def grid(self, grid: GridRecord) -> None:
        self.item = grid

    @property
    def review(self) -> bool:
        """Whether this task ends waiting on a decision in the Review tab: its
        config's attention, and the feature flag as the manager read it for
        this run. Otherwise the task confirms its own record."""
        manager = self.task_manager
        if manager is None or not getattr(manager, "review_enabled", False):
            return False
        return self.config.attention is Attention.supervised

    @property
    def result_images(self) -> Dict[str, str]:
        """The image a proposal points at, by provenance key: the file recorded
        under the config's role (the stitched overview, not its thumbnail; the
        operator is judging the image). Empty for a task with no role."""
        role = getattr(self.config, "role", None)
        return {"reference_image": role} if role else {}

    # -- where the grid is, and where its files go -----------------------------

    @property
    def stage(self) -> "SampleHandling":
        return self.microscope.sample_handling

    @property
    def slot(self) -> Optional["GridSlot"]:
        """The holder slot this grid is in right now, resolved live; None if not."""
        return self.stage.holder.find_slot_by_grid_name(self.grid.name)

    @property
    def path(self) -> Path:
        """The grid's directory: `experiment.path / grids / <name>`."""
        return self.experiment.grid_path(self.grid)

    @property
    def output_dir(self) -> Path:
        """This task's directory under the grid's, created on first use."""
        directory = self.path / self.task_name
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    # -- lifecycle -------------------------------------------------------------

    def pre_task(self) -> None:
        logging.info(
            f"Running {self.task_name}, {self.task_type} ({self.task_id}) "
            f"for grid {self.grid.name} ({self.grid.id})"
        )
        self._start_task_state()
        self._set_workflow_metadata()
        self.log_status_message("STARTED", "Started")

    def post_task(self) -> None:
        self.grid.task_state.status = TaskStatus.Completed
        self.grid.task_state.status_message = ""
        self.log_status_message("FINISHED", self.completion_note or "Finished")
        self._record_outcome()

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

    # -- outputs ---------------------------------------------------------------

    def record_output(self, role: str, output: Union[str, Path, Any, None]) -> None:
        """Record a file this run wrote, under `role`, relative to the grid's directory.

        Accepts a path or anything with a `filepath` (a saved image). Nothing that
        was never written is recorded; the same path twice is still one file.
        """
        if output is None:
            return
        filepath = getattr(output, "filepath", output)
        if filepath is None:
            return
        # Forward slashes whatever the platform: the record is a description of
        # files that travels with the experiment, not a path for this OS to open,
        # and `os.path.join` reads it back correctly on either.
        relative = Path(os.path.relpath(str(filepath), str(self.path))).as_posix()
        paths = self.grid.task_state.outputs.setdefault(role, [])
        if relative not in paths:
            paths.append(relative)
