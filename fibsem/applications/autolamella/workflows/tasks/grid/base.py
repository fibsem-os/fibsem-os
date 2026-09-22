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
import uuid
from abc import ABC, abstractmethod
from copy import deepcopy
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

from fibsem.applications.autolamella.proposals import Proposer, TaskResultProposer
from fibsem.applications.autolamella.structures import (
    Attention,
    AutoLamellaTaskStatus,
    GridRecord,
    attention_from,
    get_fields_with_metadata,
)
from fibsem.applications.autolamella.workflows.tasks.proposing import settle
from fibsem.cancellation import OperationCancelledError

if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import Experiment
    from fibsem.microscope import FibsemMicroscope
    from fibsem.microscopes._stage import GridSlot, Stage

_LIFECYCLE_STEPS = {"STARTED", "FINISHED"}


@dataclass
class GridTaskConfig(ABC):
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
    task_name: str = ""  # unique within a protocol; the key the workflow uses
    # Who decides the task's record: automated (the task confirms its own) or
    # supervised (the task ends AwaitingDecision and the Review tab decides;
    # a grid task asks nothing while it runs, so that is the whole of it).
    # One per task name, shared by every grid like the rest of the config;
    # read as automated while the review preference is off.
    attention: Attention = Attention.automated
    # The tasks, by name, whose result this one uses, as on the lamella
    # workflow: it waits while one of them awaits a decision or is still queued
    # for the grid, and is skipped when one of them did not complete. Empty:
    # it runs whatever happened before it on the grid.
    requires: List[str] = field(default_factory=list)

    @property
    def parameters(self) -> Tuple[str, ...]:
        """The task-specific fields, in declaration order: what a form shows."""
        return tuple(f.name for f in fields(self) if f.name not in _WORKFLOW_FIELDS)

    @property
    def field_metadata(self) -> Dict[str, Dict[str, Any]]:
        return get_fields_with_metadata(self.__class__)

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


class GridTask(ABC):
    """Base class for grid tasks. Subclasses implement `_run`."""

    config_cls: ClassVar[Type[GridTaskConfig]]
    config: GridTaskConfig
    # What this task type proposes for the Review tab, as on AutoLamellaTask:
    # its result, with the image it recorded under its role.
    proposer: ClassVar[Optional[Proposer]] = TaskResultProposer()

    def __init__(
        self,
        microscope: "FibsemMicroscope",
        config: GridTaskConfig,
        grid: GridRecord,
        experiment: "Experiment",
        parent_ui=None,
        task_manager=None,
    ) -> None:
        self.microscope = microscope
        self.config = config
        self.grid = grid
        self.experiment = experiment
        self.parent_ui = parent_ui
        self.task_manager = task_manager
        self.task_id = str(uuid.uuid4())
        # The manager's token, not its raw event: what counts as "cancelled" is the
        # manager's to decide. Same line as AutoLamellaTask, on purpose.
        self._stop_event = task_manager.abort_token if task_manager else None

    # -- identity --------------------------------------------------------------

    @property
    def task_type(self) -> str:
        return self.config.task_type

    @property
    def task_name(self) -> str:
        return self.config.task_name

    @property
    def display_name(self) -> str:
        return self.config.display_name

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
    def stage(self) -> "Stage":
        return self.microscope._stage

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

    def run(self) -> None:
        self.pre_task()
        self._fire_hook("task_started")
        try:
            self._run()
        except Exception as e:
            cancelled = self._is_cancellation(e)
            try:
                self.grid.task_state.status = (
                    AutoLamellaTaskStatus.Cancelled
                    if cancelled
                    else AutoLamellaTaskStatus.Failed
                )
                self.grid.task_state.status_message = (
                    "Cancelled by user." if cancelled else str(e)
                )
                self._record_outcome()
                # A failure is exactly when someone wants to look; a Stop is
                # not, whoever pressed it already knows.
                if not cancelled:
                    self._settle(failure=str(e))
            except Exception:
                logging.exception(f"Could not record the outcome of {self.task_name}")
            self._fire_hook(
                "task_cancelled" if cancelled else "task_failed", error=str(e)
            )
            raise
        finally:
            self._clear_workflow_metadata()
        self.post_task()
        self._settle()
        self._fire_hook("task_completed")

    def _settle(self, failure: str = "") -> None:
        """Propose this run's result on the grid and decide it; see
        ``proposing.settle``. A grid task asks no question, so there is never
        an inline decision."""
        settle(
            self,
            self.grid,
            proposer=type(self).proposer,
            result_images=self.result_images,
            review=self.review,
            inline_decision=None,
            experiment=self.experiment,
            failure=failure,
        )

    @abstractmethod
    def _run(self) -> None: ...

    def pre_task(self) -> None:
        logging.info(
            f"Running {self.task_name}, {self.task_type} ({self.task_id}) "
            f"for grid {self.grid.name} ({self.grid.id})"
        )
        # One task_state object per record, reused across runs and subscribed to by
        # the UI: reset field by field, never replace it.
        state = self.grid.task_state
        state.name = self.task_name
        state.start_timestamp = datetime.timestamp(datetime.now())
        state.end_timestamp = None
        state.task_id = self.task_id
        state.task_type = self.task_type
        state.status = AutoLamellaTaskStatus.InProgress
        state.status_message = ""
        state.outputs = {}
        self._set_workflow_metadata()
        self.log_status_message("STARTED", "Started")

    def post_task(self) -> None:
        self.grid.task_state.status = AutoLamellaTaskStatus.Completed
        self.grid.task_state.status_message = ""
        self.log_status_message("FINISHED", "Finished")
        self._record_outcome()

    def _record_outcome(self) -> None:
        """Freeze the finished task_state into task_history, on every terminal path."""
        self.grid.task_state.end_timestamp = datetime.timestamp(datetime.now())
        self.grid.task_history.append(deepcopy(self.grid.task_state))

    def _is_cancellation(self, exc: Exception) -> bool:
        if isinstance(exc, (OperationCancelledError, InterruptedError)):
            return True
        return bool(getattr(self.task_manager, "should_abort", False))

    def _check_for_abort(self) -> None:
        """Raise InterruptedError if this task should stop. Call between steps."""
        if self._stop_event is not None and self._stop_event.is_set():
            raise InterruptedError("Workflow aborted by user.")

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

    # -- status, metadata, hooks -----------------------------------------------

    def log_status_message(
        self, message: str, display_message: Optional[str] = None
    ) -> None:
        logging.debug(
            {
                "msg": "status",
                "timestamp": datetime.now().isoformat(),
                "grid": self.grid.name,
                "grid_id": self.grid.id,
                "task_id": self.task_id,
                "task_type": self.task_type,
                "task_name": self.task_name,
                "task_step": message,
            }
        )
        self.grid.task_state.step = message
        self.grid.task_state.status_message = display_message or ""
        record = getattr(self.microscope, "record_event", None)
        if message not in _LIFECYCLE_STEPS and record is not None:
            # STARTED / FINISHED are recorded as task_started / task_completed.
            record(
                "task_step",
                {
                    "step": message,
                    "display_message": display_message,
                    "item_type": "grid",
                    "task_type": self.task_type,
                },
            )
        signal = getattr(self.parent_ui, "step_update_signal", None)
        if message not in _LIFECYCLE_STEPS and signal is not None:
            signal.emit(display_message or message)

    def _set_workflow_metadata(self) -> None:
        """Stamp acquisitions with which grid and task produced them (FIB-466)."""
        ref = getattr(self.microscope, "experiment", None)
        setter = getattr(ref, "set_workflow_metadata", None)
        if setter is not None:
            setter(
                item_id=self.grid.id,
                item_name=self.grid.name,
                task_id=self.task_id,
                task_name=self.task_name,
            )

    def _clear_workflow_metadata(self) -> None:
        ref = getattr(self.microscope, "experiment", None)
        clear = getattr(ref, "clear_workflow_metadata", None)
        if clear is not None:
            clear()

    def _fire_hook(self, event: str, error: Optional[str] = None) -> None:
        from fibsem.hooks import fire_event

        get_run_context = getattr(self.task_manager, "hook_run_context", None)
        run_context = get_run_context() if callable(get_run_context) else {}
        fire_event(
            getattr(self.task_manager, "hook_manager", None),
            event,
            task_name=self.task_name,
            task_type=self.task_type,
            item_name=self.grid.name,
            item_id=self.grid.id,
            task_id=self.task_id,
            task_state=self.grid.task_state,
            error=error,
            **run_context,
        )
