"""The volume run: each position cuts and images its slices, one slice at a time."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

import pandas as pd

from fibsem.applications.volume.position import VolumePosition
from fibsem.applications.volume.tasks import run_volume_task
from fibsem.workflows.manager import BaseTaskManager
from fibsem.workflows.queue import WorkItem
from fibsem.workflows.status import TaskStatus

if TYPE_CHECKING:
    from fibsem.applications.volume.experiment import VolumeExperiment

SKIP_POSITION_NOT_FOUND = "position_not_found"
SKIP_TASK_REMOVED = "task_removed"
SKIP_MISSING_PREREQS = "missing_prereqs"


class VolumeTaskManager(BaseTaskManager):
    """Runs the volume tasks on each position once per slice.

    The queue holds one slice per position at a time; the next is queued when it
    drains. A run starts each position at its first slice not done, with only the
    tasks that slice has not completed, so a Stop between milling and imaging
    resumes at imaging and no slice is cut twice. A slice that ends with a task
    not completed stops that position for the run: the next slice would be cut
    on a face nobody imaged, or unaligned.
    """

    experiment: "VolumeExperiment"
    ITEM_NOUN = "positions"
    RUN_DESCRIPTION = "a volume run"
    CANCELLED_LINE = "Volume run cancelled by user."
    # The tasks each slice runs, in order: set by ``run``.
    _task_names: Tuple[str, ...] = ()

    def run(self, task_names: List[str], names: Optional[List[str]] = None) -> None:
        """Run ``task_names``, per slice, on each of ``names`` (every position
        if None)."""
        self._task_names = tuple(task_names)
        if names is None:
            names = [p.name for p in self.experiment.positions]
        for name in names:
            following = self._next_iteration(name, None)
            if following is None:
                continue
            iteration, tasks = following
            for task_name in tasks:
                self.queue.add(name, task_name, iteration=iteration)
        self._run_queue()

    def _next_iteration(
        self, item_name: str, finished: Optional[int]
    ) -> Optional[Tuple[int, List[str]]]:
        position = self.experiment.get_position_by_name(item_name)
        if position is None:
            return None
        if finished is not None and not position.iteration_completed(
            finished, self._task_names
        ):
            return None
        for iteration in range(position.iterations):
            remaining = self._remaining(position, iteration)
            if remaining:
                return iteration, remaining
        return None

    def _remaining(self, position: VolumePosition, iteration: int) -> List[str]:
        """The run's tasks slice ``iteration`` has not completed, in run order."""
        return [
            name
            for name in self._task_names
            if not position.iteration_completed(iteration, [name])
        ]

    def _record_for(self, item: WorkItem) -> Optional[VolumePosition]:
        position = self.experiment.get_position_by_name(item.item_name)
        if position is None:
            self._skip(
                item,
                item.item_name,
                f"Skipping {item.task_name}: no position named {item.item_name}.",
                SKIP_POSITION_NOT_FOUND,
            )
        return position

    def _ready(self, item: WorkItem, position: VolumePosition) -> bool:
        protocol = self.experiment.task_protocol
        if item.task_name not in protocol.task_config:
            self._skip(
                item,
                position.name,
                f"Skipping {item.task_name} on {position.name}: it is no longer in "
                "the volume protocol.",
                SKIP_TASK_REMOVED,
                item_id=position.id,
            )
            return False
        missing = [
            name
            for name in protocol.requirements(item.task_name)
            if not position.iteration_completed(item.iteration, [name])
        ]
        if missing:
            self._skip(
                item,
                position.name,
                f"Skipping {item.task_name} on {position.name} slice "
                f"{item.iteration}: {', '.join(missing)} did not complete.",
                SKIP_MISSING_PREREQS,
                item_id=position.id,
            )
            return False
        return True

    def _skip(
        self, item: WorkItem, name: str, msg: str, reason: str, item_id: str = ""
    ) -> None:
        logging.warning(msg)
        self.queue.mark_done(item, TaskStatus.Skipped)
        self._emit_report(
            item=item,
            item_name=name,
            status=TaskStatus.Skipped,
            msg=msg,
            skip_reason=reason,
        )
        self._fire_skipped_hook(item.task_name, name, reason, item_id=item_id)

    def _run_task(self, task_name: str, position: VolumePosition) -> None:
        run_volume_task(
            self.microscope,
            task_name,
            self.experiment,
            position,
            task_manager=self,
        )

    def build_run_summary_dataframe(self) -> pd.DataFrame:
        """One row per (position, slice, task) this run queued."""
        return pd.DataFrame(
            [
                {
                    "position_name": i.item_name,
                    "iteration": i.iteration,
                    "task_name": i.task_name,
                    "task_status": i.status.name,
                }
                for i in self.queue.items
            ]
        )

    def _completion_message(self) -> str:
        items = self.queue.items
        if not items:
            return "No slices to cut."
        positions = list(dict.fromkeys(i.item_name for i in items))
        ran = {(i.item_name, i.iteration) for i in items}
        unfinished = {
            (i.item_name, i.iteration)
            for i in items
            if i.status is not TaskStatus.Completed
        }
        slices = ran - unfinished
        failed = sum(1 for i in items if i.status is TaskStatus.Failed)
        parts = [f"{len(slices)} slices on {len(positions)} positions"]
        if failed:
            parts.append(f"{failed} task{'s' if failed != 1 else ''} failed")
        return "Volume run complete: " + ", ".join(parts) + "."
