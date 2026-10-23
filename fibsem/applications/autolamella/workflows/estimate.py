"""What AutoLamella's steps cost and when its tasks are scheduled: what the
estimate walk (:mod:`fibsem.workflows.estimate`) is handed for this app's runs.

A lamella step costs its own task config's estimate, a grid step what
:func:`fibsem.workflows.grid.estimate.grid_item_seconds` says, and only lamella
tasks have a schedule.
"""

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from fibsem.util.timestamps import now
from fibsem.workflows.estimate import (
    AdditionEstimate,
    RunEstimate,
    StepSeconds,
    estimate_addition,
    estimate_run,
)
from fibsem.workflows.grid.estimate import estimate_grid_run, grid_item_seconds

if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import Experiment
    from fibsem.workflows.queue import TaskQueue


def task_schedule(experiment: "Experiment") -> Dict[str, datetime]:
    """``scheduled_at`` per task name, for the tasks that have one."""
    protocol = experiment.task_protocol
    if protocol is None:
        return {}
    return {
        task.name: task.scheduled_at
        for task in protocol.workflow_config.tasks
        if task.scheduled_at is not None
    }


def step_seconds(
    experiment: "Experiment",
    exchange_seconds: float = 0.0,
    loaded: Sequence[str] = (),
) -> StepSeconds:
    """``(item, task) -> seconds``, or None for a step with nothing to offer: a
    lamella's from its own task config, a grid's from the grid protocol and the
    loader's exchange. The timeline and the queue key steps by name, so one
    lookup serves either kind of run.

    A step that cannot be priced is None rather than an error: ``estimated_time``
    divides by the sputter rate (milling/base.py), so a hand-edited protocol can
    raise, and losing a figure is a great deal better than losing the dialog or
    the slot (PyQt aborts on an exception in a slot) that shows it.
    """
    lamella_by_name = {lam.name: lam for lam in experiment.positions}
    grid_seconds = None
    if experiment.task_protocol is not None:
        grid_seconds = grid_item_seconds(experiment, exchange_seconds, loaded)

    def seconds_for(item_name: str, task_name: str) -> Optional[float]:
        try:
            lamella = lamella_by_name.get(item_name)
            if lamella is not None:
                config = lamella.task_config.get(task_name)
                return config.estimated_duration if config is not None else None
            if grid_seconds is not None:
                return grid_seconds(item_name, task_name)
            return None
        except Exception:
            logging.warning(
                "Could not estimate duration for %s on %s.",
                task_name,
                item_name,
                exc_info=True,
            )
            return None

    return seconds_for


def estimate_lamella_run(experiment: "Experiment", pairs: Sequence) -> RunEstimate:
    """What a lamella run over *pairs* (``(lamella, task)``, as its queue lays
    them out) takes: each step from its lamella's own task config, holding for
    each task's schedule."""
    return estimate_run(pairs, step_seconds(experiment), task_schedule(experiment))


def _active_elapsed(experiment: "Experiment", queue: "TaskQueue") -> Optional[float]:
    """Wall-clock since the running task started, which is what the recorded
    duration measures too; None when nothing runs."""
    active = queue.active
    if active is None:
        return None
    item = next(
        (lam for lam in experiment.positions if lam.name == active.item_name), None
    )
    if item is None:
        item = experiment.get_grid_by_name(active.item_name)
    started = item.task_state.start_timestamp if item is not None else None
    if started is None:
        return None
    return max(0.0, (now() - started).total_seconds())


def estimate_queue_addition(
    experiment: "Experiment",
    queue: "TaskQueue",
    pairs: Sequence,
    run_next: bool,
    seconds: StepSeconds,
) -> AdditionEstimate:
    """What adding ``pairs`` would cost the running ``queue``.

    Over a snapshot of the live queue: ``queue.items`` hands out copies, so the
    hypothetical is built without touching what is running. The running task's
    elapsed counts any supervised wait, so the absolute finish can run a little
    optimistic, but both walks share it, so the *delay* is unaffected.
    """
    return estimate_addition(
        queue.items,
        pairs,
        lambda item: seconds(item.item_name, item.task_name),
        run_next=run_next,
        schedule=task_schedule(experiment),
        active_elapsed=_active_elapsed(experiment, queue),
    )


# -- the grids on the stage: what a grid run's exchanges cost ------------------
# Read from the stage's last inventory read (``grid_inventory``): no hardware.


def exchange_seconds(stage: Any) -> float:
    """What one exchange costs on *stage*'s loader; nothing on a fixed holder,
    or with no sample handling."""
    if stage is None or stage.magazine is None:
        return 0.0
    return stage.magazine.exchange_seconds


def loaded_grid_names(stage: Any) -> List[str]:
    """The grids on the stage now, which a load step does not exchange."""
    if stage is None:
        return []
    try:
        return [e.name for e in stage.grid_inventory() if e.name and e.loaded]
    except Exception as e:  # noqa: BLE001 - priced as nothing loaded
        logging.warning(f"Could not read the grid inventory: {e}")
        return []


def exchanges_for(stage: Any, grid_names: Sequence[str]) -> int:
    """How many of *grid_names* are not on the stage now: the exchanges a run
    over them makes on a loader; zero on a fixed holder."""
    if stage is None or stage.magazine is None:
        return 0
    loaded = set(loaded_grid_names(stage))
    return sum(1 for name in grid_names if name not in loaded)


def run_step_seconds(experiment: "Experiment", stage: Any) -> StepSeconds:
    """:func:`step_seconds` with *stage*'s exchange and the grids on it."""
    return step_seconds(experiment, exchange_seconds(stage), loaded_grid_names(stage))


def grid_run_estimate(
    experiment: "Experiment",
    stage: Any,
    task_names: List[str],
    grid_names: List[str],
    exchanges: Optional[int] = None,
) -> Optional[RunEstimate]:
    """What a grid run of *task_names* over *grid_names* takes, with the
    exchanges *stage* makes for it unless *exchanges* says (a scan that has
    not named its grids yet); None when it cannot say, since a figure is not
    worth losing the confirmation over."""
    if exchanges is None:
        exchanges = exchanges_for(stage, grid_names)
    try:
        return estimate_grid_run(
            experiment, task_names, grid_names, exchanges, exchange_seconds(stage)
        )
    except Exception:
        logging.warning("Could not estimate the grid run.", exc_info=True)
        return None
