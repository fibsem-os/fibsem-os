"""What a grid run's steps cost: a task by its config, a load by the loader's
exchange. Walked by :mod:`fibsem.workflows.estimate` like any queue."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, List, Optional, Sequence

from fibsem.workflows.estimate import RunEstimate, StepSeconds, estimate_run
from fibsem.workflows.grid.manager import (
    LOAD_ENTRY_NAME,
    UNLOAD_ENTRY_NAME,
    plan_grid_run,
)

if TYPE_CHECKING:
    from fibsem.workflows.grid.protocol import GridExperiment


def grid_item_seconds(
    experiment: "GridExperiment", exchange_seconds: float, loaded: Sequence[str] = ()
) -> StepSeconds:
    """Per-step estimates for a grid queue, by ``(grid, step)``: what the timeline
    and Add to queue price a grid item with.

    A "Load grid" step costs ``exchange_seconds``, unless its grid is one of
    ``loaded`` (on the stage now), when the load is a no-op. The "Unload grid"
    step a run can end with costs ``exchange_seconds`` too: the loader gives
    one figure, and estimates round up. A task costs its
    config's estimate. None for anything else: a step this protocol does not have
    has nothing to offer, as on the lamella side.
    """
    grids = {grid.name for grid in experiment.grids}
    configs = experiment.grid_protocol.task_config
    on_stage = set(loaded)

    def seconds_for(grid_name: str, step: str) -> Optional[float]:
        if grid_name not in grids:
            return None
        if step == LOAD_ENTRY_NAME:
            return 0.0 if grid_name in on_stage else exchange_seconds
        if step == UNLOAD_ENTRY_NAME:
            return exchange_seconds
        config = configs.get(step)
        return config.estimated_duration if config is not None else None

    return seconds_for


def estimate_grid_run(
    experiment: "GridExperiment",
    task_names: List[str],
    grid_names: List[str],
    exchanges: int,
    exchange_seconds: float,
    now: Optional[datetime] = None,
) -> RunEstimate:
    """Estimate a grid run of ``task_names`` over ``grid_names``, as
    ``plan_grid_run`` lays it out: each grid loaded, then its tasks.

    A row for the exchanges when there are any, then a row per task with its
    grid count. ``exchanges`` is how many of the grids are not loaded now, and
    ``exchange_seconds`` what the loader says one costs; a fixed holder passes
    zero for either and prices none. A load of a grid already on the stage is
    no exchange and is left out, and so is a task the protocol does not have.
    Nothing is scheduled on a grid task.
    """
    configs = experiment.grid_protocol.task_config
    tasks = [name for name in task_names if configs.get(name) is not None]
    # Which grids need the exchange does not change the price, only how many.
    exchanged = set(grid_names[:exchanges]) if exchange_seconds > 0 else set()
    pairs = [
        (grid, step)
        for grid, step in plan_grid_run(tasks, grid_names)
        if step != LOAD_ENTRY_NAME or grid in exchanged
    ]
    return estimate_run(pairs, grid_item_seconds(experiment, exchange_seconds), now=now)
