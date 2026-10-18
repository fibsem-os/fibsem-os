"""A grid run, for every app that has grids: what the runner checks before one
starts, and the manager it runs on (see ``fibsem.workflows.runner``)."""

from __future__ import annotations

from typing import Any

from fibsem.workflows.grid.manager import GridTaskManager, plan_grid_run
from fibsem.workflows.runner import RunRequest, RunRules, RunStart, refused

# What a grid request may carry beyond its tasks and grids.
GRID_OPTIONS = ("inventory_first", "unload_at_end", "report_at_end")


def check_grid_run(experiment: Any, microscope: Any, request: RunRequest) -> RunStart:
    """The grid tasks and grids named, or why not.

    ``inventory_first`` is "Screen all grids": the run inventories and runs
    over every present grid, so it names none. Otherwise no grids named means
    every grid the experiment records.
    """
    unknown_options = [o for o in request.options if o not in GRID_OPTIONS]
    if unknown_options:
        return refused(f"unknown options: {unknown_options!r}")
    task_names = list(request.task_names)
    known_tasks = list(experiment.grid_protocol.ordered_task_names)
    unknown = [t for t in task_names if t not in known_tasks]
    if not task_names or unknown:
        return refused(
            f"unknown grid tasks: {unknown!r}" if unknown else "no tasks",
            task_names=known_tasks,
        )
    known_grids = [g.name for g in experiment.grids]
    screen_all = request.option("inventory_first")
    names = request.names
    if screen_all:
        if names is not None:
            return refused(
                "screen_all runs over every present grid; do not name grids with it"
            )
    elif names is None:
        names = tuple(known_grids)
    else:
        missing = [n for n in names if n not in known_grids]
        if missing:
            return refused(f"unknown grids: {missing!r}", grid_names=known_grids)
    if not screen_all and not names:
        return refused("no grids", grid_names=known_grids)
    details: dict = {"screen_all": screen_all}
    if not screen_all:
        details["plan"] = [
            {"grid": grid, "step": step}
            for grid, step in plan_grid_run(task_names, list(names))
        ]
    return RunStart(
        started=True,
        details=details,
        request=RunRequest(
            kind=request.kind,
            task_names=tuple(task_names),
            names=names,
            options=dict(request.options),
        ),
    )


def make_grid_manager(
    microscope: Any,
    experiment: Any,
    request: RunRequest,
    parent_ui: Any = None,
    hook_manager: Any = None,
    responder: Any = None,
    answerers: Any = None,
) -> GridTaskManager:
    return GridTaskManager(
        microscope=microscope,
        experiment=experiment,
        parent_ui=parent_ui,
        hook_manager=hook_manager,
        responder=responder,
        answerers=answerers,
        unload_at_end=request.option("unload_at_end"),
        report_at_end=request.option("report_at_end"),
        inventory_first=request.option("inventory_first"),
    )


GRID_RUNS = RunRules(check=check_grid_run, make_manager=make_grid_manager)
