"""The volume app's runs: what the runner checks before a volume run starts, and
the manager it runs on (see ``fibsem.workflows.runner``)."""

from __future__ import annotations

from typing import Any

from fibsem.applications.volume.manager import VolumeTaskManager
from fibsem.hooks import HookManager
from fibsem.workflows.runner import (
    AppRuns,
    RunRequest,
    RunRules,
    RunStart,
    not_edited,
    refused,
)


def check_volume_run(experiment: Any, microscope: Any, request: RunRequest) -> RunStart:
    """The tasks and positions named, or why not. No positions named means every
    position."""
    if request.options:
        return refused(f"unknown options: {list(request.options)!r}")
    task_names = list(request.task_names)
    known_tasks = experiment.task_protocol.ordered_task_names
    unknown = [t for t in task_names if t not in known_tasks]
    if not task_names or unknown:
        return refused(
            f"unknown volume tasks: {unknown!r}" if unknown else "no tasks",
            task_names=known_tasks,
        )
    known = [p.name for p in experiment.positions]
    if request.names is not None:
        missing = [n for n in request.names if n not in known]
        if missing:
            return refused(f"unknown positions: {missing!r}", item_names=known)
    names = tuple(request.names) if request.names is not None else tuple(known)
    if not names:
        return refused("no positions", item_names=known)
    return RunStart(
        started=True,
        request=RunRequest(
            kind=request.kind, task_names=tuple(task_names), names=names
        ),
    )


def plan_volume_addition(queue: Any, request: RunRequest, run_next: bool):
    """The run queues each position's slices itself, one at a time."""
    return not_edited(
        "A volume run queues its slices itself; start another run when it ends.",
        "refused",
    )


def make_volume_manager(
    microscope: Any,
    experiment: Any,
    request: RunRequest,
    hook_manager: Any = None,
    responder: Any = None,
    deciders: Any = None,
    images: Any = None,
) -> VolumeTaskManager:
    return VolumeTaskManager(
        microscope=microscope,
        experiment=experiment,
        hook_manager=hook_manager,
        responder=responder,
        deciders=deciders,
        images=images,
    )


RUNS = AppRuns(
    items=RunRules(
        check=check_volume_run,
        make_manager=make_volume_manager,
        plan_addition=plan_volume_addition,
    ),
    # No volume hooks yet: a run fires its events to whoever the runner adds.
    make_hooks=HookManager,
)
