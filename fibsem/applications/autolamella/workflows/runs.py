"""AutoLamella's runs: what the runner checks before a lamella run starts, the
manager it runs on, and the hooks it fires (see ``fibsem.workflows.runner``)."""

from __future__ import annotations

from typing import Any

import fibsem.config as fibsem_cfg
from fibsem.applications.autolamella.hook_defaults import build_hook_manager
from fibsem.applications.autolamella.workflows.grid_gate import run_refusal
from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
from fibsem.hooks import HookManager
from fibsem.workflows.runner import AppRuns, RunRequest, RunRules, RunStart, refused


def check_lamella_run(
    experiment: Any, microscope: Any, request: RunRequest
) -> RunStart:
    """The tasks and lamellae named, or why not. No lamellae named means every
    lamella. A lamella whose grid is not on the stage is refused by name, with
    no override, before anything moves."""
    if request.options:
        return refused(f"unknown options: {list(request.options)!r}")
    task_names = list(request.task_names)
    known_tasks = [t.name for t in experiment.task_protocol.workflow_config.tasks]
    unknown = [t for t in task_names if t not in known_tasks]
    if not task_names or unknown:
        return refused(
            f"unknown tasks: {unknown!r}" if unknown else "no tasks",
            task_names=known_tasks,
        )
    known_items = [p.name for p in experiment.positions]
    if request.names is not None:
        missing = [n for n in request.names if n not in known_items]
        if missing:
            return refused(f"unknown items: {missing!r}", item_names=known_items)
    names = tuple(request.names) if request.names is not None else tuple(known_items)
    stage = getattr(microscope, "sample_handling", None)
    if stage is not None:
        reason = run_refusal(
            experiment, stage, [p for p in experiment.positions if p.name in names]
        )
        if reason:
            return refused(reason)
    return RunStart(
        started=True,
        request=RunRequest(
            kind=request.kind, task_names=tuple(task_names), names=names
        ),
    )


def make_lamella_manager(
    microscope: Any,
    experiment: Any,
    request: RunRequest,
    parent_ui: Any = None,
    hook_manager: Any = None,
) -> TaskManager:
    return TaskManager(
        microscope=microscope,
        experiment=experiment,
        parent_ui=parent_ui,
        hook_manager=hook_manager,
    )


def lamella_run_hooks() -> HookManager:
    """The hooks for one run.

    Built per run, not once at startup, so a user's saved configuration takes
    effect on the next run without a restart, and the config dialog can never
    be editing a manager a running workflow is holding (FIB-497). The hook set
    comes from the user's saved configuration when there is one, and from
    hook_defaults.default_hooks() when there is not; re-read each time, so
    editing user-preferences.yaml by hand takes effect on the next run.
    """
    manager = build_hook_manager(fibsem_cfg.load_user_preferences().hooks)

    # Deliberately not registered yet. The trigger is proven end to end and the
    # writer is tested, but completion-summary.json is a placeholder for the real
    # artifacts (the PDF and the overview PNG), and turning it on here would start
    # writing a throwaway file into every user's lamella directories. Re-enable when
    # the artifact is the one people actually want -- see FIB-461. It comes back
    # with write_completion_summary from fibsem.applications.autolamella.tools.
    # artifacts, FunctionHook + HookEvent from fibsem.hooks, and the experiment
    # passed in.
    #
    # Per lamella, not per experiment: a lamella is what gets delivered, and an
    # experiment with one abandoned lamella never reaches completion at all, so
    # hanging the artifact off the experiment would mean it was almost never
    # written.
    #
    # A FunctionHook rather than a template-driven one because this needs the
    # experiment itself, which the context deliberately does not carry -- the path
    # is derivable from the record, and in-process hooks can reach the record. A
    # webhook could not, which is the distinction.
    #
    # HookManager.fire contains what this raises, so a summary that cannot be
    # written is logged and the run carries on. FIB-461 asks for exactly that, and
    # it comes for free rather than needing a second guard here.
    #
    # manager.register(
    #     FunctionHook(
    #         name="completion_summary",
    #         events=[HookEvent.ITEM_COMPLETED],
    #         callback=lambda ctx: write_completion_summary(experiment, ctx),
    #     )
    # )
    return manager


RUNS = AppRuns(
    items=RunRules(check=check_lamella_run, make_manager=make_lamella_manager),
    make_hooks=lamella_run_hooks,
)
