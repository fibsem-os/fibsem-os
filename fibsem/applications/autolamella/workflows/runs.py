"""AutoLamella's runs: what the runner checks before a lamella run starts or its
work joins a running queue, the manager it runs on, and the hooks it fires (see
``fibsem.workflows.runner``).

Whether a lamella run can start: every linked lamella's grid must be on the stage.

A lamella marked on a grid's stored overview belongs to that grid whether or not
the grid is on the stage (FIB-71). Running it while the grid is in the magazine
would drive the stage to that pose on whatever grid *is* there, so a run that
includes one is refused, by name, before anything moves. There is no override:
load the grid from the Grids tab, or leave that lamella out.

The inventory is *read* first (`SampleHandling.get_inventory`: instant, nothing moves) so
the answer is the hardware's, not a stale chip's. A lamella that is not linked to
a grid, or whose grid has no record in this experiment, is never refused: the
rule is about a known grid being somewhere else, not about missing bookkeeping.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Iterable, List

import fibsem.config as fibsem_cfg
from fibsem.applications.autolamella.hook_defaults import build_hook_manager
from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
from fibsem.hooks import HookManager
from fibsem.workflows.runner import AppRuns, RunRequest, RunRules, RunStart, refused

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from fibsem.applications.autolamella.structures import Experiment, Lamella
    from fibsem.services.sample_handling import SampleHandling

logger = logging.getLogger(__name__)


def lamellae_off_the_stage(
    experiment: "Experiment", stage: "SampleHandling", lamellae: Iterable["Lamella"]
) -> Dict[str, List[str]]:
    """Grid name -> the names of the given lamellae on it that cannot be run,
    because that grid is not on the stage. Empty when the run is clear."""
    try:
        stage.get_inventory()
    except Exception as e:  # noqa: BLE001 - answer from the last read instead
        logger.warning(f"Could not read the grid inventory before the run: {e}")
    try:
        loaded = {g.name for g in stage.loaded_grids}
    except Exception as e:  # noqa: BLE001 - nothing known: refuse nothing
        logger.warning(f"Could not read which grids are loaded: {e}")
        return {}

    off: Dict[str, List[str]] = {}
    for lamella in lamellae:
        grid = experiment.get_grid_for_lamella(lamella)
        if grid is None or grid.name in loaded:
            continue
        off.setdefault(grid.name, []).append(lamella.name)
    return off


def run_refusal(
    experiment: "Experiment", stage: "SampleHandling", lamellae: Iterable["Lamella"]
) -> str:
    """Why this run cannot start, naming the lamellae and their grid, or ""."""
    off = lamellae_off_the_stage(experiment, stage, lamellae)
    if not off:
        return ""
    lines = []
    for grid_name, names in off.items():
        lines.append(
            f"{', '.join(names)} {'is' if len(names) == 1 else 'are'} on "
            f"{grid_name}, which is not on the stage."
        )
    grids = list(off)
    which = grids[0] if len(grids) == 1 else "the grid"
    return (
        "Cannot run: "
        + " ".join(lines)
        + f" Load {which} from the Grids tab first, or leave "
        + (
            "that lamella"
            if sum(len(n) for n in off.values()) == 1
            else "those lamellae"
        )
        + " out."
    )


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


def give_lamellae_their_configs(experiment: Any, request: RunRequest) -> None:
    """A lamella made before a task joined the protocol has no config for it,
    and ``run_task`` raises on that: such lamellae take the base protocol's
    before their work is queued."""
    task_names = list(request.task_names)
    missing = [
        lamella.name
        for lamella in experiment.positions
        if lamella.name in request.names
        and any(t not in lamella.task_config for t in task_names)
    ]
    if missing:
        experiment.apply_lamella_config(missing, task_names)


def make_lamella_manager(
    microscope: Any,
    experiment: Any,
    request: RunRequest,
    hook_manager: Any = None,
    responder: Any = None,
    deciders: Any = None,
    images: Any = None,
) -> TaskManager:
    return TaskManager(
        microscope=microscope,
        experiment=experiment,
        hook_manager=hook_manager,
        responder=responder,
        deciders=deciders,
        images=images,
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
    items=RunRules(
        check=check_lamella_run,
        make_manager=make_lamella_manager,
        prepare=give_lamellae_their_configs,
    ),
    make_hooks=lamella_run_hooks,
)
