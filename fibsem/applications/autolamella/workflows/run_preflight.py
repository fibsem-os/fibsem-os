"""Preflight checks + confirmation summary for a lamella workflow run.

Qt-free so it stays unit-testable and keeps the run dialog thin: the UI just
renders ``RunPreflight.blocked`` / ``RunPreflight.note``.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Sequence

from fibsem.applications.autolamella.workflows.tasks.scheduling import (
    Plan,
    plan_for_run,
)

if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import Experiment
    from fibsem.microscope import FibsemMicroscope

# internal skip reasons -> human-readable text for the confirmation
_SKIP_REASONS = {
    "failure": "lamella failed",
    "missing_prereqs": "prerequisite not in this run",
    "no_lamella": "unknown lamella",
}
_MAX_SKIPS_SHOWN = 6


@dataclass
class RunPreflight:
    """Outcome of preflighting a run.

    ``blocked`` set → show it as a warning and abort. Otherwise ``note`` (if any)
    is the confirmation caption summarising exchanges and plan-time skips.
    """

    plan: Plan
    blocked: Optional[str] = None
    note: Optional[str] = None


def build_run_preflight(
    experiment: "Experiment",
    microscope: "FibsemMicroscope",
    task_names: Sequence[str],
    lamella_names: Sequence[str],
) -> RunPreflight:
    """Check a lamella run can proceed and build its confirmation note.

    On a static holder (no loader) grids that aren't loaded cannot be reached, so
    the run is blocked with a "load them manually" message. Otherwise the note
    summarises the grid-exchange cost and any work skipped at plan time.
    """
    plan = plan_for_run(experiment, microscope, task_names, lamella_names)

    stage = getattr(microscope, "_stage", None)
    if getattr(stage, "loader", None) is None:
        missing = _unreachable_grid_names(experiment, microscope, lamella_names)
        if missing:
            return RunPreflight(plan, blocked=(
                "These grids aren't in the holder and there's no autoloader to "
                f"load them: {', '.join(missing)}.\n\n"
                "Load them manually (Grids tab) or deselect their lamellae, then "
                "run again."))

    notes = [n for n in (_exchange_note(plan), _skip_note(plan)) if n]
    return RunPreflight(plan, note="\n\n".join(notes) or None)


def _unreachable_grid_names(experiment, microscope, lamella_names) -> List[str]:
    """Distinct grid names that the selected lamellae need but aren't loaded."""
    loaded = {g._id for g in experiment.get_loaded_grids(microscope)}
    names = set()
    for name in lamella_names:
        lamella = experiment.get_lamella_by_name(name)
        grid = experiment.get_grid_for_lamella(lamella) if lamella else None
        if grid is not None and grid._id not in loaded:
            names.add(grid.name)
    return sorted(names)


def _exchange_note(plan: Plan) -> Optional[str]:
    if plan.n_exchanges <= 0:
        return None
    grids = sorted(g for g in plan.items_per_grid if g is not None)
    return (
        f"Spans {len(grids)} grids ({', '.join(grids)}) — up to {plan.n_exchanges} "
        "grid exchange(s). Post-exchange realignment isn't implemented yet, so "
        "reaching positions on reloaded grids relies on loader/stage repeatability."
    )


def _skip_note(plan: Plan) -> Optional[str]:
    if not plan.skipped:
        return None
    shown = "; ".join(f"{task} on {lam} ({_SKIP_REASONS.get(reason, reason)})"
                      for lam, task, reason in plan.skipped[:_MAX_SKIPS_SHOWN])
    if len(plan.skipped) > _MAX_SKIPS_SHOWN:
        shown += f"; +{len(plan.skipped) - _MAX_SKIPS_SHOWN} more"
    return f"{len(plan.skipped)} item(s) will be skipped: {shown}."
