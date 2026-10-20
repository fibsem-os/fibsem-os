"""What may be edited while a workflow runs: everything but what the run is
working on now. The window and the App Server both ask here, so they refuse
the same edits with the same words."""

from __future__ import annotations

from typing import Any, List


def lamella_edit_refusal(runner: Any, experiment: Any, lamella: Any, what: str) -> str:
    """Why *what* on *lamella* must wait, or "": the run is on it, or on its
    grid."""
    grid = experiment.get_grid_for_lamella(lamella)
    names = [lamella.name] + ([grid.name] if grid is not None else [])
    return runner.busy_refusal(names, what)


def grid_edit_refusal(
    runner: Any, experiment: Any, grid_name: str, what: str, renaming: bool = False
) -> str:
    """Why *what* on grid *grid_name* must wait, or "": the run is on it or on
    one of its lamellae. A rename also waits while the grid is queued, since
    the queue knows it by name and would drop it."""
    grid = experiment.get_grid_by_name(grid_name)
    names = [grid_name]
    if grid is not None:
        names += [p.name for p in experiment.get_lamellae_for_grid(grid)]
    why = runner.busy_refusal(names, what)
    if why or not renaming:
        return why
    manager = runner.manager
    if manager is not None and any(
        w.item_name == grid_name for w in manager.queue.pending
    ):
        return f"{grid_name} is queued in the run; {what} when the run has finished with it."
    return ""


def busy_lamellae(runner: Any, experiment: Any) -> List[str]:
    """The lamellae an edit to every lamella must leave alone: those the run
    is on, by name."""
    return [
        p.name
        for p in experiment.positions
        if lamella_edit_refusal(runner, experiment, p, "edit it")
    ]
