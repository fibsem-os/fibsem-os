"""Edits to a grid's record that the Grids tab and the App Server both make:
a note, the lamellae on it and a removal, each saved. Renaming is :mod:`fibsem.workflows.grid.naming`'s;
a verdict is saved by ``Experiment.save_verdict``."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

__all__ = ["assign_lamellae_to_grid", "keep_grid_note", "remove_grid_record"]


def keep_grid_note(
    experiment, grid, note: str, evidence: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Keep *note* on *grid*'s record, apart from its verdict; an empty note
    clears it. Saved. The note record for the session's record, or None for
    a clearing; the caller records it, as whoever wrote it."""
    kept = grid.set_note(note, **({"evidence": evidence} if evidence else {}))
    experiment.save()
    return kept


def remove_grid_record(experiment, grid) -> List[str]:
    """Stop tracking *grid*: its files stay, and its lamellae stay with no
    grid. Saved. The names of the lamellae it leaves without one."""
    orphaned = [p.name for p in experiment.get_lamellae_for_grid(grid)]
    experiment.remove_grid(grid.name)
    experiment.save()
    return orphaned


def assign_lamellae_to_grid(experiment, grid, lamellae) -> List[str]:
    """Record *lamellae* as on *grid*, the grid record every grid view and a
    run's on-the-stage check read. Saved. The names of those that were not on
    it before."""
    moved = [p.name for p in lamellae if p.grid_id != grid.id]
    for lamella in lamellae:
        lamella.grid_id = grid.id
    experiment.save()
    return moved
