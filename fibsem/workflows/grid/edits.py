"""Edits to a grid's record that the Grids tab and the App Server both make:
a note and a removal, each saved. Renaming is :mod:`fibsem.workflows.grid.naming`'s;
a verdict is saved by ``Experiment.save_verdict``."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

__all__ = ["keep_grid_note", "remove_grid_record"]


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
