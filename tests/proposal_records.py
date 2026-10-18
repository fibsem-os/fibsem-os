"""Putting proposals on an item as a given task's, for tests that build the
record by hand."""

from typing import Any, List

from fibsem.acting import Actor


def set_proposals(item: Any, task_name: str, proposals: List[Any]) -> None:
    """Make *proposals* everything *task_name* asked about *item*, in place of
    what it had asked."""
    for proposal in proposals:
        proposal.asked_by = Actor.task(task_name)
    kept = [p for p in item.proposals if not p.asked_by_task(task_name)]
    item.proposals[:] = kept + list(proposals)
