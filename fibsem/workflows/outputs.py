"""Reading back the files an item's task runs recorded.

Runs record what they wrote on their history entry (``TaskState.outputs``),
relative to the item's directory (``BaseExperiment.item_path``), so a copied
experiment still resolves. These read that record for any item, a lamella or
a grid; nothing globs an item's directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

from fibsem.workflows.status import TaskState

if TYPE_CHECKING:
    from fibsem.workflows.experiment import BaseExperiment


def recorded_outputs(root: Any, runs: Iterable[TaskState], *roles: str) -> List[str]:
    """Absolute, existing, de-duplicated paths the given runs recorded under
    ``roles``, relative to ``root``.

    Any number of runs of a task give the union of what they produced. That one
    rule covers both shapes without branching on task or file type: a file
    written to the same name each run collapses to one path, and uniquely named
    files each count. Both guards are load-bearing, because a record can hold
    things a directory listing never could:

    * **Deduplicate** -- the same path recorded twice is still one file, whether
      one run acquired a set twice or two runs overwrote each other. Duplicates
      crowd out real images when callers slice off the last N.
    * **Require existence** -- a record can name a file since deleted, and a
      path to it would be a placeholder that never fills.
    """
    paths = (
        os.path.join(root, relpath)
        for run in runs
        for role in roles
        for relpath in run.outputs.get(role, [])
    )
    return sorted({path for path in paths if os.path.isfile(path)})


def item_outputs(
    experiment: "BaseExperiment",
    item: Any,
    *roles: str,
    task_name: Optional[str] = None,
) -> List[str]:
    """The files an item's runs recorded under ``roles``: every run's by
    default, or ``task_name``'s only."""
    runs = [
        run for run in item.task_history if task_name is None or run.name == task_name
    ]
    return recorded_outputs(experiment.item_path(item), runs, *roles)


def latest_item_output(
    experiment: "BaseExperiment", item: Any, role: str
) -> Optional[str]:
    """The most recently recorded existing file under ``role``, or None. What a
    card shows."""
    root = experiment.item_path(item)
    for run in reversed(item.task_history):
        for relpath in reversed(run.outputs.get(role, [])):
            path = os.path.join(root, relpath)
            if os.path.isfile(path):
                return path
    return None


@dataclass
class RunFiles:
    """One run's existing recorded files by role, as basenames."""

    name: str
    status: str
    files: Dict[str, List[str]] = field(default_factory=dict)


def files_by_run(root: Any, runs: Iterable[TaskState]) -> List[RunFiles]:
    """Each run in order, with its existing recorded files by role, as
    basenames: which run produced which file."""
    entries = []
    for run in runs:
        files = {}
        for role, relpaths in dict(run.outputs).items():
            names = [
                os.path.basename(rp)
                for rp in relpaths
                if os.path.isfile(os.path.join(root, rp))
            ]
            if names:
                files[role] = names
        entries.append(RunFiles(run.name, run.status.name, files))
    return entries


def all_item_outputs(experiment: "BaseExperiment", item: Any) -> List[str]:
    """Every existing file an item's runs recorded, under any role."""
    roles = {role for run in item.task_history for role in run.outputs}
    return item_outputs(experiment, item, *sorted(roles))


def item_output_named(
    experiment: "BaseExperiment", item: Any, filename: str
) -> Optional[str]:
    """The file an item's runs recorded with basename *filename*, or None."""
    return next(
        (
            path
            for path in all_item_outputs(experiment, item)
            if os.path.basename(path) == filename
        ),
        None,
    )


def saved_file(experiment: "BaseExperiment", folder: str, path: str) -> Optional[str]:
    """The file at *path* under *folder*, saved outside any task (an image the
    agent took); None when there is none, or *path* leads out of the
    experiment."""
    root = os.path.realpath(experiment.path)
    target = os.path.realpath(os.path.join(folder, path))
    if os.path.commonpath([root, target]) != root or not os.path.isfile(target):
        return None
    return target
