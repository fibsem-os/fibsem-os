"""Reading back the files a task run produced.

Runs record what they wrote on their history entry (`AutoLamellaTaskState.outputs`).
This module is the read side of that: it answers "which files does this run's images
consist of", so consumers don't each re-encode the filename convention.

Deliberately free of UI imports — the policy is about paths, not widgets, and keeping
it here lets it be tested without Qt and reused outside the review panel.
"""

from __future__ import annotations

import glob
import os
from typing import List, Optional, Sequence

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskState,
    Experiment,
    GridRecord,
    Lamella,
)


def _recorded(
    lamella: Lamella, tasks: Sequence[AutoLamellaTaskState], *roles: str
) -> List[str]:
    """Absolute, existing, de-duplicated paths recorded across the given runs.

    Every function here accepts any number of runs of the same task and returns the
    union of what they produced. That single rule covers both shapes without
    branching on task or file type: reference images are written to the same
    filename each run, so repeated runs contribute the same paths and the set
    collapses; fluorescence stacks are uniquely named, so each run contributes its
    own. The panel therefore shows one reference set and every FM acquisition.

    Both guards are load-bearing, because a record can hold things a directory
    listing never could:

    * **Deduplicate** — the same path recorded twice is still one file, whether
      that came from one run acquiring a set twice (MillCoincidentTask does, before
      and after milling) or from two runs overwriting each other. Duplicates crowd
      out real images when callers slice off the last N.
    * **Require existence** — a record can name a file that has since been deleted.
      Returning it produces a row of placeholders that never fill, where the file
      simply being absent used to mean no row at all.
    """
    paths = (
        os.path.join(lamella.path, relpath)
        for task in tasks
        for role in roles
        for relpath in task.outputs.get(role, [])
    )
    return sorted({path for path in paths if os.path.isfile(path)})


def fluorescence_images(lamella: Lamella, *tasks: AutoLamellaTaskState) -> List[str]:
    """Absolute paths to the fluorescence z-stacks the given runs produced.

    No filename fallback, unlike the reference images: fluorescence output never
    followed a discoverable convention — it is named for the lamella and a
    time-of-day stamp, with no task name — so an experiment written before runs
    recorded their outputs simply has none to find. Guessing a pattern would
    attribute files to the wrong run.
    """
    return _recorded(lamella, tasks, "fluorescence")


def final_reference_images(lamella: Lamella, *tasks: AutoLamellaTaskState) -> List[str]:
    """Absolute paths to the final reference images the given runs produced.

    Prefers what the runs recorded. Falls back to the filename convention, which
    remains the only route for experiments written before outputs existed, and for
    runs that failed before reaching post_task and so have no history entry at all.

    Sorted so both routes yield the same order: alphabetical puts the highest-res
    pair last, which is what callers slice off.
    """
    recorded = _recorded(lamella, tasks, "final_sem", "final_fib")
    if recorded:
        return recorded
    return sorted(
        {
            path
            for name in {task.name for task in tasks}
            for path in glob.glob(
                os.path.join(lamella.path, f"ref_{name}*_final_*res*.tif*")
            )
        }
    )


def grid_outputs(
    experiment: Experiment,
    grid: GridRecord,
    *roles: str,
    task_name: Optional[str] = None,
) -> List[str]:
    """Absolute, existing, de-duplicated paths a grid's task runs recorded under `roles`.

    The grid-side read model (FIB-876): a results card, the Overview tab and the
    agent server all read the history's `outputs`, and nothing globs the grid's
    directory. Paths are recorded relative to `experiment.grid_path(grid)`, so a
    copied experiment still resolves. Pass `task_name` to read one task's runs
    only; by default every run on the grid contributes.
    """
    root = experiment.grid_path(grid)
    runs = [t for t in grid.task_history if task_name is None or t.name == task_name]
    paths = (
        os.path.join(root, relpath)
        for task in runs
        for role in roles
        for relpath in task.outputs.get(role, [])
    )
    return sorted({path for path in paths if os.path.isfile(path)})


def latest_grid_output(
    experiment: Experiment, grid: GridRecord, role: str
) -> Optional[str]:
    """The most recently recorded file under `role`, or None. What a card shows."""
    for task in reversed(grid.task_history):
        for relpath in reversed(task.outputs.get(role, [])):
            path = os.path.join(experiment.grid_path(grid), relpath)
            if os.path.isfile(path):
                return path
    return None
