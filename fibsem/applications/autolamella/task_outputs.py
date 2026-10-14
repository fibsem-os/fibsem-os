"""Reading back the files a task run produced.

Runs record what they wrote on their history entry (`AutoLamellaTaskState.outputs`).
This module is the read side of that: it answers "which files does this run's images
consist of", so consumers don't each re-encode the filename convention.

Deliberately free of UI imports — the policy is about paths, not widgets, and keeping
it here lets it be tested without Qt and reused outside the History panel.
"""

from __future__ import annotations

import glob
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskState,
    Experiment,
    GridRecord,
    Lamella,
)

# Characters a folder name cannot hold on Windows (and ``/`` anywhere).
_UNSAFE_IN_FOLDER_NAMES = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def run_folder(lamella: Lamella, task_name: str, started: datetime) -> str:
    """Where one run of ``task_name`` saves what it makes:
    ``<lamella>/<task name>/<YYYYMMDD-HHMMSS>/`` (FIB-1275), so a re-run
    overwrites nothing. The lamella's shared state (``ref_alignment``, the
    coincidence setup reference) is not a run's, and stays in the lamella's
    folder (FIB-1258).

    The task name is made safe for a folder name; a second run starting in the
    same second as one already saved gets ``-2``. Not created here: a run that
    saves nothing leaves no folder. Readers find a run's files from its record
    (``outputs``), not from this name.
    """
    task_folder = _UNSAFE_IN_FOLDER_NAMES.sub("-", task_name).strip().rstrip(".")
    base = os.path.join(
        str(lamella.path), task_folder or "task", f"{started:%Y%m%d-%H%M%S}"
    )
    folder, n = base, 1
    while os.path.exists(folder):
        n += 1
        folder = f"{base}-{n}"
    return folder


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


def final_reference_images(
    lamella: Lamella, *tasks: AutoLamellaTaskState, fallback: bool = True
) -> List[str]:
    """Absolute paths to the final reference images the given runs produced.

    Prefers what the runs recorded. Falls back to the filename convention, which
    remains the only route for experiments written before outputs existed, and for
    runs that failed before reaching post_task and so have no history entry at all.
    ``fallback=False`` for a caller asking about one run of an experiment that
    records outputs: there, a run that recorded no final images made none, and the
    convention would hand it an earlier run's files of the same name.

    Sorted so both routes yield the same order: alphabetical puts the highest-res
    pair last, which is what callers slice off.
    """
    recorded = _recorded(lamella, tasks, "final_sem", "final_fib")
    if recorded or not fallback:
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


def latest_final_reference_images(lamella: Lamella, task_name: str) -> List[str]:
    """The final reference images of the latest run of ``task_name`` that left
    any: the task's images as they stand now.

    Each run of a task keeps its own images (FIB-1275), so the union over every
    run -- what :func:`final_reference_images` gives -- is every set the task
    ever made; a reader showing the task's result wants the newest. A task none
    of whose runs recorded outputs is from before outputs existed, and is found
    by the filename convention, as before.
    """
    runs = [run for run in lamella.task_history if run.name == task_name]
    for run in reversed(runs):
        found = final_reference_images(lamella, run, fallback=False)
        if found:
            return found
    if any(run.outputs for run in runs):
        return []  # its runs record what they make, and none made final images
    return final_reference_images(lamella, AutoLamellaTaskState(name=task_name))


def lamella_images(lamella: Lamella, suffix: str) -> List[str]:
    """The lamella's image files ending in ``suffix``, relative to its folder:
    those in the folder itself (every image of an experiment from before task
    run folders, and the lamella's shared state), then each run's recorded images
    kept in its own folder (FIB-1275), oldest run first.

    What a picker offers: everything that is the lamella's, wherever its run put
    it, without walking the folder tree.
    """
    flat = sorted(
        os.path.basename(path)
        for path in glob.glob(os.path.join(lamella.path, f"*{suffix}"))
    )
    elsewhere: List[str] = []
    for run in lamella.task_history:
        for relpaths in run.outputs.values():
            for relpath in relpaths:
                if (
                    relpath.endswith(suffix)
                    and os.path.dirname(relpath)
                    and relpath not in elsewhere
                    and os.path.isfile(os.path.join(lamella.path, relpath))
                ):
                    elsewhere.append(relpath)
    return flat + elsewhere


def images_by_task(lamella: Lamella) -> List[Tuple[str, List[str], List[str]]]:
    """Each task's final reference images and fluorescence stacks, in the order the
    tasks last ran: ``(task name, final images, fluorescence stacks)``.

    What a workflow animation is made of: a frame per task. One entry per task name,
    sitting where the task last ran, with the latest run's reference images (each
    run keeps its own, FIB-1275). Fluorescence stacks accumulate across runs,
    oldest first. A run that recorded nothing (cancelled, or failed before saving)
    adds nothing, and a task none of whose runs left an image has no entry.
    """
    runs: Dict[str, List[AutoLamellaTaskState]] = {}
    last: Dict[str, int] = {}
    for index, run in enumerate(lamella.task_history):
        runs.setdefault(run.name, []).append(run)
        last[run.name] = index
    steps = []
    for name in sorted(last, key=last.__getitem__):
        finals = latest_final_reference_images(lamella, name)
        stacks = fluorescence_images(lamella, *runs[name])
        if finals or stacks:
            steps.append((name, finals, stacks))
    return steps


def final_images_by_task(lamella: Lamella) -> List[Tuple[str, List[str]]]:
    """Each task's final reference images, in the order the tasks last ran: the
    tasks of ``images_by_task`` that left any."""
    return [(name, finals) for name, finals, _ in images_by_task(lamella) if finals]


def grid_outputs(
    experiment: Experiment,
    grid: GridRecord,
    *roles: str,
    task_name: Optional[str] = None,
) -> List[str]:
    """Absolute, existing, de-duplicated paths a grid's task runs recorded under `roles`.

    The grid-side read model (FIB-876): a results card, the Overview tab and the
    App Server all read the history's `outputs`, and nothing globs the grid's
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
