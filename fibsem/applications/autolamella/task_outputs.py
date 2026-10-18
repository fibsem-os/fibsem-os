"""Reading back the files a task run produced.

Runs record what they wrote on their history entry (`TaskState.outputs`), read
back for any item by `fibsem.workflows.outputs`. This module is a lamella's side
of that: it answers "which files does this run's images consist of", so
consumers don't each re-encode the filename convention.

Deliberately free of UI imports — the policy is about paths, not widgets, and keeping
it here lets it be tested without Qt and reused outside the History panel.
"""

from __future__ import annotations

import glob
import os
from typing import Dict, List, Tuple

from fibsem.applications.autolamella.structures import Lamella
from fibsem.workflows.outputs import recorded_outputs
from fibsem.workflows.status import TaskState


def fluorescence_images(lamella: Lamella, *tasks: TaskState) -> List[str]:
    """Absolute paths to the fluorescence z-stacks the given runs produced.

    No filename fallback, unlike the reference images: fluorescence output never
    followed a discoverable convention — it is named for the lamella and a
    time-of-day stamp, with no task name — so an experiment written before runs
    recorded their outputs simply has none to find. Guessing a pattern would
    attribute files to the wrong run.
    """
    return recorded_outputs(lamella.path, tasks, "fluorescence")


def final_reference_images(
    lamella: Lamella, *tasks: TaskState, fallback: bool = True
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
    recorded = recorded_outputs(lamella.path, tasks, "final_sem", "final_fib")
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
    return final_reference_images(lamella, TaskState(name=task_name))


def latest_task_image(lamella: Lamella, suffix: str) -> str:
    """The last completed task's most zoomed final image ending in ``suffix``,
    from its latest run, relative to the lamella folder; empty when it left
    none. The run's record says which, wherever it saved them (FIB-1275)."""
    task = lamella.last_completed_task
    if task is None:
        return ""
    finals = [
        f
        for f in latest_final_reference_images(lamella, task.name)
        if f.endswith(suffix)
    ]
    return os.path.relpath(finals[-1], lamella.path) if finals else ""


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
    runs: Dict[str, List[TaskState]] = {}
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
