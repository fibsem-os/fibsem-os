"""Artifacts written automatically when work finishes.

The downstream recipient -- the TEM operator, the collaborator -- never installs
fibsemOS, so what lands in the experiment directory is their only interface. Today
that artifact exists only if someone remembers to open a Qt dialog before going home.
See FIB-461.

Deliberately light on imports: these run from a hook, on the workflow's thread, so
this module must not pull in matplotlib or reportlab the way tools/reporting.py does.
"""

import json
import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus

if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import (
        AutoLamellaTaskState,
        Experiment,
        Lamella,
    )
    from fibsem.hooks import HookContext

COMPLETION_SUMMARY_FILENAME = "completion-summary.json"


def write_completion_summary(
    experiment: "Experiment", context: "HookContext"
) -> Optional[str]:
    """Write a small json into a finished lamella's directory recording that it is done.

    A placeholder for the real artifacts (the PDF and the overview PNG) while the
    trigger is proven end to end -- the point being that it is written by whatever
    finishes the work, not by a person clicking a dialog.

    Hung off ITEM_COMPLETED rather than the end of the experiment, because a lamella is
    the unit that gets delivered to a TEM, and because most experiments never complete:
    a lamella or two gets abandoned, and everything else is still worth reporting.

    Self-describing on purpose, even at a handful of keys: the file is meant to be
    forwarded, and an artifact that cannot say what it came from is a stack of bytes
    three months later. The ids are the stable keys; the names can change.

    Timestamped from the event, not from now(): the file records when the work
    finished, which is not necessarily when someone got round to writing it.
    """
    lamella = _find_item(experiment, context)
    if lamella is None:
        logging.warning(
            f"No lamella matching {context.item_name!r} in experiment "
            f"{experiment.name}; no completion summary written."
        )
        return None

    summary = {
        "experiment_id": context.experiment_id or experiment.id,
        "experiment_name": context.experiment_name or experiment.name,
        "item_id": lamella.id,
        "item_name": lamella.name,
        "completed_at": datetime.fromtimestamp(context.timestamp).isoformat(
            timespec="seconds"
        ),
        # Which event produced this. There will be more triggers -- an end-of-run
        # summary covering the lamellae that did not finish, which is when someone most
        # wants to look.
        "event": context.event,
        # Filtered on status, so a task that failed and was not retried does not appear
        # here as work delivered.
        "tasks_completed": [
            _task_record(task)
            for task in lamella.task_history
            if task.status is AutoLamellaTaskStatus.Completed
        ],
    }

    os.makedirs(lamella.path, exist_ok=True)
    path = os.path.join(lamella.path, COMPLETION_SUMMARY_FILENAME)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

    logging.info(f"Wrote completion summary for {lamella.name} to {path}")
    return path


def _task_record(task: "AutoLamellaTaskState") -> dict:
    """What one completed task contributed, and what it left behind.

    ``outputs`` is the point of this: paths are stored relative to lamella.path, and
    the artifact is written into that same directory, so they resolve as-is for
    whoever receives the folder. Reported per task rather than merged into one map,
    which would lose provenance and collide on shared role names (two tasks both
    writing final_sem).

    Named to match AutoLamellaTaskState.outputs -- the artifact and the record it
    describes should not need a glossary between them.
    """
    return {
        "name": task.name,
        # Identifies this execution, so a retried task is distinguishable from its
        # earlier attempt, and this joins to the hook event that announced it.
        "task_id": task.task_id,
        "completed_at": (
            datetime.fromtimestamp(task.end_timestamp).isoformat(timespec="seconds")
            if task.end_timestamp is not None
            else None
        ),
        "duration_s": round(task.duration, 1),
        "outputs": task.outputs,
    }


def _find_item(experiment: "Experiment", context: "HookContext") -> Optional["Lamella"]:
    """Resolve the item a context names back to the lamella it refers to.

    By id first: the name is a petname a user can edit, the id is the stable key. The
    context carries both precisely so a consumer does not have to trust the name.
    """
    if context.item_id:
        for lamella in experiment.positions:
            if lamella.id == context.item_id:
                return lamella
    for lamella in experiment.positions:
        if lamella.name == context.item_name:
            return lamella
    return None


HANDOFF_MAP_SUFFIX = "-handoff-map.pdf"


def write_handoff_map(
    experiment: "Experiment", context: "HookContext"
) -> Optional[str]:
    """Write the handoff map for the whole experiment when a run finishes.

    The artifact `write_completion_summary` was standing in for. Written into the
    experiment directory rather than a lamella's, because unlike the per-lamella summary
    this covers the *grid*: where everything is, including the lamellae that did not
    finish, which is the case people actually hit and the one they most want to look at.

    Hung off the end of a run rather than off ITEM_COMPLETED for the same reason, plus a
    practical one: the document re-renders every map and every lamella image, so
    regenerating it each time one lamella finished would repeat that work once per
    lamella for a file only the last version of which is read.

    Grid and slot come from whatever the operator last typed into the export dialog,
    which is kept on `Experiment.metadata`. Absent on a run where nobody opened it --
    and absent is right, because guessing a cassette slot into a document that travels
    with the sample would be worse than leaving it blank.

    Overwrites in place. There is one current answer to "what is on this grid", and a
    directory of dated near-duplicates makes the recipient choose between them.

    Raising is contained by `HookManager.fire`, so a map that cannot be written is
    logged and the run carries on.
    """
    from fibsem.applications.autolamella.tools.handoff_map import (
        HandoffOptions,
        generate_handoff_map,
    )

    output = os.path.join(
        str(experiment.path), f"{experiment.name}{HANDOFF_MAP_SUFFIX}"
    )
    options = HandoffOptions(
        grid=experiment.metadata.get("grid", ""),
        slot=experiment.metadata.get("slot", ""),
        note=experiment.metadata.get("handoff_note", ""),
    )
    generate_handoff_map(experiment, output, options)
    logging.info(f"Wrote the handoff map for {experiment.name} to {output}")
    return output
