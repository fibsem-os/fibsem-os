"""Artifacts written automatically when a run finishes.

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

if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import Experiment
    from fibsem.hooks import HookContext

COMPLETION_SUMMARY_FILENAME = "completion-summary.json"


def write_completion_summary(
    experiment: "Experiment", context: "HookContext"
) -> Optional[str]:
    """Write a small json recording that this experiment finished, and when.

    A placeholder for the real artifacts (the PDF and the overview PNG) while the
    trigger is proven end to end -- the point being that it is written by whatever
    finishes the run, not by a person clicking a dialog.

    Self-describing on purpose, even at four keys: the file is meant to be forwarded,
    and an artifact that cannot say which experiment it came from is a stack of bytes
    three months later. The id is the stable key; the name can change.

    Timestamped from the event, not from now(): the file records when the run
    finished, which is not necessarily when someone got round to writing it.
    """
    summary = {
        "experiment_id": context.experiment_id or experiment.id,
        "experiment_name": context.experiment_name or experiment.name,
        "completed_at": datetime.fromtimestamp(context.timestamp).isoformat(
            timespec="seconds"
        ),
        # Which event produced this. There will be more triggers -- per-item artifacts,
        # and arguably cancelled runs, which is when someone most wants to look.
        "event": context.event,
    }

    path = os.path.join(experiment.path, COMPLETION_SUMMARY_FILENAME)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

    logging.info(f"Wrote completion summary for {experiment.name} to {path}")
    return path
