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
from typing import List

from fibsem.applications.autolamella.structures import AutoLamellaTaskState, Lamella


def final_reference_images(lamella: Lamella, task: AutoLamellaTaskState) -> List[str]:
    """Absolute paths to the final reference images a task run produced.

    Prefers what the run recorded. Falls back to the filename convention, which
    remains the only route for experiments written before outputs existed, and for
    runs that failed before reaching post_task and so have no history entry at all.

    Sorted so both routes yield the same order: alphabetical puts the highest-res
    pair last, which is what callers slice off.
    """
    recorded = [
        os.path.join(lamella.path, relpath)
        for role in ("final_sem", "final_fib")
        for relpath in task.outputs.get(role, [])
    ]
    if recorded:
        return sorted(recorded)
    return sorted(
        glob.glob(os.path.join(lamella.path, f"ref_{task.name}*_final_*res*.tif*"))
    )
