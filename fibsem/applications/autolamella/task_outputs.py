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


def _recorded(lamella: Lamella, task: AutoLamellaTaskState, *roles: str) -> List[str]:
    """Absolute, existing, de-duplicated paths recorded under the given roles.

    A record can hold things a directory listing never could, so both guards are
    load-bearing rather than defensive habit:

    * **Deduplicate** — a run can acquire the same set twice (MillCoincidentTask
      does, before and after milling) and overwrite the same files, recording each
      path twice. Duplicates crowd out real images when callers slice off the last N.
    * **Require existence** — a record can name a file that has since been deleted.
      Returning it produces a row of placeholders that never fill, where the file
      simply being absent used to mean no row at all.
    """
    paths = (
        os.path.join(lamella.path, relpath)
        for role in roles
        for relpath in task.outputs.get(role, [])
    )
    return sorted({path for path in paths if os.path.isfile(path)})


def fluorescence_images(lamella: Lamella, task: AutoLamellaTaskState) -> List[str]:
    """Absolute paths to the fluorescence z-stacks a task run produced.

    No filename fallback, unlike the reference images: fluorescence output never
    followed a discoverable convention — it is named for the lamella and a
    time-of-day stamp, with no task name — so an experiment written before runs
    recorded their outputs simply has none to find. Guessing a pattern would
    attribute files to the wrong run.
    """
    return _recorded(lamella, task, "fluorescence")


def final_reference_images(lamella: Lamella, task: AutoLamellaTaskState) -> List[str]:
    """Absolute paths to the final reference images a task run produced.

    Prefers what the run recorded. Falls back to the filename convention, which
    remains the only route for experiments written before outputs existed, and for
    runs that failed before reaching post_task and so have no history entry at all.

    Sorted so both routes yield the same order: alphabetical puts the highest-res
    pair last, which is what callers slice off.
    """
    recorded = _recorded(lamella, task, "final_sem", "final_fib")
    if recorded:
        return recorded
    return sorted(
        glob.glob(os.path.join(lamella.path, f"ref_{task.name}*_final_*res*.tif*"))
    )
