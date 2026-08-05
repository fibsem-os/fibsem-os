"""Find experiments, and narrow them down to a question worth asking.

The facility-manager question is *"what did this microscope do this week?"*
Answering it needs two things: something to group by, and a way to find the
experiments. The first arrived with the session record (FIB-451); this is the
second (FIB-457).

Deliberately reads only, and never imports :class:`Experiment`. Enumerating a
directory means opening files written by other versions, half-finished runs, and
whatever else is under the root -- loading each one properly would be slow and
would let a single bad file take out the listing. ``peek_experiment`` reads the
few keys that matter and flags what it could not parse.

**What this can and cannot see.** There is no global registry of experiments: the
default root is inside the installed package, users create experiments anywhere,
and the recents list holds the last ten a machine opened. So an answer here is
complete for the roots it was given and silent about everything else. That gap
closes when the user-data directory lands (FIB-336) and the index can live
somewhere an upgrade does not destroy.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Iterable, List, Optional

from fibsem.config import ExperimentSummary, get_recent_experiments, peek_experiment

__all__ = [
    "discover_experiments",
    "filter_experiments",
    "group_by_instrument",
]

# How deep to look for an experiment.yaml below the root. An experiment directory
# sits directly under the root it was created in, so 2 covers the normal layout
# and one level of grouping (a per-project or per-user folder). Unbounded descent
# over a whole data drive is slow enough to look like a hang.
DEFAULT_MAX_DEPTH = 3


def discover_experiments(
    root: str, max_depth: int = DEFAULT_MAX_DEPTH
) -> List[ExperimentSummary]:
    """Every experiment under ``root``, newest first.

    An experiment is a directory containing ``experiment.yaml``. Descends at most
    ``max_depth`` levels; a directory that is itself an experiment is not
    descended into, so a nested run folder cannot be reported twice.

    A root that does not exist returns nothing rather than raising: pointing this
    at a drive that is not mounted is an ordinary mistake, and an empty answer
    with a warning is more useful than a traceback.
    """
    if not os.path.isdir(root):
        logging.warning("Not a directory, so nothing to enumerate: %s", root)
        return []

    found: List[ExperimentSummary] = []
    root = os.path.abspath(root)

    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath[len(root):].count(os.sep)
        if "experiment.yaml" in filenames:
            found.append(peek_experiment(os.path.join(dirpath, "experiment.yaml")))
            # Whatever is inside an experiment belongs to it. Pruning here also
            # keeps the walk off the image directories, which is most of the
            # files on disk.
            dirnames[:] = []
            continue
        if depth >= max_depth:
            dirnames[:] = []

    return sorted(found, key=lambda e: e.created_at, reverse=True)


def filter_experiments(
    experiments: Iterable[ExperimentSummary],
    instrument: Optional[str] = None,
    operator: Optional[str] = None,
    since: Optional[float] = None,
    until: Optional[float] = None,
) -> List[ExperimentSummary]:
    """Narrow a listing. Every argument left as None does not filter.

    ``instrument`` matches the serial number or the model, case-insensitively --
    a facility manager knows their instrument by whichever of those they use, and
    making them guess which one this wants is not worth a second flag.

    An experiment with no session record matches no instrument and no operator.
    It is not evidence of the instrument you asked about, and quietly including
    unknowns is how a per-instrument total comes out wrong.
    """
    results = list(experiments)

    if instrument is not None:
        needle = instrument.casefold()
        results = [
            e for e in results
            if needle in (e.instrument_serial or "").casefold()
            or needle in (e.instrument_model or "").casefold()
        ]

    if operator is not None:
        needle = operator.casefold()
        results = [e for e in results if needle in (e.operator or "").casefold()]

    if since is not None:
        results = [e for e in results if e.created_at >= since]

    if until is not None:
        results = [e for e in results if e.created_at <= until]

    return results


def group_by_instrument(
    experiments: Iterable[ExperimentSummary],
) -> Dict[Optional[str], List[ExperimentSummary]]:
    """Bucket by instrument serial, keeping the order given.

    Serial rather than model, because it is what distinguishes two of the same
    model in one facility -- the case the grouping exists for. Experiments with
    no session record collect under ``None``, which is a bucket worth seeing:
    it is how many runs cannot be attributed at all.
    """
    grouped: Dict[Optional[str], List[ExperimentSummary]] = {}
    for experiment in experiments:
        grouped.setdefault(experiment.instrument_serial, []).append(experiment)
    return grouped


def recent_experiments() -> List[ExperimentSummary]:
    """The experiments this machine has opened, newest first.

    A second source alongside a directory walk, and the only one that reaches
    experiments created outside any root the caller knows to look in. Capped at
    ten by the preference that stores it, and stored inside the installed
    package, so it is neither complete nor durable -- see FIB-336.
    """
    return get_recent_experiments(prune_missing=False)
