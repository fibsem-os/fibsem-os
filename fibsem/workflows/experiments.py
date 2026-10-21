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
from datetime import datetime
from typing import Iterable, List, Optional

from fibsem import config as fibsem_cfg
from fibsem.config import ExperimentSummary, load_user_preferences, peek_experiment
from fibsem.util.timestamps import from_posix, to_datetime

__all__ = [
    "default_experiment_directory",
    "discover_experiments",
    "created_order",
    "filter_experiments",
]

# How deep to look for an experiment.yaml below the root. An experiment directory
# sits directly under the root it was created in, so 2 covers the normal layout
# and one level of grouping (a per-project or per-user folder). Unbounded descent
# over a whole data drive is slow enough to look like a hang.
DEFAULT_MAX_DEPTH = 3


def default_experiment_directory() -> str:
    """Where a new experiment goes: the preference when it names a folder that
    exists, else the AutoLamella log folder (the Create dialog's default)."""
    preferred = load_user_preferences().experiment.default_experiment_directory
    if preferred and os.path.isdir(preferred):
        return str(preferred)
    if preferred:
        logging.warning(
            f"Preference default_experiment_directory '{preferred}' does not exist; using default."
        )
    return str(fibsem_cfg.AUTOLAMELLA_LOG_PATH)


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
        depth = dirpath[len(root) :].count(os.sep)
        if "experiment.yaml" in filenames:
            found.append(peek_experiment(os.path.join(dirpath, "experiment.yaml")))
            # Whatever is inside an experiment belongs to it. Pruning here also
            # keeps the walk off the image directories, which is most of the
            # files on disk.
            dirnames[:] = []
            continue
        if depth >= max_depth:
            dirnames[:] = []

    return sorted(found, key=created_order, reverse=True)


def created_order(summary: ExperimentSummary):
    """Sort key: when it was created, with an experiment that does not say first."""
    return summary.created_at or from_posix(0.0)


def filter_experiments(
    experiments: Iterable[ExperimentSummary],
    instrument: Optional[str] = None,
    operator: Optional[str] = None,
    since=None,
    until=None,
) -> List[ExperimentSummary]:
    """Narrow a listing. Every argument left as None does not filter.

    ``instrument`` matches the serial number or the model, case-insensitively --
    a facility manager knows their instrument by whichever of those they use, and
    making them guess which one this wants is not worth a second flag.

    An experiment with no session record matches no instrument and no operator.
    It is not evidence of the instrument you asked about, and quietly including
    unknowns is how a per-instrument total comes out wrong.

    ``since`` and ``until`` are a datetime or a POSIX time; a naive datetime is
    this machine's time. An experiment that does not say when it was created
    counts as the oldest.
    """
    results = list(experiments)

    if instrument is not None:
        needle = instrument.casefold()
        results = [
            e
            for e in results
            if needle in (e.instrument_serial or "").casefold()
            or needle in (e.instrument_model or "").casefold()
        ]

    if operator is not None:
        needle = operator.casefold()
        results = [e for e in results if needle in (e.operator or "").casefold()]

    if since is not None:
        start = _aware(to_datetime(since))
        results = [e for e in results if _aware(created_order(e)) >= start]

    if until is not None:
        end = _aware(to_datetime(until))
        results = [e for e in results if _aware(created_order(e)) <= end]

    return results


def _aware(moment: datetime) -> datetime:
    """*moment*, as this machine's time when it carries no offset."""
    return moment.astimezone() if moment.tzinfo is None else moment
