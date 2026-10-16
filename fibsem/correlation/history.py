"""A lamella's correlation history — the on-disk run folders.

Each correlation that records a result writes a run to
``<lamella>/Correlation/<timestamp>/correlation.json`` (FIB-264). This names and
writes those folders (:func:`new_run_dir`, :func:`save_run`) and reconstructs the
history from them so a new correlation can seed from the previous one (FIB-299).
Nothing else is persisted — the history *is* the folders.
"""

from __future__ import annotations

import datetime
import glob
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

from fibsem.constants import DATETIME_FILE
from fibsem.correlation.structures import CorrelationState, load_correlation_file

CORRELATION_DIRNAME = "Correlation"
CORRELATION_FILENAME = "correlation.json"

# Preference order within a run folder: the consolidated file, then either legacy.
_RUN_FILES = (CORRELATION_FILENAME, "correlation_result.json", "correlation_data.json")


def correlation_dir(lamella_path: str) -> str:
    """Where a lamella's correlation runs live: ``<lamella>/Correlation``."""
    return os.path.join(lamella_path, CORRELATION_DIRNAME)


def new_run_dir(lamella_path: str, now: Optional[datetime.datetime] = None) -> str:
    """A new run folder for a lamella, named for ``now`` but NOT created.

    It is created only when the run records something (:func:`save_run`):
    creating it on open made every open a run, and an abandoned session became
    the next open's starting coordinates (FIB-320).
    """
    now = now or datetime.datetime.now()
    return os.path.join(correlation_dir(lamella_path), now.strftime(DATETIME_FILE))


def save_run(run_dir: str, state: CorrelationState) -> str:
    """Write ``state`` as the run's ``correlation.json``, creating the folder.

    Returns the file written.
    """
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, CORRELATION_FILENAME)
    state.save(path)
    return path


def _run_file(folder: str) -> Optional[str]:
    for name in _RUN_FILES:
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    return None


def load_run(run_dir: str) -> CorrelationState:
    """The correlation saved in a run folder, whichever file it was written as."""
    path = _run_file(run_dir)
    if path is None:
        raise FileNotFoundError(f"no correlation is saved in {run_dir}")
    return load_correlation_file(path)


@dataclass
class CorrelationRun:
    """One saved correlation, loaded from its timestamped folder."""

    path: str  # the run folder
    name: str  # folder basename (the timestamp)
    state: CorrelationState


@dataclass
class LamellaCorrelation:
    """The runs found under a lamella's ``Correlation`` directory, oldest first."""

    runs: List[CorrelationRun] = field(default_factory=list)

    @staticmethod
    def discover(correlation_dir: str) -> "LamellaCorrelation":
        """Load every run folder under ``correlation_dir`` (``<lamella>/Correlation``).

        Folders are ordered by name — the timestamp format is lexicographically
        chronological. Unreadable folders are logged and skipped so one bad run
        doesn't hide the rest.
        """
        runs: List[CorrelationRun] = []
        if not os.path.isdir(correlation_dir):
            return LamellaCorrelation()
        for folder in sorted(glob.glob(os.path.join(correlation_dir, "*"))):
            if not os.path.isdir(folder):
                continue
            path = _run_file(folder)
            if path is None:
                continue
            try:
                state = load_correlation_file(path)
            except Exception:
                logging.warning("Skipping unreadable correlation run: %s", path)
                continue
            runs.append(
                CorrelationRun(path=folder, name=os.path.basename(folder), state=state)
            )
        return LamellaCorrelation(runs=runs)

    def latest(self) -> Optional[CorrelationRun]:
        """The most recent run, or None if there are none."""
        return self.runs[-1] if self.runs else None
