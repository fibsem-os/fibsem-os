"""A lamella's correlation history — a read-only view over the on-disk run folders.

Each open of the correlation dialog writes a run to
``<lamella>/Correlation/<timestamp>/correlation.json`` (FIB-264); the app has
always created these folders and thrown away the pointer. This reconstructs the
history from them so a new correlation can seed from the previous one (FIB-299).
Nothing new is persisted — the history *is* the folders.
"""
from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

from fibsem.correlation.structures import CorrelationState, load_correlation_file

# Preference order within a run folder: the consolidated file, then either legacy.
_RUN_FILES = ("correlation.json", "correlation_result.json", "correlation_data.json")


def _run_file(folder: str) -> Optional[str]:
    for name in _RUN_FILES:
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    return None


@dataclass
class CorrelationRun:
    """One saved correlation, loaded from its timestamped folder."""

    path: str          # the run folder
    name: str          # folder basename (the timestamp)
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
