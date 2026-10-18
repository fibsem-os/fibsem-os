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
from typing import Any, Dict, List, Optional

from fibsem.constants import DATETIME_FILE
from fibsem.correlation.structures import (
    CorrelationResult,
    CorrelationState,
    load_correlation_file,
)

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


def correlation_record(
    result: CorrelationResult, lamella: Any, run_folder: Optional[str], root: Any
) -> Dict[str, Any]:
    """An accepted correlation, for the experiment's record (FIB-1068): what it
    gave, how well it fits, and the run folder that holds the rest. Whoever
    moves an item's point of interest to a run's answer records this."""
    data = result.input_data
    pixel_size = data.fib_image_pixel_size if data is not None else None
    folder = run_folder
    if run_folder and root:
        try:
            folder = os.path.relpath(run_folder, str(root))
        except ValueError:  # another drive
            pass
    ri = result.refractive_index_correction_mode
    return {
        "item": {"id": lamella.id, "name": lamella.name},
        "poi": result.poi[0].px_m.to_dict(),
        "rms_px": result.rms_error,
        "rms_nm": result.rms_error * pixel_size * 1e9 if pixel_size else None,
        "fiducials": len(result.delta_2d),
        "refractive_index": (
            {"mode": ri, "factor": result.refractive_index_correction_factor}
            if ri
            else None
        ),
        "verdict": _verdict_tier(result.diagnostics),
        "seeded": result.seed is not None,
        "folder": folder,
    }


def _verdict_tier(diagnostics: Optional[dict]) -> Optional[str]:
    """The fit verdict's tier, as the dialog showed it; None for an unseeded fit."""
    if not diagnostics:
        return None
    from fibsem.correlation.verdict import FitDiagnostics, verdict

    try:
        return verdict(FitDiagnostics.from_dict(diagnostics)).tier
    except Exception:  # noqa: BLE001 - the tier is a detail of the record
        logging.debug("could not read the fit verdict", exc_info=True)
        return None
