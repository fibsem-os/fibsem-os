"""A prior FM->FIB transform from a previous correlation run (FIB-956).

The rotation and scale a correlation fits are properties of the instrument and
its mount, not of the lamella: on eight saved METEOR runs the fitted rotations
agree to about half a degree, while the transform derived from the hardware
geometry sits five degrees from all of them. So the best prior for a new run
is the *fitted* transform of a previous one -- this lamella's own if it has
one, else any other lamella's on the same system -- with the geometry as the
fallback for a first-ever run. Measured on those runs: with the fitted prior
one placed pair puts the remaining predictions within 0.3 um of the picks.

Only rotation and scale are taken. The translation is the stage offset of the
run it came from and means nothing for another; it comes from the placed
pairs. The scale is rescaled for the FIB pixel size of the current image (the
FM pixel size is the same camera and objective, and a saved run does not
record it).
"""

from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from fibsem.correlation.geometry import NominalTransform
from fibsem.correlation.history import CorrelationRun, LamellaCorrelation
from fibsem.correlation.structures import CorrelationResult

__all__ = [
    "PriorTransform",
    "experiment_runs",
    "prior_from_runs",
    "transform_from_result",
]


@dataclass(frozen=True)
class PriorTransform:
    transform: NominalTransform
    source: str  # for the status line: which run it came from


def transform_from_result(
    result: CorrelationResult,
    *,
    fm_pixel_size: float,
    fm_pixel_size_z: float,
    fib_pixel_size: float,
    translation: Optional[np.ndarray] = None,
) -> Optional[NominalTransform]:
    """The fitted transform of a saved result, in the current images' units.

    A result fitted before FIB-881 (``fm_z_scale`` 1.0 on an anisotropic stack)
    has its depth column in raw slices; it is brought to isotropic units.
    Returns None when the result holds no usable rotation.
    """
    R = np.asarray(result.rotation_quaternion, dtype=float)
    if R.shape != (3, 3) or not result.scale or not np.all(np.isfinite(R)):
        return None
    scale = float(result.scale)
    prior_fib_px = None
    if result.input_data is not None:
        prior_fib_px = result.input_data.fib_image_pixel_size
    if prior_fib_px and fib_pixel_size:
        scale *= float(prior_fib_px) / float(fib_pixel_size)
    P = scale * R[:2, :]
    zan = fm_pixel_size_z / fm_pixel_size if fm_pixel_size and fm_pixel_size_z else 1.0
    if result.fm_z_scale == 1.0 and abs(zan - 1.0) > 1e-6:
        # Fitted on raw slices: the depth column is per slice, not per xy
        # pixel. Rescaling it reproduces that fit's predictions exactly; the
        # rows are then not orthonormal, and are left so -- re-orthonormalising
        # would rotate the in-plane map away from what fitted the data. The
        # seed built from ``rotation`` completes its own proper rotation.
        P = P.copy()
        P[:, 2] /= zan
    t = np.asarray(translation, dtype=float) if translation is not None else np.zeros(2)
    return NominalTransform(
        projection=P,
        translation=t,
        scale=scale,
        fm_pixel_size=float(fm_pixel_size),
        fm_pixel_size_z=float(fm_pixel_size_z),
        fib_pixel_size=float(fib_pixel_size),
    )


def prior_from_runs(
    runs: Sequence[Tuple[str, CorrelationRun]],
    *,
    fm_pixel_size: float,
    fm_pixel_size_z: float,
    fib_pixel_size: float,
    translation: Optional[np.ndarray] = None,
) -> Optional[PriorTransform]:
    """The first run, in the given order, that holds a fitted transform.

    ``runs`` are ``(label, run)`` pairs, most specific first (this lamella's
    newest, then its older ones, then other lamellae's) -- see
    :func:`experiment_runs`.
    """
    for label, run in runs:
        result = run.state.result
        if result is None:
            continue
        try:
            transform = transform_from_result(
                result,
                fm_pixel_size=fm_pixel_size,
                fm_pixel_size_z=fm_pixel_size_z,
                fib_pixel_size=fib_pixel_size,
                translation=translation,
            )
        except Exception as exc:  # a prior is an aid; never block on one
            logging.debug(f"prior from {label} unusable: {exc}")
            continue
        if transform is not None:
            return PriorTransform(transform=transform, source=label)
    return None


@dataclass(frozen=True)
class PlacementOffset:
    """A previous run's placement offset: microns in the FIB frame, and its source."""

    offset_um: np.ndarray  # (2,)
    source: str  # the run's label
    age_days: float  # since that run was fitted


# A run whose fiducials disagree by more than this cannot have measured the
# offset (two poor Arctis fits on one lamella disagreed by 18 um in y).
MAX_RMS_UM_FOR_OFFSET = 2.0


def placement_offset_from_runs(
    runs: Sequence[Tuple[str, CorrelationRun]],
    *,
    now: Optional[float] = None,
    max_rms_um: float = MAX_RMS_UM_FOR_OFFSET,
) -> Optional[PlacementOffset]:
    """The first run, in the given order, that recorded a usable placement offset.

    Skips runs without one (written before FIB-979) and runs whose fit was
    too poor to have measured it. Same order as :func:`prior_from_runs`.
    """
    import time

    for label, run in runs:
        result = run.state.result
        offset = getattr(result, "placement_offset", None) if result else None
        if not offset or len(offset) != 2:
            continue
        fib_px = getattr(result.input_data, "fib_image_pixel_size", None)
        rms_um = result.rms_error * fib_px * 1e6 if fib_px else None
        if rms_um is None or rms_um > max_rms_um:
            logging.debug(f"placement offset from {label} skipped: rms {rms_um} um")
            continue
        age = max(0.0, ((now if now is not None else time.time()) - result.updated_at))
        return PlacementOffset(
            offset_um=np.asarray(offset, dtype=float),
            source=label,
            age_days=age / 86400.0,
        )
    return None


def experiment_runs(
    experiment_dir: str, lamella_dir: Optional[str] = None
) -> List[Tuple[str, CorrelationRun]]:
    """Every correlation run in an experiment, most useful as a prior first.

    This lamella's runs newest first, then every other lamella's newest first.
    Labels read ``"this lamella, run <name>"`` / ``"<lamella>, run <name>"``.
    """
    ordered: List[Tuple[str, CorrelationRun]] = []
    own = os.path.abspath(lamella_dir) if lamella_dir else None
    if own:
        for run in reversed(
            LamellaCorrelation.discover(os.path.join(own, "Correlation")).runs
        ):
            ordered.append((f"this lamella, run {run.name}", run))
    for folder in sorted(glob.glob(os.path.join(experiment_dir, "*"))):
        if not os.path.isdir(folder) or (own and os.path.abspath(folder) == own):
            continue
        for run in reversed(
            LamellaCorrelation.discover(os.path.join(folder, "Correlation")).runs
        ):
            ordered.append((f"{os.path.basename(folder)}, run {run.name}", run))
    return ordered
