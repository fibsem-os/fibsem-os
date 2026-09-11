"""Predicted FM fiducials: FIB fiducials through the nominal transform (FIB-956).

Finding a milled fiducial in an FM stack is the slow part of a correlation: a
spot burn is a few pixels in a 1024-pixel image, and there is nothing to say
where to look. But the FM->FIB map is mostly known before any fiducial is
picked -- its rotation and scale come from the instrument geometry
(:mod:`fibsem.correlation.geometry`), and the only thing the geometry does not
know is where the two fields of view sit relative to each other, a translation
that one confirmed pair supplies. So the FIB fiducials are projected into the
FM as *predictions*, the user drags a few onto the burns they are already
next to, and the rest are re-projected through the translation those drops
give. On the saved METEOR runs one pair puts the remaining predictions within
a few pixels of their burns.

Nothing here feeds a fit on its own. A prediction has ``status`` predicted or
suggested until the user moves it, and only positions the user has placed (or
a search has found) count as pairs. That is the line between a measurement
and a guess, and it is what stops the projection from grading its own
homework: the rotation the final fit recovers is checked against the geometry
by the user's drops, not confirmed by points the geometry itself placed.

Pure functions over :class:`Coordinate` lists, no Qt: the tab widget calls
them, and so can a review renderer or a script.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from fibsem.correlation.geometry import NominalTransform
from fibsem.correlation.structures import (
    Coordinate,
    PointProvenance,
    PointStatus,
    PointType,
    PointXYZ,
)

__all__ = [
    "MIN_PAIRS_FOR_ROTATION_REFIT",
    "Projection",
    "build_projection",
    "excluded_indices",
    "independent_pairs",
    "place_predictions",
    "predictions_for",
    "suggest_indices",
    "usable_pairs",
]

# Below this many independent pairs the re-projection moves the translation
# only and keeps the geometry's rotation: a rotation fitted to fewer points
# than it has degrees of freedom is the noise of those points, not a
# correction to the prior.
MIN_PAIRS_FOR_ROTATION_REFIT = 5


@dataclass(frozen=True)
class Projection:
    """The FM->FIB map as best known now: ``fib_xy = matrix @ fm_iso + translation``.

    ``fm_iso`` is (x, y, z) in isotropic FM pixels (z scaled by ``z_anisotropy``).
    ``z_iso`` is the depth predictions are placed at; ``note`` says what the map
    was built from, for the status line.
    """

    matrix: np.ndarray  # (2, 3)
    translation: np.ndarray  # (2,)
    z_iso: float
    z_anisotropy: float
    n_pairs: int
    rotation_refit: bool
    centred: bool
    note: str

    @property
    def z_slice(self) -> float:
        return self.z_iso / self.z_anisotropy if self.z_anisotropy else self.z_iso

    def fib_from_fm(self, x: float, y: float, z_slice: float) -> Tuple[float, float]:
        p = (
            self.matrix @ np.array([x, y, z_slice * self.z_anisotropy])
            + self.translation
        )
        return float(p[0]), float(p[1])

    def fm_from_fib(self, x: float, y: float) -> Tuple[float, float, float]:
        """The FM position at depth ``z_iso`` that lands on FIB pixel (x, y)."""
        inv = np.linalg.inv(self.matrix[:, :2])
        xy = inv @ (
            np.array([x, y]) - self.translation - self.matrix[:, 2] * self.z_iso
        )
        return float(xy[0]), float(xy[1]), self.z_slice


# ── pairs ────────────────────────────────────────────────────────────────


def usable_pairs(
    fib: Sequence[Coordinate], fm: Sequence[Coordinate]
) -> List[Tuple[Coordinate, Coordinate]]:
    """Index-paired points that may feed a fit: both usable, neither rejected."""
    return [
        (a, b)
        for a, b in zip(fib, fm)
        if a.usable and b.usable and a.status != PointStatus.REJECTED
    ]


def excluded_indices(fib: Sequence[Coordinate], fm: Sequence[Coordinate]) -> set:
    """Indices at which either point is tentative or rejected.

    The fit takes pairs by index, so a point that may not feed it takes its
    partner out too. Unpaired indices are not excluded: whether the lists
    match in length is the run gate's question, not this one's.
    """
    out = set()
    for i, c in enumerate(fib):
        if not c.usable or c.status == PointStatus.REJECTED:
            out.add(i)
    for i, c in enumerate(fm):
        if not c.usable or c.status == PointStatus.REJECTED:
            out.add(i)
    return out


def independent_pairs(
    fib: Sequence[Coordinate], fm: Sequence[Coordinate]
) -> List[Tuple[Coordinate, Coordinate]]:
    """Usable pairs whose FM position is evidence, not the projection's own.

    A prediction the user accepted without moving it sits exactly where the
    map put it; using it to refine the map would confirm the map with itself.
    """
    return [
        (a, b)
        for a, b in usable_pairs(fib, fm)
        if b.provenance != PointProvenance.PROJECTED
    ]


def suggest_indices(points: Sequence[Tuple[float, float]], k: int = 3) -> List[int]:
    """The ``k`` points that span the set best (farthest-point sampling).

    A translation solved from three points at the pattern's extremes is
    better conditioned than one from three neighbours, so these are the ones
    worth dragging first.
    """
    if len(points) <= k:
        return list(range(len(points)))
    pts = np.asarray(points, dtype=float)
    centre = pts.mean(axis=0)
    chosen = [int(np.argmax(np.linalg.norm(pts - centre, axis=1)))]
    while len(chosen) < k:
        d = np.min(
            np.stack([np.linalg.norm(pts - pts[c], axis=1) for c in chosen]), axis=0
        )
        chosen.append(int(np.argmax(d)))
    return chosen


# ── the map ──────────────────────────────────────────────────────────────


def build_projection(
    nominal: NominalTransform,
    fib: Sequence[Coordinate],
    fm: Sequence[Coordinate],
    *,
    z_slice: float,
    fm_shape: Optional[Tuple[int, int]] = None,
) -> Projection:
    """The FM->FIB map from the geometry and whatever pairs are confirmed.

    Rotation and scale are the geometry's; the translation is the mean offset
    of the independent pairs when there are any, else the stage metadata's.
    From :data:`MIN_PAIRS_FOR_ROTATION_REFIT` pairs the rotation is refitted
    (seeded from the geometry) as well. With no pairs at all, a stage
    translation that lands the FIB points off the FM image -- an odemis-written
    stack records its position in odemis's own frame -- is replaced by centring
    the FIB points on the FM image, so the first predictions are at least on
    screen; the first drop supplies the real translation.

    ``z_slice`` is the depth predictions are placed at: the slice on screen. A
    FIB point fixes a *line* through the FM volume, not a point -- on a METEOR
    one slice of depth moves the FIB position by about four pixels -- so the
    prediction is that line's crossing of the displayed slice, and the caller
    re-places predictions as the slice changes. The burns are then found by
    scrolling until a marker sits on one.
    """
    pairs = independent_pairs(fib, fm)
    zan = nominal.z_anisotropy
    z_iso = float(z_slice) * zan
    P = nominal.projection
    refit = False
    if len(pairs) >= MIN_PAIRS_FOR_ROTATION_REFIT:
        from fibsem.correlation.correlation_v2 import _fit_from_seed

        fm_iso = np.array([[b.point.x, b.point.y, b.point.z * zan] for _, b in pairs])
        fib_xy = np.array([[a.point.x, a.point.y, 0.0] for a, _ in pairs])
        try:
            R, s, _rms = _fit_from_seed(
                fm_iso, fib_xy, nominal.eulers_deg(), nominal.scale
            )
            P = s * R[:2, :]
            refit = True
        except Exception as exc:  # keep the geometry's rather than fail
            logging.debug(f"seeded refit for re-projection failed: {exc}")

    centred = False
    if pairs:
        t = np.mean(
            [
                np.array([a.point.x, a.point.y])
                - P @ np.array([b.point.x, b.point.y, b.point.z * zan])
                for a, b in pairs
            ],
            axis=0,
        )
        note = (
            f"rotation refitted from {len(pairs)} pairs"
            if refit
            else f"translation from {len(pairs)} pair{'s' if len(pairs) != 1 else ''}"
        )
    else:
        t = np.asarray(nominal.translation, dtype=float)
        note = "placed from the stage metadata"
        if fib and fm_shape is not None:
            h, w = fm_shape
            inv = np.linalg.inv(P[:, :2])
            centre_fib = np.mean([[c.point.x, c.point.y] for c in fib], axis=0)
            centre_fm = inv @ (centre_fib - t - P[:, 2] * z_iso)
            if not (0 <= centre_fm[0] < w and 0 <= centre_fm[1] < h):
                t = centre_fib - P @ np.array([w / 2, h / 2, z_iso])
                centred = True
                note = (
                    "centred on the FM image (the stage metadata gave no usable offset)"
                )
    return Projection(
        matrix=np.asarray(P, dtype=float),
        translation=np.asarray(t, dtype=float),
        z_iso=z_iso,
        z_anisotropy=zan,
        n_pairs=len(pairs),
        rotation_refit=refit,
        centred=centred,
        note=note,
    )


# ── predictions ──────────────────────────────────────────────────────────


def predictions_for(
    fib: Sequence[Coordinate], fm: Sequence[Coordinate], *, z_slice: float
) -> List[Coordinate]:
    """New predicted FM points for the FIB fiducials that have no FM partner.

    Pairs are by index, so the FIB points from ``len(fm)`` on are unpaired.
    The new points are placed at the origin; :func:`place_predictions` moves
    them. When nothing has been confirmed yet the three that span the pattern
    best are marked suggested.
    """
    new = [
        Coordinate(
            point=PointXYZ(x=0.0, y=0.0, z=float(z_slice)),
            point_type=PointType.FM,
            status=PointStatus.PREDICTED,
            provenance=PointProvenance.PROJECTED,
        )
        for _ in fib[len(fm) :]
    ]
    if new and not independent_pairs(fib, fm):
        unpaired = fib[len(fm) :]
        for i in suggest_indices([(c.point.x, c.point.y) for c in unpaired]):
            new[i].status = PointStatus.SUGGESTED
    return new


def place_predictions(
    projection: Projection, fib: Sequence[Coordinate], fm: Sequence[Coordinate]
) -> List[Coordinate]:
    """Move every tentative FM point to where its FIB partner projects.

    Confirmed points are never touched. Returns the points that moved. Once
    pairs exist the suggestion has done its job, and suggested points become
    plain predictions so the highlight does not outlive its meaning.
    """
    moved: List[Coordinate] = []
    for a, b in zip(fib, fm):
        if b.status not in PointStatus.TENTATIVE:
            continue
        x, y, z = projection.fm_from_fib(a.point.x, a.point.y)
        b.point.x, b.point.y, b.point.z = x, y, z
        if projection.n_pairs and b.status == PointStatus.SUGGESTED:
            b.status = PointStatus.PREDICTED
        moved.append(b)
    return moved
