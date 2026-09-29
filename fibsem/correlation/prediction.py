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
until the user moves it, and only positions the user has placed (or
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
    "Projection",
    "build_projection",
    "excluded_indices",
    "independent_pairs",
    "place_predictions",
    "predictions_for",
    "suggest_indices",
    "suggest_visible",
    "on_image",
    "usable_pairs",
]


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
    That is a matter of status (``accepted``), not provenance: a dragged
    prediction keeps its ``projected`` provenance -- provenance says where a
    point came from and is set once -- and is evidence all the same.
    """
    return [
        (a, b) for a, b in usable_pairs(fib, fm) if b.status != PointStatus.ACCEPTED
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
    """The FM->FIB map from the prior and whatever pairs are confirmed.

    Rotation and scale are always the prior's; the translation is the mean
    offset of the independent pairs when there are any, else the stage
    metadata's. The rotation is never refitted here: a handful of burns on
    one surface cannot pin it better than the geometry or an earlier full
    fit did (measured: five pairs refitted the rotation 18 degrees off and
    quadrupled the residual on a METEOR lamella, and tripled the ring error
    on an Arctis one), so the rotation is left to the final fit. With no
    pairs at all, a stage
    translation that lands the FIB points off the FM image -- an odemis-written
    stack records its position in odemis's own frame -- is replaced by centring
    the FIB points on the FM image, so the first predictions are at least on
    screen; the first drop supplies the real translation.

    ``z_slice`` is the depth to predict at when no pair has one: the slice on
    screen when Project is pressed. With pairs, their mean z is used -- the
    burns are on one surface, to within a few slices. A FIB point fixes a line
    through the volume, not a point, so a burn deeper or shallower than that
    plane sits a little along that line from its ring; predictions stay put
    while the user scrolls, and the drop supplies the depth.
    """
    pairs = independent_pairs(fib, fm)
    zan = nominal.z_anisotropy
    z_iso = (
        float(np.mean([b.point.z for _, b in pairs])) * zan
        if pairs
        else float(z_slice) * zan
    )
    P = nominal.projection

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
        note = f"translation from {len(pairs)} pair{'s' if len(pairs) != 1 else ''}"
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
    them. When nothing has been placed yet the three that span the pattern
    best are flagged ``suggested`` (a highlight, not a status).
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
            new[i].suggested = True
    return new


def place_predictions(
    projection: Projection,
    fib: Sequence[Coordinate],
    fm: Sequence[Coordinate],
    *,
    fm_shape: Optional[Tuple[int, int]] = None,
) -> List[Coordinate]:
    """Move every tentative FM point to where its FIB partner projects.

    Placed points are never touched. Returns the points that moved. Once
    pairs exist the suggestion has done its job and the highlight is cleared
    so it does not outlive its meaning. Before any pair, the suggestion is
    made again from where the predictions landed: a ring off the image
    (``fm_shape`` = (height, width)) cannot be dragged, so "start here" goes
    to the three best-spread predictions the user can see.
    """
    moved: List[Coordinate] = []
    for a, b in zip(fib, fm):
        if b.status not in PointStatus.TENTATIVE:
            continue
        x, y, z = projection.fm_from_fib(a.point.x, a.point.y)
        b.point.x, b.point.y, b.point.z = x, y, z
        moved.append(b)
    if projection.n_pairs:
        for b in moved:
            b.suggested = False
    else:
        suggest_visible(fib, fm, fm_shape)
    return moved


def on_image(point: PointXYZ, fm_shape: Optional[Tuple[int, int]]) -> bool:
    """Inside the FM image's axes; True when the shape is unknown."""
    if fm_shape is None:
        return True
    h, w = fm_shape
    return 0.0 <= point.x <= w and 0.0 <= point.y <= h


def suggest_visible(
    fib: Sequence[Coordinate],
    fm: Sequence[Coordinate],
    fm_shape: Optional[Tuple[int, int]] = None,
) -> None:
    """Flag the three best-spread predictions that are on the image."""
    tentative = [i for i, b in enumerate(fm) if b.status in PointStatus.TENTATIVE]
    for i in tentative:
        fm[i].suggested = False
    visible = [i for i in tentative if on_image(fm[i].point, fm_shape)]
    for k in suggest_indices([(fib[i].point.x, fib[i].point.y) for i in visible]):
        fm[visible[k]].suggested = True
