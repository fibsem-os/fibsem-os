"""Place overview tiles where their pictures say they are (experimental).

Uses the pair offsets `layout_check` measures. All tile positions are solved together
by weighted least squares from the confident pairs, with a weak pull back to the plan,
so a tile with no confident neighbours stays where it was planned. Pairs the consensus
disagrees with are dropped, worst first, so one false match on a repeating grid-bar
lattice cannot drag its neighbours.

The mean correction is removed, so the mosaic as a whole stays where the plan put it:
its centre is still the grid centre its metadata records, and its size and pixel size
do not change. Only the tiles move, relative to each other.

What this does not correct is a scale error in the stage model: a click on the mosaic
is converted back through that model, so a registered tile is off by about as far as
it was moved. `measure_registration` therefore applies it only while the largest move
is within `MAX_SHIFT`, and records the moves either way, so a run says how far its
tiles were off whether or not it was asked to align them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    Annotated,
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from fibsem.codec import NotSaved, encode
from fibsem.imaging.tiling.layout_check import (
    DEFAULT_MIN_SCORE,
    PairMeasurement,
    TileKey,
    TileLayoutCheck,
)

PixelPosition = Tuple[float, float]  # (y, x): top-left corner on the canvas

# Fewer confident pairs than this and the tiles are left as planned: on a 2x2 one
# false match would be half the evidence, and nothing could outvote it.
MIN_PAIRS = 3

# How strongly each tile is held to its planned position, relative to a pair offset
# of score 1. Small, so a confident measurement wins; non-zero, so a tile whose every
# neighbour was rejected still has a position: the plan.
PRIOR_WEIGHT = 0.01

# A pair that disagrees with the solved layout by more than this fraction of the
# tile's longer side is dropped. Generous, because real tiles do not quite fit a
# translation: on a tilted stage each row is at a different height, and pairs around
# a loop disagree by a few pixels.
MAX_RESIDUAL = 0.02

# The largest move, in metres, a tile may get from its planned place for the aligned
# mosaic to be used. A point picked on an aligned mosaic is off
# by about as far as its tile moved, so beyond this the plan is kept and the run warns.
# The Arctis's 104 % x 107 % y steps moved the corners of a 3 x 3 of 150 um tiles
# about 9 um.
MAX_SHIFT = 15e-6


def registered_positions(
    planned: Mapping[TileKey, PixelPosition],
    pairs: Sequence[PairMeasurement],
    tile_size: int,
    min_score: float = DEFAULT_MIN_SCORE,
) -> Tuple[Optional[Dict[TileKey, PixelPosition]], int, str]:
    """Each tile's registered top-left on the canvas, or None when too few pairs
    matched to say.

    Returns the positions (or None), how many pairs they were solved from, and a
    sentence saying why when None.

    Args:
        planned: Each acquired tile's planned top-left, (y, x) in pixels.
        pairs: The layout check's pair measurements.
        tile_size: The tile's longer side in pixels, which scales `MAX_RESIDUAL`.
        min_score: Pairs below this are not used.
    """
    used = [p for p in pairs if p.is_confident(min_score)]
    if len(used) < MIN_PAIRS:
        return (
            None,
            len(used),
            (
                f"{len(used)} tile pairs could be matched, and aligning them "
                f"needs {MIN_PAIRS}."
            ),
        )

    keys = list(planned)
    positions = _solve(keys, planned, used)
    limit = MAX_RESIDUAL * tile_size
    while used:
        worst = max(used, key=lambda p: _residual(p, positions))
        if _residual(worst, positions) <= limit:
            break
        logging.debug(
            f"Dropping tile pair {worst.a}-{worst.b}: disagrees with its neighbours "
            f"by {_residual(worst, positions):.1f} px"
        )
        used.remove(worst)
        positions = _solve(keys, planned, used)
    return positions, len(used), ""


@dataclass
class Registration:
    """Where a run's tiles were measured to be, and whether the mosaic used it.

    ``shifts`` is each tile's move from its planned place, ``"row,col": [dy, dx]`` in
    metres (y down, as on the canvas), recorded whether or not the tiles were moved.
    ``positions`` is what the mosaic is composed from when ``applied``; it is in
    canvas pixels and not part of the record.
    """

    requested: bool
    applied: bool = False
    note: str = ""
    pairs_used: int = 0
    shifts: Dict[str, List[float]] = field(default_factory=dict)
    largest_shift: Optional[float] = None
    limit: float = MAX_SHIFT
    positions: Annotated[Optional[Dict[TileKey, PixelPosition]], NotSaved] = field(
        default=None, repr=False, compare=False
    )

    def to_dict(self) -> Dict[str, Any]:
        return encode(self)

    def summary(self) -> Dict[str, Any]:
        """The record without the per-tile shifts."""
        out = self.to_dict()
        del out["shifts"]
        return out


def measure_registration(
    check: Optional[TileLayoutCheck],
    planned: Mapping[TileKey, PixelPosition],
    tile_size: int,
    pixel_size: Optional[float],
    requested: bool,
    limit: float = MAX_SHIFT,
) -> Registration:
    """Solve where a run's tiles are from its layout check, and decide whether the
    mosaic uses it: only when ``requested`` (``register_tiles``) and the largest move
    is within ``limit`` metres. The one step both tiled runners take.

    Never raises: a run's tiles are worth more than its registration, so a failure is
    a note in the record, and the plan is kept.
    """
    registration = Registration(requested=requested, limit=limit)
    if check is None:
        registration.note = "Tiles were left as planned: the layout check did not run."
        return _logged(registration)
    try:
        positions, registration.pairs_used, why = registered_positions(
            planned, check.pairs, tile_size=tile_size
        )
        if positions is None:
            registration.note = f"Tiles were left as planned: {why}"
            return _logged(registration)
        scale = pixel_size or 0.0
        moves = {
            key: (y - planned[key][0], x - planned[key][1])
            for key, (y, x) in positions.items()
        }
        registration.shifts = {
            f"{r},{c}": [dy * scale, dx * scale] for (r, c), (dy, dx) in moves.items()
        }
        largest = max(float(np.hypot(dy, dx)) for dy, dx in moves.values())
        if not pixel_size:
            registration.note = (
                "Tiles were left as planned: the pixel size is unknown, so how far "
                "they would move cannot be checked."
            )
            return _logged(registration)
        registration.largest_shift = largest * pixel_size
        moved = (
            f"the largest move is {registration.largest_shift * 1e6:.1f} µm "
            f"({len(moves)} tiles, {registration.pairs_used} matched pairs)"
        )
        if not requested:
            registration.note = f"Tiles were left as planned, as asked; {moved}."
        elif registration.largest_shift > limit:
            registration.note = (
                f"Tiles were left as planned: {moved}, more than the "
                f"{limit * 1e6:.0f} µm limit. The stage or the pixel size is off for "
                "this view by more than aligning can fix."
            )
        else:
            registration.applied = True
            registration.positions = positions
            registration.note = f"Tiles aligned: {moved}."
    except Exception as e:  # noqa: BLE001 - the tiles are worth more than this
        registration.note = f"Tiles were left as planned: aligning them failed ({e})."
        logging.warning(registration.note)
        return registration
    return _logged(registration)


def _logged(registration: Registration) -> Registration:
    refused = registration.requested and not registration.applied
    logging.log(logging.WARNING if refused else logging.INFO, registration.note)
    return registration


def _residual(
    pair: PairMeasurement, positions: Mapping[TileKey, PixelPosition]
) -> float:
    a, b = positions[pair.a], positions[pair.b]
    dy, dx = pair.measured  # type: ignore[misc]
    return float(np.hypot(b[0] - a[0] - dy, b[1] - a[1] - dx))


def _solve(
    keys: List[TileKey],
    planned: Mapping[TileKey, PixelPosition],
    pairs: Iterable[PairMeasurement],
) -> Dict[TileKey, PixelPosition]:
    """Weighted least squares for the corrections to the plan, mean removed.

    Solved as corrections rather than positions, which keeps the numbers small and
    makes the prior a pull towards zero. The axes are independent, so both are solved
    at once against one design matrix.
    """
    index = {k: i for i, k in enumerate(keys)}
    rows, rhs = [], []
    for p in pairs:
        if p.a not in index or p.b not in index:
            continue
        row = np.zeros(len(keys))
        row[index[p.b]], row[index[p.a]] = 1.0, -1.0
        weight = np.sqrt(max(p.score or 0.0, 0.0))
        rows.append(row * weight)
        correction = np.subtract(p.measured, p.planned)
        rhs.append(correction * weight)
    prior = np.sqrt(PRIOR_WEIGHT) * np.eye(len(keys))
    design = np.vstack([*rows, prior]) if rows else prior
    target = (
        np.vstack([*rhs, np.zeros((len(keys), 2))]) if rhs else np.zeros((len(keys), 2))
    )
    correction, *_ = np.linalg.lstsq(design, target, rcond=None)
    # The mosaic is placed by its centre; corrections must not move it as a whole.
    correction -= correction.mean(axis=0)
    return {
        k: (planned[k][0] + correction[i, 0], planned[k][1] + correction[i, 1])
        for k, i in index.items()
    }


def compose_mosaic(
    tiles: Mapping[TileKey, np.ndarray],
    positions: Mapping[TileKey, PixelPosition],
    shape: Tuple[int, int],
    dtype=None,
) -> np.ndarray:
    """Paste the tiles onto a canvas of `shape` at `positions`, clipping at its edges.

    Each seam falls halfway across the overlap rather than at the scan edge of
    whichever tile came last: every tile is pasted whole, then again cropped by half
    its overlap with each neighbour. The first pass fills any sliver the crop leaves.
    Where tiles moved apart from the canvas edge, the canvas keeps its zeros.
    """
    if not tiles:
        raise ValueError("No tiles to compose")
    first = next(iter(tiles.values()))
    canvas = np.zeros(shape, dtype=dtype or first.dtype)
    ordered = sorted(k for k in tiles if k in positions)

    for key in ordered:
        _paste(canvas, tiles[key], positions[key], (0, 0, 0, 0))

    for r, c in ordered:
        h, w = tiles[(r, c)].shape[:2]
        y, x = positions[(r, c)]
        crop = [0, 0, 0, 0]  # top, bottom, left, right
        neighbours = ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
        for side, n in enumerate(neighbours):
            if n not in positions or n not in tiles:
                continue
            ny, nx = positions[n]
            overlap = h - abs(ny - y) if side < 2 else w - abs(nx - x)
            crop[side] = max(0, int(overlap // 2))
        _paste(canvas, tiles[(r, c)], positions[(r, c)], tuple(crop))
    return canvas


def _paste(canvas: np.ndarray, tile: np.ndarray, position: PixelPosition, crop) -> None:
    top, bottom, left, right = crop
    h, w = tile.shape[:2]
    y, x = int(round(position[0])), int(round(position[1]))
    y0, x0, y1, x1 = y + top, x + left, y + h - bottom, x + w - right
    cy0, cx0 = max(y0, 0), max(x0, 0)
    cy1, cx1 = min(y1, canvas.shape[0]), min(x1, canvas.shape[1])
    if cy1 <= cy0 or cx1 <= cx0:
        return
    canvas[cy0:cy1, cx0:cx1] = tile[
        top + cy0 - y0 : top + cy1 - y0,
        left + cx0 - x0 : left + cx1 - x0,
    ]
