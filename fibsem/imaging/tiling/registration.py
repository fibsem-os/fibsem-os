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
is converted back through that model, so registered tiles are off by the scale error
times their distance from the centre. `registered_positions` therefore refuses a run
whose layout check warns, and the caller keeps the planned mosaic.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from fibsem.imaging.tiling.layout_check import (
    DEFAULT_MIN_SCORE,
    PairMeasurement,
    TileKey,
    TileLayoutSummary,
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


def registered_positions(
    planned: Mapping[TileKey, PixelPosition],
    pairs: Sequence[PairMeasurement],
    summary: TileLayoutSummary,
    tile_size: int,
    min_score: float = DEFAULT_MIN_SCORE,
) -> Tuple[Optional[Dict[TileKey, PixelPosition]], str]:
    """Each tile's registered top-left on the canvas, or None to keep the plan.

    Returns the positions (or None) and a sentence saying what was done, for the log
    and the operator.

    Args:
        planned: Each acquired tile's planned top-left, (y, x) in pixels.
        pairs: The layout check's pair measurements.
        summary: The layout check's summary of them.
        tile_size: The tile's longer side in pixels, which scales `MAX_RESIDUAL`.
        min_score: Pairs below this are not used.
    """
    if summary.warning:
        return None, "Tiles were left as planned, because they do not match the stage."
    used = [p for p in pairs if p.is_confident(min_score)]
    if len(used) < MIN_PAIRS:
        return None, (
            f"Tiles were left as planned: {len(used)} tile pairs could be matched, "
            f"and aligning them needs {MIN_PAIRS}."
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

    shift = max(
        float(
            np.hypot(positions[k][0] - planned[k][0], positions[k][1] - planned[k][1])
        )
        for k in keys
    )
    return positions, (
        f"Tiles aligned from {len(used)} matched pairs; the largest move was "
        f"{shift:.0f} px."
    )


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


def layout_summary(runner) -> Dict[str, Optional[object]]:
    """What a finished tiled run's layout came to, for a caller that is not
    watching its progress signal: the layout check's warning, and whether
    `register_tiles` moved the tiles (``applied``) and why (``note``). None
    for each that did not run."""
    registration = getattr(runner, "registration", None)
    return {
        "warning": getattr(runner, "layout_warning", None),
        "registration": (
            {"applied": registration["applied"], "note": registration["note"]}
            if registration
            else None
        ),
    }
