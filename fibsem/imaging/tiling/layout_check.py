"""Check an overview's tile layout against the pictures.

The tiles of an overview are placed on the planned grid: each one where the stage was
told to go, converted to pixels with the image's pixel size. If the pixel size or the
stage moves for that view are off, neighbouring tiles disagree at the seams -- and a
position picked on the overview is off by the same proportion, because a click goes
back through the same model.

This measures, from the images alone, how far each neighbour actually is from the
next. Nothing else is read: no stage positions, beam, tilt or rotation, so the same
check serves every beam, every view angle and every instrument. It only reports; the
mosaic is placed as planned either way.

For each pair of grid neighbours, every offset within a window around the planned one
is scored by the normalised cross-correlation of the pixels that offset makes them
share, all at once with FFTs. The median over the confident pairs, per axis, says how
much of the planned step the picture actually moved, and in which direction.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import cv2
import numpy as np
from scipy.signal import fftconvolve

TileKey = Tuple[int, int]  # (row, col)
PixelPosition = Tuple[float, float]  # (y, x)

# A pair counts only if its correlation peak reaches this. On real SEM and FIB
# overview tiles, overlaps that share features scored 0.5-0.99, most above 0.8;
# tiles with no shared content (the demo's noise) scored below 0.5.
DEFAULT_MIN_SCORE = 0.5

# The largest correction to the plan that is searched, as a fraction of the tile.
# Real steps have measured 94.7% (a tilted SEM view) to 109% (another SEM view, in
# y) of the planned one, which on a 10% overlap leaves anywhere from 15% to 1% of the
# tile shared. A finer grid-bar period than the window can alias, which is one reason
# the summary takes a median rather than trusting any single pair.
DEFAULT_MAX_DEVIATION = 0.35

# Offsets that share less than this fraction of the planned overlap are not scored:
# a sliver correlates well by chance. Low, because the largest real errors are the
# ones that leave the least overlap -- an FM row 108% of the step apart shares
# 29 px of a planned 113, and at 25% the check could not see it.
DEFAULT_MIN_OVERLAP = 0.15

# When the picture's step differs from the planned one by more than this fraction, or
# runs more than this many degrees off the grid axis, the run is flagged. Below them
# is what real data with no known calibration fault shows: a grazing-angle FIB view,
# whose rows sit at different heights, measured 1.2% and 1.0 degree. A tilted SEM
# view 5.3% out is what the check is for.
WARN_STEP_FRACTION = 0.03
WARN_ANGLE_DEG = 1.5

# Tiles are matched at a scale that brings their longer side to at most this.
_MATCH_SIZE = 768

# Pixels that barely vary cannot be located, and NCC divides by that variation; below
# this (in grey levels) a pair is skipped rather than scored.
_MIN_STD = 2.0


def measure_pair_offset(
    a: np.ndarray,
    b: np.ndarray,
    planned: PixelPosition,
    max_deviation: float = DEFAULT_MAX_DEVIATION,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
    max_size: int = _MATCH_SIZE,
) -> Optional[Tuple[PixelPosition, float]]:
    """Measure where `b` sits relative to `a`, within `max_deviation` of `planned`.

    Every candidate offset in the window is scored by the normalised cross-correlation
    of the two tiles over the pixels that offset makes them share -- not over a fixed
    template, so the window is not limited by the size of the planned overlap. All
    offsets are scored at once with FFTs (Padfield, "Masked object registration in
    the Fourier domain", 2012), on just the strips of each tile that some offset in
    the window could overlap.

    Args:
        a, b: Single-channel tiles of the same shape.
        planned: Expected offset of `b` relative to `a`, (dy, dx) in pixels, such
            that `b[t]` shows the same place as `a[t + offset]`.
        max_deviation: Largest correction searched, as a fraction of the tile's
            height (for dy) and width (for dx).
        min_overlap: Offsets that share fewer pixels than this fraction of the
            planned overlap are not scored: a sliver correlates well by chance.
        max_size: Tiles larger than this along either side are matched at a reduced
            scale. The surface costs FFTs of the strips, and the sub-pixel peak keeps
            the result to about half a reduced pixel.

    Returns:
        ((dy, dx), score) for the best offset, score in [-1, 1]; or None when the
        tiles do not overlap at the plan or are too flat to match.
    """
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(
            f"Tiles must be 2D and the same shape, got {a.shape} and {b.shape}"
        )
    h, w = a.shape
    py, px = int(round(planned[0])), int(round(planned[1]))
    if h - abs(py) <= 0 or w - abs(px) <= 0:
        return None
    dev_y, dev_x = int(round(max_deviation * h)), int(round(max_deviation * w))

    # The strips that some offset in the window could make overlap: `b` pixel t lands
    # on `a` pixel t + p + d, for |d| <= dev.
    ay0, ay1 = max(0, py - dev_y), min(h, h + py + dev_y)
    ax0, ax1 = max(0, px - dev_x), min(w, w + px + dev_x)
    by0, by1 = max(0, -py - dev_y), min(h, h - py + dev_y)
    bx0, bx1 = max(0, -px - dev_x), min(w, w - px + dev_x)
    scale = max(1, int(np.ceil(max(h, w) / max_size)))
    sa = _prepare(a[ay0:ay1, ax0:ax1], scale)
    sb = _prepare(b[by0:by1, bx0:bx1], scale)
    if float(sa.std()) < _MIN_STD or float(sb.std()) < _MIN_STD:
        return None

    ncc, count = _overlap_ncc_surface(sa, sb)
    # Index k of the surface is the strip shift k - (len(sb) - 1), in reduced pixels;
    # the tile offset is that, scaled up, plus where the strips start.
    oy = (np.arange(ncc.shape[0]) - (sb.shape[0] - 1)) * scale + ay0 - by0
    ox = (np.arange(ncc.shape[1]) - (sb.shape[1] - 1)) * scale + ax0 - bx0
    count = count * scale**2
    planned_area = (h - abs(py)) * (w - abs(px))
    valid = (
        (np.abs(oy - py) <= dev_y)[:, None]
        & (np.abs(ox - px) <= dev_x)[None, :]
        & (count >= min_overlap * planned_area)
    )
    if not valid.any():
        return None
    scores = np.where(valid, ncc, -np.inf)
    ky, kx = np.unravel_index(int(np.argmax(scores)), scores.shape)
    score = float(scores[ky, kx])
    if not np.isfinite(score):
        return None
    # A best score on the edge of the window is where the search stopped, not a peak:
    # the true offset is probably outside it.
    if not all(
        0 <= y < valid.shape[0] and 0 <= x < valid.shape[1] and valid[y, x]
        for y, x in ((ky - 1, kx), (ky + 1, kx), (ky, kx - 1), (ky, kx + 1))
    ):
        logging.debug(f"Best offset for planned {planned} is on the edge of the search")
        return None
    dy, dx = float(oy[ky]), float(ox[kx])
    # Sub-pixel: a parabola through the peak and its neighbours, along each axis.
    if 0 < ky < scores.shape[0] - 1:
        dy += scale * _parabolic_peak(*scores[ky - 1 : ky + 2, kx])
    if 0 < kx < scores.shape[1] - 1:
        dx += scale * _parabolic_peak(*scores[ky, kx - 1 : kx + 2])
    return (dy, dx), score


def _prepare(image: np.ndarray, scale: int) -> np.ndarray:
    """Float, lightly smoothed, and reduced by `scale`. Beam images are shot-noisy at
    overview dwell times, and pixel noise is uncorrelated between tiles, so it only
    dilutes the peak. Reduced by plain decimation after the blur, so reduced pixel k
    is full pixel k * scale and offsets scale back exactly."""
    image = cv2.GaussianBlur(
        np.ascontiguousarray(image, dtype=np.float64), (0, 0), 1.5 * scale
    )
    return image[::scale, ::scale]


def _overlap_ncc_surface(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """NCC of `a` and `b` over their overlap, and the overlap's pixel count, for every
    integer shift of `b` against `a`.

    Each term of the NCC is a sum over the overlap, and each such sum, for all shifts
    at once, is one FFT correlation: of the images, of their squares, or of a block of
    ones the size of the other. Index k is the shift k - (len(b) - 1) per axis.
    """

    def correlate(x, y):
        return fftconvolve(x, y[::-1, ::-1], mode="full")

    ones_a, ones_b = np.ones_like(a), np.ones_like(b)
    count = np.maximum(np.rint(correlate(ones_a, ones_b)), 1.0)
    sum_a = correlate(a, ones_b)
    sum_b = correlate(ones_a, b)
    cross = correlate(a, b) - sum_a * sum_b / count
    var_a = correlate(a * a, ones_b) - sum_a**2 / count
    var_b = correlate(ones_a, b * b) - sum_b**2 / count
    # Below this per-pixel variance the overlap is flat and the ratio is noise.
    flat = count * _MIN_STD**2
    denominator = np.sqrt(np.clip(var_a, 0, None) * np.clip(var_b, 0, None))
    ok = (var_a > flat) & (var_b > flat)
    ncc = np.full(count.shape, -np.inf)
    ncc[ok] = cross[ok] / denominator[ok]
    return ncc, count


def _parabolic_peak(left: float, centre: float, right: float) -> float:
    if not (np.isfinite(left) and np.isfinite(right)):
        return 0.0
    denominator = left - 2 * centre + right
    if denominator == 0:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))


@dataclass
class PairMeasurement:
    """One neighbour pair: `b` is right of or below `a`.

    `planned` and `measured` are `position(b) - position(a)` as (dy, dx) in pixels;
    `measured` is None when the pair could not be matched at all.
    """

    a: TileKey
    b: TileKey
    planned: PixelPosition
    measured: Optional[PixelPosition] = None
    score: Optional[float] = None

    @property
    def axis(self) -> str:
        return "x" if self.a[0] == self.b[0] else "y"

    def is_confident(self, min_score: float = DEFAULT_MIN_SCORE) -> bool:
        return self.measured is not None and (self.score or 0.0) >= min_score

    @property
    def step_fraction(self) -> Optional[float]:
        """How much of the planned step the picture moved along the pair's axis."""
        if self.measured is None:
            return None
        i = 1 if self.axis == "x" else 0
        return self.measured[i] / self.planned[i] if self.planned[i] else None

    @property
    def angle_deg(self) -> Optional[float]:
        """How far the step runs off the grid axis, in degrees; positive is clockwise
        in image coordinates (y down)."""
        if self.measured is None:
            return None
        dy, dx = self.measured
        if self.axis == "x":
            return float(np.degrees(np.arctan2(dy, dx)))
        return float(np.degrees(np.arctan2(-dx, dy)))


@dataclass
class AxisSummary:
    """The median step along one grid axis, over its confident pairs."""

    pairs: int
    step_fraction: float
    angle_deg: float

    def problems(self, axis: str) -> List[str]:
        out = []
        deviation = self.step_fraction - 1.0
        if abs(deviation) > WARN_STEP_FRACTION:
            longer = "longer" if deviation > 0 else "shorter"
            out.append(
                f"in {axis} each step was {abs(deviation):.1%} {longer} than planned"
            )
        if abs(self.angle_deg) > WARN_ANGLE_DEG:
            out.append(
                f"in {axis} the steps ran {self.angle_deg:+.1f}° off the image axis"
            )
        return out


@dataclass
class TileLayoutSummary:
    x: Optional[AxisSummary]
    y: Optional[AxisSummary]
    pairs_measured: int
    pairs_total: int

    @property
    def warning(self) -> Optional[str]:
        """A sentence for the operator when the layout is off, or None."""
        problems = [
            p
            for axis, s in (("x", self.x), ("y", self.y))
            if s is not None
            for p in s.problems(axis)
        ]
        if not problems:
            return None
        return (
            "Tile spacing differed from the plan: "
            + "; ".join(problems)
            + ". Positions picked far from the overview's centre may be off by a "
            "similar fraction of their distance from it. The pixel size or stage "
            "calibration for this view may need a check."
        )

    def describe(self) -> str:
        parts = [
            f"{axis}: {s.step_fraction:.1%} of planned step, {s.angle_deg:+.2f}° "
            f"({s.pairs} pairs)"
            for axis, s in (("x", self.x), ("y", self.y))
            if s is not None
        ]
        body = "; ".join(parts) if parts else "no confident pairs"
        return f"Tile layout check: {body} [{self.pairs_measured}/{self.pairs_total} matched]"


class TileLayoutCheck:
    """Measures neighbour pairs as the tiles of a run arrive.

    A tile's pixels are held only until every neighbour it has in the run has arrived,
    so on a typewriter or serpentine run about one row is in memory at a time, and the
    measuring happens between stage moves rather than all at the end.

    Args:
        planned: Each tile's planned top-left on the canvas, (y, x) in pixels, for
            every tile the run will acquire.
    """

    def __init__(self, planned: Mapping[TileKey, PixelPosition]):
        self.planned: Dict[TileKey, PixelPosition] = dict(planned)
        self.pairs: List[PairMeasurement] = []
        self._held: Dict[TileKey, np.ndarray] = {}
        self._arrived: set = set()

    def _neighbours(self, key: TileKey):
        r, c = key
        for n in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if n in self.planned:
                yield n

    def add(self, key: TileKey, image: np.ndarray) -> None:
        """Measure `key` against each neighbour already in, then let go of any tile
        whose neighbours are all in."""
        image = _single_channel(image)
        self._arrived.add(key)
        for n in self._neighbours(key):
            if n not in self._held:
                continue
            a, b = (n, key) if n < key else (key, n)
            data = {key: image, n: self._held[n]}
            plan = (
                self.planned[b][0] - self.planned[a][0],
                self.planned[b][1] - self.planned[a][1],
            )
            pair = PairMeasurement(a=a, b=b, planned=plan)
            result = measure_pair_offset(data[a], data[b], plan)
            if result is not None:
                pair.measured, pair.score = result
            self.pairs.append(pair)
            logging.debug(f"Tile pair {a}-{b}: {pair}")
        self._held[key] = image
        for k in [key, *self._neighbours(key)]:
            if k in self._held and all(n in self._arrived for n in self._neighbours(k)):
                del self._held[k]

    def summary(self, min_score: float = DEFAULT_MIN_SCORE) -> TileLayoutSummary:
        axes: Dict[str, Optional[AxisSummary]] = {}
        for axis in ("x", "y"):
            good = [
                p for p in self.pairs if p.axis == axis and p.is_confident(min_score)
            ]
            axes[axis] = (
                AxisSummary(
                    pairs=len(good),
                    step_fraction=float(np.median([p.step_fraction for p in good])),
                    angle_deg=float(np.median([p.angle_deg for p in good])),
                )
                if good
                else None
            )
        return TileLayoutSummary(
            x=axes["x"],
            y=axes["y"],
            pairs_measured=sum(p.is_confident(min_score) for p in self.pairs),
            pairs_total=len(self.pairs),
        )

    def record(self) -> Dict[str, Any]:
        """The summary, its thresholds and every pair, for looking back at a run's
        calibration without the tiles (the ``layout`` of `OverviewResult`)."""
        summary = self.summary()
        return {
            "summary": asdict(summary),
            "warning": summary.warning,
            "warn_step_fraction": WARN_STEP_FRACTION,
            "warn_angle_deg": WARN_ANGLE_DEG,
            "min_score": DEFAULT_MIN_SCORE,
            "pairs": [
                {
                    **asdict(p),
                    "axis": p.axis,
                    "step_fraction": p.step_fraction,
                    "angle_deg": p.angle_deg,
                }
                for p in self.pairs
            ],
        }


def _single_channel(image: np.ndarray) -> np.ndarray:
    """The plane to match on: the tile itself, or the first plane of a stack."""
    while image.ndim > 2:
        image = image[0]
    return image
