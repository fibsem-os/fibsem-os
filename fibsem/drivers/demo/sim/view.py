"""What every view of a SampleScene shares: how a beam sees the world (the
view frame and a milled pattern as the milling beam saw it), the noise and
blurs the sample is built from, and the finishing of a beam image (probe
blur, charging, scan jitter, shot noise).

Internally in micrometres.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Callable, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi

from fibsem.drivers.demo.sim import optics as sim_optics

if TYPE_CHECKING:
    from fibsem.drivers.demo.sim.scene import SampleScene

UM = 1e6
VOID = -60.0  # um; the height of nothing (through the grid, or milled away)
BRIGHTNESS = 115.0  # grey level of plain ice
CACHE_SIZE = 8
OUTSIDE = 0.04  # what a rotated view shows past the content it drew


@dataclass
class BeamCut:
    """A milled pattern as the milling beam saw it: a convex polygon in that
    view's unrotated frame, where a world point (x, y) at height h sits at
    (turn * x, turn * y * fs + k * h). Everything that projects into it, at
    its own height, is milled - so a rectangle is square in the milling view
    whatever the sample's relief."""

    polygon: np.ndarray  # (N, 2) um
    turn: float
    fs: float
    k: float


@dataclass
class ViewFrame:
    """How a view sees the world, in um: a content point (u, w) is
    (turn * x, turn * y * fs + k * h); the pixel at (column, row) offset
    (dx, dy) from the centre shows content R^-1 (dx, dy) - anchor."""

    anchor: Tuple[float, float]
    theta: float
    turn: float
    fs: float
    k: float


def polygon_span(polygon: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Where a vertical line at each `u` crosses a convex polygon: (lo, hi) in
    the second coordinate, lo > hi where it misses."""
    lo = np.full(u.shape, np.inf, dtype=np.float64)
    hi = np.full(u.shape, -np.inf, dtype=np.float64)
    n = len(polygon)
    for i in range(n):
        (u0, w0), (u1, w1) = polygon[i], polygon[(i + 1) % n]
        if u0 == u1:
            on = u == u0
            lo = np.where(on, np.minimum(lo, min(w0, w1)), lo)
            hi = np.where(on, np.maximum(hi, max(w0, w1)), hi)
            continue
        f = (u - u0) / (u1 - u0)
        on = (f >= 0) & (f <= 1)
        w = w0 + f * (w1 - w0)
        lo = np.where(on, np.minimum(lo, w), lo)
        hi = np.where(on, np.maximum(hi, w), hi)
    return lo, hi


# --------------------------------------------------------------------- noise
# what is worked out point by point over a view (the film's masks, a
# surface's texture) is done a band of rows at a time, on this many threads
WORKERS = min(4, os.cpu_count() or 1)
BAND_LEAST = 150_000  # points; a smaller view is not worth splitting
_pool: Optional[ThreadPoolExecutor] = None
_in_band = threading.local()


def by_rows(fn, *arrays):
    """`fn(*arrays)` for 2D arrays with the same rows, where each point of
    the answer comes from that point of the arrays alone: worked out a band
    of rows at a time on several threads and put back together, which gives
    what one call would. `fn` returns an array, or a tuple of them."""
    global _pool
    rows = arrays[0].shape[0]
    bands = min(WORKERS, rows, arrays[0].size // BAND_LEAST)
    if bands < 2 or getattr(_in_band, "on", False):
        return fn(*arrays)
    if _pool is None:
        _pool = ThreadPoolExecutor(WORKERS, thread_name_prefix="sim-view")
    edges = np.linspace(0, rows, bands + 1).astype(int)

    def band(k):
        _in_band.on = True
        try:
            return fn(*(a[edges[k] : edges[k + 1]] for a in arrays))
        finally:
            _in_band.on = False

    parts = list(_pool.map(band, range(bands)))
    if isinstance(parts[0], tuple):
        return tuple(np.concatenate(p) for p in zip(*parts))
    return np.concatenate(parts)


def together(*jobs) -> list:
    """What each of `jobs` (calls that do not depend on one another)
    returns, worked out side by side."""
    global _pool
    if WORKERS < 2 or len(jobs) < 2 or getattr(_in_band, "on", False):
        return [job() for job in jobs]
    if _pool is None:
        _pool = ThreadPoolExecutor(WORKERS, thread_name_prefix="sim-view")

    def run(job):
        _in_band.on = True
        try:
            return job()
        finally:
            _in_band.on = False

    return list(_pool.map(run, jobs))


def _hash(i, j, seed):
    h = (
        (i.astype(np.int64) * 73856093)
        ^ (j.astype(np.int64) * 19349663)
        ^ (seed * 83492791)
    )
    h = (h ^ (h >> 13)) * 1274126177
    return ((h ^ (h >> 16)) & 0xFFFFFF) / float(0xFFFFFF)


def value_noise(x, y, scale, seed):
    """Smooth value noise in [-1, 1] at feature size `scale`, the same at a
    world point whatever grid samples it. On an axis-aligned grid the lattice
    is hashed once and interpolated separably."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if (
        x.ndim == 2
        and x.shape == y.shape
        and min(x.shape) > 1
        and np.all(x[0] == x[-1])
        and np.all(y[:, 0] == y[:, -1])
    ):
        u, v = x[0] / scale, y[:, 0] / scale
        i, j = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
        fu, fv = u - i, v - j
        fu = (fu * fu * (3 - 2 * fu)).astype(np.float32)
        fv = (fv * fv * (3 - 2 * fv)).astype(np.float32)
        i0, j0 = i.min(), j.min()
        li = np.arange(i0, i.max() + 2)
        lj = np.arange(j0, j.max() + 2)
        lat = _hash(li[None, :], lj[:, None], seed).astype(np.float32)
        ii, jj = i - i0, j - j0
        rows = lat[:, ii] * (1 - fu) + lat[:, ii + 1] * fu
        out = rows[jj] * (1 - fv)[:, None] + rows[jj + 1] * fv[:, None]
        return 2 * out - 1
    u, v = x / scale, y / scale
    i, j = np.floor(u), np.floor(v)
    fu, fv = u - i, v - j
    fu = fu * fu * (3 - 2 * fu)
    fv = fv * fv * (3 - 2 * fv)
    i = i.astype(np.int64)
    j = j.astype(np.int64)
    a, b = _hash(i, j, seed), _hash(i + 1, j, seed)
    c, d = _hash(i, j + 1, seed), _hash(i + 1, j + 1, seed)
    out = (a * (1 - fu) + b * fu) * (1 - fv) + (c * (1 - fu) + d * fu) * fv
    return (2 * out - 1).astype(np.float32)


def fbm(x, y, scales, amps, seed, finest=0.0):
    out = np.zeros(np.broadcast(x, y).shape, dtype=np.float32)
    for k, (s, a) in enumerate(zip(scales, amps)):
        if s < finest:  # below the sampling: it would only alias
            continue
        out += a * value_noise(x, y, s, seed + 101 * k)
    return out


# --------------------------------------------------------------------- blurs
def gauss(a: np.ndarray, sigma: float, mode: str = "reflect") -> np.ndarray:
    """`ndi.gaussian_filter` of a 2D array, to the last bit, with each axis
    blurred a band at a time on several threads."""
    bands = min(WORKERS, min(a.shape), a.size // BAND_LEAST)
    if a.ndim != 2 or bands < 2 or getattr(_in_band, "on", False):
        return ndi.gaussian_filter(a, sigma, mode=mode)
    edges = np.linspace(0, a.shape[1], bands + 1).astype(int)
    down = together(
        *(
            partial(ndi.gaussian_filter1d, a[:, lo:hi], sigma, axis=0, mode=mode)
            for lo, hi in zip(edges[:-1], edges[1:])
        )
    )
    return by_rows(
        lambda b: ndi.gaussian_filter1d(b, sigma, axis=1, mode=mode),
        np.concatenate(down, axis=1),
    )


def _gauss(a: np.ndarray, sigma) -> np.ndarray:
    """A gaussian blur (sigma in samples, per axis) that stays cheap when it
    is wide: an axis blurred by more than a few samples is block-averaged,
    blurred at sigma / step and scaled back, which a blur that wide cannot
    tell from the direct one."""
    a = np.asarray(a)
    if a.dtype == np.float32 and not a.any():
        return np.zeros_like(a)  # nothing to blur
    sig = np.broadcast_to(np.asarray(sigma, dtype=float), (a.ndim,))
    steps = tuple(int(min(16, s // 4)) if s >= 8 else 1 for s in sig)
    if all(st == 1 for st in steps):
        return ndi.gaussian_filter(a, tuple(sig))
    pad = [(0, (-n) % st) for n, st in zip(a.shape, steps)]
    work = np.pad(a.astype(np.float32, copy=False), pad, mode="edge")
    shape = []
    for n, st in zip(work.shape, steps):
        shape += [n // st, st]
    small = work.reshape(shape).mean(axis=tuple(range(1, 2 * a.ndim, 2)))
    small = ndi.gaussian_filter(small, tuple(s / st for s, st in zip(sig, steps)))
    # back to the samples: each block's value sits at its centre
    coords = [(np.arange(n) - (st - 1) / 2) / st for n, st in zip(a.shape, steps)]
    grid = np.meshgrid(*coords, indexing="ij")
    out = ndi.map_coordinates(small, grid, order=1, mode="nearest")
    return out.astype(a.dtype if a.dtype.kind == "f" else np.float32)


def _smooth_top(h, sigma):
    """Smooth the solid top, leaving out the holes (VOID)."""
    solid = (h > VOID / 2).astype(np.float32)
    num, den = together(
        lambda: _gauss(np.where(solid > 0, h, 0.0), sigma),
        lambda: _gauss(solid, sigma),
    )
    return np.where(den > 1e-3, num / np.maximum(den, 1e-3), h)


# --------------------------------------------------------------------- cuts
def _cut_range(scene, region, x, y, top=None):
    """The heights (lo, hi) a milled region removes from each column at
    (x, y) um; lo > hi where it misses. The cut narrows as it goes down
    from `top` (the walls lean in: a taper)."""
    cut: Optional[BeamCut] = getattr(region, "beam_cut", None)
    if cut is None:
        # a region with no beam (a spot burn): straight down through it
        inside = scene._polygon_mask(region.points * UM, x, y)
        return np.where(inside, -1e3, 1.0), np.where(inside, 1e3, 0.0)
    w0 = cut.turn * y * cut.fs
    if x.ndim == 2 and x.shape[0] > 1 and np.array_equal(x[0], x[-1]):
        # a grid: every row has the same columns, so one row's span does
        lo, hi = polygon_span(cut.polygon, cut.turn * x[0])
        lo, hi = np.broadcast_to(lo, x.shape), np.broadcast_to(hi, x.shape)
    else:
        lo, hi = polygon_span(cut.polygon, cut.turn * x)
    if abs(cut.k) < 1e-6:
        inside = (w0 >= lo) & (w0 <= hi)
        return np.where(inside, -1e3, 1.0), np.where(inside, 1e3, 0.0)
    t = np.tan(np.deg2rad(scene.mill_taper)) if top is not None else 0.0
    ref = top if top is not None else 0.0
    if t and abs(cut.k) > 2 * t:
        # lo + t (ref - h) <= w0 + k h <= hi - t (ref - h)
        a1 = (lo + t * ref - w0) / (cut.k + t)
        b1 = (hi - t * ref - w0) / (cut.k - t)
        a, b = (a1, b1) if cut.k > 0 else (b1, a1)
    else:
        a, b = (lo - w0) / cut.k, (hi - w0) / cut.k
        if cut.k < 0:
            a, b = b, a
    miss = ~np.isfinite(lo) | (lo > hi)
    return np.where(miss, 1.0, a), np.where(miss, 0.0, b)


# --------------------------------------------------------------------- render
def content(frame: ViewFrame, pixel_size: float, resolution: Tuple[int, int]):
    """The content columns `u` and rows `w` (um) a view of `resolution` at
    `pixel_size` (m) is drawn over, and `to_view(a, order, cval)`, which lays
    something drawn over them out as the view's pixels: as it is, or turned
    where the scan is rotated (`order` 0 keeps its values, 1 blends them)."""
    width, height = int(resolution[0]), int(resolution[1])
    pix = pixel_size * UM
    ax, ay = frame.anchor
    cos_t, sin_t = np.cos(frame.theta), np.sin(frame.theta)
    # the view's pixels in the unrotated content frame
    vx = (np.arange(width) - width / 2) * pix
    vy = (np.arange(height) - height / 2) * pix
    rotated = not np.isclose(np.sin(frame.theta), 0) or np.cos(frame.theta) < 0
    if not rotated:
        return vx - ax, vy - ay, lambda a, order=1, cval=0.0: a
    corners_x = np.array([vx[0], vx[-1], vx[0], vx[-1]])
    corners_y = np.array([vy[0], vy[0], vy[-1], vy[-1]])
    cu = corners_x * cos_t + corners_y * sin_t - ax
    cw = -corners_x * sin_t + corners_y * cos_t - ay
    u = np.arange(cu.min() - pix, cu.max() + 2 * pix, pix)
    w = np.arange(cw.min() - pix, cw.max() + 2 * pix, pix)
    gx, gy = np.meshgrid(vx, vy)
    at = [
        (-gx * sin_t + gy * cos_t - ay - w[0]) / pix,
        (gx * cos_t + gy * sin_t - ax - u[0]) / pix,
    ]

    def to_view(a, order=1, cval=0.0):
        return ndi.map_coordinates(a, at, order=order, cval=cval)

    return u, w, to_view


def render(
    scene: "SampleScene",
    beam: str,
    frame: ViewFrame,
    pixel_size: float,
    resolution: Tuple[int, int],
    rng: Optional[np.random.Generator] = None,
    charge: float = 0.0,
    canvas: Callable = None,
    optics=None,
) -> np.ndarray:
    """One beam's view of the scene as a uint8 image (height, width);
    `charge` is how strongly the (SEM) beam charges it. `canvas(scene, beam,
    frame, u, w, pix, charged)` draws the view over content columns u and
    rows w (um) and returns it with the charge it takes up (None unless
    `charged`)."""
    pix = pixel_size * UM
    u, w, to_view = content(frame, pixel_size, resolution)
    canvas, q = canvas(scene, beam, frame, u, w, pix, charge > 0)
    img = to_view(canvas, 1, OUTSIDE)
    if charge > 0:
        q = to_view(q, 1, 0.0)
    if optics is not None:
        # the beam's focus, stigmation, current and dwell (see sim_optics)
        optics = (optics, 0 if beam == "SEM" else 1, pix)
    return _finish(
        scene, img, rng, q if charge > 0 else None, charge, optics, beam == "FIB"
    )


def _finish(scene, img, rng, q=None, charge=0.0, optics=None, ion=False):
    """Probe blur, charging, scan-line jitter, per-line gain and shot noise,
    as grey levels. `optics` is (the beam's Optics, 0 SEM or 1 FIB, um a
    pixel): what its focus, stigmation, current and dwell add. `ion` charges
    as the FIB does."""
    dose = 1.0
    if optics is not None:
        img = sim_optics.blur(img, *optics)
        dose = optics[0].dose(optics[1])
    img = gauss(img, 0.7)
    if rng is None:
        rng = np.random.default_rng()
    if q is not None and charge > 0:
        if ion:
            img = scene.ion_charge_up(img * BRIGHTNESS, q, charge, rng) / BRIGHTNESS
        else:
            img = scene.charge_up(img * BRIGHTNESS, q, charge) / BRIGHTNESS
    if scene.noise_sigma > 0 or scene.noise_fraction > 0:
        height, width = img.shape
        jitter = rng.normal(0, 0.35, height)
        cols = np.clip(np.arange(width)[None, :] - jitter[:, None], 0, width - 1)
        c0 = np.floor(cols).astype(np.int64)
        c1 = np.minimum(c0 + 1, width - 1)
        f = cols - c0
        r = np.arange(height)[:, None]
        img = img[r, c0] * (1 - f) + img[r, c1] * f
        img = img * (1 + rng.normal(0, 0.012, height))[:, None]
        counts = dose * 30.0 * (12.0 / max(scene.noise_sigma, 1.0)) ** 2
        img = rng.poisson(np.clip(img, 0, None) * counts) / counts
    return np.clip(img * BRIGHTNESS, 0, 255).astype(np.uint8)


def _cache(scene) -> "OrderedDict":
    cache = getattr(scene, "_view_cache", None)
    if cache is None:
        cache = OrderedDict()
        object.__setattr__(scene, "_view_cache", cache)
    return cache
