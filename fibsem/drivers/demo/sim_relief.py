"""Render a SampleScene as a height field: the `relief` renderer.

The flat renderer paints the scene onto the sample plane and squashes it for
a tilted view. This one gives everything a height - ice, bars, cells, specks -
and looks at it the way a beam does: straight along the beam, so tall things
hide what is behind them, slopes and edges shade by secondary-electron yield,
and a milled pattern is cut along the milling beam.

Positions come from the same projection the flat renderer uses: a point at
height 0 lands on the same pixel either way, and a point at height h moves as
a stage raised by h would move it. The scene's layout (grids, cells, specks)
is unchanged; only how it looks.

Internally in micrometres.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi

if TYPE_CHECKING:
    from fibsem.drivers.demo.sim_scene import SampleScene

UM = 1e6
VOID = -60.0  # um; the height of nothing (through the grid, or milled away)
FLOOR_SMOOTH = 3.0  # um; a milled floor is flat over this scale
FILM_THICKNESS = 0.25  # um; what is under the bare ice before nothing
BAR_THICKNESS = 18.0  # um
CRACK_DEPTH = 0.6  # um; a hairline crack's groove at its line
ICE_BLANKET = 3.0  # um; cells sit in the ice, the surface over them this smooth

# secondary-electron yield per material: (SEM, FIB)
ICE, CELL, METAL, CRYSTAL, HOLE, HOLDER, SPHERE, COAT, NOTHING = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    9,
)
MATERIAL = {
    ICE: (1.0, 1.0),
    CELL: (1.16, 0.9),
    METAL: (1.25, 1.35),
    CRYSTAL: (1.3, 1.2),
    HOLE: (0.86, 0.9),
    HOLDER: (0.7, 0.8),
    SPHERE: (0.2, 0.3),
    COAT: (1.08, 1.35),
    NOTHING: (0.04, 0.03),
}
# a GIS coat seen edge-on, along the top of a cut face: the metal in it
# makes it the brightest band on the face, in both beams
COAT_EDGE = (1.6, 1.7)
# how much brighter (SEM, FIB) ice or a coat reads over a grid bar
BAR_UNDER = (0.25, 0.25)
BRIGHTNESS = 115.0  # grey level of plain ice
CACHE_SIZE = 8


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


# --------------------------------------------------------------------- heights
def _gauss(a: np.ndarray, sigma) -> np.ndarray:
    """A gaussian blur (sigma in samples, per axis) that stays cheap when it
    is wide: an axis blurred by more than a few samples is block-averaged,
    blurred at sigma / step and scaled back, which a blur that wide cannot
    tell from the direct one."""
    a = np.asarray(a)
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


def _feature_height(f) -> float:
    """How tall a feature stands (um): its own height when it has one, else
    from what it is and its size."""
    if f.height > 0:
        return f.height * UM
    sigma = f.sigma * UM
    if f.kind == "cell":
        if f.part == "nucleus":
            return min(0.25 * sigma, 1.2)
        if f.part == "organelle":
            return 0.15
        return min(0.35 * sigma, 2.2)
    if f.kind == "ice":
        return 0.4
    if f.part == "frost":
        return 0.5 * sigma
    return min(0.2 * sigma, 0.5)  # contamination, the fiducial's points


def surface(
    scene: "SampleScene", x: np.ndarray, y: np.ndarray, step: Tuple[float, float]
):
    """Height (um), material, the height each column's material starts at,
    which samples a milling cut made, the curtaining per column, and how far
    each cut sample was lowered, on a grid of world points (x, y) in um (x
    varying along columns, y along rows). `step` is the sampling along
    (columns, rows) in um, for the smoothing.

    The sample before milling is worked out once per grid of points and
    kept; milling then cuts only the patterns added since the last call, so
    a view imaged again after another pass costs only the new cut."""
    base = _unmilled_cached(scene, x, y, step)
    h, mat, bottom = base["h"], base["mat"], base["bottom"]
    if not scene.milled:
        return h, mat, bottom, None, base["curtain"], None
    top, floor, cut = _mill(scene, x, y, base)
    mat = mat.copy()
    mat[top <= VOID / 2] = NOTHING
    below = (h - top).astype(np.float32)
    return (
        top.astype(np.float32),
        mat,
        floor.astype(np.float32),
        cut,
        base["curtain"],
        below,
    )


def _unmilled_cached(scene, x, y, step) -> dict:
    sx, sy = step
    key = (
        "unmilled",
        scene.seed,
        x.shape,
        round(float(x[0, 0]), 4),
        round(float(x[0, -1]), 4),
        round(float(y[0, 0]), 4),
        round(float(y[-1, 0]), 4),
        round(float(sx), 6),
        round(float(sy), 6),
        round(scene.gis_thickness * UM, 4),
        round(float(scene.curtaining), 4),
        len(scene.features),
        id(scene.features),
    )
    cache = _cache(scene)
    base = cache.get(key)
    if base is None:
        h, mat, bottom, specks, bar = _unmilled(scene, x, y, step)
        h = h.astype(np.float32)
        base = {
            "bar": bar,
            "h": h,
            "mat": mat,
            "bottom": bottom.astype(np.float32),
            "curtain": _curtain(scene, x, h, specks, sx),
            # the level a pattern's depth is measured from: the surface with
            # its small bumps smoothed away, so a floor comes out flat, not a
            # lowered copy of the top
            "level": _smooth_top(
                h, (FLOOR_SMOOTH / max(sy, 1e-3), FLOOR_SMOOTH / max(sx, 1e-3))
            ),
        }
        cache[key] = base
        while len(cache) > CACHE_SIZE:
            cache.popitem(last=False)
    else:
        cache.move_to_end(key)
    return base


def _unmilled(scene, x, y, step):
    """The sample before any milling: height, material, bottom and the
    specks on top (for the curtaining)."""
    bars, holes, rips, rim, beyond = scene.film_masks(x / UM, y / UM)
    shape = x.shape
    mat = np.zeros(shape, dtype=np.uint8)
    sx, sy = step
    # the bars stand above the squares as rounded ridges, the ice climbing
    # to them in a meniscus, so each square is a shallow well
    metal = (bars | rim).astype(np.float32)
    h = (scene.bar_ridge(x / UM, y / UM) * UM).astype(np.float32)
    mat[bars | rim] = METAL
    # the ring pressed into the grid's orientation mark
    h -= scene.centre_mark_groove(x / UM, y / UM) * UM * bars
    seed = int(scene.seed)
    # ice: long waves and fine grain, fixed to the world
    h += fbm(
        x,
        y,
        (60, 18, 5, 1.5, 0.5, 0.15),
        (0.6, 0.3, 0.12, 0.05, 0.025, 0.012),
        seed,
        2 * min(sx, sy),
    )
    # holes in a holey film are filled with ice that sits a little lower
    rim_dip = 0.0
    if holes.any():
        rim_dip = ndi.gaussian_filter(holes.astype(np.float32), (0.3 / sy, 0.3 / sx))
        h -= 0.12 * rim_dip
        mat[holes] = HOLE
    bottom = h - FILM_THICKNESS - BAR_THICKNESS * metal
    base = h.copy()

    # cells, ice plates and specks, each from its outline in the world
    lift = np.zeros(shape, dtype=np.float32)
    specks = np.zeros(shape, dtype=np.float32)
    cell = np.zeros(shape, dtype=bool)
    crystal = np.zeros(shape, dtype=bool)
    sphere = np.zeros(shape, dtype=bool)
    plate = np.zeros(shape, dtype=np.float32)
    x_lo, x_hi, y_lo, y_hi = x.min(), x.max(), y.min(), y.max()
    x_axis, y_axis = x[0], y[:, 0]
    for f in scene.features:
        fx, fy, sigma = f.x * UM, f.y * UM, f.sigma * UM
        reach = (4.0 if f.sharpness < 2 else 2.2) * sigma * (1 + f.wobble) + 1.0
        if (
            fx + reach < x_lo
            or fx - reach > x_hi
            or fy + reach < y_lo
            or fy - reach > y_hi
        ):
            continue
        # the samples are a grid: find the box from its axes
        cols = np.nonzero(np.abs(x_axis - fx) < reach)[0]
        rows = np.nonzero(np.abs(y_axis - fy) < reach)[0]
        if not len(cols) or not len(rows):
            continue
        box = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
        dx, dy = x[box] - fx, y[box] - fy
        if f.angle:
            c, s = np.cos(f.angle), np.sin(f.angle)
            dx, dy = dx * c + dy * s, -dx * s + dy * c
        sx_f = max(sigma, 0.05)
        sy_f = max(sigma * f.eccentricity, 0.05)
        r2 = (dx / sx_f) ** 2 + (dy / sy_f) ** 2
        if f.kind == "ice" or f.part == "frost":
            # an ice plate (and a frost grain) is a hexagon: its distance is the largest of the
            # three face normals'
            rx, ry = dx / sx_f, dy / sy_f
            hexd = np.maximum(
                np.abs(rx),
                np.maximum(
                    np.abs(0.5 * rx + 0.866 * ry), np.abs(0.5 * rx - 0.866 * ry)
                ),
            )
            r2 = hexd**2
        if f.wobble and f.kind != "ice" and f.part != "frost":
            t = np.arctan2(dy / sy_f, dx / sx_f)
            mod = 1.0 + f.wobble * (
                0.6 * np.sin(3 * t + f.wobble_phase)
                + 0.4 * np.sin(5 * t + 2 * f.wobble_phase)
            )
            r2 = r2 / mod**2
        profile = np.exp(-0.5 * r2**f.sharpness).astype(np.float32)
        height = _feature_height(f) * profile
        if f.part == "sphere":
            # a ball: round in section, not a dome
            height = _feature_height(f) * np.sqrt(np.clip(1 - r2, 0, None))
            profile = (r2 < 1).astype(np.float32)
        if f.kind == "cell":
            # bodies and processes are the cell's outline, overlapping pieces
            # of it; a nucleus or granule stands on top
            if f.part in ("body", "bud", "neurite"):
                lift[box] = np.maximum(lift[box], height)
            else:
                lift[box] = lift[box] + height
            cell[box] |= profile > 0.3
        else:
            specks[box] = np.maximum(specks[box], height)
            if f.kind == "ice":
                plate[box] = np.maximum(plate[box], profile)
            if f.kind == "ice" or f.part == "frost":
                crystal[box] |= profile > 0.3
            elif f.part == "sphere":
                sphere[box] |= profile > 0.5
    solid = ~rips & ~beyond
    if lift.any():
        # cells sit in the ice: the surface over them is a smooth blanket, so a
        # 12 degree lamella through one stays a clean strip
        texture = fbm(
            x, y, (3.0, 1.0, 0.35), (0.12, 0.07, 0.04), seed + 5, 2 * min(sx, sy)
        )
        texture = np.where(lift > 0.05, texture, 0)
        soft = _gauss(lift, (ICE_BLANKET / sy, ICE_BLANKET / sx))
        # fine things (a neurite, a granule) keep their shape through it
        fine = np.clip(lift - _gauss(lift, (1.0 / sy, 1.0 / sx)), 0, None)
        h = h + np.maximum(0.2 * lift + soft, 0) + 0.6 * fine + 0.3 * texture
        # a cell covers the holes under it: no dip shows through
        h = h + 0.12 * rim_dip * np.clip(lift / 0.3, 0, 1)
        mat[cell & solid] = CELL
    # hairline cracks in the film: a narrow dark groove, not through
    crack = scene.crack_depth(x / UM, y / UM) * ~(bars | holes | cell | ~solid)
    crack = crack * (1 - plate)
    h = h - CRACK_DEPTH * crack
    mat[crack > 0.2] = HOLE
    if plate.any():
        # an ice plate lies on the film: flat on top, the holes and the
        # film's ripple under it hidden
        level = _gauss(h, (3.0 / sy, 3.0 / sx))
        h = h + plate * (level - h)
    h = h + specks
    mat[crystal & solid] = CRYSTAL
    mat[sphere & solid] = SPHERE

    gis = scene.gis_thickness * UM
    if gis > 0:
        # the GIS lays an organometallic coat over everything: it fills the
        # fine detail in, so the top is smoother, and covers what it lies on
        sm = max(gis, 0.3)
        smooth = _gauss(h, (sm / sy, sm / sx))
        h = np.maximum(h, smooth) + gis
        if gis >= 0.2:
            mat[solid & (mat != NOTHING)] = COAT

    # what is torn or past the grid is nothing (a broken hole too: no ice
    # across it); the holder sits far below
    if holes.any():
        rips = rips | (holes & scene.broken_hole_mask(x / UM, y / UM))
    h = np.where(rips, VOID, h)
    mat[rips] = NOTHING
    mat[beyond] = HOLDER
    h = np.where(beyond, -30.0, h)
    bottom = np.where(beyond, -60.0, bottom)

    del base
    return h, mat, bottom, specks, bars | rim


def _curtain(scene, x, h, specks, sx):
    """Curtaining down a cut face, per column: fine stripes along the beam,
    strongest under bumps on the top (cells, specks, frost); a GIS cap
    smooths the top and takes most of them away."""
    amp = scene.curtaining * np.exp(-scene.gis_thickness / 0.5e-6)
    if amp <= 0:
        return np.zeros(x.shape[1], dtype=np.float32)
    top = np.where(h > VOID / 2, h, np.nan)
    rough = np.nan_to_num(
        np.nanstd(top - ndi.gaussian_filter(np.nan_to_num(top), 2.0), axis=0)
    )
    bumps = specks.max(axis=0)
    weight = np.clip(0.3 + 2.0 * bumps + 3.0 * rough, 0, 2)
    stripes = value_noise(x[0], np.zeros_like(x[0]), 0.25, int(scene.seed) + 71)
    stripes = stripes - ndi.gaussian_filter1d(stripes, max(1.0, 1.5 / sx))
    return (amp * 0.5 * weight * stripes).astype(np.float32)


def _smooth_top(h, sigma):
    """Smooth the solid top, leaving out the holes (VOID)."""
    solid = (h > VOID / 2).astype(np.float32)
    num = _gauss(np.where(solid > 0, h, 0.0), sigma)
    den = _gauss(solid, sigma)
    return np.where(den > 1e-3, num / np.maximum(den, 1e-3), h)


def _mill(scene, x, y, base: dict):
    """Cut each milled pattern along the beam that milled it.

    A column of sample runs from `bottom` up to `h`, and a pattern removes the
    heights that project into it. What is left above a pattern stays and the
    column's floor rises to it (a lamella, or an overhang); a pattern that
    reaches the top lowers the top instead. One piece per column is kept, the
    top one: what is below it is hidden from above. The state after the
    patterns already cut is kept in `base`, so only new ones are cut."""
    done = base.get("milled")
    regions = list(scene.milled)
    if (
        done is None
        or len(done["regions"]) > len(regions)
        or any(a is not b for a, b in zip(done["regions"], regions))
    ):
        h = base["h"]
        done = {
            "regions": [],
            "top": h.copy(),
            "floor": base["bottom"].copy(),
            "lowered": np.zeros(h.shape, dtype=bool),
            "stopped": np.zeros(h.shape, dtype=bool),
        }
    top, floor = done["top"], done["floor"]
    lowered, stopped = done["lowered"], done["stopped"]
    level = base["level"]
    for region in regions[len(done["regions"]) :]:
        a, b = _cut_range(scene, region, x, y, top)
        # a pattern cuts only as deep as it was set to: a floor is left
        # where that is short of the bottom
        limit = np.minimum(top, level - region.depth * UM)
        short = limit > a
        a = np.maximum(a, limit)
        meets = (b > a) & (a < top) & (b > floor)
        above = meets & (b < top)
        floor = np.where(above, b, floor)
        drop = meets & ~above
        top = np.where(drop, np.maximum(a, floor), top)
        lowered = lowered | drop
        stopped = np.where(drop, short, stopped)
    base["milled"] = {
        "regions": regions,
        "top": top,
        "floor": floor,
        "lowered": lowered,
        "stopped": stopped,
    }
    gone = top <= floor + 1e-6
    top = np.where(gone, VOID, top)
    # 1 = a face the beam cut through, 2 = a floor it stopped short at
    cut = np.where(lowered & ~gone, np.where(stopped, 2, 1), 0).astype(np.int8)
    return top, floor, cut


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
def render(
    scene: "SampleScene",
    beam: str,
    frame: ViewFrame,
    pixel_size: float,
    resolution: Tuple[int, int],
    rng: Optional[np.random.Generator] = None,
    charge: float = 0.0,
) -> np.ndarray:
    """One beam's view of the scene as a uint8 image (height, width);
    `charge` is how strongly the (SEM) beam charges it."""
    width, height = int(resolution[0]), int(resolution[1])
    pix = pixel_size * UM
    ax, ay = frame.anchor
    cos_t, sin_t = np.cos(frame.theta), np.sin(frame.theta)
    # the view's pixels in the unrotated content frame
    vx = (np.arange(width) - width / 2) * pix
    vy = (np.arange(height) - height / 2) * pix
    rotated = not np.isclose(np.sin(frame.theta), 0) or np.cos(frame.theta) < 0
    if rotated:
        corners_x = np.array([vx[0], vx[-1], vx[0], vx[-1]])
        corners_y = np.array([vy[0], vy[0], vy[-1], vy[-1]])
        cu = corners_x * cos_t + corners_y * sin_t - ax
        cw = -corners_x * sin_t + corners_y * cos_t - ay
        u = np.arange(cu.min() - pix, cu.max() + 2 * pix, pix)
        w = np.arange(cw.min() - pix, cw.max() + 2 * pix, pix)
    else:
        u, w = vx - ax, vy - ay
    canvas, q = _canvas(scene, beam, frame, u, w, pix)
    if rotated:
        gx, gy = np.meshgrid(vx, vy)
        cu = gx * cos_t + gy * sin_t - ax
        cw = -gx * sin_t + gy * cos_t - ay
        img = ndi.map_coordinates(
            canvas,
            [(cw - w[0]) / pix, (cu - u[0]) / pix],
            order=1,
            cval=MATERIAL[NOTHING][0],
        )
        if charge > 0:
            q = ndi.map_coordinates(
                q, [(cw - w[0]) / pix, (cu - u[0]) / pix], order=1, cval=0.0
            )
    else:
        img = canvas
    return _finish(scene, img, rng, q if charge > 0 else None, charge)


def _canvas(scene, beam, frame: ViewFrame, u, w, pix):
    """The view over content columns `u` and rows `w` (um, rows increasing
    down the image), shaded."""
    fs, k, turn = frame.fs, frame.k, frame.turn
    sigma = -1.0 if k < 0 else 1.0
    kabs = abs(k)
    # s = Y fs + kabs h runs along the beam's view; Y is the sampling axis on
    # the plane, the beam travelling toward +Y and down
    s = sigma * w
    s_lo, s_hi = s.min(), s.max()
    y0 = (s_lo - kabs * 8.0) / fs
    y1 = (s_hi + kabs * 12.0) / fs
    dy = min(0.6 * pix / fs, 2.0)
    Y = np.arange(y0, y1 + dy, dy)
    xw = turn * u  # world x per column
    yw = sigma * turn * Y  # world y per sample row
    key = (
        scene.seed,
        round(xw[0], 4),
        round(yw[0], 4),
        len(xw),
        len(Y),
        round(pix, 6),
        round(dy, 6),
        len(scene.milled),
        round(scene.gis_thickness * UM, 4),
        id(scene),
    )
    cache = _cache(scene)
    hit = cache.get(key)
    if hit is None:
        gx, gy = np.meshgrid(xw, yw)
        hit = surface(scene, gx, gy, (pix, dy))
        hit = (*hit, _unmilled_cached(scene, gx, gy, (pix, dy))["bar"])
        cache[key] = hit
        while len(cache) > CACHE_SIZE:
            cache.popitem(last=False)
    h, mat, bottom, cut, curtain, below, bar = hit
    which = 0 if beam == "SEM" else 1

    # shading from the surface normal: SE yield rises as the beam meets it
    # more obliquely; convex edges glow; hollows are darker
    hs = np.maximum(h, -1.0)
    gy_, gx_ = np.gradient(hs, dy, pix)
    norm = np.sqrt(1 + gx_ * gx_ + gy_ * gy_)
    cosi = np.clip((gy_ * kabs + fs) / norm, 0.05, 1.0)
    yield_ = cosi**-0.75
    mat_y = np.zeros(h.shape, dtype=np.float32)
    for m, v in MATERIAL.items():
        mat_y[mat == m] = v[which]
    # a bar under thin ice or a coat still shows: the ice over it is thinner
    # and the metal below gives more signal (less through a cell)
    over = bar & (mat != METAL) & (mat != NOTHING)
    if over.any():
        through = np.where(mat == CELL, 0.3, 1.0)
        mat_y = mat_y * (1 + BAR_UNDER[which] * over * through)
    box = (max(1, int(3.5 / dy)), max(1, int(3.5 / pix)))
    smooth = ndi.uniform_filter(ndi.uniform_filter(hs, box), box)
    occl = np.clip(1 + 0.25 * np.tanh((hs - smooth) / 0.6), 0.6, 1.3)
    curv = -(ndi.laplace(ndi.gaussian_filter(hs, (0.4 / dy, 0.4 / pix))) / pix**2)
    edge = 1 + 0.6 * np.tanh(np.clip(curv, 0, None) * 0.8)
    if beam == "SEM" and scene.sputter_thickness > 0:
        # a sputtered metal coat gives a little more signal
        mat_y = mat_y * (1 + 0.12 * (1 - np.exp(-scene.sputter_thickness / 5e-9)))
    # what charges in the SEM: what stands up off the film (cells, plates,
    # beads), the thicker the more
    rel = hs - _gauss(hs, (15.0 / dy, 15.0 / pix))
    insulator = np.isin(mat, (CELL, CRYSTAL, SPHERE)) | ((mat == COAT) & (rel > 0.3))
    q = np.clip(np.maximum(rel / 1.5, 0.7 * insulator), 0, 1)
    q = _gauss(q * (mat != NOTHING), (1.0 / dy, 1.0 / pix))
    if cut is not None:
        q[cut > 0] = 0.0
    # the bar below carries the charge away
    q = q * np.where(bar, 0.5, 1.0)
    if beam == "SEM":
        # the in-chamber detector sits off to one side: faces toward it are brighter
        toward = (-gx_ * 0.5 - gy_ * 0.6 + 0.6) / norm
        det = 0.75 + 0.35 * np.clip(toward, -1, 1)
    else:
        det = 1.0
    # as bright as the flat renderer's film at this tilt: plain ice reads 1,
    # a little brighter the more grazing the view
    flat = max(fs / np.hypot(fs, kabs), 0.05) ** -0.75
    shade = mat_y * yield_ * occl * edge * det / flat * (1 + 0.2 * (1 - fs))
    if cut is not None:
        # a floor a pattern stopped short at keeps the surface it was cut
        # from, lower and a little darker (fewer electrons get out); a face
        # the beam cut through is smooth and carries the curtaining
        solid = mat != NOTHING
        floor = (cut == 2) & solid
        face = (cut == 1) & solid
        shade[floor] = 0.7 * shade[floor]
        r, c = np.nonzero(face)
        shade[face] = _face(
            scene, which, 0.6 * MATERIAL[ICE][which], curtain[c], xw[c], below[r, c]
        )
    shade[mat == NOTHING] = MATERIAL[NOTHING][which]

    # project: the first sample along the beam to reach a row is what it shows
    rows_s = (Y[:, None] * fs + h * kabs - s_lo) / pix
    top = np.maximum.accumulate(rows_s, axis=0)
    floor = np.where(mat == NOTHING, h, bottom)
    rows_floor = (Y[:, None] * fs + floor * kabs - s_lo) / pix
    n_rows = len(w)
    rows = np.arange(n_rows, dtype=np.float64)
    out = np.empty((n_rows, len(u)), dtype=np.float32)
    out_q = np.zeros((n_rows, len(u)), dtype=np.float32)
    slack = dy * fs / pix + 1.0
    # a wall shows what the column is made of; under a GIS coat that is ice
    under = np.where(mat == COAT, MATERIAL[ICE][which], mat_y)
    void = MATERIAL[NOTHING][which]
    side_k = np.zeros((n_rows, len(u)), dtype=np.int64)
    side_m = np.zeros((n_rows, len(u)), dtype=bool)
    for c in range(len(u)):
        idx = np.searchsorted(top[:, c], rows)
        seen = idx < len(Y)
        kk = np.clip(idx, 0, len(Y) - 1)
        # past a column's floor is nothing - give a sample step of slack, so
        # only a real gap (an edge with nothing behind it) reads as one
        solid = rows >= rows_floor[kk, c] - slack
        # below a sample's own top the row sees its side - a wall facing
        # the beam (a bar through a tear, a trench's far wall): one even tone
        side = rows < rows_s[kk, c] - slack
        out[:, c] = np.where(seen & solid, shade[kk, c], void)
        out_q[:, c] = np.where(seen & solid & ~side, q[kk, c], 0.0)
        side_k[:, c] = kk
        side_m[:, c] = side & seen & solid
    if side_m.any():
        out[side_m] = _wall(
            scene, which, side_k, side_m, rows_s, rows, under, curtain, xw, pix, kabs
        )
    if sigma < 0:
        out, out_q = out[::-1], out_q[::-1]
    return out, out_q


def _wall(scene, which, side_k, side_m, rows_s, rows, under, curtain, xw, pix, kabs):
    """The rows that see a wall straight on (a trench's far wall, a
    lamella's face), from how far below the wall's top edge each looks."""
    r, c = np.nonzero(side_m)
    kk = side_k[r, c]
    z = (rows_s[kk, c] - rows[r]) * pix / max(kabs, 1e-3)
    return _face(scene, which, 0.8 * under[kk, c], curtain[c], xw[c], z)


def _face(scene, which, base, curtain, x, z):
    """The tone of points on a cut face, `z` um below its top edge at world
    x (um): curtains start under the top and fade down the face, each its
    own length, and a GIS coat is a bright band along the top."""
    seed = int(scene.seed)
    reach = 1.0 + 3.0 * (1 + value_noise(x * 4.0, np.zeros_like(x), 1.0, seed + 91))
    waver = value_noise(x * 12.0, z, 1.5, seed + 89)
    run = np.exp(-z / reach) * np.clip(0.4 + 1.2 * waver, 0.0, 1.5)
    value = base * (1 + curtain * run)
    gis = scene.gis_thickness * UM
    if gis > 0:
        band = np.clip((gis - z) / 0.05 + 0.5, 0.0, 1.0)
        value = value + band * (COAT_EDGE[which] - value)
    return value


def _finish(scene, img, rng, q=None, charge=0.0):
    """Probe blur, charging, scan-line jitter, per-line gain and shot noise,
    as grey levels."""
    img = ndi.gaussian_filter(img, 0.7)
    if q is not None and charge > 0:
        img = scene.charge_up(img * BRIGHTNESS, q, charge) / BRIGHTNESS
    if scene.noise_sigma > 0 or scene.noise_fraction > 0:
        if rng is None:
            rng = np.random.default_rng()
        height, width = img.shape
        jitter = rng.normal(0, 0.35, height)
        cols = np.clip(np.arange(width)[None, :] - jitter[:, None], 0, width - 1)
        c0 = np.floor(cols).astype(np.int64)
        c1 = np.minimum(c0 + 1, width - 1)
        f = cols - c0
        r = np.arange(height)[:, None]
        img = img[r, c0] * (1 - f) + img[r, c1] * f
        img = img * (1 + rng.normal(0, 0.012, height))[:, None]
        counts = 30.0 * (12.0 / max(scene.noise_sigma, 1.0)) ** 2
        img = rng.poisson(np.clip(img, 0, None) * counts) / counts
    return np.clip(img * BRIGHTNESS, 0, 255).astype(np.uint8)


# reflected light: ice and the film under it make a thin film, so thickness
# reads as fringes; their period in thickness is wavelength / (2 n)
FRINGE_PERIOD = 0.21  # um, 550 nm light in ice (n = 1.31)
FRINGE_FADE = 0.8  # um; through thicker ice the fringes wash out
REFLECTANCE = {METAL: 1.0, CRYSTAL: 1.15, SPHERE: 1.2, HOLDER: 0.2, NOTHING: 0.0}


def reflectance(scene, x: np.ndarray, y: np.ndarray, step: Tuple[float, float]):
    """What the FM sees in reflected light on a grid of world points (um),
    relative to a grid bar: the bars bright; the ice grey, in fringes as its
    thickness changes (over a cell, round its edge, down a lamella); steep or
    rough ice scattering the light away, so cells read dark and textured;
    what is milled through, nothing."""
    h, mat, bottom, _, _, below = surface(scene, x, y, step)
    sx, sy = step
    seed = int(scene.seed)
    solid = h > VOID / 2
    # the ice's own thickness, plus a slow swell across a square
    swell = 0.15 * (1 + value_noise(x, y, 40.0, seed + 301))
    thick = np.clip(h - bottom, 0, None) + swell
    fringe = 0.5 + 0.5 * np.cos(2 * np.pi * thick / FRINGE_PERIOD)
    out = 0.35 + 0.3 * fringe * np.exp(-thick / FRINGE_FADE)
    gy, gx = np.gradient(np.where(solid, h, 0.0), sy, sx)
    slope = np.hypot(gx, gy)
    out = out / (1 + (slope / 0.4) ** 2)
    out = out * (1 + 0.15 * value_noise(x, y, 0.6, seed + 302) * (mat == CELL))
    out[mat == HOLE] *= 0.85
    scatter = 1 / (1 + (slope / 0.4) ** 2)
    if below is not None:
        # a milled floor is rough: it scatters more of the light away
        scatter = scatter * np.where(below > 0.05, 0.5, 1.0)
    for m, v in REFLECTANCE.items():
        # bright, but a pit milled in it (a spot burn on a bar) still shows
        out[mat == m] = v * scatter[mat == m]
    out[~solid] = 0.0
    return out.astype(np.float32)


def _cache(scene) -> "OrderedDict":
    cache = getattr(scene, "_relief_cache", None)
    if cache is None:
        cache = OrderedDict()
        object.__setattr__(scene, "_relief_cache", cache)
    return cache
