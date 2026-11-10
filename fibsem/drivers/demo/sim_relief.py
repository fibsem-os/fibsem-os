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
FILM_THICKNESS = 0.25  # um; what is under the bare ice before nothing
BAR_THICKNESS = 18.0  # um
BAR_TOP = 0.0  # um; the film lies on top of the bars, flush with the squares
ICE_BLANKET = 3.0  # um; cells sit in the ice, the surface over them this smooth

# secondary-electron yield per material: (SEM, FIB)
ICE, CELL, METAL, CRYSTAL, HOLE, HOLDER, NOTHING = 0, 1, 2, 3, 4, 5, 9
MATERIAL = {
    ICE: (1.0, 1.0),
    CELL: (1.16, 0.9),
    METAL: (1.25, 1.35),
    CRYSTAL: (1.3, 1.2),
    HOLE: (0.86, 0.9),
    HOLDER: (0.7, 0.8),
    NOTHING: (0.04, 0.03),
}
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
    return min(0.2 * sigma, 0.5)  # contamination, the fiducial's points


def surface(
    scene: "SampleScene", x: np.ndarray, y: np.ndarray, step: Tuple[float, float]
):
    """Height (um), material, the height each column's material starts at, and
    which samples a milling cut made, on a grid of world points (x, y) in um
    (x varying along columns, y along rows). `step` is the sampling along
    (columns, rows) in um, for the smoothing."""
    bars, holes, rips, rim, beyond = scene.film_masks(x / UM, y / UM)
    shape = x.shape
    mat = np.zeros(shape, dtype=np.uint8)
    sx, sy = step
    # the film lies over the bars: their tops stand a little proud, rounded
    metal = (bars | rim).astype(np.float32)
    h = BAR_TOP * ndi.gaussian_filter(metal, (2.0 / sy, 2.0 / sx))
    mat[bars | rim] = METAL
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
        if f.wobble:
            t = np.arctan2(dy / sy_f, dx / sx_f)
            mod = 1.0 + f.wobble * (
                0.6 * np.sin(3 * t + f.wobble_phase)
                + 0.4 * np.sin(5 * t + 2 * f.wobble_phase)
            )
            r2 = r2 / mod**2
        profile = np.exp(-0.5 * r2**f.sharpness).astype(np.float32)
        height = _feature_height(f) * profile
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
                crystal[box] |= profile > 0.3
    solid = ~rips & ~beyond
    if lift.any():
        # cells sit in the ice: the surface over them is a smooth blanket, so a
        # 12 degree lamella through one stays a clean strip
        texture = fbm(
            x, y, (3.0, 1.0, 0.35), (0.12, 0.07, 0.04), seed + 5, 2 * min(sx, sy)
        )
        texture = np.where(lift > 0.05, texture, 0)
        soft = ndi.gaussian_filter(lift, (ICE_BLANKET / sy, ICE_BLANKET / sx))
        # fine things (a neurite, a granule) keep their shape through it
        fine = np.clip(lift - ndi.gaussian_filter(lift, (1.0 / sy, 1.0 / sx)), 0, None)
        h = h + np.maximum(0.2 * lift + soft, 0) + 0.6 * fine + 0.3 * texture
        # a cell covers the holes under it: no dip shows through
        h = h + 0.12 * rim_dip * np.clip(lift / 0.3, 0, 1)
        mat[cell & solid] = CELL
    h = h + specks
    mat[crystal & solid] = CRYSTAL

    # what is torn or past the grid is nothing; the holder sits far below
    h = np.where(rips, VOID, h)
    mat[rips] = NOTHING
    mat[beyond] = HOLDER
    h = np.where(beyond, -30.0, h)
    bottom = np.where(beyond, -60.0, bottom)

    cut = None
    if scene.milled:
        h, bottom, cut = _mill(scene, x, y, h, bottom)
        mat[h <= VOID / 2] = NOTHING
    del base
    return h.astype(np.float32), mat, bottom.astype(np.float32), cut


def _mill(scene, x, y, h, bottom):
    """Cut each milled pattern along the beam that milled it.

    A column of sample runs from `bottom` up to `h`, and a pattern removes the
    heights that project into it. What is left above a pattern stays and the
    column's floor rises to it (a lamella, or an overhang); a pattern that
    reaches the top lowers the top instead. One piece per column is kept, the
    top one: what is below it is hidden from above."""
    top = h.copy()
    floor = bottom.copy()
    lowered = np.zeros(h.shape, dtype=bool)
    for region in scene.milled:
        a, b = _cut_range(scene, region, x, y)
        meets = (b > a) & (a < top) & (b > floor)
        above = meets & (b < top)
        floor = np.where(above, b, floor)
        drop = meets & ~above
        top = np.where(drop, np.maximum(a, floor), top)
        lowered |= drop
    gone = top <= floor + 1e-6
    top = np.where(gone, VOID, top)
    return top, floor, lowered & ~gone


def _cut_range(scene, region, x, y):
    """The heights (lo, hi) a milled region removes from each column at
    (x, y) um; lo > hi where it misses."""
    cut: Optional[BeamCut] = getattr(region, "beam_cut", None)
    if cut is None:
        # a region with no beam (a spot burn): straight down through it
        inside = scene._polygon_mask(region.points * UM, x, y)
        return np.where(inside, -1e3, 1.0), np.where(inside, 1e3, 0.0)
    u = cut.turn * x
    w0 = cut.turn * y * cut.fs
    lo, hi = polygon_span(cut.polygon, u)
    if abs(cut.k) < 1e-6:
        inside = (w0 >= lo) & (w0 <= hi)
        return np.where(inside, -1e3, 1.0), np.where(inside, 1e3, 0.0)
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
) -> np.ndarray:
    """One beam's view of the scene as a uint8 image (height, width)."""
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
    canvas = _canvas(scene, beam, frame, u, w, pix)
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
    else:
        img = canvas
    return _finish(scene, img, rng)


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
        id(scene),
    )
    cache = _cache(scene)
    hit = cache.get(key)
    if hit is None:
        gx, gy = np.meshgrid(xw, yw)
        hit = surface(scene, gx, gy, (pix, dy))
        cache[key] = hit
        while len(cache) > CACHE_SIZE:
            cache.popitem(last=False)
    h, mat, bottom, cut = hit
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
    box = (max(1, int(3.5 / dy)), max(1, int(3.5 / pix)))
    smooth = ndi.uniform_filter(ndi.uniform_filter(hs, box), box)
    occl = np.clip(1 + 0.25 * np.tanh((hs - smooth) / 0.6), 0.6, 1.3)
    curv = -(ndi.laplace(ndi.gaussian_filter(hs, (0.4 / dy, 0.4 / pix))) / pix**2)
    edge = 1 + 0.6 * np.tanh(np.clip(curv, 0, None) * 0.8)
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
        # a face the beam cut is smooth: an even tone, darker in a trench
        # floor (fewer electrons get out)
        face = cut & (mat != NOTHING)
        shade[face] = 0.6 * MATERIAL[ICE][which]
    shade[mat == NOTHING] = MATERIAL[NOTHING][which]

    # project: the first sample along the beam to reach a row is what it shows
    rows_s = (Y[:, None] * fs + h * kabs - s_lo) / pix
    top = np.maximum.accumulate(rows_s, axis=0)
    floor = np.where(mat == NOTHING, h, bottom)
    rows_floor = (Y[:, None] * fs + floor * kabs - s_lo) / pix
    n_rows = len(w)
    rows = np.arange(n_rows, dtype=np.float64)
    out = np.empty((n_rows, len(u)), dtype=np.float32)
    slack = dy * fs / pix + 1.0
    wall = 0.8 * mat_y
    void = MATERIAL[NOTHING][which]
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
        value = np.where(side, wall[kk, c], shade[kk, c])
        out[:, c] = np.where(seen & solid, value, void)
    if sigma < 0:
        out = out[::-1]
    return out


def _finish(scene, img, rng):
    """Probe blur, scan-line jitter, per-line gain and shot noise, as grey levels."""
    img = ndi.gaussian_filter(img, 0.7)
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


def _cache(scene) -> "OrderedDict":
    cache = getattr(scene, "_relief_cache", None)
    if cache is None:
        cache = OrderedDict()
        object.__setattr__(scene, "_relief_cache", cache)
    return cache
