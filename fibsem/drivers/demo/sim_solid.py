"""Render a SampleScene as solid objects: the `solid` renderer.

The relief renderer keeps one height, one material and one bottom per point,
so nothing can lie under anything else. Here the sample is a set of objects -
grid bars, the film, cells with a nucleus, one sheet of ice over them,
particles resting on top - and at each point of the sample plane they stack up
as solid pieces, each with a bottom, a top and a material. A beam shows the
first piece it meets; a milled pattern removes what projects into it along
the milling beam, so the face it leaves shows what the sample is made of
there (ice, cell, nucleus), a particle has an underside and a lamella has
whatever was under it.

Positions come from the projection the other renderers use (see
sim_relief.ViewFrame): a point at height 0 lands on the same pixel.

Internally in micrometres. z = 0 is the top of the ice on the open film, the
plane the stage brings to the beams; the film lies ICE_THICKNESS under it.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np
from scipy import ndimage as ndi

from fibsem.drivers.demo import sim_relief
from fibsem.drivers.demo.sim_relief import UM, ViewFrame, _gauss, fbm

if TYPE_CHECKING:
    from fibsem.drivers.demo.sim_scene import SampleScene, SceneFeature

VOID, ICE, CELL, NUCLEUS, FILM, BAR, PARTICLE = 0, 1, 2, 3, 4, 5, 6
# the pieces of a stack, in the order a point is tested: the first that holds
# it is what the point is made of
SLOTS = ("particle", "nucleus", "cell", "ice", "film", "bar")
MATERIAL = {
    "particle": PARTICLE,
    "nucleus": NUCLEUS,
    "cell": CELL,
    "ice": ICE,
    "film": FILM,
    "bar": BAR,
}
NONE = -1e3  # the top of a piece that is not there

FILM_THICKNESS = 0.05
BAR_THICKNESS = 18.0
ICE_THICKNESS = 2.0  # on the open film
FILM_TOP = -ICE_THICKNESS
BAR_RISE = 1.9  # how far the ice stands up over a bar
ICE_OVER_CELL = 0.8  # the ice left over the top of a cell
ICE_BLANKET = 3.0  # the ice lies over a cell as a dome this smooth
PLATE = 0.4  # an ice plate's thickness
CELL_MAX = 5.5  # the tallest a cell stands
Z_MIN, Z_MAX = -5.0, 12.0  # the slab a view looks through
LEVEL = 0.25  # the tallest a level of that slab may be

# secondary-electron yield, (SEM, FIB): the top of a thing, and its inside
# where a cut or a wall shows it
YIELD = {
    ICE: (1.0, 1.0),
    CELL: (1.0, 1.0),
    NUCLEUS: (1.0, 1.0),
    FILM: (1.05, 1.0),
    BAR: (1.45, 1.4),
    PARTICLE: (0.5, 0.2),
}
INSIDE = {
    ICE: (0.62, 0.6),
    CELL: (0.86, 0.8),
    NUCLEUS: (0.5, 0.48),
    FILM: (1.0, 0.9),
    BAR: (1.3, 1.2),
    PARTICLE: (0.4, 0.2),
}
NOTHING = (0.1, 0.04)
TILT = (0.5, 0.42)  # how strongly a slope changes the yield
BAR_UNDER = (0.22, 0.04)  # how much a bar under thin ice adds
BAR_DEPTH = 5.0  # the ice over which a bar fades
BRIGHTNESS = (118.0, 168.0)  # grey of plain flat ice
CACHE_SIZE = 2  # views kept (one per beam)


# ------------------------------------------------------------------- sample
def _box(f: "SceneFeature", x: np.ndarray, y: np.ndarray, reach: float):
    """The rows and columns of the grid within `reach` um of a feature."""
    cols = np.nonzero(np.abs(x - f.x * UM) < reach)[0]
    rows = np.nonzero(np.abs(y - f.y * UM) < reach)[0]
    if not len(cols) or not len(rows):
        return None
    return slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)


def _radius2(f: "SceneFeature", dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Squared distance from a feature's centre in units of its size, along
    its own axes, with its outline's wobble."""
    sigma = f.sigma * UM
    if f.angle:
        c, s = np.cos(f.angle), np.sin(f.angle)
        dx, dy = dx * c + dy * s, -dx * s + dy * c
    rx, ry = dx / max(sigma, 0.05), dy / max(sigma * f.eccentricity, 0.05)
    if f.kind == "ice":
        # a plate is a hexagon: its distance is the largest of the three
        # face normals'
        return (
            np.maximum(
                np.abs(rx),
                np.maximum(
                    np.abs(0.5 * rx + 0.866 * ry), np.abs(0.5 * rx - 0.866 * ry)
                ),
            )
            ** 2
        )
    r2 = rx * rx + ry * ry
    if f.wobble:
        t = np.arctan2(ry, rx)
        mod = 1.0 + f.wobble * (
            0.6 * np.sin(3 * t + f.wobble_phase)
            + 0.4 * np.sin(5 * t + 2 * f.wobble_phase)
        )
        r2 = r2 / mod**2
    return r2


def columns(scene: "SampleScene", x: np.ndarray, y: np.ndarray, step) -> dict:
    """The stack of pieces at every point of a grid of world points, x (W,)
    and y (P,) in um; `step` is the sampling along (columns, rows). Each
    piece's top is an array (P, W), NONE where there is none."""
    X, Y = np.meshgrid(x, y)
    sx, sy = step
    f32 = np.float32
    seed = int(scene.seed)
    bars, _, rips, rim, beyond = scene.film_masks(X / UM, Y / UM)
    metal = bars | rim
    gone = rips | beyond
    # the ice is one sheet: slow waves, and a broad uneven ridge over each bar
    weight = _gauss(metal.astype(f32), (2.5 / sy, 2.5 / sx))
    swell = np.clip(1 + fbm(X, Y, (90.0, 37.0), (0.35, 0.2), seed + 23), 0.3, 1.8)
    ice = BAR_RISE * swell * weight + fbm(
        X, Y, (210.0, 120.0, 65.0), (0.4, 0.28, 0.1), seed + 11
    )

    shape = X.shape
    dome = np.zeros(shape, dtype=f32)  # how far the cells stand, by outline
    cell_id = np.full(shape, -1, dtype=np.int32)
    n_lo = np.zeros(shape, dtype=f32)
    n_hi = np.full(shape, NONE, dtype=f32)
    plate = np.zeros(shape, dtype=f32)
    bumps = np.zeros(shape, dtype=f32)
    grains = []
    x_lo, x_hi, y_lo, y_hi = x.min(), x.max(), y.min(), y.max()
    for f in scene.features:
        sigma = f.sigma * UM
        reach = (4.0 if f.sharpness < 2 else 2.2) * sigma * (1 + f.wobble) + 1.0
        fx, fy = f.x * UM, f.y * UM
        if fx + reach < x_lo or fx - reach > x_hi:
            continue
        if fy + reach < y_lo or fy - reach > y_hi:
            continue
        box = _box(f, x, y, reach)
        if box is None:
            continue
        if f.kind == "fiducial" or f.part in ("sphere", "frost"):
            grains.append((f, box))
            continue
        r2 = _radius2(f, X[box] - fx, Y[box] - fy)
        if f.kind == "contamination":
            # a speck under the ice: a low bump in the sheet
            bump = min(0.2 * sigma, 0.5) * np.exp(-0.5 * r2**f.sharpness)
            bumps[box] = np.maximum(bumps[box], bump)
        elif f.kind == "ice":
            plate[box] = np.maximum(plate[box], np.exp(-0.5 * r2**f.sharpness))
        elif f.part in ("body", "bud", "neurite"):
            # a cell stands about as tall as it is wide, up to a limit
            tall = min(2.4 * sigma * min(f.eccentricity, 1.0), CELL_MAX)
            top = (tall * np.exp(-0.5 * r2**f.sharpness)).astype(f32)
            higher = top > dome[box]
            cell_id[box] = np.where(higher & (top > 0.05), f.cell_id, cell_id[box])
            dome[box] = np.maximum(dome[box], top)
        elif f.part == "nucleus":
            # an ellipsoid inside the cell, a little off the film
            r2 = r2 / 1.4**2
            half = min(0.8 * sigma * min(f.eccentricity, 1.0), 2.0)
            t = half * np.sqrt(np.clip(1 - r2, 0, None))
            inside = r2 < 1
            n_lo[box] = np.where(inside, FILM_TOP + 0.3 + half - t, n_lo[box])
            n_hi[box] = np.where(inside, FILM_TOP + 0.3 + half + t, n_hi[box])
    in_cell = dome > 0.05
    ice = (
        ice + _gauss(dome, (ICE_BLANKET / sy, ICE_BLANKET / sx)) + PLATE * plate + bumps
    )
    ice = np.minimum(ice, Z_MAX - 3.0).astype(f32)
    cell = np.where(in_cell, np.minimum(dome - ICE_OVER_CELL, ice - 0.3), NONE).astype(
        f32
    )
    n_hi = np.minimum(n_hi, cell - 0.15)

    # particles rest on the ice: each grain a ball
    p_lo = np.full(shape, -NONE, dtype=f32)
    p_hi = np.full(shape, NONE, dtype=f32)
    p_id = np.full(shape, -1, dtype=np.int32)
    for n, (f, box) in enumerate(grains):
        r = f.sigma * UM * (1.0 if f.part == "sphere" else 1.2)
        fx, fy = f.x * UM, f.y * UM
        j = int(np.argmin(np.abs(y - fy)))
        i = int(np.argmin(np.abs(x - fx)))
        zc = float(ice[j, i]) + 0.7 * r
        d2 = r * r - (X[box] - fx) ** 2 - (Y[box] - fy) ** 2
        inside = d2 > 0
        t = np.sqrt(np.clip(d2, 0, None)).astype(f32)
        p_lo[box] = np.where(inside, np.minimum(p_lo[box], zc - t), p_lo[box])
        p_hi[box] = np.where(inside, np.maximum(p_hi[box], zc + t), p_hi[box])
        p_id[box] = np.where(inside, n, p_id[box])
    p_hi = np.minimum(p_hi, Z_MAX - 0.5)

    # where the film is torn, or past the grid, there is nothing
    for a in (ice, cell, n_hi, p_hi):
        a[gone] = NONE
    surface = np.maximum(ice, p_hi)
    return {
        "particle": (p_lo, p_hi),
        "nucleus": (n_lo, n_hi),
        "cell": (FILM_TOP, cell),
        "ice": (FILM_TOP, ice),
        "film": (
            FILM_TOP - FILM_THICKNESS,
            np.where(gone, NONE, FILM_TOP).astype(f32),
        ),
        "bar": (
            FILM_TOP - FILM_THICKNESS - BAR_THICKNESS,
            np.where(metal & ~gone, FILM_TOP - FILM_THICKNESS, NONE).astype(f32),
        ),
        "surface": surface,
        # what a pattern's depth is measured from: the top without its bumps
        "level": sim_relief._smooth_top(
            np.where(gone, sim_relief.VOID, ice),
            (sim_relief.FLOOR_SMOOTH / sy, sim_relief.FLOOR_SMOOTH / sx),
        ).astype(f32),
        "cell_id": cell_id,
        "particle_id": p_id,
        "bar_weight": weight,
    }


def _removed(scene: "SampleScene", X, Y, col: dict) -> list:
    """What each milled pattern removes: per pattern, the rows and columns it
    reaches and, there, the heights (from, to) it takes out of each column,
    with how its face shades. Kept with the columns, so only patterns milled
    since the last call are worked out."""
    done = col.get("removed")
    regions = list(scene.milled)
    if (
        done is None
        or len(done[0]) > len(regions)
        or any(a is not b for a, b in zip(done[0], regions))
    ):
        done = ([], [])
    out = list(done[1])
    surface, level = col["surface"], col["level"]
    for region in regions[len(done[0]) :]:
        a, b = sim_relief._cut_range(scene, region, X, Y, surface)
        # a pattern cuts only as deep as it was set to
        a = np.maximum(a, np.minimum(surface, level - region.depth * UM))
        hit = b > a
        if not hit.any():
            continue
        rows = np.nonzero(hit.any(axis=1))[0]
        cols = np.nonzero(hit.any(axis=0))[0]
        box = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
        a = np.where(hit, a, 1.0)[box].astype(np.float32)
        b = np.where(hit, b, 0.0)[box].astype(np.float32)
        cut = region.beam_cut
        outline = cut.polygon if cut is not None else region.points * UM
        edges = np.linalg.norm(np.roll(outline, -1, axis=0) - outline, axis=1)
        out.append(
            {
                "box": box,
                "from": a,
                "to": b,
                # a narrow cut lets few electrons back out
                "opening": float(np.clip(edges.min() / 9.0, 0.15, 1.0)),
                "fs": None if cut is None else cut.fs,
            }
        )
    col["removed"] = (regions, out)
    return out


def _part(a, sl):
    return a if np.ndim(a) == 0 else a[sl]


def _walk(col: dict, removed: list, z: np.ndarray, shift: np.ndarray, height: int):
    """What each pixel's beam meets first: the level and the material, on
    (height, W). Row r of the image at level q shows sample row r + shift[q];
    the highest level with something solid there is the first the beam meets."""
    surface = col["surface"]
    width = surface.shape[1]
    hit_q = np.full((height, width), -1, dtype=np.int16)
    hit_m = np.zeros((height, width), dtype=np.uint8)
    dz = z[1] - z[0]
    reach = {k: (float(np.min(col[k][0])), float(np.max(col[k][1]))) for k in SLOTS}
    for q, zq in enumerate(z):
        sl = slice(shift[q], shift[q] + height)
        level = np.zeros((height, width), dtype=np.uint8)
        for k in SLOTS[::-1]:
            if zq < reach[k][0] or zq >= reach[k][1]:
                continue
            lo, hi = col[k]
            level[(_part(lo, sl) <= zq) & (zq < hi[sl])] = MATERIAL[k]
        if not level.any():
            continue
        if removed:
            # the top of a column is tested at its true height, so a cut
            # meets the surface along a smooth line
            top = surface[sl]
            zt = np.where((top > zq) & (top - zq < dz), top, zq)
            for cut in removed:
                rows, cols = cut["box"]
                r0, r1 = max(rows.start, sl.start), min(rows.stop, sl.stop)
                if r1 <= r0:
                    continue
                mine = (slice(r0 - sl.start, r1 - sl.start), cols)
                theirs = slice(r0 - rows.start, r1 - rows.start)
                zz = zt[mine]
                level[mine][(cut["from"][theirs] <= zz) & (zz <= cut["to"][theirs])] = (
                    VOID
                )
        solid = level > 0
        hit_q[solid] = q
        hit_m[solid] = level[solid]
    return hit_q, hit_m


# ------------------------------------------------------------------- render
def trace(scene: "SampleScene", frame: ViewFrame, u, w, pix) -> dict:
    """What each pixel's beam meets, over content columns `u` and rows `w`
    (um): `material`, `kind` (0 nothing, 1 a top, 2 a cut face, 3 a wall) and
    height `z`, with what the shading needs. Rows run down the beam's view
    (see `flip`)."""
    fs, turn = max(frame.fs, 1e-3), frame.turn
    side = -1.0 if frame.k < 0 else 1.0
    kabs = abs(frame.k)
    height, width = len(w), len(u)
    # s = Y fs + kabs z runs down the beam's view; Y is the sampling axis on
    # the plane, the beam travelling toward +Y and down. One sample row per
    # image row, and the slab in levels no taller than a pixel row: a level
    # up is then a whole number of rows nearer the beam.
    s_lo = (side * w).min()
    dz = min(pix / kabs, LEVEL) if kabs > 1e-6 else LEVEL
    z = Z_MIN + (np.arange(int(np.ceil((Z_MAX - Z_MIN) / dz))) + 0.5) * dz
    shift = np.round((Z_MAX - z) * kabs / pix).astype(np.int64)
    dy = pix / fs
    Y = (s_lo - Z_MAX * kabs) / fs + np.arange(height + shift.max() + 1) * dy
    xw = turn * u
    yw = side * turn * Y

    key = (
        scene.seed,
        round(float(xw[0]), 4),
        round(float(yw[0]), 4),
        width,
        len(Y),
        round(float(pix), 6),
        round(float(dy), 6),
        len(scene.features),
        id(scene.features),
    )
    cache = _cache(scene)
    col = cache.get(key)
    if col is None:
        col = cache[key] = columns(scene, xw, yw, (pix, dy))
        while len(cache) > CACHE_SIZE:
            cache.popitem(last=False)
    else:
        cache.move_to_end(key)
    removed = []
    if scene.milled:
        removed = _removed(scene, *np.meshgrid(xw, yw), col)
    hit_q, hit_m = _walk(col, removed, z, shift, height)

    seen = hit_q >= 0
    q = np.clip(hit_q, 0, len(z) - 1).astype(np.int64)
    i = np.arange(height)[:, None] + shift[q]
    cols = np.broadcast_to(np.arange(width), (height, width))
    zq = z[q]

    def at(a):
        return a if np.ndim(a) == 0 else a[i, cols]

    surface = at(col["surface"])
    # what stood over the piece met, before milling: nothing (its top shows),
    # something milled away (a cut face), or something still there (a wall)
    z_up = zq + dz
    above = np.zeros((height, width), dtype=bool)
    for k in SLOTS:
        lo, hi = col[k]
        above |= (at(lo) <= z_up) & (z_up < at(hi))
    top_of = np.zeros((height, width), dtype=np.float32)
    gx = np.zeros((height, width), dtype=np.float32)
    gy = np.zeros((height, width), dtype=np.float32)
    for k in ("ice", "particle"):
        m = hit_m == MATERIAL[k]
        if not m.any():
            continue
        hi = col[k][1]
        slope = np.where(hi > NONE / 2, hi, np.nan) if k == "particle" else hi
        d_y, d_x = np.gradient(slope, dy, pix)
        gx[m] = np.nan_to_num(d_x[i[m], cols[m]])
        gy[m] = np.nan_to_num(d_y[i[m], cols[m]])
        top_of[m] = hi[i[m], cols[m]]
    cut_above = np.zeros((height, width), dtype=bool)
    opening = np.ones((height, width), dtype=np.float32)
    lean = np.ones((height, width), dtype=np.float32)
    z_face = np.full((height, width), np.nan, dtype=np.float32)
    z_up = np.where((surface > z_up) & (surface - z_up < dz), surface, z_up)
    for cut in removed:
        rows, cs = cut["box"]
        m = seen & (i >= rows.start) & (i < rows.stop)
        m &= (cols >= cs.start) & (cols < cs.stop)
        if not m.any():
            continue
        a = cut["from"][i[m] - rows.start, cols[m] - cs.start]
        b = cut["to"][i[m] - rows.start, cols[m] - cs.start]
        # a level's rows are rounded, so the piece met may lie a level lower
        here = (a <= z_up[m] + dz) & (z_up[m] <= b)
        where = np.zeros((height, width), dtype=bool)
        where[m] = here
        cut_above |= where
        # the face under a cut lies where the cut ends: its exact height
        z_face[where] = a[here]
        opening[where] = cut["opening"]
        # a face shades by how squarely this beam looks at it
        mill = np.pi / 2 if cut["fs"] is None else np.arcsin(np.clip(cut["fs"], -1, 1))
        lean[where] = abs(np.sin(np.arcsin(min(fs, 1.0)) - mill))
    kind = np.zeros((height, width), dtype=np.uint8)
    kind[seen & ~above] = 1
    kind[seen & above & cut_above] = 2
    kind[seen & above & ~cut_above] = 3
    # a sloping top is met through its side one level down: still a top
    kind[(kind == 3) & np.isin(hit_m, (ICE, PARTICLE))] = 1
    z_hit = np.where(kind == 1, np.minimum(top_of, zq + dz), zq)
    # what a cut face is made of, read at its exact height
    on_face = (kind == 2) & np.isfinite(z_face)
    if on_face.any():
        zf = np.where(on_face, z_face, zq)
        exact = np.zeros((height, width), dtype=np.uint8)
        for k in SLOTS[::-1]:
            lo, hi = col[k]
            exact[(at(lo) <= zf) & (zf < at(hi))] = MATERIAL[k]
        swap = on_face & (exact > 0)
        hit_m = np.where(swap, exact, hit_m)
        z_hit = np.where(swap, zf, z_hit)

    return {
        "material": hit_m,
        "kind": kind,
        "z": z_hit,
        "slope": (gx, gy),
        "surface": surface,
        "bar": at(col["bar_weight"]),
        "opening": opening,
        "lean": lean,
        "view": (fs, kabs),
        "flip": side < 0,  # the image's rows run the other way
    }


def canvas(scene: "SampleScene", beam: str, frame: ViewFrame, u, w, pix):
    """The view over content columns `u` and rows `w` (um, rows increasing
    down the image), shaded, and how much each pixel charges - what
    sim_relief.render lays out and finishes."""
    which = 0 if beam == "SEM" else 1
    hit = trace(scene, frame, u, w, pix)
    hit_m, kind, z_hit = hit["material"], hit["kind"], hit["z"]
    (gx, gy), (fs, kabs) = hit["slope"], hit["view"]
    shade = np.full(kind.shape, NOTHING[which], dtype=np.float32)
    mat_top = np.zeros(kind.shape, dtype=np.float32)
    mat_in = np.zeros(kind.shape, dtype=np.float32)
    for m, v in YIELD.items():
        mat_top[hit_m == m] = v[which]
        mat_in[hit_m == m] = INSIDE[m][which]
    # a top: the more the beam grazes it, the more electrons get out
    cosi = (fs + gy * kabs) / np.sqrt(1 + gx * gx + gy * gy)
    tilt = np.clip(fs / np.clip(cosi, 0.03, None), 0.0, 6.0) ** TILT[which]
    particle = hit_m == PARTICLE
    tilt = np.where(particle, np.clip(tilt, 0.8, 1.6), np.clip(tilt, 0.72, 1.4))
    top = kind == 1
    shade[top] = (mat_top * tilt)[top]
    # a bar under thin ice shows through
    bar = hit["bar"]
    under = top & (hit_m == ICE)
    over_bar = bar * np.exp(-np.clip(z_hit, 0, None) / BAR_DEPTH)
    shade[under] *= (1 + BAR_UNDER[which] * over_bar)[under]
    # a cut face or a wall shows what it was cut from, darker further down
    depth = np.clip(hit["surface"] - z_hit, 0, None)
    inside = mat_in * (0.75 + 0.35 * hit["lean"]) * np.exp(-depth / 7.0)
    shade[kind >= 2] = (inside * hit["opening"])[kind >= 2]
    # hollows are darker: fewer electrons reach the detector from a pit
    zs = np.maximum(np.where(kind > 0, z_hit, Z_MIN), FILM_TOP - 1.0)
    pit = ndi.uniform_filter(zs, max(3, int(4.0 / pix))) - zs
    shade *= np.clip(1 - 0.09 * np.clip(pit, 0, 4), 0.6, 1.0)
    # what charges in the SEM: what stands up off the film, less over a bar
    charge = np.clip((z_hit - 0.6) / 3.0, 0, 1) * top
    charge = (charge + 0.7 * (particle & top)) * (1 - 0.5 * bar)
    charge = ndi.gaussian_filter(np.clip(charge, 0, 1).astype(np.float32), 1.0 / pix)
    charge[~top] = 0
    shade *= BRIGHTNESS[which] / sim_relief.BRIGHTNESS
    if hit["flip"]:
        shade, charge = shade[::-1], charge[::-1]
    return shade, charge


def _cache(scene) -> "OrderedDict":
    cache = getattr(scene, "_solid_cache", None)
    if cache is None:
        cache = OrderedDict()
        object.__setattr__(scene, "_solid_cache", cache)
    return cache
