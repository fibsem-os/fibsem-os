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
from fibsem.drivers.demo.sim_relief import (
    UM,
    ViewFrame,
    _gauss,
    fbm,
    polygon_span,
)

if TYPE_CHECKING:
    from fibsem.drivers.demo.sim_scene import SampleScene, SceneFeature

VOID, ICE, CELL, NUCLEUS, FILM, BAR, PARTICLE, COAT = 0, 1, 2, 3, 4, 5, 6, 7
DROPLET, VESICLE = 8, 9  # what lies in a cell's cytoplasm
# the pieces of a stack, in the order a point is tested: the first that holds
# it is what the point is made of
SLOTS = ("coat", "particle", "nucleus", "organelle", "cell", "ice", "film", "bar")
MATERIAL = {
    "coat": COAT,
    "particle": PARTICLE,
    "nucleus": NUCLEUS,
    "organelle": DROPLET,  # or a vesicle: see the stack's "organelle_is"
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
ICE_CLOSE = 0.35  # and follows each cell's own shape this closely
PLATE = 0.4  # an ice plate's thickness
CELL_MAX = 5.5  # the tallest a cell stands
CLUMP_REACH = 12.0  # how far a clump of grains can reach from its speck
RPE_RADIUS = 1.1  # an epithelial cell's radius, in its feature's size
NUCLEUS_RADIUS = 1.4  # and a nucleus's
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
    COAT: (1.08, 1.3),
    DROPLET: (1.0, 1.0),
    VESICLE: (1.0, 1.0),
}
INSIDE = {
    ICE: (0.62, 0.6),
    CELL: (0.86, 0.8),
    NUCLEUS: (0.5, 0.48),
    FILM: (1.0, 0.9),
    BAR: (1.3, 1.2),
    PARTICLE: (0.4, 0.2),
    COAT: (1.9, 2.0),  # edge-on, the metal in it is the brightest band on a face
    DROPLET: (0.34, 0.4),  # lipid: the darkest thing on a face
    VESICLE: (1.08, 0.95),
}
NOTHING = (0.1, 0.04)
TILT = (0.5, 0.42)  # how strongly a slope changes the yield
BAR_UNDER = (0.22, 0.12)  # how much a bar under thin ice adds
BAR_DEPTH = 5.0  # the ice over which a bar fades
HOLE_UNDER = (0.2, 0.1)  # how much darker thin ice is with no film under it
CRACK = 0.9  # um; how deep a crack in the film grooves the ice over it
ORGANELLE_STEP = 0.5  # um; a view coarser than this shows no organelles
BRIGHTNESS = (118.0, 168.0)  # grey of plain flat ice
RIM = (1.5, 1.2)  # how bright the rim round a milled edge is
RIM_WIDTH = 1.1  # um
CURTAIN = 0.55  # how strongly curtaining streaks a cut face
GROOVE = 0.14  # um; how deep curtaining grooves a cut
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


def _clump(f: "SceneFeature", seed: int) -> list:
    """A speck of contamination as a ragged clump of grains resting on the
    ice: (x, y, radius, how far it is lifted) each, um. Most are a few small
    grains; one in ten is a big heap. Seeded by where the speck is."""
    fx, fy = f.x * UM, f.y * UM
    rng = np.random.default_rng(
        [seed & 0x7FFFFFFF, int(abs(fx) * 100) % 2**31, int(abs(fy) * 100) % 2**31]
    )
    big = rng.random() < 0.1
    n = int(rng.integers(18, 60)) if big else int(rng.integers(2, 7))
    size = rng.uniform(0.7, 1.5) if big else rng.uniform(0.2, 0.65)
    spread = size * (1.7 if big else 0.7)
    # longer one way than the other
    axis, squash = rng.uniform(0, np.pi), rng.uniform(0.35, 1.0)
    grains = []
    for k in range(n):
        a, b = rng.normal(0, spread), rng.normal(0, spread * squash)
        r = size * rng.uniform(0.2, 0.8) ** 1.5 + 0.12
        lifted = rng.uniform(0.3, 1.2) * size if big and k > n // 2 else 0.0
        grains.append(
            (
                fx + a * np.cos(axis) - b * np.sin(axis),
                fy + a * np.sin(axis) + b * np.cos(axis),
                r,
                lifted,
            )
        )
    return grains


def _lobes(f: "SceneFeature", seed: int) -> list:
    """A nucleus as one to three overlapping ellipsoids, each (x offset, y
    offset um, size, height of its underside above the film um): seeded by
    the cell, so no two nuclei are alike and each stays the same."""
    rng = np.random.default_rng([seed & 0x7FFFFFFF, 131, int(f.cell_id) & 0x7FFFFFFF])
    sigma = f.sigma * UM
    size, rise = rng.uniform(0.8, 1.1), rng.uniform(0.2, 0.9)
    lobes = [(0.0, 0.0, size, rise)]
    if rng.random() < 0.35:
        for _ in range(int(rng.integers(1, 3))):
            t, d = rng.uniform(0, 2 * np.pi), rng.uniform(0.6, 0.9) * sigma
            lobes.append(
                (
                    d * np.cos(t),
                    d * np.sin(t),
                    size * rng.uniform(0.5, 0.75),
                    rise + rng.uniform(0.0, 0.5),
                )
            )
    return lobes


def _inclusions(scene: "SampleScene", f: "SceneFeature") -> list:
    """What lies in a cell's cytoplasm, as balls (x, y, z of the centre,
    radius um, material): its lipid droplets, the ones the FM labels, and a
    scatter of vesicles. Seeded by the cell: some have many, some almost none."""
    out = [
        (px * UM, py * UM, FILM_TOP + h * UM, r * UM, DROPLET)
        for px, py, h, r in scene._fm_cell_points(f, "droplets")
    ]
    rng = np.random.default_rng(
        [int(scene.seed) & 0x7FFFFFFF, 137, int(f.cell_id) & 0x7FFFFFFF]
    )
    n = int(rng.integers(300, 5000) * rng.random() ** 1.5)
    tall = max(float(f.height or 0.0) * UM, 1.5)
    reach = 0.9 * f.sigma * UM
    r = reach * np.sqrt(rng.uniform(0, 1, n))
    t = rng.uniform(0, 2 * np.pi, n)
    z = FILM_TOP + rng.uniform(0.3, 0.8 * tall, n)
    size = rng.uniform(0.12, 0.4, n)
    fx, fy = f.x * UM, f.y * UM
    out.extend(
        (fx + r[k] * np.cos(t[k]), fy + r[k] * np.sin(t[k]), z[k], size[k], VESICLE)
        for k in range(n)
    )
    return out


def columns(scene: "SampleScene", x: np.ndarray, y: np.ndarray, step) -> dict:
    """The stack of pieces at every point of a grid of world points, x (W,)
    and y (P,) in um; `step` is the sampling along (columns, rows). Each
    piece's top is an array (P, W), NONE where there is none."""
    X, Y = np.meshgrid(x, y)
    sx, sy = step
    f32 = np.float32
    seed = int(scene.seed)
    bars, holes, rips, rim, beyond = scene.film_masks(X / UM, Y / UM)
    metal = bars | rim
    # a broken hole has no ice across it; a whole one does
    gone = rips | beyond | (scene.broken_hole_mask(X / UM, Y / UM) & ~metal)
    holes = holes & ~gone
    crack = scene.crack_depth(X / UM, Y / UM) * ~metal
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
    lift = np.zeros(shape, dtype=f32)  # the ice over the smooth-domed cells
    grains = []
    inclusions = []
    fine = max(sx, sy) <= ORGANELLE_STEP
    x_lo, x_hi, y_lo, y_hi = x.min(), x.max(), y.min(), y.max()
    for f in scene.features:
        sigma = f.sigma * UM
        reach = (4.0 if f.sharpness < 2 else 2.2) * sigma * (1 + f.wobble) + 1.0
        if f.kind == "contamination":
            reach = max(reach, CLUMP_REACH)
        fx, fy = f.x * UM, f.y * UM
        if fx + reach < x_lo or fx - reach > x_hi:
            continue
        if fy + reach < y_lo or fy - reach > y_hi:
            continue
        box = _box(f, x, y, reach)
        if box is None:
            continue
        if f.kind == "fiducial" or f.part in ("sphere", "frost"):
            grains.append((fx, fy, sigma * (1.0 if f.part == "sphere" else 1.2), 0.0))
            continue
        if f.kind == "contamination":
            grains.extend(_clump(f, seed))
            continue
        if f.part in ("body", "bud") and f.kind == "cell" and fine:
            inclusions.extend(_inclusions(scene, f))
        r2 = _radius2(f, X[box] - fx, Y[box] - fy)
        if f.kind == "ice":
            plate[box] = np.maximum(plate[box], np.exp(-0.5 * r2**f.sharpness))
        elif f.part == "body" and scene.cell_type == "rpe":
            # an epithelial cell: a smooth dome, the ice over it a little wider
            rr = np.sqrt(r2) / RPE_RADIUS
            tall = f.height * UM
            top = (tall * np.cos(0.5 * np.pi * np.minimum(rr, 1)) ** 2).astype(f32)
            higher = top > dome[box]
            cell_id[box] = np.where(higher & (top > 0.05), f.cell_id, cell_id[box])
            dome[box] = np.maximum(dome[box], top)
            over = tall * np.cos(0.5 * np.pi * np.minimum(rr / 1.2, 1)) ** 2
            lift[box] = np.maximum(lift[box], over)
        elif f.part in ("body", "bud", "neurite"):
            # a cell stands about as tall as it is wide, up to a limit
            tall = min(2.4 * sigma * min(f.eccentricity, 1.0), CELL_MAX)
            top = (tall * np.exp(-0.5 * r2**f.sharpness)).astype(f32)
            higher = top > dome[box]
            cell_id[box] = np.where(higher & (top > 0.05), f.cell_id, cell_id[box])
            dome[box] = np.maximum(dome[box], top)
        elif f.part == "nucleus":
            # an ellipsoid inside the cell, a little off the film
            # no two alike: its size, how high it sits, and sometimes lobes
            for ox, oy, size, rise in _lobes(f, seed):
                r2 = _radius2(f, X[box] - fx - ox, Y[box] - fy - oy)
                r2 = r2 / (NUCLEUS_RADIUS * size) ** 2
                half = min(0.8 * sigma * min(f.eccentricity, 1.0), 2.0) * size
                t = half * np.sqrt(np.clip(1 - r2, 0, None))
                inside = r2 < 1
                mid = FILM_TOP + rise + half
                first = inside & (n_hi[box] < NONE / 2)
                n_lo[box] = np.where(
                    inside,
                    np.where(first, mid - t, np.minimum(n_lo[box], mid - t)),
                    n_lo[box],
                )
                n_hi[box] = np.where(inside, np.maximum(n_hi[box], mid + t), n_hi[box])
    in_cell = dome > 0.05
    blanket = _gauss(dome, (ICE_BLANKET / sy, ICE_BLANKET / sx))
    # the ice lies close over each cell, and in a broad swell over a group
    close = _gauss(dome, (ICE_CLOSE / sy, ICE_CLOSE / sx)) + ICE_OVER_CELL * (
        _gauss(in_cell.astype(f32), (2 * ICE_CLOSE / sy, 2 * ICE_CLOSE / sx))
    )
    ice = ice + np.where(lift > 0, lift, np.maximum(blanket, close)) + PLATE * plate
    # a crack in the film is a groove in thin ice, open along its middle
    thin = np.exp(-np.clip(ice, 0, None) / 1.5)
    ice = ice - CRACK * crack * thin
    gone = gone | ((crack * thin > 0.8) & ~in_cell)
    ice = np.minimum(ice, Z_MAX - 3.0).astype(f32)
    cell = np.where(in_cell, np.minimum(dome - ICE_OVER_CELL, ice - 0.3), NONE).astype(
        f32
    )
    n_hi = np.minimum(n_hi, cell - 0.15)

    # droplets and vesicles in the cytoplasm: each a ball, kept inside its cell
    o_lo = np.full(shape, -NONE, dtype=f32)
    o_hi = np.full(shape, NONE, dtype=f32)
    o_is = np.full(shape, DROPLET, dtype=np.uint8)
    for fx, fy, zc, r, what in inclusions:
        if fx + r < x_lo or fx - r > x_hi or fy + r < y_lo or fy - r > y_hi:
            continue
        cols = np.nonzero(np.abs(x - fx) < r)[0]
        rows = np.nonzero(np.abs(y - fy) < r)[0]
        if not len(cols) or not len(rows):
            continue
        box = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
        d2 = r * r - (X[box] - fx) ** 2 - (Y[box] - fy) ** 2
        t = np.sqrt(np.clip(d2, 0, None)).astype(f32)
        inside = (d2 > 0) & (zc + t < cell[box] - 0.1) & (o_hi[box] < NONE / 2)
        o_lo[box] = np.where(inside, np.maximum(zc - t, FILM_TOP + 0.02), o_lo[box])
        o_hi[box] = np.where(inside, zc + t, o_hi[box])
        o_is[box] = np.where(inside, what, o_is[box])

    # particles rest on the ice: each grain a ball
    p_lo = np.full(shape, -NONE, dtype=f32)
    p_hi = np.full(shape, NONE, dtype=f32)
    p_id = np.full(shape, -1, dtype=np.int32)
    for n, (fx, fy, r, lifted) in enumerate(grains):
        cols = np.nonzero(np.abs(x - fx) < r)[0]
        rows = np.nonzero(np.abs(y - fy) < r)[0]
        if not len(cols) or not len(rows):
            continue
        box = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
        zc = float(ice[rows[len(rows) // 2], cols[len(cols) // 2]]) + 0.75 * r + lifted
        d2 = r * r - (X[box] - fx) ** 2 - (Y[box] - fy) ** 2
        inside = d2 > 0
        t = np.sqrt(np.clip(d2, 0, None)).astype(f32)
        p_lo[box] = np.where(inside, np.minimum(p_lo[box], zc - t), p_lo[box])
        p_hi[box] = np.where(inside, np.maximum(p_hi[box], zc + t), p_hi[box])
        p_id[box] = np.where(inside, n, p_id[box])
    p_hi = np.minimum(p_hi, Z_MAX - 0.5)

    # where the film is torn, or past the grid, there is nothing
    for a in (ice, cell, n_hi, o_hi, p_hi):
        a[gone] = NONE
    surface = np.maximum(ice, p_hi)
    level = sim_relief._smooth_top(
        np.where(gone, sim_relief.VOID, ice),
        (sim_relief.FLOOR_SMOOTH / sy, sim_relief.FLOOR_SMOOTH / sx),
    )
    gis = scene.gis_thickness * UM
    coat = np.full(shape, NONE, dtype=f32)
    if gis > 0:
        # the GIS lays a coat over everything, filling the fine detail in
        soft = max(gis, 0.3)
        smooth = sim_relief._smooth_top(
            np.where(gone, sim_relief.VOID, surface), (soft / sy, soft / sx)
        )
        coat = np.where(gone, NONE, np.maximum(surface, smooth) + gis).astype(f32)
        level = level + gis
    return {
        "coat": (surface, coat),
        "particle": (p_lo, p_hi),
        "nucleus": (n_lo, n_hi),
        "organelle": (o_lo, o_hi),
        "organelle_is": o_is,
        "cell": (FILM_TOP, cell),
        "ice": (FILM_TOP, ice),
        "film": (
            FILM_TOP - FILM_THICKNESS,
            np.where(gone | holes, NONE, FILM_TOP).astype(f32),
        ),
        "bar": (
            FILM_TOP - FILM_THICKNESS - BAR_THICKNESS,
            np.where(metal & ~gone, FILM_TOP - FILM_THICKNESS, NONE).astype(f32),
        ),
        "surface": np.maximum(surface, coat),
        # what a pattern's depth is measured from: the top without its bumps
        "level": level.astype(f32),
        "cell_id": cell_id,
        "particle_id": p_id,
        "bar_weight": weight,
        # thin ice across a hole in the film, with no cell on it
        "hole": (holes & ~in_cell) * np.exp(-np.clip(ice, 0, None) / 2.0).astype(f32),
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
    amp = _curtaining(scene)
    x = X[0]
    grain = col["particle"][1] > NONE / 2
    for region in regions[len(done[0]) :]:
        cut = region.beam_cut
        pieces = getattr(region, "face", None)
        if pieces is not None:
            # damage: shapes on the lamella's face, through its thickness
            a, b = sim_relief._cut_range(scene, region, X, Y)
            inside = np.zeros(X.shape, dtype=bool)
            for piece in pieces:
                inside |= scene._polygon_mask(piece, X, Y)
            b = np.where(inside, b, a - 1.0)
        else:
            a, b = sim_relief._cut_range(scene, region, X, Y, surface)
            # a pattern cuts only as deep as it was set to
            a = np.maximum(a, np.minimum(surface, level - region.depth * UM))
            if amp > 0 and cut is not None and abs(cut.k) > 0.05:
                # curtaining grooves the cut: its edges wander, more further down
                grow = GROOVE * amp / abs(cut.k)
                a = a + curtain(scene, x) * grow * np.clip((surface - a) / 2.5, 0, 1)
                b = b - curtain(scene, x, 37.3) * grow * np.clip(
                    (surface - b) / 2.5, 0, 1
                )
        hit = b > a
        if not hit.any():
            continue
        rows = np.nonzero(hit.any(axis=1))[0]
        cols = np.nonzero(hit.any(axis=0))[0]
        box = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
        # a particle the milling beam passed through leaves a stronger streak
        hard = (grain[box] & hit[box]).any(axis=0)
        a = np.where(hit, a, 1.0)[box].astype(np.float32)
        b = np.where(hit, b, 0.0)[box].astype(np.float32)
        outline = cut.polygon if cut is not None else region.points * UM
        edges = np.linalg.norm(np.roll(outline, -1, axis=0) - outline, axis=1)
        out.append(
            {
                "box": box,
                "from": a,
                "to": b,
                # a narrow cut lets few electrons back out
                "opening": 0.3
                if pieces is not None
                else float(np.clip(edges.min() / 9.0, 0.15, 1.0)),
                "outline": outline,
                "view": None if cut is None else (cut.turn, cut.fs, cut.k),
                "halo": region.halo * UM,
                "rim": pieces is None,
                "hard": hard,
            }
        )
    col["removed"] = (regions, out)
    return out


def _curtaining(scene) -> float:
    """How strong curtaining is now: a GIS coat takes most of it away."""
    return float(scene.curtaining * np.exp(-scene.gis_thickness * UM / 0.5))


def curtain(scene, x, shift: float = 0.0) -> np.ndarray:
    """Curtaining across a cut, -1..1 at each world x (um): streaks of
    uneven milling, fixed to the sample."""
    seed = int(scene.seed) % 1000

    def noise(u, scale, k):
        i = np.floor(u / scale)
        f = u / scale - i
        f = f * f * (3 - 2 * f)
        h0 = np.modf(np.sin((i + k) * 12.9898 + seed) * 43758.5453)[0]
        h1 = np.modf(np.sin((i + 1 + k) * 12.9898 + seed) * 43758.5453)[0]
        return np.abs(h0) * (1 - f) + np.abs(h1) * f

    u = np.asarray(x, dtype=np.float64) + shift
    n = 0.55 * noise(u, 0.22, 3) + 0.3 * noise(u, 0.6, 17) + 0.15 * noise(u, 1.7, 41)
    return np.clip((n - 0.5) * 3.2, -1.0, 1.0)


def _part(a, sl):
    return a if np.ndim(a) == 0 else a[sl]


def _walk(col: dict, removed: list, z, shift, rows: slice, cols: slice):
    """What each pixel's beam meets first: the level and the material, over
    image rows `rows` and columns `cols`. Row r of the image at level q shows
    sample row r + shift[q]; the highest level with something solid there is
    the first the beam meets."""
    height, width = rows.stop - rows.start, cols.stop - cols.start
    hit_q = np.full((height, width), -1, dtype=np.int16)
    hit_m = np.zeros((height, width), dtype=np.uint8)
    dz = z[1] - z[0]
    surface = col["surface"]
    pieces = []
    for k in SLOTS[::-1]:
        lo, hi = col[k]
        lo = lo if np.ndim(lo) == 0 else lo[:, cols]
        hi = hi[:, cols]
        what = col["organelle_is"][:, cols] if k == "organelle" else MATERIAL[k]
        pieces.append((float(np.min(lo)), float(np.max(hi)), lo, hi, what))
    level = np.zeros((height, width), dtype=np.uint8)
    for q, zq in enumerate(z):
        sl = slice(shift[q] + rows.start, shift[q] + rows.stop)
        level[:] = VOID
        any_piece = False
        for low, high, lo, hi, what in pieces:
            if zq < low or zq >= high:
                continue
            m = (_part(lo, sl) <= zq) & (zq < hi[sl])
            np.copyto(level, what if np.ndim(what) == 0 else what[sl], where=m)
            any_piece = True
        if not any_piece:
            continue
        if removed:
            # the top of a column is tested at its true height, so a cut
            # meets the surface along a smooth line
            for cut in removed:
                r, c = cut["box"]
                r0, r1 = max(r.start, sl.start), min(r.stop, sl.stop)
                c0, c1 = max(c.start, cols.start), min(c.stop, cols.stop)
                if r1 <= r0 or c1 <= c0:
                    continue
                mine = (
                    slice(r0 - sl.start, r1 - sl.start),
                    slice(c0 - cols.start, c1 - cols.start),
                )
                theirs = (
                    slice(r0 - r.start, r1 - r.start),
                    slice(c0 - c.start, c1 - c.start),
                )
                top = surface[r0:r1, c0:c1]
                zz = np.where((top > zq) & (top - zq < dz), top, zq)
                gone = (cut["from"][theirs] <= zz) & (zz <= cut["to"][theirs])
                level[mine][gone] = VOID
        solid = level > 0
        np.putmask(hit_q, solid, q)
        np.copyto(hit_m, level, where=solid)
    return hit_q, hit_m


def _seen(col: dict, removed: list, z, shift, height: int):
    """The walk over a whole view. The sample as it was before any milling is
    walked once and kept with its columns; a milled frame walks again only
    the part of the image the cuts can show in."""
    width = col["surface"].shape[1]
    kept = col.get("walk")
    if kept is None or kept[0] != (height, len(z)):
        whole = _walk(col, [], z, shift, slice(0, height), slice(0, width))
        kept = col["walk"] = ((height, len(z)), whole)
    hit_q, hit_m = kept[1]
    if not removed:
        return hit_q, hit_m
    hit_q, hit_m = hit_q.copy(), hit_m.copy()
    for cut in removed:
        r, c = cut["box"]
        # sample row s shows in image row s - shift[q], for every level q
        rows = slice(
            max(0, r.start - int(shift.max())), min(height, r.stop - int(shift.min()))
        )
        if rows.stop <= rows.start:
            continue
        cut["window"] = (rows, c)
    # cuts that share columns are walked together, as one window
    todo = sorted(
        (cut["window"] for cut in removed if "window" in cut), key=lambda w: w[1].start
    )
    merged = []
    for rows, c in todo:
        if merged and c.start <= merged[-1][1].stop:
            last = merged[-1]
            merged[-1] = (
                slice(min(last[0].start, rows.start), max(last[0].stop, rows.stop)),
                slice(last[1].start, max(last[1].stop, c.stop)),
            )
        else:
            merged.append((rows, c))
    for rows, c in merged:
        q, m = _walk(col, removed, z, shift, rows, c)
        hit_q[rows, c], hit_m[rows, c] = q, m
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
        round(scene.gis_thickness * UM, 4),
        round(_curtaining(scene), 4),
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
    hit_q, hit_m = _seen(col, removed, z, shift, height)

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
    for k in ("ice", "particle", "coat"):
        m = hit_m == MATERIAL[k]
        if not m.any():
            continue
        hi = col[k][1]
        slope = hi if k == "ice" else np.where(hi > NONE / 2, hi, np.nan)
        d_y, d_x = np.gradient(slope, dy, pix)
        gx[m] = np.nan_to_num(d_x[i[m], cols[m]])
        gy[m] = np.nan_to_num(d_y[i[m], cols[m]])
        top_of[m] = hi[i[m], cols[m]]
    cut_above = np.zeros((height, width), dtype=bool)
    opening = np.ones((height, width), dtype=np.float32)
    lean = np.ones((height, width), dtype=np.float32)
    z_face = np.full((height, width), np.nan, dtype=np.float32)
    z_up = np.where((surface > z_up) & (surface - z_up < dz), surface, z_up)
    open_to = np.full((height, width), np.nan, dtype=np.float32)
    across = np.zeros((height, width), dtype=np.float32)

    def over(cut):
        """The pixels a cut reaches, and what it removes above each."""
        rows, cs = cut["box"]
        m = seen & (i >= rows.start) & (i < rows.stop)
        m &= (cols >= cs.start) & (cols < cs.stop)
        a = cut["from"][i[m] - rows.start, cols[m] - cs.start]
        b = cut["to"][i[m] - rows.start, cols[m] - cs.start]
        return m, a, b

    for cut in removed:
        m, a, b = over(cut)
        if not m.any():
            continue
        # a level's rows are rounded, so the piece met may lie a level lower
        here = (a <= z_up[m] + dz) & (z_up[m] <= b)
        where = np.zeros((height, width), dtype=bool)
        where[m] = here
        cut_above |= where
        # the face under a cut lies where the cut ends: its exact height
        z_face[where] = a[here]
        open_to[where] = b[here]
        opening[where] = cut["opening"]
        # a face shades by how squarely this beam looks at it
        mill = np.pi / 2
        if cut["view"] is not None and cut["rim"]:
            mill = np.arcsin(np.clip(cut["view"][1], -1, 1))
            across[where] = abs(cut["view"][2])
        lean[where] = abs(np.sin(np.arcsin(min(fs, 1.0)) - mill))
    # patterns milled side by side make one trench: how open a face is to
    # the detector goes by all that was removed over it
    for cut in removed + removed:
        m, a, b = over(cut)
        if not m.any() or not cut["rim"]:
            continue
        more = (a <= open_to[m] + dz) & (b > open_to[m])
        where = np.zeros((height, width), dtype=bool)
        where[m] = more
        open_to[where] = b[more]
    wide = across > 0
    opening[wide] = np.clip(((open_to - z_face) * across)[wide] / 9.0, 0.15, 1.0)
    kind = np.zeros((height, width), dtype=np.uint8)
    kind[seen & ~above] = 1
    kind[seen & above & cut_above] = 2
    kind[seen & above & ~cut_above] = 3
    # a sloping top is met through its side one level down: still a top
    kind[(kind == 3) & np.isin(hit_m, (ICE, PARTICLE, COAT))] = 1
    z_hit = np.where(kind == 1, np.minimum(top_of, zq + dz), zq)
    # what a cut face is made of, read at its exact height
    on_face = (kind == 2) & np.isfinite(z_face)
    if on_face.any():
        zf = np.where(on_face, z_face, zq)
        exact = np.zeros((height, width), dtype=np.uint8)
        for k in SLOTS[::-1]:
            lo, hi = col[k]
            m = (at(lo) <= zf) & (zf < at(hi))
            exact[m] = at(col["organelle_is"])[m] if k == "organelle" else MATERIAL[k]
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
        "hole": at(col["hole"]),
        "opening": opening,
        "lean": lean,
        "view": (fs, kabs),
        "x": xw,
        "y": yw[i],
        "removed": removed,
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
    if which == 0 and scene.sputter_thickness > 0:
        # a sputtered metal coat gives a little more signal
        mat_top = mat_top * (1 + 0.12 * (1 - np.exp(-scene.sputter_thickness / 5e-9)))
    shade[top] = (mat_top * tilt)[top]
    # a bar under thin ice shows through
    bar = hit["bar"]
    under = top & np.isin(hit_m, (ICE, COAT))
    over_bar = bar * np.exp(-np.clip(z_hit, 0, None) / BAR_DEPTH)
    shade[under] *= (1 + BAR_UNDER[which] * over_bar)[under]
    shade[under] *= (1 - HOLE_UNDER[which] * hit["hole"])[under]
    # a cut face or a wall shows what it was cut from, darker further down
    depth = np.clip(hit["surface"] - z_hit, 0, None)
    inside = mat_in * (0.75 + 0.35 * hit["lean"]) * np.exp(-depth / 7.0)
    removed, x = hit["removed"], hit["x"]
    amp = _curtaining(scene)
    if amp > 0 and removed:
        # curtaining: streaks down a cut face, each its own length
        hard = np.zeros(len(x), dtype=np.float32)
        for cut in removed:
            cs = cut["box"][1]
            hard[cs] = np.maximum(hard[cs], cut["hard"])
        weight = 1 + 1.5 * ndi.gaussian_filter1d(hard, max(1.0, 0.15 / pix))
        run = np.exp(-depth / (2.5 + 7.0 * np.abs(curtain(scene, x, 91.7))))
        streak = CURTAIN * amp * curtain(scene, x) * weight
        inside = inside * np.clip(1 + streak * run, 0.25, 2.2)
    shade[kind >= 2] = (inside * hit["opening"])[kind >= 2]
    if removed:
        shade[top] *= (1 + RIM[which] * _rim(hit, pix))[top]
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


def _rim(hit: dict, pix: float) -> np.ndarray:
    """The glow round milled edges, 0..1.9 per pixel: electrons escape from
    an edge too. Measured on the surface, from each pattern's outline as its
    milling beam saw it; a spot burn's ring is narrower and brighter."""
    x, y, z = hit["x"], hit["y"], hit["z"]
    rim = np.zeros(z.shape, dtype=np.float32)
    idx = np.arange(len(x))
    for cut in hit["removed"]:
        if not cut["rim"]:
            continue
        width = 2 * cut["halo"] if cut["halo"] > 0 else RIM_WIDTH
        if cut["view"] is None:
            u, scale = x, 1.0
        else:
            turn, fs, k = cut["view"]
            u, scale = turn * x, 1.0 / max(abs(fs), 0.05)
        lo, hi = polygon_span(cut["outline"], u)
        ok = lo <= hi
        if not ok.any():
            continue
        first, last = idx[ok][0], idx[ok][-1]
        n = max(1, int(round(3 * width / pix)))
        sl = slice(max(0, first - n), min(len(x), last + n + 1))
        w = y[:, sl] if cut["view"] is None else turn * y[:, sl] * fs + k * z[:, sl]
        edge = np.clip(idx[sl], first, last)
        dw = np.maximum(lo[edge][None, :] - w, w - hi[edge][None, :])
        du = np.maximum(first - idx[sl], idx[sl] - last) * pix
        # a step in the milling view is 1 / sin longer on the surface
        d = np.hypot(np.clip(dw, 0, None) * scale, np.clip(du, 0, None)[None, :])
        glow = np.clip(1.8 * np.exp(-d / (0.5 * width)), 0, 1.25)
        glow = np.where(d < 3 * width, glow * (1.5 if cut["halo"] > 0 else 1.0), 0)
        rim[:, sl] = np.maximum(rim[:, sl], glow)
    return rim


def top(scene: "SampleScene", x: np.ndarray, y: np.ndarray, step) -> np.ndarray:
    """The height of the sample's top at a grid of world points (um), far
    below where there is nothing."""
    surface = columns(scene, x, y, step)["surface"]
    return np.where(surface > NONE / 2, surface, sim_relief.VOID)


def damage(scene, kind: str, severity: int, ends, slope: float, rng) -> list:
    """Damage to a thin lamella as shapes on its face, seen from above (world
    um): jagged bites out of its free edges, wandering cracks, or a torn hole
    with cracks running from it. `ends` are the world points of the two ends
    of the lamella's middle line on the plane z = 0; the lamella rises by
    `slope` per um of y from there."""
    (xa, ya), (xb, yb) = ends
    ts = np.arange(-60.0, 60.0, 0.05)

    def line(x):
        return ya + (yb - ya) * (x - xa) / (xb - xa) if xb != xa else ya

    def edges(x):
        # where the lamella meets the top (its front) and the film (its back)
        ys = line(x) + ts
        height = slope * ts
        surface = columns(scene, np.array([x]), ys, (0.05, 0.05))["surface"][:, 0]
        on = np.nonzero((height <= surface) & (height >= FILM_TOP))[0]
        if not len(on):
            return 0.0, 0.0
        front = on[np.argmax(height[on])]
        back = on[np.argmin(height[on])]
        return ts[front], ts[back]

    x0, x1 = min(xa, xb), max(xa, xb)
    width = x1 - x0
    front, back = edges(0.5 * (x0 + x1))
    inward = 1.0 if back >= front else -1.0

    def span(x):
        f, b = edges(x)
        return inward * f, inward * b

    pieces = []
    if kind == "ragged":
        for _ in range(9 * severity):
            x = rng.uniform(x0, x1)
            f, b = span(x)
            base, into = (f, 1) if rng.uniform() < 0.7 else (b, -1)
            half = rng.uniform(0.15, 0.9)
            deep = rng.uniform(0.4, 1.6) * severity * (rng.uniform() ** 1.5 + 0.25)
            skew = rng.uniform(-1, 1) * half
            pieces.append(
                np.array(
                    [
                        (x - half, base - into * 0.6),
                        (x + half, base - into * 0.6),
                        (x + skew + 0.25 * half, base + into * 0.8 * deep),
                        (x + skew, base + into * deep),
                    ]
                )
            )
            if rng.uniform() < 0.35:
                heading = (0 if into > 0 else np.pi) + rng.normal(0, 0.5)
                length = rng.uniform(1.0, 3.0) * severity
                pieces += _crack(
                    rng, x + skew, base + into * deep, heading, length, 0.08
                )
    elif kind == "crack":
        for _ in range(severity):
            x = rng.uniform(x0 + 0.15 * width, x1 - 0.15 * width)
            f, b = span(x)
            length = rng.uniform(0.5, 0.95) * (b - f)
            pieces += _crack(rng, x, f - 0.5, rng.normal(0, 0.25), length)
    else:
        x = rng.uniform(x0 + 0.3 * width, x1 - 0.3 * width)
        f, b = span(x)
        d = f + rng.uniform(0.3, 0.6) * (b - f)
        size = min(1.3 * severity, 0.3 * width)
        for _ in range(5 + 2 * severity):
            a, r = rng.uniform(0, 2 * np.pi), rng.uniform(0, 1.0) * size
            pieces.append(
                _blob(
                    rng,
                    x + r * np.cos(a),
                    d + 1.6 * r * np.sin(a),
                    rng.uniform(0.35, 0.8) * size,
                )
            )
        for _ in range(1 + severity):
            a = rng.uniform(0, 2 * np.pi)
            pieces += _crack(
                rng,
                x + 0.6 * size * np.sin(a),
                d + size * np.cos(a),
                a,
                rng.uniform(2.0, 6.0),
                0.12,
            )
    # from along-the-lamella back to world y
    return [
        np.stack([p[:, 0], line(p[:, 0]) + inward * p[:, 1]], axis=1) for p in pieces
    ]


def _blob(rng, x, d, radius, n=7):
    """An irregular convex piece round a point."""
    a = np.sort(rng.uniform(0, 2 * np.pi, n))
    r = radius * rng.uniform(0.55, 1.0, n)
    return np.stack([x + r * np.cos(a), d + r * np.sin(a)], axis=1)


def _crack(rng, x, d, heading, length, width=0.22):
    """A crack as thin pieces: it wanders, narrows to a tip, and may fork."""
    pieces = []
    steps = max(3, int(length / 0.9))
    for i in range(steps):
        step = rng.uniform(0.5, 1.3) * length / steps
        heading += rng.normal(0, 0.45)
        x1, d1 = x + step * np.sin(heading), d + step * np.cos(heading)
        w0 = width * (1 - i / steps) + 0.03
        w1 = width * (1 - (i + 1) / steps) + 0.03
        nx, nd = np.cos(heading), -np.sin(heading)
        pieces.append(
            np.array(
                [
                    (x - nx * w0, d - nd * w0),
                    (x + nx * w0, d + nd * w0),
                    (x1 + nx * w1, d1 + nd * w1),
                    (x1 - nx * w1, d1 - nd * w1),
                ]
            )
        )
        if i == steps // 3 and length > 4 and rng.uniform() < 0.7:
            turn = rng.choice([-1, 1]) * rng.uniform(0.6, 1.1)
            pieces += _crack(rng, x1, d1, heading + turn, 0.45 * length, 0.6 * width)
        x, d = x1, d1
    return pieces


# ------------------------------------------------------------------ from above
class Plan:
    """The sample from straight above, on a grid of world points x (W,) and
    y (H,) in um: its stacks and what milling removed. What the fluorescence
    view asks of the sample."""

    def __init__(self, scene: "SampleScene", x: np.ndarray, y: np.ndarray) -> None:
        self.x, self.y = x, y
        self.step = (
            abs(float(x[1] - x[0])) if len(x) > 1 else 1.0,
            abs(float(y[1] - y[0])) if len(y) > 1 else 1.0,
        )
        self.col = columns(scene, x, y, self.step)
        self.removed: list = []
        self.update(scene)

    def update(self, scene: "SampleScene") -> None:
        """Take in what was milled since."""
        self.removed = []
        if scene.milled:
            self.removed = _removed(scene, *np.meshgrid(self.x, self.y), self.col)

    def gone(self, z) -> np.ndarray:
        """Where the point at height z (um; one height, or one per sample)
        was milled away."""
        out = np.zeros(self.col["surface"].shape, dtype=bool)
        for cut in self.removed:
            zz = z if np.ndim(z) == 0 else z[cut["box"]]
            out[cut["box"]] |= (cut["from"] <= zz) & (zz <= cut["to"])
        return out

    def fraction(self, lo: float, hi: float, n: int = 24) -> np.ndarray:
        """How much of each column between two heights was milled away."""
        heights = np.linspace(lo, hi, n)
        return np.mean([self.gone(z) for z in heights], axis=0).astype(np.float32)

    def gone_at(self, x, y, z) -> np.ndarray:
        """Whether loose points (um) were milled away, by the sample nearest
        each; a point off the grid was not."""
        x, y, z = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (x, y, z))
        dx = self.x[1] - self.x[0] if len(self.x) > 1 else 1.0
        dy = self.y[1] - self.y[0] if len(self.y) > 1 else 1.0
        col = np.round((x - self.x[0]) / dx).astype(np.int64)
        row = np.round((y - self.y[0]) / dy).astype(np.int64)
        out = np.zeros(len(x), dtype=bool)
        for cut in self.removed:
            rows, cols = cut["box"]
            m = (row >= rows.start) & (row < rows.stop)
            m &= (col >= cols.start) & (col < cols.stop)
            if not m.any():
                continue
            a = cut["from"][row[m] - rows.start, col[m] - cols.start]
            b = cut["to"][row[m] - rows.start, col[m] - cols.start]
            out[m] |= (a <= z[m]) & (z[m] <= b)
        return out

    def top(self):
        """What is on top after milling: its height and material per sample."""
        z = self.col["surface"].copy()
        for _ in range(3):  # a cut may open onto another below it
            for cut in self.removed:
                t = z[cut["box"]]
                hit = (cut["from"] <= t) & (t <= cut["to"])
                z[cut["box"]] = np.where(hit, cut["from"] - 1e-3, t)

        def material(z):
            mat = np.zeros(z.shape, dtype=np.uint8)
            for k in SLOTS[::-1]:
                lo, hi = self.col[k]
                mat[(lo <= z) & (z <= hi)] = MATERIAL[k]
            return mat

        mat = material(z)
        # a cut that ends in a gap shows the next piece down
        below = np.full(z.shape, NONE, dtype=np.float32)
        for k in SLOTS:
            hi = self.col[k][1]
            below = np.maximum(below, np.where(hi < z, hi, NONE))
        z = np.where(mat == VOID, below, z)
        return z, np.where(mat == VOID, material(z), mat)


def plan(scene: "SampleScene", x: np.ndarray, y: np.ndarray) -> Plan:
    """The sample from above on a grid, kept between frames of one view."""
    key = (
        scene.seed,
        len(x),
        len(y),
        round(float(x[0]), 4),
        round(float(x[-1]), 4),
        round(float(y[0]), 4),
        round(float(y[-1]), 4),
        len(scene.features),
        id(scene.features),
        round(scene.gis_thickness * UM, 4),
        round(_curtaining(scene), 4),
    )
    kept = getattr(scene, "_solid_plan", None)
    if kept is None or kept[0] != key:
        kept = (key, Plan(scene, x, y))
        object.__setattr__(scene, "_solid_plan", kept)
    else:
        kept[1].update(scene)
    return kept[1]


# reflected light: ice on the film is a thin film, so thickness reads as
# fringes (see sim_relief)
REFLECTANCE = {PARTICLE: 1.15, COAT: 0.8, BAR: 1.0, FILM: 0.5}


def reflectance(scene: "SampleScene", view: Plan) -> np.ndarray:
    """What the FM sees in reflected light, relative to a grid bar: the bars
    bright through the ice over them; the ice grey, in fringes where it is
    thin and its thickness changes; steep ice (a cell's dome, a particle)
    scattering the light away; a milled face rough and dim; nothing where the
    sample was milled through."""
    z, mat = view.top()
    sx, sy = view.step
    X, Y = np.meshgrid(view.x, view.y)
    seed = int(scene.seed)
    solid = mat != VOID
    thick = np.clip(z - FILM_TOP, 0, None)
    thick = thick + 0.15 * (1 + sim_relief.value_noise(X, Y, 40.0, seed + 301))
    fringe = 0.5 + 0.5 * np.cos(2 * np.pi * thick / sim_relief.FRINGE_PERIOD)
    out = 0.35 + 0.3 * fringe * np.exp(-thick / sim_relief.FRINGE_FADE)
    gy, gx = np.gradient(np.where(solid, z, 0.0), sy, sx)
    scatter = 1 / (1 + (np.hypot(gx, gy) / 0.4) ** 2)
    # a milled face is rough: it scatters more of the light away
    scatter = scatter * np.where(z < view.col["surface"] - 0.05, 0.5, 1.0)
    out = out * scatter
    for m, v in REFLECTANCE.items():
        out[mat == m] = v * scatter[mat == m]
    # a bar shows through the ice over it
    bar = view.col["bar_weight"] * np.isin(mat, (ICE, COAT))
    out = out + bar * np.exp(-np.clip(z, 0, None) / BAR_DEPTH) * (scatter - out)
    # ice across a hole has no film behind it to reflect
    out = out * (1 - 0.5 * view.col["hole"] * (mat == ICE))
    out[~solid] = 0.0
    return out.astype(np.float32)


def _cache(scene) -> "OrderedDict":
    cache = getattr(scene, "_solid_cache", None)
    if cache is None:
        cache = OrderedDict()
        object.__setattr__(scene, "_solid_cache", cache)
    return cache
