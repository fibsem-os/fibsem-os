"""Eval cases from the simulator: the grid tasks, keyed from the scene.

The Demo microscope's sample scene is grown from a seed, so it knows every
square, rip, ice plate, frost patch, crack and cell. One overview is rendered
per seed (the SEM face on, or the FIB at the milling angle), and the key is
read from the scene, not marked by hand:

- square_finding: every square's centre, the pitch and the bar angle
- square_quality: each shown square's label, by the rules in ``RULES``
- site_selection: the good squares; damaged squares are bad picks

Sim cases carry ``tier: sim`` and are scored apart from the real cases.
Squares whose label sits close to a rule's threshold are left out of the
square_quality case, so its key is never a coin toss.
"""

import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from fibsem import config as cfg
from fibsem import utils
from fibsem.ai.evals.cases import FORMAT
from fibsem.drivers.demo.sim_scene import SampleScene, surface_foreshortening
from fibsem.geometry.projection import BeamStageProjection
from fibsem.structures.settings import BeamType, ImageSettings

TIER = "sim"

# The overview: about the real overviews' pixel size and a few hundred squares'
# worth of grid.
RESOLUTION = (4096, 2560)
HFW = 1.4e-3  # m
DWELL = 1e-6  # s, as an overview tile; the SEM charges more the longer it dwells
# The FIB overview's pose: the milling angle, in degrees
MILLING_ANGLE = 38.0
BEAMS = {"SEM": BeamType.ELECTRON, "FIB": BeamType.ION}

# What makes each label, per square, on the film inside the bars.
RULES: Dict[str, Any] = {
    # damaged: the film torn away over at least this fraction of the square
    "rip_fraction": 0.005,
    # hairline cracks are under a micron wide, so an overview doesn't show
    # them: they are measured but don't make a square damaged
    # thick: ice plates and frost over at least this fraction of the square
    "ice_fraction": 0.03,
    # good: a cell whose body is at least this far from every bar (m)
    "clearance": 3e-6,
    # a cell counts in a square when this much of its body is on the film (m2)
    "cell_area": 60e-12,
}
# A square is shown in a square_quality case only when its label holds with
# outlines 0.9x and 1.1x their drawn size and the thresholds at these factors.
MARGIN = (0.67, 1.5)
LABEL_ORDER = ("damaged", "thick", "no_cells", "cell_near_bar", "good")

# The scene options a seed draws, when the caller doesn't set them.
VARIED = {
    "cell_type": ["mammalian", "rpe", "neuron"],
    "mammalian_density": (1.5, 4.0),
    "rip_fraction": (0.03, 0.1),
    "ice_density": (0.2, 0.8),
    "frost_density": (0.1, 0.6),
    "crack_density": (0.2, 1.0),
    "rpe_island_density": (2.0, 4.0),
}
# Set for every seed: RPE as single cells and small groups of a few, not
# the simulator's wide sheets.
FIXED = {
    "rpe_island_radius": [12e-6, 35e-6],
    "rpe_occupancy": 0.6,
}

BRIEFS = {
    "square_finding": (
        "Find every grid square on this {view} of a cryo grid: torn, "
        "empty and ice-covered squares too, and squares cut by an edge. Answer "
        "each square's centre as x, y px on the full-size overview ({w} x {h} "
        "px), the grid pitch in px, and the angle of the grid bars from the "
        "image x axis in degrees (y points down)."
    ),
    "square_quality": (
        "Label each numbered grid square on this {view} of a cryo grid "
        "with the one that fits best: good (intact film, a cell clear of the "
        "grid bars), no_cells (intact film, no cell), damaged (film torn or "
        "missing, or cracked), thick (ice crystals or contamination over the "
        "square), cell_near_bar (the cells sit on or against a grid bar). The "
        "squares are given by their centres in px on the full-size overview "
        "({w} x {h} px)."
    ),
    "site_selection": (
        "Pick up to 10 grid squares on this {view} of a cryo grid to "
        "mill lamellae in, best first: intact film, no ice or contamination, "
        "and a cell clear of the grid bars. Answer each pick as the x, y px "
        "of the cell on the full-size overview ({w} x {h} px), with a reason."
    ),
}


@dataclass
class Square:
    number: int
    index: Tuple[int, int]  # the square's place in the grid's lattice
    x_px: float
    y_px: float
    in_view: bool  # the whole square is on the overview
    label: Optional[str] = None
    # the label at the low and high ends of the margins; a square whose two
    # differ is too close to call
    certain: bool = True
    measures: Dict[str, float] = field(default_factory=dict)

    def point(self) -> Dict[str, Any]:
        return {"square": self.number, "x_px": self.x_px, "y_px": self.y_px}


# --- the scene ---------------------------------------------------------------


def scene_options(
    seed: int, overrides: Optional[Dict[str, Any]] = None, hfw: float = HFW
) -> Dict:
    """The scene's options for a seed: drawn from ``VARIED``, then the
    caller's. The content covers the whole overview."""
    rng = np.random.default_rng([int(seed), 1729])
    options: Dict[str, Any] = {}
    for name, values in VARIED.items():
        if isinstance(values, list):
            options[name] = str(rng.choice(values))
        else:
            options[name] = round(float(rng.uniform(*values)), 3)
    options.update(FIXED)
    options["extent"] = 2 * math.hypot(hfw / 2, hfw / 2) + 200e-6
    options.update(overrides or {})
    options["seed"] = int(seed)
    return options


def _demo_microscope():
    # the images and the milling, without the simulator's waits
    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(
        manufacturer="Demo",
        config_path=os.path.join(
            cfg.SHIPPED_CONFIG_PATH, "microscope-configuration.yaml"
        ),
    )
    return microscope


def _face_on(microscope) -> None:
    """Tilt the stage so the SEM looks straight down on the grid."""
    best = (-1.0, 0.0)
    position = microscope.get_stage_position()
    position.r = 0.0
    for degrees in range(0, 61):
        position.t = math.radians(degrees)
        projection = BeamStageProjection.from_microscope(
            microscope, beam_type=BeamType.ELECTRON
        )
        best = max(best, (surface_foreshortening(projection, position), degrees))
    position.t = math.radians(best[1])
    microscope.move_stage_absolute(position)


@dataclass
class Overview:
    image: Any  # FibsemImage
    scene: Any  # SampleScene
    to_px: np.ndarray  # 2 x 3: world (m) to overview px
    pixel_size: float  # m
    resolution: Tuple[int, int]

    def px(self, x: float, y: float) -> Tuple[float, float]:
        u = self.to_px @ np.array([x, y, 1.0])
        return float(u[0]), float(u[1])


def render_overview(
    seed: int,
    overrides: Optional[Dict[str, Any]] = None,
    resolution: Tuple[int, int] = RESOLUTION,
    hfw: float = HFW,
    beam: str = "SEM",
    milling_angle: float = MILLING_ANGLE,
):
    """An overview of a seed's scene on the Demo microscope, and the map from
    the scene's world to the overview's pixels: the SEM looking straight down,
    or the FIB at the milling angle (foreshortened)."""
    beam_type = BEAMS[beam]
    microscope = _demo_microscope()
    try:
        if beam == "SEM":
            _face_on(microscope)
        else:
            microscope.move_to_milling_angle(milling_angle)
        scene = SampleScene(renderer="relief", **scene_options(seed, overrides, hfw))
        position = microscope.get_stage_position()
        scene.anchor(position)
        microscope._connection.sample_scene = scene
        settings = ImageSettings(
            resolution=resolution,
            hfw=hfw,
            dwell_time=DWELL,
            save=False,
            beam_type=beam_type,
        )
        image = microscope.acquire_image(image_settings=settings)
        # the same mapping the render drew with, inverted: view_to_world is
        # affine in the view offset
        projection = BeamStageProjection.from_microscope(
            microscope, beam_type=beam_type
        )
        shift = microscope.get_beam_shift(beam_type)
        try:
            current = float(microscope.get_beam_current(beam_type))
        except Exception:
            current = None
        position = microscope._connection.scene_stage_position()

        def world(dx, dy):
            return scene.view_to_world(
                beam_type,
                position,
                projection,
                dx,
                dy,
                (float(shift.x), float(shift.y)),
                current,
            )

        pixel_size = hfw / resolution[0]
        o, ex, ey = world(0, 0), world(1e-4, 0), world(0, 1e-4)
        forward = np.array(
            [
                [(ex[0] - o[0]) / 1e-4, (ey[0] - o[0]) / 1e-4],
                [(ex[1] - o[1]) / 1e-4, (ey[1] - o[1]) / 1e-4],
            ]
        )
        inverse = np.linalg.inv(forward) / pixel_size  # world offset -> px
        centre = np.array([resolution[0] / 2, resolution[1] / 2])
        to_px = np.hstack([inverse, (centre - inverse @ np.array(o))[:, None]])
        return Overview(image, scene, to_px, pixel_size, tuple(resolution))
    finally:
        microscope.disconnect()


# --- the key -----------------------------------------------------------------


def _to_world(grid, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Grid-frame (u, v) to world, the inverse of the scene's grid rotation."""
    c, s = math.cos(grid.rotation), math.sin(grid.rotation)
    return grid.x + u * c - v * s, grid.y + u * s + v * c


def _outline(feature, xs, ys, scale: float) -> np.ndarray:
    """Inside a feature's drawn outline (where its profile is over half), as
    the scene stamps it: a turned ellipse with a wobbled radius."""
    dx, dy = xs - feature.x, ys - feature.y
    c, s = math.cos(feature.angle), math.sin(feature.angle)
    rx = (dx * c + dy * s) / (feature.sigma * scale)
    ry = (-dx * s + dy * c) / (feature.sigma * max(feature.eccentricity, 1e-3) * scale)
    r2 = rx**2 + ry**2
    if feature.wobble:
        theta = np.arctan2(ry, rx)
        r2 = (
            r2
            / (
                1.0
                + feature.wobble
                * (
                    0.6 * np.sin(3 * theta + feature.wobble_phase)
                    + 0.4 * np.sin(5 * theta + 2 * feature.wobble_phase)
                )
            )
            ** 2
        )
    return r2 ** max(feature.sharpness, 0.5) < 2 * math.log(2)


def _crack_length(scene, grid, centre, half: float) -> float:
    """How much hairline crack lies on the film inside a square (m)."""
    total = 0.0
    for arm in getattr(scene, "cracks", None) or []:
        mid = (arm[1:, :2] + arm[:-1, :2]) / 2
        u = mid - np.array(_to_world(grid, np.array(centre[0]), np.array(centre[1])))
        c, s = math.cos(grid.rotation), math.sin(grid.rotation)
        lu = u[:, 0] * c + u[:, 1] * s
        lv = -u[:, 0] * s + u[:, 1] * c
        inside = (np.abs(lu) < half) & (np.abs(lv) < half)
        total += float(np.hypot(*(arm[1:, :2] - arm[:-1, :2]).T)[inside].sum())
    return total


def measure_square(scene, grid, index: Tuple[int, int], step: float = 1e-6) -> Dict:
    """What lies on the film inside one square, at each end of the margins."""
    pitch, bar = scene.grid_pitch, scene.grid_bar_width
    half = (pitch - bar) / 2
    centre = (index[0] * pitch, index[1] * pitch)
    n = int(round(2 * half / step))
    offsets = (np.arange(n) + 0.5) * step - half
    lu, lv = np.meshgrid(offsets, offsets)
    xs, ys = _to_world(grid, centre[0] + lu, centre[1] + lv)
    bars, _, rips, _, beyond = scene.film_masks(xs, ys)
    film = ~(bars | beyond)
    area = float(film.sum()) or 1.0
    broken = np.zeros_like(film)
    if getattr(scene, "film", "continuous") == "holey":
        broken = scene.broken_hole_mask(xs, ys)
    out: Dict[str, Any] = {
        "rip_fraction": float(((rips | broken) & film).sum()) / area,
        "crack_length": _crack_length(scene, grid, centre, half),
        "beyond": bool(beyond.any()),
    }
    # the film's edge, for each cell's clearance
    clear = np.abs(lu) < half - RULES["clearance"]
    clear &= np.abs(lv) < half - RULES["clearance"]
    reach = pitch
    near = [
        f
        for f in scene.features
        if abs(f.x - xs.mean()) < reach and abs(f.y - ys.mean()) < reach
    ]
    for name, scale in (("low", 0.9), ("mid", 1.0), ("high", 1.1)):
        ice = np.zeros_like(film)
        for f in near:
            if f.kind == "ice" or getattr(f, "part", "") == "frost":
                ice |= _outline(f, xs, ys, scale)
        cells, clear_cells = 0, 0
        for f in near:
            if f.kind != "cell" or f.part != "body":
                continue
            body = _outline(f, xs, ys, scale)
            on_film = float((body & film).sum()) * step * step
            if on_film < RULES["cell_area"]:
                continue
            cells += 1
            # wholly on the film and clear of every bar
            if not (body & ~clear).any() and body.any():
                clear_cells += 1
        out[name] = {
            "ice_fraction": float((ice & film).sum()) / area,
            "cells": cells,
            "clear_cells": clear_cells,
        }
    return out


def label_of(measures: Dict, end: str, factor: float = 1.0) -> str:
    m = measures[end]
    if measures["rip_fraction"] >= RULES["rip_fraction"] * factor:
        return "damaged"
    if m["ice_fraction"] >= RULES["ice_fraction"] * factor:
        return "thick"
    if not m["cells"]:
        return "no_cells"
    if not m["clear_cells"]:
        return "cell_near_bar"
    return "good"


def label_square(measures: Dict) -> Tuple[str, bool]:
    """The square's label, and whether it holds with outlines smaller and
    larger and the thresholds halved and doubled."""
    label = label_of(measures, "mid")
    ends = {
        label_of(measures, end, factor)
        for end in ("low", "mid", "high")
        for factor in (MARGIN[0], 1.0, MARGIN[1])
    }
    return label, ends == {label}


def find_squares(overview: Overview) -> List[Square]:
    """Every square whose centre is on the overview, numbered in rows from
    the top left, measured and labelled."""
    scene = overview.scene
    grid = scene.grids[0]
    pitch = scene.grid_pitch
    w, h = overview.resolution
    inverse = np.linalg.inv(overview.to_px[:, :2])
    reach = int(math.hypot(w, h) * overview.pixel_size / pitch) + 2
    centre_px = np.array([w / 2, h / 2])
    centre_world = inverse @ (centre_px - overview.to_px[:, 2])
    c, s = math.cos(grid.rotation), math.sin(grid.rotation)
    du = centre_world - np.array([grid.x, grid.y])
    i0 = int(round((du[0] * c + du[1] * s) / pitch))
    j0 = int(round((-du[0] * s + du[1] * c) / pitch))
    half_px = (pitch - scene.grid_bar_width) / 2 / overview.pixel_size
    squares = []
    for i in range(i0 - reach, i0 + reach + 1):
        for j in range(j0 - reach, j0 + reach + 1):
            x, y = _to_world(grid, np.array(i * pitch), np.array(j * pitch))
            if math.hypot(float(x) - grid.x, float(y) - grid.y) > grid.radius:
                continue
            px, py = overview.px(float(x), float(y))
            if not (0 <= px < w and 0 <= py < h):
                continue
            # the square's corners, in px
            inside = True
            for a, b in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                cx, cy = _to_world(
                    grid,
                    np.array(i * pitch + a * half_px * overview.pixel_size),
                    np.array(j * pitch + b * half_px * overview.pixel_size),
                )
                qx, qy = overview.px(float(cx), float(cy))
                inside &= 0 <= qx < w and 0 <= qy < h
            squares.append(Square(0, (i, j), round(px, 1), round(py, 1), bool(inside)))
    row = pitch / overview.pixel_size / 2
    squares.sort(key=lambda q: (round(q.y_px / row), q.x_px))
    for n, square in enumerate(squares, start=1):
        square.number = n
        if square.in_view:
            measures = measure_square(scene, grid, square.index)
            square.label, square.certain = label_square(measures)
            square.measures = measures
    return squares


def lattice(overview: Overview) -> List[Tuple[float, float]]:
    """Each lattice axis as (pitch in px, angle from the image x axis in
    degrees, y down, in [0, 90)), the longer first. Face on, the two pitches
    are the same; a tilted view foreshortens one."""
    grid = overview.scene.grids[0]
    pitch = overview.scene.grid_pitch
    zero = np.array(0.0)
    a = np.array(overview.px(*map(float, _to_world(grid, zero, zero))))
    axes = []
    for u, v in ((pitch, 0.0), (0.0, pitch)):
        b = np.array(
            overview.px(*map(float, _to_world(grid, np.array(u), np.array(v))))
        )
        d = b - a
        axes.append(
            (float(np.hypot(*d)), float(math.degrees(math.atan2(d[1], d[0])) % 90))
        )
    return sorted(axes, reverse=True)


# --- the cases ---------------------------------------------------------------


def _pick_shown(squares: List[Square], seed: int, most: int = 30) -> List[Square]:
    """Up to ``most`` squares to label, certain and whole, as even across the
    labels as the overview allows."""
    rng = np.random.default_rng([int(seed), 7])
    by_label: Dict[str, List[Square]] = {}
    for q in squares:
        if q.in_view and q.certain and q.label:
            by_label.setdefault(q.label, []).append(q)
    for items in by_label.values():
        rng.shuffle(items)
    shown: List[Square] = []
    while len(shown) < most and any(by_label.values()):
        for label in LABEL_ORDER:
            if by_label.get(label) and len(shown) < most:
                shown.append(by_label[label].pop())
    return sorted(shown, key=lambda q: q.number)


def write_cases(
    seed: int,
    out: Path,
    overrides: Optional[Dict[str, Any]] = None,
    most_labelled: int = 30,
    resolution: Tuple[int, int] = RESOLUTION,
    hfw: float = HFW,
    beam: str = "SEM",
    milling_angle: float = MILLING_ANGLE,
) -> List[Path]:
    """The three grid cases for one seed and beam, under
    ``out/<task>/sim-<seed>`` (``sim-<seed>-fib`` for the FIB). The
    square-finding case's ``reference/`` holds every square's measures and a
    preview of the key, for checking by eye; the agent sees neither. On the
    FIB the grid is foreshortened, so its pitch and angle aren't scored."""
    out = Path(out)
    name = f"sim-{int(seed):04d}" + ("-fib" if beam == "FIB" else "")
    overview = render_overview(seed, overrides, resolution, hfw, beam, milling_angle)
    squares = find_squares(overview)
    (pitch_px, angle_deg), (short_px, short_deg) = lattice(overview)
    view = (
        "SEM overview"
        if beam == "SEM"
        else f"FIB overview, taken at a {milling_angle:g} degree milling angle,"
    )
    w, h = overview.resolution
    options = scene_options(seed, overrides, hfw)

    site = out / "site-selection" / name
    image_dir = site / "experiment"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / "overview.tif"
    overview.image.save(str(image_path))
    shared = f"../../site-selection/{name}/experiment/overview.tif"

    common = {
        "format": FORMAT,
        "tier": TIER,
        "sample": f"simulated {options['cell_type']} cells",
        "instrument": "Demo (simulator)",
        "beam": beam,
        "source": {
            "simulator": {
                "seed": int(seed),
                "scene": {
                    k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in options.items()
                },
                "renderer": "relief",
                "resolution": [w, h],
                "hfw_m": hfw,
                "dwell_s": DWELL,
                **({"milling_angle_deg": milling_angle} if beam == "FIB" else {}),
            }
        },
        "size_px": [w, h],
        "pixel_size_um": round(overview.pixel_size * 1e6, 4),
    }
    keyed = {"status": "keyed", "keyed_by": "simulator", "rules": dict(RULES)}
    good = [q for q in squares if q.label == "good" and q.certain]
    bad = [q for q in squares if q.label == "damaged" and q.in_view]

    cases = {
        site: {
            "id": f"{name}-sites",
            "task": "site_selection",
            "inputs": {"overview": "experiment/overview.tif"},
            "brief": BRIEFS["site_selection"].format(w=w, h=h, view=view),
            "key": {
                **keyed,
                "good": [q.point() for q in good],
                "bad": [q.point() for q in bad],
                "hit_radius_px": 100,
            },
        },
        out / "square-finding" / name: {
            "id": f"{name}-squares",
            "task": "square_finding",
            "inputs": {"overview": shared},
            "brief": BRIEFS["square_finding"].format(w=w, h=h, view=view)
            + (
                " The view is tilted, so the grid is foreshortened: give the pitch "
                "and angle of the longer side of the squares."
                if beam == "FIB"
                else ""
            ),
            "key": {
                **keyed,
                "pitch_px": round(pitch_px, 1),
                "angle_deg": round(angle_deg, 2),
                **(
                    {
                        "short_axis": {
                            "pitch_px": round(short_px, 1),
                            "angle_deg": round(short_deg, 2),
                        }
                    }
                    if beam == "FIB"
                    else {}
                ),
                "squares": [
                    {**q.point(), **({} if q.in_view else {"optional": "edge"})}
                    for q in squares
                ],
            },
            "scoring": {
                "match_radius_px": 120,
                "min_recall": 0.8,
                "min_precision": 0.9,
                "pitch_tolerance": 0.1,
                "angle_tolerance_deg": 5,
                **({"lattice": False} if beam == "FIB" else {}),
            },
        },
    }
    shown = _pick_shown(squares, seed, most_labelled)
    cases[out / "square-quality" / name] = {
        "id": f"{name}-labels",
        "task": "square_quality",
        "inputs": {"overview": shared},
        "brief": BRIEFS["square_quality"].format(w=w, h=h, view=view),
        "squares": [q.point() for q in shown],
        "key": {**keyed, "labels": {str(q.number): q.label for q in shown}},
        "scoring": {"min_accuracy": 0.8, "unsafe_if_called_good": ["damaged"]},
    }

    written = []
    for folder, data in cases.items():
        folder.mkdir(parents=True, exist_ok=True)
        case = {**data, **{k: v for k, v in common.items() if k not in data}}
        order = ["id", "task", "tier", "format"]
        case = {k: case[k] for k in order} | {
            k: v for k, v in case.items() if k not in order
        }
        with open(folder / "case.yaml", "w") as f:
            yaml.safe_dump(case, f, sort_keys=False, width=100)
        written.append(folder)
    reference = out / "square-finding" / name / "reference"
    reference.mkdir(exist_ok=True)
    write_preview(
        np.asarray(overview.image.data), squares, reference / "preview-key.jpg"
    )
    (reference / "measures.json").write_text(
        json.dumps([q.__dict__ for q in squares], indent=1, default=str)
    )
    logging.info("sim cases for seed %s: %d squares", seed, len(squares))
    return written


LABEL_COLOURS = {
    "good": "#2ca02c",
    "no_cells": "#7f7f7f",
    "damaged": "#d62728",
    "thick": "#1f77b4",
    "cell_near_bar": "#ff7f0e",
    None: "#bbbbbb",
}


def write_preview(image: np.ndarray, squares: List[Square], path: Path) -> Path:
    """The overview with every keyed square numbered and coloured by its
    label (pale: too close to call, or cut by the edge), for checking by eye."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = image.shape[:2]
    fig, ax = plt.subplots(figsize=(16, 16 * h / w))
    ax.imshow(image[::2, ::2], cmap="gray", extent=(0, w, h, 0))
    for q in squares:
        colour = LABEL_COLOURS.get(q.label, LABEL_COLOURS[None])
        faint = not (q.in_view and q.certain)
        ax.plot(
            q.x_px,
            q.y_px,
            "o",
            ms=14,
            mfc="none",
            mec=colour,
            mew=1 if faint else 2.5,
            alpha=0.4 if faint else 1.0,
        )
        ax.annotate(
            str(q.number),
            (q.x_px, q.y_px),
            (9, 9),
            textcoords="offset points",
            color=colour,
            fontsize=8,
            alpha=0.5 if faint else 1.0,
        )
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            mfc="none",
            mec=c,
            mew=2,
            label=str(k or "no label (cut by the edge)"),
        )
        for k, c in LABEL_COLOURS.items()
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.8)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path
