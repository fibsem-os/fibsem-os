"""Live site selection on the simulator: a case whose agent takes an
overview of the loaded grid, saves likely sites, goes to each and says
whether it is good for a lamella.

The case holds the microscope configuration its App Server connects (the
simulated Arctis imaging a seeded grid) and a key worked out from that
grid: every square near the grid's centre with its label, and every cell
body on it, in stage coordinates at the SEM orientation. The run's saved
sites are scored against it (``fibsem.ai.evals.tasks.live_site_selection``).
"""

import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml

from fibsem import config as cfg
from fibsem import utils
from fibsem.ai.evals.cases import FORMAT
from fibsem.ai.evals.sim_cases import (
    RULES,
    TIER,
    _outline,
    _to_world,
    label_square,
    measure_square,
    scene_options,
)
from fibsem.drivers.demo.sim_optics import ANGLE
from fibsem.drivers.demo.sim_scene import SampleScene
from fibsem.geometry.projection import BeamStageProjection
from fibsem.structures.settings import BeamType, FibsemStagePosition

TASK = "live_site_selection"
ARCTIS = Path(cfg.SHIPPED_CONFIG_PATH) / "sim-arctis-configuration.yaml"
MICROSCOPE = "microscope.yaml"
GRID = "Grid-01"
# The part of the grid keyed, about the stage's origin (m): what a default
# overview covers (1.4 x 0.93 mm) that the stage can reach (the simulated
# Arctis's y goes to 0.378 mm).
REACH = (0.72e-3, 0.39e-3)
# Each seed's scene: the cells and film the screening thread settled on,
# the damage and ice drawn by the seed.
# No ice plates: the simulator draws none an SEM overview shows, so a square
# would be keyed thick for ice nobody can see (back once it draws them).
SCENE = {"cell_type": "rpe", "film": "holey", "extent": 2.4e-3, "ice_density": 0.0}
STEP = 0.5e-6  # m, the grid a cell's outline is measured on

BRIEF = (
    "Grid {grid} is loaded and the stage is at the SEM orientation. Find "
    "sites for lamellae on it: take a {beam} overview of the grid at the SEM "
    "orientation and pick the likely sites (a cell on intact film, clear of "
    "the grid bars, no thick ice). Go to each one, focus the SEM and make it "
    "coincident (the FIB centred on the same point as the SEM), save it there "
    "as a lamella (item_add), image it with the SEM, and say whether it is "
    "good for milling a lamella and why. Save at most {most} sites. The stage "
    "can't reach every part of the overview: a move it can't make is refused."
)
MOST = 8


def microscope_config(seed: int, overrides: Optional[Dict[str, Any]] = None) -> Dict:
    """The simulated Arctis with one grid at the stage's origin, its content
    drawn by *seed*; the stage goes where it reports."""

    config = yaml.safe_load(ARCTIS.read_text())
    sim = config["sim"]
    options = scene_options(seed, {**SCENE, **(overrides or {})})
    sample = {k: v for k, v in options.items() if k in SampleScene.CONFIG_KEYS}
    sim["sample"].update(sample)
    sim["sample"].update({"enabled": True, "grids_from_holder": False})
    sim["stage_error"] = {"enabled": False}
    return config


def _session(config_path: Path):
    """A Demo microscope on *config_path*, as the run's App Server connects
    it: the same scene, anchored at the same pose."""

    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(
        manufacturer="Demo", config_path=str(config_path), setup_logging=False
    )
    return microscope


def stage_to_world(microscope) -> np.ndarray:
    """The 2 x 3 map from a stage x, y (m, the SEM orientation, no tilt) to
    the scene's world point at the centre of the SEM's view."""

    connection = microscope._connection
    scene = connection.sample_scene
    points = []
    for x, y in ((0.0, 0.0), (1e-4, 0.0), (0.0, 1e-4)):
        microscope.move_stage_absolute(FibsemStagePosition(x=x, y=y, z=0, r=0, t=0))
        projection = BeamStageProjection.from_microscope(
            microscope, beam_type=BeamType.ELECTRON
        )
        position = connection.scene_stage_position()
        points.append(
            scene.view_to_world(BeamType.ELECTRON, position, projection, 0.0, 0.0)
        )
    (x0, y0), (x1, y1), (x2, y2) = points
    return np.array(
        [
            [(x1 - x0) / 1e-4, (x2 - x0) / 1e-4, x0],
            [(y1 - y0) / 1e-4, (y2 - y0) / 1e-4, y0],
        ]
    )


def _centre(microscope, beam_type) -> np.ndarray:
    """The scene's world point at the centre of *beam_type*'s view, where
    the stage is, the beam's shift and current offset included."""

    connection = microscope._connection
    projection = BeamStageProjection.from_microscope(microscope, beam_type=beam_type)
    shift = microscope.get_beam_shift(beam_type)
    try:
        current = float(microscope.get_beam_current(beam_type))
    except Exception:
        current = None
    return np.array(
        connection.sample_scene.view_to_world(
            beam_type,
            connection.scene_stage_position(),
            projection,
            0.0,
            0.0,
            (float(shift.x), float(shift.y)),
            current,
        )
    )


def optics_key(microscope) -> Dict[str, Any]:
    """What the key needs to say whether a site was focused and made
    coincident, at the SEM orientation: the working distance the SEM is in
    focus at for a stage z (m, linear in z), and the stage z at which the
    SEM's and the FIB's view centres meet, with how far apart they are for
    each metre of z away from it."""

    connection = microscope._connection
    zs = (0.0, -20e-6)
    focus, gap = [], []
    for z in zs:
        microscope.move_stage_absolute(FibsemStagePosition(x=0, y=0, z=z, r=0, t=0))
        focus.append(connection.scene_focus(BeamType.ELECTRON, here=True))
        gap.append(
            _centre(microscope, BeamType.ION) - _centre(microscope, BeamType.ELECTRON)
        )
    dz = zs[1] - zs[0]
    focus_per_z = (focus[1] - focus[0]) / dz
    change = (gap[1] - gap[0]) / dz  # the centres' gap, per metre of z
    # the z where the gap is least: along the direction it changes in
    rate = float(np.linalg.norm(change))
    coincident_z = zs[0] - float(gap[0] @ change) / rate**2

    return {
        # the SEM's convergence half-angle: a working distance d from focus
        # spreads the beam over ANGLE * d
        "sem_angle": ANGLE[0],
        "sem_focus_wd": round(focus[0], 9),
        "sem_focus_per_z": round(focus_per_z, 6),
        "coincident_z": round(coincident_z, 9),
        "coincidence_per_z": round(rate, 6),
    }


def _to_stage(world_map: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    inverse = np.linalg.inv(world_map[:, :2])
    s = inverse @ (np.array([x, y]) - world_map[:, 2])
    return float(s[0]), float(s[1])


def _cell(scene, grid, feature, world_map) -> Optional[Dict[str, Any]]:
    """A cell body's place, size and square in stage coordinates, and
    whether it is wholly on the film and clear of every bar."""
    reach = 4 * feature.sigma * (1 + feature.wobble) * max(1.0, feature.eccentricity)
    n = int(2 * reach / STEP) + 1
    offsets = (np.arange(n) - n // 2) * STEP
    xs, ys = np.meshgrid(feature.x + offsets, feature.y + offsets)
    body = _outline(feature, xs, ys, 1.0)
    if not body.any():
        return None
    bars, _, _, _, beyond = scene.film_masks(xs, ys)
    film = ~(bars | beyond)
    on_film = float((body & film).sum()) * STEP * STEP
    # the body in the grid's frame, to find its square and the bars
    c, s = math.cos(grid.rotation), math.sin(grid.rotation)
    du, dv = xs[body] - grid.x, ys[body] - grid.y
    lu, lv = du * c + dv * s, -du * s + dv * c
    pitch = scene.grid_pitch
    i, j = int(round(float(lu.mean()) / pitch)), int(round(float(lv.mean()) / pitch))
    half = (pitch - scene.grid_bar_width) / 2 - RULES["clearance"]
    clear = bool(
        (np.abs(lu - i * pitch) < half).all() and (np.abs(lv - j * pitch) < half).all()
    )
    cx, cy = _to_stage(world_map, float(xs[body].mean()), float(ys[body].mean()))
    radius = float(
        np.hypot(xs[body] - xs[body].mean(), ys[body] - ys[body].mean()).max()
    )
    return {
        "x": round(cx, 9),
        "y": round(cy, 9),
        "radius_um": round(radius * 1e6, 2),
        "square": [i, j],
        "on_film": on_film >= RULES["cell_area"],
        "clear": clear,
    }


def build_key(config_path: Path) -> Dict[str, Any]:
    """The key for a case's microscope: the squares near the origin with
    their labels and the cell bodies in them, in stage coordinates."""
    microscope = _session(config_path)
    try:
        scene = microscope._connection.sample_scene
        grid = scene.grids[0]
        world_map = stage_to_world(microscope)
        pitch = scene.grid_pitch
        # the squares whose centre is within REACH of the origin
        squares = []
        corners = [
            np.array(world_map @ np.array([sx * REACH[0], sy * REACH[1], 1.0]))
            for sx in (-1, 1)
            for sy in (-1, 1)
        ]
        c, s = math.cos(grid.rotation), math.sin(grid.rotation)
        grid_uv = [
            (
                (p[0] - grid.x) * c + (p[1] - grid.y) * s,
                -(p[0] - grid.x) * s + (p[1] - grid.y) * c,
            )
            for p in corners
        ]
        us, vs = zip(*grid_uv)
        for i in range(
            int(math.floor(min(us) / pitch)), int(math.ceil(max(us) / pitch)) + 1
        ):
            for j in range(
                int(math.floor(min(vs) / pitch)), int(math.ceil(max(vs) / pitch)) + 1
            ):
                wx, wy = _to_world(grid, np.array(i * pitch), np.array(j * pitch))
                if math.hypot(float(wx) - grid.x, float(wy) - grid.y) > grid.radius:
                    continue
                x, y = _to_stage(world_map, float(wx), float(wy))
                if abs(x) > REACH[0] or abs(y) > REACH[1]:
                    continue
                measures = measure_square(scene, grid, (i, j))
                label, certain = label_square(measures)
                squares.append(
                    {
                        "index": [i, j],
                        "x": round(x, 9),
                        "y": round(y, 9),
                        "label": label,
                        "certain": certain,
                    }
                )
        keyed = {tuple(q["index"]) for q in squares}
        cells = []
        for feature in scene.features:
            if feature.kind != "cell" or getattr(feature, "part", "body") != "body":
                continue
            # a cheap first cut: near a keyed square
            x, y = _to_stage(world_map, feature.x, feature.y)
            if abs(x) > REACH[0] + pitch or abs(y) > REACH[1] + pitch:
                continue
            cell = _cell(scene, grid, feature, world_map)
            if cell is not None and tuple(cell["square"]) in keyed:
                cells.append(cell)
        # the grid's frame in stage coordinates, to find a site's square
        inverse = np.linalg.inv(world_map[:, :2])
        frame = {
            "origin": list(_to_stage(world_map, grid.x, grid.y)),
            "u": [float(v) for v in inverse @ np.array([c, s])],
            "v": [float(v) for v in inverse @ np.array([-s, c])],
            "pitch": pitch,
            "bar_width": scene.grid_bar_width,
        }
        return {
            "status": "keyed",
            "keyed_by": "simulator",
            "frame": frame,
            "squares": squares,
            "cells": cells,
            "optics": optics_key(microscope),
        }
    finally:
        microscope.disconnect()


def write_case(
    seed: int,
    out: Path,
    overrides: Optional[Dict[str, Any]] = None,
    beam: str = "SEM",
    **_: Any,
) -> Path:
    """The live site selection case for one seed under
    ``out/live-site-selection/``: its overview with the SEM, or with the FIB
    (*beam* FIB, ``sim-NNNN-fib``), at the SEM orientation either way."""
    suffix = "" if beam == "SEM" else "-fib"
    folder = Path(out) / "live-site-selection" / f"sim-{int(seed):04d}{suffix}"
    folder.mkdir(parents=True, exist_ok=True)
    config = microscope_config(seed, overrides)
    (folder / MICROSCOPE).write_text(yaml.safe_dump(config, sort_keys=False))
    key = build_key(folder / MICROSCOPE)
    good = sum(1 for q in key["squares"] if q["label"] == "good")
    case = {
        "id": f"sim-{int(seed):04d}{suffix}-live-site-selection",
        "task": TASK,
        "tier": TIER,
        "playbook": "autolamella",
        "format": FORMAT,
        "sample": "simulated rpe cells on a holey film",
        "instrument": "Demo (simulator), Arctis",
        "source": {"simulator": {"seed": int(seed), "overrides": overrides or {}}},
        "sim_run": {
            "microscope": MICROSCOPE,
            "level": "drive",
            "overview_beam": "ELECTRON" if beam == "SEM" else "ION",
            "setup": [
                {"tool": "grid_load", "arguments": {"grid_name": GRID}},
                {
                    "tool": "stage_move_to_orientation",
                    "arguments": {"orientation": "SEM"},
                },
            ],
        },
        "brief": BRIEF.format(grid=GRID, most=MOST, beam=beam),
        "scoring": {"max_sites": MOST, "min_good_fraction": 0.5},
        "key": key,
        "traps": [
            "A cell on a grid bar, or with its body over the bar's edge, is "
            "not a site: there is no film under it to make a lamella of.",
            f"{good} of {len(key['squares'])} keyed squares are good.",
        ],
    }
    (folder / "case.yaml").write_text(yaml.safe_dump(case, sort_keys=False))
    return folder
