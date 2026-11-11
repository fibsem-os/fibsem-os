"""A setup placement eval case from the simulator, keyed from the scene.

One site on a grid of cells, seen by the FIB at the milling angle, as the
Setup task sees it. The agent places the lamella: its centre on the FIB image,
with the trenches above and below it as the protocol sets them.

The key is two masks the size of the image, from the scene: where the centre
puts the whole lamella on a cell with both trenches and the fiducial clear of
the grid bars and inside the image (ok), and where a trench or the lamella would
cut a bar (unsafe). Anything else, such as a lamella in bare ice, is a miss.
"""

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml
from PIL import Image

from fibsem.ai.evals.cases import FORMAT
from fibsem.ai.evals.sim_cases import TIER
from fibsem.ai.evals.sim_correlation import _arctis, _FibMap
from fibsem.drivers.demo import sim_relief
from fibsem.drivers.demo.sim_scene import SampleScene
from fibsem.structures.settings import BeamType, FibsemStagePosition, ImageSettings

PLAYBOOK = "autolamella"
MILLING_ANGLE = 15.0  # degrees
RESOLUTION = (1536, 1024)
HFW = 80e-6  # m
DWELL = 1e-6  # s
# the shipped protocol's first rough mill, in the image plane (m)
LAMELLA = {
    "width": 10e-6,
    "spacing": 4.6e-6,
    "upper_trench_height": 3.5e-6,
    "lower_trench_height": 3.5e-6,
    "fiducial_offset_x": 25e-6,
    "fiducial_height": 10e-6,
}
ON_CELL = 0.95  # of the lamella on a cell, to be ok
ON_BAR = 0.02  # of a trench or the lamella on a bar: unsafe
STEP = 2  # px between the centres the key tests

BRIEF = (
    "This FIB image at the milling angle shows one site for a lamella. Place "
    "the lamella through a cell here: give its centre as x, y px on the image. "
    "The lamella is {w:.1f} um wide and {s:.1f} um high between the trenches; "
    "the upper and lower trenches are {u:.1f} um and {lo:.1f} um high, above "
    "and below it. The fiducial goes {f:.0f} um to the right of the centre. "
    "Trenches must not cut a grid bar, and everything must stay inside the "
    "image."
)


def scene_options(seed: int, overrides: Optional[Dict[str, Any]] = None) -> Dict:
    options: Dict[str, Any] = {
        "cell_type": "rpe",
        "coincidence_offset": 0.0,
        "renderer": "relief",
        "rip_fraction": 0.0,
        "gis_thickness": 0.5e-6,
        "rpe_island_radius": [12e-6, 35e-6],
        "rpe_island_density": 3.0,
        "rpe_occupancy": 0.6,
    }
    options.update(overrides or {})
    options["seed"] = int(seed)
    return options


def _pick_cell(rng, scene: SampleScene, nearest: int = 20):
    """One of the cell bodies nearest the scene's centre, at random."""
    bodies = [
        f for f in scene.features if f.kind == "cell" and f.part in ("body", None)
    ]
    if not bodies:
        return None
    bodies.sort(key=lambda f: np.hypot(f.x, f.y))
    return bodies[int(rng.integers(min(nearest, len(bodies))))]


def _maps(scene: SampleScene, fib: _FibMap, step: int = STEP):
    """Per image px (every `step`): where the beam ray meets a cell, and
    where it meets the film over a grid bar."""
    w, h = fib.resolution
    cols = np.arange(0, w, step) + step / 2
    rows = np.arange(0, h, step) + step / 2
    vx, vy = np.meshgrid((cols - w / 2) * fib.pixel, (rows - h / 2) * fib.pixel)
    forward = np.linalg.inv(fib.inverse)

    def world(ux, uy):
        x = fib.origin[0] + forward[0, 0] * ux + forward[0, 1] * uy
        y = fib.origin[1] + forward[1, 0] * ux + forward[1, 1] * uy
        return x, y

    x0, y0 = world(vx, vy)
    um = sim_relief.UM
    bars = scene.film_masks(x0, y0)[0]

    # the surface, without the coat and on an untouched scene, for its material
    plain = deepcopy(scene)
    plain.gis_thickness = 0.0
    plain.milled = []
    s = 0.25
    gx = np.arange(np.floor(x0.min() * um) - 5, np.ceil(x0.max() * um) + 5, s)
    gy = np.arange(np.floor(y0.min() * um) - 5, np.ceil(y0.max() * um) + 5, s)
    GX, GY = np.meshgrid(gx, gy)
    height, mat = sim_relief.surface(plain, GX, GY, (s, s))[:2]

    def sample(grid, x, y):
        i = np.clip(np.round((y * um - gy[0]) / s).astype(int), 0, len(gy) - 1)
        j = np.clip(np.round((x * um - gx[0]) / s).astype(int), 0, len(gx) - 1)
        return grid[i, j]

    lift = fib.view(0.0, 0.0, 1e-6) - fib.view(0.0, 0.0)  # view m per um of height
    x, y = x0, y0
    for _ in range(4):
        z = np.clip(sample(height, x, y), 0, None)
        x, y = world(vx - lift[0] * z, vy - lift[1] * z)
    cell = sample(mat, x, y) == sim_relief.CELL
    return cell, bars


def _box_sum(table: np.ndarray, r0, r1, c0, c1) -> np.ndarray:
    return table[r1, c1] - table[r0, c1] - table[r1, c0] + table[r0, c0]


def _masks(cell: np.ndarray, bars: np.ndarray, pixel: float, step: int = STEP):
    """Ok and unsafe masks over the lamella centres, on the `step` grid."""
    rows, cols = cell.shape
    p = pixel * step

    def table(a):
        t = np.zeros((rows + 1, cols + 1))
        t[1:, 1:] = a.astype(float).cumsum(0).cumsum(1)
        return t

    tc, tb = table(cell), table(bars)
    half_w = int(round(LAMELLA["width"] / 2 / p))
    half_s = int(round(LAMELLA["spacing"] / 2 / p))
    up = int(round(LAMELLA["upper_trench_height"] / p))
    lo = int(round(LAMELLA["lower_trench_height"] / p))
    fid_x = int(round(LAMELLA["fiducial_offset_x"] / p))
    fid_h = int(round(LAMELLA["fiducial_height"] / 2 / p))
    r, c = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    inside = (
        (c - half_w >= 0)
        & (c + max(half_w, fid_x + 1) < cols)
        & (r - half_s - up >= 0)
        & (r + half_s + lo < rows)
        & (r - fid_h >= 0)
        & (r + fid_h < rows)
    )

    def clip(a, n):
        return np.clip(a, 0, n)

    c0, c1 = clip(c - half_w, cols), clip(c + half_w, cols)
    area = np.maximum((c1 - c0), 1)
    lam_r0, lam_r1 = clip(r - half_s, rows), clip(r + half_s, rows)
    on_cell = _box_sum(tc, lam_r0, lam_r1, c0, c1) / np.maximum(
        area * (lam_r1 - lam_r0), 1
    )
    upper = _box_sum(tb, clip(r - half_s - up, rows), lam_r0, c0, c1)
    lower = _box_sum(tb, lam_r1, clip(r + half_s + lo, rows), c0, c1)
    middle = _box_sum(tb, lam_r0, lam_r1, c0, c1)
    span = np.maximum(area * (up + lo + 2 * half_s), 1)
    on_bar = (upper + lower + middle) / span
    ok = inside & (on_cell >= ON_CELL) & (upper + lower == 0)
    unsafe = on_bar > ON_BAR
    return ok, unsafe & ~ok


def _save_mask(mask: np.ndarray, size: Tuple[int, int], path: Path, step: int = STEP):
    full = np.repeat(np.repeat(mask, step, 0), step, 1)[: size[1], : size[0]]
    Image.fromarray((full * 255).astype(np.uint8)).save(path)


def write_case(
    seed: int,
    out: Path,
    overrides: Optional[Dict[str, Any]] = None,
    milling_angle: float = MILLING_ANGLE,
    resolution: Tuple[int, int] = RESOLUTION,
) -> Optional[Path]:
    """The setup placement case for one seed under ``out/setup-placement/``,
    or None when no centre is ok."""
    folder = Path(out) / "setup-placement" / f"sim-{int(seed):04d}"
    rng = np.random.default_rng([int(seed), 271])
    microscope = _arctis()
    try:
        microscope.move_to_milling_angle(milling_angle)
        scene = SampleScene(**scene_options(seed, overrides))
        scene.anchor(microscope.get_stage_position())
        microscope._connection.sample_scene = scene
        target = _pick_cell(rng, scene)
        if target is None:
            logging.warning("seed %s: no cell in the scene, no case", seed)
            return None
        # bring the cell near the beam's centre, a little off so it isn't given
        jitter = rng.uniform(-8e-6, 8e-6, 2)
        tx, ty = target.x + jitter[0], target.y + jitter[1] * 3
        microscope.move_stage_relative(FibsemStagePosition(x=tx, y=-ty, z=0))
        fib = _FibMap(microscope, scene, resolution, HFW)
        cell, bars = _maps(scene, fib)
        ok, unsafe = _masks(cell, bars, fib.pixel)
        if not ok.any():
            logging.warning("seed %s: no safe placement, no case", seed)
            return None
        image = microscope.acquire_image(
            image_settings=ImageSettings(
                resolution=resolution,
                hfw=HFW,
                dwell_time=DWELL,
                save=False,
                beam_type=BeamType.ION,
            )
        )
        target_px = fib.px(target.x, target.y, 0.0)
    finally:
        microscope.disconnect()

    experiment = folder / "experiment"
    reference = folder / "reference"
    experiment.mkdir(parents=True, exist_ok=True)
    reference.mkdir(parents=True, exist_ok=True)
    image.save(str(experiment / "setup_ib.tif"))
    _save_mask(ok, resolution, reference / "ok-mask.png")
    _save_mask(unsafe, resolution, reference / "unsafe-mask.png")
    pixel_um = HFW / resolution[0] * 1e6
    case = {
        "id": f"sim-{int(seed):04d}-setup-placement",
        "task": "setup_placement",
        "tier": TIER,
        "playbook": PLAYBOOK,
        "format": FORMAT,
        "sample": "simulated RPE cells, GIS coated",
        "instrument": "Demo Arctis (simulator)",
        "source": {
            "simulator": {
                "seed": int(seed),
                "scene": scene_options(seed, overrides),
                "milling_angle_deg": milling_angle,
            }
        },
        "brief": BRIEF.format(
            w=LAMELLA["width"] * 1e6,
            s=LAMELLA["spacing"] * 1e6,
            u=LAMELLA["upper_trench_height"] * 1e6,
            lo=LAMELLA["lower_trench_height"] * 1e6,
            f=LAMELLA["fiducial_offset_x"] * 1e6,
        ),
        "inputs": {
            "fib_image": "experiment/setup_ib.tif",
            "fib_pixel_size_um": round(pixel_um, 4),
            "fib_size_px": list(resolution),
            "milling_angle_deg": milling_angle,
            "lamella_um": {k: round(v * 1e6, 2) for k, v in LAMELLA.items()},
        },
        "key": {
            "status": "keyed",
            "keyed_by": "simulator",
            "ok_mask": "reference/ok-mask.png",
            "unsafe_mask": "reference/unsafe-mask.png",
            "target_px": {"x": round(target_px[0], 1), "y": round(target_px[1], 1)},
            "ok_fraction": round(float(ok.mean()), 4),
        },
        "traps": [
            "A bar under a cell: the cell looks fine but a trench over the bar "
            "cuts metal."
        ],
    }
    with open(folder / "case.yaml", "w") as f:
        yaml.safe_dump(case, f, sort_keys=False, width=100)
    logging.info("setup placement case for seed %s: %.1f %% ok", seed, 100 * ok.mean())
    return folder
