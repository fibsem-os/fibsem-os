"""An FM target eval case from the simulator, keyed from the scene.

A grid of RPE cells on the Demo Arctis, with centrin (GFP) and cilia
(mCherry) labelled in the FM. The stage is brought to one cell, as a site
saved for a lamella is, and the case keeps an FM z-stack (GFP, mCherry, DAPI)
of it. The agent answers whether there is a usable target at the site, and
where its midpoint is in the stack. The target is a primary cilium on odd
seeds and a centriole pair on even ones.

One seed in four has none: for a cilium, the cells carry their centriole
pairs but grew no cilium; for a pair, the cells there carry no centrin.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from fibsem.ai.evals.cases import FORMAT
from fibsem.ai.evals.sim_cases import TIER
from fibsem.ai.evals.sim_correlation import CHANNELS as _ALL_CHANNELS
from fibsem.ai.evals.sim_correlation import (
    KINDS,
    MILLING_ANGLE,
    Z_RANGE,
    Z_STEP,
    _arctis,
    _fm_px,
    scene_targets,
    target_kind,
)
from fibsem.drivers.demo.sim_scene import SampleScene
from fibsem.fm.acquisition import acquire_z_stack
from fibsem.fm.structures import ZParameters
from fibsem.structures.settings import FibsemStagePosition

logger = logging.getLogger(__name__)

PLAYBOOK = "autolamella-correlative"
CHANNELS = [c for c in _ALL_CHANNELS if c.name != "reflection"]
CLEAR = 8e-6  # m: no other target this close to the target
OFFSET = (1.5e-6, 3.5e-6)  # m: the site's centre is this far off the target
NAMES = {"cilium": "a primary cilium", "centriole": "a centriole pair"}

BRIEF = (
    "This FM z-stack was taken at a site saved for a lamella; the site is the "
    "centre of the image. mCherry labels primary cilia, GFP centrin, DAPI "
    "nuclei. Is there a usable target at the site: {what} within a few microns "
    "of the centre? If yes, answer its midpoint as x, y px on the stack and the "
    "z slice where it is sharpest. If there is none, answer usable: no."
)

HOT = "Hot camera pixels are single bright pixels in every slice."
PUNCTA = "Bright mCherry dots through the cells are vesicles, not cilia."
TRAPS = {
    "cilium": [
        HOT + " A cilium is a curved line a few microns long, sharp in a few slices.",
        PUNCTA,
    ],
    "centriole": [
        HOT + " A centriole pair is two GFP spots about a micron apart, at the "
        "edge of a dim centrin pool, sharp in a few slices.",
        "A cilium (mCherry) grows from a pair; the target is the pair, not the cilium.",
    ],
}
NONE_TRAPS = {
    "cilium": [
        "Centriole pairs (GFP) sit where a cilium would grow; they are not a cilium."
    ],
    "centriole": [
        "Ice contamination glows faintly in GFP; it is a broad blob, not a pair."
    ],
}


def scene_options(seed: int, overrides: Optional[Dict[str, Any]] = None) -> Dict:
    options: Dict[str, Any] = {
        "cell_type": "rpe",
        "fm_red": "cilia",
        "fm_green": "centrin",
        **_targets(seed),
        "fm_no_target_chance": 0.0,
        "coincidence_offset": 0.0,
        "rip_fraction": 0.0,
        "rpe_density": 3.0,
    }
    options.update(overrides or {})
    options["seed"] = int(seed)
    return options


def has_none(seed: int) -> bool:
    """Seeds 3 and 8 of every eight: one cilium case, one centriole case."""
    return int(seed) % 8 in (0, 3)


def _targets(seed: int) -> Dict[str, float]:
    if target_kind(seed) == "cilium":
        cilia = 0.0 if has_none(seed) else 0.6
        return {"fm_target_fraction": 1.0, "fm_cilium_fraction": cilia}
    pairs = 0.0 if has_none(seed) else 1.0
    return {"fm_target_fraction": pairs, "fm_cilium_fraction": 0.3}


def _cell_bodies(scene) -> List:
    return [f for f in scene.features if f.kind == "cell" and f.part == "body"]


def _pick_site(
    rng, scene, kind: str, nearest: int = 20
) -> Optional[Tuple[float, float, Any]]:
    """(x, y, target or None): a site near the scene's centre. With targets of
    the kind, one with no other within CLEAR; without, a cell body."""
    targets = sorted(
        scene_targets(scene, kind), key=lambda t: np.hypot(*t["midpoint"][:2])
    )
    if targets:
        order = rng.permutation(min(nearest, len(targets)))
        for i in order:
            t = targets[int(i)]
            x, y = t["midpoint"][:2]
            others = [
                np.hypot(o["midpoint"][0] - x, o["midpoint"][1] - y)
                for o in targets
                if o is not t
            ]
            if not others or min(others) >= CLEAR:
                return float(x), float(y), t
        return None
    cells = sorted(_cell_bodies(scene), key=lambda f: np.hypot(f.x, f.y))
    if not cells:
        return None
    cell = cells[int(rng.integers(min(nearest, len(cells))))]
    return float(cell.x), float(cell.y), None


def write_case(
    seed: int,
    out: Path,
    overrides: Optional[Dict[str, Any]] = None,
    z_range: Tuple[float, float] = Z_RANGE,
    channels: Optional[List] = None,
) -> Optional[Path]:
    """The FM target case for one seed under ``out/fm-target/``, or None when
    the scene has no clear site."""
    import tifffile

    folder = Path(out) / "fm-target" / f"sim-{int(seed):04d}"
    experiment = folder / "experiment"
    rng = np.random.default_rng([int(seed), 271])
    channels = channels or CHANNELS
    microscope = _arctis()
    try:
        microscope.move_to_milling_angle(MILLING_ANGLE)
        scene = SampleScene(**scene_options(seed, overrides))
        scene.anchor(microscope.get_stage_position())
        microscope._connection.sample_scene = scene
        kind = target_kind(seed)
        site = _pick_site(rng, scene, kind)
        if site is None:
            logger.warning("seed %s: no clear site, no case", seed)
            return None
        sx, sy, target = site
        # a saved site is near its target, not on it
        a, r = rng.uniform(0, 2 * np.pi), rng.uniform(*OFFSET)
        cx, cy = sx + r * np.cos(a), sy + r * np.sin(a)
        # the scene's y runs against the stage's
        microscope.move_stage_relative(FibsemStagePosition(x=cx, y=-cy, z=0))
        microscope.move_to_device("FM")
        fm = microscope.fm
        focus = fm.objective.focus_position
        fm.objective.move_absolute(focus)
        zparams = ZParameters(zmin=z_range[0], zmax=z_range[1], zstep=Z_STEP)
        z_positions = list(zparams.generate_positions(z_init=focus))
        stack = acquire_z_stack(fm, channels, zparams)
        experiment.mkdir(parents=True, exist_ok=True)
        stack_name = "fm-zstack.ome.tiff"
        stack.save(str(experiment / stack_name))
        fm_pixel = float(fm.camera.pixel_size[0])
        film = scene._fm_film()  # targets are above the film, focus is on the ice
        width, height = fm.camera.resolution
        site_fm = _fm_px(microscope, scene, sx, sy)
        others = [
            (_fm_px(microscope, scene, *t["midpoint"][:2]), t["midpoint"][2] + film)
            for t in scene_targets(scene, kind)
            if target is None or not np.allclose(t["midpoint"], target["midpoint"])
        ]
    finally:
        microscope.disconnect()

    def slice_of(z: float) -> float:
        offsets = np.array(z_positions) - focus
        return float(np.interp(z, offsets, np.arange(len(offsets))))

    with tifffile.TiffFile(experiment / stack_name) as tif:
        series = tif.series[0]
        shape, axes = [int(n) for n in series.shape], series.axes

    others_shown = [
        {"x_px": round(x, 1), "y_px": round(y, 1), "z_slice": round(slice_of(z), 2)}
        for (x, y), z in others
        if 0 <= x < width and 0 <= y < height
    ]
    if target is not None:
        tz = float(target["midpoint"][2]) + film
        key: Dict[str, Any] = {
            "usable": "yes",
            "target": {
                "x_px": round(site_fm[0], 1),
                "y_px": round(site_fm[1], 1),
                "z_slice": round(slice_of(tz), 2),
            },
            "reason": f"the site's cell has {NAMES[kind]}",
        }
    else:
        key = {
            "usable": "no",
            "target": None,
            "reason": (
                "the cells here have centriole pairs but no cilium"
                if kind == "cilium"
                else "the cells here carry no centrin"
            ),
        }
    key.update(
        target_kind=kind,
        other_candidates=others_shown,
        status="keyed",
        keyed_by="simulator",
    )
    case = {
        "id": f"sim-{int(seed):04d}-fm-target",
        "task": "fm_target",
        "tier": TIER,
        "playbook": PLAYBOOK,
        "format": FORMAT,
        "sample": "simulated RPE cells, centrin (GFP) and cilia (mCherry) labelled",
        "instrument": "Demo Arctis (simulator)",
        "source": {
            "simulator": {
                "seed": int(seed),
                "scene": scene_options(seed, overrides),
                "site_m": [cx, cy],
            }
        },
        "brief": BRIEF.format(what=NAMES[kind]),
        "stack": {
            "file": f"experiment/{stack_name}",
            "axes": axes,
            "shape": shape,
            "channels": [c.name for c in channels],
            "pixel_size_um": round(fm_pixel * 1e6, 4),
            "z_step_um": round(Z_STEP * 1e6, 3),
            "target_channel": KINDS[kind][1],
        },
        "key": key,
        "scoring": {"xy_tolerance_um": 1.0, "z_tolerance_slices": 1},
        "traps": TRAPS[kind] + (NONE_TRAPS[kind] if target is None else []),
    }
    with open(folder / "case.yaml", "w") as f:
        yaml.safe_dump(case, f, sort_keys=False, width=100)
    if target is not None:
        reference = folder / "reference"
        reference.mkdir(parents=True, exist_ok=True)
        (reference / "target.yaml").write_text(
            yaml.safe_dump(
                {
                    "midpoint_m": [float(v) for v in target["midpoint"]],
                    "kind": kind,
                    "cilium_m": (
                        [[float(v) for v in p] for p in target["cilium"]]
                        if kind == "cilium"
                        else None
                    ),
                    "pair_m": [[float(v) for v in p] for p in target["pair"]],
                },
                sort_keys=False,
            )
        )
    logger.info("fm target case for seed %s: %s", seed, key["usable"])
    return folder
