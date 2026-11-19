"""Live correlation on the simulator: a lamella on the Arctis at the milling
orientation, near an FM target (a cilium on odd seeds, a centriole pair on
even ones), with spot burns milled round the target from the FIB. The agent
takes its own FIB image and FM stack, correlates the FM to the FIB, and puts
the lamella's point of interest on the target.

The target is not at the centre of the FIB's view: it sits a few microns off
it, as a lamella placed by eye from an overview does, so a point of interest
left at the centre misses. The key is the scene's own: the target's midpoint
in the FIB's view at the lamella's milling pose, in metres from the image
centre (y up, as a point of interest is), and in pixels of a FIB image at the
burns' field width (``fibsem.ai.evals.tasks.correlation.task_live``).
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

from fibsem.ai.evals.core.cases import FORMAT, Case
from fibsem.ai.evals.sim import live_sites as sim_sites
from fibsem.ai.evals.sim.grid import TIER
from fibsem.ai.evals.sim.live_client import LiveClient
from fibsem.ai.evals.tasks.correlation.make import (
    FIB_HFW,
    _burn_points,
    _FibMap,
    scene_targets,
    target_kind,
)
from fibsem.structures.settings import BeamType, FibsemStagePosition

TASK = "live_correlation"
ITEM = "Lam-01"
# the scene's FM labels: cilia red, centrin green; every cell a target
FM = {
    "fm_red": "cilia",
    "fm_green": "centrin",
    "fm_target_fraction": 1.0,
    "fm_cilium_fraction": 1.0,
    "fm_no_target_chance": 0.0,
}
FIB_VIEW = (1536, 1024)  # the FIB image the burns are placed on
CENTRED = 0.1e-6  # m: how near the view's centre the target is first put
# how far the target sits from the view's centre (m): along the lamella (x)
# and across it (y), either way, so the centre is never the answer
OFF_ALONG = (3e-6, 10e-6)
OFF_ACROSS = (2e-6, 6e-6)
BURN_CURRENT = 60e-12  # A
BURN_EXPOSURE = 10.0  # s a spot
# the stage may not move further than this (m) from the set-up's pose: the
# key is the target's place in the view from there
SAME_POSE = 0.2e-6

BRIEF = (
    "Lamella {item} on grid {grid} is at the milling orientation on the "
    "Arctis, near a {what} ({where}). Its spot burns were milled from the FIB "
    "at a {hfw:g} um field width, at these fractions of the image (x, y from "
    "the top left): {burns}. Without moving the stage, take a FIB image at "
    "{hfw:g} um and an FM z-stack, find the target and the burns in both, "
    "correlate the FM to the FIB, and put {item}'s point of interest on the "
    "target. Say where the target is in your FIB image: its pixels, and "
    "metres from the image centre."
)


def _target(config_path: Path, rng: np.random.Generator, kind: str) -> Dict[str, Any]:
    """A target near the stage's middle, reachable, and the milling pose that
    shows it a few microns off the centre of the FIB's view; the burns round
    it, as fractions of a FIB image from there, and where the target is in
    that image."""
    microscope = sim_sites._session(config_path)
    try:
        scene = microscope._connection.sample_scene
        world_map = sim_sites.stage_to_world(microscope)
        found = []
        for target in scene_targets(scene, kind):
            x, y, z = target["midpoint"]
            sx, sy = sim_sites._to_stage(world_map, x, y)
            if abs(sx) < sim_sites.REACH[0] and abs(sy) < sim_sites.REACH[1]:
                found.append((np.hypot(sx, sy), sx, sy, (x, y, z)))
        if not found:
            raise ValueError(f"no {kind} in reach on this grid: try another seed")
        found.sort(key=lambda f: f[0])
        _, sx, sy, mid = found[int(rng.integers(0, min(5, len(found))))]
        optics = sim_sites.optics_key(microscope, sx, sy)
        microscope.move_to_orientation("SEM")
        microscope.move_stage_absolute(
            FibsemStagePosition(x=sx, y=sy, z=optics["coincident_z"], r=0, t=0)
        )
        microscope.move_to_orientation("MILLING")
        x, y, above = mid
        z = above + scene._fm_film()
        seen = None
        for _ in range(6):
            fib = _FibMap(microscope, scene, FIB_VIEW, FIB_HFW)
            seen = fib.view(x, y, z)
            if np.hypot(*seen) <= CENTRED:
                break
            microscope.stable_move(dx=seen[0], dy=-seen[1], beam_type=BeamType.ION)
        if seen is None or np.hypot(*seen) > CENTRED:
            raise ValueError(f"could not centre the {kind} in the FIB's view")
        along = rng.uniform(*OFF_ALONG) * rng.choice((-1, 1))
        across = rng.uniform(*OFF_ACROSS) * rng.choice((-1, 1))
        # the view moves the other way to the target: the stage moves it off
        microscope.stable_move(dx=-along, dy=-across, beam_type=BeamType.ION)
        pose = microscope.get_stage_position()
        fib = _FibMap(microscope, scene, FIB_VIEW, FIB_HFW)
        vx, vy = fib.view(x, y, z)
        px, py = fib.px(x, y, z)
        burns = _burn_points(rng, x, y, fib)
        width, height = FIB_VIEW
        fractions = []
        for bx, by in burns:
            bpx, bpy = fib.px(bx, by)
            fractions.append([round(bpx / width, 4), round(bpy / height, 4)])
    finally:
        microscope.disconnect()
    return {
        "kind": kind,
        "midpoint": [float(v) for v in mid],
        "stage": {"x": round(sx, 9), "y": round(sy, 9)},
        "optics": optics,
        "milling_pose": {
            k: round(float(getattr(pose, k)), 9) for k in ("x", "y", "z", "r", "t")
        },
        # metres from the image centre, y up, as a point of interest is
        "poi_m": [round(float(vx), 9), round(float(-vy), 9)],
        "fib_px": [round(px, 1), round(py, 1)],
        "burns": fractions,
    }


def burn(operator: LiveClient, case: Case) -> None:
    """The set-up's spot burns, from the FIB at the lamella's pose, as the
    Spot Burn task mills them."""
    # the burns are fractions of the ion beam's field as it is set: the
    # field width they were placed at, as an image there sets it
    operator.call(
        "beam_acquire_image",
        beam_type="ION",
        image_settings={"hfw": FIB_HFW, "resolution": list(FIB_VIEW)},
        path=case.key["item"],
        filename="before_burns",
    )
    coordinates = [
        {"x": x, "y": y, "name": f"B{i + 1}"}
        for i, (x, y) in enumerate(case.key["burns"])
    ]
    operator.call(
        "milling_run_spot_burn",
        settings={
            "coordinates": coordinates,
            "milling_current": BURN_CURRENT,
            "exposure_time": BURN_EXPOSURE,
            "align": False,
        },
    )


# what the target is, and where the FM shows it
WHAT = {
    "cilium": ("primary cilium", "red in the FM, the cilia channel"),
    "centriole": ("centriole pair", "green in the FM, the centrin channel"),
}


def _where(kind: str) -> str:
    return f"{WHAT[kind][1]}; the FM is coincident with the FIB"


def write_case(
    seed: int, out: Path, overrides: Optional[Dict[str, Any]] = None, **_: Any
) -> Path:
    """The live correlation case for one seed under
    ``out/live-correlation/``."""
    folder = Path(out) / "live-correlation" / f"sim-{int(seed):04d}"
    folder.mkdir(parents=True, exist_ok=True)
    config = sim_sites.microscope_config(seed, {**FM, **(overrides or {})})
    (folder / sim_sites.MICROSCOPE).write_text(yaml.safe_dump(config, sort_keys=False))
    rng = np.random.default_rng([int(seed), 17])
    kind = target_kind(seed)
    target = _target(folder / sim_sites.MICROSCOPE, rng, kind)
    at = {
        "x": target["stage"]["x"],
        "y": target["stage"]["y"],
        "z": target["optics"]["coincident_z"],
        "r": 0.0,
        "t": 0.0,
    }
    shown = ", ".join(f"({x:.3f}, {y:.3f})" for x, y in target["burns"])
    case = {
        "id": f"sim-{int(seed):04d}-live-correlation",
        "task": TASK,
        "tier": TIER,
        "playbook": "autolamella",
        "format": FORMAT,
        "sample": "simulated rpe cells on a holey film, cilia red, centrin green",
        "instrument": "Demo (simulator), Arctis",
        "source": {"simulator": {"seed": int(seed), "overrides": overrides or {}}},
        "sim_run": {
            "microscope": sim_sites.MICROSCOPE,
            "level": "drive",
            "setup": [
                {"tool": "grid_load", "arguments": {"grid_name": sim_sites.GRID}},
                {
                    "tool": "stage_move_to_orientation",
                    "arguments": {"orientation": "SEM"},
                },
                {"tool": "stage_move_absolute", "arguments": at},
                {
                    "tool": "stage_move_to_orientation",
                    "arguments": {"orientation": "MILLING"},
                },
                # the target a few microns off the centre of the FIB's view
                {
                    "tool": "stage_move_absolute",
                    "arguments": target["milling_pose"],
                },
                {"tool": "item_add", "arguments": {"name": ITEM}},
                {"prepare": "fibsem.ai.evals.tasks.correlation.make_live:burn"},
            ],
        },
        "brief": BRIEF.format(
            item=ITEM,
            grid=sim_sites.GRID,
            what=WHAT[kind][0],
            where=_where(kind),
            hfw=FIB_HFW * 1e6,
            burns=shown,
        ),
        "scoring": {
            "across_tolerance_um": 0.75,
            "along_tolerance_um": 1.0,
            "same_pose_um": SAME_POSE * 1e6,
        },
        "key": {
            "status": "keyed",
            "keyed_by": "simulator",
            "item": ITEM,
            "target_kind": kind,
            "target": target["midpoint"],
            "target_poi_m": target["poi_m"],
            "target_fib_px": target["fib_px"],
            "fib_size_px": list(FIB_VIEW),
            "fib_hfw_m": FIB_HFW,
            "milling_pose": target["milling_pose"],
            "burns": target["burns"],
        },
        "traps": [
            "The target is a few microns off the centre of the FIB's view: a "
            "point of interest left at the centre misses.",
            "The simulator's FM has no refractive index: a stack's z is the true "
            "height, so an answer that applies an RI correction moves the target "
            "off the key.",
        ],
    }
    (folder / "case.yaml").write_text(yaml.safe_dump(case, sort_keys=False))
    return folder
