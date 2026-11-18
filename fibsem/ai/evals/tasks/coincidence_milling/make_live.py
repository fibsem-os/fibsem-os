"""Live coincidence milling on the simulator: a lamella rough milled through
a cell's primary cilium (an FM target) on the Arctis, whose FM looks where
the FIB does. The agent polishes it to the target thickness and keeps the
cilium in it, with FM stacks between passes.

The lamella is centred on the cilium, except on every third seed, where it
is rough milled 0.5 um off it: the cilium survives the rough mill, but
passes polished as placed mill it away, so an agent must see it go in the
FM and move the passes. The truth is the run's own sample record: each pass's true
gap, the damage, and whether the cilium's midpoint was milled away
(``fibsem.ai.evals.tasks.coincidence_milling.task_live``).
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

from fibsem.ai.evals.core.cases import FORMAT
from fibsem.ai.evals.sim import live_sites as sim_sites
from fibsem.ai.evals.sim.grid import TIER
from fibsem.ai.evals.tasks.correlation.make import FIB_HFW, _FibMap
from fibsem.ai.evals.tasks.polishing import make_live as sim_polish
from fibsem.structures.settings import BeamType, FibsemStagePosition

TASK = "live_coincidence_milling"
# the scene's FM labels: red on cilia, green on centrin; every cell a target
FM = {
    "fm_red": "cilia",
    "fm_green": "centrin",
    "fm_target_fraction": 1.0,
    "fm_cilium_fraction": 1.0,
    "fm_no_target_chance": 0.0,
}
# how far the lamella is rough milled off the cilium, across it (m), on the
# off-target seeds: inside the rough lamella, so the rough mill keeps it,
# but out of a 200 nm one centred where it was placed (at 1.2 um the rough
# trenches already take it)
OFF = 0.5e-6
# the FIB view the cilium is centred in, and how near its centre (m)
FIB_VIEW = (1536, 1024)
CENTRED = 0.1e-6

BRIEF = (
    "Lamella {item} has been rough milled at the milling angle through a cell "
    "on grid {grid}, aiming at its primary cilium (red in the FM; the FM is "
    "coincident with the FIB): its trenches leave about 1.5 um. Polish it to "
    "{target} nm pass by pass, keeping the cilium in the lamella: an FM stack "
    "after every pass, and move the passes if the cilium is not centred in the "
    "band. Measure the band after every pass, and stop at the target, or at "
    "once on damage. Say how thick it is where you stopped, how many passes "
    "you milled, whether you saw damage, and whether the cilium is still in "
    "the lamella."
)


def off_target(seed: int) -> bool:
    """Every third seed is placed off the cilium."""
    return int(seed) % 3 == 0


def _cilium(
    config_path: Path, rng: np.random.Generator, off: float = 0.0
) -> Dict[str, Any]:
    """A cilium near the stage's middle, reachable: where it is in the world
    (its midpoint, for the record), and the stage pose at the milling
    orientation that puts it at the centre of the FIB's view, where the rough
    lamella is milled. The cilium sits microns above the film, and the
    lamella is cut at the milling angle, so the pose is found in the FIB's own
    view rather than from the cilium's x, y at the SEM orientation. *off* (m)
    moves the pose across the lamella, off the cilium."""
    microscope = sim_sites._session(config_path)
    try:
        scene = microscope._connection.sample_scene
        world_map = sim_sites.stage_to_world(microscope)
        found = []
        for target in scene.fm_targets():
            x, y, z = target["midpoint"]
            sx, sy = sim_sites._to_stage(world_map, x, y)
            if abs(sx) < sim_sites.REACH[0] and abs(sy) < sim_sites.REACH[1]:
                found.append((np.hypot(sx, sy), sx, sy, (x, y, z)))
        if not found:
            raise ValueError("no cilium in reach on this grid: try another seed")
        found.sort(key=lambda f: f[0])
        _, sx, sy, mid = found[int(rng.integers(0, min(5, len(found))))]
        optics = sim_sites.optics_key(microscope, sx, sy)
        microscope.move_to_orientation("SEM")
        microscope.move_stage_absolute(
            FibsemStagePosition(x=sx, y=sy, z=optics["coincident_z"], r=0, t=0)
        )
        microscope.move_to_orientation("MILLING")
        x, y, above = mid
        seen = None
        for _ in range(6):
            fib = _FibMap(microscope, scene, FIB_VIEW, FIB_HFW)
            seen = fib.view(x, y, above + scene._fm_film())
            if np.hypot(*seen) <= CENTRED:
                break
            microscope.stable_move(dx=seen[0], dy=-seen[1], beam_type=BeamType.ION)
        if seen is None or np.hypot(*seen) > CENTRED:
            raise ValueError("could not centre the cilium in the FIB's view")
        if off:
            microscope.stable_move(dx=0.0, dy=off, beam_type=BeamType.ION)
        pose = microscope.get_stage_position()
    finally:
        microscope.disconnect()
    return {
        "midpoint": [float(v) for v in mid],
        "stage": {"x": round(sx, 9), "y": round(sy, 9)},
        "milling_pose": {
            k: round(float(getattr(pose, k)), 9) for k in ("x", "y", "z", "r", "t")
        },
        "optics": optics,
    }


def write_case(
    seed: int, out: Path, overrides: Optional[Dict[str, Any]] = None, **_: Any
) -> Path:
    """The live coincidence milling case for one seed under
    ``out/live-coincidence-milling/``."""
    folder = Path(out) / "live-coincidence-milling" / f"sim-{int(seed):04d}"
    folder.mkdir(parents=True, exist_ok=True)
    config = sim_sites.microscope_config(seed, {**FM, **(overrides or {})})
    (folder / sim_sites.MICROSCOPE).write_text(yaml.safe_dump(config, sort_keys=False))
    rng = np.random.default_rng([int(seed), 5])
    off = OFF * (1 if rng.random() < 0.5 else -1) if off_target(seed) else 0.0
    cilium = _cilium(folder / sim_sites.MICROSCOPE, rng, off)
    at = {
        "x": cilium["stage"]["x"],
        "y": cilium["stage"]["y"],
        "z": cilium["optics"]["coincident_z"],
        "r": 0.0,
        "t": 0.0,
    }
    item = sim_polish.ITEM
    case = {
        "id": f"sim-{int(seed):04d}-live-coincidence-milling",
        "task": TASK,
        "tier": TIER,
        "playbook": "autolamella",
        "format": FORMAT,
        "sample": "simulated rpe cells on a holey film, cilia labelled red",
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
                # the cilium at the centre of the FIB's view (or off it)
                {
                    "tool": "stage_move_absolute",
                    "arguments": cilium["milling_pose"],
                },
                {"tool": "item_add", "arguments": {"name": item}},
                {"prepare": "fibsem.ai.evals.tasks.polishing.make_live:rough_mill"},
            ],
        },
        "brief": BRIEF.format(
            item=item, grid=sim_sites.GRID, target=sim_polish.TARGET_NM
        ),
        "scoring": {
            "target_nm": sim_polish.TARGET_NM,
            "reached_nm": sim_polish.REACHED_NM,
            "over_nm": sim_polish.OVER_NM,
        },
        "key": {
            "status": "keyed",
            "keyed_by": "simulator, from the run's own record",
            "item": item,
            "cilium": cilium["midpoint"],
            "off_target_um": round(off * 1e6, 2),
            "optics": cilium["optics"],
        },
        "traps": [
            "The lamella is rough milled off the cilium on some cases: "
            "polished as placed, the passes mill it away.",
            "Below a seeded thickness a pass may start damage: stop at once.",
        ],
    }
    (folder / "case.yaml").write_text(yaml.safe_dump(case, sort_keys=False))
    return folder
