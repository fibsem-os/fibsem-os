"""Live coincidence milling on the simulator: a lamella rough milled through
a cell's primary cilium (an FM target) on the Arctis, whose FM looks where
the FIB does. The agent polishes it to the target thickness and keeps the
cilium in it, with FM stacks between passes.

On most seeds the lamella is centred on the cilium. On seeds 3, 6, 9, ...
it is rough milled off it, across the lamella: polished as placed, the
passes mill the cilium away, so the agent has to see that in the FM and
move its passes. The truth is the run's own sample record: each pass's true
gap, the damage, and whether the cilium's midpoint was milled away
(``fibsem.ai.evals.tasks.live_coincidence_milling``).
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

from fibsem.ai.evals.cases import FORMAT
from fibsem.ai.evals.live import sim_polish, sim_sites
from fibsem.ai.evals.sim_cases import TIER

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
# off-target seeds: past half the rough lamella, so it is milled away
OFF = 1.2e-6

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
    return int(seed) % 3 == 0


def _cilium(config_path: Path, rng: np.random.Generator) -> Dict[str, Any]:
    """A cilium near the stage's middle, reachable: where it is in the world
    (its midpoint, for the record) and at the SEM orientation's stage x, y."""
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
    finally:
        microscope.disconnect()
    return {
        "midpoint": [float(v) for v in mid],
        "stage": {"x": round(sx, 9), "y": round(sy, 9)},
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
    cilium = _cilium(folder / sim_sites.MICROSCOPE, rng)
    off = OFF * (1 if rng.random() < 0.5 else -1) if off_target(seed) else 0.0
    at = {
        "x": cilium["stage"]["x"],
        "y": round(cilium["stage"]["y"] + off, 9),
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
                {"tool": "item_add", "arguments": {"name": item}},
                {"prepare": "fibsem.ai.evals.live.sim_polish:rough_mill"},
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
