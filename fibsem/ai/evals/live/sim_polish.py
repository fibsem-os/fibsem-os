"""Live polishing on the simulator: a lamella rough milled at a good cell,
which the agent polishes to the target thickness pass by pass, measuring as
it goes, and stops: there, or at the first sign of damage.

The case holds the microscope configuration its App Server connects (the
simulated Arctis imaging a seeded grid, as for live site selection) and the
set-up that rough mills the lamella. The truth is the simulator's: the run's
App Server writes the sample's record (each pass's true gap between the
trenches, and where damage started), and the run is scored on it
(``fibsem.ai.evals.tasks.live_polishing``).
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

from fibsem.ai.evals.cases import FORMAT, Case
from fibsem.ai.evals.live import sim_sites
from fibsem.ai.evals.live.client import LiveClient
from fibsem.ai.evals.live.sim_correction import _pick_cell
from fibsem.ai.evals.sim_cases import TIER

TASK = "live_polishing"
ITEM = "Lam-01"
TARGET_NM = 200
# within this above the target counts as there; thinner than the target by
# more than this is too far (nm), as for the static polishing cases
REACHED_NM = 50
OVER_NM = 60

BRIEF = (
    "Lamella {item} has been rough milled at the milling angle, at a cell on "
    "grid {grid}: its trenches leave about 1.5 um. Polish it to {target} nm "
    "pass by pass (milling_run_stage, with the item's Polishing stage as the "
    "start), measuring the band after every pass (a FIB image of the lamella, "
    "item_measure_thickness), and stop when it is at the target, or at once "
    "if a pass shows damage (a ragged face, a crack, a break). Say how thick "
    "it is where you stopped, how many passes you milled, and whether you saw "
    "damage."
)


def rough_mill(operator: LiveClient, case: Case) -> None:
    """The set-up's rough milling: the item's own Rough Milling task, as the
    workflow would run it."""
    item = case.key["item"]
    config = operator.call(
        "item_get_task_config", item_name=item, task_name="Rough Milling"
    )
    operator.call(
        "milling_run_task", config=config["config"]["milling"]["mill_rough"], path=item
    )


def write_case(
    seed: int, out: Path, overrides: Optional[Dict[str, Any]] = None, **_: Any
) -> Path:
    """The live polishing case for one seed under ``out/live-polishing/``."""
    folder = Path(out) / "live-polishing" / f"sim-{int(seed):04d}"
    folder.mkdir(parents=True, exist_ok=True)
    config = sim_sites.microscope_config(seed, overrides)
    (folder / sim_sites.MICROSCOPE).write_text(yaml.safe_dump(config, sort_keys=False))
    grid_key = sim_sites.build_key(folder / sim_sites.MICROSCOPE)
    cell = _pick_cell(grid_key, np.random.default_rng(int(seed)))
    microscope = sim_sites._session(folder / sim_sites.MICROSCOPE)
    try:
        optics = sim_sites.optics_key(microscope, cell["x"], cell["y"])
    finally:
        microscope.disconnect()
    at = {
        "x": cell["x"],
        "y": cell["y"],
        "z": optics["coincident_z"],
        "r": 0.0,
        "t": 0.0,
    }
    case = {
        "id": f"sim-{int(seed):04d}-live-polishing",
        "task": TASK,
        "tier": TIER,
        "playbook": "autolamella",
        "format": FORMAT,
        "sample": "simulated rpe cells on a holey film",
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
                # a good cell, coincident
                {"tool": "stage_move_absolute", "arguments": at},
                {
                    "tool": "stage_move_to_orientation",
                    "arguments": {"orientation": "MILLING"},
                },
                {"tool": "item_add", "arguments": {"name": ITEM}},
                {"prepare": "fibsem.ai.evals.live.sim_polish:rough_mill"},
            ],
        },
        "brief": BRIEF.format(item=ITEM, grid=sim_sites.GRID, target=TARGET_NM),
        "scoring": {
            "target_nm": TARGET_NM,
            "reached_nm": REACHED_NM,
            "over_nm": OVER_NM,
        },
        "key": {
            "status": "keyed",
            "keyed_by": "simulator, from the run's own record",
            "item": ITEM,
            "cell": cell,
            "optics": optics,
        },
        "traps": [
            "The simulator mills a little off what is drawn: the gap a pass "
            "leaves is not its pattern's spacing. Measure, don't assume.",
            "Below a seeded thickness a pass may start damage that grows on "
            "every pass after: stop at the first sign of it.",
        ],
    }
    (folder / "case.yaml").write_text(yaml.safe_dump(case, sort_keys=False))
    return folder
