"""Live coincidence correction on the simulator: the stage is at a good
cell but at the wrong height, so the FIB is not looking where the SEM is.
The agent makes it coincident, checks that it is, and saves the site there.

The case holds the microscope configuration its App Server connects (the
simulated Arctis imaging a seeded grid, as for live site selection), the
set-up that puts the stage off coincidence, and a key with the height at
which the SEM's and FIB's view centres meet at that cell. The run's saved
site is scored against it (``fibsem.ai.evals.tasks.coincidence_correction.task_live``).
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

from fibsem.ai.evals.core.cases import FORMAT
from fibsem.ai.evals.sim import live_sites as sim_sites
from fibsem.ai.evals.sim.grid import TIER

TASK = "live_coincidence_correction"
SITE = "Site-01"
# How far off the coincident height the stage starts (m): far enough that
# the beams' centres are tens of microns apart, so a site image may not
# show the same feature in both and the holey film offers look-alikes
OFF_Z = (30e-6, 60e-6)
# Coincident within this (m): what placing a lamella needs, not what
# picking a site does
TOLERANCE = 2e-6

BRIEF = (
    "Grid {grid} is loaded and the stage is at the SEM orientation, over a "
    "cell that is a good site for a lamella. The stage is not at the right "
    "height: the FIB is not looking at the point the SEM is. Make the site "
    "coincident (the FIB centred on the same point as the SEM, within "
    "{tolerance:g} um: this site is for a lamella, not just a pick), check "
    "that it is, and save it there as a lamella named {site} (item_add). Say "
    "whether it is coincident, and how you know."
)


def realistic_config(seed: int, overrides: Optional[Dict[str, Any]] = None) -> Dict:
    """The live site selection microscope with the Arctis's own errors back
    on: the stage lands off where it reports, and the beams meet off the
    stage's height by a fixed offset and one for each current. A script
    that moves z by one measurement should not always be right."""
    config = sim_sites.microscope_config(seed, overrides)
    arctis = yaml.safe_load(sim_sites.ARCTIS.read_text())["sim"]
    sample = arctis["sample"]
    config["sim"]["sample"].update(
        {k: sample[k] for k in sim_sites.NO_BEAM_OFFSET if k in sample}
    )
    config["sim"]["stage_error"] = {**arctis["stage_error"], "seed": int(seed)}
    return config


def _pick_cell(key: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
    """A good cell (on film, clear of the bars, in a good square), the
    seed's pick of those nearest the stage's middle."""
    good_squares = {tuple(q["index"]) for q in key["squares"] if q["label"] == "good"}
    cells = [
        c
        for c in key["cells"]
        if c["on_film"] and c["clear"] and tuple(c["square"]) in good_squares
    ]
    if not cells:
        raise ValueError("no good cell on this grid: try another seed")
    cells.sort(key=lambda c: np.hypot(c["x"], c["y"]))
    return cells[int(rng.integers(0, min(5, len(cells))))]


def write_case(
    seed: int, out: Path, overrides: Optional[Dict[str, Any]] = None, **_: Any
) -> Path:
    """The live coincidence case for one seed under
    ``out/live-coincidence-correction/sim-NNNN``."""
    folder = Path(out) / "live-coincidence-correction" / f"sim-{int(seed):04d}"
    folder.mkdir(parents=True, exist_ok=True)
    config = realistic_config(seed, overrides)
    (folder / sim_sites.MICROSCOPE).write_text(yaml.safe_dump(config, sort_keys=False))
    grid_key = sim_sites.build_key(folder / sim_sites.MICROSCOPE)
    rng = np.random.default_rng(int(seed))
    cell = _pick_cell(grid_key, rng)
    microscope = sim_sites._session(folder / sim_sites.MICROSCOPE)
    try:
        optics = sim_sites.optics_key(microscope, cell["x"], cell["y"])
    finally:
        microscope.disconnect()
    sign = 1 if rng.random() < 0.5 else -1
    start_z = optics["coincident_z"] + sign * float(rng.uniform(*OFF_Z))
    start = {"x": cell["x"], "y": cell["y"], "z": round(start_z, 9), "r": 0.0, "t": 0.0}
    off_um = optics["coincidence_per_z"] * abs(start_z - optics["coincident_z"]) * 1e6
    case = {
        "id": f"sim-{int(seed):04d}-live-coincidence-correction",
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
                {"tool": "stage_move_absolute", "arguments": start},
            ],
        },
        "brief": BRIEF.format(
            grid=sim_sites.GRID, site=SITE, tolerance=TOLERANCE * 1e6
        ),
        "scoring": {"tolerance_um": TOLERANCE * 1e6},
        "key": {
            "status": "keyed",
            "keyed_by": "simulator",
            "site": SITE,
            "start": start,
            "start_off_um": round(off_um, 2),
            "cell": cell,
            "optics": optics,
        },
        "traps": [
            f"The beams start {off_um:.0f} um apart: a check at a site's "
            "field width can lock onto a look-alike hole and report a small "
            "offset; widen the field first, then narrow it, and check again "
            "after moving.",
            "Moving the stage in x or y moves off the cell: correct the "
            "height only (a vertical move).",
        ],
    }
    (folder / "case.yaml").write_text(yaml.safe_dump(case, sort_keys=False))
    return folder
