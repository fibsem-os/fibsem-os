"""A rough_milling eval case from the simulator, keyed from the 3D sample.

One lamella is rough milled on the Demo microscope at the milling angle, in
three steps: rough 1, rough 2 and stress relief. After each step the case
keeps an SEM and a FIB image. The agent says, step by step, whether the step
was milled through and whether the lamella is still in one piece.

Most seeds mill cleanly. Seeds 2, 6, 10, ... set one step too shallow, so its
trenches stop short of the film. Seeds 3, 7, 11, ... break the lamella in
rough 2 (a ragged face, a crack or a piece out), and the damage grows in the
step after.

The key is read from the sample after each step: a step is milled through
when the points just above the film under its trenches are cut away, and the
lamella is intact until the scene records damage.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from fibsem.ai.evals.cases import FORMAT
from fibsem.ai.evals.sim_cases import PLAYBOOK, TIER, _demo_microscope
from fibsem.ai.evals.sim_polishing import (
    DEPTH,
    DWELL,
    HFW,
    MILLING_ANGLE,
    OVERLAP,
    RESOLUTION,
    WIDTH,
    _site_cell,
)
from fibsem.ai.evals.sim_polishing import scene_options as _polishing_options
from fibsem.drivers.demo import sim_solid, sim_view
from fibsem.drivers.demo.sim_scene import SampleScene
from fibsem.structures.settings import (
    BeamType,
    FibsemRectangleSettings,
    ImageSettings,
)

STEPS = ("rough-1", "rough-2", "stress-relief")
GAPS = (3.0e-6, 1.0e-6)  # m: the lamella after rough 1, then rough 2
ROUGH_HEIGHT = 8e-6  # m, each rough 1 trench
RELIEF = (1.0e-6, 14e-6)  # m: each stress relief cut's width and height
RELIEF_OUT = 3e-6  # m: the cuts' distance out from the lamella's ends
SHALLOW = (0.08, 0.15)  # a shallow step's depth, as a fraction of DEPTH
THROUGH = 0.8  # the fraction of a step's floor cut away to count as through
FLOOR = 0.2e-6  # m above the film: where a step's floor is tested
BREAK = 3  # passes of damage a break shows when it happens

BRIEF = (
    "This lamella was rough milled in the steps listed below, on a {angle:g} "
    "degree milling angle: rough 1, rough 2, then stress relief cuts either "
    "side. After each step there is an SEM image and a FIB image. For each "
    "step, say whether it was milled through (its trenches reach the bottom, "
    "with no material left across them) and whether the lamella is still "
    "intact (no crack, no piece broken out, the face not eaten away)."
)


def fault(seed: int) -> str:
    """What goes wrong on a seed: nothing, a shallow step or a break."""
    return {2: "shallow", 3: "broken"}.get(int(seed) % 4, "none")


def scene_options(seed: int, overrides: Optional[Dict[str, Any]] = None) -> Dict:
    options = _polishing_options(seed, {})
    options["damage_chance"] = 0.0  # a break is set where it happens
    options.update(overrides or {})
    options["seed"] = int(seed)
    return options


def _patterns(step: str, depth: float) -> List[FibsemRectangleSettings]:
    if step == "stress-relief":
        width, height = RELIEF
        return [
            FibsemRectangleSettings(
                width=width,
                height=height,
                depth=depth,
                centre_x=sign * (WIDTH / 2 + RELIEF_OUT),
                centre_y=0.0,
            )
            for sign in (1, -1)
        ]
    if step == "rough-1":
        gap, height = GAPS[0], ROUGH_HEIGHT
    else:
        gap, height = GAPS[1], (GAPS[0] - GAPS[1]) / 2 + OVERLAP
    return [
        FibsemRectangleSettings(
            width=WIDTH,
            height=height,
            depth=depth,
            centre_x=0.0,
            centre_y=sign * (gap / 2 + height / 2),
        )
        for sign in (1, -1)
    ]


def _through(scene, regions) -> float:
    """How much of a step's floor its cuts reached, against the same cuts at
    full depth (1: all the way down). Each column a cut passes through is cut
    from its top down to the pattern's depth; the floor is reached where the
    cut goes below FLOOR over the film."""
    um = sim_view.UM
    floor = sim_solid.FILM_TOP + FLOOR * um
    reached = full = 0
    for region in regions:
        pts = np.asarray(region.points) * um
        lo, hi = pts.min(axis=0) - 25.0, pts.max(axis=0) + 25.0
        x = np.arange(lo[0], hi[0], 0.1)
        y = np.arange(lo[1], hi[1], 0.1)
        col = sim_solid.Plan(scene, x, y).col
        surface, level = col["surface"], col["level"]
        X, Y = np.meshgrid(x, y)
        a, b = sim_view._cut_range(scene, region, X, Y, surface)
        hit = (b > a) & (surface > floor)
        for depth, count in ((region.depth, "reached"), (DEPTH, "full")):
            bottom = np.maximum(a, np.minimum(surface, level - depth * um))
            n = int((hit & (bottom <= floor)).sum())
            if count == "reached":
                reached += n
            else:
                full += n
    return reached / full if full else 0.0


def mill_steps(
    seed: int,
    folder: Path,
    overrides: Optional[Dict[str, Any]] = None,
    milling_angle: float = MILLING_ANGLE,
    resolution: Tuple[int, int] = RESOLUTION,
    hfw: Dict[str, float] = HFW,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Mill the three steps and save each one's images under ``folder``:
    the step entries for the case and what each step did."""
    rng = np.random.default_rng([int(seed), 941])
    shallow = int(rng.integers(len(STEPS))) if fault(seed) == "shallow" else None
    microscope = _demo_microscope()
    try:
        microscope.move_to_milling_angle(milling_angle)
        scene = SampleScene(**scene_options(seed, overrides))
        scene.features.append(_site_cell(np.random.default_rng([int(seed), 912])))
        scene.anchor(microscope.get_stage_position())
        microscope._connection.sample_scene = scene
        folder.mkdir(parents=True, exist_ok=True)
        entries, records = [], []
        for n, step in enumerate(STEPS):
            depth = DEPTH * (rng.uniform(*SHALLOW) if n == shallow else 1.0)
            start = len(scene.milled)
            if step == "rough-2" and fault(seed) == "broken":
                # a lamella that breaks in rough milling breaks badly: start
                # the scene's damage a few passes on, so it shows at this size
                kind = str(rng.choice(["crack", "broken"]))
                scene.damage_started = {
                    "pass": len(scene.mill_passes) - BREAK,
                    "kind": kind,
                }
            for pattern in _patterns(step, depth):
                microscope.draw_pattern(pattern)
            microscope.run_milling()
            # the step's own cuts, not the damage the scene adds after them
            ours = [r for r in scene.milled[start:] if getattr(r, "face", None) is None]
            entry: Dict[str, Any] = {"name": step}
            for beam, beam_type in (("sem", BeamType.ELECTRON), ("fib", BeamType.ION)):
                image = microscope.acquire_image(
                    image_settings=ImageSettings(
                        resolution=resolution,
                        hfw=hfw[beam],
                        dwell_time=DWELL,
                        save=False,
                        beam_type=beam_type,
                    )
                )
                image.save(str(folder / f"{step}-{beam}.tif"))
                entry[beam] = f"experiment/{step}-{beam}.tif"
            entries.append(entry)
            # after the images: the sample's columns are built as it renders
            through = _through(scene, ours[:2])
            record = scene.mill_passes[-1]
            records.append(
                {
                    "step": step,
                    "depth_um": round(depth * 1e6, 2),
                    "floor_cut": round(through, 3),
                    "damage": record.get("damage")
                    or (dict(scene.damage_started) if scene.damage_started else None),
                }
            )
        return entries, records
    finally:
        microscope.disconnect()


def write_case(
    seed: int,
    out: Path,
    overrides: Optional[Dict[str, Any]] = None,
    milling_angle: float = MILLING_ANGLE,
    **render: Any,
) -> Path:
    """The rough_milling case for one seed under ``out/rough-milling/``."""
    folder = Path(out) / "rough-milling" / f"sim-{int(seed):04d}"
    entries, records = mill_steps(
        seed, folder / "experiment", overrides, milling_angle, **render
    )
    through = {r["step"]: "yes" if r["floor_cut"] >= THROUGH else "no" for r in records}
    intact = {r["step"]: "no" if r["damage"] else "yes" for r in records}
    options = scene_options(seed, overrides)
    case = {
        "id": f"sim-{int(seed):04d}-rough-milling",
        "task": "rough_milling",
        "tier": TIER,
        "playbook": PLAYBOOK,
        "format": FORMAT,
        "sample": f"simulated {options['cell_type']} cells",
        "instrument": "Demo (simulator)",
        "source": {
            "simulator": {
                "seed": int(seed),
                "scene": options,
                "milling_angle_deg": milling_angle,
                "fault": fault(seed),
            }
        },
        "brief": BRIEF.format(angle=milling_angle),
        "passes": entries,
        "key": {
            "status": "keyed",
            "keyed_by": "simulator",
            "milled_through": through,
            "intact": intact,
        },
        "traps": [
            "A trench that stops short leaves a grey floor across it in the "
            "FIB and a bridge of material under the lamella in the SEM.",
            "Curtaining stripes down a cut face are not cracks.",
        ],
    }
    reference = folder / "reference"
    reference.mkdir(parents=True, exist_ok=True)
    (reference / "steps.yaml").write_text(yaml.safe_dump(records, sort_keys=False))
    with open(folder / "case.yaml", "w") as f:
        yaml.safe_dump(case, f, sort_keys=False, width=100)
    logging.info(
        "rough milling case for seed %s: through %s, intact %s", seed, through, intact
    )
    return folder
