"""A polishing_stop eval case from the simulator, keyed from the milling.

One lamella is milled on the Demo microscope, at the milling angle, in a
seeded series of passes that thin it well past the target. The sample scene
mills a little off every pass (drift, overcut, kerf), so the true gap between
the trenches differs from the planned one, and below a seeded thickness a
pass can start damage (a ragged face, a crack or a break) that grows on every
pass after. After each pass the case keeps an SEM and a FIB image and a noisy
measured thickness, as the real cases do.

The key comes from the scene's record of each pass:

- stopping is right on any undamaged pass from the first one within
  ``REACHED`` of the target until the lamella is thinner than the target by
  more than ``OVER``
- if damage starts before the target is reached, the right stop is the last
  pass before it
- stopping earlier is early; later is unsafe when damage followed, late when
  it didn't
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from fibsem.ai.evals.cases import FORMAT
from fibsem.ai.evals.sim_cases import TIER, _demo_microscope
from fibsem.drivers.demo.sim_scene import SampleScene
from fibsem.structures.settings import (
    BeamType,
    FibsemRectangleSettings,
    ImageSettings,
)

TARGET = 200e-9  # m
REACHED = 50e-9  # m: within this above the target counts as there
OVER = 60e-9  # m: thinner than the target by more than this is too far
MILLING_ANGLE = 15.0  # degrees
# The planned gap after each pass (m): one rough pass, then polishing.
SCHEDULE = (
    2.0e-6,
    1.2e-6,
    0.9e-6,
    0.7e-6,
    0.55e-6,
    0.45e-6,
    0.37e-6,
    0.30e-6,
    0.25e-6,
    0.20e-6,
    0.16e-6,
    0.12e-6,
    0.09e-6,
)
WIDTH = 12e-6  # m, the lamella's width
ROUGH_HEIGHT = 8e-6  # m, each rough trench
OVERLAP = 0.3e-6  # m, a polishing trench reaches this far back into the last
RESOLUTION = (1024, 682)
HFW = 30e-6  # m
DWELL = 1e-6  # s
MEASURE_NOISE = 25e-9  # m, sigma of the measured thickness

BRIEF = (
    "This lamella was polished in the passes listed below, on a {angle:g} "
    "degree milling angle; the target thickness is {target:g} nm. After each "
    "pass there is an SEM image, a FIB image and a measured thickness (which "
    "can be off). Say for each pass whether you would carry on or stop there: "
    "stop once the lamella is thin enough, or before it gets damaged (a ragged "
    "or eroding front face, a crack, a break). Mark only one pass stop."
)


def _patterns(gap: float, previous: Optional[float]) -> List[FibsemRectangleSettings]:
    """A pass's two trenches: rough ones the first time, then thin strips
    from the last gap's edges to the new one's."""
    if previous is None:
        height, depth = ROUGH_HEIGHT, 3e-6
    else:
        height, depth = (previous - gap) / 2 + OVERLAP, 1e-6
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


def scene_options(seed: int, overrides: Optional[Dict[str, Any]] = None) -> Dict:
    rng = np.random.default_rng([int(seed), 911])
    options: Dict[str, Any] = {
        "cell_type": str(rng.choice(["mammalian", "rpe", "neuron"])),
        "coincidence_offset": 0.0,
        "renderer": "relief",
    }
    options.update(overrides or {})
    options["seed"] = int(seed)
    return options


def mill_series(
    seed: int,
    folder: Path,
    overrides: Optional[Dict[str, Any]] = None,
    milling_angle: float = MILLING_ANGLE,
    schedule: Tuple[float, ...] = SCHEDULE,
    resolution: Tuple[int, int] = RESOLUTION,
    hfw: float = HFW,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[dict]]:
    """Mill the series and save each pass's images under ``folder``: the
    pass entries for the case, the scene's records and where damage began."""
    microscope = _demo_microscope()
    rng = np.random.default_rng([int(seed), 912])
    try:
        microscope.move_to_milling_angle(milling_angle)
        scene = SampleScene(**scene_options(seed, overrides))
        scene.anchor(microscope.get_stage_position())
        microscope._connection.sample_scene = scene
        folder.mkdir(parents=True, exist_ok=True)
        passes = []
        previous = None
        for n, gap in enumerate(schedule, start=1):
            for pattern in _patterns(gap, previous):
                microscope.draw_pattern(pattern)
            microscope.run_milling()
            previous = gap
            name = f"P{n:02d}"
            entry: Dict[str, Any] = {"name": name}
            for beam, beam_type in (("sem", BeamType.ELECTRON), ("fib", BeamType.ION)):
                image = microscope.acquire_image(
                    image_settings=ImageSettings(
                        resolution=resolution,
                        hfw=hfw,
                        dwell_time=DWELL,
                        save=False,
                        beam_type=beam_type,
                    )
                )
                image.save(str(folder / f"{name}-{beam}.tif"))
                entry[beam] = f"experiment/{name}-{beam}.tif"
            record = scene.mill_passes[-1]
            true_mean = record.get("gap_mean") or record.get("gap") or gap
            true_min = record.get("gap") or gap
            median = true_mean + rng.normal(0, MEASURE_NOISE)
            low = min(median, true_min + rng.normal(0, MEASURE_NOISE))
            entry["measured"] = {
                "median_nm": round(median * 1e9),
                "min_nm": round(low * 1e9),
                "max_nm": round((2 * median - low) * 1e9),
            }
            passes.append(entry)
        records = [
            {
                "pass": r["pass"],
                "planned_gap_nm": _nm(r.get("planned_gap")),
                "gap_nm": _nm(r.get("gap")),
                "gap_mean_nm": _nm(r.get("gap_mean")),
                "drift_nm": [round(d * 1e9, 1) for d in r.get("drift", (0, 0))],
                "damage": r.get("damage"),
            }
            for r in scene.mill_passes
        ]
        return passes, records, scene.damage_started
    finally:
        microscope.disconnect()


def _nm(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(float(value) * 1e9, 1)


def stop_span(
    records: List[Dict[str, Any]], target: float = TARGET
) -> Tuple[List[int], bool]:
    """The passes (0-based) where stopping is right, and whether damage
    follows them (carrying on is unsafe) or not (late)."""
    damaged = [i for i, r in enumerate(records) if r.get("damage")]
    first_damage = damaged[0] if damaged else len(records)
    gaps = [r["gap_nm"] for r in records]
    reach = next(
        (
            i
            for i, g in enumerate(gaps)
            if g is not None and g <= (target + REACHED) * 1e9
        ),
        None,
    )
    if reach is None or reach >= first_damage:
        span = [first_damage - 1] if first_damage >= 1 else []
    else:
        span = [
            i
            for i in range(reach, first_damage)
            if gaps[i] is not None and gaps[i] >= (target - OVER) * 1e9
        ] or [reach]
    return span, bool(damaged)


def write_case(
    seed: int,
    out: Path,
    overrides: Optional[Dict[str, Any]] = None,
    milling_angle: float = MILLING_ANGLE,
    target: float = TARGET,
    schedule: Tuple[float, ...] = SCHEDULE,
    **render: Any,
) -> Optional[Path]:
    """The polishing_stop case for one seed under ``out/polishing-stop/``,
    or None when the lamella was damaged on its first pass."""
    folder = Path(out) / "polishing-stop" / f"sim-{int(seed):04d}"
    passes, records, started = mill_series(
        seed, folder / "experiment", overrides, milling_angle, schedule, **render
    )
    span, damage_follows = stop_span(records, target)
    if not span:
        logging.warning("seed %s: damaged on the first pass, no case", seed)
        return None
    names = [p["name"] for p in passes]
    case = {
        "id": f"sim-{int(seed):04d}-polishing",
        "task": "polishing_stop",
        "tier": TIER,
        "format": FORMAT,
        "sample": f"simulated {scene_options(seed, overrides)['cell_type']} cells",
        "instrument": "Demo (simulator)",
        "source": {
            "simulator": {
                "seed": int(seed),
                "scene": scene_options(seed, overrides),
                "milling_angle_deg": milling_angle,
                "schedule_nm": [round(g * 1e9) for g in schedule],
            }
        },
        "brief": BRIEF.format(angle=milling_angle, target=round(target * 1e9)),
        "passes": passes,
        "key": {
            "status": "keyed",
            "keyed_by": "simulator",
            "stop_after": [names[i] for i in span] if len(span) > 1 else names[span[0]],
            "target_nm": round(target * 1e9),
            "damage_started": None
            if started is None
            else {"pass": names[started["pass"]], "kind": started["kind"]},
            "rules": {
                "reached_nm": round(REACHED * 1e9),
                "over_nm": round(OVER * 1e9),
            },
        },
        "scoring": {"late": "unsafe" if damage_follows else "late"},
    }
    reference = folder / "reference"
    reference.mkdir(parents=True, exist_ok=True)
    (reference / "passes.yaml").write_text(yaml.safe_dump(records, sort_keys=False))
    with open(folder / "case.yaml", "w") as f:
        yaml.safe_dump(case, f, sort_keys=False, width=100)
    logging.info(
        "polishing case for seed %s: stop after %s, damage %s",
        seed,
        case["key"]["stop_after"],
        case["key"]["damage_started"],
    )
    return folder
