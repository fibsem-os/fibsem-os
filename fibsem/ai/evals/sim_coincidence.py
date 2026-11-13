"""A coincidence milling eval case from the simulator: target_present.

On the Demo Arctis, a lamella is polished through an FM target in a series
of passes, with a FIB image and an FM z-stack after each, as Iterative
Polishing does when the FM and FIB look at the same point. The agent says,
pass by pass, whether the target is still in the lamella.

The target is a cilium on odd seeds and a centriole pair on even ones. On
most seeds the lamella is centred on the target, so it stays to the end. On
seeds 3, 6, 9, ... the lamella sits off it, so a pass mills it away. The key
is read from the scene: whether the target's midpoint has been milled away.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from fibsem.ai.evals.cases import FORMAT
from fibsem.ai.evals.sim_cases import TIER
from fibsem.ai.evals.sim_correlation import (
    CHANNELS,
    KINDS,
    MILLING_ANGLE,
    PLAYBOOK,
    Z_STEP,
    _arctis,
    _FibMap,
    _fm_px,
    _pick_target,
    scene_options,
    target_kind,
)
from fibsem.drivers.demo import sim_solid, sim_view
from fibsem.drivers.demo.sim_scene import SampleScene
from fibsem.fm.acquisition import acquire_z_stack
from fibsem.fm.structures import ZParameters
from fibsem.structures.settings import (
    BeamType,
    FibsemRectangleSettings,
    FibsemStagePosition,
    ImageSettings,
)

# the planned lamella after each pass (m): the base, then thinner each time
SCHEDULE = (None, 2.0e-6, 1.4e-6, 1.0e-6, 0.7e-6, 0.5e-6, 0.35e-6, 0.25e-6)
WIDTH = 8e-6  # m, the lamella's width
ROUGH_HEIGHT = 6e-6  # m, each first-pass trench
OVERLAP = 0.3e-6  # m, a pass's trench reaches this far back into the last
DEPTH = 7e-6  # m
OFF = (0.3e-6, 0.6e-6)  # m: how far a lamella sits off its target, when it does
ON = 0.06e-6  # m: the most a centred lamella is off
FIB_RESOLUTION = (1536, 1024)
FIB_HFW = 40e-6  # m
DWELL = 1e-6  # s
Z_RANGE = (-2e-6, 3e-6)  # m from focus (the ice top)
GIS = 0.5e-6  # m

BRIEF = (
    "A lamella is being polished through an FM target, {what}, on a {angle:g} "
    "degree milling angle; the FM and FIB look at the same point. Before "
    "milling (pass {base}) the target is at FM x {fx:.0f}, y {fy:.0f} px, z "
    "slice {fz:.1f} in the {channel} channel. After each pass there is a FIB "
    "image and an FM stack ({channel} and reflection). For each pass, say "
    "whether the target is still in the lamella."
)


def off_target(seed: int) -> bool:
    return int(seed) % 3 == 0


def _milled(scene, x: float, y: float, z: float) -> bool:
    """Whether the point (x, y world m, z m in the focal frame) was milled
    away, on a small patch of the sample round it."""
    um = sim_view.UM
    xs = x * um + np.arange(-2.0, 2.05, 0.05)
    ys = y * um + np.arange(-2.0, 2.05, 0.05)
    return bool(sim_solid.Plan(scene, xs, ys).gone_at(x * um, y * um, z * um)[0])


def _patterns(
    cx: float, cy: float, gap: float, previous: Optional[float]
) -> List[FibsemRectangleSettings]:
    height = ROUGH_HEIGHT if previous is None else (previous - gap) / 2 + OVERLAP
    return [
        FibsemRectangleSettings(
            width=WIDTH,
            height=height,
            depth=DEPTH,
            centre_x=cx,
            centre_y=cy + sign * (gap / 2 + height / 2),
        )
        for sign in (1, -1)
    ]


def write_case(
    seed: int,
    out: Path,
    overrides: Optional[Dict[str, Any]] = None,
    schedule: Tuple[Optional[float], ...] = SCHEDULE,
    fib_resolution: Tuple[int, int] = FIB_RESOLUTION,
    z_range: Tuple[float, float] = Z_RANGE,
) -> Optional[Path]:
    """The coincidence milling case for one seed under ``out/coincidence/``,
    or None when the scene has no target of the seed's kind."""
    import tifffile

    folder = Path(out) / "coincidence" / f"sim-{int(seed):04d}"
    experiment = folder / "experiment"
    rng = np.random.default_rng([int(seed), 977])
    kind = target_kind(seed)
    what, channel_name = KINDS[kind]
    channels = [c for c in CHANNELS if c.name in (channel_name, "reflection")]
    options = scene_options(seed, {"gis_thickness": GIS, **(overrides or {})})
    microscope = _arctis()
    try:
        microscope.move_to_milling_angle(MILLING_ANGLE)
        scene = SampleScene(**options)
        scene.anchor(microscope.get_stage_position())
        microscope._connection.sample_scene = scene
        target = _pick_target(rng, scene, kind)
        if target is None:
            logging.warning("seed %s: no %s in the scene, no case", seed, kind)
            return None
        tx, ty, above = target["midpoint"]
        tz = above + scene._fm_film()
        microscope.move_stage_relative(FibsemStagePosition(x=tx, y=-ty, z=0))
        fib = _FibMap(microscope, scene, fib_resolution, FIB_HFW)
        v = fib.view(tx, ty, tz)
        sign = 1 if rng.uniform() < 0.5 else -1
        offset = sign * (rng.uniform(*OFF) if off_target(seed) else rng.uniform(0, ON))
        # pattern coordinates run y up, the view's y down
        cx, cy = float(v[0]), float(-v[1]) + offset
        zparams = ZParameters(zmin=z_range[0], zmax=z_range[1], zstep=Z_STEP)
        experiment.mkdir(parents=True, exist_ok=True)
        passes: List[Dict[str, Any]] = []
        present: Dict[str, bool] = {}
        gaps: Dict[str, Optional[float]] = {}
        previous = None
        target_fm, focus, z_positions = None, None, None
        for n, gap in enumerate(schedule):
            name = f"P{n:02d}"
            if gap is not None:
                for pattern in _patterns(cx, cy, gap, previous):
                    microscope.draw_pattern(pattern)
                microscope.run_milling()
                previous = gap
            image = microscope.acquire_image(
                image_settings=ImageSettings(
                    resolution=fib_resolution,
                    hfw=FIB_HFW,
                    dwell_time=DWELL,
                    save=False,
                    beam_type=BeamType.ION,
                )
            )
            image.save(str(experiment / f"{name}-fib.tif"))
            microscope.move_to_device("FM")
            fm = microscope.fm
            focus = fm.objective.focus_position
            fm.objective.move_absolute(focus)
            z_positions = list(zparams.generate_positions(z_init=focus))
            stack = acquire_z_stack(fm, channels, zparams)
            stack.save(str(experiment / f"{name}-fm.ome.tiff"))
            if target_fm is None:
                target_fm = _fm_px(microscope, scene, tx, ty)
                fm_pixel = float(fm.camera.pixel_size[0])
            microscope.move_to_device("FIBSEM")
            microscope.move_to_milling_angle(MILLING_ANGLE)
            passes.append(
                {
                    "name": name,
                    "fib": f"experiment/{name}-fib.tif",
                    "fm": f"experiment/{name}-fm.ome.tiff",
                }
            )
            present[name] = not _milled(scene, tx, ty, tz)
            record = scene.mill_passes[-1] if gap is not None else {}
            gaps[name] = record.get("gap")
    finally:
        microscope.disconnect()

    offsets = np.array(z_positions) - focus
    fz = float(np.interp(tz, offsets, np.arange(len(offsets))))
    with tifffile.TiffFile(folder / passes[0]["fm"]) as tif:
        axes, shape = tif.series[0].axes, [int(n) for n in tif.series[0].shape]
    names = [p["name"] for p in passes]
    lost = next((n for n in names if not present[n]), None)
    case = {
        "id": f"sim-{int(seed):04d}-coincidence",
        "task": "target_present",
        "tier": TIER,
        "playbook": PLAYBOOK,
        "format": FORMAT,
        "sample": "simulated RPE cells, centrin (GFP) and cilia (mCherry) labelled",
        "instrument": "Demo Arctis (simulator)",
        "source": {
            "simulator": {
                "seed": int(seed),
                "scene": options,
                "milling_angle_deg": MILLING_ANGLE,
                "lamella_offset_nm": round(offset * 1e9),
                "schedule_nm": [
                    None if g is None else round(g * 1e9) for g in schedule
                ],
            }
        },
        "brief": BRIEF.format(
            what=what,
            angle=MILLING_ANGLE,
            base=names[0],
            fx=target_fm[0],
            fy=target_fm[1],
            fz=fz,
            channel=channel_name,
        ),
        "inputs": {
            "fib_pixel_size_um": round(FIB_HFW / fib_resolution[0] * 1e6, 4),
            "fm_pixel_size_um": round(fm_pixel * 1e6, 4),
            "fm_z_step_um": round(Z_STEP * 1e6, 3),
            "fm_channels": [c.name for c in channels],
            "fm_axes": axes,
            "fm_shape": shape,
        },
        "passes": passes,
        "key": present,
        "traps": [
            "A target out of focus is not gone: check the slices round its "
            "height before calling it lost.",
            "A target can be milled away from one side while the lamella is "
            "still thick.",
            "A cilium milled from one end shrinks pass by pass; it counts as "
            "gone once its midpoint is milled away.",
        ],
    }
    reference = folder / "reference"
    reference.mkdir(parents=True, exist_ok=True)
    (reference / "target.yaml").write_text(
        yaml.safe_dump(
            {
                "kind": kind,
                "midpoint_m": [float(v) for v in target["midpoint"]],
                "target_fm_px": [round(target_fm[0], 1), round(target_fm[1], 1)],
                "target_z_slice": round(fz, 2),
                "lost_at": lost,
                "gap_nm": {
                    k: None if g is None else round(g * 1e9) for k, g in gaps.items()
                },
            },
            sort_keys=False,
        )
    )
    with open(folder / "case.yaml", "w") as f:
        yaml.safe_dump(case, f, sort_keys=False, width=100)
    logging.info("coincidence case for seed %s: lost at %s", seed, lost)
    return folder
