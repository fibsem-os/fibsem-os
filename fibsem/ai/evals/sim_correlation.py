"""A correlation eval case from the simulator, keyed from the scene.

A grid of RPE cells on the Demo Arctis, with centrin and cilia labelled in the
FM. One target cell's cilium is picked; about ten spot burns are milled round
it from the FIB at the milling angle; then the case keeps the FIB image and an
FM z-stack (reflection, green, red, blue) of the same place, as a real
correlation case does. The agent gets the target in the stack and the burns'
FIB pixels, finds the burns in the stack and places the target on the FIB
image.

The key is the scene's own: the target's midpoint (x, y and height) projected
into the FIB image and the FM stack, and each burn's centre in both.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

import fibsem.config as fconfig
from fibsem import utils
from fibsem.ai.evals.cases import FORMAT
from fibsem.ai.evals.sim_cases import TIER
from fibsem.drivers.demo import sim_relief
from fibsem.drivers.demo.sim_scene import (
    MIN_FORESHORTENING,
    SampleScene,
    surface_foreshortening,
)
from fibsem.fm.acquisition import acquire_z_stack
from fibsem.fm.structures import (
    REFLECTION,
    ChannelSettings,
    EmissionFilter,
    ZParameters,
)
from fibsem.geometry.projection import BeamStageProjection, FMStageProjection
from fibsem.structures.settings import BeamType, FibsemStagePosition, ImageSettings

PLAYBOOK = "autolamella-correlative"
MILLING_ANGLE = 15.0  # degrees
FIB_RESOLUTION = (1536, 1024)
FIB_HFW = 100e-6  # m
DWELL = 1e-6  # s
BURNS = 10
BURN_RADIUS = (8e-6, 14e-6)  # m on the surface, from the target
Z_RANGE = (-2e-6, 8e-6)  # m from focus (the film)
Z_STEP = 0.5e-6  # m
CHANNELS = [
    ChannelSettings(
        name="reflection",
        excitation_wavelength=550,
        emission_filter=REFLECTION,
        color="gray",
    ),
    ChannelSettings(
        name="green",
        excitation_wavelength=488,
        emission_filter=EmissionFilter.band("520 nm", 520),
        color="green",
    ),
    ChannelSettings(
        name="red",
        excitation_wavelength=561,
        emission_filter=EmissionFilter.band("610 nm", 610),
        color="red",
    ),
    ChannelSettings(
        name="blue",
        excitation_wavelength=365,
        emission_filter=EmissionFilter.band("450 nm", 450),
        color="blue",
    ),
]

BRIEF = (
    "The target, a primary cilium's midpoint, is at FM x {fx:.0f}, y {fy:.0f} "
    "px, z slice {fz:.1f} in the fluorescence stack (red channel: cilia; green: "
    "centrin). Correlate the stack with the FIB image and place that target on "
    "the FIB image. The spot burns were milled at these FIB px (x, y): {burns}; "
    "find each in the FM stack (reflection shows them) and use them as "
    "fiducials. Answer the target as x, y px on the FIB image, with the "
    "fiducial pairs you used and your fit error."
)


def scene_options(seed: int, overrides: Optional[Dict[str, Any]] = None) -> Dict:
    options: Dict[str, Any] = {
        "cell_type": "rpe",
        "fm_red": "cilia",
        "fm_green": "centrin",
        "fm_target_fraction": 1.0,
        "fm_cilium_fraction": 1.0,
        "fm_no_target_chance": 0.0,
        "coincidence_offset": 0.0,
        "renderer": "relief",
        "rip_fraction": 0.0,
        "rpe_island_radius": [12e-6, 35e-6],
        "rpe_island_density": 3.0,
        "rpe_occupancy": 0.6,
    }
    options.update(overrides or {})
    options["seed"] = int(seed)
    return options


def _arctis():
    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(
        manufacturer="Demo",
        config_path=os.path.join(
            fconfig.SHIPPED_CONFIG_PATH, "sim-arctis-configuration.yaml"
        ),
    )
    microscope.stage.sim_error = None
    return microscope


class _FibMap:
    """World (m) to FIB image px at one pose, for points on the film or above
    it: the inverse of the mapping the scene renders the beam with."""

    def __init__(self, microscope, scene, resolution, hfw):
        self.scene = scene
        self.beam = BeamType.ION
        self.projection = BeamStageProjection.from_microscope(
            microscope, beam_type=self.beam
        )
        shift = microscope.get_beam_shift(self.beam)
        self.shift = (float(shift.x), float(shift.y))
        try:
            self.current = float(microscope.get_beam_current(self.beam))
        except Exception:
            self.current = None
        self.position = microscope._connection.scene_stage_position()
        self.resolution = resolution
        self.pixel = hfw / resolution[0]

        def world(dx, dy):
            return scene.view_to_world(
                self.beam,
                self.position,
                self.projection,
                dx,
                dy,
                self.shift,
                self.current,
            )

        o, ex, ey = world(0, 0), world(1e-4, 0), world(0, 1e-4)
        forward = np.array(
            [
                [(ex[0] - o[0]) / 1e-4, (ey[0] - o[0]) / 1e-4],
                [(ex[1] - o[1]) / 1e-4, (ey[1] - o[1]) / 1e-4],
            ]
        )
        self.inverse = np.linalg.inv(forward)  # world offset -> view m
        self.origin = np.array(o)

    def view(self, x: float, y: float, z: float = 0.0) -> np.ndarray:
        """View offset (m from the centre, y down) of a world point z above
        the film."""
        v = self.inverse @ (np.array([x, y]) - self.origin)
        if z:
            args = (self.beam, self.position, self.projection, self.shift, self.current)
            ax0, ay0, _, _ = self.scene._view_anchor(*args)
            ax1, ay1, _, _ = self.scene._view_anchor(*args, lift=z)
            v = v + np.array([ax1 - ax0, ay1 - ay0])
        return v

    def px(self, x: float, y: float, z: float = 0.0) -> Tuple[float, float]:
        v = self.view(x, y, z) / self.pixel
        return (
            float(self.resolution[0] / 2 + v[0]),
            float(self.resolution[1] / 2 + v[1]),
        )


def _fm_px(microscope, scene, x: float, y: float) -> Tuple[float, float]:
    """World (m) to FM camera px, as render_fm draws it."""
    fm = microscope.fm
    width, height = fm.camera.resolution
    base = FMStageProjection.from_microscope(microscope)
    projection = FMStageProjection(
        geometry=base.geometry,
        pixel_size=fm.camera.pixel_size[0],
        shape=(height, width),
    )
    pose = microscope._connection.scene_stage_position_at_fm()
    reference = scene.reference_position
    if scene.tilt_axis_offset:
        reference = scene._non_eucentric_reference(pose, projection)
    reference = BeamStageProjection._compucentric_corrected(
        reference, pose, projection.geometry.rotation_centre
    )
    ax, ay = projection.to_plane(reference, pose)
    fs = max(surface_foreshortening(projection, pose), MIN_FORESHORTENING)
    turn = scene._turn(pose)
    pix = projection.pixel_size
    return (
        float(width / 2 + (x / turn + ax) / pix),
        float(height / 2 + (y * fs / turn + ay) / pix),
    )


def _surface_height(scene, x: float, y: float) -> float:
    """The sample's top (m above the film) at a world point, as the relief
    renderer builds it."""
    um = sim_relief.UM  # um per m
    step = 0.25
    offsets = np.arange(-20, 21) * step
    gx, gy = np.meshgrid(x * um + offsets, y * um + offsets)
    h = sim_relief.surface(scene, gx, gy, (step, step))[0]
    return float(h[20, 20]) / um


def _pick_target(rng, scene, nearest: int = 20):
    """One of the cilia nearest the scene's centre, at random."""
    targets = sorted(scene.fm_targets(), key=lambda t: np.hypot(*t["midpoint"][:2]))
    if not targets:
        return None
    return targets[int(rng.integers(min(nearest, len(targets))))]


def _centred(fib: "_FibMap", x: float, y: float, z: float, margin: float = 0.3) -> bool:
    w, h = fib.resolution
    px, py = fib.px(x, y, z)
    return margin * w < px < (1 - margin) * w and margin * h < py < (1 - margin) * h


def _burn_points(rng, x0: float, y0: float, fib: _FibMap) -> List[Tuple[float, float]]:
    """About ten world points round the target, spread in angle, all in the
    FIB view."""
    w, h = fib.resolution
    points = []
    angles = np.linspace(0, 2 * np.pi, BURNS, endpoint=False)
    angles = angles + rng.uniform(0, 2 * np.pi) + rng.normal(0, 0.15, BURNS)
    for a in angles:
        r = rng.uniform(*BURN_RADIUS)
        x, y = x0 + r * np.cos(a), y0 + r * np.sin(a)
        px, py = fib.px(x, y)
        if 20 < px < w - 20 and 20 < py < h - 20:
            points.append((float(x), float(y)))
    return points


def write_case(
    seed: int,
    out: Path,
    overrides: Optional[Dict[str, Any]] = None,
    milling_angle: float = MILLING_ANGLE,
    fib_resolution: Tuple[int, int] = FIB_RESOLUTION,
    z_range: Tuple[float, float] = Z_RANGE,
    channels: Optional[List[ChannelSettings]] = None,
) -> Optional[Path]:
    """The correlation case for one seed under ``out/correlation/``, or None
    when no cilium sits in the view."""
    folder = Path(out) / "correlation" / f"sim-{int(seed):04d}"
    experiment = folder / "experiment"
    rng = np.random.default_rng([int(seed), 913])
    microscope = _arctis()
    try:
        microscope.move_to_milling_angle(milling_angle)
        scene = SampleScene(**scene_options(seed, overrides))
        scene.anchor(microscope.get_stage_position())
        microscope._connection.sample_scene = scene
        target = _pick_target(rng, scene)
        if target is None:
            logging.warning("seed %s: no cilium in the scene, no case", seed)
            return None
        tx, ty, tz = target["midpoint"]
        # bring the target under the beam: the scene's y runs against the stage's
        microscope.move_stage_relative(FibsemStagePosition(x=tx, y=-ty, z=0))
        fib = _FibMap(microscope, scene, fib_resolution, FIB_HFW)
        if not _centred(fib, tx, ty, tz):
            logging.warning(
                "seed %s: target not centred at %s", seed, fib.px(tx, ty, tz)
            )
            return None
        burns = _burn_points(rng, tx, ty, fib)
        # spot burns from the FIB view: image coordinates, y up
        image_points = []
        for x, y in burns:
            vx, vy = fib.view(x, y)
            image_points.append((float(vx), float(-vy)))
        scene.burn(
            image_points,
            BeamType.ION,
            fib.position,
            fib.projection,
            fib.shift,
            fib.current,
        )
        experiment.mkdir(parents=True, exist_ok=True)
        image = microscope.acquire_image(
            image_settings=ImageSettings(
                resolution=fib_resolution,
                hfw=FIB_HFW,
                dwell_time=DWELL,
                save=False,
                beam_type=BeamType.ION,
            )
        )
        image.save(str(experiment / "spot_burns_ib.tif"))

        microscope.move_to_device("FM")
        fm = microscope.fm
        focus = fm.objective.focus_position
        fm.objective.move_absolute(focus)
        zparams = ZParameters(zmin=z_range[0], zmax=z_range[1], zstep=Z_STEP)
        z_positions = list(zparams.generate_positions(z_init=focus))
        stack = acquire_z_stack(fm, channels or CHANNELS, zparams)
        stack_name = "fm-zstack.ome.tiff"
        stack.save(str(experiment / stack_name))
        fm_pixel = float(fm.camera.pixel_size[0])

        def slice_of(z: float) -> float:
            offsets = np.array(z_positions) - focus
            return float(np.interp(z, offsets, np.arange(len(offsets))))

        target_fib = fib.px(tx, ty, tz)
        target_fm = _fm_px(microscope, scene, tx, ty)
        # a burn sits where the film-plane aim lands, drawn on the surface there
        burns_fib = [fib.px(x, y, _surface_height(scene, x, y)) for x, y in burns]
        burns_fm = [_fm_px(microscope, scene, x, y) for x, y in burns]
        film_slice = slice_of(0.0)
    finally:
        microscope.disconnect()

    shown = ", ".join(f"({x:.0f}, {y:.0f})" for x, y in burns_fib)
    case = {
        "id": f"sim-{int(seed):04d}-correlation",
        "task": "correlation",
        "tier": TIER,
        "playbook": PLAYBOOK,
        "format": FORMAT,
        "sample": "simulated RPE cells, centrin (green) and cilia (red) labelled",
        "instrument": "Demo Arctis (simulator)",
        "source": {
            "simulator": {
                "seed": int(seed),
                "scene": scene_options(seed, overrides),
                "milling_angle_deg": milling_angle,
            }
        },
        "brief": BRIEF.format(
            fx=target_fm[0], fy=target_fm[1], fz=slice_of(tz), burns=shown
        ),
        "inputs": {
            "fib_image": "experiment/spot_burns_ib.tif",
            "fib_pixel_size_um": round(FIB_HFW / fib_resolution[0] * 1e6, 4),
            "fib_size_px": list(fib_resolution),
            "fm_stack": f"experiment/{stack_name}",
            "fm_pixel_size_um": round(fm_pixel * 1e6, 4),
            "fm_z_step_um": round(Z_STEP * 1e6, 3),
        },
        "key": {
            "status": "keyed",
            "keyed_by": "simulator",
            "target_fib_px": {
                "x": round(target_fib[0], 1),
                "y": round(target_fib[1], 1),
            },
            "target_fm": {
                "x_px": round(target_fm[0], 1),
                "y_px": round(target_fm[1], 1),
                "z_slice": round(slice_of(tz), 2),
            },
            "target_height_um": round(tz * 1e6, 3),
            "fiducials_fm_px": [
                [round(x, 1), round(y, 1), round(film_slice, 2)] for x, y in burns_fm
            ],
            "fiducials_fib_px": [[round(x, 1), round(y, 1)] for x, y in burns_fib],
        },
        "scoring": {"across_tolerance_um": 0.75, "along_tolerance_um": 1.0},
        "traps": [
            "The simulator's FM has no refractive index: a stack's z is the true "
            "height, so an answer that applies an RI correction moves the target "
            "off the key."
        ],
    }
    with open(folder / "case.yaml", "w") as f:
        yaml.safe_dump(case, f, sort_keys=False, width=100)
    reference = folder / "reference"
    reference.mkdir(parents=True, exist_ok=True)
    (reference / "target.yaml").write_text(
        yaml.safe_dump(
            {
                "midpoint_m": [float(v) for v in target["midpoint"]],
                "cilium_m": [[float(v) for v in p] for p in target["cilium"]],
                "burns_world_m": [list(p) for p in burns],
            },
            sort_keys=False,
        )
    )
    logging.info("correlation case for seed %s: %s", seed, case["key"]["target_fib_px"])
    return folder
