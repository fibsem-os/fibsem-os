"""A synthetic sample the simulated microscope images through its projections.

The simulator's default imaging (file sequences, or the SEM/FIB text cards)
serves unrelated pictures per beam: nothing about them reflects the stage
position, so geometry between the views - coincidence above all - cannot be
measured, and the check/align control flow cannot be exercised on Demo.

This module renders both beam views as projections of ONE synthetic sample
scene:

- SEM view: the scene top-down, centred on the stage (x, y).
- FIB view: the same scene compressed along y by the geometric foreshortening
  ratio for the current milling angle, and displaced along y by
  (z_stage - z_coincident) * sin(column_tilt) - the same projection a real
  height error produces. The two views get different base contrast so they
  are not trivially identical.

`z_coincident` is captured on the first render as the stage z minus the
configured initial offset, so the simulator boots off-coincidence by that
amount and a vertical move that changes stage z visibly corrects the FIB
view, exactly as on hardware.

Opt-in via the simulator config (`sim: sample: {enabled: true, ...}`),
defaulting off - the file-sequence and text-card modes remain the default
for workflow/UI simulation. The `sample` block also carries the scene's
options (grid pitch, cell size and count, noise, the height offsets); see
SampleScene for the fields and `SampleScene.from_config`.
"""

from __future__ import annotations

import logging
import math
import zlib
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import binary_dilation as ndi_binary_dilation
from scipy.ndimage import gaussian_filter as ndi_gaussian
from scipy.ndimage import zoom as ndi_zoom

from fibsem.devices.fm import mount_transform_from_name
from fibsem.drivers.demo import sim_optics, sim_solid, sim_view
from fibsem.geometry.projection import (
    BeamStageProjection,
    FMStageProjection,
    surface_foreshortening,
)
from fibsem.structures.settings import BeamType, FibsemStagePosition

logger = logging.getLogger(__name__)

# Keep the render well-defined at any pose: below this foreshortening the
# view is a degenerate grazing projection (features smear to infinity).
MIN_FORESHORTENING = 0.035  # ~ sin(2 deg)


@dataclass(frozen=True)
class Fluorophore:
    """A dye on part of the sample: excitation and emission peaks (nm) with
    the widths a channel's excitation line and emission band are matched
    against."""

    name: str
    excitation_peak: float
    emission_peak: float
    excitation_width: float = 25.0  # nm, sigma
    emission_width: float = 35.0  # nm, sigma

    def response(self, excitation, emission) -> float:
        """How strongly this dye shows in a channel: the product of how well
        the excitation line drives it and how much of its emission the
        collected band admits. Either missing counts as fully open."""

        def gauss(value, peak, width):
            # None, or a non-numeric name ("Fluorescence": a generic
            # multi-band filter), is an open band - the other side decides
            try:
                value = float(value)
            except (TypeError, ValueError):
                return 1.0
            return float(np.exp(-0.5 * ((value - peak) / width) ** 2))

        return gauss(excitation, self.excitation_peak, self.excitation_width) * gauss(
            emission, self.emission_peak, self.emission_width
        )


# what carries which dye: the fiducial a DAPI-like blue, every cell a
# GFP-like green, a seeded subset of cells an mCherry-like red as well
# peaks sit on real dyes and on the simulated light source's lines (365,
# 450, 550, 635 nm): 365 drives the fiducial, 450 and 488 the cells, 550 and
# 561 the subset, 635 the far-red label (none unless one is set)
FIDUCIAL_DYE = Fluorophore("dapi-like", 365.0, 460.0, excitation_width=30.0)
CELL_DYE = Fluorophore("gfp-like", 470.0, 510.0, excitation_width=30.0)
SUBSET_DYE = Fluorophore("mcherry-like", 560.0, 610.0, excitation_width=30.0)
FAR_RED_DYE = Fluorophore("cy5-like", 640.0, 670.0, excitation_width=30.0)
# in reflection the bars dominate, the fiducial reflects, cells are faint
REFLECTION_WEIGHTS = (1.0, 0.25, 0.0, 0.6, 0.0)
# the FM's fluorescence intensity for a fully-dyed part, on the beam
# intensity scale the stamper uses
FM_INTENSITY = 90.0
FM_CELL_DEPTH = 9.0  # um; the cell thickness that glows at FM_INTENSITY
FM_GLOW_HAZE = 2.5e-6  # m; how far out-of-focus light from a cell spreads
FM_PSF_SIGMA = 0.13e-6  # m; an in-focus point source
CILIUM_SIGMA = 0.09e-6  # m; a ~250 nm wide cilium, under the PSF
# a source spread dimmer than this (canvas scale; the camera's noise is ~0.3)
# is not drawn: far from focus it would cover the frame for nothing
FM_FAINT = 0.05
# a faint outline of the bars leaks into every fluorescence channel
BARS_IN_FLUORESCENCE = 0.05


def fm_channel_weights(
    emission, excitation=None
) -> Tuple[float, float, float, float, float]:
    """(bars, green, red, blue, far red) intensity weights for a channel:
    the bars, then the green (cells, centrin), red (the subset, cilia), blue
    (nucleus and fiducial) and far-red (membrane, mitochondria) labels.

    Reflection is an emission of None (or named so): the excitation light
    imaged straight back, so the grid bars dominate. Otherwise each dye
    responds to the excitation line AND the collected emission band - a
    non-numeric band such as "Fluorescence" is open, and the excitation
    line alone decides. So 450 or 488 excitation shows the cells, 550/561
    the red subset, 365 the fiducial, 635 nothing; a mismatched pair (488
    excitation collected at 610, say) shows a weak bleed rather than
    nothing.
    """
    if emission is None or (
        isinstance(emission, str) and "reflect" in emission.lower()
    ):
        return REFLECTION_WEIGHTS
    return (
        BARS_IN_FLUORESCENCE,
        CELL_DYE.response(excitation, emission),
        SUBSET_DYE.response(excitation, emission),
        FIDUCIAL_DYE.response(excitation, emission),
        FAR_RED_DYE.response(excitation, emission),
    )


def _blur(image: np.ndarray, sigma_px: float) -> np.ndarray:
    """A gaussian blur that stays cheap when it is wide: past a few pixels the
    frame is block-averaged by `step`, blurred at sigma / step, and scaled
    back with a linear zoom, which is indistinguishable from the direct blur
    at that width and costs step^2 less."""
    step = int(min(8, sigma_px // 4)) if sigma_px >= 12 else 1
    if step == 1:
        return ndi_gaussian(image, sigma_px)
    h, w = image.shape
    hh, ww = (h // step) * step, (w // step) * step
    small = (
        image[:hh, :ww].reshape(hh // step, step, ww // step, step).mean(axis=(1, 3))
    )
    small = ndi_gaussian(small, sigma_px / step)
    back = ndi_zoom(small, (h / small.shape[0], w / small.shape[1]), order=1)
    return back[:h, :w]


def _lattice_draw(i: np.ndarray, j: np.ndarray, seed: int) -> np.ndarray:
    """A deterministic pseudo-random number in [0, 1) per integer lattice
    cell - a hash, so a square or a hole draws the same value from every
    view and every session with the same seed."""
    h = (i * 73856093) ^ (j * 19349663) ^ (seed * 83492791)
    return ((h * 2654435761) & 0xFFFFFFFF) / 2**32


# the orientation mark at the grid's centre: a square this many bar widths across
CENTRE_MARK_SIZE = 2.4
# coating: how fast the GIS and the sputter coater lay it down, and the
# thicknesses that take away about two-thirds of the SEM charging
GIS_RATE = 0.04e-6  # m/s with the valve open
SPUTTER_RATE = 0.5e-9  # m/s
SPUTTER_DISCHARGE = 5e-9  # m
GIS_DISCHARGE = 1e-6  # m

FIDUCIAL_ARM_LENGTH = 8e-6  # m, half-length of each fiducial cross arm
FIDUCIAL_CLEARING = 34e-6  # m, no cell body centre nearer the cross than this
FIDUCIAL_POINT_SPACING = 0.5e-6  # m
# a cell's and a nucleus's radius, in the size of the feature that draws it
RPE_RADIUS = sim_solid.RPE_RADIUS
NUCLEUS_RADIUS = sim_solid.NUCLEUS_RADIUS


@dataclass
class SceneFeature:
    """One blob on the sample surface, in world coordinates.

    `sharpness` is the supergaussian order: 1 is a plain gaussian, higher
    values give a flat top with an increasingly sharp rim (cell-like).
    """

    x: float  # m
    y: float  # m
    sigma: float  # m, along the feature's long axis
    intensity: float
    sharpness: float = 1.0
    kind: str = "cell"  # "cell" | "fiducial" | "contamination" | "ice"
    # shape: minor/major axis ratio, orientation in the world plane, and an
    # outline wobble (0 = a clean ellipse) with its own phase
    eccentricity: float = 1.0
    angle: float = 0.0  # rad
    wobble: float = 0.0
    wobble_phase: float = 0.0
    # which part of a cell this is - the FM dyes by part
    part: str = "body"  # "body" | "nucleus" | "organelle" | "bud" | "neurite"
    cell_id: int = -1
    # how much the feature hides what is under it (film, holes, bars): 0 is
    # additive (a mound on top), 1 replaces the background inside its
    # outline - a cell body or an ice plate is opaque, a nucleus sits on top
    opacity: float = 0.0
    # m; a feature with height is a mound the beams see in relief: its
    # slopes shade toward the detector and brighten with steepness
    height: float = 0.0


@dataclass
class SceneGrid:
    """One grid on the holder: where it sits in the world, how it lies, and
    the seed its content grows from. A grid's seed comes from its name, so
    the same grid shows the same cells whichever slot it is loaded in."""

    name: str
    x: float  # m, world position of the grid centre
    y: float
    rotation: float  # rad
    radius: float  # m, the usable film area
    seed: int
    extent: float  # m, content is grown over +/- extent/2 about the centre


def _lamella_gap(pass_corners, all_corners):
    """The gap (m) a pass's patterns leave about their middle, between the
    nearest cut below and above it, over the middle 80 % of the patterns'
    width: (narrowest, mean, (x0, x1, lo, hi) of the narrowest band), or
    None when the middle is cut or nothing lies either side."""
    if len(pass_corners) < 2:
        return None
    pts = np.vstack(pass_corners)
    y0 = float(np.mean([c[:, 1].mean() for c in pass_corners]))
    xa, xb = pts[:, 0].min(), pts[:, 0].max()
    x0, x1 = xa + 0.1 * (xb - xa), xb - 0.1 * (xb - xa)
    xs = np.linspace(x0, x1, 41)
    below = np.full(xs.shape, -np.inf)
    above = np.full(xs.shape, np.inf)
    for c in all_corners:
        lo, hi = sim_view.polygon_span(np.asarray(c, dtype=float), xs)
        hit = lo <= hi
        if np.any(hit & (lo <= y0) & (hi >= y0)):
            return None
        below = np.where(hit & (hi < y0), np.maximum(below, hi), below)
        above = np.where(hit & (lo > y0), np.minimum(above, lo), above)
    if not (np.all(np.isfinite(below)) and np.all(np.isfinite(above))):
        return None
    gap = above - below
    return (
        float(gap.min()),
        float(gap.mean()),
        (float(x0), float(x1), float(below.max()), float(above.min())),
    )


@dataclass
class MilledRegion:
    """A pattern that was milled: a convex polygon on the sample surface,
    in world coordinates (metres), that every later view shows as a trench.
    Kept as corners rather than a shape description because the
    view-to-world mapping at the milling pose is a foreshortening: a
    rectangle rotated in the FIB image is not a rectangle of any rotation
    on the surface."""

    points: np.ndarray  # (N, 2) world coordinates, in order round the outline
    depth: float = 1.0  # m, recorded only
    # m; width of the bright rim round the region. A trench has none; a spot
    # burn has one, the redeposition halo that makes the mark easy to find in
    # the reflection channel and the SEM.
    halo: float = 0.0
    # how the milling beam saw it, for the cut along the beam (a
    # sim_view.BeamCut); None cuts straight down
    beam_cut: Optional[object] = None
    # damage: convex pieces on the sample plane (um);
    # when given, only what lies under one of them is removed
    face: Optional[list] = None


MILL_DEPTH = 90.0  # how dark a trench reads in the FM reflection (canvas scale)
# A spot burn: what a 60 pA, 10 s exposure leaves on a real grid. The mark is
# a dark disc like any milled region, with a bright rim round it.
SPOT_BURN_DIAMETER = 0.5e-6  # m
SPOT_BURN_HALO = 0.2e-6  # m
FM_HALO = 70.0  # and in the FM reflection (canvas scale)


class FeatureIndex:
    """Where a scene's features reach, to find the ones a view touches."""

    def __init__(self, scene: "SampleScene") -> None:
        self.features = scene.features
        n = len(self.features)
        self._x = np.array([f.x for f in self.features], dtype=np.float64)
        self._y = np.array([f.y for f in self.features], dtype=np.float64)
        self._reach = np.array(
            [
                (4.0 if f.sharpness < 2 else 2.2) * f.sigma * (1 + f.wobble)
                for f in self.features
            ],
            dtype=np.float64,
        ).reshape(n)

    def within(self, x_lo: float, x_hi: float, y_lo: float, y_hi: float) -> np.ndarray:
        """Indices of the features whose reach touches a world box."""
        return np.flatnonzero(
            (self._x + self._reach >= x_lo)
            & (self._x - self._reach <= x_hi)
            & (self._y + self._reach >= y_lo)
            & (self._y - self._reach <= y_hi)
        )

    def is_for(self, scene: "SampleScene") -> bool:
        return self.features is scene.features and len(self.features) == len(self._x)


@dataclass
class SampleScene:
    """A synthetic cryo-grid scene rendered per-beam by projection.

    The sample is a regular grid mesh (bars at a known pitch, a hole centred
    on the world origin) with randomly clustered cell-like blobs scattered
    over the film, plus a fiducial-like cross at the origin. The mesh is
    deliberately periodic: it reproduces the rival-correlation-peak trap that
    real grid bars create, so window constraints are testable.
    """

    coincidence_offset: float = 10e-6  # m, initial height error at boot
    seed: int = 24
    # what grows on the film: "mammalian" (adherent, fried-egg: a wide flat
    # body with an off-centre nuclear mound and organelle speckle, sparse),
    # "yeast" (compact bright ovoids in clusters, some budding), "bacteria"
    # (dense small rods), "rpe" (big smooth domes standing apart, each with a nucleus),
    # "neuron" (a round soma with long branching processes), "mixed", or
    # "none" for bare film
    cell_type: str = "rpe"
    # yeast: clusters over the extent
    n_clusters: int = 120  # yeast clumps per 800 x 800 um
    cells_per_cluster: Tuple[int, int] = (8, 40)  # inclusive range
    cell_size: Tuple[float, float] = (1.8e-6, 2.8e-6)  # m, radius range per cell
    cluster_spread: float = 20e-6  # m, how far a clump reaches from its centre
    # mammalian: count per 150 x 150 um, body and nucleus radii
    mammalian_density: float = 7.0
    mammalian_radius: Tuple[float, float] = (18e-6, 30e-6)  # m
    nucleus_radius: Tuple[float, float] = (5e-6, 7e-6)  # m
    # bacteria: count per 150 x 150 um, rod length
    bacteria_density: float = 160.0
    bacteria_length: Tuple[float, float] = (2.0e-6, 3.5e-6)  # m
    # rpe: count per 150 x 150 um, and the cell radius
    rpe_density: float = 3.0
    rpe_radius: Tuple[float, float] = (11e-6, 21e-6)  # m
    # neuron: somas per 150 x 150 um, soma radius, process length
    neuron_density: float = 1.2
    soma_radius: Tuple[float, float] = (5e-6, 9e-6)  # m
    neurite_length: Tuple[float, float] = (60e-6, 220e-6)  # m
    # m, features are scattered over +/- extent/2 (on the grid's disc); None
    # fills the whole grid
    extent: Optional[float] = None
    grid_pitch: float = 125e-6  # m, mesh pitch (~200 mesh)
    grid_bar_width: float = 35e-6  # m
    grid_intensity: float = 90.0  # scales the bars' reflection leak into the FM
    # the support film in each square: "continuous", or "holey" - a
    # Quantifoil-style lattice of round holes (diameter, centre-to-centre
    # pitch), aligned with the grid, a seeded few of them broken/merged.
    # Continuous by default: the lattice is a 4 um periodic pattern the
    # fine-pass correlator aliases on (false refusals at the fiducial, and a
    # false convergence at 80 um on the simulator), which the measurement
    # does not handle yet - see the xfail in test_sim_scene_realism.py
    film: str = "holey"
    hole_diameter: float = 2.0e-6  # m  (R2/2: 2 um holes, 2 um apart)
    hole_pitch: float = 4.0e-6  # m
    broken_hole_fraction: float = 0.006
    # a seeded fraction of squares with the film torn away (a rip)
    rip_fraction: float = 0.03
    # ice crystals: flat bright plates with a bright rim, per 100 x 100 um
    ice_density: float = 0.4
    ice_size: Tuple[float, float] = (6e-6, 16e-6)  # m, plate radius
    # ice contamination: patches of frost (small bright crystal grains), per
    # 100 x 100 um
    frost_density: float = 0.15
    # dark spheres (round beads standing on the film), per 100 x 100 um
    sphere_density: float = 0.4
    sphere_size: Tuple[float, float] = (0.4e-6, 2.0e-6)  # m, radius
    # hairline cracks in the film (a dark line, not torn open), per 100 x 100 um
    crack_density: float = 0.3
    # a fiducial-like cross at each grid centre: a landmark for eyeballing
    # navigation and for tests, absent on a real grid - off by default
    fiducial: bool = False
    # the bigger square with one rounded corner at the bar crossing by the
    # grid's centre, which tells the grid's orientation
    centre_mark: bool = True
    # how strongly the SEM charges an uncoated grid (cells, ice): bright
    # patches and streaks along the scan, worse with a longer dwell; 0 = never
    charging: float = 1.0
    # milling is not perfect: each pass drifts by a seeded offset (m, sigma)
    # and a pattern cuts a little past its edges (m, up to)
    mill_drift: float = 20e-9
    mill_overcut: float = 30e-9
    # the beam's tail: a pattern cuts wider at higher current (half the spot
    # size) and its walls lean in going down (degrees), so a lamella is
    # thicker at its bottom; a cut face carries vertical curtaining stripes
    mill_kerf: bool = True
    mill_taper: float = 1.5
    curtaining: float = 1.0
    # a thin lamella can be damaged: below a seeded thickness (m, range) each
    # polishing pass has this chance of starting damage (a ragged front face,
    # a crack across, a piece broken out), which grows with each pass after
    damage_below: Tuple[float, float] = (150e-9, 300e-9)
    damage_chance: float = 0.4
    # the coat on the grid, laid down by the GIS and the sputter coater (or
    # set for a grid that comes in coated); m
    gis_thickness: float = 0.0
    sputter_thickness: float = 0.0
    # the grid's usable radius; beyond it the metal rim, then the holder
    grid_radius: float = 1.4e-3  # m
    grid_rim_width: float = 150e-6  # m
    noise_sigma: float = 12.0  # gaussian noise layer over the final image
    # blend of full-range uniform noise, like the simulator's default
    # random-noise images (0 = none, 1 = pure noise)
    noise_fraction: float = 0.15
    # grids never load perfectly straight; drawn from +/- this range per seed
    grid_rotation: Optional[float] = None  # rad; random when None
    grid_rotation_range: float = np.deg2rad(45.0)
    # contamination: small specks over film and bars - the aperiodic content
    # a real grid carries, which is what breaks a mesh-pitch alias
    contamination_density: float = 15.0  # specks per 100 x 100 um
    contamination_size: Tuple[float, float] = (0.8e-6, 3.0e-6)  # m, sigma range
    # a fixed per-beam misalignment ("electron"/"ion" -> (dx, dy) m): the
    # view shifted as by a beam shift the microscope does not report - the
    # persistent lateral offset the coincidence measurement calls dx
    beam_offset: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    # changing beam current moves the beam a little (aperture/lens
    # alignment): a seeded per-(beam, current) offset drawn with this sigma
    # (m), so the milling-current alignment has something real to undo.
    # Opt-in (0 = off): on, every acquisition carries a current-dependent
    # lateral offset, which tests with exact expectations must account for
    current_offset_scale: float = 0.0
    # the beams' settings show in the image: a working distance away from
    # focus and stigmation off zero blur it, and the current sets the probe
    # size and the noise (see sim_optics). False draws every image sharp.
    # On for a scene from the configuration; a scene built in code asks
    beam_optics: bool = False
    # focus is at the beam's eucentric height, give or take this (m): each
    # beam's own seeded error, which an autofocus takes out
    focus_fuzz: float = 15e-6
    # a session that starts wrong, for something to notice and put right:
    # the scan rotation each beam boots at (degrees, {electron: 180}), and
    # how the FM camera is really mounted (a mount transform's name; None is
    # as the configuration says). Mounted another way, the FM image is
    # flipped until the camera's mount transform is set to match
    scan_rotation_at_start: Dict[str, float] = field(default_factory=dict)
    fm_camera_mount: Optional[str] = None
    _current_offsets: Dict[Tuple[str, float], Tuple[float, float]] = field(
        default_factory=dict, repr=False
    )
    # how far above the stage tilt axis the sample surface sits (m, along the
    # stage z axis). 0 is a eucentric stage; a real stage is not, and a tilt
    # change then swings the surface about the axis, costing coincidence
    tilt_axis_offset: float = 0.0
    # the coincident stage position, captured on first render (current
    # position with z offset by coincidence_offset); every view is rendered
    # as the beam projection of the scene relative to this reference
    reference_position: Optional[FibsemStagePosition] = None
    features: List[SceneFeature] = field(default_factory=list)
    milled: List[MilledRegion] = field(default_factory=list)
    # one record per milling pass: the drift, the planned and true gap
    # between its trenches, and any damage (see `mill`)
    mill_passes: List[dict] = field(default_factory=list)
    # where a thin lamella's damage started: {"pass", "kind"}, or None
    damage_started: Optional[dict] = None
    _cuts: List[np.ndarray] = field(default_factory=list, repr=False)
    # the grids on the holder. With no calibrated, occupied slot handed in
    # (`holder_slots`, set by the simulator from its sample holder before
    # each acquisition) there is one grid, centred on the world anchor -
    # the scene as it always was. With slots, one grid per occupied slot,
    # centred where that slot's stage position falls in the world; an empty
    # or uncalibrated slot shows the holder
    grids: List[SceneGrid] = field(default_factory=list)
    # opt-in: the sample holder's slot positions are an operator file shared
    # by every configuration in the directory, so a test-drive holder must
    # not move the grids under every test that runs the Demo
    grids_from_holder: bool = False
    holder_slots: List[Tuple[str, float, FibsemStagePosition]] = field(
        default_factory=list, repr=False, compare=False
    )
    _holder_signature: Optional[tuple] = field(default=None, repr=False, compare=False)
    # fraction of cells that also carry the red fluorophore (seeded subset)
    red_fraction: float = 0.4
    # defocus model for the FM: blur sigma grows by this many pixels per
    # micron the objective sits away from its focus position
    fm_blur_px_per_um: float = 0.6
    # FM targets in the cytoplasmic (GFP) channel: a seeded fraction of cells
    # carry a diffuse centrin pool with a centriole pair at its edge (two
    # diffraction-limited spots ~1 um apart, at a height in the cell), and a
    # seeded chance that a scene carries none; a spot is sharp within about
    # `fm_axial_range` (m) of its height and fades either side
    fm_target_fraction: float = 0.5
    fm_no_target_chance: float = 0.2
    fm_axial_range: float = 0.5e-6
    # what each FM channel labels (blue is always the nucleus and fiducial):
    # green "cytoplasm" (the whole cell, with the centrin pairs), "centrin"
    # (centrin-GFP: the pairs and pools, the cell only faint) or
    # "lipid_droplets"; red "subset" (a seeded subset of whole cells) or
    # "cilia" (a membrane label on the primary cilium, grown from the mother
    # centriole in a seeded fraction of the target cells); far red "none",
    # "membrane" (the cell outlines) or "mitochondria"
    fm_green: str = "cytoplasm"
    fm_red: str = "subset"
    fm_far_red: str = "none"
    fm_cilium_fraction: float = 0.7
    # the camera's hot pixels: the same pixels, as bright, in every frame
    fm_hot_pixels: float = 2e-4
    _index: Optional[FeatureIndex] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.features:
            return
        rng = np.random.default_rng(self.seed)
        if self.grid_rotation is None:
            self.grid_rotation = float(
                rng.uniform(-self.grid_rotation_range, self.grid_rotation_range)
            )
        if not self.grids:
            self.grids = [self._default_grid()]
        # the default grid grows from the generator that drew its rotation,
        # as it always did, so a seed still means the same scene
        self._populate(rng)

    def _default_grid(self) -> SceneGrid:
        """The one grid at the world anchor: the scene as it always was."""
        return SceneGrid(
            name="default",
            x=0.0,
            y=0.0,
            rotation=float(self.grid_rotation or 0.0),
            radius=self.grid_radius,
            seed=self.seed,
            extent=self.extent or 2 * self.grid_radius,
        )

    def _populate(self, rng: Optional[np.random.Generator] = None) -> None:
        """Grow every grid's content from its own seed (`rng` for the first
        grid, when given)."""
        self.features = []
        self.cracks: List[np.ndarray] = []
        self.fm_spots: List[dict] = []
        for grid in self.grids:
            self._generate_grid(grid, rng)
            rng = None

    def _generate_grid(
        self, grid: SceneGrid, rng: Optional[np.random.Generator] = None
    ) -> None:
        if rng is None:
            rng = np.random.default_rng(grid.seed)
        half = grid.extent / 2
        start = len(self.features)
        self._generate_cells(rng, half)
        # fiducial-like cross at the grid centre: a dense line of small
        # blobs along each arm, so it goes through the same projection as
        # every other feature
        n_arm = int(2 * FIDUCIAL_ARM_LENGTH / FIDUCIAL_POINT_SPACING) + 1
        arm = np.linspace(-FIDUCIAL_ARM_LENGTH, FIDUCIAL_ARM_LENGTH, n_arm)
        for offset in arm if self.fiducial else []:
            for x, y in (
                (float(offset), float(offset)),
                (float(offset), -float(offset)),
            ):
                self.features.append(
                    SceneFeature(
                        x=x, y=y, sigma=0.4e-6, intensity=220.0, kind="fiducial"
                    )
                )
        # contamination: everywhere, film and bars alike
        n_specks = int(self.contamination_density * (grid.extent / 100e-6) ** 2)
        for _ in range(n_specks):
            x, y = rng.uniform(-half, half, size=2)
            self.features.append(
                SceneFeature(
                    x=float(x),
                    y=float(y),
                    sigma=float(rng.uniform(*self.contamination_size)),
                    intensity=float(rng.uniform(40, 120)),
                    sharpness=2.0,
                    kind="contamination",
                )
            )
        # ice crystals: sparse flat plates, irregular outline, bright rim
        n_ice = int(self.ice_density * (grid.extent / 100e-6) ** 2)
        for _ in range(n_ice):
            x, y = rng.uniform(-half, half, size=2)
            self.features.append(
                SceneFeature(
                    x=float(x),
                    y=float(y),
                    sigma=float(rng.uniform(*self.ice_size)),
                    intensity=70.0,
                    sharpness=6.0,
                    eccentricity=float(rng.uniform(0.7, 1.0)),
                    angle=float(rng.uniform(0, np.pi)),
                    wobble=0.18,
                    wobble_phase=float(rng.uniform(0, 2 * np.pi)),
                    kind="ice",
                    opacity=0.9,
                )
            )
        # frost and dark spheres from a generator of their own, so the rest
        # of a seed's scene is laid out as it always was
        extra = np.random.default_rng([int(grid.seed) & 0x7FFFFFFF, 31])
        self._generate_frost(extra, grid, half)
        self._generate_spheres(extra, grid, half)
        self._generate_cracks(extra, grid, half)
        # grown over a square; only what lies on the grid's disc is kept, a
        # cell whole or not at all (by where its body is)
        mine = self.features[start:]
        off = {
            f.cell_id
            for f in mine
            if f.kind == "cell"
            and f.part == "body"
            and (
                np.hypot(f.x, f.y) > grid.radius
                # the fiducial is cut into clear ice, not into a cell
                or (self.fiducial and np.hypot(f.x, f.y) < FIDUCIAL_CLEARING)
            )
        }
        self.features[start:] = [
            f
            for f in mine
            if (f.kind == "cell" and f.cell_id >= 0 and f.cell_id not in off)
            or (
                (f.kind != "cell" or f.cell_id < 0)
                and np.hypot(f.x, f.y) <= grid.radius
            )
        ]
        # grown about the origin; carried to where the grid sits
        for f in self.features[start:]:
            f.x += grid.x
            f.y += grid.y
        self._generate_centrioles(grid, start)

    def _generate_frost(self, rng, grid: SceneGrid, half: float) -> None:
        """Ice contamination: patches of small bright crystal grains, dense
        in the middle and thinning out."""
        for _ in range(int(self.frost_density * (grid.extent / 100e-6) ** 2)):
            cx, cy = rng.uniform(-half, half, size=2)
            spread = float(rng.uniform(2e-6, 8e-6))
            for _ in range(int(rng.integers(20, 120))):
                x, y = rng.normal(0, spread, size=2)
                self.features.append(
                    SceneFeature(
                        x=float(cx + x),
                        y=float(cy + y),
                        sigma=float(rng.uniform(0.15e-6, 0.6e-6)),
                        intensity=float(rng.uniform(80, 140)),
                        sharpness=3.0,
                        eccentricity=float(rng.uniform(0.5, 1.0)),
                        angle=float(rng.uniform(0, np.pi)),
                        kind="contamination",
                        part="frost",
                    )
                )

    def _generate_spheres(self, rng, grid: SceneGrid, half: float) -> None:
        """Dark spheres: round beads standing on the film, darker than it in
        both beams."""
        for _ in range(int(self.sphere_density * (grid.extent / 100e-6) ** 2)):
            x, y = rng.uniform(-half, half, size=2)
            r = float(rng.uniform(*self.sphere_size))
            self.features.append(
                SceneFeature(
                    x=float(x),
                    y=float(y),
                    sigma=r,
                    intensity=-60.0,
                    sharpness=3.0,
                    kind="contamination",
                    part="sphere",
                    height=1.1 * r,
                )
            )

    def _generate_cracks(self, rng, grid: SceneGrid, half: float) -> None:
        """Hairline cracks: wandering lines across the film, some with a
        branch, kept as (N, 3) world points (x, y, half-width)."""
        step = 0.5e-6
        n = int(self.crack_density * (grid.extent / 100e-6) ** 2)
        for _ in range(n):
            start = rng.uniform(-half, half, size=2) + (grid.x, grid.y)
            heading = float(rng.uniform(0, 2 * np.pi))
            length = float(rng.uniform(8e-6, 60e-6))
            width = float(rng.uniform(0.25e-6, 0.5e-6))
            arms = [(start, heading, length)]
            if rng.uniform() < 0.4:
                arms.append(
                    (
                        None,
                        float(rng.choice([-1, 1]) * rng.uniform(0.4, 1.1)),
                        0.5 * length,
                    )
                )
            trunk = None
            for origin, turn, arm_length in arms:
                if origin is None:
                    k = int(rng.integers(len(trunk) // 4, 3 * len(trunk) // 4 + 1))
                    origin, angle = trunk[k, :2], heading + turn
                else:
                    angle = turn
                m = max(2, int(arm_length / step))
                angles = angle + np.cumsum(rng.normal(0, 0.18, size=m))
                pts = origin + np.cumsum(
                    step * np.stack([np.cos(angles), np.sin(angles)], axis=1), axis=0
                )
                pts = np.vstack([origin, pts])
                # it narrows to nothing at its tip
                w = width * np.linspace(1.0, 0.3, len(pts))
                arm = np.column_stack([pts, w])
                trunk = arm if trunk is None else trunk
                self.cracks.append(arm)

    def _generate_centrioles(self, grid: SceneGrid, start: int) -> None:
        """Centrin in a seeded fraction of this grid's cells: a dim pool and,
        at its edge, a centriole pair at a height in the cell (kept in
        `fm_spots`, seen only by the FM). Some scenes carry none."""
        rng = np.random.default_rng([int(grid.seed) & 0x7FFFFFFF, 59])
        if rng.uniform() < self.fm_no_target_chance:
            return
        for f in self.features[start:]:
            if f.kind != "cell" or f.part != "body":
                continue
            if rng.uniform() >= self.fm_target_fraction:
                continue
            tall = max(float(f.height or 0.0), 1.5e-6)
            pool_sigma = float(rng.uniform(1.0e-6, 1.5e-6))
            a = rng.uniform(0, 2 * np.pi)
            reach = float(rng.uniform(0.2, 0.5)) * f.sigma
            px, py = f.x + reach * np.cos(a), f.y + reach * np.sin(a)
            z = float(rng.uniform(0.3e-6, 0.8 * tall))
            b = rng.uniform(0, 2 * np.pi)
            edge = 0.8 * pool_sigma
            mx, my = px + edge * np.cos(b), py + edge * np.sin(b)
            sep = float(rng.uniform(0.8e-6, 1.2e-6))
            c = rng.uniform(0, 2 * np.pi)
            dz = float(rng.normal(0, 0.15e-6))
            pair = [
                (
                    mx + s_ * sep / 2 * np.cos(c),
                    my + s_ * sep / 2 * np.sin(c),
                    z + s_ * dz / 2,
                )
                for s_ in (-1, 1)
            ]
            # the primary cilium: a rod from the mother centriole, leaning
            # up and out of the cell (drawn for every target so the layout
            # is the same whichever label is shown)
            grows = rng.uniform() < self.fm_cilium_fraction
            length = float(rng.uniform(2e-6, 5e-6))
            azimuth = rng.uniform(0, 2 * np.pi)
            rise = rng.uniform(0.0, np.deg2rad(80))  # some lie almost flat
            # it bends as it goes: a smoothly varying curvature, so it
            # sweeps round over a micron or two rather than jittering
            steps = max(int(length / 0.1e-6), 2)
            bend = ndi_gaussian(rng.normal(0, 1, steps + 16), 6)[8:-8]
            bend = 0.08 * bend / (bend.std() + 1e-9) + rng.normal(0, 0.03)
            turn_ = azimuth + np.cumsum(bend)
            lift = np.clip(rise + np.cumsum(rng.normal(0, 0.03, steps)), -0.2, 1.4)
            step = 0.1e-6 * np.column_stack(
                [
                    np.cos(lift) * np.cos(turn_),
                    np.cos(lift) * np.sin(turn_),
                    np.sin(lift),
                ]
            )
            path = np.vstack([pair[0], np.array(pair[0]) + np.cumsum(step, axis=0)])
            cilium = (
                [tuple(float(v) for v in path[0]), tuple(float(v) for v in path[-1])]
                if grows
                else None
            )
            self.fm_spots.append(
                {
                    "cell_id": int(f.cell_id),
                    "cilium": cilium,
                    "cilium_path": path.tolist() if grows else None,
                    "pool": (float(px), float(py), z, pool_sigma),
                    "pair": [tuple(float(v) for v in p) for p in pair],
                    "midpoint": (float(mx), float(my), z),
                }
            )

    def fm_targets(self) -> List[dict]:
        """The scene's target cells: per cell, the centriole pair's midpoint
        (x, y world m; z m above the film, the objective's distance from
        focus at which the pair is sharpest), the two spots, the centrin
        pool, and the primary cilium's base and tip (None if it has none).
        With red labelling cilia, only the cells with a cilium are listed,
        and the midpoint is the cilium's."""
        spots = getattr(self, "fm_spots", [])
        if self.fm_red != "cilia":
            return [dict(t) for t in spots]
        out = []
        for t in spots:
            if t.get("cilium") is None:
                continue
            (x0, y0, z0), (x1, y1, z1) = t["cilium"]
            mid = ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
            out.append({**t, "midpoint": mid})
        return out

    def crack_depth(self, xs_world: np.ndarray, ys_world: np.ndarray) -> np.ndarray:
        """How far (0..1) each world point sits inside a hairline crack: 1 on
        its line, 0 past its half-width. Film only is the caller's to mask."""
        xs_world, ys_world = np.broadcast_arrays(xs_world, ys_world)
        depth = np.zeros(xs_world.shape, dtype=np.float32)
        cracks = getattr(self, "cracks", None)
        if not cracks or self.crack_density <= 0:
            return depth
        x_lo, x_hi = float(xs_world.min()), float(xs_world.max())
        y_lo, y_hi = float(ys_world.min()), float(ys_world.max())
        for arm in cracks:
            pad = arm[:, 2].max()
            if (
                arm[:, 0].max() + pad < x_lo
                or arm[:, 0].min() - pad > x_hi
                or arm[:, 1].max() + pad < y_lo
                or arm[:, 1].min() - pad > y_hi
            ):
                continue
            near = (
                (xs_world >= arm[:, 0].min() - pad)
                & (xs_world <= arm[:, 0].max() + pad)
                & (ys_world >= arm[:, 1].min() - pad)
                & (ys_world <= arm[:, 1].max() + pad)
            )
            idx = np.nonzero(near)
            if not idx[0].size:
                continue
            px, py = xs_world[idx][:, None], ys_world[idx][:, None]
            ax, ay, aw = arm[:-1, 0], arm[:-1, 1], arm[:-1, 2]
            dx, dy = arm[1:, 0] - ax, arm[1:, 1] - ay
            t = np.clip(
                ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy + 1e-30), 0, 1
            )
            dist = np.hypot(px - ax - t * dx, py - ay - t * dy) / aw
            inside = np.clip(1 - dist.min(axis=1), 0, None)
            depth[idx] = np.maximum(depth[idx], inside.astype(np.float32))
        return depth

    def deposit(self, seconds: float) -> None:
        """A GIS deposition with the valve open *seconds*: a thicker coat."""
        self.gis_thickness += GIS_RATE * max(float(seconds), 0.0)
        logger.debug(
            {"msg": "sample_scene_deposited", "gis_thickness": self.gis_thickness}
        )

    def sputter(self, seconds: float) -> None:
        """A sputter coat of *seconds*: a thin metal layer."""
        self.sputter_thickness += SPUTTER_RATE * max(float(seconds), 0.0)
        logger.debug(
            {
                "msg": "sample_scene_sputtered",
                "sputter_thickness": self.sputter_thickness,
            }
        )

    def charge_strength(self, dwell_time: Optional[float]) -> float:
        """How much the SEM charges the sample now: the scene's `charging`,
        less what the coat carries away, more the longer each pixel dwells.
        0 with no dwell given."""
        if not dwell_time or self.charging <= 0:
            return 0.0
        coat = np.exp(-self.sputter_thickness / SPUTTER_DISCHARGE) / (
            1.0 + self.gis_thickness / GIS_DISCHARGE
        )
        dwell = float(np.clip(np.sqrt(dwell_time / 1e-6), 0.0, 2.5))
        return float(self.charging * coat * dwell)

    @staticmethod
    def charge_up(data: np.ndarray, charge: np.ndarray, strength: float) -> np.ndarray:
        """SEM charging on a grey-level image: the insulating parts (`charge`,
        0..1 per pixel) glow and lose their detail, and the film round them
        goes dark."""
        if strength <= 0 or not charge.any():
            return data
        size = max(data.shape) / 1024.0
        reach = ndi_gaussian(charge, 25.0 * size)
        halo = np.clip(reach - charge, 0, None)
        k = np.clip(strength * charge, 0, 1)
        blurred = ndi_gaussian(data, 2.0 * size)
        out = data + 0.6 * k * (blurred - data)
        out = out + 40.0 * strength * np.tanh(1.5 * charge) - 60.0 * strength * halo
        return out

    def place_grids(self, projection: BeamStageProjection) -> None:
        """Lay the grids out from the holder's slots, when they changed.

        Each occupied slot with a position puts a grid where that stage
        position falls in the world (the plane offset from the anchor, at
        the anchor's pose, un-foreshortened), turned by its own seeded
        rotation. Content is seeded from the grid's name, so exchanging
        grids between slots moves their cells with them. With no such slot
        the single default grid at the anchor stays.
        """
        signature = tuple(
            (name, float(radius), float(pos.x or 0.0), float(pos.y or 0.0))
            for name, radius, pos in self.holder_slots
            if pos is not None
        )
        if signature == self._holder_signature:
            return
        had_holder_grids = bool(self._holder_signature)
        self._holder_signature = signature
        if not signature:
            # no slot to place: the default grid from construction stands,
            # with whatever features were set on it. Regrow only when
            # holder grids are being taken away
            if had_holder_grids:
                self.grids = [self._default_grid()]
                self._populate()
            return
        else:
            reference = self.reference_position
            fs = max(surface_foreshortening(projection, reference), MIN_FORESHORTENING)
            grids = []
            for name, radius, pos in self.holder_slots:
                if pos is None:
                    continue
                at = deepcopy(reference)
                at.x, at.y = float(pos.x or 0.0), float(pos.y or 0.0)
                px, py = projection.to_plane(at, reference)
                seed = zlib.crc32(f"{self.seed}:{name}".encode()) & 0x7FFFFFFF
                rotation = float(
                    np.random.default_rng(seed).uniform(
                        -self.grid_rotation_range, self.grid_rotation_range
                    )
                )
                grids.append(
                    SceneGrid(
                        name=name,
                        x=float(px),
                        y=float(py) / fs,
                        rotation=rotation,
                        radius=float(radius),
                        seed=seed,
                        # a holder grid is populated edge to edge
                        extent=2 * float(radius),
                    )
                )
        self.grids = grids
        self._populate()
        # trenches were cut in grids that are no longer where they were
        self.milled = []
        self.mill_passes, self._cuts, self.damage_started = [], [], None
        logger.debug(
            {
                "msg": "sample_scene_grids_placed",
                "grids": [
                    {"name": g.name, "x": g.x, "y": g.y, "radius": g.radius}
                    for g in grids
                ],
                "features": len(self.features),
            }
        )

    def _view_anchor(
        self,
        beam_type: BeamType,
        stage_position: FibsemStagePosition,
        projection: BeamStageProjection,
        beam_shift: Tuple[float, float] = (0.0, 0.0),
        beam_current: Optional[float] = None,
        lift: float = 0.0,
    ) -> Tuple[float, float, float, float]:
        """Where the world anchor appears in a view (m, scan frame, y down),
        the foreshortening and the scan rotation. `lift` raises the anchor
        (m, stage z): where a point that high appears."""
        reference = self.reference_position
        if self.tilt_axis_offset:
            reference = self._non_eucentric_reference(stage_position, projection)
        if lift:
            reference = deepcopy(reference)
            reference.z = (reference.z or 0.0) + lift
        ax, ay = projection.to_plane(reference, stage_position)
        fs = max(surface_foreshortening(projection, stage_position), MIN_FORESHORTENING)
        theta = float(projection.scan_rotation or 0.0)
        if np.isclose(theta, np.pi):
            ax, ay = -ax, -ay
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        ax, ay = ax * cos_t - ay * sin_t, ax * sin_t + ay * cos_t
        offset = self.beam_offset.get(beam_type.name.lower(), (0.0, 0.0))
        current_offset = self.current_offset(beam_type, beam_current)
        ax = ax + beam_shift[0] + offset[0] + current_offset[0]
        ay = ay - (beam_shift[1] + offset[1] + current_offset[1])
        return ax, ay, fs, theta

    def view_frame(
        self,
        beam_type: BeamType,
        stage_position: FibsemStagePosition,
        projection: BeamStageProjection,
        beam_shift: Tuple[float, float] = (0.0, 0.0),
        beam_current: Optional[float] = None,
    ):
        """How a view sees the world: the anchor in
        the unrotated frame and how far a point moves per unit height."""
        args = (beam_type, stage_position, projection, beam_shift, beam_current)
        ax, ay, fs, theta = self._view_anchor(*args)
        _, ay_up, _, _ = self._view_anchor(*args, lift=1e-6)
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        def unrotate(x, y):
            return x * cos_t + y * sin_t, -x * sin_t + y * cos_t

        a = unrotate(ax, ay)
        a_up = unrotate(ax, ay_up)
        return sim_view.ViewFrame(
            anchor=(a[0] * 1e6, a[1] * 1e6),
            theta=theta,
            turn=self._turn(stage_position),
            fs=float(fs),
            # a height moves a point across the view by the cosine of the
            # glancing angle whose sine is the foreshortening, the way a
            # stage raise moves it
            k=float(np.copysign(np.sqrt(max(0.0, 1 - fs * fs)), a_up[1] - a[1])),
        )

    def view_to_world(
        self,
        beam_type: BeamType,
        stage_position: FibsemStagePosition,
        projection: BeamStageProjection,
        dx: float,
        dy: float,
        beam_shift: Tuple[float, float] = (0.0, 0.0),
        beam_current: Optional[float] = None,
    ) -> Tuple[float, float]:
        """World coordinates of a point seen at (dx, dy) metres from the
        centre of a beam's view (plane convention: y down) - the inverse of
        the mapping render() draws with: un-shift, un-rotate the scan,
        un-foreshorten."""
        ax, ay, fs, theta = self._view_anchor(
            beam_type, stage_position, projection, beam_shift, beam_current
        )
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        vx, vy = dx - ax, dy - ay
        turn = self._turn(stage_position)
        return (
            float(turn * (vx * cos_t + vy * sin_t)),
            float(turn * (-vx * sin_t + vy * cos_t) / fs),
        )

    def mill(
        self,
        patterns,
        beam_type: BeamType,
        stage_position: FibsemStagePosition,
        projection: BeamStageProjection,
        beam_shift: Tuple[float, float] = (0.0, 0.0),
        beam_current: Optional[float] = None,
    ) -> List[MilledRegion]:
        """Commit milling patterns to the world.

        Patterns are in microscope image coordinates of the milling beam's
        view at this pose (metres from the centre, y up); each becomes a
        region on the sample surface that every later view - either beam,
        any tilt - renders as a trench. Rectangles keep their rotation,
        circles become ellipses (foreshortened back onto the surface),
        lines are thin rectangles, bitmaps their bounding box.
        """

        def world(cx, cy):
            return self.view_to_world(
                beam_type, stage_position, projection, cx, -cy, beam_shift, beam_current
            )

        frame = self.view_frame(
            beam_type, stage_position, projection, beam_shift, beam_current
        )

        def cut(corners):
            """The pattern as the milling beam sees it, for a cut along it."""
            c, s_ = np.cos(frame.theta), np.sin(frame.theta)
            polygon = np.array(
                [
                    (
                        (x * c - y * s_) * 1e6 - frame.anchor[0],
                        (-x * s_ - y * c) * 1e6 - frame.anchor[1],
                    )
                    for x, y in corners
                ]
            )
            return sim_view.BeamCut(
                polygon=polygon, turn=frame.turn, fs=frame.fs, k=frame.k
            )

        n_pass = len(self.mill_passes)
        rng = np.random.default_rng([int(self.seed) & 0x7FFFFFFF, 47, n_pass])
        drift = rng.normal(0.0, self.mill_drift, 2) if self.mill_drift > 0 else (0, 0)
        kerf = 0.0
        if self.mill_kerf and beam_current:
            # half the spot: ~15 nm at 100 pA, ~110 nm at 15 nA
            kerf = 0.5 * 30e-9 * (float(beam_current) / 1e-10) ** 0.4
        planned: List[np.ndarray] = []
        actual: List[np.ndarray] = []
        regions: List[MilledRegion] = []
        for p in patterns:
            depth = float(getattr(p, "depth", 1e-6) or 1e-6)
            drawn = len(planned)
            if hasattr(p, "start_x"):  # a line: a thin rectangle along it
                half = float(getattr(p, "width", 0.5e-6) or 0.5e-6) / 2
                dx_, dy_ = p.end_x - p.start_x, p.end_y - p.start_y
                length = np.hypot(dx_, dy_) or 1e-12
                nx, ny = -dy_ / length * half, dx_ / length * half
                corners = [
                    (p.start_x + nx, p.start_y + ny),
                    (p.end_x + nx, p.end_y + ny),
                    (p.end_x - nx, p.end_y - ny),
                    (p.start_x - nx, p.start_y - ny),
                ]
            elif hasattr(p, "radius"):  # a circle: a polygon round it
                t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
                corners = [
                    (
                        p.centre_x + p.radius * np.cos(a),
                        p.centre_y + p.radius * np.sin(a),
                    )
                    for a in t
                ]
            else:  # a rectangle (or a bitmap's bounding box), with its rotation
                rot = float(getattr(p, "rotation", 0.0) or 0.0)
                c, s_ = np.cos(rot), np.sin(rot)
                hw, hh = float(p.width) / 2, float(p.height) / 2
                planned.append(
                    np.array(
                        [
                            (p.centre_x + x * c - y * s_, p.centre_y + x * s_ + y * c)
                            for x, y in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))
                        ]
                    )
                )
                grow = float(rng.uniform(0.0, self.mill_overcut)) + kerf
                hw, hh = hw + grow, hh + grow
                corners = [
                    (p.centre_x + x * c - y * s_, p.centre_y + x * s_ + y * c)
                    for x, y in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))
                ]
            if len(planned) == drawn:  # not a rectangle: drawn as cut
                planned.append(np.array(corners, dtype=float))
            corners = [(x + drift[0], y + drift[1]) for x, y in corners]
            actual.append(np.array(corners, dtype=float))
            points = np.array([world(x, y) for x, y in corners])
            regions.append(
                MilledRegion(points=points, depth=depth, beam_cut=cut(corners))
            )
        record = self._record_pass(planned, actual, drift)
        record["current"] = beam_current
        record["kerf"] = kerf
        if record.get("gap") is not None:
            depth = min(float(getattr(p, "depth", 1e-6) or 1e-6) for p in patterns)
            lean = np.tan(np.deg2rad(self.mill_taper))
            record["gap_at_depth"] = record["gap"] + 2 * lean * depth
        damage = self._damage(record, rng)
        if damage and abs(frame.k) > 1e-6:
            regions.append(self._solid_damage(record, frame, world, cut, rng))
            damage = []
        for corners in damage:
            points = np.array([world(x, y) for x, y in corners])
            regions.append(MilledRegion(points=points, beam_cut=cut(corners)))
        self.milled.extend(regions)
        logger.debug(
            {
                "msg": "sample_scene_milled",
                "regions": len(regions),
                "total": len(self.milled),
            }
        )
        return regions

    def _solid_damage(self, record, frame, world, cut, rng) -> MilledRegion:
        """A damaged lamella: irregular shapes on the
        lamella's face (see sim_solid.damage), cut through its thickness."""
        um = sim_view.UM
        x0, x1, lo, hi = record["band"]
        mid = 0.5 * (lo + hi)
        ends = [tuple(v * um for v in world(x, mid)) for x in (x0, x1)]
        pieces = sim_solid.damage(
            self,
            record["damage"]["kind"],
            record["damage"]["severity"],
            ends,
            -frame.turn * frame.fs / frame.k,
            rng,
            thickness=record["gap"] * um,
        )
        slack = 0.6e-6  # past the faces: a lamella is thicker further down
        band = [(x0, lo - slack), (x1, lo - slack), (x1, hi + slack), (x0, hi + slack)]

        def area(p):
            return abs(
                np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
            )

        # seen from above (the FM, the flat view) it is the biggest piece
        return MilledRegion(
            points=max(pieces, key=area) / um, beam_cut=cut(band), face=pieces
        )

    def _record_pass(self, planned, actual, drift) -> dict:
        """Note a milling pass: what was drawn, what was cut, and the gap the
        pass leaves between the trenches either side of its patterns' middle
        (the lamella's true thickness, m), from everything milled at this
        pose so far."""
        record = {
            "pass": len(self.mill_passes),
            "drift": (float(drift[0]), float(drift[1])),
            "planned_gap": _lamella_gap(planned, planned),
            "gap": None,
            "damage": None,
        }
        self._cuts.extend(actual)
        record["actual"] = actual
        measured = _lamella_gap(actual, self._cuts)
        if measured is not None:
            record["gap"], record["gap_mean"], record["band"] = measured
        if record["planned_gap"] is not None:
            record["planned_gap"] = record["planned_gap"][1]
        self.mill_passes.append(record)
        return record

    def _damage(self, record: dict, rng: np.random.Generator) -> List[np.ndarray]:
        """A thin lamella's damage, as extra cuts (pattern coordinates): it
        starts on a pass below the seeded thickness, by chance, and grows on
        every pass after."""
        if record.get("gap") is None or record["gap"] <= 0:
            return []
        threshold = float(
            np.random.default_rng([int(self.seed) & 0x7FFFFFFF, 53]).uniform(
                *self.damage_below
            )
        )
        started = self.damage_started
        if started is None:
            if record["gap"] >= threshold or rng.uniform() >= self.damage_chance:
                return []
            kind = str(rng.choice(["ragged", "crack", "broken"]))
            self.damage_started = started = {"pass": record["pass"], "kind": kind}
        severity = record["pass"] - started["pass"] + 1
        record["damage"] = {"kind": started["kind"], "severity": severity}
        x0, x1, lo, hi = record["band"]
        width, gap = x1 - x0, hi - lo
        cuts: List[np.ndarray] = []

        def box(xa, xb, ya, yb):
            return np.array([(xa, ya), (xb, ya), (xb, yb), (xa, yb)])

        if started["kind"] == "ragged":
            # bites out of the front face, more each pass
            for _ in range(4 * severity):
                xc = rng.uniform(x0, x1)
                half = rng.uniform(0.1e-6, 0.5e-6)
                bite = gap * rng.uniform(0.3, 0.9)
                if rng.uniform() < 0.5:
                    cuts.append(box(xc - half, xc + half, hi - bite, hi + 1e-9))
                else:
                    cuts.append(box(xc - half, xc + half, lo - 1e-9, lo + bite))
        elif started["kind"] == "crack":
            # a thin crack across, a new one each pass
            for _ in range(severity):
                xc = rng.uniform(x0 + 0.2 * width, x1 - 0.2 * width)
                lean = np.tan(np.deg2rad(rng.uniform(-30, 30))) * gap
                half = 25e-9
                cuts.append(
                    np.array(
                        [
                            (xc - half, lo - 1e-9),
                            (xc + half, lo - 1e-9),
                            (xc + half + lean, hi + 1e-9),
                            (xc - half + lean, hi + 1e-9),
                        ]
                    )
                )
        else:
            # a piece broken out, wider each pass
            xc = rng.uniform(x0 + 0.25 * width, x1 - 0.25 * width)
            half = min(0.5e-6 * severity, 0.45 * width)
            cuts.append(box(xc - half, xc + half, lo - 1e-9, hi + 1e-9))
        return cuts

    def burn(
        self,
        points,
        beam_type: BeamType,
        stage_position: FibsemStagePosition,
        projection: BeamStageProjection,
        beam_shift: Tuple[float, float] = (0.0, 0.0),
        beam_current: Optional[float] = None,
        diameter: float = SPOT_BURN_DIAMETER,
        halo: float = SPOT_BURN_HALO,
    ) -> List[MilledRegion]:
        """Commit spot burns to the world.

        Points are in microscope image coordinates of the burning beam's view
        at this pose (metres from the centre, y up), the same convention
        `mill()` takes. Each becomes a disc of `diameter` on the sample
        surface, foreshortened back the way a circle pattern is, so every
        later view of either beam and the FM shows it as a small dark mark
        with a bright rim of width `halo`.
        """

        def world(cx, cy):
            return self.view_to_world(
                beam_type, stage_position, projection, cx, -cy, beam_shift, beam_current
            )

        frame = self.view_frame(
            beam_type, stage_position, projection, beam_shift, beam_current
        )
        t = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        r = diameter / 2
        regions: List[MilledRegion] = []
        for x, y in points:
            corners = [(x + r * np.cos(a), y + r * np.sin(a)) for a in t]
            pts = np.array([world(cx, cy) for cx, cy in corners])
            if abs(frame.k) > 1e-6:
                # the beam meets the surface, not the film plane: on a cell
                # standing h up, it lands back along the beam by h k / fs
                pts = pts + (0.0, self._along_beam_to_surface(pts.mean(axis=0), frame))
            regions.append(MilledRegion(points=pts, depth=diameter, halo=halo))
        self.milled.extend(regions)
        logger.debug(
            {
                "msg": "sample_scene_burned",
                "points": len(regions),
                "total": len(self.milled),
            }
        )
        return regions

    def _along_beam_to_surface(self, point, frame) -> float:
        """How far (m, world y) the point a beam aims at on the film plane
        moves to where the beam ray meets the sample's surface: a point at
        height h shows where the plane point k h / (turn fs) further on
        does, so the hit is where turn fs (y - y0) + k h(y) = 0, the highest
        such point when the ray crosses the surface more than once."""
        um = sim_view.UM
        x0, y0 = float(point[0]) * um, float(point[1]) * um
        reach = min(abs(frame.k / frame.fs) * 10.0 + 2.0, 60.0)  # um, for 10 um tall
        dy = 0.1
        ts = np.arange(-reach, reach + dy, dy)
        xs = x0 + np.arange(-24, 25) * 0.25
        h = sim_solid.top(self, xs, y0 + ts, (0.25, dy))[:, 24].astype(float)
        f = frame.turn * frame.fs * ts + frame.k * h
        solid = h > sim_view.VOID / 2
        cross = np.nonzero(
            solid[:-1] & solid[1:] & (np.sign(f[:-1]) != np.sign(f[1:]))
        )[0]
        if not len(cross):
            return 0.0
        best = cross[np.argmax(h[cross])]
        # the crossing between the two samples
        t = ts[best] + dy * f[best] / (f[best] - f[best + 1])
        return float(t) / um

    @staticmethod
    def _polygon_mask(pts: np.ndarray, xs_world: np.ndarray, ys_world: np.ndarray):
        """Inside every edge's half-plane of one convex polygon."""
        inside = np.ones(np.broadcast(xs_world, ys_world).shape, dtype=bool)
        # orientation of the polygon decides which side is "inside"
        area = 0.0
        for i in range(len(pts)):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % len(pts)]
            area += x0 * y1 - x1 * y0
        sign = 1.0 if area > 0 else -1.0
        for i in range(len(pts)):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % len(pts)]
            cross = (x1 - x0) * (ys_world - y0) - (y1 - y0) * (xs_world - x0)
            inside &= sign * cross >= 0
        return inside

    def milled_mask(self, xs_world: np.ndarray, ys_world: np.ndarray) -> np.ndarray:
        """Where milled regions lie, from broadcastable world coordinates."""
        mask = np.zeros(np.broadcast(xs_world, ys_world).shape, dtype=bool)
        for r in self.milled:
            mask |= self._polygon_mask(r.points, xs_world, ys_world)
        return mask

    def halo_mask(
        self, xs_world: np.ndarray, ys_world: np.ndarray, pixel_size: float
    ) -> Optional[np.ndarray]:
        """The bright rims round the regions that have one: each region's
        mask grown by its halo width (in view pixels), less every milled
        region. None when nothing has a halo."""
        rims = [r for r in self.milled if r.halo > 0]
        if not rims:
            return None
        shape = np.broadcast(xs_world, ys_world).shape
        grown = np.zeros(shape, dtype=bool)
        for r in rims:
            mark = self._polygon_mask(r.points, xs_world, ys_world)
            if not mark.any():
                continue
            iterations = max(1, int(np.ceil(r.halo / pixel_size)))
            grown |= ndi_binary_dilation(mark, iterations=iterations)
        return grown & ~self.milled_mask(xs_world, ys_world)

    def film_masks(
        self, xs_world: np.ndarray, ys_world: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """What the support is at each world point: (bars, holes, rips,
        rim, beyond) boolean masks, from broadcastable world coordinates.

        Per grid, in the grid's own frame (centre, rotation): the mesh and
        the hole lattice are rotated together; holes exist only on film
        (not on bars), a seeded fraction of them broken into larger
        openings; a seeded fraction of squares is ripped (film gone inside
        an irregular patch). Past the grid's radius is the metal rim; past
        every rim, the holder.
        """
        shape = np.broadcast(xs_world, ys_world).shape
        bars = np.zeros(shape, dtype=bool)
        holes = np.zeros(shape, dtype=bool)
        rips = np.zeros(shape, dtype=bool)
        rim = np.zeros(shape, dtype=bool)
        inside_any = np.zeros(shape, dtype=bool)
        x_lo, x_hi = float(np.min(xs_world)), float(np.max(xs_world))
        y_lo, y_hi = float(np.min(ys_world)), float(np.max(ys_world))
        for grid in self.grids:
            reach = grid.radius + self.grid_rim_width
            if (
                grid.x + reach < x_lo
                or grid.x - reach > x_hi
                or grid.y + reach < y_lo
                or grid.y - reach > y_hi
            ):
                continue
            g_bars, g_holes, g_rips, inside, ring = self._grid_masks(
                grid, xs_world - grid.x, ys_world - grid.y
            )
            bars |= g_bars & inside
            holes |= g_holes & inside
            rips |= g_rips & inside
            rim |= ring
            inside_any |= inside
        rim &= ~inside_any
        beyond = ~(inside_any | rim)
        return bars, holes, rips, rim, beyond

    def _centre_mark(self, x_rot: np.ndarray, y_rot: np.ndarray) -> np.ndarray:
        """The orientation mark at the bar crossing nearest the grid's
        centre: a bigger square of metal with one corner rounded."""
        half = CENTRE_MARK_SIZE * self.grid_bar_width / 2
        corner = 0.6 * half  # radius of the rounded corner
        dx = x_rot - self.grid_pitch / 2
        dy = y_rot - self.grid_pitch / 2
        square = (np.abs(dx) <= half) & (np.abs(dy) <= half)
        cut = (dx > half - corner) & (dy > half - corner)
        rounded = np.hypot(dx - (half - corner), dy - (half - corner)) <= corner
        return square & (~cut | rounded)

    def broken_hole_mask(
        self, xs_world: np.ndarray, ys_world: np.ndarray
    ) -> np.ndarray:
        """Where the film's holes are broken (no ice across them), at world
        points: the seeded few torn patches `film_masks` draws."""
        mask = np.zeros(np.broadcast(xs_world, ys_world).shape, dtype=bool)
        if self.film != "holey" or self.broken_hole_fraction <= 0:
            return mask
        for grid in self.grids:
            cos_r, sin_r = np.cos(grid.rotation), np.sin(grid.rotation)
            xs, ys = xs_world - grid.x, ys_world - grid.y
            x_rot = xs * cos_r + ys * sin_r
            y_rot = -xs * sin_r + ys * cos_r
            mask |= self._broken_holes(x_rot, y_rot, grid.seed)
        return mask

    def _grid_masks(self, grid: SceneGrid, xs: np.ndarray, ys: np.ndarray):
        """One grid's (bars, holes, rips, inside, rim) from coordinates
        relative to its centre."""
        cos_r, sin_r = np.cos(grid.rotation), np.sin(grid.rotation)
        x_rot = xs * cos_r + ys * sin_r
        y_rot = -xs * sin_r + ys * cos_r

        def _near_bar(w):
            return np.abs((w % self.grid_pitch) - self.grid_pitch / 2) < (
                self.grid_bar_width / 2
            )

        bars = _near_bar(x_rot) | _near_bar(y_rot)
        if self.centre_mark:
            bars |= self._centre_mark(x_rot, y_rot)
        radius = np.hypot(xs, ys)
        inside = radius <= grid.radius
        rim = (radius > grid.radius) & (radius <= grid.radius + self.grid_rim_width)

        # a per-square seeded draw, from the square's lattice index. The
        # bars sit at half-pitch (the grid centre is a square's centre), so
        # the square frame is the bar frame shifted by half a pitch
        half_pitch = self.grid_pitch / 2
        sq_i = np.floor((x_rot + half_pitch) / self.grid_pitch).astype(np.int64)
        sq_j = np.floor((y_rot + half_pitch) / self.grid_pitch).astype(np.int64)
        square_draw = _lattice_draw(sq_i, sq_j, grid.seed)
        # local coordinates within the square, zero at its centre
        lx = ((x_rot + half_pitch) % self.grid_pitch) - half_pitch
        ly = ((y_rot + half_pitch) % self.grid_pitch) - half_pitch

        # the rip: each ripped square its own tear. Every quantity is a
        # seeded per-square draw, so a rip is the same from every view:
        # its size (a tear at one edge to most of the square gone), its
        # orientation, where it starts (usually at a bar, running inward)
        # and how ragged its outline is. Rips cluster: a square beside a
        # ripped one rips more readily
        torn = square_draw < self.rip_fraction
        beside_torn = np.zeros_like(torn)
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            beside_torn |= (
                _lattice_draw(sq_i + di, sq_j + dj, grid.seed) < self.rip_fraction
            )
        torn |= beside_torn & (square_draw < 3 * self.rip_fraction)
        torn &= ~bars
        rips = np.zeros_like(bars)
        if torn.any():
            # few squares are torn: work the tears out only in those
            ti, tj = sq_i[torn], sq_j[torn]

            def draw(k):
                return _lattice_draw(ti, tj, grid.seed + k)

            size = self.grid_pitch * (0.15 + 0.35 * draw(11))  # semi-major axis
            aspect = 0.4 + 0.6 * draw(12)
            phi = np.pi * draw(13)
            # the tear starts somewhere between the square's centre and a bar
            psi = 2 * np.pi * draw(14)
            start = (half_pitch - 0.7 * size) * (0.3 + 0.7 * draw(15))
            ux = lx[torn] - start * np.cos(psi)
            uy = ly[torn] - start * np.sin(psi)
            cos_p, sin_p = np.cos(phi), np.sin(phi)
            ex = (ux * cos_p + uy * sin_p) / size
            ey = (-ux * sin_p + uy * cos_p) / (size * aspect)
            angle = np.arctan2(ey, ex)
            ragged = 0.1 + 0.35 * draw(16)
            phase = 2 * np.pi * draw(17)
            wob = 1 + ragged * (
                np.sin(3 * angle + phase) + 0.6 * np.sin(5 * angle + 2 * phase)
            )
            rips[torn] = np.hypot(ex, ey) < wob

        holes = np.zeros_like(bars)
        if self.film == "holey":
            hx = (x_rot % self.hole_pitch) - self.hole_pitch / 2
            hy = (y_rot % self.hole_pitch) - self.hole_pitch / 2
            whole = np.hypot(hx, hy) < self.hole_diameter / 2
            holes = (whole | self._broken_holes(x_rot, y_rot, grid.seed)) & ~bars
        return bars, holes, rips, inside, rim

    def _broken_holes(self, x_rot: np.ndarray, y_rot: np.ndarray, seed: int):
        """Where the film is broken round a seeded few holes, in a grid's
        frame: each a torn patch of its own size, stretch and ragged outline
        about its hole, a little bigger than the hole was."""
        if self.broken_hole_fraction <= 0:
            return np.zeros(np.broadcast(x_rot, y_rot).shape, dtype=bool)
        # a frame asks twice for the same points (the film, then the ice)
        x_rot, y_rot = np.asarray(x_rot), np.asarray(y_rot)
        key = (
            seed, x_rot.shape, y_rot.shape, self.broken_hole_fraction,
            self.hole_pitch, self.hole_diameter,
            float(x_rot.flat[0]), float(x_rot.flat[-1]),
            float(y_rot.flat[0]), float(y_rot.flat[-1]),
        )  # fmt: skip
        last = getattr(self, "_broken_last", None)
        if last is not None and last[0] == key:
            return last[1].copy()
        pitch = self.hole_pitch
        out = np.zeros(np.broadcast(x_rot, y_rot).shape, dtype=bool)
        hi0 = np.floor(x_rot / pitch).astype(np.int64)
        hj0 = np.floor(y_rot / pitch).astype(np.int64)
        # a tear can reach past its own cell: look at the neighbours' holes too
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                broken = (
                    _lattice_draw(hi0 + di, hj0 + dj, seed + 7)
                    < self.broken_hole_fraction
                )
                if not broken.any():
                    continue
                # few holes are broken: work out the tears only round those
                hi, hj = (hi0 + di)[broken], (hj0 + dj)[broken]
                hx = np.broadcast_to(x_rot, broken.shape)[broken] - (hi + 0.5) * pitch
                hy = np.broadcast_to(y_rot, broken.shape)[broken] - (hj + 0.5) * pitch
                size = self.hole_diameter * (
                    0.55 + 0.5 * _lattice_draw(hi, hj, seed + 8)
                )
                phi = np.pi * _lattice_draw(hi, hj, seed + 9)
                stretch = 1.0 + 0.5 * _lattice_draw(hi, hj, seed + 10)
                phase = 2 * np.pi * _lattice_draw(hi, hj, seed + 11)
                c, s_ = np.cos(phi), np.sin(phi)
                u = (hx * c + hy * s_) / stretch
                v = -hx * s_ + hy * c
                t = np.arctan2(v, u)
                ragged = (
                    1
                    + 0.15 * np.sin(2 * t + phase)
                    + 0.1 * np.sin(3 * t + 2 * phase)
                    + 0.04 * np.sin(5 * t + 3 * phase)
                )
                out[broken] |= np.hypot(u, v) < size * ragged
        self._broken_last = (key, out.copy())
        return out

    def _generate_cells(self, rng: np.random.Generator, half: float) -> None:
        kinds = {
            "yeast": [("yeast", 1.0)],
            "mammalian": [("mammalian", 1.0)],
            "bacteria": [("bacteria", 1.0)],
            "rpe": [("rpe", 1.0)],
            "neuron": [("neuron", 1.0)],
            "mixed": [("yeast", 0.5), ("mammalian", 0.4), ("bacteria", 0.3)],
            "none": [],  # bare film: fiducial, bars and contamination only
        }
        if self.cell_type not in kinds:
            raise ValueError(
                f"cell_type must be one of {sorted(kinds)}, got {self.cell_type!r}"
            )
        fields = (2 * half / 150e-6) ** 2  # how many 150 um fields the extent is
        next_id = 0
        for kind, fraction in kinds[self.cell_type]:
            if kind == "yeast":
                next_id = self._generate_yeast(rng, half, fraction, next_id)
            elif kind == "mammalian":
                n = int(round(self.mammalian_density * fields * fraction))
                next_id = self._generate_mammalian(rng, half, n, next_id)
            elif kind == "rpe":
                n = int(round(self.rpe_density * fields * fraction))
                next_id = self._generate_rpe(rng, half, n, next_id)
            elif kind == "neuron":
                n = int(round(self.neuron_density * fields * fraction))
                next_id = self._generate_neurons(rng, half, n, next_id)
            else:
                n = int(round(self.bacteria_density * fields * fraction))
                next_id = self._generate_bacteria(rng, half, n, next_id)

    def _generate_yeast(self, rng, half, fraction, next_id) -> int:
        """Clumps of small round cells packed against each other: each new
        cell touches one already placed, some with a budding daughter."""
        # n_clusters is per 800 x 800 um, so a bigger extent is as dense
        per = (2 * half / 800e-6) ** 2
        for _ in range(int(round(self.n_clusters * fraction * per))):
            cx, cy = (float(v) for v in rng.uniform(-half, half, size=2))
            lo, hi = self.cells_per_cluster
            want = int(rng.integers(lo, hi + 1))
            placed: List[Tuple[float, float, float]] = []
            for _ in range(20 * want):
                if len(placed) >= want:
                    break
                r = float(rng.uniform(*self.cell_size))
                if placed:
                    px, py, pr = placed[int(rng.integers(len(placed)))]
                    t = float(rng.uniform(0, 2 * np.pi))
                    x, y = px + (pr + r) * np.cos(t), py + (pr + r) * np.sin(t)
                    if np.hypot(x - cx, y - cy) > self.cluster_spread:
                        continue
                    if any(
                        np.hypot(x - qx, y - qy) < 0.9 * (r + qr)
                        for qx, qy, qr in placed
                    ):
                        continue
                else:
                    x, y = cx, cy
                placed.append((float(x), float(y), r))
            for x, y, r in placed:
                self.features.append(
                    SceneFeature(
                        x=x,
                        y=y,
                        sigma=r,
                        intensity=float(rng.uniform(60, 110)),
                        sharpness=2.5,
                        eccentricity=float(rng.uniform(0.85, 1.0)),
                        angle=float(rng.uniform(0, np.pi)),
                        cell_id=next_id,
                        opacity=1.0,
                        height=1.5 * r,
                    )
                )
                # a compact nucleus for the FM (the beams barely see it)
                self.features.append(
                    SceneFeature(
                        x=x,
                        y=y,
                        sigma=r * 0.35,
                        intensity=8.0,
                        sharpness=2.0,
                        part="nucleus",
                        cell_id=next_id,
                    )
                )
                if rng.random() < 0.2:  # budding daughter
                    t = rng.uniform(0, 2 * np.pi)
                    self.features.append(
                        SceneFeature(
                            x=x + 1.3 * r * np.cos(t),
                            y=y + 1.3 * r * np.sin(t),
                            sigma=r * 0.5,
                            intensity=70.0,
                            sharpness=2.5,
                            part="bud",
                            cell_id=next_id,
                            opacity=1.0,
                            height=0.8 * r,
                        )
                    )
                next_id += 1
        return next_id

    def _generate_mammalian(self, rng, half, n, next_id) -> int:
        # adherent cells tile the film rather than cluster: keep them apart
        min_distance = 1.6 * self.mammalian_radius[1]
        # one distance check per try, against the cells placed in the
        # neighbouring buckets (a bucket is min_distance across). The tries
        # are drawn in one go and the generator then advanced by exactly the
        # draws used, so the layout is what one draw per try gives
        tries = 60 * max(n, 1)
        state = rng.bit_generator.state
        drawn = rng.uniform(-half, half, size=(tries, 2)).tolist()
        rng.bit_generator.state = state
        placed_x: List[float] = []
        placed_y: List[float] = []
        buckets: Dict[Tuple[int, int], List[int]] = {}
        limit2 = min_distance * min_distance
        used = 0
        for x, y in drawn:
            if len(placed_x) >= n:
                break
            used += 1
            bx, by = math.floor(x / min_distance), math.floor(y / min_distance)
            clear = True
            for key in (
                (bx - 1, by - 1), (bx - 1, by), (bx - 1, by + 1),
                (bx, by - 1), (bx, by), (bx, by + 1),
                (bx + 1, by - 1), (bx + 1, by), (bx + 1, by + 1),
            ):  # fmt: skip
                for k in buckets.get(key, ()):
                    ddx, ddy = x - placed_x[k], y - placed_y[k]
                    if ddx * ddx + ddy * ddy <= limit2:
                        clear = False
                        break
                if not clear:
                    break
            if clear:
                buckets.setdefault((bx, by), []).append(len(placed_x))
                placed_x.append(x)
                placed_y.append(y)
        if used:
            rng.uniform(-half, half, size=(used, 2))
        count = len(placed_x)
        px, py = np.array(placed_x), np.array(placed_y)
        for x, y in zip(px[:count].tolist(), py[:count].tolist()):
            radius = float(rng.uniform(*self.mammalian_radius))
            angle = float(rng.uniform(0, np.pi))
            ecc = float(rng.uniform(0.6, 1.0))
            phase = float(rng.uniform(0, 2 * np.pi))
            # the flat spread body: low, wide, irregular outline
            self.features.append(
                SceneFeature(
                    x=x,
                    y=y,
                    sigma=radius,
                    intensity=28.0,
                    sharpness=2.5,
                    eccentricity=ecc,
                    angle=angle,
                    wobble=0.25,
                    wobble_phase=phase,
                    cell_id=next_id,
                    opacity=0.95,
                )
            )
            # the nuclear mound, off centre
            nx = x + float(rng.normal(0, 0.2 * radius))
            ny = y + float(rng.normal(0, 0.2 * radius))
            nucleus = float(rng.uniform(*self.nucleus_radius))
            self.features.append(
                SceneFeature(
                    x=nx,
                    y=ny,
                    sigma=nucleus,
                    intensity=75.0,
                    sharpness=2.0,
                    eccentricity=0.85,
                    angle=angle,
                    part="nucleus",
                    cell_id=next_id,
                )
            )
            # organelle speckle in the cytoplasm
            for _ in range(int(rng.integers(15, 30))):
                t = rng.uniform(0, 2 * np.pi)
                r = rng.uniform(1.2 * nucleus, 0.9 * radius)
                self.features.append(
                    SceneFeature(
                        x=x
                        + r * np.cos(t) * np.cos(angle)
                        - r * np.sin(t) * ecc * np.sin(angle),
                        y=y
                        + r * np.cos(t) * np.sin(angle)
                        + r * np.sin(t) * ecc * np.cos(angle),
                        sigma=1.2e-6,
                        intensity=12.0,
                        sharpness=2.0,
                        part="organelle",
                        cell_id=next_id,
                    )
                )
            next_id += 1
        return next_id

    def _generate_rpe(self, rng, half, n, next_id) -> int:
        """Epithelial cells standing apart: each a big smooth dome a few
        micrometres tall, with a nucleus a little off its centre."""
        placed = np.zeros((0, 3))
        tries = 0
        while len(placed) < n and tries < 40 * n:
            # in batches: a cell is kept if it stands clear of those before it
            tries += 256
            batch = np.column_stack(
                [
                    rng.uniform(-half, half, 256),
                    rng.uniform(-half, half, 256),
                    rng.uniform(*self.rpe_radius, 256),
                ]
            )
            for x, y, radius in batch:
                gap = np.hypot(placed[:, 0] - x, placed[:, 1] - y)
                if np.any(gap < 1.8 * (placed[:, 2] + radius)):
                    continue
                placed = np.vstack([placed, (x, y, radius)])
                if len(placed) >= n:
                    break
        for x, y, radius in placed:
            self.features.append(
                SceneFeature(
                    x=float(x),
                    y=float(y),
                    sigma=float(radius) / RPE_RADIUS,
                    intensity=45.0,
                    sharpness=2.5,
                    cell_id=next_id,
                    opacity=1.0,
                    height=float(rng.uniform(3.0e-6, 5.5e-6)),
                    # no cell is a circle: each is stretched its own way,
                    # with an uneven outline
                    eccentricity=float(rng.uniform(0.62, 1.0)),
                    angle=float(rng.uniform(0, np.pi)),
                    wobble=float(rng.uniform(0.06, 0.22)),
                    wobble_phase=float(rng.uniform(0, 2 * np.pi)),
                )
            )
            self.features.append(
                SceneFeature(
                    x=float(x + rng.uniform(-3e-6, 3e-6)),
                    y=float(y + rng.uniform(-3e-6, 3e-6)),
                    sigma=0.4 * float(radius) / NUCLEUS_RADIUS,
                    intensity=20.0,
                    sharpness=2.0,
                    part="nucleus",
                    cell_id=next_id,
                    height=0.6e-6,
                )
            )
            next_id += 1
        return next_id

    def _generate_neurons(self, rng, half, n, next_id) -> int:
        """A round soma with a few long processes: each a wandering path of
        thin overlapping segments that tapers, and may branch once."""
        for _ in range(n):
            x, y = (float(v) for v in rng.uniform(-half, half, size=2))
            soma = float(rng.uniform(*self.soma_radius))
            self.features.append(
                SceneFeature(
                    x=x,
                    y=y,
                    sigma=soma,
                    intensity=60.0,
                    sharpness=2.5,
                    eccentricity=float(rng.uniform(0.75, 1.0)),
                    angle=float(rng.uniform(0, np.pi)),
                    wobble=0.1,
                    wobble_phase=float(rng.uniform(0, 2 * np.pi)),
                    cell_id=next_id,
                    opacity=1.0,
                    height=0.6 * soma,
                )
            )
            self.features.append(
                SceneFeature(
                    x=x,
                    y=y,
                    sigma=0.5 * soma,
                    intensity=15.0,
                    sharpness=2.0,
                    part="nucleus",
                    cell_id=next_id,
                    height=0.5e-6,
                )
            )
            paths = []
            start = float(rng.uniform(0, 2 * np.pi))
            count = int(rng.integers(3, 7))
            for k in range(count):
                heading = start + 2 * np.pi * k / count + float(rng.normal(0, 0.3))
                length = float(rng.uniform(*self.neurite_length))
                paths.append((x, y, heading, length, 1.0))
            while paths:
                px, py, heading, length, width = paths.pop()
                px += soma * 0.9 * np.cos(heading) if width == 1.0 else 0.0
                py += soma * 0.9 * np.sin(heading) if width == 1.0 else 0.0
                step = 1.5e-6
                travelled = 0.0
                branched = width < 1.0
                while travelled < length:
                    heading += float(rng.normal(0, 0.12))
                    px += step * np.cos(heading)
                    py += step * np.sin(heading)
                    travelled += step
                    taper = width * (1.0 - 0.6 * travelled / length)
                    self.features.append(
                        SceneFeature(
                            x=float(px),
                            y=float(py),
                            sigma=1.2e-6,
                            intensity=40.0,
                            sharpness=2.0,
                            eccentricity=0.6 * taper,
                            angle=float(heading),
                            part="neurite",
                            cell_id=next_id,
                            height=0.6e-6 * taper,
                        )
                    )
                    if (
                        not branched
                        and travelled > 0.4 * length
                        and rng.random() < 0.02
                    ):
                        branched = True
                        turn = float(rng.choice([-1, 1]) * rng.uniform(0.4, 0.9))
                        rest = length - travelled
                        paths.append((px, py, heading + turn, 0.7 * rest, 0.7 * taper))
            next_id += 1
        return next_id

    def _generate_bacteria(self, rng, half, n, next_id) -> int:
        for _ in range(n):
            x, y = rng.uniform(-half, half, size=2)
            length = float(rng.uniform(*self.bacteria_length))
            width = float(rng.uniform(0.5e-6, 0.7e-6))
            self.features.append(
                SceneFeature(
                    x=float(x),
                    y=float(y),
                    sigma=length / 2,
                    intensity=float(rng.uniform(60, 110)),
                    sharpness=4.0,
                    eccentricity=width / (length / 2),
                    angle=float(rng.uniform(0, np.pi)),
                    cell_id=next_id,
                    opacity=1.0,
                )
            )
            next_id += 1
        return next_id

    # the configuration keys `sim: sample:` accepts, with their units
    CONFIG_KEYS = (
        "coincidence_offset",  # m
        "tilt_axis_offset",  # m
        "grids_from_holder",
        "seed",
        "cell_type",  # mammalian | yeast | bacteria | mixed
        "n_clusters",
        "cells_per_cluster",  # [min, max]
        "cell_size",  # [min, max] m
        "cluster_spread",  # m
        "mammalian_density",  # per 150 x 150 um
        "mammalian_radius",  # [min, max] m
        "nucleus_radius",  # [min, max] m
        "bacteria_density",  # per 150 x 150 um
        "bacteria_length",  # [min, max] m
        "rpe_density",  # per 150 x 150 um
        "rpe_radius",  # [min, max] m
        "neuron_density",  # per 150 x 150 um
        "soma_radius",  # [min, max] m
        "neurite_length",  # [min, max] m
        "extent",  # m
        "grid_pitch",  # m
        "grid_bar_width",  # m
        "grid_intensity",
        "noise_sigma",
        "noise_fraction",
        "fiducial",  # the central cross; off for realistic imaging
        "grid_rotation",  # degrees; null = random within grid_rotation_range
        "grid_rotation_range",  # degrees
        "contamination_density",  # specks per 100 x 100 um
        "contamination_size",  # [min, max] m
        "beam_offset",  # {electron: [dx, dy], ion: [dx, dy]} m
        "current_offset_scale",  # m, sigma of the per-current beam offset
        "beam_optics",  # focus, stigmation and current show in the image
        "focus_fuzz",  # m, how far focus is from the eucentric height, at most
        "scan_rotation_at_start",  # {electron: deg, ion: deg} the beams boot at
        "fm_camera_mount",  # the mount transform the FM camera really has
        "film",  # continuous | holey
        "hole_diameter",  # m
        "hole_pitch",  # m
        "broken_hole_fraction",
        "rip_fraction",  # of squares
        "ice_density",  # per 100 x 100 um
        "ice_size",  # [min, max] m
        "frost_density",  # frost patches per 100 x 100 um
        "sphere_density",  # dark spheres per 100 x 100 um
        "sphere_size",  # [min, max] m, radius
        "crack_density",  # hairline cracks per 100 x 100 um
        "grid_radius",  # m
        "grid_rim_width",  # m
        "mill_drift",  # m, sigma of each milling pass's drift
        "mill_overcut",  # m, how far past its edges a pattern may cut
        "mill_kerf",  # a higher current cuts wider
        "mill_taper",  # degrees the trench walls lean in going down
        "curtaining",  # strength of the stripes down a cut face (0 = none)
        "damage_below",  # [min, max] m, the seeded thickness damage can start below
        "damage_chance",  # per polishing pass below it
        "fm_target_fraction",  # cells with a centrin pool and centriole pair
        "fm_no_target_chance",  # chance a grid carries no pair at all
        "fm_axial_range",  # m, how far from its height an FM spot stays sharp
        "fm_green",  # cytoplasm | centrin | lipid_droplets
        "fm_red",  # subset | cilia
        "fm_far_red",  # none | membrane | mitochondria
        "fm_cilium_fraction",  # of the target cells, those with a cilium
        "fm_hot_pixels",  # fraction of FM camera pixels that are hot
        "charging",  # SEM charging strength on an uncoated grid (0 = never)
        "gis_thickness",  # m, a GIS coat the grid comes in with
        "sputter_thickness",  # m, a sputter coat the grid comes in with
        "centre_mark",  # the orientation square at the grid's centre
    )

    @classmethod
    def from_config(cls, config: dict) -> "SampleScene":
        """Build a scene from the `sim: sample:` block (unknown keys rejected,
        angles in degrees, ranges as two-element lists)."""
        unknown = set(config) - set(cls.CONFIG_KEYS) - {"enabled"}
        if unknown:
            raise ValueError(f"Unknown sim.sample keys: {sorted(unknown)}")
        kwargs = {k: v for k, v in config.items() if k in cls.CONFIG_KEYS}
        kwargs.setdefault("beam_optics", True)
        if kwargs.get("fm_camera_mount") is not None:
            mount_transform_from_name(kwargs["fm_camera_mount"])  # or it raises
        for key in (
            "cells_per_cluster",
            "cell_size",
            "contamination_size",
            "mammalian_radius",
            "nucleus_radius",
            "bacteria_length",
            "ice_size",
            "sphere_size",
            "damage_below",
            "rpe_radius",
            "soma_radius",
            "neurite_length",
        ):
            if key in kwargs:
                kwargs[key] = tuple(kwargs[key])
        if "beam_offset" in kwargs:
            kwargs["beam_offset"] = {
                str(k).lower(): (float(v[0]), float(v[1]))
                for k, v in (kwargs["beam_offset"] or {}).items()
            }
        if "cells_per_cluster" in kwargs:
            kwargs["cells_per_cluster"] = tuple(
                int(v) for v in kwargs["cells_per_cluster"]
            )
        if kwargs.get("grid_rotation") is not None:
            kwargs["grid_rotation"] = float(np.deg2rad(kwargs["grid_rotation"]))
        if "grid_rotation_range" in kwargs:
            kwargs["grid_rotation_range"] = float(
                np.deg2rad(kwargs["grid_rotation_range"])
            )
        return cls(**kwargs)

    def anchor(self, stage_position: FibsemStagePosition) -> None:
        """Fix the world at a stage position (with the boot height error).

        Called once at connect so the world exists before any image: moving
        straight to a saved position and acquiring must show that position's
        surroundings, not anchor the world (fiducial included) there.
        """
        reference = deepcopy(stage_position)
        reference.z = (reference.z or 0.0) - self.coincidence_offset
        self.reference_position = reference
        logger.debug(
            {
                "msg": "coincidence_scene_anchored",
                "reference_position": reference.to_dict(),
                "coincidence_offset": self.coincidence_offset,
            }
        )

    def render(
        self,
        beam_type: BeamType,
        stage_position: FibsemStagePosition,
        hfw: float,
        resolution: Tuple[int, int],
        projection: BeamStageProjection,
        rng: Optional[np.random.Generator] = None,
        beam_shift: Tuple[float, float] = (0.0, 0.0),
        beam_current: Optional[float] = None,
        dwell_time: Optional[float] = None,
        optics: Optional[sim_optics.Optics] = None,
    ) -> np.ndarray:
        """Render one beam's view of the scene at the current stage position.

        All world-to-view mapping goes through the supplied BeamStageProjection
        - the same machinery the minimap/overview and click handlers use - so
        navigation, tiled stitching, reprojection, scan rotation, and the
        per-beam projection of a height error (the coincidence displacement)
        are all consistent with the app by construction.

        Args:
            beam_type: which beam's view to render.
            stage_position: current stage position.
            hfw: horizontal field width in metres.
            resolution: (width, height) in pixels.
            projection: the beam's stage projection (from the live geometry).
            rng: noise source; a fresh default generator when omitted.
            beam_shift: the beam's current shift (m); the configured
                `beam_offset` for the beam is added to it.
            beam_current: the beam current (A); each distinct current carries
                its own seeded offset (see current_offset_scale).
            optics: the beam's focus, stigmation and current, which blur the
                image and set its noise; None draws it sharp.

        Returns:
            uint8 image of shape (height, width).
        """
        width, height = int(resolution[0]), int(resolution[1])
        if self.reference_position is None:
            self.anchor(stage_position)
        self.place_grids(projection)
        frame = self.view_frame(
            beam_type, stage_position, projection, beam_shift, beam_current
        )
        beam = "SEM" if beam_type is BeamType.ELECTRON else "FIB"
        charge = 0.0
        if beam_type is BeamType.ELECTRON:
            charge = self.charge_strength(dwell_time)
        return sim_view.render(
            self,
            beam,
            frame,
            hfw / width,
            (width, height),
            rng,
            charge,
            sim_solid.canvas,
            self._optics_here(optics, stage_position, projection, frame),
        )

    def render_fm(
        self,
        stage_position: FibsemStagePosition,
        resolution: Tuple[int, int],
        projection: FMStageProjection,
        weights: Tuple[float, float, float, float] = (0.05, 1.0, 0.0, 0.0),
        defocus: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Render the fluorescence camera's view of the scene.

        The same world the beams image, through the FM's own projection
        (camera tilt, image transform, pixel size), so a feature the SEM
        shows sits where the FM projection says it does. Channels are what
        a cryo-CLEM grid looks like: `reflection` shows the grid bars (and
        the cells faintly), `green` the cells, `red` a seeded subset of them,
        `blue` the fiducial; anything else renders dark. `defocus` (m, the
        objective's distance from its focus position) blurs the image.

        Args:
            stage_position: current stage position.
            resolution: (width, height) in pixels, after binning.
            projection: the FM stage projection (from the live geometry).
            weights: per-structure intensities (see fm_channel_weights).
            defocus: objective distance from focus, in metres.
            rng: noise source; a fresh default generator when omitted.

        Returns:
            uint16 image of shape (height, width).
        """
        width, height = int(resolution[0]), int(resolution[1])
        pixel_size = projection.pixel_size
        cx, cy = width / 2, height / 2
        if self.reference_position is None:
            self.anchor(stage_position)
        self.place_grids(projection)

        reference = self.reference_position
        if self.tilt_axis_offset:
            reference = self._non_eucentric_reference(stage_position, projection)
        # the FM's projection has no half-turn flip: reflect the anchor as the
        # beams' does, then turn the world about it (see _turn)
        reference = BeamStageProjection._compucentric_corrected(
            reference, stage_position, projection.geometry.rotation_centre
        )
        ax, ay = projection.to_plane(reference, stage_position)
        fs = max(surface_foreshortening(projection, stage_position), MIN_FORESHORTENING)
        turn = self._turn(stage_position)

        bars, cells, red, fiducial = weights[:4]
        far_red = weights[4] if len(weights) > 4 else 0.0
        # contamination autofluoresces faintly
        contamination = 0.12
        canvas = np.zeros((height, width), dtype=np.float32)

        xs_world = turn * (
            (np.arange(width, dtype=np.float32) - cx)[None, :] * pixel_size - ax
        )
        ys_world = (
            turn
            * ((np.arange(height, dtype=np.float32) - cy)[:, None] * pixel_size - ay)
            / fs
        )
        # the sample seen from above: what was milled, at what height
        self._plan = self._fm_plan(xs_world, ys_world)
        if bars >= 1.0:
            return self._fm_reflection(xs_world, ys_world, defocus, rng)
        _, _, _, rim, beyond = self.film_masks(xs_world, ys_world)
        if bars > 0:
            # a little reflected light leaks into a fluorescence channel
            plan, step = self._plan
            leak = sim_solid.reflectance(self, plan)
            canvas += (
                0.3
                * bars
                * self.grid_intensity
                * self._fm_full(leak, step, height, width)
            )
        # ice barely fluoresces
        ice = 0.04

        # only the features in view - the loop itself is the cost, most
        # of the scene returning untouched from its bounding-box check
        in_view = self.feature_index().within(
            float(xs_world.min()),
            float(xs_world.max()),
            float(ys_world.min()),
            float(ys_world.max()),
        )
        dyed: dict = {}  # per cell, how strongly its whole body is dyed
        for idx in in_view:
            f = self.features[idx]
            if f.kind == "fiducial":
                weight = fiducial * f.intensity
            elif f.kind == "contamination":
                weight = contamination * f.intensity
            elif f.kind == "ice":
                weight = ice * f.intensity
            else:
                # dyes by part: the DNA dye lives in the nucleus, the
                # cytoplasmic one everywhere; the subset dye follows the cell
                # a centrin-GFP cell is only faint in green outside its
                # centrosome; with droplets labelled, faint autofluorescence
                glow = {"cytoplasm": 1.0, "centrin": 0.3}.get(self.fm_green, 0.15)
                # the cilia label also sits faintly over the whole cell
                red_glow = 0.2 if self.fm_red == "cilia" else 0.0
                body_dye = glow * cells + (
                    red if self._in_red_subset(f.cell_id) else red_glow * red
                )
                if f.part == "nucleus":
                    weight = FM_INTENSITY * (fiducial * 1.0 + body_dye * 0.5)
                elif f.part == "organelle":
                    weight = FM_INTENSITY * body_dye * 0.6
                else:
                    weight = FM_INTENSITY * body_dye
                if f.part in ("body", "nucleus", "bud"):
                    # the solid sample: a dye through the cell glows by how
                    # much cell there is (see _fm_glow), not as a flat shape
                    if f.part != "nucleus":
                        dyed[int(f.cell_id)] = max(
                            dyed.get(int(f.cell_id), 0.0), FM_INTENSITY * body_dye
                        )
                    weight = FM_INTENSITY * fiducial if f.part == "nucleus" else 0.0
            if weight <= 0:
                continue
            if f.part == "organelle" and f.sigma < 1e-6:
                # a small granule takes up little dye, and sits at a height
                # in its cell: sharp there, spreading and dimming away from it
                f, weight = self._fm_at_height(f, 0.25 * weight, defocus)
            u = cx + (turn * f.x + ax) / pixel_size
            v = cy + (turn * f.y * fs + ay) / pixel_size
            self._stamp_feature(
                canvas, f, u, v, pixel_size, fs, np.pi * (turn < 0), intensity=weight
            )
        # milled regions after the features: a trench or a burn removes the
        # material, so in reflection it reads dark through whatever cell was
        # there, and a burn's rim stays visible on top of it
        if self.milled:
            # a cell is a column of dye a few um tall: a cut along the
            # milling beam takes away the part of each column it passes
            # through, so from above a trench reaches as far as the SEM sees
            canvas *= 1.0 - self.milled_fraction(xs_world, ys_world)
        if dyed:
            canvas += self._fm_glow(dyed, xs_world, ys_world, pixel_size)
        if bars > 0 and self.milled:
            canvas[self.milled_mask(xs_world, ys_world)] = -bars * MILL_DEPTH
            halo = self.halo_mask(xs_world, ys_world, pixel_size)
            if halo is not None:
                canvas[halo] += bars * FM_HALO
        canvas[rim] = 0.0
        canvas[beyond] = 0.0

        if defocus:
            # a retracted objective is millimetres from focus: cap the blur so
            # the frame reads as "out of focus" without costing more than the
            # rest of the render. A wide blur is done on a decimated frame and
            # scaled back - what a 40 px gaussian removes, decimating by 8
            # already removed
            sigma_px = min(abs(defocus) * 1e6 * self.fm_blur_px_per_um, 40.0)
            if sigma_px > 0.3:
                canvas = _blur(canvas, sigma_px)

        view = (ax, ay, cx, cy, pixel_size, fs, turn)
        if getattr(self, "fm_spots", None):
            if cells > 0 and self.fm_green in ("cytoplasm", "centrin"):
                self._stamp_centrin(canvas, cells, defocus, *view)
            if red > 0 and self.fm_red == "cilia":
                self._stamp_cilia(canvas, red, defocus, *view)
        bodies = [
            self.features[i]
            for i in in_view
            if self.features[i].kind == "cell" and self.features[i].part == "body"
        ]
        if red > 0 and self.fm_red == "cilia":
            # and in vesicles: bright puncta through the cells, the decoys
            self._stamp_organelles(canvas, red, "puncta", bodies, defocus, view)
        if cells > 0 and self.fm_green == "lipid_droplets":
            self._stamp_organelles(canvas, cells, "droplets", bodies, defocus, view)
        if far_red > 0 and self.fm_far_red == "mitochondria":
            self._stamp_organelles(canvas, far_red, "mito", bodies, defocus, view)
        if far_red > 0 and self.fm_far_red == "membrane":
            gone = (
                self.milled_fraction(xs_world, ys_world) > 0.5 if self.milled else None
            )
            self._stamp_membrane(
                canvas, far_red, bodies, defocus, view, gone, (xs_world, ys_world)
            )

        return self._fm_camera(canvas, rng)

    def _fm_reflection(self, xs_world, ys_world, defocus, rng):
        """Reflected light from the same 3D sample the beams see, so ice
        thickness reads as fringes, cells as dark textured mounds and a
        lamella or trench where the milling left it. Worked out on at most
        ~512 samples a side and scaled up: the optics blur finer detail."""
        plan, step = self._plan
        r = sim_solid.reflectance(self, plan)
        return self._fm_reflected(r, step, xs_world, ys_world, defocus, rng)

    def _fm_plan(self, xs_world, ys_world):
        """The solid sample from above on the FM's pixels (at most ~512 a
        side: the optics blur finer detail), and that decimation."""
        um = sim_view.UM
        height, width = ys_world.shape[0], xs_world.shape[1]
        step = max(1, int(np.ceil(max(height, width) / 512)))
        xs = xs_world[0, ::step].astype(np.float64) * um
        ys = ys_world[::step, 0].astype(np.float64) * um
        return sim_solid.plan(self, xs, ys), step

    @staticmethod
    def _fm_full(r: np.ndarray, step: int, height: int, width: int) -> np.ndarray:
        """A decimated frame back at the camera's pixels."""
        if step > 1:
            r = ndi_zoom(r, step, order=1)[:height, :width]
            if r.shape != (height, width):
                r = np.pad(
                    r,
                    ((0, height - r.shape[0]), (0, width - r.shape[1])),
                    mode="edge",
                )
        return r

    def _fm_glow(self, dyed: dict, xs_world, ys_world, pixel_size) -> np.ndarray:
        """The glow of cells dyed through: brightest where the cell is
        thickest, gone where it was milled away, with the haze a widefield
        FM adds from above and below the focal plane."""
        plan, step = self._plan
        depth, cell_id = sim_solid.cell_depth(plan)
        ids = np.array(sorted(dyed), dtype=np.int64)
        level = np.array([dyed[i] for i in ids], dtype=np.float32)
        at = np.clip(np.searchsorted(ids, cell_id), 0, len(ids) - 1)
        glow = np.where(ids[at] == cell_id, level[at], 0.0) * depth / FM_CELL_DEPTH
        height, width = ys_world.shape[0], xs_world.shape[1]
        glow = self._fm_full(glow.astype(np.float32), step, height, width)
        sharp = _blur(glow, max(FM_PSF_SIGMA / pixel_size, 0.6))
        haze = _blur(glow, min(FM_GLOW_HAZE / pixel_size, 60.0))
        # a real cell is not evenly dyed: brighter and dimmer patches
        um = sim_view.UM
        wx, wy = np.broadcast_arrays(xs_world * um, ys_world * um)
        patchy = 1 + 0.3 * sim_view.value_noise(wx, wy, 6.0, int(self.seed) + 71)
        return (0.3 * sharp + 0.7 * haze) * np.clip(patchy, 0.5, 1.5)

    def _fm_reflected(self, r, step, xs_world, ys_world, defocus, rng):
        """The reflection frame from the sample's reflectance."""
        height, width = ys_world.shape[0], xs_world.shape[1]
        canvas = self._fm_full(r, step, height, width) * self.grid_intensity
        if self.milled:
            # a spot burn's rim reflects brightly round its pit: a fiducial
            pixel = abs(float(xs_world[0, -1] - xs_world[0, 0])) / max(width - 1, 1)
            halo = self.halo_mask(xs_world, ys_world, pixel)
            if halo is not None:
                canvas[halo] += FM_HALO
        if defocus:
            sigma_px = min(abs(defocus) * 1e6 * self.fm_blur_px_per_um, 40.0)
            if sigma_px > 0.3:
                canvas = _blur(canvas, sigma_px)
        return self._fm_camera(canvas, rng)

    def _fm_camera(self, canvas: np.ndarray, rng) -> np.ndarray:
        """The canvas as the camera reads it."""
        height, width = canvas.shape
        if rng is None:
            rng = np.random.default_rng()
        # 16-bit camera: a dim background, shot-noise-like gaussian noise
        data = 400.0 + canvas * 120.0
        data += rng.normal(0, 40.0 + 0.05 * np.sqrt(np.maximum(data, 0)), canvas.shape)
        if self.fm_hot_pixels > 0:
            # the camera's own: the same pixels at the same level every frame
            hot = np.random.default_rng(
                [int(self.seed) & 0x7FFFFFFF, 61, width, height]
            )
            n = int(self.fm_hot_pixels * width * height)
            rows = hot.integers(0, height, n)
            cols = hot.integers(0, width, n)
            data[rows, cols] = hot.uniform(4000, 30000, n)
        return np.clip(data, 0, 65535).astype(np.uint16)

    def _fm_at_height(self, f: SceneFeature, weight: float, defocus: float):
        """A small feature blurred for its distance from the focal plane: its
        height is seeded from where it is, so it is the same in every view."""
        rng = np.random.default_rng(
            [
                int(self.seed) & 0x7FFFFFFF,
                67,
                int(f.x * 1e8) & 0x7FFFFFFF,
                int(f.y * 1e8) & 0x7FFFFFFF,
            ]
        )
        z = float(rng.uniform(0.3e-6, 2.0e-6)) + self._fm_film()
        spread = ((float(defocus or 0.0) - z) / max(self.fm_axial_range, 1e-9)) ** 2
        sigma = float(np.hypot(f.sigma, FM_PSF_SIGMA * np.sqrt(spread)))
        return replace(f, sigma=sigma, sharpness=1.0), weight * (f.sigma / sigma) ** 2

    def _stamp_centrin(self, canvas, weight, defocus, ax, ay, cx, cy, pix, fs, turn):
        """The centrin pools and centriole pairs, each blurred by how far it
        sits from the focal plane (the objective's distance from focus is the
        height in focus): a spot is sharp within `fm_axial_range` and spreads
        and dims either side."""
        z_plane = float(defocus or 0.0)
        z_r = max(self.fm_axial_range, 1e-9)
        height, width = canvas.shape

        def stamp(x, y, z, sigma0, peak):
            # the point spread widens away from focus; a wide source (the
            # pool) barely changes, a point (a centriole) spreads and dims
            dz = (z_plane - z - self._fm_film()) / z_r
            sigma_m = float(np.hypot(sigma0, FM_PSF_SIGMA * abs(dz)))
            amp = weight * peak * (sigma0 / sigma_m) ** 2
            if amp < FM_FAINT:
                return  # spread so far it is lost in the camera noise
            sigma = max(sigma_m / pix, 0.6)
            u = cx + (turn * x + ax) / pix
            v = cy + (turn * y * fs + ay) / pix
            r = int(np.ceil(4 * sigma))
            u0, u1 = max(int(u) - r, 0), min(int(u) + r + 2, width)
            v0, v1 = max(int(v) - r, 0), min(int(v) + r + 2, height)
            if u0 >= u1 or v0 >= v1:
                return
            uu = np.arange(u0, u1)[None, :] - u
            vv = np.arange(v0, v1)[:, None] - v
            canvas[v0:v1, u0:u1] += amp * np.exp(-(uu**2 + vv**2) / (2 * sigma**2))

        for spot in self.fm_spots:
            x, y, z, sigma = spot["pool"]
            if not self.milled_through(x, y, z):
                stamp(x, y, z, sigma, 0.6 * FM_INTENSITY)
            for x, y, z in spot["pair"]:
                if not self.milled_through(x, y, z):
                    stamp(x, y, z, FM_PSF_SIGMA, 2.0 * FM_INTENSITY)

    def _stamp_cilia(self, canvas, weight, defocus, ax, ay, cx, cy, pix, fs, turn):
        """The primary cilia: a thin bending line of membrane label with a
        bright head at its base, points along it each blurred for its own
        height."""
        pts, heads = [], []
        for spot in self.fm_spots:
            if spot.get("cilium") is None:
                continue
            path = np.asarray(spot["cilium_path"], dtype=float)
            # points every ~40 nm along the bending path
            fine = np.linspace(0, len(path) - 1, 2 * len(path) + 1)
            k = np.floor(fine).astype(int).clip(0, len(path) - 2)
            frac = (fine - k)[:, None]
            pts.append(path[k] * (1 - frac) + path[k + 1] * frac)
            heads.append(path[0])
        view = (ax, ay, cx, cy, pix, fs, turn)
        if pts:
            self._stamp_points(
                canvas,
                np.concatenate(pts),
                CILIUM_SIGMA,
                0.45 * FM_INTENSITY * weight,
                defocus,
                view,
            )
            # the base, where the receptor gathers: a brighter head
            self._stamp_points(
                canvas,
                np.array(heads),
                0.2e-6,
                2.5 * FM_INTENSITY * weight,
                defocus,
                view,
            )

    def _fm_cell_points(self, f: SceneFeature, kind: str) -> np.ndarray:
        """A cell's labelled organelles as (N, 3) world points (x, y, height
        m), drawn from the seed and the cell, so they are the same in every
        view: lipid droplets (a few round spheres, each one point with its
        radius as the 4th column) or mitochondria (many short wandering
        tubules, as points along them)."""
        cache = self.__dict__.setdefault("_fm_points", {})
        key = (kind, int(f.cell_id), round(f.x, 9), round(f.y, 9))
        if key in cache:
            return cache[key]
        code = {"droplets": 73, "mito": 79, "puncta": 83}[kind]
        rng = np.random.default_rng(
            [int(self.seed) & 0x7FFFFFFF, code, int(f.cell_id) & 0x7FFFFFFF]
        )
        tall = max(float(f.height or 0.0), 1.5e-6)
        reach = 0.8 * f.sigma
        if kind == "puncta":
            n = int(rng.integers(4, 16))
            r = reach * np.sqrt(rng.uniform(0, 1, n))
            t = rng.uniform(0, 2 * np.pi, n)
            pts = np.column_stack(
                [
                    f.x + r * np.cos(t),
                    f.y + r * np.sin(t),
                    rng.uniform(0.3e-6, 0.8 * tall, n),
                ]
            )
        elif kind == "droplets":
            n = int(rng.integers(3, 11))
            r = reach * np.sqrt(rng.uniform(0, 1, n))
            t = rng.uniform(0, 2 * np.pi, n)
            pts = np.column_stack(
                [
                    f.x + r * np.cos(t),
                    f.y + r * np.sin(t),
                    rng.uniform(0.3e-6, 0.8 * tall, n),
                    rng.uniform(0.25e-6, 0.8e-6, n),
                ]
            )
        else:
            tubes = []
            for _ in range(int(rng.integers(20, 40))):
                r = reach * np.sqrt(rng.uniform(0.05, 1))
                t = rng.uniform(0, 2 * np.pi)
                start = np.array(
                    [
                        f.x + r * np.cos(t),
                        f.y + r * np.sin(t),
                        rng.uniform(0.3e-6, 0.6 * tall),
                    ]
                )
                steps = int(rng.integers(8, 30))  # 0.8 - 3 um long
                heading = rng.uniform(0, 2 * np.pi) + np.cumsum(
                    rng.normal(0, 0.35, steps)
                )
                step = 0.1e-6 * np.column_stack(
                    [np.cos(heading), np.sin(heading), rng.normal(0, 0.3, steps)]
                )
                tubes.append(start + np.cumsum(step, axis=0))
            pts = np.concatenate(tubes)
        cache[key] = pts
        return pts

    def _stamp_organelles(self, canvas, weight, kind, bodies, defocus, view):
        """Every in-view cell's lipid droplets or mitochondria."""
        if not bodies:
            return
        pts = np.concatenate([self._fm_cell_points(f, kind) for f in bodies])
        if kind == "droplets":
            # a filled sphere reads as a bright disc its own size
            for radius in np.unique(np.round(pts[:, 3], 7)):
                group = pts[np.round(pts[:, 3], 7) == radius]
                self._stamp_points(
                    canvas,
                    group[:, :3],
                    0.5 * radius,
                    1.5 * FM_INTENSITY * weight,
                    defocus,
                    view,
                )
        elif kind == "puncta":
            self._stamp_points(
                canvas, pts, FM_PSF_SIGMA, 1.2 * FM_INTENSITY * weight, defocus, view
            )
        else:
            self._stamp_points(
                canvas, pts, 0.12e-6, 0.4 * FM_INTENSITY * weight, defocus, view
            )

    def _stamp_membrane(
        self, canvas, weight, bodies, defocus, view, gone=None, world=None
    ):
        """The plasma membrane: each in-view cell's outline, a bright ring
        where its body's edge falls, blurred by the FM's own defocus."""
        ax, ay, cx, cy, pix, fs, turn = view
        mask = np.zeros_like(canvas)
        for f in bodies:
            u = cx + (turn * f.x + ax) / pix
            v = cy + (turn * f.y * fs + ay) / pix
            self._stamp_feature(
                mask, f, u, v, pix, fs, np.pi * (turn < 0), intensity=1.0
            )
        mask = np.clip(mask, 0, 1)
        # the outline: a thin band where the body's edge falls (its
        # half-height contour), over a faint top
        if world is not None:
            # a real outline wanders, thickens and fades along its length,
            # with brighter patches (ruffles, contacts) and gaps
            um = sim_view.UM
            wx, wy = np.broadcast_arrays(world[0] * um, world[1] * um)
            seed = int(self.seed) + 97
            mask = mask + 0.12 * sim_view.value_noise(wx, wy, 2.5, seed)
            patchy = sim_view.value_noise(wx, wy, 1.2, seed + 1)
            spots = np.clip(sim_view.value_noise(wx, wy, 0.4, seed + 2), 0, 1) ** 3
        else:
            patchy, spots = 0.0, 0.0
        band = np.exp(-(((mask - 0.5) / 0.12) ** 2))
        ring = band * np.clip(0.75 + 0.6 * patchy + 2.0 * spots, 0.1, 2.5)
        ring = ring + 0.1 * np.clip(mask, 0, 1)
        if gone is not None:
            ring[gone] = 0.0  # milled away
        blur = abs(float(defocus or 0.0)) / max(self.fm_axial_range, 1e-9)
        if blur > 0.5:
            ring = _blur(ring, min(blur * FM_PSF_SIGMA / pix, 40.0))
        canvas += FM_INTENSITY * weight * ring

    def _stamp_points(self, canvas, pts, sigma0, peak, defocus, view):
        """Many point-like sources at once (N, 3: x, y, height m), each
        blurred for its distance from the focal plane: splatted into one
        image per blur width, then each blurred once. A source's peak in
        focus is `peak`; out of focus it spreads and dims."""
        ax, ay, cx, cy, pix, fs, turn = view
        height, width = canvas.shape
        u = cx + (turn * pts[:, 0] + ax) / pix
        v = cy + (turn * pts[:, 1] * fs + ay) / pix
        dz = float(defocus or 0.0) - pts[:, 2] - self._fm_film()
        dz = dz / max(self.fm_axial_range, 1e-9)
        sigma_m = np.hypot(sigma0, FM_PSF_SIGMA * np.abs(dz))
        amp = peak * (sigma0 / sigma_m) ** 2
        sigma_px = np.maximum(sigma_m / pix, 0.6)
        keep = amp >= FM_FAINT
        keep &= (u > -4 * sigma_px) & (u < width + 4 * sigma_px)
        keep &= (v > -4 * sigma_px) & (v < height + 4 * sigma_px)
        plan = getattr(self, "_plan", None)
        if self.milled and plan is not None:
            # what a pattern cut away is gone
            um = sim_view.UM
            keep &= ~plan[0].gone_at(
                pts[:, 0] * um, pts[:, 1] * um, (pts[:, 2] + self._fm_film()) * um
            )
        elif self.milled:
            for i in np.flatnonzero(keep):
                keep[i] = not self.milled_through(*pts[i, :3])
        if not np.any(keep):
            return
        u, v, amp, sigma_px = u[keep], v[keep], amp[keep], sigma_px[keep]
        level = np.round(np.log(sigma_px) / np.log(1.3)).astype(int)
        for lv in np.unique(level):
            sel = level == lv
            s_px = float(1.3**lv)
            img = np.zeros_like(canvas)
            # bilinear splat, so a source between pixels stays put
            u0, v0 = np.floor(u[sel]).astype(int), np.floor(v[sel]).astype(int)
            fu, fv = u[sel] - u0, v[sel] - v0
            mass = amp[sel] * 2 * np.pi * s_px**2
            for du, dv, wgt in (
                (0, 0, (1 - fu) * (1 - fv)),
                (1, 0, fu * (1 - fv)),
                (0, 1, (1 - fu) * fv),
                (1, 1, fu * fv),
            ):
                uu, vv = u0 + du, v0 + dv
                ok = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
                np.add.at(img, (vv[ok], uu[ok]), (mass * wgt)[ok])
            canvas += _blur(img, s_px)

    def milled_fraction(
        self, xs_world: np.ndarray, ys_world: np.ndarray, top: float = 3e-6
    ) -> np.ndarray:
        """How much of the column from the film up to `top` (m) the milled
        patterns cut away at each world point (0..1), each cut along the
        beam that milled it."""
        um = sim_view.UM
        plan = getattr(self, "_plan", None)
        if plan is not None and plan[0].col["surface"].shape == (
            len(ys_world[:: plan[1], 0]),
            len(xs_world[0, :: plan[1]]),
        ):
            # the solid sample: its cells stand taller, from the film up
            film = sim_solid.FILM_TOP
            gone = plan[0].fraction(film + 0.05, film + 2 * top * um)
            return self._fm_full(gone, plan[1], ys_world.shape[0], xs_world.shape[1])
        xs, ys = np.broadcast_arrays(xs_world * um, ys_world * um)
        heights = np.linspace(0.05, top * um, 24)
        gone = np.zeros((len(heights),) + xs.shape, dtype=bool)
        for region in self.milled:
            lo, hi = sim_view._cut_range(self, region, xs, ys)
            for i, z in enumerate(heights):
                gone[i] |= (lo <= z) & (z <= hi)
        return gone.mean(axis=0).astype(np.float32)

    def milled_through(self, x: float, y: float, z: float) -> bool:
        """Whether a milled pattern has cut away the point at (x, y) world
        m, `z` m above the film: each pattern is cut along the beam that
        milled it, so a lamella keeps only what lies inside its slab."""
        um = sim_view.UM
        plan = getattr(self, "_plan", None)
        if plan is not None:
            return bool(plan[0].gone_at(x * um, y * um, (z + self._fm_film()) * um)[0])
        return self._milled_away(x, y, z)

    def _milled_away(self, x: float, y: float, z: float) -> bool:
        """`milled_through`, from the sample round the point as it is now,
        whatever view was drawn last."""
        um = sim_view.UM
        near = np.arange(-16, 17) * 0.25  # wide enough to level a floor
        here = sim_solid.Plan(self, x * um + near, y * um + near)
        return bool(here.gone_at(x * um, y * um, (z + self._fm_film()) * um)[0])

    def record(self, points: Optional[Dict[str, tuple]] = None) -> dict:
        """What was done to the sample, as plain values that go straight to
        JSON: each milling pass (drawn and cut gaps, drift, current, damage),
        every milled region, the damage, and whether each target - and each
        of `points`, a name to (x, y world m, z m above the film) - was
        milled away."""

        def plain(v):
            if isinstance(v, dict):
                return {str(k): plain(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [plain(x) for x in v]
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, np.generic):
                return v.item()
            return v

        targets = []
        for n, t in enumerate(self.fm_targets()):
            x, y, z = t["midpoint"]
            targets.append(
                {
                    "target": n,
                    "midpoint": [x, y, z],
                    "milled": self._milled_away(x, y, z),
                }
            )
        return plain(
            {
                "seed": self.seed,
                "passes": self.mill_passes,
                "damage_started": self.damage_started,
                "regions": [
                    {
                        "points": r.points,
                        "depth": r.depth,
                        "spot_burn": r.halo > 0,
                        "damage": r.face is not None,
                    }
                    for r in self.milled
                ],
                "targets": targets,
                "points": {
                    name: {"point": list(p), "milled": self._milled_away(*p)}
                    for name, p in (points or {}).items()
                },
            }
        )

    def _fm_film(self) -> float:
        """The height (m) of the film in the focal frame: heights in the FM
        are given above the film, and the solid sample's film lies under its
        ice, below the plane the stage brings into focus."""
        return sim_solid.FILM_TOP / sim_view.UM

    def _in_red_subset(self, cell_id: int) -> bool:
        """Whether a cell carries the subset dye: drawn from the seed and
        the cell's id, so the answer is the same whichever features are in
        view (a draw in feature order would change with the view)."""
        rng = np.random.default_rng([self.seed + 1, int(cell_id) & 0x7FFFFFFF])
        if self.fm_red != "subset":
            return False  # the red label is on something else
        return bool(rng.random() < self.red_fraction)

    def current_offset(
        self, beam_type: BeamType, beam_current: Optional[float]
    ) -> Tuple[float, float]:
        """The beam offset that goes with a beam current: drawn once per
        (beam, current) from the scene's seed, so it is reproducible across
        a session and across sessions with the same seed - a lookup table
        nobody wrote down, like the real one."""
        if beam_current is None or self.current_offset_scale <= 0:
            return (0.0, 0.0)
        key = (beam_type.name.lower(), float(f"{beam_current:.6g}"))
        if key not in self._current_offsets:
            # a deterministic hash - Python's own is salted per process
            rng = np.random.default_rng(
                [self.seed, zlib.crc32(key[0].encode()), int(abs(key[1]) * 1e15)]
            )
            self._current_offsets[key] = tuple(
                float(v) for v in rng.normal(0, self.current_offset_scale, size=2)
            )
        return self._current_offsets[key]

    def focus_depth(
        self, stage_position: FibsemStagePosition, projection: BeamStageProjection
    ) -> float:
        """How much further along the beam than the eucentric point the
        sample is at the view's centre (m): a surface that sits low is further
        away, more so the more the beam grazes it."""
        reference = self.reference_position
        if reference is None:
            return 0.0
        if self.tilt_axis_offset:
            reference = self._non_eucentric_reference(stage_position, projection)
        height = (reference.z or 0.0) - (stage_position.z or 0.0)
        fs = max(surface_foreshortening(projection, stage_position), MIN_FORESHORTENING)
        return float(-height / fs)

    def _optics_here(self, optics, stage_position, projection, frame):
        """The beam's optics on this view of the sample: focus goes by how
        far the surface is, at the centre and across a tilted view."""
        if optics is None or not self.beam_optics:
            return None
        # down the unrotated view the surface runs away from the beam (or
        # toward it), by the view's tilt
        slope = float(np.copysign(np.sqrt(max(0.0, 1 - frame.fs**2)), frame.k))
        slope /= max(frame.fs, MIN_FORESHORTENING)
        return replace(
            optics,
            defocus=optics.defocus - self.focus_depth(stage_position, projection),
            tilt=(slope * np.sin(frame.theta), -slope * np.cos(frame.theta)),
        )

    def focus_offset(self, beam_type: BeamType) -> float:
        """How far the beam's focus is from its eucentric height (m): drawn
        once per beam from the scene's seed, within `focus_fuzz`."""
        if self.focus_fuzz <= 0:
            return 0.0
        rng = np.random.default_rng(
            [self.seed, zlib.crc32(beam_type.name.lower().encode()), 77]
        )
        return float(rng.uniform(-self.focus_fuzz, self.focus_fuzz))

    def _turn(self, stage_position: FibsemStagePosition) -> float:
        """-1 when the stage is half a turn from the anchor, else 1.

        The sample turns with the stage. The projection already reflects the
        anchor at a half turn (`_compucentric_corrected`, by the same test);
        this turns the world about it, so the view shows what a move by
        `target_position` puts under it."""
        r, r0 = stage_position.r, self.reference_position.r
        if r is None or r0 is None:
            return 1.0
        return -1.0 if np.isclose(abs(np.rad2deg(r0 - r)), 180, atol=2) else 1.0

    def _non_eucentric_reference(
        self, stage_position: FibsemStagePosition, projection: BeamStageProjection
    ) -> FibsemStagePosition:
        """The world anchor as a non-eucentric stage actually carries it.

        The projection tilts the world about the point the stage reports - a
        eucentric stage. On a real stage the surface sits `tilt_axis_offset`
        above the tilt axis, so a tilt change swings it about the axis. The
        textbook eucentric-height model, anchored at the SEM orientation
        (the apex, where the offset vector is vertical): a point h above the
        axis walks ALONG THE SURFACE (the same in both views - only which
        feature is centred changes) and changes height, which is what costs
        coincidence. See tilt_swing for the model.

        The walk follows the stable-move direction - on a pre-tilted shuttle
        that is not the stage y axis but (cos p, -sin p) in stage y/z, with p
        the corrected pre-tilt; along the stage y axis alone it would leave
        the surface plane and read as a height error nothing can correct.
        The sag goes on the stage z axis exactly as the boot
        `coincidence_offset` does, so the same correction path restores it.
        """
        from fibsem.alignment.coincidence import tilt_swing
        from fibsem.geometry.frames import StageModel

        pretilt = StageModel.from_geometry(projection.geometry).surface_slope(
            stage_position.r or 0.0
        )
        # the apex - where the offset vector is vertical - is the SEM
        # orientation: tilt = shuttle pre-tilt (0 on a compustage)
        apex = np.deg2rad(projection.geometry.shuttle_pre_tilt)
        dy, dz = tilt_swing(
            self.tilt_axis_offset,
            self.reference_position.t or 0.0,
            stage_position.t or 0.0,
            apex,
            pretilt,
        )
        reference = deepcopy(self.reference_position)
        reference.y = (reference.y or 0.0) + dy
        reference.z = (reference.z or 0.0) + dz
        return reference

    def feature_index(self) -> FeatureIndex:
        """The scene's feature index, rebuilt when the feature list is
        replaced or grows (a test swapping in its own features)."""
        current = self._index
        if current is None or not current.is_for(self):
            current = self._index = FeatureIndex(self)
        return current

    def _stamp_feature(
        self,
        targets,
        f: SceneFeature,
        u: float,
        v: float,
        pixel_size: float,
        fs: float,
        scan_rotation: float,
        intensity: Optional[float] = None,
    ) -> None:
        """Stamp one feature at view position (u, v): its ellipse in pixels,
        foreshortened along the view's y, turned with the scan. `targets` is
        either one canvas (with `intensity`) or a list of (canvas,
        intensity) pairs drawn with one profile."""
        if intensity is not None:
            targets = [(targets, intensity)]
        sigma_x = f.sigma / pixel_size
        self._stamp_blob(
            targets,
            u,
            v,
            sigma_x,
            sigma_x * f.eccentricity * fs,
            f.sharpness,
            angle=f.angle + scan_rotation,
            wobble=f.wobble,
            wobble_phase=f.wobble_phase,
            opacity=f.opacity if intensity is None else 0.0,
        )

    @staticmethod
    def _stamp_blob(
        targets,
        u: float,
        v: float,
        sigma_x: float,
        sigma_y: float,
        sharpness: float = 1.0,
        angle: float = 0.0,
        wobble: float = 0.0,
        wobble_phase: float = 0.0,
        opacity: float = 0.0,
    ) -> None:
        """Add one supergaussian blob, clipped to its bounding box.

        sharpness 1 = gaussian; higher = flat top with a sharp rim. `angle`
        turns the (sigma_x, sigma_y) ellipse; `wobble` modulates its radius
        with a few harmonics of the polar angle for an irregular outline.
        `opacity` composites instead of adding: inside the outline the
        background (film, holes, bars) is hidden by that fraction. `targets`
        are (canvas, intensity) pairs sharing the one profile.
        """
        height, width = targets[0][0].shape
        # a flat-top blob (sharpness >= 2) is already ~0 by 2 sigma; only a
        # plain gaussian needs the 4 sigma box - and a box 4x smaller is
        # what makes a field of 30 um cells affordable
        tails = 4.0 if sharpness < 2 else 2.2
        reach = tails * max(sigma_x, sigma_y) * (1 + wobble)
        if u + reach < 0 or u - reach > width or v + reach < 0 or v - reach > height:
            return
        x0 = max(0, int(u - reach))
        x1 = min(width, int(u + reach) + 1)
        y0 = max(0, int(v - reach))
        y1 = min(height, int(v + reach) + 1)
        if x0 >= x1 or y0 >= y1:
            return
        xs = np.arange(x0, x1, dtype=np.float32)
        ys = np.arange(y0, y1, dtype=np.float32)
        dx = xs[None, :] - u
        dy = ys[:, None] - v
        if angle:
            c, s_ = np.cos(angle), np.sin(angle)
            dx, dy = dx * c + dy * s_, -dx * s_ + dy * c
        rx = dx / max(sigma_x, 0.5)
        ry = dy / max(sigma_y, 0.5)
        r2 = rx**2 + ry**2
        if wobble:
            theta = np.arctan2(ry, rx)
            mod = 1.0 + wobble * (
                0.6 * np.sin(3 * theta + wobble_phase)
                + 0.4 * np.sin(5 * theta + 2 * wobble_phase)
            )
            r2 = r2 / mod**2
        profile = np.exp(-0.5 * r2**sharpness)
        if opacity > 0:
            # the cover is flat out to the outline and drops sharply there,
            # whatever the intensity profile does - a body hides the film
            # under all of itself, not only under its brightest part
            hide = opacity * np.exp(-0.5 * r2 ** max(sharpness, 6.0))
            for canvas, intensity in targets:
                region = canvas[y0:y1, x0:x1]
                canvas[y0:y1, x0:x1] = region * (1 - hide) + intensity * profile
        else:
            for canvas, intensity in targets:
                canvas[y0:y1, x0:x1] += intensity * profile
