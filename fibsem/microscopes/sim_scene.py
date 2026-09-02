"""A synthetic sample the simulated microscope images through its projections (FIB-874).

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
import zlib
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import binary_dilation as ndi_binary_dilation
from scipy.ndimage import gaussian_filter as ndi_gaussian

from fibsem.projection import (
    BeamStageProjection,
    FMStageProjection,
    surface_foreshortening,
)
from fibsem.structures import BeamType, FibsemStagePosition

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
# 561 the subset, 635 nothing on this sample
FIDUCIAL_DYE = Fluorophore("dapi-like", 365.0, 460.0, excitation_width=30.0)
CELL_DYE = Fluorophore("gfp-like", 470.0, 510.0, excitation_width=30.0)
SUBSET_DYE = Fluorophore("mcherry-like", 560.0, 610.0, excitation_width=30.0)
# in reflection the bars dominate, the fiducial reflects, cells are faint
REFLECTION_WEIGHTS = (1.0, 0.25, 0.0, 0.6)
# the FM's fluorescence intensity for a fully-dyed part, on the beam
# intensity scale the stamper uses
FM_INTENSITY = 90.0
# a faint outline of the bars leaks into every fluorescence channel
BARS_IN_FLUORESCENCE = 0.05


def fm_channel_weights(emission, excitation=None) -> Tuple[float, float, float, float]:
    """(bars, cells, red subset, fiducial) intensity weights for a channel.

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
    )


def _lattice_draw(i: np.ndarray, j: np.ndarray, seed: int) -> np.ndarray:
    """A deterministic pseudo-random number in [0, 1) per integer lattice
    cell - a hash, so a square or a hole draws the same value from every
    view and every session with the same seed."""
    h = (i * 73856093) ^ (j * 19349663) ^ (seed * 83492791)
    return ((h * 2654435761) & 0xFFFFFFFF) / 2**32


# what the beams see of the support, on the canvas scale
HOLE_DEPTH = 45.0  # film absent inside a hole
HOLE_RIM = 25.0  # the bright rim around a hole
RIP_DEPTH = 80.0  # film torn away
RIM_INTENSITY = 120.0  # the grid's metal rim
BEYOND_INTENSITY = -60.0  # the holder, beyond the grid

FIDUCIAL_ARM_LENGTH = 8e-6  # m, half-length of each fiducial cross arm
FIDUCIAL_POINT_SPACING = 0.5e-6  # m


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
    part: str = "body"  # "body" | "nucleus" | "organelle" | "bud"
    cell_id: int = -1
    # how much the feature hides what is under it (film, holes, bars): 0 is
    # additive (a mound on top), 1 replaces the background inside its
    # outline - a cell body or an ice plate is opaque, a nucleus sits on top
    opacity: float = 0.0


@dataclass
class MilledRegion:
    """A pattern that was milled: a convex polygon on the sample surface,
    in world coordinates (metres), that every later view shows as a trench
    (FIB-877). Kept as corners rather than a shape description because the
    view-to-world mapping at the milling pose is a foreshortening: a
    rectangle rotated in the FIB image is not a rectangle of any rotation
    on the surface."""

    points: np.ndarray  # (N, 2) world coordinates, in order round the outline
    depth: float = 1.0  # m, recorded only


MILL_DEPTH = 90.0  # how dark a trench reads in the FM reflection (canvas scale)
MILL_FLOOR = 0.25  # the fraction of the local beam intensity a trench keeps


@dataclass
class SampleScene:
    """A synthetic cryo-grid scene rendered per-beam by projection.

    The sample is a regular grid mesh (bars at a known pitch, a hole centred
    on the world origin) with randomly clustered cell-like blobs scattered
    over the film, plus a fiducial-like cross at the origin. The mesh is
    deliberately periodic: it reproduces the rival-correlation-peak trap that
    real grid bars create (FIB-711), so window constraints are testable.
    """

    coincidence_offset: float = 10e-6  # m, initial height error at boot
    seed: int = 24
    # what grows on the film: "mammalian" (adherent, fried-egg: a wide flat
    # body with an off-centre nuclear mound and organelle speckle, sparse),
    # "yeast" (compact bright ovoids in clusters, some budding), "bacteria"
    # (dense small rods), "mixed", or "none" for bare film
    cell_type: str = "mammalian"
    # yeast: clusters over the extent
    n_clusters: int = 35  # cell clusters scattered over the extent
    cells_per_cluster: Tuple[int, int] = (3, 8)  # inclusive range
    cell_size: Tuple[float, float] = (4.5e-6, 12.0e-6)  # m, sigma range per cell
    cluster_spread: float = 15e-6  # m, how far blobs scatter around a cluster
    # mammalian: count per 150 x 150 um, body and nucleus radii
    mammalian_density: float = 7.0
    mammalian_radius: Tuple[float, float] = (18e-6, 30e-6)  # m
    nucleus_radius: Tuple[float, float] = (5e-6, 7e-6)  # m
    # bacteria: count per 150 x 150 um, rod length
    bacteria_density: float = 160.0
    bacteria_length: Tuple[float, float] = (2.0e-6, 3.5e-6)  # m
    extent: float = 800e-6  # m, features are scattered over +/- extent/2
    grid_pitch: float = 125e-6  # m, mesh pitch (~200 mesh)
    grid_bar_width: float = 35e-6  # m
    grid_intensity: float = 90.0
    # the support film in each square: "continuous", or "holey" - a
    # Quantifoil-style lattice of round holes (diameter, centre-to-centre
    # pitch), aligned with the grid, a seeded few of them broken/merged.
    # Continuous by default: the lattice is a 4 um periodic pattern the
    # fine-pass correlator aliases on (false refusals at the fiducial, and a
    # false convergence at 80 um on the simulator), which the measurement
    # does not handle yet - see the xfail in test_sim_scene_realism.py
    film: str = "continuous"
    hole_diameter: float = 2.0e-6  # m  (R2/2: 2 um holes, 2 um apart)
    hole_pitch: float = 4.0e-6  # m
    broken_hole_fraction: float = 0.02
    # a seeded fraction of squares with the film torn away (a rip)
    rip_fraction: float = 0.03
    # ice crystals: flat bright plates with a bright rim, per 100 x 100 um
    ice_density: float = 0.4
    ice_size: Tuple[float, float] = (6e-6, 16e-6)  # m, plate radius
    # the fiducial-like cross at the world origin: handy for eyeballing
    # navigation, absent on a real grid - off for realistic imaging
    fiducial: bool = True
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
    # fraction of cells that also carry the red fluorophore (seeded subset)
    red_fraction: float = 0.4
    # defocus model for the FM: blur sigma grows by this many pixels per
    # micron the objective sits away from its focus position
    fm_blur_px_per_um: float = 0.6

    def __post_init__(self) -> None:
        if self.features:
            return
        rng = np.random.default_rng(self.seed)
        if self.grid_rotation is None:
            self.grid_rotation = float(
                rng.uniform(-self.grid_rotation_range, self.grid_rotation_range)
            )
        half = self.extent / 2
        self._generate_cells(rng, half)
        # fiducial-like cross at the world origin: a dense line of small
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
        n_specks = int(self.contamination_density * (self.extent / 100e-6) ** 2)
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
        n_ice = int(self.ice_density * (self.extent / 100e-6) ** 2)
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
        reference = self.reference_position
        if self.tilt_axis_offset:
            reference = self._non_eucentric_reference(stage_position, projection)
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
        vx, vy = dx - ax, dy - ay
        return float(vx * cos_t + vy * sin_t), float((-vx * sin_t + vy * cos_t) / fs)

    def mill(
        self,
        patterns,
        beam_type: BeamType,
        stage_position: FibsemStagePosition,
        projection: BeamStageProjection,
        beam_shift: Tuple[float, float] = (0.0, 0.0),
        beam_current: Optional[float] = None,
    ) -> List[MilledRegion]:
        """Commit milling patterns to the world (FIB-877).

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

        regions: List[MilledRegion] = []
        for p in patterns:
            depth = float(getattr(p, "depth", 1e-6) or 1e-6)
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
                corners = [
                    (p.centre_x + x * c - y * s_, p.centre_y + x * s_ + y * c)
                    for x, y in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))
                ]
            points = np.array([world(x, y) for x, y in corners])
            regions.append(MilledRegion(points=points, depth=depth))
        self.milled.extend(regions)
        logging.info(
            {
                "msg": "sample_scene_milled",
                "regions": len(regions),
                "total": len(self.milled),
            }
        )
        return regions

    def milled_mask(self, xs_world: np.ndarray, ys_world: np.ndarray) -> np.ndarray:
        """Where milled regions lie, from broadcastable world coordinates:
        inside every edge's half-plane of each (convex) polygon."""
        mask = np.zeros(np.broadcast(xs_world, ys_world).shape, dtype=bool)
        for r in self.milled:
            pts = r.points
            inside = np.ones_like(mask)
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
            mask |= inside
        return mask

    def film_masks(
        self, xs_world: np.ndarray, ys_world: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """What the support is at each world point: (bars, holes, rips,
        rim, beyond) boolean masks, from broadcastable world coordinates.

        The mesh and the hole lattice are rotated together; holes exist
        only on film (not on bars), a seeded fraction of them broken into
        larger openings; a seeded fraction of squares is ripped (film gone
        inside an irregular patch). Past `grid_radius` is the metal rim,
        past that the holder.
        """
        cos_r, sin_r = np.cos(self.grid_rotation), np.sin(self.grid_rotation)
        x_rot = xs_world * cos_r + ys_world * sin_r
        y_rot = -xs_world * sin_r + ys_world * cos_r

        def _near_bar(w):
            return np.abs((w % self.grid_pitch) - self.grid_pitch / 2) < (
                self.grid_bar_width / 2
            )

        bars = _near_bar(x_rot) | _near_bar(y_rot)
        radius = np.hypot(xs_world, ys_world)
        rim = radius > self.grid_radius
        beyond = radius > self.grid_radius + self.grid_rim_width

        # a per-square seeded draw, from the square's lattice index. The
        # bars sit at half-pitch (the world origin is a square's centre), so
        # the square frame is the bar frame shifted by half a pitch
        half_pitch = self.grid_pitch / 2
        sq_i = np.floor((x_rot + half_pitch) / self.grid_pitch).astype(np.int64)
        sq_j = np.floor((y_rot + half_pitch) / self.grid_pitch).astype(np.int64)
        square_draw = _lattice_draw(sq_i, sq_j, self.seed)
        # local coordinates within the square, zero at its centre
        lx = ((x_rot + half_pitch) % self.grid_pitch) - half_pitch
        ly = ((y_rot + half_pitch) % self.grid_pitch) - half_pitch
        # the rip: an irregular patch in the ripped squares
        angle = np.arctan2(ly, lx)
        wob = 1 + 0.3 * np.sin(3 * angle + 6.0 * square_draw) + 0.2 * np.sin(5 * angle)
        patch = np.hypot(lx / (0.55 * self.grid_pitch), ly / (0.4 * self.grid_pitch))
        rips = (square_draw < self.rip_fraction) & (patch < 0.6 * wob) & ~bars

        holes = np.zeros_like(bars)
        if self.film == "holey":
            hi = np.floor(x_rot / self.hole_pitch).astype(np.int64)
            hj = np.floor(y_rot / self.hole_pitch).astype(np.int64)
            hx = (x_rot % self.hole_pitch) - self.hole_pitch / 2
            hy = (y_rot % self.hole_pitch) - self.hole_pitch / 2
            broken = _lattice_draw(hi, hj, self.seed + 7) < self.broken_hole_fraction
            r_hole = np.where(broken, 0.8 * self.hole_diameter, self.hole_diameter / 2)
            holes = (np.hypot(hx, hy) < r_hole) & ~bars
        return bars, holes, rips, rim, beyond

    def _generate_cells(self, rng: np.random.Generator, half: float) -> None:
        kinds = {
            "yeast": [("yeast", 1.0)],
            "mammalian": [("mammalian", 1.0)],
            "bacteria": [("bacteria", 1.0)],
            "mixed": [("yeast", 0.5), ("mammalian", 0.4), ("bacteria", 0.3)],
            "none": [],  # bare film: fiducial, bars and contamination only
        }
        if self.cell_type not in kinds:
            raise ValueError(
                f"cell_type must be one of {sorted(kinds)}, got {self.cell_type!r}"
            )
        fields = (self.extent / 150e-6) ** 2  # how many 150 um fields the extent is
        next_id = 0
        for kind, fraction in kinds[self.cell_type]:
            if kind == "yeast":
                next_id = self._generate_yeast(rng, half, fraction, next_id)
            elif kind == "mammalian":
                n = int(round(self.mammalian_density * fields * fraction))
                next_id = self._generate_mammalian(rng, half, n, next_id)
            else:
                n = int(round(self.bacteria_density * fields * fraction))
                next_id = self._generate_bacteria(rng, half, n, next_id)

    def _generate_yeast(self, rng, half, fraction, next_id) -> int:
        for _ in range(int(round(self.n_clusters * fraction))):
            cx, cy = rng.uniform(-half, half, size=2)
            lo, hi = self.cells_per_cluster
            for _ in range(int(rng.integers(lo, hi + 1))):
                x = float(cx + rng.normal(0, self.cluster_spread))
                y = float(cy + rng.normal(0, self.cluster_spread))
                sigma = float(rng.uniform(*self.cell_size))
                angle = float(rng.uniform(0, np.pi))
                self.features.append(
                    SceneFeature(
                        x=x,
                        y=y,
                        sigma=sigma,
                        intensity=float(rng.uniform(40, 110)),
                        sharpness=3.0,
                        eccentricity=float(rng.uniform(0.8, 1.0)),
                        angle=angle,
                        cell_id=next_id,
                        opacity=1.0,
                    )
                )
                # a compact nucleus for the FM (the beams barely see it)
                self.features.append(
                    SceneFeature(
                        x=x,
                        y=y,
                        sigma=sigma * 0.35,
                        intensity=8.0,
                        sharpness=2.0,
                        part="nucleus",
                        cell_id=next_id,
                    )
                )
                if rng.random() < 0.3:  # budding daughter
                    t = rng.uniform(0, 2 * np.pi)
                    self.features.append(
                        SceneFeature(
                            x=x + 1.1 * sigma * np.cos(t),
                            y=y + 1.1 * sigma * np.sin(t),
                            sigma=sigma * 0.55,
                            intensity=70.0,
                            sharpness=3.0,
                            part="bud",
                            cell_id=next_id,
                            opacity=1.0,
                        )
                    )
                next_id += 1
        return next_id

    def _generate_mammalian(self, rng, half, n, next_id) -> int:
        # adherent cells tile the film rather than cluster: keep them apart
        placed: List[Tuple[float, float]] = []
        min_distance = 1.6 * self.mammalian_radius[1]
        for _ in range(60 * max(n, 1)):
            if len(placed) >= n:
                break
            x, y = rng.uniform(-half, half, size=2)
            if all(np.hypot(x - px_, y - py_) > min_distance for px_, py_ in placed):
                placed.append((float(x), float(y)))
        for x, y in placed:
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
        "film",  # continuous | holey
        "hole_diameter",  # m
        "hole_pitch",  # m
        "broken_hole_fraction",
        "rip_fraction",  # of squares
        "ice_density",  # per 100 x 100 um
        "ice_size",  # [min, max] m
        "grid_radius",  # m
        "grid_rim_width",  # m
    )

    @classmethod
    def from_config(cls, config: dict) -> "SampleScene":
        """Build a scene from the `sim: sample:` block (unknown keys rejected,
        angles in degrees, ranges as two-element lists)."""
        unknown = set(config) - set(cls.CONFIG_KEYS) - {"enabled"}
        if unknown:
            raise ValueError(f"Unknown sim.sample keys: {sorted(unknown)}")
        kwargs = {k: v for k, v in config.items() if k in cls.CONFIG_KEYS}
        for key in (
            "cells_per_cluster",
            "cell_size",
            "contamination_size",
            "mammalian_radius",
            "nucleus_radius",
            "bacteria_length",
            "ice_size",
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
        logging.info(
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

        Returns:
            uint8 image of shape (height, width).
        """
        width, height = int(resolution[0]), int(resolution[1])
        pixel_size = hfw / width
        cx, cy = width / 2, height / 2

        if self.reference_position is None:
            self.anchor(stage_position)

        # where the world anchor appears in the CURRENT view. Asked this way
        # round (anchor projected into the current pose, not the current
        # position into the anchor's pose) the trig is evaluated at the
        # current tilt/rotation - so the world persists through tilt changes
        # exactly the way the app draws saved positions on views at other
        # poses, instead of resetting. The offset carries the lateral
        # travel, the per-beam projection of the height error (near zero in
        # the SEM view, the view tilt's worth in the FIB view), and the
        # scan-rotation/manufacturer conventions
        reference = self.reference_position
        if self.tilt_axis_offset:
            reference = self._non_eucentric_reference(stage_position, projection)
        ax, ay = projection.to_plane(reference, stage_position)

        fs = max(surface_foreshortening(projection, stage_position), MIN_FORESHORTENING)
        # scan rotation turns the raster, so the content turns with it - the
        # full angle here, not only the half-turn the app's projection models,
        # so an intermediate rotation shows the app's limitation honestly
        theta = float(projection.scan_rotation or 0.0)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        # the projection already flips the anchor offset at a half-turn;
        # undo that and apply the full rotation here so every angle works
        if np.isclose(theta, np.pi):
            ax, ay = -ax, -ay
        ax, ay = ax * cos_t - ay * sin_t, ax * sin_t + ay * cos_t
        # a beam shift moves the content in the view, in the scan frame
        offset = self.beam_offset.get(beam_type.name.lower(), (0.0, 0.0))
        current_offset = self.current_offset(beam_type, beam_current)
        ax = ax + beam_shift[0] + offset[0] + current_offset[0]
        ay = ay - (beam_shift[1] + offset[1] + current_offset[1])

        canvas = np.zeros((height, width), dtype=np.float32)

        # grid mesh bars: world (sample-plane) coordinates per pixel, through
        # the inverse of the same mapping the features use: un-shift,
        # un-rotate the scan, un-foreshorten
        vx = (np.arange(width, dtype=np.float32) - cx)[None, :] * pixel_size - ax
        vy = (np.arange(height, dtype=np.float32) - cy)[:, None] * pixel_size - ay
        xs_world = vx * cos_t + vy * sin_t
        ys_world = (-vx * sin_t + vy * cos_t) / fs

        bars, holes, rips, rim, beyond = self.film_masks(xs_world, ys_world)
        canvas[bars] += self.grid_intensity
        # holes: film absent, with a bright rim one pixel-ish wide
        canvas[holes] -= HOLE_DEPTH
        hole_rim = ndi_binary_dilation(holes, iterations=2) & ~holes & ~bars
        canvas[hole_rim] += HOLE_RIM

        for f in self.features:
            # project (foreshorten), rotate with the scan, then shift
            px_, py_ = f.x, f.y * fs
            u = cx + (px_ * cos_t - py_ * sin_t + ax) / pixel_size
            v = cy + (px_ * sin_t + py_ * cos_t + ay) / pixel_size
            self._stamp_feature(canvas, f, u, v, pixel_size, fs, theta)

        # what is torn or off-grid overrides whatever was drawn there
        canvas[rips] = -RIP_DEPTH
        canvas[rim] = RIM_INTENSITY
        canvas[beyond] = BEYOND_INTENSITY

        if rng is None:
            rng = np.random.default_rng()
        # soft-compress so overlapping cells don't saturate into flat,
        # texture-free regions (hard clipping kills correlatable structure)
        canvas = 180.0 * np.tanh(canvas / 180.0)
        if beam_type is BeamType.ION:
            data = 170.0 - 0.6 * canvas
        else:
            data = 60.0 + canvas
        # a trench is dark in BOTH beams - it is not surface contrast but a
        # hole in the sample - so it goes on after the per-beam mapping
        if self.milled:
            trench = self.milled_mask(xs_world, ys_world)
            data = np.where(trench, data * MILL_FLOOR, data)
        data += rng.normal(0, self.noise_sigma, canvas.shape)
        if self.noise_fraction > 0:
            uniform = rng.uniform(0, 255, canvas.shape)
            data = (1 - self.noise_fraction) * data + self.noise_fraction * uniform
        return np.clip(data, 0, 255).astype(np.uint8)

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

        reference = self.reference_position
        if self.tilt_axis_offset:
            reference = self._non_eucentric_reference(stage_position, projection)
        ax, ay = projection.to_plane(reference, stage_position)
        fs = max(surface_foreshortening(projection, stage_position), MIN_FORESHORTENING)

        bars, cells, red, fiducial = weights[:4]
        # contamination reflects strongly and autofluoresces faintly
        contamination = 0.9 if bars >= 1.0 else 0.12
        canvas = np.zeros((height, width), dtype=np.float32)

        xs_world = (np.arange(width, dtype=np.float32) - cx)[None, :] * pixel_size - ax
        ys_world = (
            (np.arange(height, dtype=np.float32) - cy)[:, None] * pixel_size - ay
        ) / fs
        bar_mask, holes, rips, rim, beyond = self.film_masks(xs_world, ys_world)
        if bars > 0:
            canvas[bar_mask] += bars * self.grid_intensity
            # in reflection the holes and rips read dark, the rim bright
            canvas[holes] -= bars * HOLE_DEPTH
            canvas[rips] -= bars * RIP_DEPTH
            if self.milled:
                canvas[self.milled_mask(xs_world, ys_world)] = -bars * MILL_DEPTH
        # ice reflects strongly and barely fluoresces
        ice = 0.9 if bars >= 1.0 else 0.04

        subset_rng = np.random.default_rng(self.seed + 1)
        in_subset: Dict[int, bool] = {}
        for f in self.features:
            if f.kind == "fiducial":
                weight = fiducial * f.intensity
            elif f.kind == "contamination":
                weight = contamination * f.intensity
            elif f.kind == "ice":
                weight = ice * f.intensity
            else:
                if f.cell_id not in in_subset:
                    in_subset[f.cell_id] = subset_rng.random() < self.red_fraction
                # dyes by part: the DNA dye lives in the nucleus, the
                # cytoplasmic one everywhere; the subset dye follows the cell
                body_dye = cells + (red if in_subset[f.cell_id] else 0.0)
                if f.part == "nucleus":
                    weight = FM_INTENSITY * (fiducial * 1.0 + body_dye * 0.5)
                elif f.part == "organelle":
                    weight = FM_INTENSITY * body_dye * 0.6
                else:
                    weight = FM_INTENSITY * body_dye
            if weight <= 0:
                continue
            u = cx + (f.x + ax) / pixel_size
            v = cy + (f.y * fs + ay) / pixel_size
            self._stamp_feature(canvas, f, u, v, pixel_size, fs, 0.0, intensity=weight)
        canvas[rim] = RIM_INTENSITY if bars >= 1.0 else 0.0
        canvas[beyond] = 0.0

        if defocus:
            # a retracted objective is millimetres from focus: cap the blur so
            # the render stays cheap and the frame reads as "out of focus"
            sigma_px = min(abs(defocus) * 1e6 * self.fm_blur_px_per_um, 40.0)
            if sigma_px > 0.3:
                canvas = ndi_gaussian(canvas, sigma_px)

        if rng is None:
            rng = np.random.default_rng()
        # 16-bit camera: a dim background, shot-noise-like gaussian noise
        data = 400.0 + canvas * 120.0
        data += rng.normal(0, 40.0 + 0.05 * np.sqrt(np.maximum(data, 0)), canvas.shape)
        return np.clip(data, 0, 65535).astype(np.uint16)

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
        from fibsem.transformations import _projection_terms

        _, pretilt, _ = _projection_terms(
            projection.geometry, stage_position.r or 0.0, stage_position.t or 0.0
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

    def _stamp_feature(
        self,
        canvas: np.ndarray,
        f: SceneFeature,
        u: float,
        v: float,
        pixel_size: float,
        fs: float,
        scan_rotation: float,
        intensity: Optional[float] = None,
    ) -> None:
        """Stamp one feature at view position (u, v): its ellipse in pixels,
        foreshortened along the view's y, turned with the scan."""
        sigma_x = f.sigma / pixel_size
        self._stamp_blob(
            canvas,
            u,
            v,
            sigma_x,
            sigma_x * f.eccentricity * fs,
            f.intensity if intensity is None else intensity,
            f.sharpness,
            angle=f.angle + scan_rotation,
            wobble=f.wobble,
            wobble_phase=f.wobble_phase,
            opacity=f.opacity if intensity is None else 0.0,
        )

    @staticmethod
    def _stamp_blob(
        canvas: np.ndarray,
        u: float,
        v: float,
        sigma_x: float,
        sigma_y: float,
        intensity: float,
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
        background (film, holes, bars) is hidden by that fraction.
        """
        height, width = canvas.shape
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
            cover = opacity * np.exp(-0.5 * r2 ** max(sharpness, 6.0))
            region = canvas[y0:y1, x0:x1]
            canvas[y0:y1, x0:x1] = region * (1 - cover) + intensity * profile
        else:
            canvas[y0:y1, x0:x1] += intensity * profile
