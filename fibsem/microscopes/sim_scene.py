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
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
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
    sigma: float  # m
    intensity: float
    sharpness: float = 1.0


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
    n_clusters: int = 35  # cell clusters scattered over the extent
    cells_per_cluster: Tuple[int, int] = (3, 8)  # inclusive range
    cell_size: Tuple[float, float] = (4.5e-6, 12.0e-6)  # m, sigma range per cell
    cluster_spread: float = 15e-6  # m, how far blobs scatter around a cluster
    extent: float = 400e-6  # m, features are scattered over +/- extent/2
    grid_pitch: float = 125e-6  # m, mesh pitch (~200 mesh)
    grid_bar_width: float = 35e-6  # m
    grid_intensity: float = 90.0
    noise_sigma: float = 12.0  # gaussian noise layer over the final image
    # blend of full-range uniform noise, like the simulator's default
    # random-noise images (0 = none, 1 = pure noise)
    noise_fraction: float = 0.15
    # grids never load perfectly straight; drawn from +/- this range per seed
    grid_rotation: Optional[float] = None  # rad; random when None
    grid_rotation_range: float = np.deg2rad(45.0)
    # how far above the stage tilt axis the sample surface sits (m, along the
    # stage z axis). 0 is a eucentric stage; a real stage is not, and a tilt
    # change then swings the surface about the axis, costing coincidence
    tilt_axis_offset: float = 0.0
    # the coincident stage position, captured on first render (current
    # position with z offset by coincidence_offset); every view is rendered
    # as the beam projection of the scene relative to this reference
    reference_position: Optional[FibsemStagePosition] = None
    features: List[SceneFeature] = field(default_factory=list)
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
        for _ in range(self.n_clusters):
            cx, cy = rng.uniform(-half, half, size=2)
            lo, hi = self.cells_per_cluster
            for _ in range(int(rng.integers(lo, hi + 1))):
                self.features.append(
                    SceneFeature(
                        x=float(cx + rng.normal(0, self.cluster_spread)),
                        y=float(cy + rng.normal(0, self.cluster_spread)),
                        sigma=float(rng.uniform(*self.cell_size)),
                        intensity=float(rng.uniform(40, 110)),
                        sharpness=3.0,
                    )
                )
        # fiducial-like cross at the world origin: a dense line of small
        # blobs along each arm, so it goes through the same projection as
        # every other feature
        n_arm = int(2 * FIDUCIAL_ARM_LENGTH / FIDUCIAL_POINT_SPACING) + 1
        for offset in np.linspace(-FIDUCIAL_ARM_LENGTH, FIDUCIAL_ARM_LENGTH, n_arm):
            for x, y in (
                (float(offset), float(offset)),
                (float(offset), -float(offset)),
            ):
                self.features.append(
                    SceneFeature(x=x, y=y, sigma=0.4e-6, intensity=220.0)
                )

    # the configuration keys `sim: sample:` accepts, with their units
    CONFIG_KEYS = (
        "coincidence_offset",  # m
        "tilt_axis_offset",  # m
        "seed",
        "n_clusters",
        "cells_per_cluster",  # [min, max]
        "cell_size",  # [min, max] m
        "cluster_spread",  # m
        "extent",  # m
        "grid_pitch",  # m
        "grid_bar_width",  # m
        "grid_intensity",
        "noise_sigma",
        "noise_fraction",
        "grid_rotation",  # degrees; null = random within grid_rotation_range
        "grid_rotation_range",  # degrees
    )

    @classmethod
    def from_config(cls, config: dict) -> "SampleScene":
        """Build a scene from the `sim: sample:` block (unknown keys rejected,
        angles in degrees, ranges as two-element lists)."""
        unknown = set(config) - set(cls.CONFIG_KEYS) - {"enabled"}
        if unknown:
            raise ValueError(f"Unknown sim.sample keys: {sorted(unknown)}")
        kwargs = {k: v for k, v in config.items() if k in cls.CONFIG_KEYS}
        for key in ("cells_per_cluster", "cell_size"):
            if key in kwargs:
                kwargs[key] = tuple(kwargs[key])
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
        flip = -1.0 if np.isclose(projection.scan_rotation, np.pi) else 1.0

        canvas = np.zeros((height, width), dtype=np.float32)

        # grid mesh bars: world (sample-plane) coordinates per pixel, through
        # the inverse of the same mapping the features use
        xs_world = flip * ((np.arange(width, dtype=np.float32) - cx) * pixel_size - ax)
        ys_world = (
            flip * ((np.arange(height, dtype=np.float32) - cy) * pixel_size - ay) / fs
        )

        def _near_bar(w: np.ndarray) -> np.ndarray:
            return (
                np.abs((w % self.grid_pitch) - self.grid_pitch / 2)
                < self.grid_bar_width / 2
            )

        # the mesh is rotated in the world plane, so the bar test runs on
        # rotated world coordinates (full 2D, no longer separable)
        cos_r, sin_r = np.cos(self.grid_rotation), np.sin(self.grid_rotation)
        xw = xs_world[None, :]
        yw = ys_world[:, None]
        x_rot = xw * cos_r + yw * sin_r
        y_rot = -xw * sin_r + yw * cos_r
        bar_mask = _near_bar(x_rot) | _near_bar(y_rot)
        canvas[bar_mask] += self.grid_intensity

        for f in self.features:
            u = cx + (flip * f.x + ax) / pixel_size
            v = cy + (flip * f.y * fs + ay) / pixel_size
            sigma_x = f.sigma / pixel_size
            sigma_y = sigma_x * fs
            self._stamp_blob(canvas, u, v, sigma_x, sigma_y, f.intensity, f.sharpness)

        if rng is None:
            rng = np.random.default_rng()
        # soft-compress so overlapping cells don't saturate into flat,
        # texture-free regions (hard clipping kills correlatable structure)
        canvas = 180.0 * np.tanh(canvas / 180.0)
        if beam_type is BeamType.ION:
            data = 170.0 - 0.6 * canvas
        else:
            data = 60.0 + canvas
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

        bars, cells, red, fiducial = weights
        canvas = np.zeros((height, width), dtype=np.float32)

        if bars > 0:
            xs_world = (np.arange(width, dtype=np.float32) - cx) * pixel_size - ax
            ys_world = (
                (np.arange(height, dtype=np.float32) - cy) * pixel_size - ay
            ) / fs
            cos_r, sin_r = np.cos(self.grid_rotation), np.sin(self.grid_rotation)
            xw = xs_world[None, :]
            yw = ys_world[:, None]
            x_rot = xw * cos_r + yw * sin_r
            y_rot = -xw * sin_r + yw * cos_r
            half = self.grid_bar_width / 2
            bar_mask = (
                np.abs((x_rot % self.grid_pitch) - self.grid_pitch / 2) < half
            ) | (np.abs((y_rot % self.grid_pitch) - self.grid_pitch / 2) < half)
            canvas[bar_mask] += bars * self.grid_intensity

        red_rng = np.random.default_rng(self.seed + 1)
        for f in self.features:
            is_fiducial = f.sharpness == 1.0 and f.sigma < 1e-6
            if is_fiducial:
                weight = fiducial
            else:
                in_subset = red_rng.random() < self.red_fraction
                weight = cells + (red if in_subset else 0.0)
            if weight <= 0:
                continue
            u = cx + (f.x + ax) / pixel_size
            v = cy + (f.y * fs + ay) / pixel_size
            sigma_x = f.sigma / pixel_size
            self._stamp_blob(
                canvas, u, v, sigma_x, sigma_x * fs, weight * f.intensity, f.sharpness
            )

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

    @staticmethod
    def _stamp_blob(
        canvas: np.ndarray,
        u: float,
        v: float,
        sigma_x: float,
        sigma_y: float,
        intensity: float,
        sharpness: float = 1.0,
    ) -> None:
        """Add one supergaussian blob, clipped to its bounding box.

        sharpness 1 = gaussian; higher = flat top with a sharp rim.
        """
        height, width = canvas.shape
        x0 = max(0, int(u - 4 * sigma_x))
        x1 = min(width, int(u + 4 * sigma_x) + 1)
        y0 = max(0, int(v - 4 * sigma_y))
        y1 = min(height, int(v + 4 * sigma_y) + 1)
        if x0 >= x1 or y0 >= y1:
            return
        xs = np.arange(x0, x1, dtype=np.float32)
        ys = np.arange(y0, y1, dtype=np.float32)
        rx = (xs[None, :] - u) / max(sigma_x, 0.5)
        ry = (ys[:, None] - v) / max(sigma_y, 0.5)
        r2 = rx**2 + ry**2
        canvas[y0:y1, x0:x1] += intensity * np.exp(-0.5 * r2**sharpness)
