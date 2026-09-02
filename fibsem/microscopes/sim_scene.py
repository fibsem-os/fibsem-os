"""Shared-scene projection rendering for the simulated microscope (FIB-874).

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

Opt-in via the simulator config (`sim: coincidence_projection: true`),
defaulting off - the file-sequence and text-card modes remain the default
for workflow/UI simulation.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from fibsem.projection import BeamStageProjection, surface_foreshortening
from fibsem.structures import BeamType, FibsemStagePosition

# Keep the render well-defined at any pose: below this foreshortening the
# view is a degenerate grazing projection (features smear to infinity).
MIN_FORESHORTENING = 0.035  # ~ sin(2 deg)

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
class CoincidenceScene:
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
            for _ in range(int(rng.integers(3, 9))):
                self.features.append(
                    SceneFeature(
                        x=float(cx + rng.normal(0, self.cluster_spread)),
                        y=float(cy + rng.normal(0, self.cluster_spread)),
                        sigma=float(rng.uniform(4.5e-6, 12.0e-6)),
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
