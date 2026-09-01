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
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from fibsem.structures import BeamType, FibsemStagePosition
from fibsem.transformations import convert_stage_tilt_to_milling_angle

# Keep the projection well-defined at any stage tilt: outside this range the
# flat-surface foreshortening model degenerates (grazing or edge-on views).
MIN_GLANCING_ANGLE = np.deg2rad(2.0)

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
    # reference stage (y, z) captured on first render; the height error is
    # measured relative to this, in the stage-carried surface frame
    reference_stage_yz: Optional[Tuple[float, float]] = None
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

    def render(
        self,
        beam_type: BeamType,
        stage_position: FibsemStagePosition,
        hfw: float,
        resolution: Tuple[int, int],
        pretilt: float,
        column_tilt: float,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Render one beam's view of the scene at the current stage position.

        Args:
            beam_type: ELECTRON (top-down) or ION (foreshortened + displaced).
            stage_position: current stage position; x/y centre the view, z
                drives the coincidence displacement, t the foreshortening.
            hfw: horizontal field width in metres.
            resolution: (width, height) in pixels.
            pretilt: shuttle pre-tilt in RADIANS.
            column_tilt: ion column tilt in RADIANS.
            rng: noise source; a fresh default generator when omitted.

        Returns:
            uint8 image of shape (height, width).
        """
        width, height = int(resolution[0]), int(resolution[1])
        pixel_size = hfw / width
        cx, cy = width / 2, height / 2
        sx = stage_position.x or 0.0
        sy = stage_position.y or 0.0
        sz = stage_position.z or 0.0
        tilt = stage_position.t or 0.0

        if self.reference_stage_yz is None:
            self.reference_stage_yz = (sy, sz)
            logging.info(
                {
                    "msg": "coincidence_scene_init",
                    "reference_stage_yz": self.reference_stage_yz,
                    "coincidence_offset": self.coincidence_offset,
                }
            )

        # Stage y/z axes tilt with the stage, and stable moves follow the
        # pre-tilted sample surface (dz = -dy*tan(pretilt)), so:
        # - the view centre in the sample plane is the chamber-lateral
        #   position, y*cos(t) - z*sin(t)  (verified against stable_move:
        #   a requested image dy lands exactly dy in this frame)
        # - the height error is measured against the stage-carried surface,
        #   so surface-following moves leave it unchanged and z-only moves
        #   (vertical_move) change it one-to-one
        y0, z0 = self.reference_stage_yz
        height_error = (sz - z0) + (sy - y0) * np.tan(pretilt) + self.coincidence_offset
        vy = sy * np.cos(tilt) - sz * np.sin(tilt)
        vy0 = y0 * np.cos(tilt) - z0 * np.sin(tilt)
        view_y = vy - vy0

        if beam_type is BeamType.ION:
            milling_angle = convert_stage_tilt_to_milling_angle(
                stage_tilt=tilt, pretilt=pretilt, column_tilt=column_tilt
            )
            milling_angle = float(
                np.clip(
                    milling_angle, MIN_GLANCING_ANGLE, column_tilt - MIN_GLANCING_ANGLE
                )
            )
            stretch = np.sin(column_tilt - milling_angle) / np.sin(milling_angle)
            dy_px = height_error * np.sin(column_tilt) / pixel_size
        else:
            stretch = 1.0
            dy_px = 0.0

        canvas = np.zeros((height, width), dtype=np.float32)

        # grid mesh bars: computed in world coordinates per pixel row/column,
        # through the same view transform as the features (hole at the origin)
        xs_world = (np.arange(width, dtype=np.float32) - cx) * pixel_size + sx
        v_sem_rows = (np.arange(height, dtype=np.float32) - cy - dy_px) * stretch + cy
        ys_world = (v_sem_rows - cy) * pixel_size + view_y

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
            u = (f.x - sx) / pixel_size + cx
            v_sem = cy + (f.y - view_y) / pixel_size
            v = cy + (v_sem - cy) / stretch + dy_px
            sigma_x = f.sigma / pixel_size
            sigma_y = sigma_x / stretch
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
