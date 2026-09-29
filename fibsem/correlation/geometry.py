"""The nominal FM->FIB transform, from the geometry the images were taken under.

The correlation fit solves for a rotation between the FM volume and the FIB image
that is a property of the instrument, not of the lamella: on eight saved METEOR runs
the fitted rotations agree to about half a degree, and on the Arctis the fitted tilt
is 90 degrees minus the milling angle to a fraction of a degree. With FIB-milled
fiducials -- all on one plane -- the fit cannot tell that rotation from its mirror
image, and the sign it picks decides which way the refractive-index depth correction
renders (FIB-880). So the rotation is built here from what the metadata already says,
and the fit is *seeded* with it (FIB-881).

Nothing geometric is derived in this module. The in-plane part of the map is the
composition of two projections the repo already trusts -- FM pixel -> stage point
(:func:`fibsem.fm.reprojection.project_image_point`) and stage point -> FIB pixel
(:class:`fibsem.projection.BeamStageProjection`) -- probed numerically, so scan
rotation, camera flips, pre-tilt and foreshortening come out of code that is verified
against the instrument, not out of a second derivation that would have to agree with
it. The depth column is a displacement along the sample normal put through the same
beam projection.

Verified against the saved fits in ``tests/correlation/test_nominal_transform.py``:
Arctis (compustage, scan rotation pi, FLIP_XY) and METEOR (Aquilos2 offset mount).
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from fibsem.fm.reprojection import project_image_point
from fibsem.projection import BeamStageProjection
from fibsem.structures import (
    BeamType,
    CameraImageTransform,
    FibsemHardwareGeometry,
    FibsemStagePosition,
    Point,
)

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from fibsem.fm.structures import FluorescenceImage
    from fibsem.structures import FibsemImage


class NominalTransformError(ValueError):
    """The images do not record enough geometry to build the nominal transform.

    Raised rather than guessed: a seed built on a guessed camera flip or stage pose
    lands the fit on the wrong branch as confidently as a good one lands it on the
    right one, and the point of seeding is to stop that being a coin flip.
    """


@dataclass(frozen=True)
class NominalTransform:
    """The FM->FIB projection the geometry predicts, in the fit's own terms.

    ``projection`` is the 2x3 matrix ``P`` with ``fib_px = P @ fm_xyz + translation``
    for FM coordinates in *isotropic* units: x and y in FM pixels, z in FM pixels of
    depth (metres / xy pixel size). In those units the rows of ``P`` are orthogonal
    with norm ``scale``, so ``P / scale`` completes to a proper rotation -- the same
    ``R`` the pyto fit stores as ``rotation_quaternion`` -- which is what a seed needs.

    ``translation`` is where FM pixel (0, 0) *on the sample surface* lands in the FIB
    image according to the stage positions alone. It carries the stage repeatability
    and the FM/FIB frame offset (tens of micrometres on the one Arctis dataset checked)
    and is informational here: the fit solves the translation from the fiducials.
    """

    projection: np.ndarray  # (2, 3) FIB px per isotropic FM unit
    translation: np.ndarray  # (2,) FIB px
    scale: float  # FIB px per FM xy px
    fm_pixel_size: float  # m, xy
    fm_pixel_size_z: float  # m, per slice
    fib_pixel_size: float  # m

    # ── derived ───────────────────────────────────────────────────────────

    @property
    def rotation(self) -> np.ndarray:
        """The 3x3 rotation whose first two rows are ``projection / scale``."""
        return _complete_rotation(self.projection / self.scale)

    @property
    def z_anisotropy(self) -> float:
        """FM slice thickness in xy pixels: what raw slice indices must be scaled by."""
        return self.fm_pixel_size_z / self.fm_pixel_size

    @property
    def dimage_dz_px_per_slice(self) -> np.ndarray:
        """Where one FM *slice* of depth lands in the FIB image, in pixels (x, y).

        This is the axis and gain the refractive-index depth correction moves along.
        """
        return self.projection[:, 2] * self.z_anisotropy

    @property
    def foreshortening(self) -> float:
        """Ratio of the in-plane singular values: sin(milling angle) on a FIB image."""
        s = np.linalg.svd(self.projection[:, :2], compute_uv=False)
        return float(s[1] / s[0]) if s[0] else 0.0

    def eulers_deg(self) -> np.ndarray:
        """Euler angles in pyto's x-convention, degrees: the form ``correlate`` seeds with."""
        from fibsem.correlation.pyto.rigid_3d import Rigid3D

        return np.degrees(Rigid3D.extract_euler(self.rotation, mode="x", ret="one"))

    def mirrored(self) -> "NominalTransform":
        """The other branch: the same in-plane map with the depth axis reversed.

        Negating the z component of both projection rows keeps them orthonormal, so the
        result completes to a proper rotation of its own. It is the solution a
        coplanar-fiducial fit cannot distinguish from this one.
        """
        mirrored = self.projection.copy()
        mirrored[:, 2] *= -1
        return dataclasses.replace(self, projection=mirrored)

    def project(self, fm_xyz_isotropic: np.ndarray) -> np.ndarray:
        """FIB pixels for (N, 3) FM points in isotropic units."""
        pts = np.asarray(fm_xyz_isotropic, dtype=float).reshape(-1, 3)
        return (self.projection @ pts.T).T + self.translation

    def angle_to(self, rotation: np.ndarray) -> float:
        """Geodesic angle in degrees between this rotation and another 3x3 one."""
        return rotation_angle_deg(self.rotation, np.asarray(rotation, dtype=float))


# ── building it ───────────────────────────────────────────────────────────


def fm_geometry_for(
    fib_geometry: FibsemHardwareGeometry,
    transform: CameraImageTransform,
    camera_tilt: Optional[float] = None,
) -> FibsemHardwareGeometry:
    """The FM's geometry record, for an FM image that did not record one.

    FM images written before 2026-08-04 carry no ``geometry``; the instrument terms
    are the FIB image's, and only the camera's two are missing. The camera tilt is
    derived the way :attr:`FluorescenceMicroscope.camera_tilt` derives it -- a half
    turn under a compustage, the ion column tilt on an offset mount -- unless given.
    The transform cannot be derived and must be supplied (from the run's log or the
    FM configuration in force at the time).
    """
    if camera_tilt is None:
        camera_tilt = (
            180.0 if fib_geometry.is_compustage else float(fib_geometry.fib_column_tilt)
        )
    return dataclasses.replace(
        fib_geometry, camera_tilt=float(camera_tilt), transform=transform
    )


def nominal_transform(
    fib_image: "FibsemImage",
    fm_image: "FluorescenceImage",
    fm_geometry: Optional[FibsemHardwareGeometry] = None,
    fm_pose: Optional[FibsemStagePosition] = None,
) -> NominalTransform:
    """Build the nominal FM->FIB transform from two images' metadata.

    Args:
        fib_image: the FIB image the fiducials are picked on. Needs
            ``metadata.hardware_geometry``, the stage position and the ion beam's scan
            rotation.
        fm_image: the FM stack. Needs xy and z pixel sizes and a stage position.
        fm_geometry: the geometry the FM image was taken under. Taken from
            ``fm_image.metadata.geometry`` when recorded; required otherwise (see
            :func:`fm_geometry_for`).
        fm_pose: the stage pose (r, t) the FM image was taken at. Taken from the FM
            metadata when it carries both, else completed from the geometry: the
            mounting fixes where the FM images (see ``_complete_fm_pose``).

    Raises:
        NominalTransformError: when any of the above is missing.
    """
    fib_md = getattr(fib_image, "metadata", None)
    fib_geometry = getattr(fib_md, "hardware_geometry", None)
    if fib_md is None or fib_geometry is None:
        raise NominalTransformError(
            "The FIB image records no hardware geometry; cannot build the nominal transform."
        )
    state = fib_md.microscope_state
    fib_pose = getattr(state, "stage_position", None)
    beam_settings = getattr(state, "ion_beam", None)
    scan_rotation = getattr(beam_settings, "scan_rotation", None)
    if fib_pose is None or scan_rotation is None:
        raise NominalTransformError(
            "The FIB image records no stage position or scan rotation; cannot build the nominal transform."
        )
    fib_pixel_size = getattr(getattr(fib_md, "pixel_size", None), "x", None)
    if not fib_pixel_size:
        raise NominalTransformError("The FIB image records no pixel size.")
    fib_shape = tuple(np.asarray(fib_image.data).shape[:2])

    fm_md = getattr(fm_image, "metadata", None)
    fm_pixel_size = getattr(fm_md, "pixel_size_x", None)
    fm_pixel_size_z = getattr(fm_md, "pixel_size_z", None)
    if not fm_pixel_size or not fm_pixel_size_z:
        raise NominalTransformError(
            "The FM image records no xy or z pixel size; cannot build the nominal transform."
        )
    if fm_geometry is None:
        fm_geometry = getattr(fm_md, "geometry", None)
    if fm_geometry is None:
        raise NominalTransformError(
            "The FM image records no geometry (camera tilt and transform) and none was given. "
            "Stacks written before 2026-08-04 need it supplied from the run's configuration."
        )
    fm_shape = tuple(np.asarray(fm_image.data).shape[-2:])
    if fm_pose is None:
        fm_pose = _complete_fm_pose(
            getattr(fm_md, "stage_position", None), fib_geometry
        )
    if fm_pose is None:
        raise NominalTransformError(
            "The FM image records no stage position, and no pose was given."
        )

    return nominal_transform_from_geometry(
        fib_geometry=fib_geometry,
        fib_pose=fib_pose,
        scan_rotation=float(scan_rotation),
        fib_pixel_size=float(fib_pixel_size),
        fib_shape=fib_shape,
        fm_geometry=fm_geometry,
        fm_pose=fm_pose,
        fm_pixel_size=float(fm_pixel_size),
        fm_pixel_size_z=float(fm_pixel_size_z),
        fm_shape=fm_shape,
        is_tescan=_is_tescan(fib_md),
    )


def nominal_transform_from_geometry(
    fib_geometry: FibsemHardwareGeometry,
    fib_pose: FibsemStagePosition,
    scan_rotation: float,
    fib_pixel_size: float,
    fib_shape: Tuple[int, int],
    fm_geometry: FibsemHardwareGeometry,
    fm_pose: FibsemStagePosition,
    fm_pixel_size: float,
    fm_pixel_size_z: float,
    fm_shape: Tuple[int, int],
    is_tescan: bool = False,
) -> NominalTransform:
    """The image-free form of :func:`nominal_transform`: every term given explicitly."""
    if is_tescan:
        logging.warning(
            "Nominal correlation transform on a Tescan geometry is unverified: "
            "the depth-axis sign was calibrated on ThermoFisher instruments only."
        )
    beam = BeamStageProjection(
        geometry=fib_geometry,
        beam_type=BeamType.ION,
        scan_rotation=scan_rotation,
        is_tescan=is_tescan,
    )

    def fib_px(position: FibsemStagePosition) -> np.ndarray:
        dx, dy = beam.to_plane(position, fib_pose)  # metres in the plane, y down
        return np.array(
            [
                fib_shape[1] / 2 + dx / fib_pixel_size,
                fib_shape[0] / 2 + dy / fib_pixel_size,
            ]
        )

    def compose(u: float, v: float) -> np.ndarray:
        return fib_px(
            project_image_point(
                Point(u, v), fm_pose, fm_pixel_size, fm_shape, fm_geometry
            )
        )

    # In-plane block: both projections are linear in the offset, so unit probes read
    # the columns off exactly. Probed about the FM centre, where the pose is recorded.
    cu, cv = fm_shape[1] / 2, fm_shape[0] / 2
    origin = compose(cu, cv)
    col_x = compose(cu + 1.0, cv) - origin
    col_y = compose(cu, cv + 1.0) - origin

    # Depth column: one FM xy pixel of distance along the sample normal at the FIB
    # pose, through the beam projection. The normal is perpendicular, in the stage
    # y-z plane, to the direction an in-plane displacement travels.
    col_z = _depth_column(beam, fib_pose, fm_pixel_size) / fib_pixel_size

    projection = np.column_stack([col_x, col_y, col_z])
    scale = fm_pixel_size / fib_pixel_size
    translation = origin - projection[:, :2] @ np.array([cu, cv])
    return NominalTransform(
        projection=projection,
        translation=translation,
        scale=scale,
        fm_pixel_size=fm_pixel_size,
        fm_pixel_size_z=fm_pixel_size_z,
        fib_pixel_size=fib_pixel_size,
    )


def _depth_column(
    beam: BeamStageProjection, fib_pose: FibsemStagePosition, length: float
) -> np.ndarray:
    """FIB-plane displacement (metres, y down) for ``length`` metres of sample depth.

    The sign is the one thing here that is calibrated rather than composed. Of the two
    normals to the surface, the one with the positive stage-z component is taken as
    *into the sample*: on the Arctis DEV-TEST stack the reflection channel (the milled
    surface) focuses at lower slice indices than the fluorescence (the cells beneath),
    so a rising slice index is deeper, and every saved fit -- two Arctis, eight METEOR
    -- maps a rising slice index onto the FIB-image direction this choice produces.
    A wrong sign here would put every seeded fit on the mirror branch, which the
    fixture tests would show as a systematic failure rather than a subtle one.
    """
    probe = 1.0e-6
    moved = beam.from_plane(0.0, probe, fib_pose)
    along = np.array(
        [(moved.y or 0.0) - (fib_pose.y or 0.0), (moved.z or 0.0) - (fib_pose.z or 0.0)]
    )
    norm = np.linalg.norm(along)
    if not norm:
        raise NominalTransformError(
            "The beam projection reports no in-plane travel; cannot orient the sample normal."
        )
    along /= norm
    normal = np.array([-along[1], along[0]])
    if normal[1] < 0:
        normal = -normal
    position = dataclasses.replace(fib_pose)
    position.y = (fib_pose.y or 0.0) + normal[0] * length
    position.z = (fib_pose.z or 0.0) + normal[1] * length
    dx, dy = beam.to_plane(position, fib_pose)
    return np.array([dx, dy])


def _complete_fm_pose(
    recorded: Optional[FibsemStagePosition], fib_geometry: FibsemHardwareGeometry
) -> Optional[FibsemStagePosition]:
    """A usable FM pose from the recorded position, completing r and t from the geometry.

    Both mountings image at a pose the hardware fixes, so a recorded position
    missing rotation and tilt can be completed rather than refused:

    * a **compustage** turns the grid over to face the objective underneath it:
      r = 0, t = -180 deg (``get_orientation("FM")``);
    * an **offset mount** sits parallel to the ion column, so the sample is imaged
      at the stage's FIB orientation: r = ``rotation_180``,
      t = ``fib_column_tilt - shuttle_pre_tilt`` (``get_orientation("FIB")``). An
      odemis-written METEOR stack records its position in odemis's own frame with no
      r or t; with this pose the composed transform lands 5-6 degrees from all seven
      saved METEOR fits, versus 30 degrees with the FIB image's own (milling) pose.
    """
    if recorded is None:
        return None
    if recorded.r is not None and recorded.t is not None:
        return recorded
    if fib_geometry.is_compustage:
        r, t = 0.0, np.radians(-180.0)
    else:
        r = np.radians(fib_geometry.rotation_180)
        t = np.radians(fib_geometry.fib_column_tilt - fib_geometry.shuttle_pre_tilt)
    return dataclasses.replace(
        recorded,
        r=float(r) if recorded.r is None else recorded.r,
        t=float(t) if recorded.t is None else recorded.t,
    )


def _is_tescan(fib_metadata) -> bool:
    from fibsem import manufacturers

    info = getattr(fib_metadata, "system_info", None)
    manufacturer = getattr(info, "manufacturer", None)
    return bool(manufacturer) and manufacturers.is_tescan(manufacturer)


# ── rotation helpers ──────────────────────────────────────────────────────


def _complete_rotation(rows: np.ndarray) -> np.ndarray:
    """A proper rotation whose first two rows are the given (nearly) orthonormal pair.

    The pair is re-orthonormalised first, so a projection built in slightly
    anisotropic units still yields a valid rotation to seed from.
    """
    r0 = np.asarray(rows[0], dtype=float)
    r1 = np.asarray(rows[1], dtype=float)
    r0 = r0 / np.linalg.norm(r0)
    r1 = r1 - np.dot(r1, r0) * r0
    r1 = r1 / np.linalg.norm(r1)
    r2 = np.cross(r0, r1)
    return np.vstack([r0, r1, r2])


def rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Geodesic angle between two rotation matrices, in degrees."""
    cos = (np.trace(a.T @ b) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def branch_of(
    rotation: np.ndarray, nominal: NominalTransform
) -> Tuple[str, float, float]:
    """Which branch a fitted rotation sits on: ``("nominal" | "mirror", angle_to_nominal, angle_to_mirror)``."""
    to_nominal = nominal.angle_to(rotation)
    to_mirror = nominal.mirrored().angle_to(rotation)
    return ("nominal" if to_nominal <= to_mirror else "mirror", to_nominal, to_mirror)
