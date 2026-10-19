"""One correlation fit, without the GUI: the seed, the fit and what it measured.

The correlation tab runs these on its worker thread; anything else that fits a
correlation (a service, a script) calls the same functions, so a fit made
outside the tab is seeded, judged and recorded the way the tab's is.

- :func:`seed_for` builds the transform the fit starts from: a previous run's
  fitted rotation and scale (FIB-956), else the images' geometry (FIB-881),
  placed with a previous run's calibrated offset (FIB-979).
- :func:`fit_transform` runs the fit from that seed and attaches the verdict's
  evidence (FIB-956) and the placement offset this run measured (FIB-979).
- :func:`fm_to_fib` maps more FM points through a fitted result.
- :func:`run_correlation_from_data` is the fit itself: the rigid solve
  (:mod:`fibsem.correlation.solver`) on the picked points, both branches
  checked, packaged as a :class:`CorrelationResult`.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from fibsem.conversions import image_to_microscope_image_coordinates_px
from fibsem.correlation.config import MIN_FIDUCIAL_PAIRS
from fibsem.correlation.geometry import (
    NominalTransform,
    branch_of,
    mirror_rotation,
    rotation_angle_deg,
)
from fibsem.correlation.history import CorrelationRun
from fibsem.correlation.prediction import independent_pairs, pair_translation
from fibsem.correlation.prior import (
    PriorTransform,
    placement_offset_from_runs,
    prior_from_runs,
)
from fibsem.correlation.refractive_index import apply_z_surface_correction
from fibsem.correlation.solver import (
    MIN_PAIRS,
    RigidFit,
    Start,
    correlate,
    fit_from_seed,
)
from fibsem.correlation.structures import (
    Coordinate,
    CorrelationInputData,
    CorrelationPointOfInterest,
    CorrelationResult,
    PointStatus,
    PointXYZ,
)
from fibsem.structures import Point

__all__ = [
    "MAX_PRIOR_ANGLE_DEG",
    "Seed",
    "diagnostics_for",
    "fit_transform",
    "fm_to_fib",
    "geometry_transform",
    "measure_placement_offset",
    "prior_transform",
    "run_correlation_from_data",
    "seed_for",
]

Runs = Sequence[Tuple[str, CorrelationRun]]

# A previous run's rotation further than this from the geometry's is not
# a better prior but a different answer: the one Arctis run checked was an
# unseeded fit 62 degrees off (its depth column a tenth of the geometry's),
# while the METEOR fits sit 5-6 degrees from their derived geometry.
MAX_PRIOR_ANGLE_DEG = 15.0


@dataclass(frozen=True)
class Seed:
    """The transform a fit starts from, and where it came from."""

    # What seeds the fit, or None for an unseeded fit.
    transform: Optional[NominalTransform] = None
    # Why the geometry gave no seed (empty when it did), for the status line.
    note: str = ""
    # The geometry's translation before any placement offset: the stage
    # metadata's placement, which this run's miss is measured against.
    metadata_translation: Optional[np.ndarray] = None
    # The previous run the rotation and scale came from, if any.
    prior: Optional[PriorTransform] = None
    # ``(source, angle)`` of a previous run passed over for disagreeing with
    # the geometry by more than :data:`MAX_PRIOR_ANGLE_DEG`.
    prior_rejected: Optional[Tuple[str, float]] = None


def geometry_transform(
    fib_image, fm_image, assumed_fm_transform=None
) -> Tuple[Optional[NominalTransform], str]:
    """The geometry's own FM->FIB transform, or ``(None, why)``; no prior, no offset.

    ``assumed_fm_transform`` is the camera transform to assume for an FM stack
    that recorded no geometry (the host's assumption; the library never
    guesses, FIB-881).
    """
    from fibsem.correlation.geometry import (
        NominalTransformError,
        fm_geometry_for,
        nominal_transform,
    )

    if fib_image is None or fm_image is None:
        return None, ""
    fm_geometry = None
    fib_geometry = getattr(fib_image.metadata, "hardware_geometry", None)
    if (
        assumed_fm_transform is not None
        and fib_geometry is not None
        and getattr(fm_image.metadata, "geometry", None) is None
    ):
        fm_geometry = fm_geometry_for(fib_geometry, assumed_fm_transform)
    try:
        return nominal_transform(fib_image, fm_image, fm_geometry=fm_geometry), ""
    except NominalTransformError as exc:
        logging.info(f"Correlation fit not seeded from geometry: {exc}")
        return None, str(exc)
    except Exception as exc:  # a seed is an aid; never block the run on it
        logging.warning(f"Could not build the nominal correlation transform: {exc}")
        return None, "the nominal transform could not be built"


def prior_transform(
    fib_image,
    fm_image,
    runs: Runs,
    *,
    translation=None,
    assumed_fm_transform=None,
) -> Tuple[Optional[PriorTransform], Optional[Tuple[str, float]]]:
    """The fitted transform of a previous run, in these images' units, or None.

    Returns ``(prior, rejected)``. A run whose rotation disagrees with the
    geometry by more than :data:`MAX_PRIOR_ANGLE_DEG` is passed over and named
    in ``rejected`` as ``(source, angle)``; the geometry is used.
    """
    if not runs or fib_image is None or fm_image is None:
        return None, None
    fm_md = fm_image.metadata
    fib_md = fib_image.metadata
    fm_px = getattr(fm_md, "pixel_size_x", None)
    fm_pz = getattr(fm_md, "pixel_size_z", None)
    fib_px = getattr(getattr(fib_md, "pixel_size", None), "x", None)
    if not (fm_px and fm_pz and fib_px):
        return None, None
    prior = prior_from_runs(
        runs,
        fm_pixel_size=fm_px,
        fm_pixel_size_z=fm_pz,
        fib_pixel_size=fib_px,
        translation=translation,
    )
    geometry = (
        geometry_transform(fib_image, fm_image, assumed_fm_transform)[0]
        if prior is not None
        else None
    )
    if geometry is not None:
        angle = geometry.angle_to(prior.transform.rotation)
        if angle > MAX_PRIOR_ANGLE_DEG:
            logging.info(
                f"prior from {prior.source} is {angle:.0f} deg from the geometry; "
                "using the geometry"
            )
            return None, (prior.source, angle)
    return prior, None


def seed_for(
    fib_image,
    fm_image,
    runs: Runs = (),
    *,
    assumed_fm_transform=None,
    use_placement_offset: bool = True,
) -> Seed:
    """The transform a fit of these images starts from, or an unseeded :class:`Seed`.

    ``runs`` are previous correlation runs, ``(label, run)`` pairs most
    specific first (see :func:`fibsem.correlation.prior.experiment_runs`). An
    image that records no geometry gets an unseeded fit rather than a seed built
    on a guess (FIB-881), unless ``assumed_fm_transform`` names the camera
    transform to assume. The stage metadata misses by microns to tens of
    microns; a previous run measured that miss, so the placement adds it unless
    ``use_placement_offset`` is False (FIB-979). A previous run's fitted
    rotation and scale beat the geometry's; its translation is that run's stage
    offset, so the geometry's is used.
    """
    if fib_image is None or fm_image is None:
        return Seed()
    geometry, note = geometry_transform(fib_image, fm_image, assumed_fm_transform)
    metadata_translation = (
        np.asarray(geometry.translation, dtype=float) if geometry is not None else None
    )
    offset = placement_offset_from_runs(runs) if runs else None
    if (
        geometry is not None
        and offset is not None
        and use_placement_offset
        and geometry.fib_pixel_size
    ):
        geometry = dataclasses.replace(
            geometry,
            translation=metadata_translation
            + offset.offset_um / (geometry.fib_pixel_size * 1e6),
        )
    prior, rejected = prior_transform(
        fib_image,
        fm_image,
        runs,
        translation=geometry.translation if geometry is not None else None,
        assumed_fm_transform=assumed_fm_transform,
    )
    if prior is not None:
        transform, note = prior.transform, ""
    else:
        transform = geometry
    return Seed(
        transform=transform,
        note=note,
        metadata_translation=metadata_translation,
        prior=prior,
        prior_rejected=rejected,
    )


def diagnostics_for(
    data: CorrelationInputData, nominal, rows: Optional[List[int]] = None
) -> Optional[dict]:
    """The fit verdict's evidence for a seeded fit, or None when it cannot be
    computed (no seed, too few pairs, no pixel sizes). Never raises: the
    verdict is an aid, the result is the product.

    ``rows`` are each fitted pair's row in the user's lists (the fit skips
    rejected pairs), so the verdict names the row the user sees.
    """
    if nominal is None or len(data.fib_coordinates) < MIN_FIDUCIAL_PAIRS:
        return None
    try:
        from fibsem.correlation.verdict import diagnose

        fm_md = getattr(data.fm_image, "metadata", None)
        fib_px = data.fib_image_pixel_size
        fm_px = getattr(fm_md, "pixel_size_x", None)
        fm_pz = getattr(fm_md, "pixel_size_z", None)
        if not (fib_px and fm_px and fm_pz):
            return None
        poi = data.poi_coordinates[0].point if data.poi_coordinates else None
        return diagnose(
            [[c.point.x, c.point.y] for c in data.fib_coordinates],
            [[c.point.x, c.point.y, c.point.z] for c in data.fm_coordinates],
            nominal,
            fib_pixel_size=fib_px,
            fm_pixel_size=fm_px,
            fm_pixel_size_z=fm_pz,
            poi=[poi.x, poi.y, poi.z] if poi is not None else None,
            accepted=[c.status == PointStatus.ACCEPTED for c in data.fm_coordinates],
            indices=rows,
        ).to_dict()
    except Exception as exc:
        logging.warning(f"Fit diagnostics not computed: {exc}")
        return None


def measure_placement_offset(
    result: CorrelationResult,
    nominal: Optional[NominalTransform],
    metadata_translation: Optional[np.ndarray],
) -> Optional[list]:
    """This run's placement miss: where its fiducials put FM (0, 0) in the FIB
    image minus where the stage metadata put it, in microns (FIB-979).

    Measured with the seed's rotation and scale, the way the next lamella
    will apply it, from the pairs the user placed: an accepted prediction
    sits exactly where the previous offset put it and would only echo that
    offset back. None when the run was not seeded or no pair was placed.
    """
    data = result.input_data
    if nominal is None or metadata_translation is None or data is None:
        return None
    pairs = independent_pairs(data.fib_coordinates, data.fm_coordinates)
    fib_px = data.fib_image_pixel_size
    if not pairs or not fib_px:
        return None
    try:
        t_pairs = pair_translation(nominal.projection, pairs, nominal.z_anisotropy)
        offset_um = (t_pairs - metadata_translation) * fib_px * 1e6
        return [float(offset_um[0]), float(offset_um[1])]
    except Exception as exc:  # a record for the next lamella; never block
        logging.debug(f"placement offset not measured: {exc}")
        return None


def fit_transform(
    data: CorrelationInputData,
    seed: Optional[Seed] = None,
    rows: Optional[List[int]] = None,
) -> CorrelationResult:
    """Fit ``data`` from ``seed`` (unseeded when None).

    The result carries the verdict's evidence (``diagnostics``) and the
    placement offset this run measured (``placement_offset``). Raises what the
    fit raises; the diagnostics and the offset never do.
    """
    seed = seed or Seed()
    result = run_correlation_from_data(data, nominal=seed.transform)
    result.diagnostics = diagnostics_for(data, seed.transform, rows=rows)
    result.placement_offset = measure_placement_offset(
        result, seed.transform, seed.metadata_translation
    )
    return result


def fm_to_fib(
    result: CorrelationResult, fm_points: Sequence[Sequence[float]]
) -> List[CorrelationPointOfInterest]:
    """Where FM points land in the FIB image under a fitted ``result``.

    ``fm_points`` are ``(x, y, z)`` in pixels and slices of the stack the
    result was fitted on. The answer is in every coordinate system a fitted POI
    is (image pixels, centred pixels, metres from the image centre). No
    refractive-index correction is applied: these are points, not a target at
    depth.
    """
    points = np.asarray(fm_points, dtype=float).reshape(-1, 3).copy()
    points[:, 2] *= result.fm_z_scale
    data = result.input_data
    return _reproject_poi_via_transform(
        {
            "rotation_matrix": result.rotation_matrix,
            "scale": result.scale,
            "translation_around_rotation_center_zero": result.translation,
        },
        points,
        data.fib_image_shape if data is not None else None,
        data.fib_image_pixel_size if data is not None else None,
    )


# ── the fit itself ───────────────────────────────────────────────────────


def run_correlation(
    fib_coords: np.ndarray,
    fm_coords: np.ndarray,
    poi_coords: np.ndarray,
    image_props: tuple,
    rotation_center: tuple,
    fib_image_filename: str = "",
    fm_image_filename: str = "",
    seed: Optional[Start] = None,
) -> dict:
    """Fit the FIB and FM fiducials and package the fit as a run records it."""
    # run the correlation
    correlation_results = correlate(
        markers_3d=fm_coords,
        markers_2d=fib_coords,
        poi_3d=poi_coords,
        rotation_center=rotation_center,
        imageProps=image_props,
        seed=seed,
    )
    # input data
    input_data = {
        "fib_coordinates": fib_coords.tolist(),
        "fm_coordinates": fm_coords.tolist(),
        "poi_coordinates": poi_coords.tolist(),
        "image_properties": {
            "fib_image_filename": fib_image_filename,
            "fib_image_shape": list(image_props[0]),
            "fib_pixel_size_um": float(image_props[1]),
            "fm_image_filename": fm_image_filename,
            "fm_image_shape": list(image_props[2]),
        },
        "rotation_center": list(rotation_center),
        "rotation_center_custom": list(rotation_center),
        "method": "multi-point",
    }

    # output data
    correlation_data = parse_correlation_result_v2(
        cor_ret=correlation_results, input_data=input_data
    )

    return correlation_data


def _microscope_coordinates(
    x: float, y: float, fib_shape: Tuple[int, ...], pixel_size_m: float
) -> Tuple[Point, Point, Point, Point]:
    """One FIB image pixel in every coordinate system a POI is recorded in.

    Returns ``(image_px, px, px_um, px_m)``: the image pixel as given, the
    microscope image pixel (centred, y up, via
    :func:`fibsem.conversions.image_to_microscope_image_coordinates_px` at
    subpixel precision), and that in micrometres and in metres.
    ``pixel_size_m`` is in metres. The correlation output and the
    reprojection both go through here, so they cannot drift apart.
    """
    image_px = Point(x, y)
    px = image_to_microscope_image_coordinates_px(
        image_px, fib_shape, subpixel_precision=True
    )
    pixel_size_um = pixel_size_m * 1e6
    px_um = Point(px.x * pixel_size_um, px.y * pixel_size_um)
    px_m = Point(px.x * pixel_size_m, px.y * pixel_size_m)
    return image_px, px, px_um, px_m


# convert 2D image coordinates to microscope image coordinates
def convert_poi_to_microscope_coordinates(
    poi_coordinates: np.ndarray, fib_image_shape: tuple, pixel_size_um: float
) -> list:
    poi_image_coordinates: list = []
    pixel_size_m = pixel_size_um * 1e-6

    for i in range(poi_coordinates.shape[1]):
        image_px, px, px_um, px_m = _microscope_coordinates(
            float(poi_coordinates[0, i]),
            float(poi_coordinates[1, i]),
            fib_image_shape,
            pixel_size_m,
        )
        poi_image_coordinates.append(
            {
                "image_px": [image_px.x, image_px.y],
                "px": [px.x, px.y],
                "px_um": [px_um.x, px_um.y],  # micrometers
                "px_m": [px_m.x, px_m.y],  # meters
            }
        )

    return poi_image_coordinates


def extract_transformation_data(
    transf: RigidFit, mod_translation, reproj_3d, delta_2d
) -> dict:
    # difference between points after transforming 3D points to 2D
    delta_2d_mean_abs_err = np.absolute(delta_2d).mean(axis=1)

    transformation_data = {
        "transformation": {
            "scale": float(transf.scale),
            "rotation_eulers": transf.eulers_deg().tolist(),
            "rotation_matrix": transf.rotation.tolist(),
            "translation_around_rotation_center_custom": mod_translation.tolist(),
            "translation_around_rotation_center_zero": transf.translation.tolist(),
        },
        "error": {
            "reprojected_3d": reproj_3d.tolist(),
            "delta_2d": delta_2d.tolist(),
            "mean_absolute_error": delta_2d_mean_abs_err.tolist(),
            "rms_error": float(transf.rms),
        },
    }

    return transformation_data


def parse_correlation_result_v2(cor_ret: dict, input_data: dict) -> dict:
    # point of interest data
    spots_2d = cor_ret["output"][
        "reprojected_2d_poi"
    ]  # (points of interest in 2D image)
    fib_image_shape = input_data["image_properties"]["fib_image_shape"]
    pixel_size_um = input_data["image_properties"]["fib_pixel_size_um"]

    poi_image_coordinates = convert_poi_to_microscope_coordinates(
        spots_2d, fib_image_shape, pixel_size_um
    )

    # transformation data
    transf = cor_ret["output"]["transform"]  # transformation matrix
    reproj_3d = cor_ret["output"][
        "reprojected_3d_coordinates"
    ]  # reprojected 3D points to 2D points
    delta_2d = cor_ret["output"][
        "reprojection_error"
    ]  # difference between reprojected 3D points and 2D points (in pixels)
    mod_translation = cor_ret["output"][
        "modified_translation"
    ]  # translation around rotation center
    transformation_data = extract_transformation_data(
        transf=transf,
        mod_translation=mod_translation,
        reproj_3d=reproj_3d,
        delta_2d=delta_2d,
    )

    correlation_data = {"input": input_data, "output": {}}
    correlation_data["output"].update(transformation_data)
    correlation_data["output"].update({"poi": poi_image_coordinates})

    return correlation_data


def _coords_to_array(coords: list[Coordinate]) -> np.ndarray:
    return np.array(
        [[c.point.x, c.point.y, c.point.z] for c in coords], dtype=np.float32
    ).reshape(-1, 3)


def _reproject_poi_via_transform(
    transformation: dict,
    poi_coords: np.ndarray,
    fib_shape: Optional[Tuple[int, ...]],
    pixel_size: Optional[float],
) -> List[CorrelationPointOfInterest]:
    """Project (N, 3) FM-space POIs into the FIB image using a fitted transform.

    Reconstructs ``y = s * R @ x + d`` from the parsed transformation data.
    ``rotation_matrix`` is the 3x3 rotation matrix; ``d`` is the translation
    around rotation center zero — the same parameters ``correlate()`` uses to
    reproject the POI.
    """
    R = np.asarray(transformation["rotation_matrix"], dtype=float)
    if R.shape != (3, 3):
        # Unreachable from the only call site — extract_transformation_data
        # always writes the 3x3 matrix — but kept, and kept loud. A (3,) R broadcasts
        # cleanly through the matmul below and yields plausible garbage instead
        # of raising, and a broken transform here is the same broken transform
        # that produced the result's own POI, so there is nothing worth
        # salvaging. This used to log a warning and return [], which dropped the
        # ghost marker and left the run looking fine.
        raise ValueError(
            f"Cannot reproject POI: expected a 3x3 rotation matrix, got {R.shape}"
        )
    s = float(transformation["scale"])
    d = np.asarray(
        transformation["translation_around_rotation_center_zero"], dtype=float
    )
    projected = s * (R @ poi_coords.T.astype(float)) + d[:, None]  # (3, N)

    pois: List[CorrelationPointOfInterest] = []
    for i in range(projected.shape[1]):
        x, y = float(projected[0, i]), float(projected[1, i])
        if fib_shape is not None and pixel_size is not None:
            # pixel_size is in metres (CorrelationInputData)
            image_px, px, _, px_m = _microscope_coordinates(x, y, fib_shape, pixel_size)
            poi = CorrelationPointOfInterest(image_px=image_px, px=px, px_m=px_m)
        else:
            poi = CorrelationPointOfInterest(image_px=Point(x, y))
        pois.append(poi)
    return pois


# ── seeding the fit from the geometry (FIB-881) ───────────────────────────


def _fit_from_seed(
    fm_coords, fib_coords, rotation, scale
) -> Tuple[np.ndarray, float, float]:
    """The fit started at a rotation and scale, with nudged restarts: ``(R, s, rms)``."""
    fit = fit_from_seed(fm_coords, fib_coords, rotation, scale)
    return fit.rotation, fit.scale, fit.rms


# The two branches of an unseeded fit are told apart only by the depth of the
# fiducials; when they fit within this of each other, the points cannot say
# which is real (FIB-1249). Relative, with a 1 px floor for very good fits.
_MIRROR_CLOSE_FRACTION = 0.25
_MIRROR_CLOSE_PX = 1.0


def _unseeded_mirror_warning(rms: float, rms_mirror: Optional[float]) -> Optional[str]:
    """The warning for an unseeded fit whose mirror branch fits nearly as well."""
    if rms_mirror is None:
        return None
    close = max(_MIRROR_CLOSE_PX, _MIRROR_CLOSE_FRACTION * min(rms, rms_mirror))
    if abs(rms_mirror - rms) > close:
        return None
    return (
        f"The mirror branch fits nearly as well (RMS {rms_mirror:.1f} vs "
        f"{rms:.1f} px): the fiducials barely set the depth direction, so the "
        "target may be on the wrong side. Add fiducials at other depths, or "
        "check the target against the FM image."
    )


def _fm_z_scale(data: CorrelationInputData) -> float:
    """FM slice thickness in xy pixels, or 1.0 when the stack does not say.

    The fit is rigid, so it needs isotropic units; a raw stack's slice index is
    not one. Fitting on raw slices leaves the depth gain wrong by this factor
    (2-5x on the data checked), which shrinks the refractive-index correction by
    the same amount. Applied transiently: the picked coordinates keep their slices.
    """
    fm_md = getattr(data.fm_image, "metadata", None)
    xy = getattr(fm_md, "pixel_size_x", None)
    z = getattr(fm_md, "pixel_size_z", None)
    if not xy or not z:
        return 1.0
    scale = float(z) / float(xy)
    return scale if np.isfinite(scale) and scale > 0 else 1.0


def run_correlation_from_data(
    data: CorrelationInputData,
    nominal=None,
) -> CorrelationResult:
    """Run correlation from a CorrelationInputData struct, returning a CorrelationResult.

    Args:
        data: the picked fiducials, POIs and images.
        nominal: a :class:`fibsem.correlation.geometry.NominalTransform` built from
            the images' metadata. When given, the fit is *seeded* with its rotation
            and scale and the branch the fit lands on is checked against it; the
            result records both. When None the fit runs unseeded as before, and the
            branch check only reports how well the mirror branch also fits.
    """

    if data.surface_coordinate is not None and data.fm_surface_coordinate is not None:
        raise ValueError(
            "Both surface_coordinate (FIB) and fm_surface_coordinate (FM) are set — "
            "only one surface point is supported (applying both would double-correct)."
        )

    fib_coords = _coords_to_array(data.fib_coordinates)
    fm_coords = _coords_to_array(data.fm_coordinates)
    poi_coords = (
        _coords_to_array(data.poi_coordinates)
        if data.poi_coordinates
        else np.zeros((0, 3), dtype=np.float32)
    )

    # Isotropic units for the rigid fit: slices -> xy pixels (see _fm_z_scale).
    fm_z_scale = _fm_z_scale(data)
    if fm_z_scale != 1.0:
        fm_coords = fm_coords.copy()
        fm_coords[:, 2] *= fm_z_scale
        poi_coords = poi_coords.copy()
        poi_coords[:, 2] *= fm_z_scale
        logging.info(
            f"FM z scaled by {fm_z_scale:.3f} (slice thickness in xy pixels) for the fit"
        )

    # Pre-correlation refractive-index correction: scale POI z about the FM
    # surface z. Applied transiently — data.poi_coordinates keeps the picked z,
    # so re-running never double-applies.
    pre_correction_applied = False
    poi_coords_original = poi_coords
    if (
        data.fm_surface_coordinate is not None
        and data.ri_pre_correction_factor is not None
        and len(poi_coords)
    ):
        surface_z = data.fm_surface_coordinate.point.z * fm_z_scale
        factor = data.ri_pre_correction_factor
        poi_coords = apply_z_surface_correction(poi_coords, surface_z, factor)
        pre_correction_applied = True
        logging.info(
            f"Pre-correlation RI correction: factor={factor:.4f}, "
            f"surface_z={surface_z:.2f}, "
            f"poi_z {poi_coords_original[:, 2].tolist()} -> {poi_coords[:, 2].tolist()}"
        )

    # image_props: (fib_shape, pixel_size_um, fm_shape_3d)
    fib_shape = data.fib_image_shape
    pixel_size = data.fib_image_pixel_size
    fm_shape = data.fm_image_shape
    if fib_shape is not None and pixel_size is not None and fm_shape is not None:
        image_props = (fib_shape, pixel_size * 1e6, fm_shape[1:])  # (Z, Y, X)
    else:
        image_props = None

    # rotation center: FM volume centre (matching app.py default)
    if fm_shape is not None:
        halfmax = int(max(fm_shape[1:]) * 0.5)
        rotation_center = (halfmax, halfmax, halfmax)
    else:
        rotation_center = (0, 0, 0)

    # Seeded fit: solve both branches from the geometry's rotation and keep the
    # one the geometry says is real. The final packaging run below restarts at
    # that solution, so it reproduces it rather than re-solving from scratch.
    seed: Optional[dict] = None
    branch_check: Optional[dict] = None
    run_kwargs: dict = {}
    if nominal is not None and len(fm_coords) >= MIN_PAIRS:
        r_n, s_n, rms_n = _fit_from_seed(
            fm_coords, fib_coords, nominal.rotation, nominal.scale
        )
        mirror = nominal.mirrored()
        r_m, s_m, rms_m = _fit_from_seed(
            fm_coords, fib_coords, mirror.rotation, mirror.scale
        )
        # the solve started at the nominal seed may itself have crossed over
        on_nominal = [
            (r, s, rms)
            for (r, s, rms) in ((r_n, s_n, rms_n), (r_m, s_m, rms_m))
            if branch_of(r, nominal)[0] == "nominal"
        ]
        on_mirror = [
            (r, s, rms)
            for (r, s, rms) in ((r_n, s_n, rms_n), (r_m, s_m, rms_m))
            if branch_of(r, nominal)[0] == "mirror"
        ]
        chosen = (
            min(on_nominal, key=lambda c: c[2]) if on_nominal else (r_n, s_n, rms_n)
        )
        rms_other = min(c[2] for c in on_mirror) if on_mirror else None
        _, angle_nominal, angle_mirror = branch_of(chosen[0], nominal)
        warning = None
        if not on_nominal:
            warning = (
                "Neither seeded solve stayed on the geometry's branch; the fit is "
                "unconstrained by the seed."
            )
        elif (
            rms_other is not None
            and rms_other < 0.5 * chosen[2]
            and chosen[2] - rms_other > 2.0
        ):
            warning = (
                f"The mirror branch fits far better (RMS {rms_other:.1f} vs {chosen[2]:.1f} px): "
                "check the scan rotation, camera transform and stage poses the seed was built from."
            )
        seed = {
            "eulers_deg": [float(a) for a in nominal.eulers_deg()],
            "scale": float(nominal.scale),
            "foreshortening": float(nominal.foreshortening),
            "dimage_dz_px_per_slice": [
                float(v) for v in nominal.dimage_dz_px_per_slice
            ],
        }
        branch_check = {
            "selected": "nominal",
            "rms_selected": float(chosen[2]),
            "rms_mirror": None if rms_other is None else float(rms_other),
            "angle_to_nominal_deg": float(angle_nominal),
            "angle_to_mirror_deg": float(angle_mirror),
            "warning": warning,
        }
        run_kwargs["seed"] = (chosen[0], chosen[1])

    correlation_data = run_correlation(
        fib_coords=fib_coords,
        fm_coords=fm_coords,
        poi_coords=poi_coords,
        image_props=image_props,
        rotation_center=rotation_center,
        fib_image_filename=data.fib_image_filename or "",
        fm_image_filename=data.fm_image_filename or "",
        **run_kwargs,
    )

    out = correlation_data["output"]
    transf = out["transformation"]
    err = out["error"]

    # Unseeded: solve the *other* branch of what was found as well, and keep the
    # better. The unseeded search covers every rotation, but a mirror that fits
    # nearly as well is a warning in its own right (FIB-1249), so its RMS is
    # measured here. When the mirror wins, the final fit is re-run from it, as
    # the seeded path does; when the first fit was better, it is returned as is.
    if branch_check is None and len(fm_coords) >= MIN_PAIRS:
        try:
            fitted_r = np.asarray(transf["rotation_matrix"], dtype=float)
            r_m, s_m, rms_m = _fit_from_seed(
                fm_coords,
                fib_coords,
                mirror_rotation(fitted_r),
                float(transf["scale"]),
            )
            crossed = rotation_angle_deg(r_m, fitted_r) < rotation_angle_deg(
                r_m, mirror_rotation(fitted_r)
            )
            rms_first = float(err["rms_error"])
            switched = not crossed and rms_m < rms_first - 1e-6
            if switched:
                correlation_data = run_correlation(
                    fib_coords=fib_coords,
                    fm_coords=fm_coords,
                    poi_coords=poi_coords,
                    image_props=image_props,
                    rotation_center=rotation_center,
                    fib_image_filename=data.fib_image_filename or "",
                    fm_image_filename=data.fm_image_filename or "",
                    seed=(r_m, s_m),
                )
                out = correlation_data["output"]
                transf = out["transformation"]
                err = out["error"]
            rms_here = float(err["rms_error"])
            rms_other = rms_first if switched else (None if crossed else float(rms_m))
            branch_check = {
                "selected": "unseeded",
                "rms_selected": rms_here,
                "rms_mirror": rms_other,
                "switched_to_mirror": switched,
                "angle_to_nominal_deg": None,
                "angle_to_mirror_deg": None,
                "warning": _unseeded_mirror_warning(rms_here, rms_other),
            }
        except Exception as exc:  # the check is advisory; never fail a run over it
            logging.debug(f"Mirror-branch check skipped: {exc}")
    if branch_check is not None:
        z = fm_coords[:, 2] / fm_z_scale
        branch_check["fiducial_z_span_slices"] = float(np.ptp(z)) if len(z) else 0.0

    d2d = err["delta_2d"]  # [[x1,x2,...], [y1,y2,...]]
    r3d = err["reprojected_3d"]  # [[x1,...], [y1,...], [z1,...]]
    n_markers = len(d2d[0])

    # Ghost markers: where the POIs would land without the pre-correction,
    # reprojected through the same fitted transform (no re-fit).
    poi_uncorrected: List[CorrelationPointOfInterest] = []
    if pre_correction_applied:
        poi_uncorrected = _reproject_poi_via_transform(
            transf, poi_coords_original, fib_shape, pixel_size
        )

    return CorrelationResult(
        poi=[
            CorrelationPointOfInterest(
                image_px=Point(p["image_px"][0], p["image_px"][1]),
                px=Point(p["px"][0], p["px"][1]),
                px_m=Point(p["px_m"][0], p["px_m"][1]),
            )
            for p in out["poi"]
        ],
        poi_uncorrected=poi_uncorrected,
        scale=transf["scale"],
        rotation_eulers=transf["rotation_eulers"],
        rotation_matrix=transf["rotation_matrix"],
        translation=transf["translation_around_rotation_center_zero"],
        translation_custom=transf["translation_around_rotation_center_custom"],
        rms_error=err["rms_error"],
        mean_absolute_error=err["mean_absolute_error"],
        delta_2d=[Point(d2d[0][i], d2d[1][i]) for i in range(n_markers)],
        reprojected_3d=[
            PointXYZ(r3d[0][i], r3d[1][i], r3d[2][i]) for i in range(len(r3d[0]))
        ],
        input_data=data,
        refractive_index_correction_factor=(
            data.ri_pre_correction_factor if pre_correction_applied else None
        ),
        refractive_index_correction_mode="pre" if pre_correction_applied else None,
        fm_z_scale=fm_z_scale,
        seed=seed,
        branch_check=branch_check,
    )
