from __future__ import annotations

import datetime
import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import yaml

from fibsem.constants import DATETIME_FILE
from fibsem.conversions import image_to_microscope_image_coordinates_px
from fibsem.correlation.refractive_index import apply_z_surface_correction
from fibsem.correlation.solver import (
    MIN_PAIRS,
    RigidFit,
    fit_from_seed,
    fit_rigid,
    fit_unseeded,
)
from fibsem.correlation.structures import (
    Coordinate,
    CorrelationInputData,
    CorrelationPointOfInterest,
    CorrelationResult,
    PointXYZ,
)
from fibsem.structures import Point

# A seed for :func:`correlate`: a 3x3 rotation and a scale (FIB px per FM px).
Seed = Tuple[np.ndarray, float]


def correlate(
    markers_3d: np.ndarray,
    markers_2d: np.ndarray,
    poi_3d: np.ndarray,
    rotation_center: List[float],
    imageProps: list = None,
    seed: Optional[Seed] = None,
) -> dict:
    """
    Fit the rigid transform from the 3D markers to the 2D markers and reproject the
    points of interest (POI) into the 2D image.

    Args:
        markers_3d: array of correlation marker positions for 3D image
        markers_2d: array of correlation marker positions for 2D image
        poi_3d:     array of points of interest for 3D image
        rotation_center: center of rotation for the 3D image (x,y,z)
        imageProps: properties of the images
            ([2d_image_shape, 2d_image_pixel_size_um, 3d_image_shape])
        seed: ``(rotation, scale)`` to restart the fit from, exactly (one local solve).
            None searches every rotation (:func:`fibsem.correlation.solver.fit_unseeded`).
    Returns:
        Dictionary with input and output data:
            input: {
                "markers_3d": np.ndarray[float],    # 3D marker positions
                "markers_2d": np.ndarray[float],    # 2D marker positions
                "poi_3d": np.ndarray[float],        # 3D point of interest positions
                "rotation_center": list[float],     # center of rotation for the 3D image
                "imageProps": list                  # properties of the images
            },
            output: {
                "transform": RigidFit,                              # the fit
                "reprojected_3d_coordinates": np.ndarray[float],    # reprojected 3D marker positions in 2D image
                "reprojected_2d_poi": np.ndarray[float],            # reprojected 3D poi in 2D image
                "reprojection_error": np.ndarray[float],            # reprojection error between reprojected 3D markers and 2D markers
                "center_of_mass_3d_markers": list[float],           # center of mass of 3D markers
                "modified_translation": list[float]                 # modified translation (rotation center not at 0,0,0)

    """
    assert markers_3d.shape[1] == 3, "Markers 3D do not have 3 dimensions"

    # coordinate arrays
    mark_3d = markers_3d.T  # fm markers (3D)
    mark_2d = markers_2d[:, :2].T  # fib markers (2D)
    poi_3d = poi_3d.T  # points of interest (3D)

    if seed is None:
        transf = fit_unseeded(mark_3d.T, mark_2d.T)
    else:
        rotation, scale = seed
        transf = fit_rigid(mark_3d.T, mark_2d.T, [(np.asarray(rotation), scale)])

    if imageProps:
        # the translation for a cubic volume (offset added to coordinates): the same
        # rotation and scale, so one solve restarted at this fit reproduces them
        shape_2d, pixel_size, shape_3d = imageProps
        offset = (max(shape_3d) - np.array(shape_3d)) * 0.5
        mark_3d_cube = np.copy(mark_3d) + offset[::-1, np.newaxis]
        transf_cube = fit_rigid(
            mark_3d_cube.T, mark_2d.T, [(transf.rotation, transf.scale)]
        )
    else:
        transf_cube = transf

    # reproject_points of interest
    reprojected_poi_2d = None
    if len(poi_3d) > 0:
        reprojected_poi_2d = transf.project(poi_3d.T).T

    # transform markers
    reprojected_coordinates_3d = transf.project(mark_3d.T).T

    # calculate translation if rotation center is not at (0,0,0)
    modified_translation = transf_cube.translation_about(rotation_center)

    # center of mass of 3D markers
    cm_3D_markers = mark_3d.mean(axis=-1).tolist()

    # delta calc,real
    reprojection_error = reprojected_coordinates_3d[:2, :] - mark_2d

    return {
        "input": {
            "markers_3d": mark_3d,
            "markers_2d": mark_2d,
            "poi_3d": poi_3d,
            "rotation_center": rotation_center,
            "imageProps": imageProps,
        },
        "output": {
            "transform": transf,
            "reprojected_3d_coordinates": reprojected_coordinates_3d,
            "reprojected_2d_poi": reprojected_poi_2d,
            "reprojection_error": reprojection_error,
            "center_of_mass_3d_markers": cm_3D_markers,
            "modified_translation": modified_translation,
        },
    }


def run_correlation(
    fib_coords: np.ndarray,
    fm_coords: np.ndarray,
    poi_coords: np.ndarray,
    image_props: tuple,
    rotation_center: tuple,
    path: Optional[str] = None,
    fib_image_filename: str = "",
    fm_image_filename: str = "",
    seed: Optional[Seed] = None,
) -> dict:
    """Run the correlation between the FIB and FM images"""
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

    # full correlation data
    full_correlation_data = {
        "metadata": {
            "timestamp": datetime.datetime.now().strftime(DATETIME_FILE),
            "data_path": path,
            "csv_path": os.path.join(path, "data.csv") if path is not None else path,
            "project_path": path,  # TODO: add project path
        },
        "correlation": correlation_data,
    }
    if path is not None:
        save_correlation_data(full_correlation_data, path)

    return correlation_data


##### CORRELATION RESULTS #####


# convert 2D image coordinates to microscope image coordinates
def convert_poi_to_microscope_coordinates(
    poi_coordinates: np.ndarray, fib_image_shape: tuple, pixel_size_um: float
) -> list:
    poi_image_coordinates: list = []

    for i in range(poi_coordinates.shape[1]):
        px = poi_coordinates[:, i]  # (x, y, z) in pixel coordinates
        px = [float(px[0]), float(px[1])]
        pt_px = image_to_microscope_image_coordinates_px(
            Point(px[0], px[1]), fib_image_shape, subpixel_precision=True
        )
        px_x, px_y = pt_px.x, pt_px.y
        pt_um = (
            px_x * pixel_size_um,
            px_y * pixel_size_um,
        )  # point in microscope image coordinates (um)
        poi_image_coordinates.append(
            {
                "image_px": px,
                "px": [px_x, px_y],
                "px_um": [pt_um[0], pt_um[1]],  # micrometers
                "px_m": [pt_um[0] * 1e-6, pt_um[1] * 1e-6],
            }  # meters
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


def save_correlation_data(data: dict, path: str) -> str:
    correlation_data_filename = os.path.join(path, "correlation_data.yaml")
    with open(correlation_data_filename, "w") as file:
        yaml.dump(data, file)

    logging.info(f"Correlation data saved to: {correlation_data_filename}")

    return correlation_data_filename


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
        poi = CorrelationPointOfInterest(image_px=Point(x, y))
        if fib_shape is not None and pixel_size is not None:
            # pixel_size is in metres here (CorrelationInputData), not the µm
            # that convert_poi_to_microscope_coordinates takes.
            poi.px = image_to_microscope_image_coordinates_px(
                Point(x, y), fib_shape, subpixel_precision=True
            )
            poi.px_m = Point(poi.px.x * pixel_size, poi.px.y * pixel_size)
        pois.append(poi)
    return pois


# ── seeding the fit from the geometry (FIB-881) ───────────────────────────


def _fit_from_seed(
    fm_coords, fib_coords, rotation, scale
) -> Tuple[np.ndarray, float, float]:
    """The fit started at a rotation and scale, with nudged restarts: ``(R, s, rms)``."""
    fit = fit_from_seed(fm_coords, fib_coords, rotation, scale)
    return fit.rotation, fit.scale, fit.rms


def _mirror_rotation(rotation: np.ndarray) -> np.ndarray:
    """The other branch of a fitted rotation: depth axis reversed, still proper."""
    from fibsem.correlation.geometry import _complete_rotation

    rows = np.asarray(rotation, dtype=float)[:2].copy()
    rows[:, 2] *= -1
    return _complete_rotation(rows)


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
    path: Optional[str] = None,
    nominal=None,
) -> CorrelationResult:
    """Run correlation from a CorrelationInputData struct, returning a CorrelationResult.

    Args:
        data: the picked fiducials, POIs and images.
        path: where to write the legacy YAML, if anywhere.
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
        from fibsem.correlation.geometry import branch_of

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
        path=path,
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
                _mirror_rotation(fitted_r),
                float(transf["scale"]),
            )
            from fibsem.correlation.geometry import rotation_angle_deg

            crossed = rotation_angle_deg(r_m, fitted_r) < rotation_angle_deg(
                r_m, _mirror_rotation(fitted_r)
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
                    path=path,
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
