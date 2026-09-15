from __future__ import annotations

import contextlib
import datetime
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

from fibsem.constants import DATETIME_FILE
from fibsem.conversions import image_to_microscope_image_coordinates_px
from fibsem.correlation.pyto.rigid_3d import (
    Rigid3D,  # NOTE: this is still a 3DCT dependency, migrate
)
from fibsem.correlation.structures import (
    Coordinate,
    CorrelationInputData,
    CorrelationPointOfInterest,
    CorrelationResult,
    PointXYZ,
    apply_z_surface_correction,
)
from fibsem.structures import Point

DEFAULT_OPTIMIZATION_PARAMETERS = {
    "random_rotations": True,
    "rotation_init": "gl2",
    "restrict_rotations": 0.1,
    "scale": None,
    "random_scale": True,
    "scale_init": "gl2",
    "ninit": 10,
}


def correlate(
    markers_3d: np.ndarray,
    markers_2d: np.ndarray,
    poi_3d: np.ndarray,
    rotation_center: List[float],
    imageProps: list = None,
    optimiser_params: Dict = DEFAULT_OPTIMIZATION_PARAMETERS,
) -> dict:
    """
    Iteratively calculate the correlation between 3D and 2D markers and reproject the points of interest (POI) into the 2D image

    Args:
        markers_3d: array of correlation marker positions for 3D image
        markers_2d: array of correlation marker positions for 2D image
        poi_3d:     array of points of interest for 3D image
        rotation_center: center of rotation for the 3D image (x,y,z)
        imageProps: properties of the images
            ([2d_image_shape, 2d_image_pixel_size_um, 3d_image_shape])
        optimiser_params: dictionary with optimization parameters
            {
                'random_rotations': bool,      # random rotations
                'rotation_init': float,        # initial rotation in degrees
                'restrict_rotations': float,   # restrict rotations
                'scale': float,                # scale
                'random_scale': bool,          # random scale
                'scale_init': float,           # initial scale
                'ninit': float                 # number of iterations
            }
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
                "transform": Rigid3D,                               # transformation object
                "reprojected_3d_coordinates": np.ndarray[float],    # reprojected 3D marker positions in 2D image
                "reprojected_2d_poi": np.ndarray[float],            # reprojected 3D poi in 2D image
                "reprojection_error": np.ndarray[float],            # reprojection error between reprojected 3D markers and 2D markers
                "center_of_mass_3d_markers": list[float],           # center of mass of 3D markers
                "modified_translation": list[float]                 # modified translation (rotation center not at 0,0,0)

    """
    # TODO: convert imageProps to a dataclass or dict?

    # read optimization parameters
    random_rotations = optimiser_params.get(
        "random_rotations", DEFAULT_OPTIMIZATION_PARAMETERS["random_rotations"]
    )
    rotation_init = optimiser_params.get(
        "rotation_init", DEFAULT_OPTIMIZATION_PARAMETERS["rotation_init"]
    )
    restrict_rotations = optimiser_params.get(
        "restrict_rotations", DEFAULT_OPTIMIZATION_PARAMETERS["restrict_rotations"]
    )
    scale = optimiser_params.get("scale", DEFAULT_OPTIMIZATION_PARAMETERS["scale"])
    random_scale = optimiser_params.get(
        "random_scale", DEFAULT_OPTIMIZATION_PARAMETERS["random_scale"]
    )
    scale_init = optimiser_params.get(
        "scale_init", DEFAULT_OPTIMIZATION_PARAMETERS["scale_init"]
    )
    ninit: float = optimiser_params.get(
        "ninit", DEFAULT_OPTIMIZATION_PARAMETERS["ninit"]
    )

    assert markers_3d.shape[1] == 3, "Markers 3D do not have 3 dimensions"

    # coordinate arrays
    mark_3d = markers_3d.T  # fm markers (3D)
    mark_2d = markers_2d[:, :2].T  # fib markers (2D)
    poi_3d = poi_3d.T  # points of interest (3D)

    # convert Eulers in degrees to Caley-Klein params. A seed is three x-convention
    # Euler angles in degrees (see fibsem.correlation.geometry.NominalTransform.eulers_deg);
    # 'gl2' asks the solver for its own 2D-affine initialisation, None for its default.
    if rotation_init is None or isinstance(rotation_init, str):
        einit = rotation_init
    else:
        rotation_init_rad = np.deg2rad(np.asarray(rotation_init, dtype=float))
        if rotation_init_rad.shape != (3,):
            raise ValueError(
                "rotation_init must be 'gl2', None, or three Euler angles in degrees; "
                f"got {rotation_init!r}"
            )
        einit = Rigid3D.euler_to_ck(angles=rotation_init_rad, mode="x")

    # establish correlation
    # Suppress stdout and stderr
    with open(os.devnull, "w") as fnull, contextlib.redirect_stdout(
        fnull
    ), contextlib.redirect_stderr(fnull):
        transf = Rigid3D.find_32(
            x=mark_3d,
            y=mark_2d,
            scale=scale,
            randome=random_rotations,
            einit=einit,
            einit_dist=restrict_rotations,
            randoms=random_scale,
            sinit=scale_init,
            ninit=ninit,
        )

    if imageProps:
        # establish correlation for cubic rotation (offset added to coordinates)
        shape_2d, pixel_size, shape_3d = imageProps
        offset = (max(shape_3d) - np.array(shape_3d)) * 0.5

        mark_3d_cube = np.copy(mark_3d) + offset[::-1, np.newaxis]
        # Suppress stdout and stderr
        with open(os.devnull, "w") as fnull, contextlib.redirect_stdout(
            fnull
        ), contextlib.redirect_stderr(fnull):
            transf_cube = Rigid3D.find_32(
                x=mark_3d_cube,
                y=mark_2d,
                scale=scale,
                randome=random_rotations,
                einit=einit,
                einit_dist=restrict_rotations,
                randoms=random_scale,
                sinit=scale_init,
                ninit=ninit,
            )
    else:
        transf_cube = transf

    # reproject_points of interest
    reprojected_poi_2d = None
    if len(poi_3d) > 0:
        reprojected_poi_2d = transf.transform(x=poi_3d)

    # transform markers
    reprojected_coordinates_3d = transf.transform(x=mark_3d)

    # calculate translation if rotation center is not at (0,0,0)
    modified_translation = transf_cube.recalculate_translation(
        rotation_center=rotation_center
    )

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


def save_results(correlation_results: dict, results_file: str):
    """
    Save the results of the correlation to a file (old .txt format)
    """
    from tdct.correlation import write_results

    # write transformation params and correlation
    write_results(
        transf=correlation_results["output"]["transform"],
        res_file_name=results_file,
        spots_3d=correlation_results["input"]["poi_3d"],
        spots_2d=correlation_results["output"]["reprojected_2d_poi"],
        markers_3d=correlation_results["input"]["markers_3d"],
        transformed_3d=correlation_results["output"]["reprojected_3d_coordinates"],
        markers_2d=correlation_results["input"]["markers_2d"],
        rotation_center=correlation_results["input"]["rotation_center"],
        modified_translation=correlation_results["output"]["modified_translation"],
        imageProps=correlation_results["input"]["imageProps"],
    )


def run_correlation(
    fib_coords: np.ndarray,
    fm_coords: np.ndarray,
    poi_coords: np.ndarray,
    image_props: tuple,
    rotation_center: tuple,
    path: Optional[str] = None,
    fib_image_filename: str = "",
    fm_image_filename: str = "",
    optimiser_params: Optional[Dict] = None,
) -> dict:
    """Run the correlation between the FIB and FM images"""
    # run the correlation
    correlation_results = correlate(
        markers_3d=fm_coords,
        markers_2d=fib_coords,
        poi_3d=poi_coords,
        rotation_center=rotation_center,
        imageProps=image_props,
        optimiser_params=optimiser_params or DEFAULT_OPTIMIZATION_PARAMETERS,
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


def extract_transformation_data(transf, mod_translation, reproj_3d, delta_2d) -> dict:
    # extract eulers in degrees
    eulers = transf.extract_euler(r=transf.q, mode="x", ret="one")
    eulers = eulers * 180 / np.pi

    # RMS error
    rms_error = transf.rmsError

    # difference between points after transforming 3D points to 2D
    delta_2d_mean_abs_err = np.absolute(delta_2d).mean(axis=1)

    transformation_data = {
        "transformation": {
            "scale": float(transf.s_scalar),
            "rotation_eulers": eulers.tolist(),
            "rotation_quaternion": transf.q.tolist(),
            "translation_around_rotation_center_custom": mod_translation.tolist(),
            "translation_around_rotation_center_zero": transf.d.tolist(),
        },
        "error": {
            "reprojected_3d": reproj_3d.tolist(),
            "delta_2d": delta_2d.tolist(),
            "mean_absolute_error": delta_2d_mean_abs_err.tolist(),
            "rms_error": float(rms_error),
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
    )


def _reproject_poi_via_transform(
    transformation: dict,
    poi_coords: np.ndarray,
    fib_shape: Optional[Tuple[int, ...]],
    pixel_size: Optional[float],
) -> List[CorrelationPointOfInterest]:
    """Project (N, 3) FM-space POIs into the FIB image using a fitted transform.

    Reconstructs ``y = s * R @ x + d`` from the parsed transformation data.
    The ``rotation_quaternion`` field stores ``Rigid3D.q``, which is the 3x3
    rotation matrix (see ``Rigid3D.transform``); ``d`` is the translation
    around rotation center zero — the same parameters ``correlate()`` uses to
    reproject the POI.
    """
    R = np.asarray(transformation["rotation_quaternion"], dtype=float)
    if R.shape != (3, 3):
        # Unreachable from the only call site — extract_transformation_data
        # always writes Rigid3D.q — but kept, and kept loud. A (3,) R broadcasts
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

# Restarts scattered about a seed, in Cayley-Klein distance. The solver's
# constrained optimiser sometimes fails to leave the exact seed (it returned the
# seed untouched on three of seven METEOR runs), and a single radius of restarts
# occasionally settles in a poor local minimum; the exact seed plus two radii of
# restarts found the optimum on every saved fit. Cheap: each is one local solve.
_SEED_JITTERS = (0.05, 0.2)
_SEED_RESTARTS = 8
# The restarts draw from numpy's global RNG (pyto's `make_random_ck`). Fixed here
# so a seeded fit is a function of its inputs: the same picks give the same
# answer on every run, and a flaky restart cannot flip which local minimum wins.
_SEED_RNG = 20260909


def seeded_optimiser_params(
    eulers_deg, scale: float, jitter: float = 0.0, ninit: int = 1
) -> Dict:
    """``correlate`` parameters for a fit started at a known rotation and scale."""
    return {
        "random_rotations": jitter > 0,
        "rotation_init": [float(a) for a in eulers_deg],
        "restrict_rotations": jitter,
        "scale": None,  # still fitted; the seed is a start, not a constraint
        "random_scale": False,
        "scale_init": float(scale),
        "ninit": ninit if jitter > 0 else 1,
    }


def _fit_from_seed(
    fm_coords, fib_coords, eulers_deg, scale
) -> Tuple[np.ndarray, float, float]:
    """Best of the exact seed and jittered restarts about it: ``(R, s, rms)``."""
    candidates = [seeded_optimiser_params(eulers_deg, scale)]
    candidates += [
        seeded_optimiser_params(eulers_deg, scale, jitter, _SEED_RESTARTS)
        for jitter in _SEED_JITTERS
    ]
    state = np.random.get_state()
    np.random.seed(_SEED_RNG)
    try:
        best = None
        for params in candidates:
            out = correlate(
                markers_3d=fm_coords,
                markers_2d=fib_coords,
                poi_3d=np.zeros((0, 3), dtype=np.float32),
                rotation_center=(0, 0, 0),
                imageProps=None,
                optimiser_params=params,
            )
            transf = out["output"]["transform"]
            candidate = (
                np.asarray(transf.q, dtype=float),
                float(transf.s_scalar),
                float(transf.rmsError),
            )
            if best is None or candidate[2] < best[2]:
                best = candidate
    finally:
        np.random.set_state(state)
    return best


def _eulers_deg(rotation: np.ndarray) -> List[float]:
    from fibsem.correlation.pyto.rigid_3d import Rigid3D

    return [
        float(a)
        for a in np.degrees(Rigid3D.extract_euler(rotation, mode="x", ret="one"))
    ]


def _mirror_rotation(rotation: np.ndarray) -> np.ndarray:
    """The other branch of a fitted rotation: depth axis reversed, still proper."""
    from fibsem.correlation.geometry import _complete_rotation

    rows = np.asarray(rotation, dtype=float)[:2].copy()
    rows[:, 2] *= -1
    return _complete_rotation(rows)


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
    if nominal is not None and len(fm_coords) >= 3:
        from fibsem.correlation.geometry import branch_of

        r_n, s_n, rms_n = _fit_from_seed(
            fm_coords, fib_coords, nominal.eulers_deg(), nominal.scale
        )
        mirror = nominal.mirrored()
        r_m, s_m, rms_m = _fit_from_seed(
            fm_coords, fib_coords, mirror.eulers_deg(), mirror.scale
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
        run_kwargs["optimiser_params"] = seeded_optimiser_params(
            _eulers_deg(chosen[0]), chosen[1]
        )

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

    # Unseeded: report how well the *other* branch of what was found also fits.
    # A mirror that fits as well is the coin flip FIB-880 describes -- the
    # depth direction of this result is not supported by the fiducials.
    if branch_check is None and len(fm_coords) >= 3:
        try:
            fitted_r = np.asarray(transf["rotation_quaternion"], dtype=float)
            r_m, s_m, rms_m = _fit_from_seed(
                fm_coords,
                fib_coords,
                _eulers_deg(_mirror_rotation(fitted_r)),
                float(transf["scale"]),
            )
            from fibsem.correlation.geometry import rotation_angle_deg

            crossed = rotation_angle_deg(r_m, fitted_r) < rotation_angle_deg(
                r_m, _mirror_rotation(fitted_r)
            )
            rms_here = float(err["rms_error"])
            branch_check = {
                "selected": "unseeded",
                "rms_selected": rms_here,
                "rms_mirror": None if crossed else float(rms_m),
                "angle_to_nominal_deg": None,
                "angle_to_mirror_deg": None,
                "warning": None
                if crossed or abs(rms_m - rms_here) > 1.0
                else "The mirror branch fits the fiducials as well: the depth direction of this fit is a coin flip.",
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
        rotation_quaternion=transf["rotation_quaternion"],
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
