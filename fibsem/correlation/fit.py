"""One correlation fit, without the GUI: the seed, the fit and what it measured.

The correlation tab runs these on its worker thread; anything else that fits a
correlation (a service, a script) calls the same functions, so a fit made
outside the tab is seeded, judged and recorded the way the tab's is.

- :func:`seed_for` builds the transform the fit starts from: a previous run's
  fitted rotation and scale (FIB-956), else the images' geometry (FIB-881),
  placed with a previous run's calibrated offset (FIB-979).
- :func:`fit_transform` runs the fit from that seed and attaches the verdict's
  evidence (FIB-956) and the placement offset this run measured (FIB-979).
- :func:`predict` maps more FM points through a fitted result.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from fibsem.correlation.correlation_v2 import run_correlation_from_data
from fibsem.correlation.geometry import NominalTransform
from fibsem.correlation.history import CorrelationRun
from fibsem.correlation.prediction import independent_pairs
from fibsem.correlation.prior import (
    PriorTransform,
    placement_offset_from_runs,
    prior_from_runs,
)
from fibsem.correlation.structures import (
    CorrelationInputData,
    CorrelationPointOfInterest,
    CorrelationResult,
    PointStatus,
)

__all__ = [
    "MAX_PRIOR_ANGLE_DEG",
    "Seed",
    "diagnostics_for",
    "fit_transform",
    "geometry_transform",
    "measure_placement_offset",
    "predict",
    "prior_transform",
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
    if nominal is None or len(data.fib_coordinates) < 4:
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
        zan = nominal.z_anisotropy
        fib = np.array([[a.point.x, a.point.y] for a, _ in pairs])
        fm = np.array([[b.point.x, b.point.y, b.point.z * zan] for _, b in pairs])
        t_pairs = np.mean(fib - (nominal.projection @ fm.T).T, axis=0)
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


def predict(
    result: CorrelationResult, fm_points: Sequence[Sequence[float]]
) -> List[CorrelationPointOfInterest]:
    """Where FM points land in the FIB image under a fitted ``result``.

    ``fm_points`` are ``(x, y, z)`` in pixels and slices of the stack the
    result was fitted on. The answer is in every coordinate system a fitted POI
    is (image pixels, centred pixels, metres from the image centre). No
    refractive-index correction is applied: these are points, not a target at
    depth.
    """
    from fibsem.correlation.correlation_v2 import _reproject_poi_via_transform

    points = np.asarray(fm_points, dtype=float).reshape(-1, 3).copy()
    points[:, 2] *= result.fm_z_scale
    data = result.input_data
    return _reproject_poi_via_transform(
        {
            "rotation_quaternion": result.rotation_quaternion,
            "scale": result.scale,
            "translation_around_rotation_center_zero": result.translation,
        },
        points,
        data.fib_image_shape if data is not None else None,
        data.fib_image_pixel_size if data is not None else None,
    )
