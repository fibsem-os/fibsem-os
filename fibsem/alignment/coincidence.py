"""SEM/FIB coincidence measurement (FIB-868).

Coincidence error is a height error: the sample surface sitting above or below
the fixed intersection point of the two beam axes. Because the beams are
separated by a tilt about a horizontal axis, that height error projects into
the FIB view along image-y only, while leaving the SEM view stationary. This
module measures that residual by cross-beam correlation and reports it as a
height error in metres, refusing rather than guessing when the measurement
cannot be trusted.

The measurement is a pure image computation: no hardware access, no stage
motion. Correction (vertical_move) is layered on top by callers.

Method: both images are contrast-normalised locally, band-passed (difference
of Gaussians) and reduced to gradient magnitude, so only structure that
survives the SEM->FIB modality change contributes (topography: milled edges,
fiducials, cracks, contamination). The FIB image is stretched along y by the
geometric foreshortening ratio so both views share the SEM projection, then a
windowed cross-correlation finds the residual shift. The estimate is computed
twice with independent band-pass scales and accepted only when the two agree
(a window-size-invariant validity gate; a top-2-peak ratio degenerates when
the search window is tight). A peak on the search-window boundary is refused
outright.

Validated offline on a reference cryo-lamella dataset: sub-micron
repeatability when topographic features are present. See FIB-868 for the
full findings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import fftconvolve

from fibsem.structures import FibsemImage
from fibsem.transformations import convert_stage_tilt_to_milling_angle

DEFAULT_FIB_COLUMN_TILT = np.deg2rad(52.0)

# Independent band-pass scales (sigma pairs, px) for the agreement gate.
AGREEMENT_BANDS: Tuple[Tuple[float, float], Tuple[float, float]] = ((2, 8), (4, 16))
DEFAULT_AGREEMENT_TOLERANCE = 0.75e-6  # m
DEFAULT_CAPTURE_RANGE = 20e-6  # m
WINDOW_EDGE_MARGIN_PX = 3

REFUSAL_BAND_DISAGREEMENT = "band-disagreement"
REFUSAL_WINDOW_EDGE = "window-edge"


@dataclass
class CoincidenceMeasurement:
    """Result of one coincidence measurement.

    Sign convention: dx/dy are where the FIB-view scene sits relative to the
    SEM-view scene, in the SEM image convention (x right, y down), after the
    geometric perspective correction. dz is the inferred height error along
    the chamber-vertical axis (dz = dy / sin(column_tilt)).

    dx is diagnostic only: a height error cannot produce it. A persistent dx
    indicates beam misalignment (correctable by beam shift, see FIB-873),
    scan-rotation mismatch, or a failed SEM centring - never fold it into a
    vertical correction.
    """

    dx: float  # m, lateral residual (diagnostic)
    dy: float  # m, vertical residual in the FIB image plane
    dz: float  # m, inferred height error
    band_disagreement: float  # m, distance between the two band estimates
    is_reliable: bool
    refusal_reason: Optional[str] = None  # set when is_reliable is False
    seeded: bool = False
    method: str = "crossbeam-xcorr"


def fib_view_y_stretch(
    milling_angle: float, column_tilt: float = DEFAULT_FIB_COLUMN_TILT
) -> float:
    """Foreshortening ratio mapping the FIB view onto the SEM projection.

    For a locally flat surface at the given milling angle (FIB glancing
    angle), the two views of the surface differ by a y-only scale of
    sin(column_tilt - milling_angle) / sin(milling_angle).

    Args:
        milling_angle: angle between the FIB beam and the sample surface, in
            RADIANS.
        column_tilt: angle between the SEM and FIB columns, in RADIANS.
    """
    if not 0 < milling_angle < column_tilt:
        raise ValueError(
            f"milling_angle must be in (0, column_tilt) radians, "
            f"got {milling_angle}. (Degrees passed by mistake?)"
        )
    return np.sin(column_tilt - milling_angle) / np.sin(milling_angle)


def height_error_from_fib_shift(
    dy: float, column_tilt: float = DEFAULT_FIB_COLUMN_TILT
) -> float:
    """Convert a FIB-view vertical residual (m) to a height error (m).

    A chamber-vertical displacement dz appears in the FIB view displaced by
    dz * sin(column_tilt) along image-y, and is invisible to a vertical SEM.
    """
    return dy / np.sin(column_tilt)


def _local_normalise(image: np.ndarray, sigma: float = 32) -> np.ndarray:
    mean = ndi.gaussian_filter(image, sigma)
    var = ndi.gaussian_filter((image - mean) ** 2, sigma)
    return (image - mean) / (np.sqrt(var) + 1e-6)


def _preprocess(image: np.ndarray, s1: float, s2: float) -> np.ndarray:
    """Local normalise -> DoG band-pass -> gradient magnitude, zero-mean."""
    x = _local_normalise(image.astype(np.float32))
    x = ndi.gaussian_filter(x, s1) - ndi.gaussian_filter(x, s2)
    x = np.hypot(ndi.sobel(x, axis=1), ndi.sobel(x, axis=0))
    return (x - x.mean()) / (x.std() + 1e-6)


def _stretch_y(image: np.ndarray, scale: float) -> np.ndarray:
    """Scale an image about its centre along y only."""
    h, w = image.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy = h / 2
    src_y = (yy - cy) / scale + cy
    return ndi.map_coordinates(image, [src_y, xx], order=1, mode="constant", cval=0)


def _windowed_xcorr(
    ref: np.ndarray,
    other: np.ndarray,
    center_yx: Tuple[float, float],
    half_yx: Tuple[float, float],
) -> Tuple[Tuple[int, int], bool]:
    """Cross-correlate with a Hann window, peak restricted near center_yx.

    Returns ((dy, dx) in pixels, peak_on_window_edge). The Hann window also
    centre-weights the measurement, so structure near the frame centre (the
    point whose height matters) dominates over peripheral features.
    """
    window = np.outer(np.hanning(ref.shape[0]), np.hanning(ref.shape[1]))
    corr = fftconvolve(ref * window, (other * window)[::-1, ::-1], mode="same")
    cy, cx = np.array(corr.shape) // 2
    oy, ox = int(round(center_yx[0])), int(round(center_yx[1]))
    hy, hx = int(round(half_yx[0])), int(round(half_yx[1]))
    y0, y1 = max(0, cy + oy - hy), min(corr.shape[0], cy + oy + hy + 1)
    x0, x1 = max(0, cx + ox - hx), min(corr.shape[1], cx + ox + hx + 1)
    search = corr[y0:y1, x0:x1]
    py, px = np.unravel_index(np.argmax(search), search.shape)
    on_edge = (
        py < WINDOW_EDGE_MARGIN_PX
        or px < WINDOW_EDGE_MARGIN_PX
        or py >= search.shape[0] - WINDOW_EDGE_MARGIN_PX
        or px >= search.shape[1] - WINDOW_EDGE_MARGIN_PX
    )
    return (py + y0 - cy, px + x0 - cx), on_edge


def measure_coincidence(
    sem: np.ndarray,
    fib: np.ndarray,
    pixel_size: float,
    milling_angle: float,
    column_tilt: float = DEFAULT_FIB_COLUMN_TILT,
    prior: Optional[Tuple[float, float]] = None,
    capture_range: float = DEFAULT_CAPTURE_RANGE,
    agreement_tolerance: float = DEFAULT_AGREEMENT_TOLERANCE,
) -> CoincidenceMeasurement:
    """Measure the SEM/FIB coincidence residual from one image pair.

    Pure image computation; performs no acquisition and no motion.

    Args:
        sem: SEM (electron) image, 2D.
        fib: FIB (ion) image of the same scene, 2D, same shape and pixel size.
        pixel_size: metres per pixel (shared by both images).
        milling_angle: FIB-beam-to-surface angle in RADIANS.
        column_tilt: SEM-to-FIB column angle in RADIANS.
        prior: expected (dx, dy) residual in metres, e.g. the previous
            measurement at this site. Centres the search window.
        capture_range: half-width of the search window in metres. Keep well
            under one grid pitch: periodic structure creates rival
            correlation peaks one pitch apart (FIB-711).
        agreement_tolerance: max distance (m) between the two independent
            band estimates for the result to be reliable.

    Returns:
        CoincidenceMeasurement. When is_reliable is False the residuals are
        the (untrustworthy) mean estimate and refusal_reason says why; act on
        reliable measurements only.
    """
    if sem.shape != fib.shape:
        raise ValueError(f"image shapes differ: {sem.shape} vs {fib.shape}")

    stretch = fib_view_y_stretch(milling_angle, column_tilt)
    fib_stretched = _stretch_y(fib.astype(np.float32), stretch)

    if prior is not None:
        center = (prior[1] / pixel_size * stretch, prior[0] / pixel_size)
    else:
        center = (0.0, 0.0)
    half_px = capture_range / pixel_size
    half_yx = (half_px * stretch, half_px)

    shifts = []
    on_edge_any = False
    for s1, s2 in AGREEMENT_BANDS:
        ref = _preprocess(sem, s1, s2)
        other = _preprocess(fib_stretched, s1, s2)
        (dy_px, dx_px), on_edge = _windowed_xcorr(ref, other, center, half_yx)
        shifts.append((dy_px, dx_px))
        on_edge_any = on_edge_any or on_edge

    (dy_a, dx_a), (dy_b, dx_b) = shifts
    disagreement = np.hypot((dy_a - dy_b) / stretch, dx_a - dx_b) * pixel_size
    dy = ((dy_a + dy_b) / 2 / stretch) * pixel_size
    dx = ((dx_a + dx_b) / 2) * pixel_size
    dz = height_error_from_fib_shift(dy, column_tilt)

    refusal: Optional[str] = None
    if on_edge_any:
        refusal = REFUSAL_WINDOW_EDGE
    elif disagreement > agreement_tolerance:
        refusal = REFUSAL_BAND_DISAGREEMENT

    measurement = CoincidenceMeasurement(
        dx=float(dx),
        dy=float(dy),
        dz=float(dz),
        band_disagreement=float(disagreement),
        is_reliable=refusal is None,
        refusal_reason=refusal,
        seeded=prior is not None,
    )
    logging.debug(
        {
            "msg": "measure_coincidence",
            "dx": measurement.dx,
            "dy": measurement.dy,
            "dz": measurement.dz,
            "band_disagreement": measurement.band_disagreement,
            "is_reliable": measurement.is_reliable,
            "refusal_reason": measurement.refusal_reason,
            "seeded": measurement.seeded,
        }
    )
    return measurement


def geometry_from_metadata(image: FibsemImage) -> Tuple[float, float, float]:
    """Extract (pixel_size, milling_angle, column_tilt) from image metadata.

    Pixel size is per-image (hfw varies between workflow stages - never share
    one value across a dataset). The milling angle is derived from the
    recorded stage tilt plus the hardware geometry (shuttle pre-tilt, ion
    column tilt), so a saved pair is self-sufficient - usable offline with no
    microscope connection.
    """
    md = image.metadata
    pixel_size = md.pixel_size.x
    column_tilt = np.deg2rad(md.hardware_geometry.fib_column_tilt)
    pretilt = np.deg2rad(md.hardware_geometry.shuttle_pre_tilt)
    stage_tilt = md.microscope_state.stage_position.t
    milling_angle = convert_stage_tilt_to_milling_angle(
        stage_tilt=stage_tilt, pretilt=pretilt, column_tilt=column_tilt
    )
    return pixel_size, milling_angle, column_tilt


def measure_coincidence_from_images(
    sem_image: FibsemImage,
    fib_image: FibsemImage,
    milling_angle: Optional[float] = None,
    column_tilt: Optional[float] = None,
    prior: Optional[Tuple[float, float]] = None,
    capture_range: float = DEFAULT_CAPTURE_RANGE,
    agreement_tolerance: float = DEFAULT_AGREEMENT_TOLERANCE,
) -> CoincidenceMeasurement:
    """measure_coincidence for a FibsemImage pair.

    Pixel size, milling angle, and column tilt default to values derived from
    the SEM image's metadata (see geometry_from_metadata); pass milling_angle
    or column_tilt (RADIANS) only to override the recorded geometry.
    """
    pixel_size, md_milling_angle, md_column_tilt = geometry_from_metadata(sem_image)
    return measure_coincidence(
        sem=sem_image.data,
        fib=fib_image.data,
        pixel_size=pixel_size,
        milling_angle=md_milling_angle if milling_angle is None else milling_angle,
        column_tilt=md_column_tilt if column_tilt is None else column_tilt,
        prior=prior,
        capture_range=capture_range,
        agreement_tolerance=agreement_tolerance,
    )
