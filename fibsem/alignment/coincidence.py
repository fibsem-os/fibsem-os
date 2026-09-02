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
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import fftconvolve

from fibsem.structures import FibsemImage, FibsemStagePosition

if TYPE_CHECKING:
    from fibsem.microscope import FibsemMicroscope
    from fibsem.structures import ImageSettings

DEFAULT_FIB_COLUMN_TILT = np.deg2rad(52.0)

# Independent band-pass scales (sigma pairs, px) for the agreement gate.
AGREEMENT_BANDS: Tuple[Tuple[float, float], Tuple[float, float]] = ((2, 8), (4, 16))
DEFAULT_AGREEMENT_TOLERANCE = 0.75e-6  # m
DEFAULT_CAPTURE_RANGE = 20e-6  # m
WINDOW_EDGE_MARGIN_PX = 3

REFUSAL_BAND_DISAGREEMENT = "band-disagreement"
REFUSAL_WINDOW_EDGE = "window-edge"
REFUSAL_LATERAL_OFFSET = "lateral-offset"
# A height error cannot produce dx, and a real beam misalignment is a few
# microns at most (FIB-873 measured ~1.3 um) - a lock further out in x is a
# rival peak, and both bands agreeing on it does not make it right. On the
# simulator's mesh, rival locks at 9 um passed a 10 um bound.
DEFAULT_MAX_LATERAL_OFFSET = 5e-6  # m


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
    coarse: bool = False  # taken at the wide coarse field of view
    # the FIB->SEM y stretch the pair was measured with: dy is in the FIB
    # plane, dy * y_stretch is the same displacement seen in the SEM plane
    y_stretch: float = 1.0
    # what the measurement was run with, so a saved pair can be replayed
    # with the same inputs
    capture_range: float = DEFAULT_CAPTURE_RANGE
    agreement_tolerance: float = DEFAULT_AGREEMENT_TOLERANCE
    max_lateral_offset: float = DEFAULT_MAX_LATERAL_OFFSET
    prior: Optional[Tuple[float, float]] = None
    # the pair this was measured from, kept for diagnostics (check_coincidence
    # fills them in; the pure array path leaves them None)
    sem_image: Optional[FibsemImage] = field(default=None, repr=False)
    fib_image: Optional[FibsemImage] = field(default=None, repr=False)


@dataclass
class CoincidenceGeometry:
    """The three numbers the measurement needs, derived per image pair.

    Derived from BeamStageProjection - the calibrated stage/view model the
    app's movement, overview and reprojection code share - rather than from
    standalone trigonometry. The earlier hand formulas (a sin ratio for the
    stretch, sin(column_tilt) for dz) disagreed with the calibrated model at
    tilt: stage axes tilt with the stage, and each beam has its own
    foreshortening.
    """

    pixel_size: float  # m/px, shared by the pair
    y_stretch: float  # multiply FIB-view y by this to reach the SEM projection
    dz_per_dy: float  # stage-z error per metre of measured FIB-plane dy


_Z_PROBE = 1.0e-6  # m; the projections are linear, any probe length works


def geometry_from_images(
    sem_image: FibsemImage, fib_image: FibsemImage
) -> CoincidenceGeometry:
    """Derive the measurement geometry from an image pair's own metadata.

    Pixel size is per-image (hfw varies between workflow stages - never share
    one value across a dataset). The stretch is the ratio of the two beams'
    surface foreshortenings, and dz_per_dy inverts how a stage-z move
    projects into the two views (it moves both: the FIB view by its view
    tilt, and the SEM view by the sin(tilt) stage-axis leak), all read from
    the projections the images record - so a saved pair is self-sufficient,
    usable offline with no microscope connection.

    A pair taken nearer the flipped (rotation_180 / FIB-orientation) side
    than the reference side is measured - the projections carry the
    half-turn - but logged: validated on the simulator only, where the loop
    converges exactly at the FIB orientation, and not yet on hardware.

    Raises:
        ValueError: when the pair's metadata cannot supply the projections.
    """
    from copy import deepcopy

    from fibsem.movement import angle_difference
    from fibsem.projection import BeamStageProjection, surface_foreshortening

    md = sem_image.metadata
    pixel_size = md.pixel_size.x
    stage_position = md.microscope_state.stage_position

    stage_rotation = stage_position.r or 0.0
    rotation_reference = np.deg2rad(md.hardware_geometry.rotation_reference)
    rotation_180 = np.deg2rad(md.hardware_geometry.rotation_180)
    if angle_difference(stage_rotation, rotation_180) < angle_difference(
        stage_rotation, rotation_reference
    ):
        logging.warning(
            "Coincidence measurement on the flipped (FIB-orientation) side: "
            "validated on the simulator only, not yet on hardware (FIB-868)."
        )

    sem_projection = BeamStageProjection.from_image(sem_image)
    fib_projection = BeamStageProjection.from_image(fib_image)
    if sem_projection is None or fib_projection is None:
        raise ValueError(
            "Image metadata does not carry the beam geometry needed to "
            "derive the coincidence measurement projections."
        )

    fs_sem = surface_foreshortening(sem_projection, stage_position)
    fs_fib = surface_foreshortening(fib_projection, stage_position)
    y_stretch = fs_sem / fs_fib

    probed = deepcopy(stage_position)
    probed.z = (probed.z or 0.0) + _Z_PROBE
    k_fib = fib_projection.to_plane(probed, stage_position)[1] / _Z_PROBE
    k_sem = sem_projection.to_plane(probed, stage_position)[1] / _Z_PROBE
    # the measured dy is FIB-plane displacement RELATIVE to the SEM view;
    # a stage-z error moves both views, the SEM part seen through the stretch
    k_relative = k_fib - k_sem / y_stretch
    if abs(k_relative) < 1e-6:
        raise ValueError(
            "Degenerate pose: a stage-z move is invisible to the pair, so "
            "no height error can be inferred here."
        )
    return CoincidenceGeometry(
        pixel_size=pixel_size, y_stretch=y_stretch, dz_per_dy=1.0 / k_relative
    )


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
    y_stretch: float,
    dz_per_dy: float,
    prior: Optional[Tuple[float, float]] = None,
    capture_range: float = DEFAULT_CAPTURE_RANGE,
    agreement_tolerance: float = DEFAULT_AGREEMENT_TOLERANCE,
    max_lateral_offset: float = DEFAULT_MAX_LATERAL_OFFSET,
) -> CoincidenceMeasurement:
    """Measure the SEM/FIB coincidence residual from one image pair.

    Pure image computation; performs no acquisition and no motion.

    Args:
        sem: SEM (electron) image, 2D.
        fib: FIB (ion) image of the same scene, 2D, same shape and pixel size.
        pixel_size: metres per pixel (shared by both images).
        y_stretch: foreshortening ratio mapping FIB-view y onto the SEM
            projection (see geometry_from_images).
        dz_per_dy: stage-z error per metre of measured dy (see
            geometry_from_images).
        prior: expected (dx, dy) residual in metres, e.g. the previous
            measurement at this site. Centres the search window.
        capture_range: half-width of the search window in metres. Keep well
            under one grid pitch: periodic structure creates rival
            correlation peaks one pitch apart (FIB-711).
        agreement_tolerance: max distance (m) between the two independent
            band estimates for the result to be reliable.
        max_lateral_offset: |dx| (m) beyond which a lock is refused as a
            rival peak - a height error cannot move x.

    Returns:
        CoincidenceMeasurement. When is_reliable is False the residuals are
        the (untrustworthy) mean estimate and refusal_reason says why; act on
        reliable measurements only.
    """
    if sem.shape != fib.shape:
        raise ValueError(f"image shapes differ: {sem.shape} vs {fib.shape}")

    stretch = y_stretch
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
    dz = dy * dz_per_dy

    refusal: Optional[str] = None
    if on_edge_any:
        refusal = REFUSAL_WINDOW_EDGE
    elif disagreement > agreement_tolerance:
        refusal = REFUSAL_BAND_DISAGREEMENT
    elif abs(dx) > max_lateral_offset:
        refusal = REFUSAL_LATERAL_OFFSET

    measurement = CoincidenceMeasurement(
        dx=float(dx),
        dy=float(dy),
        y_stretch=float(stretch),
        dz=float(dz),
        band_disagreement=float(disagreement),
        is_reliable=refusal is None,
        refusal_reason=refusal,
        seeded=prior is not None,
        capture_range=capture_range,
        agreement_tolerance=agreement_tolerance,
        max_lateral_offset=max_lateral_offset,
        prior=None if prior is None else (float(prior[0]), float(prior[1])),
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


def measure_coincidence_from_images(
    sem_image: FibsemImage,
    fib_image: FibsemImage,
    geometry: Optional[CoincidenceGeometry] = None,
    prior: Optional[Tuple[float, float]] = None,
    capture_range: float = DEFAULT_CAPTURE_RANGE,
    agreement_tolerance: float = DEFAULT_AGREEMENT_TOLERANCE,
    max_lateral_offset: float = DEFAULT_MAX_LATERAL_OFFSET,
) -> CoincidenceMeasurement:
    """measure_coincidence for a FibsemImage pair.

    The geometry defaults to values derived from the pair's own metadata
    (see geometry_from_images); pass one explicitly only to override the
    recorded geometry.
    """
    if geometry is None:
        geometry = geometry_from_images(sem_image, fib_image)
    return measure_coincidence(
        sem=sem_image.data,
        fib=fib_image.data,
        pixel_size=geometry.pixel_size,
        y_stretch=geometry.y_stretch,
        dz_per_dy=geometry.dz_per_dy,
        prior=prior,
        capture_range=capture_range,
        agreement_tolerance=agreement_tolerance,
        max_lateral_offset=max_lateral_offset,
    )


DEFAULT_ALIGNMENT_HFW = 150e-6  # m
DEFAULT_ALIGNMENT_RESOLUTION = (1536, 1024)
DEFAULT_TOLERANCE = 1e-6  # m, height error below which the pair is coincident
DEFAULT_MAX_ITERATIONS = 3

# The coarse (establish) pass: a wide view buys capture range at the cost of
# precision. Its search window must stay well under one grid pitch - the mesh
# is periodic, and rival correlation peaks sit exactly one pitch apart
# (FIB-711 by construction).
# Sized for height errors up to ~100 um: the FIB-view displacement of a
# height error, stretched into the correlation space, must still fit the
# frame - and the search window must stay under one grid pitch (125 um),
# where the mesh's rival correlation peaks sit.
DEFAULT_COARSE_HFW = 900e-6  # m
DEFAULT_COARSE_CAPTURE_RANGE = 100e-6  # m, just under one 125 um grid pitch
DEFAULT_COARSE_AGREEMENT_TOLERANCE = 2e-6  # m, looser: coarse pixels are ~4x bigger
DEFAULT_COARSE_MAX_LATERAL_OFFSET = 20e-6  # m

REASON_CONVERGED = "converged"
REASON_MAX_ITERATIONS = "max-iterations"


@dataclass
class CoincidenceAlignment:
    """The full history of one ensure_coincident run.

    `converged` is only ever set from a fresh measurement taken AFTER the
    last correction - success is measured, never assumed from having moved
    (FIB-809 philosophy). When a measurement refuses, the loop stops and
    `reason` carries the refusal; the stage is left where the last reliable
    correction put it.
    """

    measurements: List["CoincidenceMeasurement"]
    converged: bool
    reason: str  # REASON_CONVERGED, REASON_MAX_ITERATIONS, or a refusal_reason
    coarse_used: bool = False
    # counted, not derived from the history: a coarse measurement adds an
    # entry without a move
    moves_applied: int = 0

    @property
    def final(self) -> "CoincidenceMeasurement":
        return self.measurements[-1]


PROGRESS_MEASURING = "measuring"
PROGRESS_MEASURED = "measured"
PROGRESS_MOVING = "moving"


@dataclass
class CoincidenceProgress:
    """One step of an ensure_coincident run, as reported to `on_progress`.

    Emitted before each acquisition (MEASURING), after each measurement
    (MEASURED, with the measurement) and before each corrective move
    (MOVING, with the measurement driving it). `iteration` counts the
    corrective moves applied so far.
    """

    stage: str
    iteration: int
    max_iterations: int
    coarse: bool = False
    measurement: Optional[CoincidenceMeasurement] = None

    def describe(self) -> str:
        """A one-line, operator-facing description of the step."""
        pass_name = "coarse" if self.coarse else "fine"
        if self.stage == PROGRESS_MEASURING:
            if self.coarse:
                return "Fine pass refused - measuring at the coarse field of view..."
            return f"Measuring coincidence ({pass_name}, {self.iteration}/{self.max_iterations} moves so far)..."
        m = self.measurement
        assert m is not None
        if self.stage == PROGRESS_MEASURED:
            if not m.is_reliable:
                return (
                    f"{pass_name.capitalize()} measurement refused: "
                    f"{m.refusal_reason} (band disagreement {m.band_disagreement * 1e6:.2f} um)."
                )
            return (
                f"{pass_name.capitalize()} measurement: dz {m.dz * 1e6:+.2f} um "
                f"(dx {m.dx * 1e6:+.2f} um)."
            )
        return (
            f"Correcting height by {m.dz * 1e6:+.2f} um "
            f"(move {self.iteration + 1}/{self.max_iterations})..."
        )


ProgressCallback = Callable[[CoincidenceProgress], None]


def _default_image_settings() -> "ImageSettings":
    from fibsem.structures import ImageSettings

    return ImageSettings(
        hfw=DEFAULT_ALIGNMENT_HFW,
        resolution=DEFAULT_ALIGNMENT_RESOLUTION,
        dwell_time=0.2e-6,  # keep the ion dose down: this runs repeatedly
        autocontrast=False,
        save=False,
    )


def check_coincidence(
    microscope: "FibsemMicroscope",
    image_settings: Optional["ImageSettings"] = None,
    prior: Optional[Tuple[float, float]] = None,
    capture_range: float = DEFAULT_CAPTURE_RANGE,
    agreement_tolerance: float = DEFAULT_AGREEMENT_TOLERANCE,
    max_lateral_offset: float = DEFAULT_MAX_LATERAL_OFFSET,
) -> CoincidenceMeasurement:
    """Acquire an eb/ib pair at the current position and measure coincidence.

    No stage motion. The geometry comes from the acquired images' own
    metadata, so this works identically live and when replaying saved pairs.
    """
    from copy import deepcopy

    from fibsem.structures import BeamType

    settings = deepcopy(image_settings or _default_image_settings())
    settings.beam_type = BeamType.ELECTRON
    sem_image = microscope.acquire_image(image_settings=settings)
    settings.beam_type = BeamType.ION
    fib_image = microscope.acquire_image(image_settings=settings)
    measurement = measure_coincidence_from_images(
        sem_image,
        fib_image,
        prior=prior,
        capture_range=capture_range,
        agreement_tolerance=agreement_tolerance,
        max_lateral_offset=max_lateral_offset,
    )
    measurement.sem_image = sem_image
    measurement.fib_image = fib_image
    return measurement


def ensure_coincident(
    microscope: "FibsemMicroscope",
    tolerance: float = DEFAULT_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    image_settings: Optional["ImageSettings"] = None,
    capture_range: float = DEFAULT_CAPTURE_RANGE,
    agreement_tolerance: float = DEFAULT_AGREEMENT_TOLERANCE,
    relaxation: float = 1.0,
    coarse_hfw: Optional[float] = DEFAULT_COARSE_HFW,
    coarse_capture_range: float = DEFAULT_COARSE_CAPTURE_RANGE,
    coarse_agreement_tolerance: float = DEFAULT_COARSE_AGREEMENT_TOLERANCE,
    coarse_max_lateral_offset: float = DEFAULT_COARSE_MAX_LATERAL_OFFSET,
    on_progress: Optional[ProgressCallback] = None,
    reference: "BeamType" = None,
) -> CoincidenceAlignment:
    """Measure the SEM/FIB coincidence and correct it until within tolerance.

    `reference` says which view keeps its centre. ELECTRON (the default):
    the height correction moves the stage along the SEM axis, so what sits
    at the SEM centre stays there and the FIB view slides onto it. ION: what
    sits at the FIB centre stays there - the case in the workflow, where the
    operator chose the site in the FIB view - achieved by first recentring
    the SEM on the FIB-centred feature with a stable move, then applying the
    same height correction.

    The loop: measure -> vertical move by the measured residual -> re-measure,
    stopping when a FRESH measurement is within tolerance, an iteration limit
    is reached, or the measurement chain refuses. It never reports success
    from having moved: convergence is always a measured fact.

    When the fine measurement refuses - typically because the error is
    outside its capture range - the loop escalates ONCE per refusal to a
    coarse pass: the same measurement at a much wider field of view, whose
    window buys capture range at the cost of precision. A reliable coarse
    measurement drives a corrective move that only needs to land within the
    fine pass's reach; the next fine measurement takes over. If the coarse
    pass refuses too, the loop stops without moving - the escalation beyond
    this point (spot burn, operator) is a policy decision that does not
    belong in here.

    dx is never corrected here - a persistent lateral offset is not a height
    error (beam misalignment, FIB-873) and no stage move can change it: the
    stage carries both views together. It rides along as a diagnostic, and a
    large one refuses the measurement outright (a rival peak, not an offset).

    Args:
        microscope: the microscope connection.
        tolerance: height error (m) below which the views count as coincident.
        max_iterations: maximum number of corrective moves.
        image_settings: per-acquisition settings; a low-dose default when omitted.
        capture_range: fine search half-width (m), centred on zero every
            iteration - never re-seeded at a previous lock.
        agreement_tolerance: band-agreement gate for fine measurements.
        relaxation: under-relaxation passed to vertical_move; 1.0 is exact.
        coarse_hfw: field width (m) for the coarse escalation; None disables
            it, making a fine refusal terminal.
        coarse_capture_range: coarse search half-width (m). Keep well under
            one grid pitch: the mesh is periodic and rival peaks sit one
            pitch apart.
        coarse_agreement_tolerance: band-agreement gate for the coarse pass
            (looser: its pixels are ~4x larger).
        coarse_max_lateral_offset: |dx| bound for the coarse pass (looser:
            it only has to land within the fine pass's reach, and the fine
            measurement that follows re-verifies under the strict bound).
        on_progress: called with a CoincidenceProgress before every
            acquisition, after every measurement and before every move, from
            the calling thread - a GUI must marshal it across itself.
        reference: the view whose centre is preserved (ELECTRON or ION).

    Returns:
        CoincidenceAlignment with the full measurement history (coarse
        measurements included) and whether the coarse pass was needed.
    """
    from copy import deepcopy

    from fibsem.structures import BeamType

    if reference is None:
        reference = BeamType.ELECTRON
    if reference not in (BeamType.ELECTRON, BeamType.ION):
        raise ValueError(f"reference must be ELECTRON or ION, got {reference}")

    measurements: List[CoincidenceMeasurement] = []
    coarse_used = False
    moves = 0

    def report(stage: str, coarse: bool = False, measurement=None) -> None:
        if on_progress is None:
            return
        on_progress(
            CoincidenceProgress(
                stage=stage,
                iteration=moves,
                max_iterations=max_iterations,
                coarse=coarse,
                measurement=measurement,
            )
        )

    def fine_check(prior: Optional[Tuple[float, float]] = None):
        report(PROGRESS_MEASURING)
        measurement = check_coincidence(
            microscope,
            image_settings=image_settings,
            prior=prior,
            capture_range=capture_range,
            agreement_tolerance=agreement_tolerance,
        )
        measurements.append(measurement)
        report(PROGRESS_MEASURED, measurement=measurement)
        return measurement

    def coarse_check():
        report(PROGRESS_MEASURING, coarse=True)
        settings = deepcopy(image_settings or _default_image_settings())
        settings.hfw = coarse_hfw
        measurement = check_coincidence(
            microscope,
            image_settings=settings,
            capture_range=coarse_capture_range,
            agreement_tolerance=coarse_agreement_tolerance,
            max_lateral_offset=coarse_max_lateral_offset,
        )
        measurement.coarse = True
        measurements.append(measurement)
        report(PROGRESS_MEASURED, coarse=True, measurement=measurement)
        return measurement

    measurement = fine_check()

    reason = REASON_MAX_ITERATIONS
    for _ in range(max_iterations):
        if measurement.is_reliable and abs(measurement.dz) <= tolerance:
            reason = REASON_CONVERGED
            break

        if not measurement.is_reliable:
            if coarse_hfw is None:
                reason = measurement.refusal_reason or REFUSAL_BAND_DISAGREEMENT
                break
            coarse_used = True
            measurement = coarse_check()
            if not measurement.is_reliable:
                reason = measurement.refusal_reason or REFUSAL_BAND_DISAGREEMENT
                break
            # a coarse move only needs to land within the fine pass's reach

        report(PROGRESS_MOVING, coarse=measurement.coarse, measurement=measurement)
        if reference is BeamType.ION:
            # bring the FIB-centred feature to the SEM centre first: the
            # height correction preserves the SEM centre, so this is what
            # leaves the FIB view where the operator put it. A stable move
            # follows the surface and leaves the height error unchanged
            # y only: a stage move shifts both views together, so it cannot
            # change dx - chasing dx walked the stage sideways on a false lock
            microscope.stable_move(
                dx=0.0,
                dy=-measurement.dy * measurement.y_stretch,
                beam_type=BeamType.ELECTRON,
            )
        microscope.vertical_move(dy=measurement.dy, dx=0, relaxation=relaxation)
        moves += 1

        # after a correction the residual should be near zero; the window is
        # NOT re-seeded at the previous dx - that made a rival peak
        # self-confirming (the window followed the alias)
        measurement = fine_check()
    else:
        # loop exhausted: the final measurement still decides the verdict
        if measurement.is_reliable and abs(measurement.dz) <= tolerance:
            reason = REASON_CONVERGED
        elif not measurement.is_reliable:
            reason = measurement.refusal_reason or REFUSAL_BAND_DISAGREEMENT

    result = CoincidenceAlignment(
        measurements=measurements,
        converged=reason == REASON_CONVERGED,
        reason=reason,
        coarse_used=coarse_used,
        moves_applied=moves,
    )
    logging.info(
        {
            "msg": "ensure_coincident",
            "converged": result.converged,
            "reason": result.reason,
            "moves_applied": result.moves_applied,
            "coarse_used": result.coarse_used,
            "final_dz": result.final.dz,
            "final_dx": result.final.dx,
            "tolerance": tolerance,
        }
    )
    return result


@dataclass
class TiltAlignment:
    """The record of one tilt_coincident run: every tilt visited, in order,
    with the coincidence alignment run there."""

    tilts: List[float]  # rad
    alignments: List["CoincidenceAlignment"]
    converged: bool  # coincident AT the target tilt
    reason: str
    # the height of the surface above the tilt axis, estimated from the sag
    # each tilt segment cost (None when no segment could measure it)
    tilt_axis_offset: Optional[float] = None
    # the surface walk h * sin(dt) the tilt produced, and whether it was
    # undone (a stable move back to the patch that was centred before)
    walk: float = 0.0
    walk_undone: bool = False

    @property
    def moves_applied(self) -> int:
        return sum(a.moves_applied for a in self.alignments)


# a segment must change cos(t - apex) by at least this much to resolve h:
# below it the height lever is too short and noise dominates (~2 deg near apex)
MIN_HEIGHT_LEVER = 6e-4
MIN_WALK_TO_UNDO = 0.2e-6  # m


def tilt_swing(
    tilt_axis_offset: float,
    tilt_from: float,
    tilt_to: float,
    apex: float,
    pretilt: float,
) -> Tuple[float, float]:
    """Where a tilt from `tilt_from` to `tilt_to` carries a surface point
    sitting `tilt_axis_offset` above the tilt axis, as a (dy, dz)
    displacement in stage axes.

    The eucentric-height model, anchored at a physical `apex`: the tilt at
    which the offset vector from axis to surface is vertical - the SEM
    orientation, where the surface normal points up the column. Relative to
    it the point sits at height h * cos(t - apex) and lateral h * sin(t -
    apex); a segment changes those by the difference. Anchoring at an apex
    rather than at "wherever the segment started" is what keeps h's sign
    right on the way back: after a correction at the milling angle, the
    return tilt sees the opposite height change, and a start-relative
    h * (1 - cos dt) read that as a negative h and undid the walk backwards.

    The walk follows the stable-move direction (cos p, -sin p) in stage y/z
    for corrected pre-tilt p - not the stage y axis, which leaves the surface
    plane on a pre-tilted shuttle; the height goes on the stage z axis,
    where a coincidence height error lives. Shared with the simulator's
    scene, which renders exactly this.
    """
    walk = tilt_axis_offset * (np.sin(tilt_to - apex) - np.sin(tilt_from - apex))
    rise = tilt_axis_offset * (np.cos(tilt_to - apex) - np.cos(tilt_from - apex))
    return float(walk * np.cos(pretilt)), float(-walk * np.sin(pretilt) + rise)


def tilt_axis_offset_from_height_change(
    dz: float, tilt_from: float, tilt_to: float, apex: float
) -> Optional[float]:
    """h from the height error a tilt segment produced (dz = the z move that
    cancels it, so the surface moved by -dz). None when the segment is too
    flat to tell."""
    dcos = np.cos(tilt_to - apex) - np.cos(tilt_from - apex)
    if abs(dcos) < MIN_HEIGHT_LEVER:
        return None
    return float(-dz / dcos)


DEFAULT_MAX_TILT_SPLITS = 2


def tilt_coincident(
    microscope: "FibsemMicroscope",
    target_tilt: float,
    reference: "BeamType" = None,
    max_splits: int = DEFAULT_MAX_TILT_SPLITS,
    on_progress: Optional[ProgressCallback] = None,
    undo_walk: bool = True,
    **ensure_kwargs,
) -> TiltAlignment:
    """Tilt the stage to `target_tilt` (rad) and restore coincidence there.

    A stage is rarely eucentric: the surface sits some height h off the tilt
    axis, so tilting swings it - mostly a walk along the surface, plus a
    height change that costs coincidence. Both are small relative to the
    measurement's reach (~h * sin(dt): even 100 um over 23 deg lands inside
    the coarse window), so the loop tilts straight to the target and measures
    there. Stepping is the fallback, not the cadence: when the alignment at a
    tilt refuses, the segment from the last coincident tilt is halved - align
    at the midpoint, then try the target again - at most `max_splits` times.
    Nothing here needs a reference image from the starting tilt, which is
    what made the stepped-and-correlate predecessor drift.

    `reference` is passed through to ensure_coincident; ION (the default
    here, unlike ensure_coincident's) keeps what the operator centred in the
    FIB view.

    Coincidence is not identity: the same swing that costs the height also
    walks the surface under the beams by h * sin(dt), so a different patch
    is centred afterwards. The sag the alignment measures gives h directly
    (sag = h * (1 - cos dt), assuming the stage was coincident before the
    tilt), so the walk is known without any image tracking, and with
    `undo_walk` one stable move at the end brings the original patch back
    under the crosshair. The estimate is reported as `tilt_axis_offset`.

    Args:
        microscope: the microscope connection.
        target_tilt: stage tilt to end at, in radians (absolute).
        reference: which view keeps its centre (see ensure_coincident).
        max_splits: how many times a refused segment may be halved.
        on_progress: forwarded to every ensure_coincident call.
        undo_walk: after converging at the target, stable-move back by the
            walk the tilt produced, so what was centred before stays centred.
        **ensure_kwargs: forwarded to ensure_coincident (tolerance, ranges...).

    Returns:
        TiltAlignment; `converged` means the LAST alignment, at the target
        tilt, measured coincident.
    """
    from fibsem.structures import BeamType

    if reference is None:
        reference = BeamType.ION

    start_pose = microscope.get_stage_position()
    start_tilt = float(start_pose.t or 0.0)
    last_coincident_tilt = start_tilt
    apex = _tilt_apex(microscope)
    pending: List[float] = [float(target_tilt)]  # target stays at the bottom
    tilts: List[float] = []
    alignments: List[CoincidenceAlignment] = []
    offset_estimates: List[float] = []
    splits = 0

    while pending:
        goal = pending[-1]
        pose = microscope.get_stage_position()
        pose.t = goal
        microscope.move_stage_absolute(pose)

        result = ensure_coincident(
            microscope, reference=reference, on_progress=on_progress, **ensure_kwargs
        )
        tilts.append(goal)
        alignments.append(result)

        if result.converged:
            estimate = _segment_offset_estimate(
                result, last_coincident_tilt, goal, apex
            )
            if estimate is not None:
                offset_estimates.append(estimate)
            pending.pop()
            last_coincident_tilt = goal
            continue
        if result.reason == REASON_MAX_ITERATIONS or splits >= max_splits:
            # not a refusal we can shrink our way out of, or out of splits
            break
        splits += 1
        pending.append((last_coincident_tilt + goal) / 2)

    converged = bool(alignments) and not pending and alignments[-1].converged
    reason = REASON_CONVERGED if converged else alignments[-1].reason

    tilt_axis_offset = float(np.mean(offset_estimates)) if offset_estimates else None
    walk = 0.0
    walk_undone = False
    if converged and tilt_axis_offset is not None:
        walk = tilt_axis_offset * (
            np.sin(float(target_tilt) - apex) - np.sin(start_tilt - apex)
        )
        if undo_walk and abs(walk) > MIN_WALK_TO_UNDO:
            _undo_surface_walk(microscope, start_pose, tilt_axis_offset)
            walk_undone = True
    logging.info(
        {
            "msg": "tilt_coincident",
            "target_tilt": target_tilt,
            "tilts": tilts,
            "converged": converged,
            "reason": reason,
            "splits": splits,
            "tilt_axis_offset": tilt_axis_offset,
            "walk": walk,
            "walk_undone": walk_undone,
        }
    )
    return TiltAlignment(
        tilts=tilts,
        alignments=alignments,
        converged=converged,
        reason=reason,
        tilt_axis_offset=tilt_axis_offset,
        walk=walk,
        walk_undone=walk_undone,
    )


def _tilt_apex(microscope: "FibsemMicroscope") -> float:
    """The tilt at which the surface normal is vertical: the SEM orientation."""
    return float(microscope.get_orientation("SEM").t or 0.0)


def _segment_offset_estimate(
    alignment: CoincidenceAlignment, tilt_from: float, tilt_to: float, apex: float
) -> Optional[float]:
    """h from one tilt segment: the first reliable measurement at the new
    tilt is the whole height error the segment produced (the stage was
    coincident before it)."""
    first = next((m for m in alignment.measurements if m.is_reliable), None)
    if first is None:
        return None
    return tilt_axis_offset_from_height_change(first.dz, tilt_from, tilt_to, apex)


def _undo_surface_walk(
    microscope: "FibsemMicroscope",
    start_pose: FibsemStagePosition,
    tilt_axis_offset: float,
) -> None:
    """Bring the patch that was centred before the tilt back to the centre.

    The h-model says where that patch now sits in stage coordinates (the
    start pose carried by the swing); projecting it into the SEM view at the
    ACTUAL current pose - after every correction the alignment made - gives
    the displacement to undo, with no image tracking and no bookkeeping of
    the individual moves. One stable move follows the surface, so it does
    not cost the coincidence just restored.
    """
    from fibsem.projection import BeamStageProjection
    from fibsem.structures import BeamType
    from fibsem.transformations import _projection_terms

    projection = BeamStageProjection.from_microscope(microscope, BeamType.ELECTRON)
    if projection is None:
        logging.warning("Cannot undo the tilt walk: no SEM projection available")
        return
    now = microscope.get_stage_position()
    _, pretilt, _ = _projection_terms(projection.geometry, now.r or 0.0, now.t or 0.0)
    dy, dz = tilt_swing(
        tilt_axis_offset,
        float(start_pose.t or 0.0),
        float(now.t or 0.0),
        _tilt_apex(microscope),
        pretilt,
    )
    patch = deepcopy(start_pose)
    patch.y = (patch.y or 0.0) + dy
    patch.z = (patch.z or 0.0) + dz
    ax, ay = projection.to_plane(patch, now)
    microscope.stable_move(dx=-ax, dy=-ay, beam_type=BeamType.ELECTRON)
