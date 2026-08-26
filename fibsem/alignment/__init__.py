from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from threading import Event as ThreadingEvent
from typing import TYPE_CHECKING, Optional

import numpy as np

from fibsem import acquire, utils
from fibsem.structures import (
    BeamType,
    FibsemImage,
    ImageSettings,
    Point,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from fibsem.microscope import FibsemMicroscope


from fibsem.alignment.methods import (
    crosscorrelation_v2,
    shift_from_crosscorrelation,
    shift_from_skimage_phase_correlation,
)
from fibsem.alignment.plotting import (
    _alignment_save_path,
    plot_multi_step_alignment,
)
from fibsem.alignment.verified import (
    AlignmentContext,
    DEFAULT_ALIGNMENT_CONTEXT,
    VALIDATION_MODES,
    DEFAULT_VALIDATION_MODE,
    ValidationMode,
    ValidatedAlignment,
    AlignmentFailedError,
    correlation_psr,
    get_validation_mode,
    get_configured_validation,
    get_configured_clip_fraction,
    get_configured_convergence_tolerance,
    DEFAULT_CONVERGENCE_TOLERANCE_PX,
    should_abort_on_failure,
    shift_from_crosscorrelation_validated,
)

ALIGNMENT_SUBDIR = "Alignment"


def _nan_to_none(value: float) -> Optional[float]:
    """NaN -> None, so the value survives json.dump as `null` rather than `NaN`."""
    return None if value is None or np.isnan(value) else float(value)


def _none_to_nan(value: Optional[float]) -> float:
    """Inverse of _nan_to_none. Absent keys (pre-FIB-711 records) also read as NaN."""
    return float("nan") if value is None else float(value)


class AlignmentSubsystem(Enum):
    BEAM_SHIFT = "beam-shift"
    STAGE = "stage"
    STAGE_VERTICAL = "stage-vertical"


class AlignmentMethod(Enum):
    CROSS_CORRELATION = "cross-correlation"
    SKIMAGE_PHASE_CORRELATION = "skimage-phase-correlation"


DEFAULT_ALIGNMENT_METHOD = AlignmentMethod.CROSS_CORRELATION


@dataclass
class AlignmentIteration:
    """Result of a single alignment step."""

    shift: Point  # (x, y) shift applied, in metres
    score: float  # alignment quality score; higher = better
    image: FibsemImage  # new image acquired during this alignment step
    xcorr: Optional[np.ndarray] = None  # cross-correlation map (bandpass method only)
    success: bool = True  # False if score < minimum_response (shift was zeroed)
    method: Optional[AlignmentMethod] = None  # which method produced this result
    # Validated-mode decision (FIB-711). Under an unvalidated mode accepted is
    # always True and disagreement_px is NaN, matching historical behaviour.
    accepted: bool = True  # False when the estimate could not be corroborated
    reason: str = "accepted"  # why it was rejected, when it was
    disagreement_px: float = float("nan")  # estimator vs validator distance
    psr: float = float("nan")  # peak-to-sidelobe ratio (diagnostic only)
    clipped: bool = False  # the shift was bounded to the ROI before being applied

    @property
    def shift_px(self) -> Point:
        """Shift in pixels, derived from shift (metres) and image pixel size."""
        if self.image is None or self.image.metadata is None:
            return Point(0.0, 0.0)
        px = self.image.metadata.pixel_size.x
        py = self.image.metadata.pixel_size.y
        return Point(self.shift.x / px, self.shift.y / py)

    def to_dict(self) -> dict:
        return {
            "shift": self.shift.to_dict(),
            "score": self.score,
            "shift_px": self.shift_px.to_dict(),
            "success": self.success,
            "method": self.method.value if self.method else None,
            "accepted": self.accepted,
            "reason": self.reason,
            # NaN is not valid JSON -- json.dump would write a bare `NaN` token that
            # strict parsers (JS JSON.parse, jq, the workflow monitor) reject. These
            # are NaN on every unvalidated run, so this would be in every record.
            "disagreement_px": _nan_to_none(self.disagreement_px),
            "psr": _nan_to_none(self.psr),
            "clipped": self.clipped,
        }

    @staticmethod
    def from_dict(d: dict, image: FibsemImage) -> "AlignmentIteration":
        return AlignmentIteration(
            shift=Point.from_dict(d["shift"]),
            score=d["score"],
            image=image,
            success=d.get("success", True),
            method=AlignmentMethod(d["method"]) if d.get("method") else None,
            # pre-FIB-711 runs applied every shift unconditionally
            accepted=d.get("accepted", True),
            reason=d.get("reason", "accepted"),
            disagreement_px=_none_to_nan(d.get("disagreement_px")),
            psr=_none_to_nan(d.get("psr")),
            clipped=d.get("clipped", False),
        )


@dataclass
class AlignmentDifferential:
    """Pairwise comparison of shift estimates across alignment methods."""

    results: "list[AlignmentIteration]"  # one per method, in method order
    shifts_px: "dict[str, Point]"  # method name → shift in pixels
    max_disagreement_px: float  # max pairwise |shift_a − shift_b| in pixels
    agreement: bool  # True if max_disagreement_px < threshold
    consensus_shift: Optional[Point] = None  # score-weighted mean shift in metres

    def to_dict(self) -> dict:
        return {
            "shifts_px": {k: v.to_dict() for k, v in self.shifts_px.items()},
            "max_disagreement_px": float(self.max_disagreement_px),
            "agreement": bool(self.agreement),
            "consensus_shift": self.consensus_shift.to_dict()
            if self.consensus_shift
            else None,
        }

    @staticmethod
    def from_dict(d: dict) -> "AlignmentDifferential":
        return AlignmentDifferential(
            results=[],  # images not serialised; reload via AlignmentResult.load()
            shifts_px={k: Point.from_dict(v) for k, v in d["shifts_px"].items()},
            max_disagreement_px=d["max_disagreement_px"],
            agreement=d["agreement"],
            consensus_shift=Point.from_dict(d["consensus_shift"])
            if d.get("consensus_shift")
            else None,
        )


@dataclass
class AlignmentResult:
    """Inputs and per-step results for a single multi-step alignment operation."""

    name: str
    reference_image: FibsemImage
    subsystem: AlignmentSubsystem
    method: AlignmentMethod
    results: list[AlignmentIteration]
    final_image: Optional[FibsemImage] = None
    validation: Optional[AlignmentDifferential] = None
    # Outcome under a validated mode (FIB-711). Under an unvalidated mode every
    # step is accepted, so aligned is True whenever any step ran.
    mode: str = DEFAULT_VALIDATION_MODE
    aligned: bool = True  # at least one step produced an applied shift
    n_accepted: int = 0
    n_rejected: int = 0
    outcome: str = "aligned"  # aligned | all_steps_rejected | stop_event | no_steps
    # How far the FINAL image still sits from the reference, in pixels (FIB-809).
    # This is the only field that answers "did this run end up aligned?" -- every
    # other field describes individual steps, and a run whose steps all look
    # reasonable can still finish in the wrong place. None when no final image was
    # acquired. Recorded only; nothing acts on it yet.
    final_residual_px: Optional[float] = None
    # Whether that residual is within the configured tolerance. None when no final
    # image was acquired, so "not converged" and "not measured" stay distinguishable
    # -- collapsing them would report every stop_event run as a convergence failure.
    converged: Optional[bool] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "subsystem": self.subsystem.value,
            "method": self.method.value,
            "results": [r.to_dict() for r in self.results],
            "validation": self.validation.to_dict() if self.validation else None,
            "mode": self.mode,
            "aligned": self.aligned,
            "n_accepted": self.n_accepted,
            "n_rejected": self.n_rejected,
            "outcome": self.outcome,
            "final_residual_px": _nan_to_none(self.final_residual_px),
            "converged": self.converged,
        }

    def save(
        self, base_path: str, plot: bool = True, plot_title: Optional[str] = None
    ) -> str:
        """Save to <base_path>/<name>/. Returns the run directory path."""
        import json

        run_dir = os.path.join(base_path, self.name)
        os.makedirs(run_dir, exist_ok=True)
        self.reference_image.save(path=os.path.join(run_dir, "reference_image"))
        for i, r in enumerate(self.results):
            r.image.save(path=os.path.join(run_dir, f"new_image_{i:02d}"))
        if self.final_image:
            self.final_image.save(path=os.path.join(run_dir, "final_image"))
        fpath = os.path.join(run_dir, "data.json")
        with open(fpath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        if plot:
            try:
                self.plot(title=plot_title, path=run_dir)
            except Exception as e:
                logging.warning(f"Failed to save alignment plot: {e}")
        return run_dir

    def plot(
        self, title: Optional[str] = None, save: bool = True, path: Optional[str] = None
    ) -> "Figure":
        # surface the run verdict on the figure, so a saved plot shows at a glance
        # whether the alignment was trusted
        if self.n_rejected:
            suffix = f"{self.outcome} — {self.n_rejected}/{len(self.results)} steps refused"
            title = f"{title} [{suffix}]" if title else suffix
        fig = plot_multi_step_alignment(
            self.reference_image,
            self.results,
            save=save,
            title=title,
            final_image=self.final_image,
            path=path,
            validation=self.validation,
            final_residual_px=self.final_residual_px,
            converged=self.converged,
        )
        return fig

    @classmethod
    def load(cls, run_dir: str) -> "AlignmentResult":
        """Load a previously saved AlignmentResult from its directory."""
        import json

        with open(os.path.join(run_dir, "data.json")) as f:
            d = json.load(f)

        reference_image = FibsemImage.load(os.path.join(run_dir, "reference_image.tif"))

        results = []
        for i, rd in enumerate(d["results"]):
            image = FibsemImage.load(os.path.join(run_dir, f"new_image_{i:02d}.tif"))
            results.append(AlignmentIteration.from_dict(rd, image))

        final_path = os.path.join(run_dir, "final_image.tif")
        final_image = (
            FibsemImage.load(final_path) if os.path.exists(final_path) else None
        )

        validation = (
            AlignmentDifferential.from_dict(d["validation"])
            if d.get("validation")
            else None
        )

        # Records written before these fields existed have no verdict. Derive one from
        # the steps rather than defaulting to the dataclass values, which assert success
        # -- reloading a refused run as "aligned" would invert exactly the fact this
        # change exists to record.
        n_accepted = sum(1 for r in results if r.accepted)
        n_rejected = len(results) - n_accepted
        default_outcome = (
            "no_steps" if not results
            else "all_steps_rejected" if n_accepted == 0
            else "aligned"
        )
        return cls(
            name=d["name"],
            reference_image=reference_image,
            subsystem=AlignmentSubsystem(d["subsystem"]),
            method=AlignmentMethod(d["method"]),
            results=results,
            final_image=final_image,
            validation=validation,
            mode=d.get("mode", DEFAULT_VALIDATION_MODE),
            aligned=d.get("aligned", n_accepted > 0),
            n_accepted=d.get("n_accepted", n_accepted),
            n_rejected=d.get("n_rejected", n_rejected),
            outcome=d.get("outcome", default_outcome),
            # absent from records written before FIB-809
            final_residual_px=d.get("final_residual_px"),
            converged=d.get("converged"),
        )


def _acquire_from_reference_image(
    microscope: FibsemMicroscope,
    ref_image: FibsemImage,
    use_autocontrast: bool = False,
    use_autofocus: bool = False,
) -> FibsemImage:
    """Acquire a new image with the same settings as the reference image."""
    image_settings = ImageSettings.fromFibsemImage(ref_image)
    image_settings.autocontrast = False
    image_settings.save = False

    if use_autocontrast:
        microscope.autocontrast(
            beam_type=image_settings.beam_type, reduced_area=image_settings.reduced_area
        )
    if use_autofocus:
        microscope.auto_focus(
            beam_type=image_settings.beam_type, reduced_area=image_settings.reduced_area
        )

    return acquire.acquire_image(microscope, settings=image_settings)


def _calculate_shift(
    ref_image: FibsemImage,
    new_image: FibsemImage,
    method: AlignmentMethod = DEFAULT_ALIGNMENT_METHOD,
    alignment_validation: Optional[str] = None,
    clip_fraction: Optional[float] = None,
) -> AlignmentIteration:
    """Calculate the shift between two images using the specified alignment method.

    `alignment_validation` selects an `VALIDATION_MODES` entry. A validated mode such as
    "verified" may return an iteration with `accepted=False` and a zeroed shift,
    meaning the estimate could not be corroborated and nothing should be moved.
    It applies to the cross-correlation method only -- the phase-correlation
    method has no validated variant and is always accepted.

    `clip_fraction` bounds an accepted shift to that fraction of the half-ROI;
    None uses the user preference. Also cross-correlation only.
    """
    logging.debug(f"Calculating shift using method: {method.value}...")
    if method is AlignmentMethod.SKIMAGE_PHASE_CORRELATION:
        result = shift_from_skimage_phase_correlation(ref_image, new_image)
        result.method = method
        return result
    else:
        validated = shift_from_crosscorrelation_validated(
            ref_image, new_image, mode=alignment_validation, clip_fraction=clip_fraction
        )
        if not validated.accepted:
            # Per-step detail only. The run-level summary in multi_step_alignment_v2
            # is the one that warns -- a validated run legitimately refuses steps, and
            # a warning per step buries everything else in the log during a session.
            logging.debug(
                "Alignment step rejected: %s. metrics=%s",
                validated.reason,
                validated.to_dict(),
            )
        return AlignmentIteration(
            shift=validated.shift,
            score=validated.score,
            image=new_image,
            xcorr=validated.xcorr,
            method=method,
            accepted=validated.accepted,
            reason=validated.reason,
            disagreement_px=validated.disagreement_px,
            psr=validated.psr,
            clipped=validated.clipped,
        )


def compare_alignment_methods(
    ref_image: FibsemImage,
    new_image: FibsemImage,
    agreement_threshold_px: float = 2.0,
) -> AlignmentDifferential:
    """Run all alignment methods on the same image pair and compare their shift estimates.

    Args:
        ref_image: Reference image.
        new_image: Image to align to the reference.
        agreement_threshold_px: Methods are considered in agreement when the
            maximum pairwise shift difference is below this value (in pixels).
            Defaults to 2.0.

    Returns:
        AlignmentDifferential containing per-method results, shifts in pixels,
        the maximum pairwise disagreement, and an agreement flag.
    """
    pixel_size = new_image.metadata.pixel_size.x if new_image.metadata else 1.0

    methods = list(AlignmentMethod)
    # clip_fraction=0: this is a measurement, not a movement. Clipping would
    # understate a large residual, and it applies only to the cross-correlation
    # path, so it would also inflate the disagreement against the two phase
    # methods and could flip `agreement` on exactly the runs that matter most.
    results = [
        _calculate_shift(ref_image, new_image, method, clip_fraction=0.0)
        for method in methods
    ]

    shifts_px = {
        method.value: Point(r.shift.x / pixel_size, r.shift.y / pixel_size)
        for method, r in zip(methods, results)
    }

    points = list(shifts_px.values())
    max_disagreement = 0.0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            diff = np.hypot(points[i].x - points[j].x, points[i].y - points[j].y)
            max_disagreement = max(max_disagreement, diff)

    logging.debug(
        {
            "msg": "compare_alignment_methods",
            "shifts_px": {k: (v.x, v.y) for k, v in shifts_px.items()},
            "max_disagreement_px": max_disagreement,
            "agreement": max_disagreement < agreement_threshold_px,
        }
    )

    valid = [r for r in results if r.success]
    if not valid:
        valid = results  # fall back to all results if every method failed
    total_score = sum(r.score for r in valid) or 1.0
    consensus_shift = Point(
        x=sum(r.shift.x * r.score for r in valid) / total_score,
        y=sum(r.shift.y * r.score for r in valid) / total_score,
    )

    return AlignmentDifferential(
        results=results,
        shifts_px=shifts_px,
        max_disagreement_px=max_disagreement,
        agreement=max_disagreement < agreement_threshold_px,
        consensus_shift=consensus_shift,
    )


def _apply_shift(
    microscope: FibsemMicroscope,
    dx: float,
    dy: float,
    beam_type: BeamType,
    subsystem: AlignmentSubsystem = AlignmentSubsystem.BEAM_SHIFT,
):
    """Apply the calculated shift to the microscope subsystem."""
    if subsystem is AlignmentSubsystem.BEAM_SHIFT:
        microscope.beam_shift(-dx, dy, beam_type)
    elif subsystem is AlignmentSubsystem.STAGE:
        microscope.stable_move(
            dx=dx,
            dy=-dy,
            beam_type=beam_type,
        )
    elif subsystem is AlignmentSubsystem.STAGE_VERTICAL:
        if beam_type is BeamType.ELECTRON and hasattr(microscope, "move_coincident_from_sem"):
            microscope.move_coincident_from_sem(dx=dx, dy=-dy)  # type: ignore
            return
        microscope.vertical_move(dy=-dy, dx=dx)


def align_with_reference_image(
    microscope: FibsemMicroscope,
    ref_image: FibsemImage,
    use_autocontrast: bool = False,
    use_autofocus: bool = False,
    subsystem: AlignmentSubsystem = AlignmentSubsystem.BEAM_SHIFT,
    method: AlignmentMethod = DEFAULT_ALIGNMENT_METHOD,
    alignment_validation: Optional[str] = None,
    context: AlignmentContext = DEFAULT_ALIGNMENT_CONTEXT,
) -> AlignmentIteration:
    """Align to a reference image. Delegates to beam_shift_alignment_v2."""
    return beam_shift_alignment_v2(
        microscope=microscope,
        ref_image=ref_image,
        use_autocontrast=use_autocontrast,
        use_autofocus=use_autofocus,
        subsystem=subsystem,
        method=method,
        alignment_validation=alignment_validation,
        context=context,
    )


def beam_shift_alignment_v2(
    microscope: FibsemMicroscope,
    ref_image: FibsemImage,
    use_autocontrast: bool = False,
    use_autofocus: bool = False,
    subsystem: AlignmentSubsystem = AlignmentSubsystem.BEAM_SHIFT,
    method: AlignmentMethod = DEFAULT_ALIGNMENT_METHOD,
    alignment_validation: Optional[str] = None,
    context: AlignmentContext = DEFAULT_ALIGNMENT_CONTEXT,
):
    """Aligns the images by adjusting the beam shift instead of moving the stage.

    This method uses cross-correlation between the reference image and a new image to calculate the
    optimal beam shift for alignment. This approach offers increased precision, but a lower range
    compared to stage movement.

        Args:
        microscope (FibsemMicroscope): An OpenFIBSEM microscope client.
        ref_image (FibsemImage): The reference image to align to.
        use_autocontrast (bool): Whether to use autocontrast for the new image. Defaults to False.
        use_autofocus (bool): Whether to use autofocus before acquiring the new image. Defaults to False.
        subsystem (AlignmentSubsystem): The subsystem to use for alignment.
            BEAM_SHIFT applies correction via beam shift (default).
            STAGE moves the stage instead. STAGE_VERTICAL uses vertical stage movement.
        method (AlignmentMethod): Cross-correlation method to use. Defaults to DEFAULT_ALIGNMENT_METHOD.
        alignment_validation (Optional[str]): Name of the alignment mode, a key of `VALIDATION_MODES`.
            A validated mode such as "verified" may refuse to move: check `accepted` on the
            result. Defaults to None, i.e. `DEFAULT_VALIDATION_MODE`.

    Raises:
        ValueError: If the reference image does not have a valid beam type.

    """
    if ref_image.metadata is None or ref_image.metadata.beam_type is None:
        raise ValueError("Reference image must have a valid beam type for alignment.")

    new_image = _acquire_from_reference_image(
        microscope=microscope,
        ref_image=ref_image,
        use_autocontrast=use_autocontrast,
        use_autofocus=use_autofocus,
    )

    # Resolve here too, not only in multi_step_alignment_v2: this is a public entry
    # point that moves the beam, and a global safety preference that some callers
    # silently ignore is not a safety preference. multi_step_alignment_v2 resolves
    # first and passes a concrete name, so this is a no-op for that path.
    if alignment_validation is None:
        alignment_validation = get_configured_validation(context)

    result = _calculate_shift(ref_image, new_image, method, alignment_validation=alignment_validation)

    if not result.accepted:
        # the estimate could not be corroborated: move nothing at all.
        # _calculate_shift already logged the detail; do not log the same event twice.
        return result

    _apply_shift(
        microscope=microscope,
        dx=result.shift.x,
        dy=result.shift.y,
        beam_type=ref_image.metadata.beam_type,
        subsystem=subsystem,
    )

    logging.info(
        f"Beam Shift Alignment: dx: {result.shift.x}, dy: {result.shift.y}, score: {result.score}"
    )

    return result


def _measure_final_residual(
    ref_image: FibsemImage,
    final_image: FibsemImage,
    validation: Optional[AlignmentDifferential],
    method: AlignmentMethod,
) -> Optional[float]:
    """How far the final image still sits from the reference, in pixels (FIB-809).

    This answers a different question from anything per-step: not "was that
    measurement trustworthy?" but "did the run end up where the reference says it
    should be?". A run whose every step looked reasonable can still finish in the
    wrong place, and nothing recorded before this could show that.

    Reuses the differential when `validate` already computed it, so the common path
    costs nothing. Measured with validation off and clipping off deliberately: a
    validated measurement can be zeroed by a refusal, which would record a badly
    misaligned run as a perfect one, and a clipped measurement understates exactly
    the large residuals this exists to catch.

    Returns None if the residual cannot be measured, never raises -- this is a
    diagnostic, and failing to record it must not fail the alignment.
    """
    try:
        if validation is not None:
            shift_px = validation.shifts_px.get(method.value)
            if shift_px is not None:
                return float(np.hypot(shift_px.x, shift_px.y))
        it = _calculate_shift(
            ref_image, final_image, method,
            alignment_validation="none", clip_fraction=0.0,
        )
        px = final_image.metadata.pixel_size.x
        py = final_image.metadata.pixel_size.y
        return float(np.hypot(it.shift.x / px, it.shift.y / py))
    except Exception as e:  # noqa: BLE001 - a diagnostic must never break a run
        logging.warning(f"Could not measure the final alignment residual: {e}")
        return None


def multi_step_alignment_v2(
    microscope: FibsemMicroscope,
    ref_image: FibsemImage,
    steps: int = 3,
    use_autocontrast: bool = False,
    use_autofocus: bool = False,
    subsystem: AlignmentSubsystem = AlignmentSubsystem.BEAM_SHIFT,
    stop_event: Optional[ThreadingEvent] = None,
    run_name: str = "AlignmentResult",
    acquire_final_image: bool = True,
    validate: bool = True,
    path: Optional[str] = None,
    method: AlignmentMethod = DEFAULT_ALIGNMENT_METHOD,
    alignment_validation: Optional[str] = None,
    context: AlignmentContext = DEFAULT_ALIGNMENT_CONTEXT,
) -> AlignmentResult:
    """Runs the beam shift alignment multiple times.

    `alignment_validation` selects a `VALIDATION_MODES` entry; None uses the user's
    preference for `context` (`user-preferences.yaml`, `alignment.validation_workflow`
    or `alignment.validation_milling`), which defaults to `DEFAULT_VALIDATION_MODE`.
    The two contexts are configured separately because refusing a shift costs 23% of
    good workflow alignments but only 8% of milling ones -- see `verified.py`. Under a validated mode a step that cannot be
    corroborated applies no shift and the loop continues, because the next step
    re-acquires and a fresh image often correlates cleanly. If every step is
    rejected the returned result says so via `aligned` / `outcome`, and the caller
    should surface that rather than mill at an unverified position.

    Separately, `final_residual_px` / `converged` record whether the run actually
    ended up matching the reference (FIB-809). That is a different question from
    whether any step was trustworthy, and a run can be `aligned=True` with every
    step accepted and still be `converged=False`.

    `converged` is deliberately NOT wired to `abort_on_failure`. That flag currently
    means "every step was refused", and silently giving it a second, unrelated
    trigger would start failing tasks for users who enabled it for the first reason
    -- a behaviour change smuggled in behind a diagnostic. The tolerance also rests
    on one instrument. Warn now; wire it to a stop only once the threshold has been
    checked somewhere else.
    """
    # None means "whatever the user configured"; resolve it once, here, so the whole
    # run and its saved record agree on which mode was used
    if alignment_validation is None:
        alignment_validation = get_configured_validation(context)
    # fail before touching the microscope if the mode name is wrong
    get_validation_mode(alignment_validation)

    alignment_results = []
    aborted = False
    for i in range(steps):
        if stop_event is not None and stop_event.is_set():
            aborted = True
            break
        # only use autocontrast on first step
        use_autocontrast = use_autocontrast if i == 0 else False
        use_autofocus = use_autofocus if i == 0 else False
        result = beam_shift_alignment_v2(
            microscope=microscope,
            ref_image=ref_image,
            use_autocontrast=use_autocontrast,
            use_autofocus=use_autofocus,
            subsystem=subsystem,
            method=method,
            alignment_validation=alignment_validation,
            context=context,
        )
        alignment_results.append(result)

    # Cancelled before any step completed: there is nothing meaningful to
    # persist. Skip the final-image acquisition (which would touch the
    # microscope after a stop request) and the save (which would otherwise
    # dump an empty AlignmentResult directory, into the CWD when no path is
    # set), and return an empty result.
    if aborted and not alignment_results:
        return AlignmentResult(
            name=run_name,
            reference_image=ref_image,
            subsystem=subsystem,
            method=method,
            results=[],
            mode=alignment_validation,
            aligned=False,
            outcome="stop_event" if aborted else "no_steps",
        )

    if validate:
        acquire_final_image = True

    final_image = None
    validation = None
    final_residual_px = None
    converged = None
    if acquire_final_image:
        final_image = _acquire_from_reference_image(
            microscope=microscope,
            ref_image=ref_image,
            use_autocontrast=False,
            use_autofocus=False,
        )
        if validate:
            validation = compare_alignment_methods(ref_image, final_image)
            if not validation.agreement:
                logging.warning(
                    f"Alignment validation failed: max disagreement "
                    f"{validation.max_disagreement_px:.2f}px across methods."
                )
        final_residual_px = _measure_final_residual(
            ref_image, final_image, validation, method
        )
        if final_residual_px is not None:
            tolerance = get_configured_convergence_tolerance()
            converged = final_residual_px <= tolerance
            if not converged:
                # The one warning that describes the OUTCOME rather than a step. A
                # run can reach here with every step accepted and still be wrong,
                # which is the whole reason this check exists.
                logging.warning(
                    "Alignment '%s': finished %.1fpx from the reference "
                    "(tolerance %.1fpx). The steps completed, but the position "
                    "does not match the reference image.",
                    run_name, final_residual_px, tolerance,
                )

    n_accepted = sum(1 for r in alignment_results if r.accepted)
    n_rejected = len(alignment_results) - n_accepted
    if not alignment_results:
        outcome = "stop_event" if aborted else "no_steps"
    elif n_accepted == 0:
        outcome = "all_steps_rejected"
    elif aborted:
        outcome = "stop_event"
    else:
        outcome = "aligned"

    disagreements = [
        r.disagreement_px
        for r in alignment_results
        if not r.accepted and not np.isnan(r.disagreement_px)
    ]
    if n_rejected:
        # One warning per run, carrying the range of disagreements so a marginal
        # refusal (a few px over tolerance) is distinguishable at a glance from a
        # gross one (a false lock, tens to hundreds of px) without opening data.json.
        spread = (
            f", disagreement {min(disagreements):.1f}-{max(disagreements):.1f}px"
            if disagreements else ""
        )
        logging.warning(
            "Alignment '%s' [%s]: %d/%d steps refused%s. %s",
            run_name, alignment_validation, n_rejected, len(alignment_results), spread,
            "No shift was applied at all -- the position is unverified."
            if n_accepted == 0 else "The accepted steps were applied.",
        )

    ts = utils.current_timestamp_v3(timeonly=True)
    run = AlignmentResult(
        name=f"{run_name}-{ts}",
        reference_image=ref_image,
        subsystem=subsystem,
        method=method,
        results=alignment_results,
        final_image=final_image,
        validation=validation,
        mode=alignment_validation,
        aligned=n_accepted > 0,
        n_accepted=n_accepted,
        n_rejected=n_rejected,
        outcome=outcome,
        final_residual_px=final_residual_px,
        converged=converged,
    )

    save_path: str = path if path is not None else _alignment_save_path(ref_image)[0]
    run.save(save_path, plot_title=run_name)
    if n_rejected:
        # the path lives here rather than in the error message, where it wrapped to two
        # lines in the workflow list and buried the reason
        logging.warning("Alignment diagnostics: %s", save_path)

    # Raised only after the run is saved: the plot and data.json are the evidence for
    # why the alignment was refused, and they are most wanted precisely when it fails.
    # "aborted" is a user Stop, not an alignment failure, so it is excluded.
    if outcome == "all_steps_rejected" and should_abort_on_failure():
        # Written to be read in the workflow list, where it is the only explanation the
        # operator gets. So: no lamella/task prefix (the UI already shows both), no mode
        # name, no absolute path (it wrapped to two lines and buried everything else).
        # What it does carry is the one number that says how badly the two measurements
        # disagreed, and what to do about it. The path stays in the log warning above.
        tolerance = get_validation_mode(alignment_validation).tolerance_px
        by = (
            f" by {max(disagreements):.0f} px (limit {tolerance:.0f} px)"
            if disagreements else ""
        )
        raise AlignmentFailedError(
            f"Could not verify the alignment: two independent measurements "
            f"disagreed{by}. The beam was not moved and milling was stopped rather "
            f"than risk cutting in the wrong place. Check the alignment images for "
            f"this task."
        )

    return run


def _eucentric_tilt_alignment(
    microscope: FibsemMicroscope,
    image_settings: ImageSettings,
    target_angle: float,
    step_size: float,
    beam_type: Optional[BeamType] = None,
    show: bool = False,
) -> None:
    """Perform eucentric tilt alignment by moving the stage in steps towards the target angle,
    acquiring images at each step, and performing alignment.
    Args:
        microscope (FibsemMicroscope): The microscope to use for alignment.
        image_settings (ImageSettings): The image settings to use for image acquisition.
        target_angle (float): The target tilt angle in degrees.
        step_size (float): The step size in degrees.
        beam_type (Optional[BeamType]): The beam type to use for image acquisition. If None, both beams are used.
        show (bool): Whether to show the images at each step. Defaults to False.
    Returns:
        None
    """
    import matplotlib.pyplot as plt

    from fibsem.structures import FibsemStagePosition

    stage_position = microscope.get_stage_position()
    current_angle = np.degrees(stage_position.t)

    n_steps = int(abs(int(current_angle) - target_angle) // step_size)

    logging.info(
        f"Current Tilt: {current_angle}, Target Tilt:  {target_angle}, Step Size: {step_size},  Num Steps: {n_steps}"
    )
    steps = np.linspace(current_angle, target_angle, num=n_steps)

    image_settings.hfw = 150e-6
    image_settings.save = False
    if beam_type is not None:
        image_settings.beam_type = beam_type
        reference_image = acquire.acquire_image(microscope, image_settings)
    else:
        ref_sem_image, ref_fib_image = acquire.acquire_channels(
            microscope, image_settings
        )

    fib_images = []
    sem_images = []

    for i, angle in enumerate(steps[1:]):
        microscope.move_stage_absolute(FibsemStagePosition(t=np.radians(angle)))

        if beam_type is not None:
            beam_shift_alignment_v2(
                microscope, reference_image, subsystem=AlignmentSubsystem.STAGE
            )
        else:
            beam_shift_alignment_v2(
                microscope, ref_sem_image, subsystem=AlignmentSubsystem.STAGE
            )
            beam_shift_alignment_v2(
                microscope, ref_fib_image, subsystem=AlignmentSubsystem.STAGE_VERTICAL
            )

        sem_image, fib_image = acquire.acquire_channels(microscope, image_settings)

        if show:
            fig, ax = plt.subplots(1, 2, figsize=(10, 7))
            ax[0].imshow(sem_image.data, cmap="gray")
            ax[0].plot(
                sem_image.data.shape[1] // 2, sem_image.data.shape[0] // 2, "y+", ms=50
            )
            ax[1].imshow(fib_image.data, cmap="gray")
            ax[1].plot(
                fib_image.data.shape[1] // 2, fib_image.data.shape[0] // 2, "y+", ms=50
            )
            plt.show()

        sem_images.append(sem_image)
        fib_images.append(fib_image)
        if beam_type is None:
            ref_sem_image = sem_image
            ref_fib_image = fib_image
        elif beam_type is BeamType.ELECTRON:
            reference_image = sem_image
        elif beam_type is BeamType.ION:
            reference_image = fib_image

    # TODO: have a metric to measure if it failed? how??
    final_position = microscope.get_stage_position()
    diff = stage_position - final_position
    logging.info(f"Start Position: {stage_position.pretty}")
    logging.info(f"Final Position: {final_position.pretty}")
    logging.info(f"Difference: {diff.pretty}")

    return sem_images, fib_images
