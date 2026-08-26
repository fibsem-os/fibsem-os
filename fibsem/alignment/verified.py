"""Alignment validation: a second opinion that can refuse to move, and a bounded clip.

The intensity cross-correlation used in production cannot see its own catastrophic
failures: where the correlation surface is bi-modal it returns the wrong peak,
sharp and confident, and the beam is shifted by that amount before milling
(FIB-711).

A difference-of-Gaussians correlation gets many of those right, but it is *not* a
better method -- run as the estimator it is worse. What makes the two useful
together is that they largely fail on disjoint images, so the intensity path stays
the **estimator** and the band-pass runs as a **validator** whose shift is
discarded. It only decides whether the estimator is trusted.

Measured cost and benefit (FIB-807 audit, 306 steps, 204 with a successor step to
score against). Scoring uses an outcome oracle -- a step's shift worked if the
next step's residual is small -- NOT the magnitude of the shift. Magnitude does
not separate good from bad here: of 34 steps above 40 px, 17 worked and 17 failed.

    subsystem   bad shifts caught   good shifts refused
    workflow          11 / 12            22 / 96  (23%)
    milling            7 / 11             7 / 85  ( 8%)

Workflow alignment aligns *to* a reference and legitimately makes ~130 px
first-step corrections, so refusing large shifts is expensive there. Milling drift
correction runs from an already-aligned position where a large shift really is
suspect. That is why validation is configured per context rather than globally,
and why milling defaults to `verified` while workflow does not.

`clip_fraction` is the cheaper, always-on guard. A shift is expressed as a
fraction of the half-ROI: beyond 1.0 the two images overlap by less than half, so
the fiducial cannot have been in both. Clipping at 0.8 flags 11 steps of which 10
are genuinely bad -- 10:1, against the 1:1 of an absolute pixel threshold. It
clips rather than rejects, so a correct large shift is only slowed: the step
applies the bounded move and the next step walks the remainder.

Design and evidence: FIB-711, FIB-719, FIB-807.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np

from fibsem.alignment.methods import (
    get_preprocessing_profile,
    shift_from_crosscorrelation,
)
from fibsem.structures import FibsemImage, Point

# Fourier band-pass parameters shared by both passes. These are the values
# `_calculate_shift` has always used for the production correlation.
_BANDPASS = dict(lowpass=50, highpass=4, sigma=5)


class AlignmentFailedError(Exception):
    """An alignment could not be trusted and the caller asked to stop rather than continue.

    Raised only when the user has set `alignment.abort_on_failure`. Deliberately not
    an OperationCancelledError: that means "the user pressed Stop", and a task ending
    this way is a genuine failure that should be recorded as one.
    """


class AlignmentContext(Enum):
    """Which alignment is running, because the two have different priors.

    WORKFLOW aligns *to* a stored reference image and legitimately makes large
    first-step corrections. MILLING corrects drift from an already-aligned
    position, where a large shift is much more likely to be a false lock. They get
    separate validation settings for that reason -- see the module docstring for
    the measured difference.
    """

    WORKFLOW = "workflow"
    MILLING = "milling"


DEFAULT_ALIGNMENT_CONTEXT = AlignmentContext.WORKFLOW


@dataclass(frozen=True)
class ValidationMode:
    """An estimator, and optionally a validator that must agree with it.

    Attributes:
        estimator: the preprocessing profile whose shift is used.
        validator: an independent profile run on the same image pair. Its shift is
            never used; it only decides whether the estimator is trusted. None
            disables validation, which is the historical unconditional behaviour.
        tolerance_px: the two shifts must agree within this distance, in pixels.
    """

    estimator: str
    validator: Optional[str] = None
    tolerance_px: float = 3.0


#: How much corroboration an alignment needs before its shift is applied.
#:
#: "none" is the historical production path and stays the default: one
#: correlation, no confidence, the shift is always applied.
#:
#: "verified" keeps that same estimator and adds a difference-of-Gaussians second
#: opinion that must agree before the shift is applied. On the harvested corpus
#: this removed every catastrophic beam shift (5/383 -> 0/383) while still
#: accepting 92% of alignments. Checked independently against a second corpus
#: from an unrelated site (306 iterations): zero rejections across the 192
#: iterations of tasks that never fail, and it fires on the two tasks that do.
#:
#: Note the estimator is the same either way. Running the band-pass *as* the
#: estimator is a worse alignment (44 -> 54 wild shifts on that corpus), so it
#: is deliberately not offered here; use `plot_preprocessing_comparison` to compare
#: the two estimators on a given image pair instead.
VALIDATION_MODES = {
    "none": ValidationMode(estimator="zscore", validator=None),
    "verified": ValidationMode(estimator="zscore", validator="dog", tolerance_px=3.0),
}
DEFAULT_VALIDATION_MODE = "none"

#: Clip a shift to this fraction of the half-ROI. A shift of 1.0 means the two
#: images overlap by less than half, so the fiducial cannot have been in both --
#: a geometric bound, not a fitted one. 0.8 is the operating point measured in
#: FIB-807: it flags 11 steps of which 10 are genuinely bad. Set 0 to disable.
#:
#: This CLIPS rather than rejects, so a correct large shift is not lost: the step
#: applies the bounded move and the next step walks the remainder. That is what
#: makes a non-unity default acceptable -- the cost of over-clipping is a slower
#: convergence, not a refused alignment.
DEFAULT_CLIP_FRACTION = 0.8

#: A run is converged when its final image sits within this many pixels of the
#: reference. Insensitive across 10-20 px on the reference corpus: the residuals
#: are bimodal (95 of 102 below 10 px, then a 14.5 px gap), so every value in that
#: range flags the same 7 runs. Diagnostic only -- it drives a warning, not a stop.
DEFAULT_CONVERGENCE_TOLERANCE_PX = 10.0


def get_configured_validation(
    context: AlignmentContext = DEFAULT_ALIGNMENT_CONTEXT,
) -> str:
    """The user's alignment validation setting for `context`, from user-preferences.yaml.

    Split per context because the cost differs sharply between them: refusing a
    shift costs 23% of good workflow alignments but only 8% of milling ones, while
    catching a comparable share of the bad. One global tolerance cannot serve both.

    Falls back to the default if preferences cannot be read, because refusing to
    align because a preferences file is unreadable would be a worse failure than
    aligning the way we always have.
    """
    try:
        from fibsem.config import load_user_preferences

        a = load_user_preferences().alignment
        mode = a.validation_milling if context is AlignmentContext.MILLING else a.validation_workflow
    except Exception as e:  # noqa: BLE001 - never let prefs break alignment
        logging.warning(f"Could not read alignment validation preference: {e}")
        return DEFAULT_VALIDATION_MODE
    if mode not in VALIDATION_MODES:
        logging.warning(
            f"Unknown alignment validation {mode!r} in user preferences for "
            f"{context.value}; using {DEFAULT_VALIDATION_MODE!r}."
        )
        return DEFAULT_VALIDATION_MODE
    return mode


def get_configured_clip_fraction() -> float:
    """The user's clip fraction, from user-preferences.yaml. 0 disables clipping."""
    try:
        from fibsem.config import load_user_preferences

        v = float(load_user_preferences().alignment.clip_fraction)
    except Exception as e:  # noqa: BLE001 - never let prefs break alignment
        logging.warning(f"Could not read alignment clip preference: {e}")
        return DEFAULT_CLIP_FRACTION
    if v < 0:
        logging.warning(f"Negative alignment clip_fraction {v}; clipping disabled.")
        return 0.0
    return v


def should_abort_on_failure() -> bool:
    """Whether a failed alignment should stop the task, from user preferences.

    Defaults to False -- the historical behaviour -- if preferences cannot be read.
    """
    try:
        from fibsem.config import load_user_preferences

        return bool(load_user_preferences().alignment.abort_on_failure)
    except Exception as e:  # noqa: BLE001 - never let prefs break alignment
        logging.warning(f"Could not read alignment abort preference: {e}")
        return False


def get_configured_convergence_tolerance() -> float:
    """The user's convergence tolerance in pixels, from user-preferences.yaml.

    Falls back to the default if preferences cannot be read -- a diagnostic
    threshold must never be the reason an alignment fails to run.
    """
    try:
        from fibsem.config import load_user_preferences

        v = float(load_user_preferences().alignment.convergence_tolerance_px)
    except Exception as e:  # noqa: BLE001 - never let prefs break alignment
        logging.warning(f"Could not read convergence tolerance preference: {e}")
        return DEFAULT_CONVERGENCE_TOLERANCE_PX
    if v <= 0:
        logging.warning(
            f"Non-positive convergence tolerance {v}; using "
            f"{DEFAULT_CONVERGENCE_TOLERANCE_PX}."
        )
        return DEFAULT_CONVERGENCE_TOLERANCE_PX
    return v


def get_validation_mode(mode: Optional[str]) -> ValidationMode:
    """Resolve a validation name to a `ValidationMode`.

    Args:
        mode: a key of `VALIDATION_MODES`, or None for the default.

    Raises:
        ValueError: if the name is not a known validation setting.
    """
    if mode is None:
        mode = DEFAULT_VALIDATION_MODE
    if mode not in VALIDATION_MODES:
        raise ValueError(
            f"Unknown alignment validation {mode!r}. "
            f"Available: {sorted(VALIDATION_MODES)}"
        )
    return VALIDATION_MODES[mode]


def correlation_psr(xcorr: np.ndarray, exclude_radius: int = 8) -> float:
    """Peak-to-sidelobe ratio of a correlation surface.

    (peak - mean) / std over the surface with a disk around the peak excluded.
    Scale-invariant, so it is comparable between methods.

    Recorded as a diagnostic only. On the harvested corpus a PSR threshold is
    redundant once estimator/validator agreement is required, and gating on both
    costs acceptance without improving safety -- where the estimator is wrong its
    PSR is 2.1-2.6, where it is right 2.3-8.1, and those ranges overlap.
    """
    surf = np.asarray(xcorr, dtype=np.float64)
    h, w = surf.shape
    row, col = np.unravel_index(np.argmax(surf), surf.shape)
    peak = float(surf[row, col])
    yy, xx = np.ogrid[0:h, 0:w]
    outside = surf[((yy - row) ** 2 + (xx - col) ** 2) > exclude_radius**2]
    if outside.size == 0:
        return 0.0
    mu, sd = float(outside.mean()), float(outside.std())
    return (peak - mu) / sd if sd > 1e-20 else 0.0


def clip_shift_to_roi(
    dx: float,
    dy: float,
    shape: Tuple[int, int],
    pixel_size: Tuple[float, float],
    fraction: float = DEFAULT_CLIP_FRACTION,
) -> Tuple[float, float, bool]:
    """Bound a shift to `fraction` of the half-ROI, per axis, preserving direction.

    Scales both components by a single factor so the direction is unchanged and
    only the magnitude is bounded -- clipping the axes independently would rotate
    the correction.

    Args:
        dx, dy: shift in metres.
        shape: (height, width) of the alignment ROI, in pixels.
        pixel_size: (x, y) metres per pixel.
        fraction: bound as a fraction of the half-ROI. <= 0 disables clipping.

    Returns:
        (dx, dy, was_clipped), in metres.
    """
    if fraction <= 0:
        return dx, dy, False
    h, w = shape
    px, py = pixel_size
    limit_x = fraction * (w / 2) * px
    limit_y = fraction * (h / 2) * py
    # how far over the bound the worse axis is; scale both by that so direction holds
    over = max(
        abs(dx) / limit_x if limit_x > 0 else 0.0,
        abs(dy) / limit_y if limit_y > 0 else 0.0,
    )
    if over <= 1.0:
        return dx, dy, False
    return dx / over, dy / over, True


@dataclass
class ValidatedAlignment:
    """A translation estimate with an explicit accept/reject decision.

    `accepted` is the only field a caller may act on: when it is False the shift
    fields are zero and nothing should be moved.
    """

    shift: Point  # metres; (0, 0) when rejected
    shift_px: Point  # pixels; (0, 0) when rejected
    score: float
    accepted: bool
    reason: str  # "accepted", or why it was rejected
    disagreement_px: float  # distance between estimator and validator shifts
    psr: float  # estimator peak-to-sidelobe ratio (diagnostic)
    mode: str = DEFAULT_VALIDATION_MODE
    clipped: bool = False  # the shift was bounded to the ROI before being applied
    xcorr: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        """Loggable summary; excludes the correlation surface."""
        return {
            "shift": self.shift.to_dict(),
            "shift_px": self.shift_px.to_dict(),
            "score": self.score,
            "accepted": self.accepted,
            "reason": self.reason,
            "disagreement_px": self.disagreement_px,
            "psr": self.psr,
            "mode": self.mode,
            "clipped": self.clipped,
        }


def shift_from_crosscorrelation_validated(
    ref_image: FibsemImage,
    new_image: FibsemImage,
    mode: Optional[str] = None,
    clip_fraction: Optional[float] = None,
) -> ValidatedAlignment:
    """Translation estimate with an explicit accept/reject decision and a bounded shift.

    Registers the same image pair twice: once with the mode's estimator, whose
    shift is returned, and once with its validator, whose shift is used only to
    decide whether the estimator can be trusted. If the two disagree by more than
    the mode's tolerance the result is rejected and the shift is zeroed -- the
    caller must not move anything.

    An accepted shift is then clipped to `clip_fraction` of the half-ROI. The two
    guards are deliberately independent and applied in this order:

    - Validation asks "did the two methods find the same peak?". That question is
      about the raw measurements, so it is answered BEFORE any clipping. Clipping
      first would let a 300 px estimate and a 200 px estimate both clip to the
      bound, agree exactly, and be accepted -- a false accept manufactured by the
      guard meant to make things safer.
    - Clipping asks "is this move physically plausible given the overlap?". It
      bounds what is applied without discarding it, so a correct large shift is
      only slowed: the next step walks the remainder.

    A mode with no validator returns `accepted=True` unconditionally, which is the
    historical behaviour, but is still clipped.

    Args:
        ref_image: the reference image to align to.
        new_image: the newly acquired image.
        mode: a key of `VALIDATION_MODES`. Defaults to None, i.e. `DEFAULT_VALIDATION_MODE`.
        clip_fraction: bound as a fraction of the half-ROI. None uses the user
            preference; 0 disables clipping.

    Returns:
        ValidatedAlignment. Act only on `accepted`.
    """
    m = get_validation_mode(mode)
    mode_name = mode if mode is not None else DEFAULT_VALIDATION_MODE
    if clip_fraction is None:
        clip_fraction = get_configured_clip_fraction()

    px = new_image.metadata.pixel_size.x
    py = new_image.metadata.pixel_size.y
    shape = new_image.data.shape

    dx, dy, xcorr, score = shift_from_crosscorrelation(
        ref_image, new_image, **get_preprocessing_profile(m.estimator), **_BANDPASS
    )
    psr = correlation_psr(xcorr)

    if m.validator is None:
        accepted, disagreement = True, float("nan")
        reason = "accepted (no validator configured)"
    else:
        # Independent second opinion. Its shift is never applied, so its own
        # accuracy does not matter -- only whether it corroborates the estimator.
        # save=False so the validator pass does not double the diagnostics on disk.
        vdx, vdy, _, _ = shift_from_crosscorrelation(
            ref_image,
            new_image,
            save=False,
            **get_preprocessing_profile(m.validator),
            **_BANDPASS,
        )
        # raw, unclipped -- see the note on ordering above
        disagreement = float(np.hypot(dx / px - vdx / px, dy / py - vdy / py))
        accepted = disagreement <= m.tolerance_px
        reason = (
            "accepted"
            if accepted
            else (
                f"{m.estimator}/{m.validator} disagree by {disagreement:.1f} px "
                f"(tolerance {m.tolerance_px:.1f} px)"
            )
        )

    clipped = False
    if accepted:
        dx, dy, clipped = clip_shift_to_roi(dx, dy, shape, (px, py), clip_fraction)
        if clipped:
            logging.debug(
                f"Alignment shift clipped to {clip_fraction:.2f} of the half-ROI: "
                f"applying {dx / px:.1f}, {dy / py:.1f} px"
            )
    else:
        dx = dy = 0.0

    result = ValidatedAlignment(
        shift=Point(dx, dy),
        shift_px=Point(dx / px, dy / py),
        score=score,
        accepted=accepted,
        reason=reason,
        disagreement_px=disagreement,
        psr=psr,
        mode=mode_name,
        clipped=clipped,
        xcorr=xcorr,
    )
    logging.debug({"msg": "shift_from_crosscorrelation_validated", **result.to_dict()})
    return result
