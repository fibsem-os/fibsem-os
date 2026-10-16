"""Bounded Gaussian fits for a spot in an image and a peak in a profile (FIB-1218).

The correlation point fits (FIB hole, FM reflection hole, FM target) each grew
their own fit on top of two unbounded helpers, and each was patched on its own
for the same faults: a fit that runs out of its window, a "dark" fit that turns
bright, a z outside the stack, a failure reported as "unchanged", and no way to
tell a spot from fitted noise. This module is the one fit they move onto.

* :func:`fit_spot_2d` fits an axis-aligned 2D Gaussian in a window of an image.
* :func:`fit_peak_1d` fits a 1D Gaussian in a window of a profile, such as the
  mean brightness of each plane of a z-stack around a point.

Both are bounded (centre inside the window, width in a sensible range, the
amplitude's sign set by :class:`Polarity`), deterministic, cap their
evaluations so a failure is quick, and return a :class:`FitResult` that says
whether the fit found something, with the numbers to judge it and a reason
when it did not.

**The edge rule.** A fit whose centre lies within about one width of the edge
of its window may be the best fit, or an extrapolation from a window that holds
only one side of the feature. The two can't be told apart from one window, so
the fit is repeated once in a window centred on that answer: it is kept when it
then sits inside, and reported :attr:`FitOutcome.AT_EDGE` when it doesn't, or
when the window can't move because it is already at the edge of the data.

Pixels and planes only. This module imports nothing from ``fibsem.correlation``,
so it can move to ``fibsem/fitting.py`` when a second user appears.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Literal, Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares

__all__ = [
    "FitOutcome",
    "FitResult",
    "Polarity",
    "fit_peak_1d",
    "fit_spot_2d",
]


class Polarity(str, Enum):
    """Whether the feature is darker or brighter than its surroundings."""

    DARK = "dark"
    BRIGHT = "bright"


class FitOutcome(str, Enum):
    FITTED = "fitted"  # a feature of the expected polarity and size, inside the window
    AT_EDGE = "at_edge"  # the best fit is at the window's edge and can't be confirmed
    FAILED = "failed"  # no feature: no contrast, wrong width, or no convergence


Start = Literal["center", "extreme"]

# Evaluations before a fit gives up. Today's unbounded fits allow 10000, which
# makes every failure slow; at 200, two real METEOR burns with a bright halo
# ran out before converging.
MAX_NFEV = 1000
# A fitted amplitude below this many times the window's pixel noise is noise.
# Saved FIB picks score 7x and up (median 45x); a flat or noise-only window
# scores below this. Texture scores high too, so this is not what tells a burn
# from texture: width, polarity and the edge rule are.
MIN_CONTRAST = 3.0
# The narrowest width that is a feature rather than a single pixel.
MIN_SIGMA = 0.5


@dataclass(frozen=True)
class FitResult:
    """What a fit found, in the input's own coordinates.

    ``position`` and ``sigma`` are ``(x, y)`` for a 2D fit and ``(z,)`` for a
    1D fit; ``window`` holds each axis's first and last index searched. They are
    set whenever the least-squares fit ran, including a FAILED or AT_EDGE one,
    so a caller can show where it went; only a FITTED result is an answer.
    """

    outcome: FitOutcome
    position: Optional[Tuple[float, ...]] = None
    sigma: Optional[Tuple[float, ...]] = None
    amplitude: Optional[float] = None  # > 0; the polarity gives its sign
    offset: Optional[float] = None  # the background level
    noise: Optional[float] = None  # the window's pixel noise (see _pixel_noise)
    contrast_to_noise: Optional[float] = None  # amplitude / noise
    residual: Optional[float] = None  # RMS of what the model doesn't explain
    window: Optional[Tuple[Tuple[int, int], ...]] = None
    recentred: bool = False  # the edge rule moved the window once
    reason: Optional[str] = None  # why it is not FITTED, or how it got there

    @property
    def ok(self) -> bool:
        return self.outcome is FitOutcome.FITTED


def fit_spot_2d(
    image: np.ndarray,
    x: float,
    y: float,
    half_width: int,
    polarity: Polarity,
    *,
    sigma: float = 2.0,
    start: Start = "center",
    min_contrast: float = MIN_CONTRAST,
    recentre: bool = True,
    max_nfev: int = MAX_NFEV,
) -> FitResult:
    """Fit a 2D Gaussian spot near ``(x, y)`` in ``image``.

    The window is ``half_width`` pixels either side of the rounded point,
    clamped to the image. The fit starts ``sigma`` pixels wide, at the point
    (``start="center"``) or at the window's darkest or brightest pixel after
    light smoothing (``start="extreme"``). See the module docstring for the
    edge rule and :class:`FitResult` for what comes back.
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"fit_spot_2d needs a 2D image, got shape {image.shape}")
    height, width = image.shape
    xi, yi = int(round(x)), int(round(y))
    x0, x1 = max(0, xi - half_width), min(width - 1, xi + half_width)
    y0, y1 = max(0, yi - half_width), min(height - 1, yi + half_width)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return FitResult(
            FitOutcome.FAILED,
            window=((x0, x1), (y0, y1)),
            reason=f"window too small to fit: x {x0}-{x1}, y {y0}-{y1}",
        )
    roi = image[y0 : y1 + 1, x0 : x1 + 1]
    ny, nx = roi.shape
    yy, xx = np.indices(roi.shape, dtype=float)
    sign = -1.0 if polarity is Polarity.DARK else 1.0

    if start == "extreme":
        smooth = ndimage.gaussian_filter(roi, sigma=1.0)
        pick = np.argmin if polarity is Polarity.DARK else np.argmax
        sy0, sx0 = (float(v) for v in np.unravel_index(pick(smooth), roi.shape))
    else:
        sx0 = float(np.clip(x - x0, 0, nx - 1))
        sy0 = float(np.clip(y - y0, 0, ny - 1))

    background = float(np.percentile(roi, 75 if polarity is Polarity.DARK else 25))
    at_start = float(roi[int(round(sy0)), int(round(sx0))])
    amplitude0 = max(abs(at_start - background), 1e-6 * (1.0 + abs(background)))
    sigma_max = max(nx, ny) / 2.0
    s0 = float(np.clip(sigma, MIN_SIGMA, sigma_max))

    def model(p):
        a, cx, cy, sx, sy, b = p
        g = np.exp(-((xx - cx) ** 2) / (2 * sx**2) - ((yy - cy) ** 2) / (2 * sy**2))
        return b + sign * a * g

    lower = [0.0, 0.0, 0.0, MIN_SIGMA, MIN_SIGMA, -np.inf]
    upper = [np.inf, nx - 1.0, ny - 1.0, sigma_max, sigma_max, np.inf]
    p0 = [amplitude0, sx0, sy0, s0, s0, background]
    p, residual, converged = _least_squares(
        lambda q: (model(q) - roi).ravel(), p0, lower, upper, max_nfev
    )
    a, cx, cy, sx, sy, b = (float(v) for v in p)
    noise = _noise_floor(_pixel_noise(roi), b)
    result = FitResult(
        FitOutcome.FITTED,
        position=(cx + x0, cy + y0),
        sigma=(sx, sy),
        amplitude=a,
        offset=b,
        noise=noise,
        contrast_to_noise=_contrast(a, noise),
        residual=residual,
        window=((x0, x1), (y0, y1)),
    )
    early = _failure(result, converged, max_nfev, min_contrast, None)
    if early is not None:
        return early

    # The edge rule, per axis: an edge the window could move past, or one at
    # the edge of the image, where the fall-off can't be seen. A window holding
    # only one side of a feature makes it look wider than it is, so the width
    # check waits for the re-centred fit. Contrast and a collapse to one pixel
    # don't: re-centring on nothing would only go looking somewhere else.
    edges = []
    for c, s, lo, hi, n, axis in (
        (cx, sx, x0, x1, width, "x"),
        (cy, sy, y0, y1, height, "y"),
    ):
        near_lo = c < max(s, 1.0)
        near_hi = (hi - lo) - c < max(s, 1.0)
        if near_lo or near_hi:
            at_data_edge = (near_lo and lo == 0) or (near_hi and hi == n - 1)
            edges.append((axis, at_data_edge))
    at_data = any(at for _, at in edges)
    if edges and recentre and not at_data:
        return _after_recentre(
            fit_spot_2d(
                image,
                cx + x0,
                cy + y0,
                half_width,
                polarity,
                sigma=max(sx, sy),
                start="center",
                min_contrast=min_contrast,
                recentre=False,
                max_nfev=max_nfev,
            ),
            result,
        )
    failed = _failure(result, converged, max_nfev, min_contrast, sigma_max)
    if failed is not None:
        return failed
    if edges:
        where = "image" if at_data else "window"
        return _at_edge(result, f"at the {where} edge in {_axes(edges)}")
    return result


def fit_peak_1d(
    profile: np.ndarray,
    center: float,
    half_width: int,
    polarity: Polarity,
    *,
    sigma: float = 1.5,
    start: Start = "extreme",
    min_contrast: float = MIN_CONTRAST,
    recentre: bool = True,
    max_nfev: int = MAX_NFEV,
) -> FitResult:
    """Fit a 1D Gaussian peak (or dip) near index ``center`` of ``profile``.

    The window is ``half_width`` samples either side of the rounded centre,
    clamped to the profile, so the answer is always an index inside the data,
    never one that would wrap or fall off the end of a stack. Starts as
    :func:`fit_spot_2d` does; the default starts at the window's extreme,
    since a profile is usually computed through the clicked point already.
    """
    profile = np.asarray(profile, dtype=float)
    if profile.ndim != 1:
        raise ValueError(f"fit_peak_1d needs a 1D profile, got shape {profile.shape}")
    n = profile.size
    ci = int(round(center))
    i0, i1 = max(0, ci - half_width), min(n - 1, ci + half_width)
    if i1 - i0 < 3:
        return FitResult(
            FitOutcome.FAILED,
            window=((i0, i1),),
            reason=f"window too small to fit: {i0}-{i1}",
        )
    seg = profile[i0 : i1 + 1]
    t = np.arange(seg.size, dtype=float)
    sign = -1.0 if polarity is Polarity.DARK else 1.0

    if start == "extreme":
        smooth = ndimage.gaussian_filter1d(seg, sigma=1.0)
        pick = np.argmin if polarity is Polarity.DARK else np.argmax
        s0c = float(pick(smooth))
    else:
        s0c = float(np.clip(center - i0, 0, seg.size - 1))

    background = float(np.percentile(seg, 75 if polarity is Polarity.DARK else 25))
    at_start = float(seg[int(round(s0c))])
    amplitude0 = max(abs(at_start - background), 1e-6 * (1.0 + abs(background)))
    sigma_max = seg.size / 2.0
    s0 = float(np.clip(sigma, MIN_SIGMA, sigma_max))

    def model(p):
        a, c, s, b = p
        return b + sign * a * np.exp(-((t - c) ** 2) / (2 * s**2))

    lower = [0.0, 0.0, MIN_SIGMA, -np.inf]
    upper = [np.inf, seg.size - 1.0, sigma_max, np.inf]
    p, residual, converged = _least_squares(
        lambda q: model(q) - seg,
        [amplitude0, s0c, s0, background],
        lower,
        upper,
        max_nfev,
    )
    a, c, s, b = (float(v) for v in p)
    noise = _noise_floor(_pixel_noise(seg), b)
    result = FitResult(
        FitOutcome.FITTED,
        position=(c + i0,),
        sigma=(s,),
        amplitude=a,
        offset=b,
        noise=noise,
        contrast_to_noise=_contrast(a, noise),
        residual=residual,
        window=((i0, i1),),
    )
    early = _failure(result, converged, max_nfev, min_contrast, None)
    if early is not None:
        return early

    # The edge rule, as in fit_spot_2d.
    near_lo = c < max(s, 1.0)
    near_hi = (i1 - i0) - c < max(s, 1.0)
    at_data_edge = (near_lo and i0 == 0) or (near_hi and i1 == n - 1)
    if at_data_edge and _turns_over(seg, c, a, sign, near_lo, near_hi):
        # A peak near the end of the stack whose fall-off is in the data is a
        # fit: e.g. a target 5 planes from the end, 40% down by the last plane.
        near_lo = near_lo and i0 != 0
        near_hi = near_hi and i1 != n - 1
        at_data_edge = False
    at_edge = near_lo or near_hi
    if at_edge and recentre and not at_data_edge:
        return _after_recentre(
            fit_peak_1d(
                profile,
                c + i0,
                half_width,
                polarity,
                sigma=s,
                start="center",
                min_contrast=min_contrast,
                recentre=False,
                max_nfev=max_nfev,
            ),
            result,
        )
    failed = _failure(result, converged, max_nfev, min_contrast, sigma_max)
    if failed is not None:
        return failed
    if at_edge:
        where = "end of the data" if at_data_edge else "window edge"
        return _at_edge(result, f"at the {where}")
    return result


# -- shared ------------------------------------------------------------------


def _least_squares(residual, p0, lower, upper, max_nfev):
    """Bounded least squares; returns (params, residual RMS, converged).

    Trust-region reflective with a forward-difference Jacobian: deterministic
    for a given input, and the bounds are honoured rather than penalised.
    """
    p0 = np.clip(np.asarray(p0, dtype=float), lower, upper)
    sol = least_squares(
        residual,
        p0,
        bounds=(lower, upper),
        method="trf",
        x_scale="jac",
        max_nfev=max_nfev,
    )
    r = sol.fun
    dof = max(r.size - len(p0), 1)
    noise = float(np.sqrt(np.sum(r**2) / dof))
    return sol.x, noise, bool(sol.status > 0)


# The standard deviation of (white noise - its Gaussian blur at sigma 1), as a
# fraction of the noise's own: the high-pass below removes this much of it.
_HIGH_PASS_STD = {1: 0.696, 2: 0.872}


def _pixel_noise(data: np.ndarray) -> float:
    """The data's pixel-to-pixel noise: a robust spread of what a light blur
    removes, scaled back to the noise's own standard deviation.

    Not the fit's residual: a burn's bright halo, or any structure the model
    doesn't describe, inflates the residual, and real METEOR burns then read as
    no contrast. This measures only the noise.
    """
    high = data - ndimage.gaussian_filter(data, sigma=1.0)
    mad = float(np.median(np.abs(high - np.median(high))))
    return 1.4826 * mad / _HIGH_PASS_STD[data.ndim]


# A peak near the end of the data counts as seen when the profile falls by at
# least this fraction of its fitted amplitude between the peak and the end.
TURN_OVER = 0.25


def _turns_over(seg, c, amplitude, sign, near_lo, near_hi) -> bool:
    """Whether the profile visibly turns over between the peak at ``c`` and the
    end of the data it is near: falls (or, for a dark dip, rises) by at least
    :data:`TURN_OVER` of the amplitude.

    A profile that only rises to the end of a stack has its peak on the last
    sample and no fall at all. The real FM target fitted bright in a channel
    where it is dark sits 2 planes from the start with a fall of 2 against an
    amplitude of 180. Neither turns over; a target 5 planes from the end that is
    40 % down by the last plane does.
    """
    k = int(round(min(max(c, 0), seg.size - 1)))
    end = 0 if near_lo else seg.size - 1
    drop = sign * (seg[k] - seg[end])
    return drop >= TURN_OVER * amplitude


def _noise_floor(noise: float, offset: float) -> float:
    """The noise, never below a millionth of the background level: a perfectly
    flat window has none, and any amplitude would then read as infinite
    contrast."""
    return max(noise, 1e-6 * (1.0 + abs(offset)))


def _contrast(amplitude: float, noise: float) -> float:
    return float(amplitude / noise) if noise > 0 else float("inf")


def _failure(result, converged, max_nfev, min_contrast, sigma_max):
    """The FAILED result for a fit that found no feature, or None.

    ``sigma_max`` None skips the too-wide check (see the edge rule).
    """
    if not converged:
        reason = f"the fit did not converge in {max_nfev} evaluations"
    elif result.contrast_to_noise < min_contrast:
        reason = (
            f"no contrast: the feature is {result.contrast_to_noise:.1f}x the "
            f"noise (needs {min_contrast:g}x)"
        )
    elif sigma_max is not None and any(s >= sigma_max * 0.999 for s in result.sigma):
        reason = "too wide to be a spot: the fit spans the whole window"
    elif any(s <= MIN_SIGMA * 1.001 for s in result.sigma):
        reason = "too narrow to be a spot: a single pixel"
    else:
        return None
    return _with(result, FitOutcome.FAILED, reason)


def _at_edge(result: FitResult, reason: str) -> FitResult:
    return _with(result, FitOutcome.AT_EDGE, reason)


def _after_recentre(again: FitResult, first: FitResult) -> FitResult:
    """The re-centred fit, marked; still at the edge, it stays unconfirmed.

    Re-centring confirms a feature that straddles the first window's edge,
    which leaves its centre at most about two widths outside it with its side
    still inside. An answer further out is a different feature, found by
    looking somewhere else, and is not kept.
    """
    if again.ok:
        for p, s, (lo, hi) in zip(again.position, again.sigma, first.window):
            reach = 2.0 * max(s, 1.0)
            if not lo - reach <= p <= hi + reach:
                return _with(
                    again,
                    FitOutcome.AT_EDGE,
                    "at the window edge; re-centring found a different feature "
                    "outside the first window",
                    recentred=True,
                )
    if again.ok:
        return _with(again, again.outcome, "re-centred once", recentred=True)
    if again.outcome is FitOutcome.AT_EDGE:
        return _with(
            again,
            again.outcome,
            f"still {again.reason} after re-centring once",
            recentred=True,
        )
    return _with(again, again.outcome, again.reason, recentred=True)


def _with(result: FitResult, outcome: FitOutcome, reason: str, **kw) -> FitResult:
    return replace(result, outcome=outcome, reason=reason, **kw)


def _axes(edges) -> str:
    return " and ".join(axis for axis, _ in edges)
