"""Fitting one picked point: refine a click onto the feature under it.

A point fit (hole centroid, gaussian) proposes a refined position for a
coordinate. It never moves the coordinate: the proposal is a
:class:`PointFitResult`, which the correlation tab shows for the user to accept
or reject and which anything else fitting points (a service, a script) gets
back the same way.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from fibsem.correlation.config import FitSettings
from fibsem.correlation.structures import Coordinate, PointXYZ

__all__ = [
    "FitStatus",
    "PointFitResult",
    "refine_point",
    "humanize_fit_error",
]

# Per-axis displacement below which a fit is treated as "no change" (the fit
# fell back to the input). Well under the 3-decimal coordinate precision, so
# genuine sub-pixel refinements read as real moves, not "no change".
_UNCHANGED_EPS = 0.001


class FitStatus(Enum):
    OK = "ok"  # fit moved the point
    UNCHANGED = "unchanged"  # fit landed on (≈) the input position
    ERROR = "error"  # fit raised / produced no position


@dataclass
class PointFitResult:
    """Outcome of auto-fitting one coordinate (runtime only)."""

    coordinate: Coordinate
    method: str
    channel: Optional[int]
    initial: PointXYZ
    fitted: Optional[PointXYZ]
    status: FitStatus
    message: Optional[str] = None
    channel_name: Optional[str] = None  # display label for `channel` (FM only)
    detail: Optional[str] = None  # raw error, surfaced as a tooltip
    diagnostic: object = None  # PointFitDiagnostic; rendered on demand, not serialised

    @property
    def delta_px(self) -> float:
        """XY displacement between the initial and fitted position (pixels)."""
        if self.fitted is None:
            return 0.0
        dx = self.fitted.x - self.initial.x
        dy = self.fitted.y - self.initial.y
        return (dx * dx + dy * dy) ** 0.5

    @property
    def delta_z(self) -> float:
        if self.fitted is None:
            return 0.0
        return self.fitted.z - self.initial.z

    @property
    def delta(self) -> Optional[tuple]:
        """Per-axis (dx, dy, dz) displacement, or None when there's no fit."""
        if self.fitted is None:
            return None
        return (
            self.fitted.x - self.initial.x,
            self.fitted.y - self.initial.y,
            self.fitted.z - self.initial.z,
        )

    @staticmethod
    def classify(
        initial: PointXYZ,
        fitted: Optional[PointXYZ],
        *,
        error: Optional[str] = None,
    ) -> FitStatus:
        """Coarse status from the initial/fitted positions (no fit-fn changes)."""
        if error is not None or fitted is None:
            return FitStatus.ERROR
        deltas = (
            fitted.x - initial.x,
            fitted.y - initial.y,
            fitted.z - initial.z,
        )
        moved = any(abs(d) >= _UNCHANGED_EPS for d in deltas)
        return FitStatus.OK if moved else FitStatus.UNCHANGED


def humanize_fit_error(exc: Exception) -> str:
    """Turn a raw fit exception into an actionable, non-jargon message.

    scipy surfaces failures like *"Optimal parameters not found: Number of calls
    to function has reached maxfev = 800"* — meaningless to a user. Map the known
    modes to plain guidance; the raw text is still logged (and shown as a tooltip
    via :attr:`PointFitResult.detail`) so debugging isn't lost.
    """
    msg = str(exc)
    if "maxfev" in msg or "Optimal parameters not found" in msg:
        return (
            "The fit didn't converge — the feature may be too faint, or not "
            "centred in the search region. Try re-centring your click, or a "
            "different channel / method."
        )
    if isinstance(exc, IndexError) or "out of bounds" in msg or "too close" in msg:
        return "The point is too close to the image edge to fit a region around it."
    # the fit's answer left its window (FIB-1219), or found no burn (FIB-1216)
    if "fell outside the" in msg or "no burn-shaped" in msg:
        return (
            "No clear burn here: the fit's answer fell outside the region it "
            "searched, so the point stays where you placed it. Check the "
            "channel, or place it on the burn by hand."
        )
    if "plane(s) around z" in msg:  # the reflection fit's own refusal (FIB-980)
        return (
            "Too few planes around this z to fit a hole through. Pick a slice "
            "further from the ends of the stack, or acquire a deeper stack."
        )
    if isinstance(exc, (ValueError, ZeroDivisionError)) and (
        "empty" in msg or "zero-size" in msg or "float division" in msg
    ):
        return "The search region is empty or featureless — nothing to fit."
    return "The fit failed unexpectedly (see log for details)."


def refine_point(
    coord: Coordinate,
    side: str,
    method: str,
    *,
    fib_image=None,
    fm_image=None,
    channel: Optional[int] = None,
    channel_name: Optional[str] = None,
    settings: Optional[FitSettings] = None,
) -> Optional[PointFitResult]:
    """Fit ``coord`` WITHOUT mutating it.

    ``side`` is ``"fib"`` or ``"fm"``. A FIB point fits a hole in
    ``fib_image``'s filtered data (``method`` "Hole"); an FM point fits
    ``fm_image``'s ``channel`` (``method`` "Hole" for a reflection hole,
    "Gaussian" for a fluorescent target). Returns None when no fit applies
    (method "None", image missing); otherwise a :class:`PointFitResult`
    carrying the proposed position, coarse status and the diagnostic figure.
    A fit that raises comes back as an ERROR result, never as an exception.
    """
    from fibsem.imaging import localise

    settings = settings or FitSettings()
    x, y, z = coord.point.x, coord.point.y, coord.point.z
    initial = PointXYZ(x, y, z)
    fitted, diag, error, error_detail, attempted = None, None, None, None, False
    if side != "fm":
        channel, channel_name = None, None

    try:
        if side == "fib":
            if method == "Hole" and fib_image is not None:
                attempted = True
                # pass the sub-pixel click (not int) so the diagnostic's
                # input marker lands where the user clicked — FIB-282.
                xr, yr, diag = localise.hole_fitting_FIB(fib_image.filtered_data, x, y)
                fitted = PointXYZ(float(xr), float(yr), z)
        elif (
            method != "None"
            and fm_image is not None
            and channel is not None
            and channel >= 0
        ):
            img = fm_image.data[channel]
            if method == "Hole":
                attempted = True
                xr, yr, zr, diag = localise.hole_fitting_reflection(
                    img,
                    x,
                    y,
                    z=int(z),  # sub-pixel x/y (FIB-282)
                    cutout=settings.reflection_cutout,
                )
                fitted = PointXYZ(float(xr), float(yr), float(zr))
            elif method == "Gaussian":
                attempted = True
                xr, yr, zr, diag = localise.target_fitting_fluorescence(
                    img,
                    x,
                    y,
                    int(z),  # sub-pixel x/y (FIB-282)
                    cutout=settings.fluorescence_cutout,
                )
                fitted = PointXYZ(float(xr), float(yr), float(zr))
    except Exception as exc:
        logging.exception("Point fit failed")
        error_detail = str(exc)  # raw text -> log + tooltip
        error = humanize_fit_error(exc)  # user-facing, actionable
        attempted = True

    if not attempted:
        return None

    status = PointFitResult.classify(initial, fitted, error=error)
    return PointFitResult(
        coordinate=coord,
        method=method,
        channel=channel,
        channel_name=channel_name,
        initial=initial,
        fitted=fitted,
        status=status,
        message=error,
        detail=error_detail,
        diagnostic=diag,
    )
