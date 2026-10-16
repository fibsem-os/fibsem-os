"""The outcome of fitting one correlation point, and how it is shown.

A point fit proposes a refined position for a coordinate. Rather than mutating
the coordinate, the fit is captured as a :class:`PointFitResult`, which the fit
panel (:mod:`fibsem.ui.correlation.widgets.fit_panel`) shows for the user to
accept, change or keep their click. This module holds the result, its coarse
status, the plain-words failure messages and the small formatting helpers the
panel uses. The Confirm fit dialog that used to show it was replaced by the
fit panel (FIB-1218).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QWidget,
)

from fibsem.correlation.structures import Coordinate, PointXYZ
from fibsem.ui.tokens import (
    BODY_MUTED_STYLE,
    BODY_STYLE,
    TEXT_COLOR,
    state_color,
)

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
    diagnostic: object = None  # FitDiagnostic; rendered on demand, not serialised

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


# (background, foreground) per status — napari-dark chips
# Chip colours by outcome: the semantic text colour on the panel ground, with a
# border in the same colour, rather than a tinted background of its own.
_STATUS_STYLE = {
    FitStatus.OK: state_color("ok"),
    FitStatus.UNCHANGED: state_color("warn"),
    FitStatus.ERROR: state_color("error"),
}


def _status_text(result: "PointFitResult") -> str:
    if result.status is FitStatus.OK:
        return "Fit ok"
    if result.status is FitStatus.UNCHANGED:
        return "No change — fit ≈ original"
    return "Fit failed"  # the (long) reason goes in a separate wrapped line


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
            "No clear burn here, so the point stays where you placed it. "
            "Change the window or the channel, or place it on the burn by hand."
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


def _fmt_xyz(p: PointXYZ) -> str:
    return f"{p.x:.3f}, {p.y:.3f}, {p.z:.3f}"


def _fmt_delta(initial: PointXYZ, fitted: PointXYZ) -> str:
    return (
        f"{fitted.x - initial.x:+.3f}, "
        f"{fitted.y - initial.y:+.3f}, "
        f"{fitted.z - initial.z:+.3f}"
    )


def _kv(key: str, value: str, value_color: str = TEXT_COLOR) -> QWidget:
    w = QWidget()
    row = QHBoxLayout(w)
    row.setContentsMargins(0, 0, 0, 0)
    k = QLabel(key)
    k.setStyleSheet(BODY_MUTED_STYLE)
    v = QLabel(value)
    v.setStyleSheet(f"{BODY_STYLE} color: {value_color};")
    v.setTextFormat(Qt.TextFormat.PlainText)
    row.addWidget(k)
    row.addStretch(1)
    row.addWidget(v)
    return w
