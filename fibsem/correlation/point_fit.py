"""Fitting one picked point: refine a click onto the feature under it.

A point fit (hole centroid, gaussian) proposes a refined position for a
coordinate. It never moves the coordinate: the proposal is a
:class:`PointFitResult`, which the correlation tab shows for the user to accept
or reject and which anything else fitting points (a service, a script) gets
back the same way.

The fitters (:mod:`fibsem.imaging.localise`) return a :class:`PointFitDiagnostic`,
the *data* needed to draw the diagnostic figure, without touching matplotlib.
:func:`plot_fit_diagnostic` renders it on demand, in a light palette by
default or the dark palette (``dark=True``) the confirm dialog uses, so the
compute path stays matplotlib-free and a figure is built only when one is shown.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

from fibsem.correlation.config import FitSettings
from fibsem.correlation.structures import Coordinate, PointXYZ
from fibsem.theme.tokens import CANVAS_BG, NEUTRAL_200, NEUTRAL_300, SURFACE_COLOR

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from fibsem.imaging.localise import PointFitDiagnostic

__all__ = [
    "FitStatus",
    "PointFitResult",
    "refine_point",
    "humanize_fit_error",
    "plot_fit_diagnostic",
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
    # the fit's answer left its window, or found no burn
    if "fell outside the" in msg or "no burn-shaped" in msg:
        return (
            "No clear burn here: the fit's answer fell outside the region it "
            "searched, so the point stays where you placed it. Check the "
            "channel, or place it on the burn by hand."
        )
    if "plane(s) around z" in msg:  # the reflection fit's own refusal
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
                # input marker lands where the user clicked.
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
                    z=int(z),  # sub-pixel x/y
                    cutout=settings.reflection_cutout,
                )
                fitted = PointXYZ(float(xr), float(yr), float(zr))
            elif method == "Gaussian":
                attempted = True
                xr, yr, zr, diag = localise.target_fitting_fluorescence(
                    img,
                    x,
                    y,
                    int(z),  # sub-pixel x/y
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


# ── the diagnostic figure ────────────────────────────────────────────────

# input (red) / fitted (green) are the only saturated colours in every panel;
# everything else is greyscale so the eye goes to the two markers.
_C_IN = "#e53935"
_C_FIT = "#43a047"


def plot_fit_diagnostic(d: PointFitDiagnostic, *, dark: bool = False):
    """Render a :class:`PointFitDiagnostic` to a matplotlib ``Figure``.

    Uses the object-oriented ``Figure`` API (no pyplot), so the figure isn't
    held in pyplot's global registry — it's freed with its canvas and needs no
    ``plt.close``. Pass ``dark=True`` for the dark confirm-dialog palette.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if d.has_z:
        fig = Figure(figsize=(9, 4.5))
        FigureCanvasAgg(
            fig
        )  # gives tight_layout a renderer; replaced by the Qt canvas on embed
        ax_z, ax_xy = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.4]})
        _draw_z_panel(ax_z, d)
    else:
        fig = Figure(figsize=(5, 5))
        FigureCanvasAgg(fig)
        ax_xy = fig.subplots(1, 1)

    _draw_xy_panel(ax_xy, d)
    fig.suptitle(d.title)

    if dark:
        _apply_dark_theme(fig)
    fig.tight_layout()
    return fig


def _draw_z_panel(ax, d: PointFitDiagnostic) -> None:
    ax.plot(d.z_axis, d.z_signal, color="0.55", lw=1.3)
    if d.z_fit is not None:
        ax.plot(d.z_axis, d.z_fit, color="0.15", ls="--", lw=1.2)
    ax.axvline(d.z_input, color=_C_IN, ls="--", lw=1.6, label=f"input {d.z_input:.1f}")
    ax.axvline(
        d.z_fitted, color=_C_FIT, ls="--", lw=1.6, label=f"fitted {d.z_fitted:.1f}"
    )
    ax.set_title("z", fontsize=10)
    ax.set_xlabel("z slice", fontsize=8)
    if d.z_inverted:
        # the hole is dark, so the signal is inverted — the peak is the hole
        ax.set_ylabel("signal (inverted)", fontsize=8)
    ax.set_yticks([])
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, framealpha=0.6)


def _draw_xy_panel(ax, d: PointFitDiagnostic) -> None:
    ax.imshow(d.roi_xy, cmap="gray")
    ax.plot(d.input_xy[0], d.input_xy[1], "+", color=_C_IN, ms=16, mew=2, label="input")
    if d.fitted_xy is not None:
        ax.plot(
            d.fitted_xy[0],
            d.fitted_xy[1],
            "+",
            color=_C_FIT,
            ms=16,
            mew=2,
            label="fitted",
        )
    elif d.xy_message:
        # A failed / out-of-region fit has no marker to show; say so instead of
        # letting matplotlib silently chase a marker off-axes.
        ax.text(
            0.5,
            0.5,
            d.xy_message,
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=_C_IN,
            fontsize=11,
            bbox=dict(boxstyle="round", fc="black", ec=_C_IN, alpha=0.65),
        )
    n = d.roi_xy.shape[0]
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_title(d.xy_title)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.6)
    ax.axis("off")


# --- dark theme -----------------------------------------------------------
# The dark palette for the confirm dialog, which passes dark=True. Only theme-agnostic
# chrome is touched — the saturated input/fitted markers and the grayscale image
# are left as authored (they read correctly on dark).
_FIG_BG = CANVAS_BG  # == the dialog background: figure margins blend in
_AXES_BG = SURFACE_COLOR  # a subtle lift so the z line-plot reads as a panel
_FG = NEUTRAL_300  # labels, ticks, legend text
_FG_TITLE = NEUTRAL_200  # titles / suptitle, a touch brighter
_SPINE = "#4a4d53"  # no token; the shared plot theme replaces it
# A line darker than this is invisible on the dark panel; the only such artist
# is the gaussian-fit overlay (drawn at 0.15 grey). Lift it to a light grey.
_DARK_LINE_LUM = 0.25
_DARK_LINE_TARGET = "0.8"


def _luminance(color) -> float:
    from matplotlib.colors import to_rgb

    r, g, b = to_rgb(color)
    return 0.299 * r + 0.587 * g + 0.114 * b


def _apply_dark_theme(fig) -> None:
    fig.set_facecolor(_FIG_BG)
    for txt in fig.texts:  # figure-level text == the suptitle
        txt.set_color(_FG_TITLE)
    for ax in fig.axes:
        ax.set_facecolor(_AXES_BG)
        ax.title.set_color(_FG_TITLE)
        ax.xaxis.label.set_color(_FG)
        ax.yaxis.label.set_color(_FG)
        ax.tick_params(colors=_FG)
        for spine in ax.spines.values():
            spine.set_color(_SPINE)
        for line in ax.get_lines():
            if _luminance(line.get_color()) < _DARK_LINE_LUM:
                line.set_color(_DARK_LINE_TARGET)
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(_AXES_BG)
            legend.get_frame().set_edgecolor(_SPINE)
            for text in legend.get_texts():
                text.set_color(_FG)
