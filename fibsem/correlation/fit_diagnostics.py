"""Fit-diagnostic data + rendering, split from the fit compute (FIB-281).

The correlation fit functions (:mod:`fibsem.correlation.util`) compute a refined
coordinate and return a :class:`FitDiagnostic` — the *data* needed to draw the
diagnostic figure — without touching matplotlib. :func:`plot_fit_diagnostic`
renders it, in a light palette by default or the napari-dark palette
(``dark=True``) used by the confirm dialog.

This keeps the compute path matplotlib-free, builds a figure only when one is
actually shown (no build-then-discard), makes the diagnostics testable on data
rather than on figure internals, and turns the confirm dialog's dark theme into
a render parameter (it no longer re-themes a figure after the fact).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# input (red) / fitted (green) are the only saturated colours in every panel;
# everything else is greyscale so the eye goes to the two markers.
_C_IN = "#e53935"
_C_FIT = "#43a047"


@dataclass
class FitDiagnostic:
    """Everything needed to render a fit's diagnostic figure — no matplotlib.

    The XY panel is always present; the z panel is present only when
    :attr:`z_signal` is set (the FIB hole fit is XY-only). Marker positions are
    in ROI pixel coordinates; :attr:`input_xy` carries the sub-pixel click so the
    input marker lands exactly where the user clicked (FIB-282).
    """

    title: str
    roi_xy: np.ndarray                                 # image shown in the XY panel
    input_xy: Tuple[float, float]                      # red input marker (ROI coords)
    fitted_xy: Optional[Tuple[float, float]] = None    # green marker, or None → not drawn
    xy_title: str = "XY"
    xy_message: Optional[str] = None                   # shown when there is no fitted marker
    # z panel — left None for an XY-only (FIB) fit
    z_axis: Optional[np.ndarray] = None
    z_signal: Optional[np.ndarray] = None
    z_fit: Optional[np.ndarray] = None
    z_input: Optional[float] = None
    z_fitted: Optional[float] = None
    z_inverted: bool = False                           # reflection: signal is inverted

    @property
    def has_z(self) -> bool:
        """True for the FM fits (z + XY); False for the XY-only FIB fit."""
        return self.z_signal is not None


def plot_fit_diagnostic(d: FitDiagnostic, *, dark: bool = False):
    """Render a :class:`FitDiagnostic` to a matplotlib ``Figure``.

    Uses the object-oriented ``Figure`` API (no pyplot), so the figure isn't
    held in pyplot's global registry — it's freed with its canvas and needs no
    ``plt.close``. Pass ``dark=True`` for the napari-dark confirm-dialog palette.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if d.has_z:
        fig = Figure(figsize=(9, 4.5))
        FigureCanvasAgg(fig)  # gives tight_layout a renderer; replaced by the Qt canvas on embed
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


def _draw_z_panel(ax, d: FitDiagnostic) -> None:
    ax.plot(d.z_axis, d.z_signal, color="0.55", lw=1.3)
    if d.z_fit is not None:
        ax.plot(d.z_axis, d.z_fit, color="0.15", ls="--", lw=1.2)
    ax.axvline(d.z_input, color=_C_IN, ls="--", lw=1.6, label=f"input {d.z_input:.1f}")
    ax.axvline(d.z_fitted, color=_C_FIT, ls="--", lw=1.6, label=f"fitted {d.z_fitted:.1f}")
    ax.set_title("z", fontsize=10)
    ax.set_xlabel("z slice", fontsize=8)
    if d.z_inverted:
        # the hole is dark, so the signal is inverted — the peak is the hole
        ax.set_ylabel("signal (inverted)", fontsize=8)
    ax.set_yticks([])
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, framealpha=0.6)


def _draw_xy_panel(ax, d: FitDiagnostic) -> None:
    ax.imshow(d.roi_xy, cmap="gray")
    ax.plot(d.input_xy[0], d.input_xy[1], "+", color=_C_IN, ms=16, mew=2, label="input")
    if d.fitted_xy is not None:
        ax.plot(d.fitted_xy[0], d.fitted_xy[1], "+", color=_C_FIT, ms=16, mew=2,
                label="fitted")
    elif d.xy_message:
        # A failed / out-of-region fit has no marker to show; say so instead of
        # letting matplotlib silently chase a marker off-axes.
        ax.text(0.5, 0.5, d.xy_message, transform=ax.transAxes, ha="center",
                va="center", color=_C_IN, fontsize=11,
                bbox=dict(boxstyle="round", fc="black", ec=_C_IN, alpha=0.65))
    n = d.roi_xy.shape[0]
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_title(d.xy_title)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.6)
    ax.axis("off")


# --- dark theme -----------------------------------------------------------
# The napari-dark palette for the confirm dialog. This is the exact re-theming
# that shipped in #144 (formerly _apply_dark_theme in fit_confirmation_dialog);
# it now lives here so the dialog just passes dark=True. Only theme-agnostic
# chrome is touched — the saturated input/fitted markers and the grayscale image
# are left as authored (they read correctly on dark).
_FIG_BG = "#1e2124"    # == the dialog background: figure margins blend in
_AXES_BG = "#262930"   # a subtle lift so the z line-plot reads as a panel
_FG = "#d0d0d0"        # labels, ticks, legend text
_FG_TITLE = "#e0e0e0"  # titles / suptitle, a touch brighter
_SPINE = "#4a4d53"
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
