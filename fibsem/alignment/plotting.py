from __future__ import annotations

import os
from typing import Optional, TYPE_CHECKING

import numpy as np

from fibsem import utils
from fibsem.constants import DATETIME_DISPLAY
from fibsem.alignment.verified import DEFAULT_CONVERGENCE_TOLERANCE_PX
from fibsem.structures import ImageSettings

if TYPE_CHECKING:
    from fibsem.alignment import AlignmentDifferential, AlignmentIteration
    from fibsem.structures import FibsemImage
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes


def _plot_image_with_crosshair(ax: Axes, data: np.ndarray, title: str) -> None:
    """Plot an image with a yellow crosshair at the centre."""
    ax.imshow(data, cmap="gray")
    cy, cx = data.shape[0] // 2, data.shape[1] // 2
    ax.axhline(cy, color="yellow", linewidth=2, alpha=0.7)
    ax.axvline(cx, color="yellow", linewidth=2, alpha=0.7)
    ax.set_title(title)
    ax.axis("off")


def _alignment_save_path(ref_image: FibsemImage) -> tuple:
    """Return (ref_path, prefix, ts) for saving alignment plots."""
    from datetime import datetime
    from fibsem.alignment import ALIGNMENT_SUBDIR

    ref_settings = ImageSettings.fromFibsemImage(ref_image)
    ref_filename = ref_settings.filename
    ref_path = ref_settings.path if ref_settings.path is not None else os.getcwd()
    ref_path = os.path.join(ref_path, ALIGNMENT_SUBDIR)
    os.makedirs(ref_path, exist_ok=True)
    from fibsem.config import REFERENCE_FILENAME

    prefix = (
        ref_filename.split(REFERENCE_FILENAME)[0]
        if REFERENCE_FILENAME in ref_filename
        else ref_filename + "_"
    )
    ts = utils.current_timestamp_v2()
    return ref_path, prefix, ts


# Verdict colours. Semantic, and deliberately not the inferno ramp used for the
# correlation surfaces -- a reader should never have to ask whether a colour means
# "this is the peak" or "this is the verdict".
_OK, _BAD, _DIM = "#2F6F58", "#B23257", "#6E6880"
_UNSURE = "#B45F0E"
_CROSSHAIR = "#FFD166"


def _panel(
    ax: Axes,
    data: np.ndarray,
    title: str,
    subtitle: str = "",
    colour: Optional[str] = None,
    xcorr: Optional[np.ndarray] = None,
    scale: float = 1.0,
) -> None:
    """One image in the timeline, with an optional correlation surface inset.

    The inset sits on the step it belongs to rather than in a separate strip: the
    surface is read as "sharp peak or mush", and that judgement is only useful
    next to the image that produced it.
    """
    ax.imshow(data, cmap="gray")
    h, w = data.shape
    ax.axhline(h / 2, color=_CROSSHAIR, lw=0.7, alpha=0.9)
    ax.axvline(w / 2, color=_CROSSHAIR, lw=0.7, alpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9.5 * scale, color=colour or "black", pad=4)
    if subtitle:
        ax.set_xlabel(subtitle, fontsize=8 * scale, color=colour or _DIM, labelpad=3)
    for spine in ax.spines.values():
        spine.set_color(colour or "#cccccc")
        spine.set_linewidth(1.6 if colour else 0.8)
    if xcorr is not None:
        # Sized off the SHORTER side so the inset stays readable on a wide frame,
        # where a fraction-of-width box would be a thin letterbox.
        frac_w = 0.34 if w <= h else 0.34 * (h / w)
        frac_h = 0.34 if h <= w else 0.34 * (w / h)
        inset = ax.inset_axes([0.98 - frac_w, 0.02, frac_w, frac_h])
        # Stretch to the bulk of the surface, not its single brightest pixel. A
        # sharp peak is one bright pixel against a mid-grey field; on the default
        # scale that renders as an empty dark box at inset size, which is exactly
        # the judgement -- sharp peak or mush -- the inset exists to support.
        surf = np.asarray(xcorr, dtype=float)
        lo, hi = np.percentile(surf, 2), np.percentile(surf, 99.5)
        inset.imshow(surf, cmap="inferno",
                     vmin=lo, vmax=hi if hi > lo else None)
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_color(_CROSSHAIR)
            spine.set_linewidth(0.9)


def plot_multi_step_alignment(
    ref_image: FibsemImage,
    alignment_results: list[AlignmentIteration],
    title: Optional[str] = None,
    save: bool = True,
    final_image: Optional[FibsemImage] = None,
    path: Optional[str] = None,
    validation: Optional[AlignmentDifferential] = None,
    final_residual_px: Optional[float] = None,
    converged: Optional[bool] = None,
    tolerance_px: Optional[float] = None,
):
    """Plot an alignment run as a timeline, with the verdict stated once.

    Reference, each step, and the final image sit in one row, because that is the
    order they happened in. Correlation surfaces are insets on their own step. The
    lower row carries the verdict and the only chart that answers the question the
    plot exists for: did the run get closer to the reference?

    That chart plots `measured_px` -- what each step SAW -- not the shift it
    applied. A refused step applies nothing, so plotting applied shifts draws a
    refused run converging neatly to zero, which is the opposite of what happened.

    Args:
        ref_image: the reference image aligned to.
        alignment_results: the per-step results.
        final_image: image acquired after the last step, if one was taken.
        validation: cross-method comparison on the final image.
        final_residual_px: how far the final image sits from the reference.
        converged: whether that is within tolerance. None when not measured.
        tolerance_px: the limit used, drawn on the chart.

    Returns:
        matplotlib.figure.Figure
    """
    from datetime import datetime

    from matplotlib.figure import Figure
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Rectangle

    ref_path, prefix, ts = _alignment_save_path(ref_image)
    if path is not None:
        ref_path = path
    ref_filename = ImageSettings.fromFibsemImage(ref_image).filename
    timestamp_str = datetime.now().strftime(DATETIME_DISPLAY)
    if title is None:
        title = f"Multi-Step Alignment — {ref_filename} — {timestamp_str}"
    else:
        title = f"{title} — {timestamp_str}"

    n = len(alignment_results)
    n_cols = n + (2 if final_image is not None else 1)
    if tolerance_px is None:
        tolerance_px = DEFAULT_CONVERGENCE_TOLERANCE_PX

    # Adapt the row heights to the image shape. Alignment ROIs range from tall
    # and narrow (a fiducial crop, ~366x464) to short and wide (a full frame,
    # ~1312x853); a fixed row height leaves a dead band above the chart on the
    # wide ones and squashes the tall ones.
    _CHART_H = 2.45          # inches for the verdict + chart row
    _MAX_W = 20.0            # a figure wider than this is unreadable anywhere
    grid_cols_pre = max(n_cols, 3)
    # Narrow the columns rather than growing without bound: a run with many steps
    # would otherwise produce a figure several feet wide.
    _PANEL_W = min(3.05, _MAX_W / grid_cols_pre)
    h_px, w_px = ref_image.data.shape
    aspect = h_px / w_px if w_px else 1.0
    # keep the image row within sane bounds regardless of an extreme aspect
    top_h = min(max(_PANEL_W * aspect, 1.7), 4.4)
    grid_cols = max(n_cols, 3)
    fig = Figure(figsize=(_PANEL_W * grid_cols, top_h + _CHART_H))
    gs = GridSpec(2, grid_cols, figure=fig, height_ratios=[top_h, _CHART_H],
                  hspace=0.34, wspace=0.30)

    # Narrow columns need smaller labels, and the verdict block needs more than one
    # of them or its text runs into the chart.
    scale = min(1.0, _PANEL_W / 3.05)
    v_cols = 1 if _PANEL_W >= 2.6 else 2

    n_refused = sum(1 for r in alignment_results if not r.accepted)
    colour = _BAD if converged is False else (
        _UNSURE if (validation is not None and not validation.agreement) else _OK)

    # --- row 0: the timeline, in the order it happened -----------------------
    _panel(fig.add_subplot(gs[0, 0]), ref_image.data, "Reference", "the target",
           scale=scale)
    for i, r in enumerate(alignment_results):
        step_colour = None if r.accepted else _BAD
        if r.accepted:
            sub = f"applied {r.shift_px.x:+.0f}, {r.shift_px.y:+.0f} px"
            if r.clipped:
                sub += "  (clipped)"
        else:
            sub = f"REFUSED ({r.disagreement_px:.0f} px apart)"
        _panel(fig.add_subplot(gs[0, 1 + i]), r.image.data, f"Step {i + 1}",
               sub, step_colour, r.xcorr, scale=scale)
    if final_image is not None:
        sub = f"{final_residual_px:.1f} px out" if final_residual_px is not None else ""
        _panel(fig.add_subplot(gs[0, n + 1]), final_image.data, "Final", sub, colour,
               scale=scale)

    # --- row 1, left: the verdict, in exactly one place ----------------------
    axv = fig.add_subplot(gs[1, 0:v_cols])
    axv.axis("off")
    # The verdict rests on ONE method's measurement of the final image. When the
    # methods disagree about where that image sits, a green pass would be claiming
    # more than the data supports -- so agreement gates the confident verdict, but
    # never rescues a bad residual: a large residual is a failure either way.
    corroborated = validation is None or validation.agreement
    if converged is None:
        verdict = "NOT MEASURED" if final_image is None else "NO VERDICT"
        vcolour = _DIM
    elif not converged:
        verdict = "NOT CONVERGED"
        vcolour = _BAD
    elif not corroborated:
        verdict = "UNCERTAIN"
        vcolour = _UNSURE
    else:
        verdict = "ALIGNED"
        vcolour = _OK
    axv.add_patch(Rectangle((0, 0.06), 1, 0.88, transform=axv.transAxes,
                            facecolor=vcolour, alpha=0.10,
                            edgecolor=vcolour, linewidth=1.4))
    axv.text(0.5, 0.78, verdict, transform=axv.transAxes, fontsize=12 * scale,
             fontweight="bold", color=vcolour, ha="center", va="center")
    lines = []
    if final_residual_px is not None:
        lines.append(f"{final_residual_px:.1f} px out  (limit {tolerance_px:.0f})")
    lines.append(f"{n - n_refused}/{n} steps applied")
    if validation is not None:
        if validation.agreement:
            lines.append(f"methods agree ({validation.max_disagreement_px:.1f} px)")
        else:
            lines.append(f"methods DISAGREE by {validation.max_disagreement_px:.0f} px")
            lines.append("so the number above")
            lines.append("is not corroborated")
    axv.text(0.5, 0.34, "\n".join(lines), transform=axv.transAxes, fontsize=8.0 * scale,
             color="#333333", ha="center", va="center", family="monospace",
             linespacing=1.7)

    # --- row 1, right: did it get closer? ------------------------------------
    axc = fig.add_subplot(gs[1, v_cols:])
    xs = list(range(1, n + 1))
    # what each step SAW, not what it applied -- see the docstring
    ys = [r.measured_px for r in alignment_results]
    labels = [(f"before step {i}" if n <= 5 else f"S{i}") for i in xs]
    if final_residual_px is not None:
        xs.append(n + 1)
        ys.append(final_residual_px)
        labels.append("final")

    finite = [(x, y) for x, y in zip(xs, ys) if y is not None and np.isfinite(y)]
    # Records written before measured_px existed have no per-step values. Say so:
    # a chart showing only the final point otherwise reads as "one measurement"
    # rather than "the step measurements were never recorded".
    n_missing = sum(1 for r in alignment_results if not np.isfinite(r.measured_px))
    if n_missing and finite:
        axc.text(0.5, 0.92, f"step measurements not recorded for this run "
                            f"({n_missing}/{n})", transform=axc.transAxes,
                 ha="center", va="top", fontsize=7.5, color=_DIM, style="italic")
    if finite:
        fx, fy = zip(*finite)
        axc.axhspan(0, tolerance_px, color=_OK, alpha=0.07)
        axc.axhline(tolerance_px, color=_OK, lw=1.1, ls=":")
        if final_residual_px is not None and len(fx) > 1:
            axc.plot(fx[:-1], fy[:-1], "o-", color="#4A4458", lw=1.8, ms=6)
            axc.plot(fx[-2:], fy[-2:], "--", color=colour, lw=1.5)
            axc.plot(fx[-1], fy[-1], "o", color=colour, ms=9)
        else:
            axc.plot(fx, fy, "o-", color="#4A4458", lw=1.8, ms=6)
        for x, y in finite:
            axc.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                         xytext=(0, 9), ha="center", fontsize=8, color=_DIM)
        # a refused step measured something and moved nothing; say so on the point
        for i, r in enumerate(alignment_results):
            if not r.accepted and np.isfinite(r.measured_px):
                axc.annotate("refused", (i + 1, r.measured_px),
                             textcoords="offset points", xytext=(0, -15),
                             ha="center", fontsize=7.5, color=_BAD)
        axc.text(fx[0], tolerance_px, " tolerance", fontsize=7.5, color=_OK,
                 va="bottom", ha="left")
        axc.margins(x=0.07, y=0.22)
    else:
        axc.text(0.5, 0.5, "no measurements recorded", transform=axc.transAxes,
                 ha="center", va="center", fontsize=9, color=_DIM)

    axc.set_xticks(xs)
    axc.set_xticklabels(labels, fontsize=8)
    axc.set_ylabel("px from reference", fontsize=8.5, labelpad=2)
    axc.tick_params(labelsize=8)
    axc.set_title("Convergence", fontsize=9.5, loc="left", pad=6)
    axc.grid(axis="y", alpha=0.18, lw=0.6)
    for spine in ("top", "right"):
        axc.spines[spine].set_visible(False)

    fig.suptitle(title, fontsize=11, y=0.98)
    if save:
        save_path = os.path.join(ref_path, "figure.png")
        fig.savefig(save_path, dpi=90, bbox_inches="tight", facecolor="white")
    return fig


def plot_preprocessing_comparison(
    ref_image: FibsemImage,
    new_image: FibsemImage,
    lowpass: int = 50,
    highpass: int = 4,
    sigma: int = 5,
    title: Optional[str] = None,
    save: bool = False,
    path: Optional[str] = None,
):
    """Compare the zscore and dog preprocessing profiles on one image pair.

    Shows, per profile, what the correlation was fed and the surface it produced,
    and reports how far apart the two answers are. That distance is the quantity a
    validated alignment mode gates on: on a healthy pair the profiles agree to
    around a pixel, and where the intensity path false-locks they disagree by tens
    to hundreds of pixels (see FIB-711).

    Use this to inspect a suspect alignment, or to check the agreement tolerance
    against a new instrument before trusting a validated mode on it.

    Args:
        ref_image: the reference image.
        new_image: the newly acquired image.
        lowpass, highpass, sigma: Fourier band-pass parameters, applied to both profiles.
        title: figure title. Defaults to a generated one.
        save: whether to write the figure next to the other alignment diagnostics.
        path: directory to save into. Defaults to the usual alignment save path.

    Returns:
        matplotlib.figure.Figure
    """
    from matplotlib.figure import Figure

    from fibsem.alignment.methods import (
        PREPROCESSING_PROFILES,
        get_preprocessing_profile,
        shift_from_crosscorrelation,
    )

    pixel_size = new_image.metadata.pixel_size.x if new_image.metadata else 1.0
    kw = dict(lowpass=lowpass, highpass=highpass, sigma=sigma)

    results = {}
    for name in PREPROCESSING_PROFILES:
        dx, dy, xcorr, score = shift_from_crosscorrelation(
            ref_image, new_image, **get_preprocessing_profile(name), **kw
        )
        results[name] = (dx / pixel_size, dy / pixel_size, xcorr, score)

    (zx, zy, _, _) = results["zscore"]
    (dx_, dy_, _, _) = results["dog"]
    disagreement = float(np.hypot(zx - dx_, zy - dy_))

    if title is None:
        title = "Preprocessing comparison"
    fig = Figure(figsize=(4 * len(results), 8))
    fig.suptitle(f"{title} — profiles disagree by {disagreement:.1f} px")
    axes = fig.subplots(2, len(results))

    for col, (name, (sx, sy, xcorr, score)) in enumerate(results.items()):
        prof = get_preprocessing_profile(name)
        _plot_image_with_crosshair(
            axes[0, col], xcorr, f"{name} — xcorr (score {score:.2f})"
        )
        axes[1, col].axis("off")
        axes[1, col].text(
            0.02,
            0.95,
            f"shift  ({sx:+.1f}, {sy:+.1f}) px\n"
            f"preprocess   {prof['preprocess']}\n"
            f"rect mask    {prof['use_rect_mask']}\n"
            f"hann window  {prof['use_hann_window']}\n"
            f"subpixel     {prof['subpixel']}",
            va="top",
            family="monospace",
            fontsize=9,
            transform=axes[1, col].transAxes,
        )

    if save:
        save_path = path if path is not None else _alignment_save_path(ref_image)[0]
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(
            os.path.join(save_path, "preprocessing_comparison.png"), bbox_inches="tight"
        )
    return fig
