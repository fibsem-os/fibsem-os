from __future__ import annotations

import os
from typing import Optional, TYPE_CHECKING

import numpy as np

from fibsem import utils
from fibsem.constants import DATETIME_DISPLAY
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


def plot_multi_step_alignment(
    ref_image: FibsemImage,
    alignment_results: list[AlignmentIteration],
    title: Optional[str] = None,
    save: bool = True,
    final_image: Optional[FibsemImage] = None,
    path: Optional[str] = None,
    validation: Optional[AlignmentDifferential] = None,
    final_residual_px: Optional[float] = None,
):
    """Plot the reference image and each alignment step with cross-correlation maps.

    Args:
        ref_image: The reference image used for alignment.
        alignment_results: List of AlignmentIteration from multi_step_alignment_v2.
        save: Whether to save the figure to disk. Defaults to True.
        final_image: Optional post-alignment image acquired after all steps. When provided,
            a third row is added comparing the reference and final images side by side.

    Returns:
        matplotlib.figure.Figure
    """
    from datetime import datetime

    from matplotlib.figure import Figure

    ref_path, prefix, ts = _alignment_save_path(ref_image)
    if path is not None:
        ref_path = path
    ref_filename = ImageSettings.fromFibsemImage(ref_image).filename
    timestamp_str = datetime.now().strftime(DATETIME_DISPLAY)
    if title is None:
        title = f"Multi-Step Alignment — {ref_filename} — {timestamp_str}"
    else:
        title = f"{title} — {timestamp_str}"

    # row 0 = images, row 1 = xcorr/convergence, row 2 = ref vs final (when final_image provided)
    n_cols = 1 + len(alignment_results)
    n_rows = 3 if final_image is not None else 2
    fig = Figure(figsize=(4 * n_cols, 4 * n_rows))
    axes = fig.subplots(n_rows, n_cols)
    fig.suptitle(title)

    # row 0: reference + each alignment step image
    _plot_image_with_crosshair(axes[0, 0], ref_image.data, "Reference")
    for i, r in enumerate(alignment_results):
        _plot_image_with_crosshair(axes[0, 1 + i], r.image.data, f"Step {i + 1}")
        pixel_size = r.image.metadata.pixel_size.x if r.image.metadata else 1.0
        dx_px = r.shift.x / pixel_size
        dy_px = r.shift.y / pixel_size
        colour = "lime" if r.score >= 0.7 else ("orange" if r.score >= 0.5 else "red")
        label = f"dx={dx_px:.1f}px  dy={dy_px:.1f}px"
        if not r.accepted:
            # a refused step moved nothing, so the reported shift is not a correction
            colour = "red"
            label = f"REFUSED ({r.disagreement_px:.1f}px)"
        axes[0, 1 + i].text(
            0.04, 0.04,
            label,
            transform=axes[0, 1 + i].transAxes,
            color=colour, fontsize=7, va="bottom", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6, edgecolor="none"),
        )
        if r.accepted:
            cx, cy = r.image.data.shape[1] // 2, r.image.data.shape[0] // 2
            axes[0, 1 + i].annotate(
                "", xy=(cx + dx_px, cy + dy_px), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="->", color=colour, lw=2),
            )
        else:
            for spine in axes[0, 1 + i].spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(3)

    # row 1: convergence chart + xcorr maps
    step_nums = list(range(1, len(alignment_results) + 1))
    ax_conv: Axes = axes[1, 0]
    if step_nums:
        ax_conv.plot(
            step_nums,
            [abs(r.shift.x) * 1e9 for r in alignment_results],
            "o-",
            label="dx",
        )
        ax_conv.plot(
            step_nums,
            [abs(r.shift.y) * 1e9 for r in alignment_results],
            "s-",
            label="dy",
        )
        ax_conv.legend(fontsize="small")
        ax_conv.set_xticks(step_nums)
    ax_conv.set_xlabel("Step")
    ax_conv.set_ylabel("Shift (nm)")
    n_refused = sum(1 for r in alignment_results if not r.accepted)
    if n_refused:
        ax_conv.set_title(f"Convergence — {n_refused} step(s) refused", color="red")
        ax_conv.text(
            0.5, 0.5,
            f"{n_refused}/{len(alignment_results)} steps refused;\n"
            "their shifts are zeroed, so a flat\nline here means nothing moved",
            transform=ax_conv.transAxes, ha="center", va="center",
            fontsize=8, color="red", style="italic",
        )
    else:
        ax_conv.set_title("Convergence")

    for i, r in enumerate(alignment_results):
        col = 1 + i
        colour = "lime" if r.score >= 0.7 else ("orange" if r.score >= 0.5 else "red")
        if r.xcorr is not None:
            axes[1, col].imshow(r.xcorr, cmap="inferno")
        extra = ""
        if not np.isnan(r.disagreement_px):
            verdict = "accepted" if r.accepted else "REFUSED"
            extra = f"\nvalidator {r.disagreement_px:.1f}px -> {verdict}"
        if not r.accepted:
            colour = "red"
        axes[1, col].set_title(
            f"XCorr {i + 1}\ndx={r.shift.x * 1e9:.1f}nm, dy={r.shift.y * 1e9:.1f}nm\n"
            f"score={r.score:.2f}{extra}",
            fontsize="small",
            color=colour,
        )
        axes[1, col].axis("off")

    # row 2: ref vs final comparison
    if final_image is not None:
        for c in range(n_cols):
            axes[2, c].axis("off")
        _plot_image_with_crosshair(axes[2, 0], ref_image.data, "Reference")
        # The residual is the run-level answer -- did this end up aligned? -- so it
        # goes in the title rather than the corner text, which is per-method detail.
        final_title = "Final"
        if final_residual_px is not None:
            final_title = f"Final  -  {final_residual_px:.1f}px from the reference"
        _plot_image_with_crosshair(axes[2, n_cols - 1], final_image.data, final_title)
        if validation is not None:
            colour = "lime" if validation.agreement else "orange"
            lines = []
            for method_name, shift_pt in validation.shifts_px.items():
                mag = np.hypot(shift_pt.x, shift_pt.y)
                lines.append(f"{method_name}: dx={shift_pt.x:.1f}  dy={shift_pt.y:.1f}  |{mag:.1f}|px")
            axes[2, n_cols - 1].text(
                0.04, 0.04, "\n".join(lines),
                transform=axes[2, n_cols - 1].transAxes,
                color=colour, fontsize=7, va="bottom", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6, edgecolor="none"),
            )

    fig.tight_layout()
    if save:
        save_path = os.path.join(ref_path, "figure.png")
        fig.savefig(save_path, dpi=80)
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
