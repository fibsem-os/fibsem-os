from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from fibsem import utils
from fibsem.constants import DATETIME_DISPLAY
from fibsem.structures import ImageSettings

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from fibsem.alignment import AlignmentDifferential, AlignmentIteration
    from fibsem.alignment.coincidence import (
        CoincidenceAlignment,
        CoincidenceMeasurement,
    )
    from fibsem.structures import FibsemImage


def _plot_image_with_crosshair(
    ax: Axes, data: np.ndarray, title: str, crosshair: str
) -> None:
    """Plot an image with a crosshair at the centre, in the theme's ``crosshair``."""
    ax.imshow(data, cmap="gray")
    cy, cx = data.shape[0] // 2, data.shape[1] // 2
    ax.axhline(cy, color=crosshair, linewidth=1, alpha=0.7)
    ax.axvline(cx, color=crosshair, linewidth=1, alpha=0.7)
    ax.set_title(title)
    ax.axis("off")


def _alignment_save_path(ref_image: FibsemImage) -> tuple:
    """Return (ref_path, prefix, ts) for saving alignment plots."""

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
    from fibsem.theme.plots import DARK, plot_theme

    with plot_theme(DARK) as theme:
        return _plot_multi_step_alignment(
            ref_image,
            alignment_results,
            title,
            save,
            final_image,
            path,
            validation,
            theme,
        )


def _plot_multi_step_alignment(
    ref_image, alignment_results, title, save, final_image, path, validation, theme
):
    from datetime import datetime

    from matplotlib.figure import Figure

    # Only derived when no path was given: deriving it creates <reference's
    # folder>/Alignment, which left an empty folder beside the reference image
    # whenever the caller said where the run goes.
    if path is not None:
        ref_path = path
    else:
        ref_path, _, _ = _alignment_save_path(ref_image)
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
    _plot_image_with_crosshair(axes[0, 0], ref_image.data, "Reference", theme.crosshair)
    for i, r in enumerate(alignment_results):
        _plot_image_with_crosshair(
            axes[0, 1 + i], r.image.data, f"Step {i + 1}", theme.crosshair
        )
        pixel_size = r.image.metadata.pixel_size.x if r.image.metadata else 1.0
        dx_px = r.shift.x / pixel_size
        dy_px = r.shift.y / pixel_size
        colour = theme.score(r.score)
        axes[0, 1 + i].text(
            0.04,
            0.04,
            f"dx={dx_px:.1f}px  dy={dy_px:.1f}px",
            transform=axes[0, 1 + i].transAxes,
            color=colour,
            fontsize=7,
            va="bottom",
            fontfamily="monospace",
            bbox=theme.label_box(),
        )
        cx, cy = r.image.data.shape[1] // 2, r.image.data.shape[0] // 2
        axes[0, 1 + i].annotate(
            "",
            xy=(cx + dx_px, cy + dy_px),
            xytext=(cx, cy),
            arrowprops=dict(arrowstyle="->", color=colour, lw=2),
        )

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
    ax_conv.set_title("Convergence")

    for i, r in enumerate(alignment_results):
        col = 1 + i
        colour = theme.score(r.score)
        if r.xcorr is not None:
            axes[1, col].imshow(r.xcorr, cmap="inferno")
        else:  # not saved with the run, so not there when it is reloaded
            axes[1, col].text(
                0.5,
                0.5,
                "map not saved",
                ha="center",
                va="center",
                color=theme.text_muted,
                fontsize=8,
                transform=axes[1, col].transAxes,
            )
        axes[1, col].set_title(
            f"XCorr {i + 1}\ndx={r.shift.x * 1e9:.1f}nm, dy={r.shift.y * 1e9:.1f}nm\nscore={r.score:.2f}",
            fontsize="small",
            color=colour,
        )
        axes[1, col].axis("off")

    # row 2: ref vs final comparison
    if final_image is not None:
        for c in range(n_cols):
            axes[2, c].axis("off")
        _plot_image_with_crosshair(
            axes[2, 0], ref_image.data, "Reference", theme.crosshair
        )
        _plot_image_with_crosshair(
            axes[2, n_cols - 1], final_image.data, "Final", theme.crosshair
        )
        if validation is not None:
            colour = theme.good if validation.agreement else theme.warn
            lines = []
            for method_name, shift_pt in validation.shifts_px.items():
                mag = np.hypot(shift_pt.x, shift_pt.y)
                lines.append(
                    f"{method_name}: dx={shift_pt.x:.1f}  dy={shift_pt.y:.1f}  |{mag:.1f}|px"
                )
            axes[2, n_cols - 1].text(
                0.04,
                0.04,
                "\n".join(lines),
                transform=axes[2, n_cols - 1].transAxes,
                color=colour,
                fontsize=7,
                va="bottom",
                fontfamily="monospace",
                bbox=theme.label_box(),
            )

    fig.tight_layout()
    if save:
        save_path = os.path.join(ref_path, "figure.png")
        fig.savefig(save_path, dpi=80)
    return fig


def plot_coincidence_alignment(
    result: "CoincidenceAlignment", title: Optional[str] = None
):
    """One figure for a whole ensure_coincident run.

    A row per measurement: the raw SEM and FIB pair (the context the
    operator recognises), then what the correlator actually saw - both
    images local-normalised, band-passed and reduced to gradient magnitude,
    the FIB stretched to the SEM projection and shifted by the measured
    residual, shown as a checkerboard against the SEM so a good lock reads
    as continuous structure across the tiles and a wrong one as broken
    edges. A short residual plot along the bottom carries the convergence
    story, and the title the verdict. Measurements without
    their image pair are skipped.

    Returns:
        matplotlib.figure.Figure
    """
    from fibsem.theme.plots import DARK, plot_theme

    with plot_theme(DARK) as theme:
        return _plot_coincidence_alignment(result, title, theme)


def _plot_coincidence_alignment(result, title, theme):
    from datetime import datetime

    from matplotlib.figure import Figure
    from matplotlib.gridspec import GridSpec
    from scipy import ndimage as ndi

    from fibsem.alignment.coincidence import (
        AGREEMENT_BANDS,
        _preprocess,
        _stretch_y,
        geometry_from_images,
    )

    rows = [
        m
        for m in result.measurements
        if m.sem_image is not None and m.fib_image is not None
    ]
    n = max(len(rows), 1)
    fig = Figure(figsize=(12, 2.8 * n + 2.4))
    grid = GridSpec(
        n + 1,
        3,
        figure=fig,
        height_ratios=[1] * n + [0.6],
        wspace=0.04,
        hspace=0.12,
        left=0.03,
        right=0.99,
        top=0.93,
        bottom=0.06,
    )

    verdict = "CONVERGED" if result.converged else f"NOT COINCIDENT ({result.reason})"
    header = (
        f"{title or 'Coincidence alignment'} — {verdict}, "
        f"{result.moves_applied} move(s)"
        f"{', coarse pass used' if result.coarse_used else ''} — "
        f"{datetime.now().strftime(DATETIME_DISPLAY)}"
    )
    fig.suptitle(
        header, fontsize=12, color=theme.good if result.converged else theme.warn
    )

    s1, s2 = AGREEMENT_BANDS[0]
    for i, m in enumerate(rows):
        geometry = geometry_from_images(m.sem_image, m.fib_image)
        px = geometry.pixel_size
        stretch = geometry.y_stretch
        sem = m.sem_image.data.astype(np.float32)
        fib = m.fib_image.data.astype(np.float32)
        pass_name = "coarse" if m.coarse else "fine"
        hfw_um = m.sem_image.metadata.image_settings.hfw * 1e6

        ax = fig.add_subplot(grid[i, 0])
        _plot_image_with_crosshair(
            ax, sem, f"{i + 1}. SEM ({pass_name}, {hfw_um:.0f} um)", theme.crosshair
        )
        ax = fig.add_subplot(grid[i, 1])
        _plot_image_with_crosshair(ax, fib, f"{i + 1}. FIB", theme.crosshair)

        # the correlator's view, FIB shifted by the measured residual
        ref = _preprocess(sem, s1, s2)
        other = _preprocess(_stretch_y(fib, stretch), s1, s2)
        other = ndi.shift(
            other, (m.dy / px * stretch, m.dx / px), order=1, mode="constant"
        )
        tile = max(16, min(ref.shape) // 8)
        yy, xx = np.mgrid[0 : ref.shape[0], 0 : ref.shape[1]]
        checker = ((yy // tile) + (xx // tile)) % 2 == 0
        board = np.where(checker, ref, other)
        ax = fig.add_subplot(grid[i, 2])
        lo, hi = np.percentile(board, [1, 99])
        ax.imshow(board, cmap="gray", vmin=lo, vmax=hi)
        ax.set_title("Correlator: SEM / aligned FIB checkerboard", fontsize=8)
        ax.axis("off")
        colour = theme.good if m.is_reliable else theme.bad
        ax.text(
            0.03,
            0.04,
            ("RELIABLE" if m.is_reliable else f"REFUSED ({m.refusal_reason})")
            + f"\ndx={m.dx * 1e6:+.2f}  dy={m.dy * 1e6:+.2f}  dz={m.dz * 1e6:+.2f} um"
            + f"\nband-disagreement={m.band_disagreement * 1e6:.2f} um  rival={m.rival_ratio:.2f}",
            transform=ax.transAxes,
            color=colour,
            fontsize=6.5,
            va="bottom",
            fontfamily="monospace",
            bbox=theme.label_box(),
        )

    # a short strip along the bottom, not a column-high panel
    ax = fig.add_subplot(grid[n, :])
    idx = np.arange(1, len(result.measurements) + 1)
    dz = np.array([m.dz for m in result.measurements]) * 1e6
    dx = np.array([m.dx for m in result.measurements]) * 1e6
    reliable = np.array([m.is_reliable for m in result.measurements])
    ax.plot(idx, dz, "o-", color=theme.series[0], label="dz (height)")
    ax.plot(idx, dx, "s--", color=theme.text_muted, label="dx (lateral)")
    if (~reliable).any():
        ax.plot(
            idx[~reliable],
            dz[~reliable],
            "x",
            color=theme.bad,
            ms=12,
            mew=2,
            label="refused",
        )
    ax.axhline(0, color=theme.text_muted, lw=0.8)
    ax.set_xticks(idx)
    ax.set_xlabel("measurement")
    ax.set_ylabel("um")
    ax.set_title("Residuals", fontsize=10)
    ax.legend(fontsize="x-small", loc="upper right", ncol=3)
    ax.grid(alpha=0.3)
    return fig


def save_coincidence_diagnostics(
    result: "CoincidenceAlignment", path: str, prefix: str = ""
) -> str:
    """Save an ensure_coincident run as a replayable case.

    Writes a run directory `<path>/<prefix>coincidence_<ts>/` holding every
    measured pair as `NN_<pass>_sem.tif` / `NN_<pass>_fib.tif` with full
    metadata (so they can be re-measured offline with
    the same inputs), `run.json` with the verdict and every measurement's
    numbers, and `summary.png`, the figure from plot_coincidence_alignment.
    Measurements without their image pair (the pure array path) are skipped.

    Returns:
        the run directory.
    """
    import json
    from datetime import datetime

    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_dir = os.path.join(path, f"{prefix}coincidence_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    records = []
    for i, m in enumerate(result.measurements, start=1):
        pass_name = "coarse" if m.coarse else "fine"
        record = {
            "index": i,
            "pass": pass_name,
            "dx": m.dx,
            "dy": m.dy,
            "dz": m.dz,
            "band_disagreement": m.band_disagreement,
            "is_reliable": m.is_reliable,
            "refusal_reason": m.refusal_reason,
            "seeded": m.seeded,
            "y_stretch": m.y_stretch,
            "method": m.method,
            "capture_range": m.capture_range,
            "agreement_tolerance": m.agreement_tolerance,
            "max_lateral_offset": m.max_lateral_offset,
            "max_rival_ratio": m.max_rival_ratio,
            "rival_ratio": m.rival_ratio,
            "prior": m.prior,
        }
        if m.sem_image is not None and m.fib_image is not None:
            record["sem"] = f"{i:02d}_{pass_name}_sem.tif"
            record["fib"] = f"{i:02d}_{pass_name}_fib.tif"
            m.sem_image.save(os.path.join(run_dir, record["sem"]))
            m.fib_image.save(os.path.join(run_dir, record["fib"]))
        records.append(record)

    with open(os.path.join(run_dir, "run.json"), "w") as f:
        json.dump(
            {
                "converged": result.converged,
                "reason": result.reason,
                "moves_applied": result.moves_applied,
                "coarse_used": result.coarse_used,
                "measurements": records,
            },
            f,
            indent=2,
        )

    fig = plot_coincidence_alignment(result, title=f"{prefix}coincidence")
    fig.savefig(os.path.join(run_dir, "summary.png"), dpi=80)
    return run_dir
