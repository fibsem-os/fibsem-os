"""Diagnostic plotting for AutoContrastBrightnessResult."""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from fibsem.autofunctions.acb import AutoContrastBrightnessResult

logger = logging.getLogger(__name__)

MAX_THUMBS = 5


def _best_thumb_indices(result: "AutoContrastBrightnessResult") -> list[int]:
    """Return up to MAX_THUMBS indices ranked by closeness of mean to 0.5."""
    ranked = sorted(range(len(result.iterations)),
                    key=lambda i: abs(result.iterations[i].stats.mean - 0.5))
    return sorted(ranked[:MAX_THUMBS])


def plot_acb_result(
    result: "AutoContrastBrightnessResult",
    save_path: Optional[str] = None,
) -> None:
    """Diagnostic plot for a hardware ACB run.

    Top row — probe image thumbnails for the best iterations (closest mean to 0.5).
    Bottom row — per-iteration traces for brightness, contrast, mean, saturation,
                 and range utilisation.

    Args:
        result: ``AutoContrastBrightnessResult`` returned by
            ``hardware_auto_contrast_brightness``.
        save_path: Path to save the figure. If ``None``, saves to an ``acb/``
            directory in the current working directory.
    """
    import matplotlib.gridspec as gridspec
    from matplotlib.figure import Figure

    iters = result.iterations
    n = len(iters)
    xs = list(range(n))

    thumb_indices = _best_thumb_indices(result)
    n_thumbs = len(thumb_indices)

    N_METRICS = 5  # 1 combined (brightness/contrast/mean/median) + 4 solo
    fig = Figure(figsize=(max(10, N_METRICS * 2.2), 7), layout="constrained")
    gs = gridspec.GridSpec(2, max(n_thumbs, N_METRICS), figure=fig,
                           height_ratios=[1.4, 1])
    fig.get_layout_engine().set(hspace=0.08, wspace=0.1, h_pad=0.2, w_pad=0.05)

    # ── top row: probe thumbnails ────────────────────────────────────────────
    final_idx = n - 1
    for col, idx in enumerate(thumb_indices):
        ax = fig.add_subplot(gs[0, col])
        img = iters[idx].image.data
        ax.imshow(img[::4, ::4], cmap="gray")
        is_final = idx == final_idx
        label = f"iter {idx}"
        if is_final:
            label += " ★"
        ax.set_title(
            f"{label}\nb={iters[idx].brightness:.2f} c={iters[idx].contrast:.2f}\n"
            f"mean={iters[idx].stats.mean:.3f}",
            fontsize=6, pad=2,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        border_color = "limegreen" if is_final else "#444444"
        border_width = 2.5 if is_final else 0.5
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(border_width)

    # ── bottom row: metric traces ────────────────────────────────────────────
    # combined 0-1 panel: brightness, contrast, mean, median
    combined = [
        ("brightness", [it.brightness        for it in iters], "tab:blue"),
        ("contrast",   [it.contrast          for it in iters], "tab:orange"),
        ("mean",       [it.stats.mean        for it in iters], "tab:green"),
        ("median",     [it.stats.median      for it in iters], "tab:olive"),
    ]
    # remaining individual panels
    solo_metrics = [
        ("saturation hi",     [it.stats.saturation_hi         for it in iters], "tab:red",    0.005),
        ("range utilisation", [it.stats.range_utilisation     for it in iters], "tab:purple", None),
        ("SNR",               [it.stats.snr                   for it in iters], "tab:cyan",   None),
        ("entropy (bits)",    [it.stats.entropy               for it in iters], "tab:brown",  None),
    ]
    N_METRICS = 1 + len(solo_metrics)  # 1 combined + 4 solo

    ax_combined = fig.add_subplot(gs[1, 0])
    for label, values, color in combined:
        ax_combined.plot(xs, values, "o-", color=color, markersize=4, linewidth=1.2, label=label)
    ax_combined.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_combined.set_ylim(0, 1)
    ax_combined.set_title("brightness / contrast / mean / median", fontsize=7, pad=2)
    ax_combined.set_xlabel("iteration", fontsize=6)
    ax_combined.tick_params(labelsize=6)
    ax_combined.legend(fontsize=5, loc="lower right")

    for col, (label, values, color, hline) in enumerate(solo_metrics, start=1):
        ax = fig.add_subplot(gs[1, col])
        ax.plot(xs, values, "o-", color=color, markersize=4, linewidth=1.2)
        ax.set_title(label, fontsize=7, pad=2)
        ax.set_xlabel("iteration", fontsize=6)
        ax.tick_params(labelsize=6)
        if hline is not None:
            ax.axhline(hline, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    status = "converged" if result.converged else "not converged"
    fig.suptitle(f"Auto Contrast/Brightness — {n} iterations, {status}", fontsize=9)

    _save_figure(fig, save_path)


def _save_figure(fig, save_path: Optional[str]) -> None:
    if save_path is None:
        os.makedirs("acb", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join("acb", f"acb_{ts}.png")
    fig.savefig(save_path, dpi=100)
    logger.info("ACB diagnostic plot saved to %s", save_path)
