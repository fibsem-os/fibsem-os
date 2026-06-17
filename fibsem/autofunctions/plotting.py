"""Diagnostic plotting for autofunctions results."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fibsem.autofunctions.acb import AutoContrastBrightnessResult
    from fibsem.autofunctions.autofocus import AutoFocusResult

logger = logging.getLogger(__name__)

MAX_THUMBS = 5
PASS_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]


# ── ACB ──────────────────────────────────────────────────────────────────────

def _best_thumb_indices(result: "AutoContrastBrightnessResult") -> list[int]:
    """Return up to MAX_THUMBS indices ranked by closeness of mean to 0.5."""
    ranked = sorted(range(len(result.iterations)),
                    key=lambda i: abs(result.iterations[i].stats.mean - 0.5))
    return sorted(ranked[:MAX_THUMBS])


def plot_acb_result(
    result: "AutoContrastBrightnessResult",
    save_path: str,
) -> None:
    """Diagnostic plot for a hardware ACB run.

    Top row — probe image thumbnails for the best iterations (closest mean to 0.5).
    Bottom row — per-iteration traces for brightness, contrast, mean, saturation,
                 and range utilisation.
    """
    import matplotlib.gridspec as gridspec
    from matplotlib.figure import Figure

    iters = result.iterations
    n = len(iters)
    xs = list(range(n))

    thumb_indices = _best_thumb_indices(result)
    n_thumbs = len(thumb_indices)

    N_METRICS = 5
    fig = Figure(figsize=(max(10, N_METRICS * 2.2), 7), layout="constrained")
    gs = gridspec.GridSpec(2, max(n_thumbs, N_METRICS), figure=fig,
                           height_ratios=[1.4, 1])
    fig.get_layout_engine().set(hspace=0.08, wspace=0.1, h_pad=0.2, w_pad=0.05)

    final_idx = n - 1
    for col, idx in enumerate(thumb_indices):
        ax = fig.add_subplot(gs[0, col])
        img = iters[idx].image.data
        ax.imshow(img[::4, ::4], cmap="gray")
        is_final = idx == final_idx
        label = f"iter {idx}" + (" ★" if is_final else "")
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

    combined = [
        ("brightness", [it.brightness        for it in iters], "tab:blue"),
        ("contrast",   [it.contrast          for it in iters], "tab:orange"),
        ("mean",       [it.stats.mean        for it in iters], "tab:green"),
        ("median",     [it.stats.median      for it in iters], "tab:olive"),
    ]
    solo_metrics = [
        ("saturation hi",     [it.stats.saturation_hi     for it in iters], "tab:red",    0.005),
        ("range utilisation", [it.stats.range_utilisation for it in iters], "tab:purple", None),
        ("SNR",               [it.stats.snr               for it in iters], "tab:cyan",   None),
        ("entropy (bits)",    [it.stats.entropy           for it in iters], "tab:brown",  None),
    ]

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
    fig.clf()


# ── Autofocus ─────────────────────────────────────────────────────────────────

def plot_autofocus_result(
    result: "AutoFocusResult",
    save_path: str,
) -> None:
    """Diagnostic plot: one row per pass, curve on left, thumbnails on right."""
    import numpy as np
    import matplotlib.gridspec as gridspec
    from matplotlib.figure import Figure

    iters = result.iterations
    n = len(iters)
    method = result.settings.method if result.settings else "?"

    best_idx = int(max(range(n), key=lambda i: iters[i].focus_score))
    best_wd = iters[best_idx].working_distance

    initial_wd = getattr(result, "initial_working_distance", None)
    initial_z_um = (initial_wd - best_wd) * 1e6 if initial_wd is not None else None

    z_um = [(it.working_distance - best_wd) * 1e6 for it in iters]
    raw_scores = [it.focus_score for it in iters]
    score_max = max(raw_scores) if raw_scores else 1.0
    norm_scores = [s / score_max for s in raw_scores]

    pass_indices = sorted(set(it.pass_index for it in iters))
    n_passes = len(pass_indices)

    per_pass_thumbs = []
    for pi in pass_indices:
        global_idx = [i for i, it in enumerate(iters) if it.pass_index == pi]
        if len(global_idx) <= MAX_THUMBS:
            sampled = list(global_idx)
        else:
            step = len(global_idx) / MAX_THUMBS
            sampled = [global_idx[min(round(i * step), len(global_idx) - 1)] for i in range(MAX_THUMBS)]
            if best_idx in global_idx and best_idx not in sampled:
                sampled[-1] = best_idx
            sampled = sorted(set(sampled))
        per_pass_thumbs.append(sampled)

    n_thumb_cols = max(len(row) for row in per_pass_thumbs)

    thumb_w = 1.4
    fig_w = 3.5 + n_thumb_cols * thumb_w
    fig_h = 2.0 * n_passes
    fig = Figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(
        n_passes, 1 + n_thumb_cols,
        figure=fig,
        width_ratios=[3.0] + [1.0] * n_thumb_cols,
        wspace=0.08, hspace=0.25,
    )

    for row, pi in enumerate(pass_indices):
        color = PASS_COLORS[pi % len(PASS_COLORS)]
        idx = [i for i, it in enumerate(iters) if it.pass_index == pi]

        ax = fig.add_subplot(gs[row, 0])
        ax.plot(
            [z_um[i] for i in idx], [norm_scores[i] for i in idx],
            "o-", color=color, markersize=3, linewidth=1.2,
        )
        ax.axvline(0, color="limegreen", linestyle="--", linewidth=0.8)
        if initial_z_um is not None:
            ax.axvline(initial_z_um, color="gray", linestyle=":", linewidth=0.8)
        ax.set_ylabel(f"pass {pi}", fontsize=7, color=color)
        ax.set_ylim(0, 1.1)
        ax.tick_params(labelsize=6)
        ax.yaxis.set_tick_params(labelleft=False)
        if row < n_passes - 1:
            for lbl in ax.get_xticklabels():
                lbl.set_visible(False)
        else:
            ax.set_xlabel("Z offset (µm)", fontsize=8)
        if row == 0:
            ax.set_title(f"Autofocus ({method})", fontsize=9)

        thumb_indices = per_pass_thumbs[row]
        for col, tidx in enumerate(thumb_indices):
            ax_img = fig.add_subplot(gs[row, 1 + col])
            img = iters[tidx].image.data
            ds = max(1, img.shape[0] // 96)
            ax_img.imshow(img[::ds, ::ds], cmap="gray", interpolation="nearest")
            z_off = z_um[tidx]
            wd_mm = iters[tidx].working_distance * 1e3
            ax_img.set_title(f"{z_off:+.0f}µm\n{wd_mm:.3f}mm", fontsize=5, pad=1)
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            is_best = tidx == best_idx
            for spine in ax_img.spines.values():
                spine.set_edgecolor("limegreen" if is_best else color)
                spine.set_linewidth(2.0 if is_best else 0.5)

        for col in range(len(thumb_indices), n_thumb_cols):
            fig.add_subplot(gs[row, 1 + col]).set_visible(False)

    _save_figure(fig, save_path)
    # fig.clf()
    return fig


# ── shared ────────────────────────────────────────────────────────────────────

def _save_figure(fig, save_path: str) -> None:
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    logger.info("Plot saved to %s", save_path)
