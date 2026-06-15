"""Diagnostic plotting for AutoFocusResult."""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from fibsem.autofunctions.autofocus import AutoFocusResult

logger = logging.getLogger(__name__)

MAX_THUMBS = 10   # up to 5 per row, 2 rows
PASS_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]


def plot_autofocus_result(
    result: "AutoFocusResult",
    save_path: Optional[str] = None,
) -> None:
    """Diagnostic plot for an image-based auto-focus sweep.

    Left panel — normalised focus score vs Z offset, one line per pass.
    Right panel — up to 10 probe thumbnails from the final pass in two rows;
                  best image highlighted in green.

    Args:
        result: ``AutoFocusResult`` returned by ``run_auto_focus``.
        save_path: Path to save the figure. If ``None``, saves to an ``autofocus/``
            directory in the current working directory.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    iters = result.iterations
    n = len(iters)
    method = result.settings.method if result.settings else "?"

    best_idx = int(max(range(n), key=lambda i: iters[i].focus_score))
    best_wd = iters[best_idx].working_distance

    initial_wd = getattr(result, "initial_working_distance", None)
    initial_z_um = (initial_wd - best_wd) * 1e6 if initial_wd is not None else None

    z_um = [(it.working_distance - best_wd) * 1e6 for it in iters]
    raw_scores = [it.focus_score for it in iters]
    score_max = max(raw_scores) or 1.0
    norm_scores = [s / score_max for s in raw_scores]

    pass_indices = sorted(set(it.pass_index for it in iters))
    n_passes = len(pass_indices)

    # one row of thumbnails per pass, up to MAX_THUMBS columns each
    # build per-pass sample lists
    per_pass_thumbs = []
    for pi in pass_indices:
        global_idx = [i for i, it in enumerate(iters) if it.pass_index == pi]
        if len(global_idx) <= MAX_THUMBS:
            per_pass_thumbs.append(global_idx)
        else:
            step = len(global_idx) / MAX_THUMBS
            sampled = sorted(set(
                [global_idx[min(round(i * step), len(global_idx) - 1)] for i in range(MAX_THUMBS)]
            ))
            per_pass_thumbs.append(sampled[:MAX_THUMBS])

    n_cols_img = max(len(row) for row in per_pass_thumbs)

    fig = plt.figure(figsize=(4 + n_cols_img * 1.5, 2 + n_passes * 1.5))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, n_cols_img * 0.9], wspace=0.15)

    # ── left: one curve subplot per pass, sharing x-axis ────────────────────
    gs_curves = gridspec.GridSpecFromSubplotSpec(
        n_passes, 1, subplot_spec=gs[0], hspace=0.08,
    )
    ax_curves = []
    for pi in pass_indices:
        ax = fig.add_subplot(gs_curves[pi], sharex=ax_curves[0] if ax_curves else None)
        ax_curves.append(ax)
        idx = [i for i, it in enumerate(iters) if it.pass_index == pi]
        color = PASS_COLORS[pi % len(PASS_COLORS)]
        ax.plot(
            [z_um[i] for i in idx], [norm_scores[i] for i in idx],
            "o-", color=color, markersize=3, linewidth=1.2,
        )
        ax.axvline(0, color="limegreen", linestyle="--", linewidth=1.0)
        if initial_z_um is not None:
            ax.axvline(initial_z_um, color="gray", linestyle="--", linewidth=1.0)
        ax.set_ylabel(f"p{pi}", fontsize=7, color=color)
        ax.set_ylim(0, 1.05)
        ax.tick_params(labelsize=6)
        ax.yaxis.set_tick_params(labelleft=False)
        if pi < n_passes - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel("Z position (µm)", fontsize=8)

    ax_curves[0].set_title(f"Autofocus ({method})", fontsize=9)

    # ── right: one row of thumbnails per pass ────────────────────────────────
    gs_imgs = gridspec.GridSpecFromSubplotSpec(
        n_passes, n_cols_img, subplot_spec=gs[1], hspace=0.05, wspace=0.05
    )

    for row, (pi, thumb_indices) in enumerate(zip(pass_indices, per_pass_thumbs)):
        color = PASS_COLORS[pi % len(PASS_COLORS)]
        for col, idx in enumerate(thumb_indices):
            ax = fig.add_subplot(gs_imgs[row, col])
            img = iters[idx].image.data
            ds = max(1, img.shape[0] // 96)
            ax.imshow(img[::ds, ::ds], cmap="gray", interpolation="nearest")
            z_off = z_um[idx]
            score = norm_scores[idx]
            ax.set_title(f"{idx}: {z_off:+.0f} µm ({score:.2f})", fontsize=6, pad=1)
            ax.set_xticks([])
            ax.set_yticks([])
            is_best = idx == best_idx
            for spine in ax.spines.values():
                spine.set_edgecolor("limegreen" if is_best else color)
                spine.set_linewidth(2.0 if is_best else 0.8)
        # hide unused columns in this row
        for col in range(len(thumb_indices), n_cols_img):
            try:
                fig.add_subplot(gs_imgs[row, col]).set_visible(False)
            except Exception:
                pass

    fig.tight_layout()
    _save_figure(fig, save_path)
    plt.close(fig)


def _save_figure(fig, save_path: Optional[str]) -> None:
    if save_path is None:
        os.makedirs("autofocus", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join("autofocus", f"autofocus_{ts}.png")
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    logger.info("AutoFocus diagnostic plot saved to %s", save_path)
