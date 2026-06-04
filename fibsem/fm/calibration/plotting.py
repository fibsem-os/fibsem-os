"""Autofocus diagnostic plotting utilities."""
from __future__ import annotations

import logging
import os
from datetime import datetime
from math import ceil
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from fibsem.fm.structures import AutoFocusResult


def _select_thumb_indices(result: "AutoFocusResult", max_thumbs: int = 10) -> list:
    """Return up to max_thumbs image indices: best + next-best by score, re-sorted by index."""
    n = len(result.images)
    ranked = sorted(range(n), key=lambda i: result.scores[i], reverse=True)
    return sorted(ranked[:max_thumbs])


def _render_thumb_row(gs, stage_row: int, n_cols: int, result: "AutoFocusResult",
                      thumb_indices: list, z_um: list, stage_label: str,
                      score_ylim: Optional[tuple] = None) -> None:
    """Render one row (score curve + thumbnails) into an existing GridSpec."""
    fig = gs.figure

    best_z_um = z_um[result.best_idx]

    ax_score = fig.add_subplot(gs[stage_row, 0])
    ax_score.plot(z_um, result.scores, "o-", color="tab:blue", markersize=3)
    ax_score.axvline(best_z_um, color="limegreen", linestyle="--", alpha=0.8)
    ax_score.set_ylabel("Score", fontsize=7)
    ax_score.set_title(stage_label, fontsize=7, pad=2)
    ax_score.tick_params(labelsize=6)
    if score_ylim is not None:
        ax_score.set_ylim(score_ylim)

    for i, idx in enumerate(thumb_indices):
        if i >= n_cols:
            break
        ax = fig.add_subplot(gs[stage_row, 1 + i])
        img = result.images[idx]
        if img.ndim > 2:
            img = img.reshape(-1, img.shape[-2], img.shape[-1])[0]
        ax.imshow(img[::4, ::4], cmap="gray")
        ax.set_title(f"{idx}: {z_um[idx]:.1f} μm ({result.scores[idx]:.2f})", fontsize=6, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        is_best = idx == result.best_idx
        border_color = "limegreen" if is_best else "#444444"
        border_width = 3 if is_best else 0.5
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(border_width)


def _save_figure(fig, save_path: Optional[str]) -> None:
    if save_path is None:
        os.makedirs("autofocus", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join("autofocus", f"autofocus_{ts}.png")
    fig.savefig(save_path, dpi=100)
    logging.info(f"Autofocus diagnostic plot saved to {save_path}")


def plot_autofocus(
    result: "AutoFocusResult",
    save_path: Optional[str] = None,
) -> None:
    """Generate a diagnostic plot for autofocus results.

    If result.iterations is populated (CoarseFine or Iterative strategies), renders
    one row per stage. Otherwise renders a single-result layout.

    Args:
        result: AutoFocusResult returned by run_autofocus or a strategy.
        save_path: Optional path to save the figure. If None, saves to autofocus/ directory.
    """
    import matplotlib.gridspec as gridspec
    from matplotlib.figure import Figure

    MAX_THUMBS = 10

    # --- Multi-stage (iterative / coarse-fine) ---
    if result.iterations:
        stages = result.iterations
        n_stages = len(stages)

        all_thumb_indices = [_select_thumb_indices(s, MAX_THUMBS) for s in stages]
        n_cols = max(len(t) for t in all_thumb_indices)

        all_scores = [s for stage in stages for s in stage.scores]
        score_ylim = (min(0, min(all_scores)), max(all_scores) * 1.05)

        fig = Figure(figsize=(2 + n_cols * 1.5, n_stages * 1.8), layout="constrained")
        gs = gridspec.GridSpec(n_stages, 1 + n_cols, figure=fig,
                               width_ratios=[2] + [1] * n_cols)
        fig.get_layout_engine().set(hspace=0.0, wspace=0.02, h_pad=0.1, w_pad=0.02)

        for row, (stage, thumb_indices) in enumerate(zip(stages, all_thumb_indices)):
            z_um = [z * 1e6 for z in stage.z_positions]
            is_final = row == n_stages - 1
            label = f"{'★ ' if is_final else ''}Stage {row + 1}"
            _render_thumb_row(gs, row, n_cols, stage, thumb_indices, z_um, label, score_ylim)

        _save_figure(fig, save_path)
        return

    # --- Single-stage ---
    z_um = [z * 1e6 for z in result.z_positions]
    thumb_indices = _select_thumb_indices(result, MAX_THUMBS)

    n_thumbs = len(thumb_indices)
    n_rows = 2 if n_thumbs > 5 else 1
    n_cols = ceil(n_thumbs / n_rows)

    thumb_height = 1.8 if n_rows > 1 else 2.5
    fig = Figure(figsize=(2 + n_cols * 1.5, n_rows * thumb_height), layout="constrained")
    gs = gridspec.GridSpec(n_rows, 1 + n_cols, figure=fig,
                           width_ratios=[2] + [1] * n_cols)
    fig.get_layout_engine().set(hspace=0.0, wspace=0.02, h_pad=0.1, w_pad=0.02)

    best_z_um = z_um[result.best_idx]
    ax_score = fig.add_subplot(gs[:, 0])
    ax_score.plot(z_um, result.scores, "o-", color="tab:blue", markersize=3)
    ax_score.axvline(best_z_um, color="limegreen", linestyle="--", alpha=0.8,
                     label=f"Best: {best_z_um:.1f} μm")
    ax_score.set_xlabel("Z position (μm)")
    ax_score.set_ylabel("Focus score")
    title = f"Autofocus ({result.method})"
    if result.roi is not None:
        title += f"\nROI: l={result.roi.left:.2f} t={result.roi.top:.2f} w={result.roi.width:.2f} h={result.roi.height:.2f}"
    ax_score.set_title(title, fontsize="small")
    ax_score.legend(fontsize="small")

    for i, idx in enumerate(thumb_indices):
        row = i // n_cols
        col = i % n_cols
        ax = fig.add_subplot(gs[row, 1 + col])
        img = result.images[idx]
        if img.ndim > 2:
            img = img.reshape(-1, img.shape[-2], img.shape[-1])[0]
        ax.imshow(img[::4, ::4], cmap="gray")
        is_best = idx == result.best_idx
        ax.set_title(f"{idx}: {z_um[idx]:.1f} μm ({result.scores[idx]:.2f})", fontsize=6, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        border_color = "limegreen" if is_best else "#444444"
        border_width = 3 if is_best else 0.5
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(border_width)

    _save_figure(fig, save_path)
