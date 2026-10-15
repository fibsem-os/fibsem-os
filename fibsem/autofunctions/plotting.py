"""Diagnostic plotting for autofunctions results."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from fibsem.autofunctions.acb import AutoContrastBrightnessResult
    from fibsem.autofunctions.autofocus import AutoFocusResult
    from fibsem.autofunctions.brightness_contrast import BrightnessContrastResult

logger = logging.getLogger(__name__)

MAX_THUMBS = 5


# ── ACB ──────────────────────────────────────────────────────────────────────


def _best_thumb_indices(result: "AutoContrastBrightnessResult") -> list[int]:
    """Return up to MAX_THUMBS indices ranked by closeness of mean to 0.5."""
    ranked = sorted(
        range(len(result.iterations)),
        key=lambda i: abs(result.iterations[i].stats.mean - 0.5),
    )
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
    from fibsem.theme.plots import DARK, plot_theme

    with plot_theme(DARK) as theme:
        return _plot_acb_result(result, save_path, theme)


def _plot_acb_result(result, save_path, theme):
    import matplotlib.gridspec as gridspec
    from matplotlib.figure import Figure

    iters = result.iterations
    n = len(iters)
    xs = list(range(n))

    thumb_indices = _best_thumb_indices(result)
    n_thumbs = len(thumb_indices)

    N_METRICS = 5
    fig = Figure(figsize=(max(10, N_METRICS * 2.2), 7), layout="constrained")
    gs = gridspec.GridSpec(
        2, max(n_thumbs, N_METRICS), figure=fig, height_ratios=[1.4, 1]
    )
    fig.get_layout_engine().set(hspace=0.08, wspace=0.1, h_pad=0.2, w_pad=0.05)

    final_idx = n - 1
    for col, idx in enumerate(thumb_indices):
        ax = fig.add_subplot(gs[0, col])
        img = iters[idx].image.filtered_data
        ax.imshow(img[::4, ::4], cmap="gray")
        is_final = idx == final_idx
        label = f"iter {idx}" + (" ★" if is_final else "")
        ax.set_title(
            f"{label}\nb={iters[idx].brightness:.2f} c={iters[idx].contrast:.2f}\n"
            f"mean={iters[idx].stats.mean:.3f}",
            fontsize=6,
            pad=2,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        border_color = theme.good if is_final else theme.grid
        border_width = 2.5 if is_final else 0.5
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(border_width)

    # the four on one axis take the theme's first four colours; a metric on an
    # axis of its own needs no colour to tell it apart, so each takes the first
    combined = [
        ("brightness", [it.brightness for it in iters], theme.series[0]),
        ("contrast", [it.contrast for it in iters], theme.series[1]),
        ("mean", [it.stats.mean for it in iters], theme.series[2]),
        ("median", [it.stats.median for it in iters], theme.series[3]),
    ]
    solo = theme.series[0]
    solo_metrics = [
        ("saturation hi", [it.stats.saturation_hi for it in iters], solo, 0.005),
        (
            "range utilisation",
            [it.stats.range_utilisation for it in iters],
            solo,
            None,
        ),
        ("SNR", [it.stats.snr for it in iters], solo, None),
        ("entropy (bits)", [it.stats.entropy for it in iters], solo, None),
    ]

    ax_combined = fig.add_subplot(gs[1, 0])
    for label, values, color in combined:
        ax_combined.plot(
            xs, values, "o-", color=color, markersize=4, linewidth=1.2, label=label
        )
    ax_combined.axhline(
        0.5, color=theme.text_muted, linestyle="--", linewidth=0.8, alpha=0.6
    )
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
            ax.axhline(
                hline, color=theme.text_muted, linestyle="--", linewidth=0.8, alpha=0.6
            )

    status = "converged" if result.converged else "not converged"
    fig.suptitle(f"Auto Contrast/Brightness — {n} iterations, {status}", fontsize=9)
    _save_figure(fig, save_path)
    return fig


# ── Autofocus ─────────────────────────────────────────────────────────────────


def plot_autofocus_result(
    result: "AutoFocusResult",
    save_path: Optional[str] = None,
) -> None:
    """Diagnostic plot: one row per pass, curve on left, thumbnails on right.

    If ``save_path`` is None, the figure is returned and nothing is written.
    """
    from fibsem.theme.plots import DARK, plot_theme

    with plot_theme(DARK) as theme:
        return _plot_autofocus_result(result, save_path, theme)


def _plot_autofocus_result(result, save_path, theme):
    import matplotlib.gridspec as gridspec
    import numpy as np
    from matplotlib.figure import Figure

    iters = result.iterations
    n = len(iters)
    # method may live on settings (FocusMethod enum) or directly on the result
    # (str); fall back gracefully and always render the string form.
    _method = result.settings.method if result.settings else result.method
    method = getattr(_method, "value", _method) or "?"

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
            sampled = [
                global_idx[min(round(i * step), len(global_idx) - 1)]
                for i in range(MAX_THUMBS)
            ]
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
        n_passes,
        1 + n_thumb_cols,
        figure=fig,
        width_ratios=[3.0] + [1.0] * n_thumb_cols,
        wspace=0.08,
        hspace=0.25,
    )

    for row, pi in enumerate(pass_indices):
        color = theme.series[pi % len(theme.series)]
        idx = [i for i, it in enumerate(iters) if it.pass_index == pi]

        ax = fig.add_subplot(gs[row, 0])
        ax.plot(
            [z_um[i] for i in idx],
            [norm_scores[i] for i in idx],
            "o-",
            color=color,
            markersize=3,
            linewidth=1.2,
        )
        ax.axvline(0, color=theme.good, linestyle="--", linewidth=0.8)
        if initial_z_um is not None:
            ax.axvline(
                initial_z_um, color=theme.text_muted, linestyle=":", linewidth=0.8
            )
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
            img = iters[tidx].image
            ds = max(1, img.shape[0] // 96)
            ax_img.imshow(img[::ds, ::ds], cmap="gray", interpolation="nearest")
            z_off = z_um[tidx]
            wd_mm = iters[tidx].working_distance * 1e3
            ax_img.set_title(f"{z_off:+.0f}µm\n{wd_mm:.3f}mm", fontsize=5, pad=1)
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            is_best = tidx == best_idx
            for spine in ax_img.spines.values():
                spine.set_edgecolor(theme.good if is_best else color)
                spine.set_linewidth(2.0 if is_best else 0.5)

        for col in range(len(thumb_indices), n_thumb_cols):
            fig.add_subplot(gs[row, 1 + col]).set_visible(False)

    _save_figure(fig, save_path)
    return fig


# ── Brightness and contrast ───────────────────────────────────────────────────


def plot_brightness_contrast_result(
    result: "BrightnessContrastResult",
    save_path: Optional[str] = None,
):
    """Diagnostic plot for `Imaging.auto_brightness_contrast`.

    Top row: every probe image in order, with the pixels that clipped painted
    over it (areas that counted solid, thin features that were let clip faint),
    framed by how the routine judged it: accepted, adjusting, or unusable.
    Bottom row: the controls at each image, the clipped fractions against their
    limits, each image's histogram, and the centre and spread against their
    targets.

    If ``save_path`` is None, the figure is returned and nothing is written.
    """
    from fibsem.theme.plots import DARK, plot_theme

    with plot_theme(DARK) as theme:
        return _plot_brightness_contrast_result(result, save_path, theme)


def _clip_overlay(image, settings, theme):
    """``image`` as RGB, with the clipped pixels painted: counted areas solid, thin
    features that were let clip at a third."""
    import numpy as np
    from matplotlib.colors import to_rgb

    from fibsem.autofunctions.brightness_contrast import clipped_masks

    info = np.iinfo(image.dtype) if np.issubdtype(image.dtype, np.integer) else None
    grey = image / (info.max if info else 1.0)
    rgb = np.repeat(grey[..., None], 3, axis=2).astype(float)
    masks = clipped_masks(image, settings)
    for end, colour in (("white", theme.bad), ("black", theme.series[0])):
        colour = np.array(to_rgb(colour))
        rgb[masks[end]] = 0.67 * rgb[masks[end]] + 0.33 * colour
        rgb[masks[f"{end}_counted"]] = colour
    return rgb


def _plot_brightness_contrast_result(result, save_path, theme):
    import matplotlib.gridspec as gridspec
    import numpy as np
    from matplotlib.figure import Figure

    from fibsem.autofunctions.brightness_contrast import (
        BrightnessContrastSettings,
        problems,
        unusable,
    )

    settings = result.settings or BrightnessContrastSettings()
    iters = result.iterations
    n = len(iters)
    xs = np.arange(1, n + 1)
    n_cols = max(n, 4)

    # the top row as tall as the probe images are at this width, and their titles
    shape = next((it.image.shape for it in iters if it.image is not None), (2, 3))
    image_height = 2.3 * shape[0] / shape[1] + 0.6
    fig = Figure(figsize=(2.3 * n_cols, image_height + 2.8), layout="constrained")
    gs = gridspec.GridSpec(2, n_cols, figure=fig, height_ratios=[image_height, 2.8])

    def judged(stats):
        if not problems(stats, settings):
            return "accepted", theme.good
        if unusable(stats, settings):
            return "unusable", theme.bad
        return "adjusting", theme.warn

    ended = max(
        (
            i
            for i, it in enumerate(iters)
            if (it.brightness, it.contrast) == (result.brightness, result.contrast)
        ),
        default=n - 1,
    )
    for i, it in enumerate(iters):
        ax = fig.add_subplot(gs[0, i])
        ax.set_xticks([])
        ax.set_yticks([])
        verdict, colour = judged(it.stats)
        if it.image is not None:
            ax.imshow(_clip_overlay(it.image, settings, theme), interpolation="nearest")
        else:
            ax.text(
                0.5,
                0.5,
                "no image",
                ha="center",
                va="center",
                color=theme.text_muted,
                transform=ax.transAxes,
                fontsize=7,
            )
        mark = " (ended)" if i == ended else ""
        ax.set_title(
            f"{i + 1}: {verdict}{mark}\n"
            f"B {it.brightness:.3f}  C {it.contrast:.3f}\n"
            f"clipped {it.stats.black:.1%} black, {it.stats.white:.1%} white",
            fontsize=6.5,
            pad=2,
            color=colour if i == ended else theme.text_strong,
        )
        for spine in ax.spines.values():
            spine.set_edgecolor(colour)
            spine.set_linewidth(3 if i == ended else 1.5)
    for i in range(n, n_cols):
        fig.add_subplot(gs[0, i]).set_visible(False)

    bottom = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[1, :])

    # the controls
    ax = fig.add_subplot(bottom[0])
    ax.plot(
        xs,
        [it.brightness for it in iters],
        "o-",
        label="brightness",
        color=theme.series[0],
        markersize=4,
    )
    ax.plot(
        xs,
        [it.contrast for it in iters],
        "s-",
        label="contrast",
        color=theme.series[1],
        markersize=4,
    )
    for value, colour in (
        (result.initial_brightness, theme.series[0]),
        (result.initial_contrast, theme.series[1]),
    ):
        if value is not None:
            ax.axhline(value, color=colour, linestyle=":", linewidth=0.8)
    ax.set_title("controls (dotted: before the run)", fontsize=7)
    ax.set_xlabel("image", fontsize=7)
    ax.set_xticks(xs)
    ax.legend(fontsize=6)

    # clipping: what counted, and every clipped pixel
    ax = fig.add_subplot(bottom[1])
    floor = 1e-4
    for end, colour in (("black", theme.series[0]), ("white", theme.bad)):
        counted = [max(getattr(it.stats, end), floor) for it in iters]
        total = [max(getattr(it.stats, f"{end}_total"), floor) for it in iters]
        ax.plot(xs, counted, "o-", color=colour, markersize=4, label=f"{end} (counted)")
        ax.plot(
            xs,
            total,
            "o--",
            color=colour,
            markersize=3,
            alpha=0.6,
            linewidth=0.9,
            label=f"{end} (all)",
        )
    ax.axhline(
        settings.max_clipped, color=theme.text_muted, linestyle="--", linewidth=0.8
    )
    ax.axhline(
        settings.max_clipped_total, color=theme.text_muted, linestyle=":", linewidth=0.8
    )
    ax.set_yscale("log")
    ax.set_ylim(floor * 0.8, 1.2)
    ax.set_title(
        f"clipped fraction (limits {settings.max_clipped:.0%} counted, "
        f"{settings.max_clipped_total:.0%} all)",
        fontsize=7,
    )
    ax.set_xlabel("image", fontsize=7)
    ax.set_xticks(xs)
    ax.legend(fontsize=5, ncol=2)

    # histograms
    ax = fig.add_subplot(bottom[2])
    cmap = theme.series
    for i, it in enumerate(iters):
        if it.image is None:
            continue
        data = it.image
        top = np.iinfo(data.dtype).max if np.issubdtype(data.dtype, np.integer) else 1.0
        ax.hist(
            (data / top).ravel(),
            bins=64,
            range=(0, 1),
            histtype="step",
            color=cmap[i % len(cmap)],
            linewidth=1.1 if i == ended else 0.7,
            label=str(i + 1),
        )
    ax.set_yscale("log")
    ax.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax.set_xlim(0, 1)
    ax.set_title("histograms (by image)", fontsize=7)
    ax.set_xlabel("intensity", fontsize=7)
    ax.legend(fontsize=5, ncol=3)

    # centre and spread against the targets
    ax = fig.add_subplot(bottom[3])
    ax.axhspan(
        settings.min_centre, settings.max_centre, color=theme.series[2], alpha=0.12
    )
    ax.plot(
        xs,
        [it.stats.centre for it in iters],
        "o-",
        color=theme.series[2],
        markersize=4,
        label="centre (allowed band)",
    )
    ax.plot(
        xs,
        [it.stats.spread for it in iters],
        "s-",
        color=theme.series[3],
        markersize=4,
        label="spread",
    )
    ax.axhline(
        settings.min_spread, color=theme.series[3], linestyle="--", linewidth=0.8
    )
    ax.set_ylim(0, 1)
    low, high = settings.spread_percentiles
    ax.set_title(
        f"centre, spread ({low:g}th to {high:g}th pct; dashed: min)", fontsize=7
    )
    ax.set_xlabel("image", fontsize=7)
    ax.set_xticks(xs)
    ax.legend(fontsize=5)

    for a in fig.axes:
        a.tick_params(labelsize=6)

    status = (
        "converged"
        if result.converged
        else ("UNUSABLE, raised" if not result.usable else "short of the targets")
    )
    start = ", started from the last good settings" if result.started_from_last else ""
    beam = f" {result.beam_type}" if result.beam_type else ""
    fig.suptitle(
        f"Brightness and contrast{beam}: {n} image(s), {status}{start}", fontsize=9
    )
    _save_figure(fig, save_path)
    return fig


# ── shared ────────────────────────────────────────────────────────────────────


def _save_figure(fig, save_path: Optional[str] = None) -> None:
    # No path means the caller did not say where its run writes, so nothing is
    # saved: guessing one (it used to be ``autofocus/`` under the working
    # directory) scattered plots wherever the app was started from.
    if save_path is None:
        return
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    logger.debug("Plot saved to %s", save_path)
