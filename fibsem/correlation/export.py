"""Exporting a correlation: the CSV, the saved plot and the overlay figure.

Pure functions over a :class:`CorrelationResult` and matplotlib ``Figure``\\ s,
no Qt and no pyplot: a figure built here is never registered with pyplot's
global figure manager, so it needs no ``plt.close`` and cannot disturb an
interactive backend (it is freed with its canvas). The correlation tab and the
correlation service both export through this module.

- :func:`write_csv` writes a result's input coordinates and per-marker
  reprojection error.
- :func:`correlation_plot` is the tab's Save Plot figure: the FIB and FM canvases
  side by side, exactly as they are shown (:func:`render_canvas_to_axes`).
- :func:`overlay_figure` is the overlay the service saves beside each run: the
  FIB image with the picked fiducials, where the fit puts the FM fiducials, and
  the points of interest.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any, Sequence

from fibsem.theme.tokens import CANVAS_BG

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from matplotlib.figure import Figure

    from fibsem.correlation.structures import CorrelationResult

__all__ = [
    "correlation_plot",
    "overlay_figure",
    "render_canvas_to_axes",
    "save_correlation_plot",
    "save_overlay",
    "write_csv",
]

PLOT_DPI = 150  # the saved plot's dpi
_PLOT_PANEL_INCHES = 8.0  # each panel of the 16x8 two-panel figure


# ── CSV ───────────────────────────────────────────────────────────────────


def write_csv(result: "CorrelationResult", filename: str) -> None:
    """Export input coordinates and reprojection error to a CSV file."""
    df_input = result.to_input_dataframe()
    df_result = result.to_result_dataframe()
    with open(filename, "w") as f:
        f.write("# Input Coordinates\n")
        df_input.to_csv(f, index=False)
        f.write("\n# Marker Reprojection Error\n")
        df_result.to_csv(f, index=False)


# ── the saved plot ────────────────────────────────────────────────────────


def render_canvas_to_axes(canvas, ax, *, dpi: int = PLOT_DPI) -> None:
    """Draw whatever *canvas* is currently showing into *ax*.

    Deliberately canvas-agnostic: it needs a matplotlib figure and nothing else,
    so it works for ``ImagePointCanvas`` and the shared ``FibsemImageCanvas``
    alike and does not have to be rewritten when the correlation tab swaps from
    one to the other (FIB-535). It replaces ``ImagePointCanvas.render_to_axes``,
    which re-created every artist by hand and had no equivalent on the shared
    stack.

    ``savefig`` re-renders the figure offscreen at the requested dpi, so
    resolution comes from *dpi* rather than from the on-screen widget size — the
    reason this is not a plain ``buffer_rgba()`` grab, which would cap the export
    at however large the window happened to be.

    One deliberate behaviour change: what lands on the page is now what is on the
    screen, aspect ratio included. ``render_to_axes`` re-fitted the content to the
    panel, which stretched a square image into a 2:1 rectangle — measuring off
    that plot gave the wrong answer.
    """
    from matplotlib.image import imread

    fig = canvas.figure
    # Render large enough that scaling into the panel never has to upsample.
    width_inches = max(fig.get_size_inches()[0], 0.1)
    render_dpi = max(dpi, int(dpi * _PLOT_PANEL_INCHES / width_inches))

    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=render_dpi,
        facecolor=fig.get_facecolor(),
        pad_inches=0,
    )
    buf.seek(0)
    ax.imshow(imread(buf))
    ax.axis("off")


def correlation_plot(fib_canvas, fm_canvas, *, dpi: int = PLOT_DPI) -> "Figure":
    """The FIB and FM canvases side by side, titled, on the canvas background."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(16, 8), facecolor=CANVAS_BG)
    FigureCanvasAgg(fig)  # gives tight_layout a renderer
    ax_fib, ax_fm = fig.subplots(1, 2)
    for ax in (ax_fib, ax_fm):
        ax.set_facecolor(CANVAS_BG)

    render_canvas_to_axes(fib_canvas, ax_fib, dpi=dpi)
    ax_fib.set_title("FIB", color="white", fontsize=12)

    render_canvas_to_axes(fm_canvas, ax_fm, dpi=dpi)
    ax_fm.set_title("FM", color="white", fontsize=12)

    fig.tight_layout()
    return fig


def save_correlation_plot(path: str, fib_canvas, fm_canvas) -> None:
    """Write :func:`correlation_plot` to *path* as a PNG."""
    fig = correlation_plot(fib_canvas, fm_canvas, dpi=PLOT_DPI)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())


# ── the overlay ───────────────────────────────────────────────────────────


def overlay_figure(fib_image: Any, data: Any, result: Any) -> "Figure":
    """The FIB image with the fiducials picked (o), where the fit puts the FM
    fiducials (x), and the points of interest (+)."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(8, 8 * fib_image.data.shape[0] / fib_image.data.shape[1]))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(fib_image.data, cmap="gray")
    _scatter(ax, [(c.point.x, c.point.y) for c in data.fib_coordinates], "o", "cyan")
    _scatter(ax, [(p.x, p.y) for p in result.reprojected_3d], "x", "yellow")
    _scatter(ax, [(p.image_px.x, p.image_px.y) for p in result.poi], "+", "magenta")
    ax.set_axis_off()
    return fig


def save_overlay(path: str, fib_image: Any, data: Any, result: Any) -> None:
    """Write :func:`overlay_figure` to *path* as a PNG."""
    overlay_figure(fib_image, data, result).savefig(path, dpi=100)


def _scatter(ax: Any, points: Sequence, marker: str, colour: str) -> None:
    if not points:
        return
    xs, ys = zip(*points)
    if marker == "o":
        ax.scatter(xs, ys, marker=marker, s=80, facecolors="none", edgecolors=colour)
    else:
        ax.scatter(xs, ys, marker=marker, s=80, color=colour)
