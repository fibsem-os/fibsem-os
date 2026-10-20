"""The agent's pictures as PNGs, drawn with matplotlib off any GUI thread: a
frame with what a request would do on it (`pictures`), or the agent's own
image with its marks. Patterns are drawn as the milling plots draw them
(``add_shape_mpl``), in the agent's colours."""

from __future__ import annotations

import io
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fibsem.ai.agent.pictures import (
    AREA,
    CROSS,
    DOT,
    FILL,
    FILL_ALPHA,
    PICTURE_WIDTH,
    TARGET,
    _Layer,
    legend_entries,
)
from fibsem.theme.tokens import PURPLE_COLOR

_DPI = 100
# Points per pixel of the PNG.
_PT = 72 / _DPI
# The evidence marks' dark halo, so they read on a grey overview.
_HALO = (0, 0, 0, 200 / 255)


def _canvas(data: np.ndarray):
    """A figure showing *data* (2D grey or RGB, uint8) at most `PICTURE_WIDTH`
    wide, its axes in *data*'s pixels: ``(figure, axes, scale, width)``, the
    PNG's *width* in pixels."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    height, width = data.shape[:2]
    scale = min(1.0, PICTURE_WIDTH / width)
    out_width, out_height = max(1, round(width * scale)), max(1, round(height * scale))
    figure = Figure(figsize=(out_width / _DPI, out_height / _DPI), dpi=_DPI)
    FigureCanvasAgg(figure)
    axes = figure.add_axes((0, 0, 1, 1))
    axes.imshow(
        data,
        cmap="gray",
        vmin=0,
        vmax=255,
        extent=(0, width, height, 0),
        interpolation="antialiased",
    )
    axes.set_xlim(0, width)
    axes.set_ylim(height, 0)
    axes.set_axis_off()
    return figure, axes, scale, out_width


def _png(figure) -> bytes:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=_DPI)
    return buffer.getvalue()


def _mark(axes, shape: str, x: float, y: float, arm: float, colour, width) -> None:
    """One point, *arm* its reach (axes units): "+", a crosshair (a ring, its
    lines stopping short of the centre), or a dot."""
    from matplotlib.patches import Circle

    if shape == DOT:
        axes.add_patch(Circle((x, y), arm * 0.35, color=colour, linewidth=0))
        return
    if shape == TARGET:
        axes.add_patch(Circle((x, y), arm * 0.6, fill=False, ec=colour, lw=width))
        gap = arm * 0.25
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            axes.plot(
                [x + dx * gap, x + dx * arm],
                [y + dy * gap, y + dy * arm],
                color=colour,
                linewidth=width,
                solid_capstyle="butt",
            )
        return
    axes.plot([x - arm, x + arm], [y, y], color=colour, linewidth=width)
    axes.plot([x, x], [y - arm, y + arm], color=colour, linewidth=width)


def _legend(axes, entries: List[Tuple[str, str, str]], out_width: int) -> None:
    """*entries* (`legend_entries`) in a dark box at the top left, sized to
    read at the card's width."""
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if not entries:
        return
    size = max(10.0, out_width / 38) * _PT
    handles = []
    for colour, shape, text in entries:
        if shape not in (CROSS, TARGET, DOT):
            handles.append(
                Patch(
                    facecolor=to_rgba(colour, FILL_ALPHA / 255)
                    if shape == FILL
                    else "none",
                    edgecolor=colour,
                    linestyle="--" if shape == AREA else "-",
                    linewidth=1.5,
                    label=text,
                )
            )
            continue
        marker = {CROSS: "+", TARGET: "o", DOT: "o"}[shape]
        handles.append(
            Line2D(
                [],
                [],
                linestyle="",
                marker=marker,
                markersize=size,
                markeredgewidth=1.5,
                color=colour,
                markerfacecolor=colour if shape == DOT else "none",
                label=text,
            )
        )
    legend = axes.legend(
        handles=handles,
        loc="upper left",
        fontsize=size,
        labelcolor="white",
        facecolor="black",
        edgecolor="none",
        framealpha=170 / 255,
        handlelength=1.4,
    )
    legend.get_frame().set_boxstyle("round", pad=0.2)


def _evidence_marks(axes, marks: Optional[Dict[str, Any]], scale, out_width) -> None:
    """The agent's *marks* (its image's pixels) in the agent's purple: a cross
    per point, a box per area, each with its label and a dark halo, sized for
    the picture shown a third as wide (FIB-1503)."""
    from matplotlib import patheffects
    from matplotlib.patches import Rectangle

    if not marks:
        return
    line = max(3.0, out_width / 110)  # PNG pixels
    halo = [
        patheffects.withStroke(linewidth=line * 2.2 * _PT, foreground=_HALO),
    ]
    text_halo = [
        patheffects.withStroke(linewidth=line * 1.2 * _PT, foreground=_HALO),
    ]
    font = max(18.0, out_width / 20) * _PT
    size = line * 5 / scale  # the cross's reach, in the image's pixels
    gap = line / scale
    right = out_width / scale
    for point in marks.get("points") or []:
        x, y = float(point["x"]), float(point["y"])
        for xs, ys in (([x - size, x + size], [y, y]), ([x, x], [y - size, y + size])):
            axes.plot(
                xs,
                ys,
                color=PURPLE_COLOR,
                linewidth=line * _PT,
                solid_capstyle="round",
                path_effects=halo,
            )
        label = point.get("label")
        if label:
            # Where it would run off the picture's right edge, under the mark
            # and ending at the edge.
            beside = x + size + gap
            fits = beside + 0.6 * font / _PT * len(label) / scale < right
            axes.text(
                beside if fits else right - gap,
                y - gap if fits else y + size + gap,
                label,
                ha="left" if fits else "right",
                va="baseline" if fits else "top",
                color=PURPLE_COLOR,
                fontsize=font,
                fontweight="bold",
                path_effects=text_halo,
            )
    for area in marks.get("areas") or []:
        x, y = float(area["x"]), float(area["y"])
        axes.add_patch(
            Rectangle(
                (x, y),
                float(area["width"]),
                float(area["height"]),
                fill=False,
                edgecolor=PURPLE_COLOR,
                linewidth=line * _PT,
                path_effects=halo,
            )
        )
        if area.get("label"):
            axes.text(
                x,
                y - gap,
                area["label"],
                va="baseline",
                color=PURPLE_COLOR,
                fontsize=font,
                fontweight="bold",
                path_effects=text_halo,
            )


def picture_png(
    frame, layers: List[_Layer], marks: Optional[Dict[str, Any]] = None
) -> bytes:
    """*frame* with *layers* drawn on it, their legend at the top left, and
    the agent's evidence *marks* (its pixels) when given: a PNG at most
    `PICTURE_WIDTH` wide (FIB-1411). Patterns are drawn against this frame's
    field."""
    from matplotlib.patches import Rectangle

    from fibsem.imaging.export import _to_uint8
    from fibsem.milling.patterning.plotting import PatternStyle, add_shape_mpl

    figure, axes, scale, out_width = _canvas(_to_uint8(frame.data))
    # Seen at the card's width, a third of this or less.
    line = max(2.0, out_width / 150)  # PNG pixels
    arm = 6 * line / scale
    for layer in layers:
        style = PatternStyle(
            line_width=line * _PT,
            fill=FILL_ALPHA / 255 if layer.fill else 0.0,
            edge=1.0,
        )
        for pattern in layer.patterns:
            try:
                add_shape_mpl(pattern, frame, layer.colour, axes, style=style)
            except Exception:  # noqa: BLE001 - one shape that cannot be drawn
                continue
        for (x0, y0), (x1, y1) in layer.areas:
            axes.add_patch(
                Rectangle(
                    (min(x0, x1), min(y0, y1)),
                    abs(x1 - x0),
                    abs(y1 - y0),
                    fill=False,
                    edgecolor=layer.colour,
                    linewidth=line * _PT,
                    linestyle="--",
                )
            )
        for shape in (CROSS, TARGET, DOT):
            for x, y in layer.points(shape):
                _mark(axes, shape, x, y, arm, layer.colour, line * 1.5 * _PT)
    _evidence_marks(axes, marks, scale, out_width)
    _legend(axes, legend_entries(layers), out_width)
    return _png(figure)


def _evidence_data(path: str) -> Optional[np.ndarray]:
    """The evidence image's pixels as uint8: a beam image (TIFF) as the
    pictures draw a frame, anything else as RGB; None when it does not read."""
    if os.path.splitext(path)[1].lower() in (".tif", ".tiff"):
        from fibsem.imaging.export import _to_uint8
        from fibsem.structures import FibsemImage

        try:
            return _to_uint8(FibsemImage.load(path).data)
        except Exception:  # noqa: BLE001 - try it as a plain image
            pass
    from PIL import Image

    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"))
    except Exception:  # noqa: BLE001 - shown without it
        return None


def evidence_png(evidence: Optional[Dict[str, Any]]) -> Optional[bytes]:
    """The image of the agent's *evidence*, its marks drawn on it, at most
    `PICTURE_WIDTH` wide: a PNG, or None when there is no image to show."""
    path = (evidence or {}).get("image")
    data = _evidence_data(str(path)) if path else None
    if data is None:
        return None
    figure, axes, scale, out_width = _canvas(data)
    _evidence_marks(axes, (evidence or {}).get("marks"), scale, out_width)
    return _png(figure)
