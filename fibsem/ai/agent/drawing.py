"""The agent's pictures as PNGs, drawn with matplotlib off any GUI thread: a
frame with what a request would do on it (`pictures`), or the agent's own
image with its marks. Drawn in the app's own styles: patterns as the milling
view draws them (``add_shape_mpl``, a colour per stage), points as the editors
do, with a legend, a chip saying whose it is, the marks as
``fibsem.imaging.marks`` draws them, and the export's scalebar and metadata
bar (``fibsem.imaging.export``)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fibsem.ai.agent.pictures import (
    AREA,
    CROSS,
    DASHED,
    DOT,
    FILL,
    PICTURE_WIDTH,
    TARGET,
    _Layer,
    legend_entries,
    stage_colour,
)
from fibsem.imaging.labels import Box
from fibsem.imaging.marks import (
    PT,
    artist_box,
    load_marked_image,
    mark_colour,
    marked_picture,
    png_bytes,
)
from fibsem.theme.tokens import PURPLE_COLOR

# How see-through a pattern to mill is: the milling view's.
_FILL = 0.3


def _mark(axes, shape: str, x: float, y: float, arm: float, colour, width) -> None:
    """One point, *arm* its reach (axes units): "+", a crosshair (a ring, its
    lines stopping short of the centre), or a dot."""
    from matplotlib.patches import Circle

    if shape == DOT:
        axes.add_patch(Circle((x, y), arm * 0.3, color=colour, linewidth=0))
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


def _legend(axes, entries: List[Tuple[str, str, str]], out_width: int) -> List[Box]:
    """*entries* (`legend_entries`) in a dark box at the top left, sized to
    read at the card's width; answers the place it takes."""
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if not entries:
        return []
    size = max(10.0, out_width / 42) * PT
    handles = []
    for colour, shape, text in entries:
        if shape not in (CROSS, TARGET, DOT):
            handles.append(
                Patch(
                    facecolor=to_rgba(colour, _FILL) if shape == FILL else "none",
                    edgecolor=colour,
                    linestyle="--" if shape in (AREA, DASHED) else "-",
                    linewidth=1.5,
                    label=text,
                )
            )
            continue
        handles.append(
            Line2D(
                [],
                [],
                linestyle="",
                marker={CROSS: "+", TARGET: "o", DOT: "o"}[shape],
                markersize=size * (0.6 if shape == DOT else 1.0),
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
    return [artist_box(axes.figure, legend)]


def _chip(axes, text: str, out_width: int) -> List[Box]:
    """Whose the picture is, in the agent's purple at the top right; answers
    the place it takes."""
    if not text:
        return []
    chip = axes.text(
        0.985,
        0.975,
        text,
        transform=axes.transAxes,
        ha="right",
        va="top",
        fontsize=max(11.0, out_width / 40) * PT,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.45", facecolor=PURPLE_COLOR, edgecolor="none"),
    )
    return [artist_box(axes.figure, chip)]


def _patterns(axes, frame, layer: _Layer, line: float) -> None:
    """*layer*'s patterns: to mill, each stage in the milling view's colour;
    else outlined in the layer's colour, dashed when it is."""
    from fibsem.milling.patterning.plotting import PatternStyle, add_shape_mpl

    order = {name: i for i, name in enumerate(layer.stage_names())}
    style = PatternStyle(
        line_width=line * PT,
        fill=_FILL if layer.fill else 0.0,
        edge=1.0,
        line_style="--" if layer.dashed else "-",
    )
    for index, pattern in enumerate(layer.patterns):
        colour = layer.colour
        if layer.fill and index < len(layer.names):
            colour = stage_colour(order[layer.names[index]])
        try:
            add_shape_mpl(pattern, frame, colour, axes, style=style)
        except Exception:  # noqa: BLE001 - one shape that cannot be drawn
            continue


def _export_image(frame):
    """*frame* as the export renders it: with its metadata, or plain when
    that does not read."""
    from fibsem.imaging.export import ExportImage, _to_uint8, from_fibsem_image

    try:
        return from_fibsem_image(frame)
    except Exception:  # noqa: BLE001 - metadata it cannot read: no bar
        return ExportImage(rgb=_to_uint8(frame.data), pixel_size=None, kind="Image")


def picture_png(
    frame,
    layers: List[_Layer],
    marks: Optional[Dict[str, Any]] = None,
    chip: str = "",
) -> bytes:
    """*frame* with *layers* drawn on it, their legend at the top left, *chip*
    at the top right, and the agent's evidence *marks* (its pixels) when
    given; the scalebar and metadata bar under it. A PNG at most
    `PICTURE_WIDTH` wide (FIB-1411). Patterns are drawn against this frame's
    field."""
    from matplotlib.patches import Rectangle

    from fibsem.imaging.export import default_options

    image = _export_image(frame)

    def decorate(figure, axes, scale):
        out_width = figure.bbox.width
        line = max(1.5, out_width / 340)  # PNG pixels
        arm = 8 * line / scale
        for layer in layers:
            _patterns(axes, frame, layer, line)
            for (x0, y0), (x1, y1) in layer.areas:
                axes.add_patch(
                    Rectangle(
                        (min(x0, x1), min(y0, y1)),
                        abs(x1 - x0),
                        abs(y1 - y0),
                        fill=False,
                        edgecolor=layer.colour,
                        linewidth=line * 1.3 * PT,
                        linestyle="--",
                    )
                )
            for shape in (CROSS, TARGET, DOT):
                for x, y in layer.points(shape):
                    _mark(axes, shape, x, y, arm, layer.colour, line * 1.3 * PT)
        return _legend(axes, legend_entries(layers), out_width) + _chip(
            axes, chip, out_width
        )

    rgb = marked_picture(
        image,
        marks,
        mark_colour("agent"),
        default_options(image),
        PICTURE_WIDTH,
        decorate,
    )
    return png_bytes(rgb)


def evidence_png(evidence: Optional[Dict[str, Any]], chip: str = "") -> Optional[bytes]:
    """The image of *evidence*, its marks drawn on it in the colour of whoever
    marked it and *chip* at the top right, at most `PICTURE_WIDTH` wide (a
    beam image or a stack's projection with its scalebar and metadata bar): a
    PNG, or None when there is no image."""
    evidence = evidence or {}
    path = evidence.get("image")
    image, options = (
        load_marked_image(str(path), evidence.get("channels")) if path else (None, None)
    )
    if image is None:
        return None
    rgb = marked_picture(
        image,
        evidence.get("marks"),
        mark_colour(evidence.get("by")),
        options,
        PICTURE_WIDTH,
        lambda figure, axes, scale: _chip(axes, chip, figure.bbox.width),
    )
    return png_bytes(rgb)
