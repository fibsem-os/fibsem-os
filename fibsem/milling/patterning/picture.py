"""Milling stages drawn on a saved ion image as a picture to keep or look at:
each stage's patterns in the milling view's colour with a legend, the stages
of related tasks behind them in black (as the lamella editor shows them), and
optionally each rectangle's scan direction, the alignment area and the point
of interest. The export's scale bar and metadata bar go under it
(`fibsem.imaging.marks`). Drawn with matplotlib off any GUI thread.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from fibsem.imaging.marks import (
    DRAWN_WIDTH,
    PT,
    artist_box,
    legend,
    marked_picture,
    options_for,
)
from fibsem.milling.patterning.plotting import (
    COLOURS,
    PatternStyle,
    _rect_pattern_to_image_pixels,
    add_shape_mpl,
)
from fibsem.structures.image import FibsemImage
from fibsem.structures.settings import FibsemRectangle, FibsemRectangleSettings, Point

# The milling view's fill, and its colour for stages drawn behind.
_FILL = 0.3
BACKGROUND_COLOUR = "black"
# The point of interest in white, not the editor's magenta: a stage's colour
# here. The alignment area in the editor's colour.
POI_COLOUR = "white"
ALIGNMENT_COLOUR = "limegreen"
SCAN_COLOUR = "yellow"
_SCANS = ("TopToBottom", "BottomToTop")


def _shapes(stage) -> List[Any]:
    try:
        return list(stage.define_patterns())
    except Exception:  # noqa: BLE001 - a stage that can't define its patterns
        return []


def _draw_stage(axes, image, stage, colour, line, fill, zorder) -> None:
    style = PatternStyle(line_width=line * PT, fill=fill, edge=1.0)
    for shape in _shapes(stage):
        try:
            add_shape_mpl(shape, image, colour, axes, zorder=zorder, style=style)
        except Exception:  # noqa: BLE001 - one shape that cannot be drawn
            continue


def _scan_arrows(axes, image, stage, line, scale) -> None:
    """The scan direction of each of *stage*'s rectangles, an arrow down its
    middle, as the milling view's scan direction overlay draws it."""
    from matplotlib.patches import FancyArrowPatch

    pixel_size = image.metadata.pixel_size.x
    for shape in _shapes(stage):
        direction = getattr(shape, "scan_direction", None)
        if not isinstance(shape, FibsemRectangleSettings) or direction not in _SCANS:
            continue
        cx, cy, _, height = _rect_pattern_to_image_pixels(
            shape, pixel_size, image.data.shape
        )
        margin = height * 0.15
        top, bottom = cy - height / 2 + margin, cy + height / 2 - margin
        start, end = (top, bottom) if direction == "TopToBottom" else (bottom, top)
        # The head at most a third of the arrow, in points.
        head = min(line * 4, (bottom - top) * scale / 3) * PT
        axes.add_patch(
            FancyArrowPatch(
                (cx, start),
                (cx, end),
                arrowstyle=f"-|>,head_length={head},head_width={head / 2}",
                mutation_scale=1,
                color=SCAN_COLOUR,
                linewidth=line * PT,
                zorder=20,
            )
        )


def milling_picture(
    image: FibsemImage,
    stages: Sequence,
    background: Sequence = (),
    scan_direction: bool = False,
    alignment_area: Optional[FibsemRectangle] = None,
    poi: Optional[Point] = None,
    metadata_bar: bool = True,
    max_width: int = DRAWN_WIDTH,
) -> np.ndarray:
    """*stages* drawn on *image* (an ion image with its metadata), each in the
    milling view's colour and named in the legend, *background* stages behind
    them in black; each rectangle's scan direction when *scan_direction*; the
    *alignment_area* (fractions of the image) and the *poi* (metres from the
    image centre) when given. RGB, at most *max_width* wide, with the scale
    bar and, when *metadata_bar*, the metadata bar under it."""
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Patch, Rectangle

    from fibsem.conversions import microscope_image_to_image_coordinates
    from fibsem.imaging.export import from_fibsem_image

    export = from_fibsem_image(image)
    options = options_for(export, metadata_bar=metadata_bar)
    height, width = image.data.shape[:2]

    def decorate(figure, axes, scale):
        line = max(1.5, figure.bbox.width / 400)  # PNG pixels
        for stage in background:
            _draw_stage(axes, image, stage, BACKGROUND_COLOUR, line, _FILL, 2)
        handles = []
        drawn_before = len(axes.patches)
        for index, stage in enumerate(stages):
            colour = COLOURS[index % len(COLOURS)]
            _draw_stage(axes, image, stage, colour, line, _FILL, 4)
            handles.append(
                Patch(
                    facecolor=to_rgba(colour, _FILL),
                    edgecolor=colour,
                    linewidth=1.5,
                    label=stage.name,
                )
            )
            if scan_direction:
                _scan_arrows(axes, image, stage, line, scale)
        if background:
            handles.append(
                Patch(
                    facecolor=to_rgba(BACKGROUND_COLOUR, _FILL),
                    edgecolor="white",
                    linewidth=0.8,
                    label="Related tasks",
                )
            )
        if alignment_area is not None:
            rect = alignment_area
            axes.add_patch(
                Rectangle(
                    (rect.left * width, rect.top * height),
                    rect.width * width,
                    rect.height * height,
                    fill=False,
                    edgecolor=ALIGNMENT_COLOUR,
                    linewidth=line * PT,
                    linestyle="--",
                    zorder=6,
                )
            )
            handles.append(
                Patch(
                    facecolor="none",
                    edgecolor=ALIGNMENT_COLOUR,
                    linestyle="--",
                    linewidth=1.5,
                    label="Alignment area",
                )
            )
        # The point of interest's label keeps clear of the stages.
        taken = [artist_box(figure, p) for p in axes.patches[drawn_before:]]
        return taken + legend(axes, handles)

    marks: Optional[Dict[str, Any]] = None
    if poi is not None:
        pixel = microscope_image_to_image_coordinates(
            poi, image.data.shape[:2], image.metadata.pixel_size.x
        )
        marks = {
            "points": [{"x": pixel.x, "y": pixel.y, "shape": "target", "label": "POI"}]
        }
    return marked_picture(export, marks, POI_COLOUR, options, max_width, decorate)
