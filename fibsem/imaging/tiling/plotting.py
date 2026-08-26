"""Matplotlib rendering of tile grids, stage positions, and minimaps.

Split from the geometry so that consuming the geometry does not cost a matplotlib
import. Pure presentation: nothing here computes a position, it only draws ones
computed elsewhere.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from fibsem import constants
from fibsem.conversions import is_inside_image_bounds
from fibsem.imaging.reduce import downsample
from fibsem.imaging.tiling.geometry import TilePosition
from fibsem.imaging.tiling.reprojection import reproject_stage_positions_onto_image2
from fibsem.structures import (
    FibsemImage,
    FibsemStagePosition,
    OverviewAcquisitionSettings,
)

POSITION_COLOURS = [
    "lime",
    "blue",
    "cyan",
    "magenta",
    "hotpink",
    "yellow",
    "orange",
    "red",
]

# What a grid position and the live stage position are drawn in. Constants rather than
# literals inside the draw loop, because they used to be reached by sniffing the
# position's name for "Grid" / "Current Position".
GRID_POSITION_COLOUR = "red"
CURRENT_POSITION_COLOUR = "yellow"

# How a position a human has flagged is drawn. Literals rather than an import of
# `fibsem.ui.tokens.ERROR_COLOR` / `WARN_COLOR`, which they mirror: `fibsem/ui/__init__`
# eagerly imports the whole Qt widget stack, so importing anything from that package --
# tokens included, despite tokens itself being Qt-free -- makes the importing module
# unimportable wherever PyQt5 is absent. The PDF report is one such module, and CI is one
# such place. Keep these in step with tokens.py by hand.
DEFECT_FAILURE_COLOUR = "#d04040"
DEFECT_REWORK_COLOUR = "#e0a030"

# Figure width in inches when the size is derived from the image. The height follows the
# image's aspect, so a 3:1 overview gets a 3:1 figure.
_AUTO_FIGURE_WIDTH_IN = 10.0

# Clear space between a marker's edge and its label, in points.
_LABEL_GAP_POINTS = 6.0

# What a marker's label sits on. A chip, rather than the black outline it replaces, for
# one reason that decides it: an outline only *partly* darkens what is behind the glyphs,
# so the contrast still depends on the image. One overview holds bright ice and black
# vacuum, and cyan-over-ice stayed marginal however thick the stroke was drawn. A chip
# fixes the ground, so the contrast is chosen once and holds everywhere on the page. It
# also stops the outline thickening the letterforms, which at 8-9pt was most of why the
# old labels looked crude.
#
# The values are the real-space canvas's, not new ones: `RulerOverlay` labels a point on
# the image with exactly this -- `round,pad=0.3` on `CANVAS_BG`, a 0.8pt edge in the
# artist's own colour, alpha 0.8 -- and `FibsemCanvasBase` gives its title the same chip
# without the edge. The exported map is the canvas on paper, so it should be the same
# object. The ruler's alpha rather than the title's 0.55 because the ruler is the sibling
# that sits *on the image*, where what is behind the chip is not known in advance.
#
# Literals rather than an import of `fibsem.ui.stylesheets.CANVAS_BG`, for the reason
# given above `DEFECT_FAILURE_COLOUR`: importing anything from `fibsem.ui` drags in Qt,
# and this module has to render from a workflow hook and on CI. Keep them in step by
# hand. `dict(...)` the chip at the call site -- matplotlib keeps a reference to the dict
# it is handed, and two annotations sharing one is a bug waiting to be written.
_LABEL_CHIP_PAD = 0.3
_LABEL_CHIP = {
    "boxstyle": f"round,pad={_LABEL_CHIP_PAD}",
    "facecolor": "#1e2124",  # CANVAS_BG
    "alpha": 0.8,
    "edgecolor": "none",
    "linewidth": 0.8,
}

# The description under a name: smaller, and the canvas's text-on-a-dark-panel grey
# rather than the marker's colour. On this figure colour means one thing -- what a human
# flagged -- and a description repeating it turns a grid of unflagged lamellae into a
# wall of cyan with nothing standing out.
_SUB_LABEL_SCALE = 0.7
_MIN_LABEL_POINTS = 6.0
_SUB_LABEL_COLOUR = "#e8e8e8"

# How the scalebar is drawn, again the canvas's own: white on a `CANVAS_BG` panel at 0.6,
# and 8pt rather than matplotlib's 10pt default, which is large against everything else
# on the figure (FIB-583). This was black-on-white, which is the one combination that
# cannot be mistaken for part of the canvas.
_SCALEBAR = {
    "color": "white",
    "box_color": "#1e2124",  # CANVAS_BG
    "box_alpha": 0.6,
    "font_properties": {"size": 8},
}

# How near an edge a marker must be before its label is placed on the other side of it.
# A fraction of the image rather than a pixel count, because overviews differ by orders
# of magnitude in pixel size.
_EDGE_FRACTION = 0.18


def figsize_for_image(
    image_shape: Tuple[int, ...], width_in: float = _AUTO_FIGURE_WIDTH_IN
) -> Tuple[float, float]:
    """A figure the same shape as the image, so the axes fills it.

    A square figure holding a wide image spends the difference on blank paper, and
    ``bbox_inches="tight"`` cannot reclaim it: the whitespace is inside the axes
    rectangle, not around it. Measured on a real 3:1 overview drawn at ``figsize=(15,
    15)``, the axes occupied 0.355 of the figure height -- so nearly two thirds of the
    exported PNG was white.

    Args:
        image_shape: the image's ``.shape``; only the first two entries are read.
        width_in: figure width in inches. The height is derived from it.
    Returns:
        ``(width, height)`` in inches, for ``figsize``.
    """
    height = float(image_shape[0]) if len(image_shape) > 0 else 0.0
    width = float(image_shape[1]) if len(image_shape) > 1 else 0.0
    if width <= 0 or height <= 0:
        return (width_in, width_in)
    # Clamped at both ends: an extreme mosaic would otherwise produce a figure feet wide
    # and an inch tall, which has no room for a label, or the reverse.
    aspect = min(max(height / width, 0.2), 5.0)
    return (width_in, width_in * aspect)


def _label_placement(
    x: float,
    y: float,
    image_shape: Tuple[int, ...],
    marker_half_points: float,
) -> Tuple[Tuple[float, float], str, str]:
    """Where a marker's label goes: an offset in points, plus how to align it.

    Two things this fixes, both of which made labels unreadable on real overviews.

    The offset is in **points**, derived from the marker's own size, rather than in image
    pixels. A fixed pixel offset puts the label *inside* its marker at any real scale --
    an overview pixel is tens of nanometres, so the old ``+10, -10`` was well within the
    crosshair.

    And the label is pushed away from whichever edge the marker is near, instead of being
    clipped at it. ``clip_on`` truncates silently, so a name running off the right-hand
    edge looked like a lamella that had simply been given a shorter name.
    """
    height = float(image_shape[0]) if len(image_shape) > 0 else 0.0
    width = float(image_shape[1]) if len(image_shape) > 1 else 0.0
    gap = marker_half_points + _LABEL_GAP_POINTS

    if width > 0 and x > width * (1.0 - _EDGE_FRACTION):
        dx, ha = -gap, "right"
    else:
        dx, ha = gap, "left"

    # Screen terms, not data terms: an offset in points is measured in display space, so
    # a positive dy is up the page whichever way the image's y-axis runs.
    if height > 0 and y < height * _EDGE_FRACTION:
        dy, va = -gap, "top"
    else:
        dy, va = gap, "bottom"

    return (dx, dy), ha, va


def _draw_position_label(
    ax: plt.Axes,
    x: float,
    y: float,
    label: str,
    colour: str,
    fontsize: float,
    image_shape: Tuple[int, ...],
    marker_half_points: float,
    description: str = "",
) -> None:
    """Draw a marker's name, and its description beneath it if there is one.

    The two are drawn as separate artists rather than one two-line string so the
    description can be smaller, and they are stacked so that the block reads downwards
    from the name wherever :func:`_label_placement` put it.

    Each sits on its own chip -- see :data:`_LABEL_CHIP` for why that beats the outline
    it replaces.

    Once they are chips rather than bare text the gap between them has to be *derived*
    from the chip geometry rather than picked. The literal it replaces happens to abut at
    the sizes in use today, but it does not track the padding: raising
    :data:`_LABEL_CHIP_PAD` from 0.25 to 0.7 moves the two chips from touching to 13px of
    overlap, because the boxes grow and the gap does not. Half of each chip's height is
    the value that keeps them meeting whatever the padding and the sub-size become.
    """
    (dx, dy), ha, va = _label_placement(x, y, image_shape, marker_half_points)

    sub_size = max(_MIN_LABEL_POINTS, fontsize * _SUB_LABEL_SCALE)
    # Half of each chip's height, so their edges meet. A chip is the text's own height
    # plus its padding above and below, which is what the `1 + 2 * pad` is.
    line_gap = (
        fontsize * (1 + 2 * _LABEL_CHIP_PAD) + sub_size * (1 + 2 * _LABEL_CHIP_PAD)
    ) / 2.0

    # Above the marker, the name goes on top and the description sits between it and the
    # marker; below the marker, the order is the other way up. Either way the name is the
    # line further from the marker, so a column of markers reads name-first.
    if description:
        name_dy = dy + line_gap if va == "bottom" else dy
        sub_dy = dy if va == "bottom" else dy - line_gap
    else:
        name_dy, sub_dy = dy, dy

    # The name's chip is ringed in the marker's own colour, which is what ties a label
    # to the crosshair it belongs to when several sit close together -- and is what
    # `RulerOverlay` does with `edgecolor=self._color`. The description's is not: two
    # abutting chips with different edges read as two objects, and the ring on the block
    # as a whole is the name's.
    name_chip = dict(_LABEL_CHIP, edgecolor=colour)
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(dx, name_dy),
        textcoords="offset points",
        fontsize=fontsize,
        color=colour,
        ha=ha,
        va=va,
        bbox=name_chip,
        annotation_clip=False,
    )

    if description:
        ax.annotate(
            description,
            xy=(x, y),
            xytext=(dx, sub_dy),
            textcoords="offset points",
            fontsize=sub_size,
            # Neutral, not the marker's colour. On this figure colour means one thing --
            # what a human flagged -- and a description repeating it makes a grid of
            # unflagged lamellae into a wall of cyan with nothing standing out.
            color=_SUB_LABEL_COLOUR,
            ha=ha,
            va=va,
            bbox=dict(_LABEL_CHIP),
            annotation_clip=False,
        )


def plot_tile_positions(
    tiles: list[TilePosition],
    settings: OverviewAcquisitionSettings,
    ax: Optional[plt.Axes] = None,
    stage_positions: Optional[list[FibsemStagePosition]] = None,
) -> Figure:
    """Plot the tile grid with traversal order, for debugging and validation.

    Beam-side adapter over :func:`plot_tile_grid`, which takes the field of view
    directly -- a fluorescence camera has a real FOV in both axes rather than an
    `hfw` with the vertical inferred, the same asymmetry `compute_tile_grid_from_fov`
    exists for.

    Args:
        tiles: Ordered list of TilePosition objects (acquisition order).
        settings: Overview acquisition settings (for FOV dimensions and labels).
        ax: Optional existing axes to draw on; creates a new figure if None.
        stage_positions: Optional list of pre-computed FibsemStagePosition objects
            (same length as tiles). When provided, the actual projected positions are
            overlaid as white crosses + dotted path so you can compare the ideal grid
            against the real stage coordinates returned by project_stable_move.
    Returns:
        The matplotlib Figure.
    """
    image_width, image_height = settings.image_settings.resolution
    tile_fov_x = settings.image_settings.hfw
    tile_fov_y = tile_fov_x * (image_height / image_width)

    return plot_tile_grid(
        tiles,
        fov_x=tile_fov_x,
        fov_y=tile_fov_y,
        ax=ax,
        stage_positions=stage_positions,
        title=(
            f"{settings.tile_order.value.title()} — {settings.nrows}×{settings.ncols} tiles, "
            f"{settings.overlap * 100:.0f}% overlap"
        ),
    )


def plot_tile_grid(
    grid: list[TilePosition],
    fov_x: float,
    fov_y: float,
    order: Optional[list[TilePosition]] = None,
    ax: Optional[plt.Axes] = None,
    stage_positions: Optional[list[FibsemStagePosition]] = None,
    title: Optional[str] = None,
) -> Figure:
    """Draw a tile grid: what gets acquired, in what order, and what gets skipped.

    Skipped tiles are drawn hollow and unnumbered rather than omitted. A sparse
    overview otherwise looks identical to a smaller dense one, and the difference
    between "not acquired" and "acquired but dark" is the whole reason the mask is
    recorded in the first place.

    Args:
        grid: Every tile in the grid, enabled or not. Defines what is drawn.
        fov_x: Tile width in metres.
        fov_y: Tile height in metres.
        order: The enabled tiles in traversal order. Defaults to the enabled tiles in
            the order `grid` gives them. Numbering and the path follow this list.
        ax: Optional existing axes to draw on; creates a new figure if None.
        stage_positions: Optional projected positions, same length as `order`. Overlaid
            as white crosses and a dotted path, so the ideal grid can be compared with
            the real stage coordinates the projection returned.
        title: Optional axes title.
    Returns:
        The matplotlib Figure.
    """
    import matplotlib.patches as mpatches

    if order is None:
        order = [t for t in grid if t.enabled]

    tile_fov_x = fov_x * constants.SI_TO_MICRO  # µm
    tile_fov_y = fov_y * constants.SI_TO_MICRO

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = ax.get_figure()

    def centre(tile: TilePosition) -> Tuple[float, float]:
        return tile.dx * constants.SI_TO_MICRO, tile.dy * constants.SI_TO_MICRO

    # skipped tiles first, so the acquired ones draw over them
    for tile in grid:
        if tile.enabled:
            continue
        cx, cy = centre(tile)
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (cx - tile_fov_x / 2, cy - tile_fov_y / 2),
                tile_fov_x,
                tile_fov_y,
                boxstyle="round,pad=0.01",
                linewidth=1,
                edgecolor="#5a5f6e",
                facecolor="none",
                linestyle="--",
            )
        )
        ax.text(cx, cy, "—", ha="center", va="center", fontsize=8, color="#5a5f6e")

    for order_idx, tile in enumerate(order):
        cx, cy = centre(tile)
        colour = POSITION_COLOURS[tile.row % len(POSITION_COLOURS)]
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (cx - tile_fov_x / 2, cy - tile_fov_y / 2),
                tile_fov_x,
                tile_fov_y,
                boxstyle="round,pad=0.01",
                linewidth=1,
                edgecolor="white",
                facecolor=colour,
                alpha=0.4,
            )
        )
        ax.text(
            cx,
            cy,
            str(order_idx),
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            fontweight="bold",
        )

    # traversal path -- through the acquired tiles only, so a jump over a skipped
    # region is visible as a long arrow rather than hidden
    if len(order) > 1:
        pts = [centre(t) for t in order]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color="white", lw=1.0),
            )

    # overlay actual projected stage positions (if provided)
    if stage_positions is not None and len(stage_positions) > 0 and order:
        # Anchored on the first tile *visited*, not on the grid origin. The projection
        # centres the grid on wherever the stage already is, so its coordinates differ
        # from the grid's by a constant offset. Subtracting only `stage_positions[0]`
        # leaves the overlay displaced by that offset, which looks exactly like a
        # geometry error and hides the thing actually worth seeing: whether the *shape*
        # of the projected path matches the grid it came from.
        ref = stage_positions[0]
        anchor_x, anchor_y = centre(order[0])
        sxs = [
            anchor_x + (sp.x - ref.x) * constants.SI_TO_MICRO for sp in stage_positions
        ]
        sys_ = [
            anchor_y + (sp.y - ref.y) * constants.SI_TO_MICRO for sp in stage_positions
        ]
        ax.plot(sxs, sys_, linestyle=":", color="white", lw=0.8, alpha=0.6)
        ax.plot(
            sxs,
            sys_,
            marker="x",
            color="white",
            ms=6,
            markeredgewidth=1.5,
            linestyle="none",
        )

    sym = constants.MICRON_SYMBOL
    ax.set_xlabel(f"X ({sym})")
    ax.set_ylabel(f"Y ({sym})")
    ax.set_aspect("equal")
    ax.set_facecolor("#1e2027")
    fig.patch.set_facecolor("#1e2027")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    if title:
        ax.set_title(title)
    ax.autoscale_view()
    fig.tight_layout()
    return fig


def plot_stage_positions_on_image(
    image: FibsemImage,
    positions: List[FibsemStagePosition],
    show: bool = False,
    bound: bool = True,
    color: Optional[str] = None,
    show_scalebar: bool = False,
    show_names: bool = True,
    figsize: Optional[Tuple[int, int]] = None,
) -> Figure:
    """Plot stage positions reprojected on an image. Assumes image is flat to beam.

    A thin adapter over :func:`plot_minimap`, which is the one renderer. It used to be a
    second, near-identical implementation, and the two had drifted: different marker
    sizes, different label offsets, a scalebar added through a different call. That
    mattered because this one drew the PDF report's overview page while `plot_minimap`
    drew the dialog's preview -- so what you tuned on screen was not what landed in the
    report.

    The one behaviour kept from the old implementation is `color=None` meaning "walk the
    colour cycle", which is what the statistics plots rely on to tell tracks apart.

    Args:
        image: The image.
        positions: The positions.
        show: Whether to show the plot.
        bound: Whether to only plot points inside the image.
        color: The colour of the points. None walks POSITION_COLOURS per position.
        show_scalebar: Whether to add a scalebar.
        show_names: Whether to label each position with its name.
        figsize: Figure size in inches; None sizes it to the image's aspect.
    Returns:
        The matplotlib figure."""
    per_position: Optional[Dict[str, str]] = None
    if color is None:
        # Named, because plot_minimap keys its overrides by name. A position with no
        # name is given the same placeholder plot_minimap would give it, so the two
        # agree on what to look up.
        per_position = {}
        for i, pos in enumerate(positions):
            name = getattr(pos, "name", None) or f"Position {i:02d}"
            per_position[name] = POSITION_COLOURS[i % len(POSITION_COLOURS)]

    return plot_minimap(
        image=image,
        positions=positions,
        show=show,
        bound=bound,
        color=color if color is not None else POSITION_COLOURS[0],
        colors=per_position,
        show_scalebar=show_scalebar,
        show_names=show_names,
        fontsize=14,
        figsize=figsize,
    )


def plot_minimap(
    image: FibsemImage,
    positions: List[FibsemStagePosition],
    current_position: Optional[FibsemStagePosition] = None,
    grid_positions: Optional[List[FibsemStagePosition]] = None,
    show: bool = False,
    bound: bool = True,
    color: str = "cyan",
    show_scalebar: bool = False,
    show_names: bool = True,
    show_descriptions: bool = False,
    descriptions: Optional[Dict[str, str]] = None,
    colors: Optional[Dict[str, str]] = None,
    show_grid_radius: bool = False,
    fontsize: int = 12,
    markersize: int = 20,
    figsize: Optional[Tuple[int, int]] = None,
    ax: Optional[plt.Axes] = None,
) -> Figure:
    """Plot stage positions reprojected on an image as matplotlib figure. Assumes image is flat to beam.
    Args:
        image: The image.
        positions: The positions.
        current_position: Optional current position to highlight
        grid_positions: Optional grid positions to show
        show: Whether to show the plot.
        bound: Whether to only plot points inside the image.
        color: The color of the points, for any position `colors` does not name.
        show_scalebar: Whether to show a scalebar
        show_names: Whether to show position names as labels
        descriptions: Position name -> free text, drawn under the name when
            `show_descriptions` is set.
        colors: Position name -> colour, overriding `color` for those positions.
            How a caller says that some positions differ from the rest -- a defective
            lamella, say -- without this function needing to know why.
        fontsize: Font size for position name labels (default: 14)
        figsize: Figure size in inches. None (the default) sizes the figure to the
            image's aspect -- see :func:`figsize_for_image` for why that is not merely
            tidier. Ignored when `ax` is supplied.
    Returns:
        The matplotlib figure."""
    if image.metadata is None or image.metadata.microscope_state is None:
        raise ValueError(
            "Image metadata or microscope state is not set. Cannot reproject stage positions."
        )

    # Which list a position came from decides how it is drawn, so that is tracked here
    # rather than recovered later. It used to be recovered later, by testing whether the
    # position's *name* contained "Grid" or "Current Position" -- which quietly drew a
    # lamella called "Grid square 4" in the grid colour, and would have drawn one called
    # "Current Position" in yellow.
    all_positions = list(positions)
    kinds = ["position"] * len(all_positions)
    if current_position is not None:
        all_positions.append(current_position)
        kinds.append("current")
    if grid_positions is not None:
        all_positions.extend(grid_positions)
        kinds.extend(["grid"] * len(grid_positions))

    # construct matplotlib figure/axes
    if ax is None:
        if figsize is None:
            figsize = figsize_for_image(image.data.shape)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.imshow(image.data, cmap="gray")

    # reproject stage positions onto image
    points = reproject_stage_positions_onto_image2(image=image, positions=all_positions)

    marker_entries: List[dict] = []
    for i, pt in enumerate(points):
        # if points outside image, don't plot
        if bound and not is_inside_image_bounds(
            (pt.y, pt.x), (image.data.shape[0], image.data.shape[1])
        ):
            continue

        if pt.name is None:
            pt.name = f"Position {i:02d}"

        kind = kinds[i] if i < len(kinds) else "position"
        if kind == "grid":
            c = GRID_POSITION_COLOUR
        elif kind == "current":
            c = CURRENT_POSITION_COLOUR
        else:
            # A caller that knows something about an individual position -- that this
            # lamella was flagged defective, say -- says so here. Everything unnamed in
            # the mapping keeps the single `color`, so callers with nothing to
            # distinguish are unaffected.
            c = (colors or {}).get(pt.name, color)

        marker_entries.append(
            {
                "point": (pt.x, pt.y),
                "color": c,
                "label": pt.name,
                "description": descriptions.get(pt.name, "") if descriptions else "",
            }
        )

        # show grid radius
        if kind == "grid" and show_grid_radius:
            r_pixels = 1000e-6 / image.metadata.pixel_size.x
            ax.add_artist(
                plt.Circle(
                    (pt.x, pt.y), radius=r_pixels, color=c, fill=False, linewidth=5
                )
            )

    if marker_entries:
        scatter_array = np.array([entry["point"] for entry in marker_entries])
        scatter_colors = [entry["color"] for entry in marker_entries]
        ax.scatter(
            scatter_array[:, 0],
            scatter_array[:, 1],
            c=scatter_colors,
            marker="+",
            s=markersize**2,
            linewidths=2,
        )

        if show_names:
            # `s` is an area in points squared, so a marker drawn at `markersize ** 2`
            # spans `markersize` points and reaches half that from its centre.
            marker_half_points = markersize / 2.0
            for entry in marker_entries:
                x, y = entry["point"]
                _draw_position_label(
                    ax,
                    x,
                    y,
                    label=entry["label"],
                    colour=entry["color"],
                    fontsize=fontsize,
                    image_shape=image.data.shape,
                    marker_half_points=marker_half_points,
                    description=(entry["description"] if show_descriptions else ""),
                )

    if show_scalebar:
        try:
            # add scalebar
            from matplotlib_scalebar.scalebar import ScaleBar

            ax.add_artist(
                ScaleBar(
                    dx=image.metadata.pixel_size.x,
                    location="lower right",
                    **_SCALEBAR,
                )
            )
        except Exception as e:
            logging.debug(f"Could not add scalebar: {e}")

    ax.axis("off")
    if show:
        plt.show()

    return fig


# How many pixels each placed image is drawn at when compositing. A stitched mosaic is
# tens of thousands of pixels across and a PDF page is a few thousand at 200 dpi, so the
# full array is thrown away by the rasteriser anyway -- reducing first turns a
# multi-second render into a fast one. Box-averaged, so a feature smaller than a drawn
# pixel dims into its neighbours rather than disappearing.
_COMPOSITE_MAX_PX = 3000


def compose_overview_extent(
    image: FibsemImage,
    reference: FibsemImage,
    centre: Tuple[float, float],
) -> Tuple[float, float, float, float]:
    """Where *image* lands on *reference*'s pixel grid, as an imshow extent.

    *centre* is where *image* was acquired, already reprojected into *reference*'s
    pixels. The rest is the scale difference between the two: an image acquired at a
    finer pixel size covers proportionally less ground per pixel, so it is drawn smaller.

    The same arithmetic `FibsemRealSpaceCanvas._extent_for` does, and deliberately
    duplicated in about six lines rather than reached for through that class -- the
    canvas is a `FigureCanvasQTAgg`, and this runs from a workflow hook that has no
    display. What is Qt about the canvas is the widget around this, not this.
    """
    scale = image.metadata.pixel_size.x / reference.metadata.pixel_size.x
    half_w = image.data.shape[1] * scale / 2.0
    half_h = image.data.shape[0] * scale / 2.0
    cx, cy = centre
    # y descending, because an image's rows run downwards and imshow was given
    # origin="upper" for the reference.
    return (cx - half_w, cx + half_w, cy + half_h, cy - half_h)


def plot_overview_composite(
    images: List[FibsemImage],
    positions: List[FibsemStagePosition],
    color: str = "cyan",
    colors: Optional[Dict[str, str]] = None,
    descriptions: Optional[Dict[str, str]] = None,
    show_names: bool = True,
    show_descriptions: bool = False,
    show_scalebar: bool = True,
    fontsize: int = 10,
    markersize: int = 20,
    figsize: Optional[Tuple[int, int]] = None,
    ax: Optional[plt.Axes] = None,
) -> Figure:
    """Draw several overviews of the *same view* on one set of axes, and mark positions.

    Only ever call this with images that register with each other -- same beam, same
    stage orientation. Two overviews taken through different beams, or at different
    orientations, are pictures from different directions: compositing them would place
    real pixels at coordinates they were never acquired at, and the result would look
    exactly as authoritative as a correct one. `OverviewView` in the Overview tab draws
    the same line for the same reason.

    Placement is relative to `images[0]`, whose pixel grid becomes the frame. Every other
    image's acquisition position is reprojected into that grid by the *same* function the
    position markers use, so the images and the markers agree by construction rather than
    by two implementations happening to match.

    Args:
        images: overviews of one view. The first is the frame; order otherwise decides
            what draws over what, so pass coarse before fine.
        positions: stage positions to mark, named.
        color: marker colour for anything `colors` does not name.
        colors: position name -> colour, for positions that differ from the rest.
        descriptions: position name -> free text, drawn under the name.
        figsize: None sizes the figure to the composed extent's aspect.
    Returns:
        The matplotlib figure.
    """
    if not images:
        raise ValueError("plot_overview_composite needs at least one image")

    reference = images[0]
    if reference.metadata is None or reference.metadata.microscope_state is None:
        raise ValueError(
            "The reference overview has no microscope state, so nothing can be placed "
            "relative to it."
        )

    placements: List[Tuple[FibsemImage, Tuple[float, float, float, float]]] = []
    ref_h, ref_w = reference.data.shape[0], reference.data.shape[1]
    placements.append((reference, (-0.5, ref_w - 0.5, ref_h - 0.5, -0.5)))

    for image in images[1:]:
        state = getattr(image.metadata, "microscope_state", None)
        centre_position = getattr(state, "stage_position", None)
        if centre_position is None:
            logging.warning(
                "Skipping an overview with no recorded stage position: it cannot be "
                "placed relative to the others."
            )
            continue
        try:
            point = reproject_stage_positions_onto_image2(
                image=reference, positions=[centre_position]
            )[0]
        except Exception as e:
            logging.warning(f"Could not place an overview on the composite: {e}")
            continue
        placements.append(
            (image, compose_overview_extent(image, reference, (point.x, point.y)))
        )

    x0 = min(p[1][0] for p in placements)
    x1 = max(p[1][1] for p in placements)
    y_bottom = max(p[1][2] for p in placements)
    y_top = min(p[1][3] for p in placements)

    if ax is None:
        if figsize is None:
            span_w = abs(x1 - x0)
            span_h = abs(y_bottom - y_top)
            figsize = figsize_for_image((span_h, span_w))
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for image, extent in placements:
        ax.imshow(
            downsample(image.data, _COMPOSITE_MAX_PX),
            cmap="gray",
            extent=extent,
            origin="upper",
            aspect="equal",
            interpolation="nearest",
        )

    ax.set_xlim(x0, x1)
    ax.set_ylim(y_bottom, y_top)

    points = reproject_stage_positions_onto_image2(image=reference, positions=positions)
    marker_entries = []
    for i, pt in enumerate(points):
        # Bounded against the *composed* extent rather than the reference image, which
        # is the point of compositing: a lamella off the edge of the first overview but
        # inside a second one is on this page, and must be drawn.
        if not (x0 <= pt.x <= x1 and y_top <= pt.y <= y_bottom):
            continue
        if pt.name is None:
            pt.name = f"Position {i:02d}"
        marker_entries.append(
            {
                "point": (pt.x, pt.y),
                "color": (colors or {}).get(pt.name, color),
                "label": pt.name,
                "description": (descriptions or {}).get(pt.name, ""),
            }
        )

    if marker_entries:
        scatter = np.array([e["point"] for e in marker_entries])
        ax.scatter(
            scatter[:, 0],
            scatter[:, 1],
            c=[e["color"] for e in marker_entries],
            marker="+",
            s=markersize**2,
            linewidths=2,
        )
        if show_names:
            shape = (abs(y_bottom - y_top), abs(x1 - x0))
            for entry in marker_entries:
                x, y = entry["point"]
                _draw_position_label(
                    ax,
                    x,
                    y,
                    label=entry["label"],
                    colour=entry["color"],
                    fontsize=fontsize,
                    # Offsets are measured from the *drawn* extent, not the reference
                    # image, so a marker near the edge of the composite flips inwards
                    # rather than a marker near the edge of one tile of it.
                    image_shape=shape,
                    marker_half_points=markersize / 2.0,
                    description=entry["description"] if show_descriptions else "",
                )

    if show_scalebar:
        try:
            from matplotlib_scalebar.scalebar import ScaleBar

            ax.add_artist(
                ScaleBar(
                    dx=reference.metadata.pixel_size.x,
                    location="lower right",
                    **_SCALEBAR,
                )
            )
        except Exception as e:
            logging.debug(f"Could not add a scalebar: {e}")

    ax.axis("off")
    return fig
