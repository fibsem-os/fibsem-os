"""Matplotlib rendering of tile grids, stage positions, and minimaps.

Split from the geometry so that consuming the geometry does not cost a matplotlib
import. Pure presentation: nothing here computes a position, it only draws ones
computed elsewhere.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from fibsem import constants
from fibsem.conversions import is_inside_image_bounds
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

# Figure width in inches when the size is derived from the image. The height follows the
# image's aspect, so a 3:1 overview gets a 3:1 figure.
_AUTO_FIGURE_WIDTH_IN = 10.0

# Clear space between a marker's edge and its label, in points.
_LABEL_GAP_POINTS = 6.0

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
    """
    (dx, dy), ha, va = _label_placement(x, y, image_shape, marker_half_points)

    # A halo, rather than the transparency it replaces. A single overview holds bright
    # ice and black vacuum, and text at alpha 0.75 was hard to read on both.
    halo = [path_effects.withStroke(linewidth=2.0, foreground="black", alpha=0.7)]

    line_gap = fontsize + 2.0
    sub_size = max(6.0, fontsize * 0.7)

    # Above the marker, the name goes on top and the description sits between it and the
    # marker; below the marker, the order is the other way up. Either way the name is the
    # line further from the marker, so a column of markers reads name-first.
    if description:
        name_dy = dy + line_gap if va == "bottom" else dy
        sub_dy = dy if va == "bottom" else dy - line_gap
    else:
        name_dy, sub_dy = dy, dy

    ax.annotate(
        label,
        xy=(x, y),
        xytext=(dx, name_dy),
        textcoords="offset points",
        fontsize=fontsize,
        color=colour,
        ha=ha,
        va=va,
        path_effects=halo,
        annotation_clip=False,
    )

    if description:
        ax.annotate(
            description,
            xy=(x, y),
            xytext=(dx, sub_dy),
            textcoords="offset points",
            fontsize=sub_size,
            color=colour,
            ha=ha,
            va=va,
            path_effects=halo,
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
    """Plot stage positions reprojected on an image as matplotlib figure. Assumes image is flat to beam.
    Args:
        image: The image.
        positions: The positions.
        show: Whether to show the plot.
        bound: Whether to only plot points inside the image.
        color: The color of the points. (None -> default colour cycle)
        figsize: Figure size in inches. None (the default) sizes the figure to the
            image's aspect -- see :func:`figsize_for_image`.
    Returns:
        The matplotlib figure."""
    if image.metadata is None or image.metadata.microscope_state is None:
        raise ValueError(
            "Image metadata or microscope state is not set. Cannot reproject stage positions."
        )

    # reproject stage positions onto image
    points = reproject_stage_positions_onto_image2(image=image, positions=positions)

    # construct matplotlib figure
    if figsize is None:
        figsize = figsize_for_image(image.data.shape)
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    ax.imshow(image.data, cmap="gray")

    # `ms` is the marker's width in points, so it reaches half that from its centre.
    marker_size_points = 20
    for i, pt in enumerate(points):
        # if points outside image, don't plot
        if bound and not is_inside_image_bounds(
            (pt.y, pt.x), (image.data.shape[0], image.data.shape[1])
        ):
            continue

        if color is None:
            c = POSITION_COLOURS[i % len(POSITION_COLOURS)]
        else:
            c = color
        ax.plot(
            pt.x,
            pt.y,
            ms=marker_size_points,
            c=c,
            marker="+",
            markeredgewidth=2,
            label=f"{pt.name}",
        )

        if show_names:
            # draw position name next to point
            _draw_position_label(
                ax,
                pt.x,
                pt.y,
                label=pt.name if pt.name else f"Position {i:02d}",
                colour=c,
                fontsize=14,
                image_shape=image.data.shape,
                marker_half_points=marker_size_points / 2.0,
            )

    if show_scalebar:
        try:
            # add scalebar
            from matplotlib_scalebar.scalebar import ScaleBar

            scalebar = ScaleBar(
                dx=image.metadata.pixel_size.x,
                color="black",
                box_color="white",
                box_alpha=0.5,
                location="lower right",
            )
            ax.add_artist(scalebar)
        except Exception as e:
            logging.debug(f"Could not add scalebar: {e}")

    ax.axis("off")
    if show:
        plt.show()

    return fig


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
        color: The color of the points.
        show_scalebar: Whether to show a scalebar
        show_names: Whether to show position names as labels
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

    all_positions = list(positions)
    if current_position is not None:
        all_positions.append(current_position)
    if grid_positions is not None:
        all_positions.extend(grid_positions)

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

        c = color
        if "Grid" in pt.name:
            c = "red"
        elif "Current Position" in pt.name:
            c = "yellow"

        marker_entries.append(
            {
                "point": (pt.x, pt.y),
                "color": c,
                "label": pt.name,
                "description": descriptions.get(pt.name, "") if descriptions else "",
            }
        )

        # show grid radius
        if c == "red" and show_grid_radius:
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
                    color="black",
                    box_color="white",
                    box_alpha=0.5,
                    location="lower right",
                )
            )
        except Exception as e:
            logging.debug(f"Could not add scalebar: {e}")

    ax.axis("off")
    if show:
        plt.show()

    return fig
