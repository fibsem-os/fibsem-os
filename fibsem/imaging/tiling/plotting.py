"""Matplotlib rendering of tile grids, stage positions, and minimaps.

Split from the geometry so that consuming the geometry does not cost a matplotlib
import. Pure presentation: nothing here computes a position, it only draws ones
computed elsewhere.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

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

# Figure width in inches when the size is derived from the image. The height follows
# the image's aspect, so a 3:1 overview gets a 3:1 figure.
_AUTO_FIGURE_WIDTH_IN = 10.0


def figsize_for_image(
    image_shape: Tuple[int, ...], width_in: float = _AUTO_FIGURE_WIDTH_IN
) -> Tuple[float, float]:
    """A figure the same shape as the image, so the image fills it.

    A wide overview in a square figure spends the difference on blank paper, and
    ``bbox_inches="tight"`` cannot reclaim it: the gap sits inside the bounding box,
    between the title and the image, rather than around it.

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
    # Clamped, so an extreme mosaic cannot produce a figure a foot wide and a fraction
    # of an inch tall, with no room left for a title or a label.
    aspect = min(max(height / width, 0.2), 5.0)
    return (width_in, width_in * aspect)


# Clear space between the end of a marker's arm and its label, in points.
_LABEL_GAP_POINTS = 6.0

# How near an edge a marker must be before its label goes on the other side of it. A
# fraction of the image rather than a pixel count, because overviews differ by orders
# of magnitude in pixel count.
_EDGE_FRACTION = 0.18


def _label_placement(
    x: float,
    y: float,
    image_shape: Tuple[int, ...],
    marker_half_points: float,
) -> Tuple[Tuple[float, float], str, str]:
    """Where a marker's label goes: an offset in points, and how to align the text.

    In points, derived from the marker's size, rather than a fixed count of image
    pixels: an overview pixel is tens of nanometres, so a ten-pixel offset put the
    label inside the crosshair at any real scale.

    And away from whichever edge the marker is near, rather than clipped at it -- a
    clipped name gives no sign it was cut, and reads as a lamella with a shorter name.

    Returns:
        ``((dx, dy), ha, va)`` for ``annotate(..., textcoords="offset points")``.
    """
    height = float(image_shape[0]) if len(image_shape) > 0 else 0.0
    width = float(image_shape[1]) if len(image_shape) > 1 else 0.0
    gap = marker_half_points + _LABEL_GAP_POINTS

    if width > 0 and x > width * (1.0 - _EDGE_FRACTION):
        dx, ha = -gap, "right"
    else:
        dx, ha = gap, "left"

    # Display space, not data space: a positive dy is up the page whichever way the
    # image's y-axis runs. Image y grows downwards, so a small y is the top edge.
    if height > 0 and y < height * _EDGE_FRACTION:
        dy, va = -gap, "top"
    else:
        dy, va = gap, "bottom"

    return (dx, dy), ha, va


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
    figsize: Optional[Tuple[float, float]] = None,
) -> Figure:
    """Plot stage positions reprojected on an image as matplotlib figure. Assumes image is flat to beam.

    An adapter over :func:`plot_minimap`, which draws the overview plot dialog's
    preview. This used to be a second renderer, and the two had drifted, so the
    overview page in the PDF report was not what the dialog showed. What it keeps of
    its own: a colour per position when ``color`` is None, which the statistics plots
    use to tell tracks apart, and its 14 pt labels.

    Args:
        image: The image.
        positions: The positions.
        show: Whether to show the plot.
        bound: Whether to only plot points inside the image.
        color: The color of the points. (None -> default colour cycle)
        figsize: Figure size in inches. None sizes it to the image's aspect.
    Returns:
        The matplotlib figure."""
    colors = None
    if color is None:
        colors = [
            POSITION_COLOURS[i % len(POSITION_COLOURS)] for i in range(len(positions))
        ]
    return plot_minimap(
        image,
        positions,
        show=show,
        bound=bound,
        color=color or POSITION_COLOURS[0],
        colors=colors,
        show_scalebar=show_scalebar,
        show_names=show_names,
        fontsize=14,
        markersize=20,
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
    colors: Optional[Sequence[str]] = None,
    show_scalebar: bool = False,
    show_names: bool = True,
    show_descriptions: bool = False,
    descriptions: Optional[Dict[str, str]] = None,
    show_grid_radius: bool = False,
    fontsize: int = 12,
    markersize: int = 20,
    figsize: Optional[Tuple[float, float]] = None,
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
        colors: Optional colour per entry of ``positions``, overriding ``color``.
        show_scalebar: Whether to show a scalebar
        show_names: Whether to show position names as labels
        fontsize: Font size for position name labels (default: 14)
        figsize: Figure size in inches. None sizes it to the image's aspect.
    Returns:
        The matplotlib figure."""
    if image.metadata is None or image.metadata.microscope_state is None:
        raise ValueError(
            "Image metadata or microscope state is not set. Cannot reproject stage positions."
        )

    # What each entry is, by which list it came from. Grid and current positions used
    # to be told apart by their names -- so a lamella whose name contained "Grid" was
    # drawn red, with the grid's radius around it.
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

        kind = kinds[i]
        if kind == "grid":
            c = "red"
        elif kind == "current":
            c = "yellow"
        elif colors is not None and i < len(colors):
            c = colors[i]
        else:
            c = color

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
            for entry in marker_entries:
                x, y = entry["point"]
                # `s` is the marker's area in points squared, so its arm is half
                # of markersize.
                (dx, dy), ha, va = _label_placement(
                    x, y, image.data.shape, marker_half_points=markersize / 2
                )
                description = entry["description"] if show_descriptions else ""
                # The name is the line further from the marker, so a column of
                # markers reads name-first: above the marker, the description sits
                # between them; below it, under the name.
                line_gap = fontsize + 2
                name_dy, sub_dy = dy, dy
                if description:
                    if va == "bottom":
                        name_dy = dy + line_gap
                    else:
                        sub_dy = dy - line_gap
                ax.annotate(
                    entry["label"],
                    xy=(x, y),
                    xytext=(dx, name_dy),
                    textcoords="offset points",
                    ha=ha,
                    va=va,
                    fontsize=fontsize,
                    color=entry["color"],
                    alpha=0.75,
                )
                # description as a smaller subtitle with the name
                if description:
                    ax.annotate(
                        description,
                        xy=(x, y),
                        xytext=(dx, sub_dy),
                        textcoords="offset points",
                        ha=ha,
                        va=va,
                        fontsize=max(6, int(round(fontsize * 0.7))),
                        color=entry["color"],
                        alpha=0.6,
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
