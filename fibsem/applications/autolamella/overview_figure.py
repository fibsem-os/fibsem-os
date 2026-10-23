"""The experiment's lamellae drawn on an overview: the overview plot dialog's
figure, and the picture the agent cites (`experiment_render_markers`), drawn
as marks so their numbers are placed clear of each other.

Pure: no Qt. Positions are the lamellae's milling poses (`milling_positions`),
reprojected the same way for both.
"""

from __future__ import annotations

import copy
import re
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple

from matplotlib.figure import Figure

from fibsem.imaging.tiling.plotting import plot_minimap
from fibsem.structures import FibsemImage, FibsemStagePosition

if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import Lamella

# Colours by how a lamella stands, when drawn by status.
OK_COLOR = "#6be37a"
FAILED_COLOR = "#e05555"

_NUMBERED = re.compile(r"^(\d+)-")


def short_name(name: str) -> str:
    """A petname's number ("02-alive-skunk" is "02"); any other name whole."""
    match = _NUMBERED.match(name or "")
    return match.group(1) if match else name


def milling_positions(
    lamellae: Iterable["Lamella"], short: bool = False
) -> List[Tuple["Lamella", FibsemStagePosition]]:
    """Each lamella with a milling pose and that pose's stage position, named
    for it (copied: the lamella's own is left as it is). A position missing a
    coordinate can't be reprojected, and is left out."""
    placed = []
    for lamella in lamellae:
        pose = lamella.milling_pose
        position = pose.stage_position if pose is not None else None
        if position is None or None in (position.x, position.y, position.r, position.t):
            continue
        position = copy.deepcopy(position)
        position.name = short_name(lamella.name) if short else lamella.name
        placed.append((lamella, position))
    return placed


def overview_figure(
    image: FibsemImage,
    lamellae: Iterable["Lamella"],
    *,
    color: str = "cyan",
    by_status: bool = False,
    short_names: bool = False,
    outline: bool = False,
    fontsize: int = 12,
    markersize: int = 20,
    show_scalebar: bool = False,
    show_names: bool = True,
    show_descriptions: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
) -> Figure:
    """*lamellae* on the beam overview *image*: in *color*, or green and red
    for failed ones when *by_status*; names as numbers when *short_names*."""
    placed = milling_positions(lamellae, short=short_names)
    colors = None
    if by_status:
        colors = [FAILED_COLOR if lam.is_failure else OK_COLOR for lam, _ in placed]
    descriptions = {position.name: lamella.description for lamella, position in placed}
    return plot_minimap(
        image,
        [position for _, position in placed],
        current_position=None,
        grid_positions=None,
        color=color,
        colors=colors,
        fontsize=fontsize,
        markersize=markersize,
        show_scalebar=show_scalebar,
        show_names=show_names,
        show_descriptions=show_descriptions,
        descriptions=descriptions,
        figsize=figsize,
        outline=outline,
    )


def markers_marks(
    image: FibsemImage,
    lamellae: Iterable["Lamella"],
    fov: bool = False,
    current: Optional[FibsemStagePosition] = None,
) -> dict:
    """*lamellae* on *image* as marks (`fibsem.imaging.marks`): a cross at
    each milling position named by its number, green or red for a failed one;
    with *fov*, the field of view a lamella image there covers, as the
    Overview canvas draws it; the *current* stage position as a yellow target.
    Positions off the image are left out."""
    from fibsem.config import POSITION_FOV_HEIGHT, POSITION_FOV_WIDTH
    from fibsem.imaging.tiling.reprojection import (
        reproject_stage_positions_onto_image2,
    )
    from fibsem.theme.tokens import CURRENT_POSITION_COLOUR

    placed = milling_positions(lamellae, short=True)
    positions = [position for _, position in placed]
    if current is not None:
        positions.append(current)
    pixels = reproject_stage_positions_onto_image2(image, positions)
    height, width = image.data.shape[:2]
    size = image.metadata.pixel_size.x
    box = (POSITION_FOV_WIDTH / size, POSITION_FOV_HEIGHT / size)
    points, areas = [], []
    for index, pixel in enumerate(pixels):
        if not (0 <= pixel.x < width and 0 <= pixel.y < height):
            continue
        if index == len(placed):
            points.append(
                {
                    "x": pixel.x,
                    "y": pixel.y,
                    "shape": "target",
                    "label": "stage",
                    "colour": CURRENT_POSITION_COLOUR,
                }
            )
            continue
        lamella, position = placed[index]
        colour = FAILED_COLOR if lamella.is_failure else OK_COLOR
        points.append(
            {"x": pixel.x, "y": pixel.y, "label": position.name, "colour": colour}
        )
        if fov:
            areas.append(
                {
                    "x": pixel.x - box[0] / 2,
                    "y": pixel.y - box[1] / 2,
                    "width": box[0],
                    "height": box[1],
                    "colour": colour,
                }
            )
    return {"points": points, "areas": areas}


def save_markers_picture(
    image: FibsemImage,
    lamellae: Iterable["Lamella"],
    path: str,
    fov: bool = False,
    current: Optional[FibsemStagePosition] = None,
) -> bytes:
    """*lamellae* on *image* as the agent cites them (`markers_marks`), each
    number placed clear of the others, with the scale bar and metadata bar.
    Saved as a PNG at *path*; answers a JPEG preview of it."""
    from fibsem.imaging.export import from_fibsem_image
    from fibsem.imaging.marks import (
        DRAWN_WIDTH,
        jpeg_bytes,
        marked_picture,
        options_for,
        save_drawn,
    )

    export = from_fibsem_image(image)
    marks = markers_marks(image, lamellae, fov=fov, current=current)
    rgb = marked_picture(export, marks, OK_COLOR, options_for(export), DRAWN_WIDTH)
    save_drawn(rgb, path)
    return jpeg_bytes(rgb)
