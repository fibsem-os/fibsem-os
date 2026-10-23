"""The experiment's lamellae drawn on an overview: one figure for the overview
plot dialog and for the picture the agent cites (`experiment_render_markers`).

Pure: no Qt. Positions are the lamellae's milling poses, reprojected by
`plot_minimap`.
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


def save_markers_picture(
    image: FibsemImage, lamellae: Iterable["Lamella"], path: str
) -> bytes:
    """*lamellae* on *image* as the agent cites them: numbers, green and red,
    outlined, with a scale bar. Saved as a PNG at *path*; answers a JPEG
    preview of it."""
    import io
    import os

    height, width = image.data.shape[:2]
    fig = overview_figure(
        image,
        lamellae,
        by_status=True,
        short_names=True,
        outline=True,
        fontsize=22,
        markersize=40,
        show_scalebar=True,
        figsize=(8, 8 * height / width),
    )
    saved = {"bbox_inches": "tight", "pad_inches": 0, "facecolor": "black"}
    buf = io.BytesIO()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=200, **saved)
    fig.savefig(buf, format="jpeg", dpi=96, pil_kwargs={"quality": 85}, **saved)
    return buf.getvalue()
