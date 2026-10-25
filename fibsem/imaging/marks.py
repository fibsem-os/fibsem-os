"""Marks on an image, drawn with matplotlib off any GUI thread: the one marks
vocabulary (`fibsem.workflows.evidence`) as evidence, the draw tools and the
request card show it. Points (a cross or a ring crosshair), areas, circles and
lines (an arrow, or measured as the ruler measures), each with its label
placed by `fibsem.imaging.labels`; the agent's in its purple, the operator's
in the accent blue. The picture keeps the export's scale bar and metadata bar
(`fibsem.imaging.export.render_export`).
"""

from __future__ import annotations

import copy
import io
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from fibsem.fm.preview import is_fluorescence_image
from fibsem.imaging.export import (
    ExportImage,
    ExportOptions,
    _display_rgb,
    default_options,
    overlay_boxes,
    render_export,
)
from fibsem.imaging.labels import Box, LabelPlacer
from fibsem.theme.tokens import ACCENT_COLOR, AGENT_MARK_COLOR

_DPI = 100
# Points per pixel of the PNG.
PT = 72 / _DPI
# The dark edge under a mark's lines, so they read on a bright image.
_HALO = (0, 0, 0, 0.6)

# Whose marks they are, by evidence's `by`.
MARK_COLOURS = {"agent": AGENT_MARK_COLOR, "operator": ACCENT_COLOR}


def mark_colour(by: Optional[str]) -> str:
    return MARK_COLOURS.get(by or "agent", AGENT_MARK_COLOR)


def canvas(data: np.ndarray, max_width: int):
    """A figure showing *data* (2D grey or RGB, uint8) at most *max_width*
    wide, its axes in *data*'s pixels: ``(figure, axes, scale)``, PNG pixels
    to one of *data*'s."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    height, width = data.shape[:2]
    scale = min(1.0, max_width / width)
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
    return figure, axes, scale


def rgb_of(figure) -> np.ndarray:
    figure.canvas.draw()
    return np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()


def png_bytes(rgb: np.ndarray) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return buffer.getvalue()


def jpeg_bytes(rgb: np.ndarray, max_width: int = 1024) -> bytes:
    from PIL import Image

    image = Image.fromarray(rgb)
    image.thumbnail((max_width, max_width * 4))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def artist_box(figure, artist) -> Box:
    """Where *artist* is drawn, in the PNG's pixels from its top left."""
    extent = artist.get_window_extent(figure.canvas.get_renderer())
    height = figure.bbox.height
    return (extent.x0, height - extent.y1, extent.x1, height - extent.y0)


def legend(axes, handles: List[Any]) -> List[Box]:
    """*handles* in a dark box at the top left, sized to read at the
    picture's width; answers the place it takes."""
    if not handles:
        return []
    size = max(10.0, axes.figure.bbox.width / 42) * PT
    for handle in handles:
        if hasattr(handle, "set_markersize"):
            handle.set_markersize(size * handle.get_markersize())
    drawn = axes.legend(
        handles=handles,
        loc="upper left",
        fontsize=size,
        labelcolor="white",
        facecolor="black",
        edgecolor="none",
        framealpha=170 / 255,
        handlelength=1.4,
    )
    drawn.get_frame().set_boxstyle("round", pad=0.2)
    return [artist_box(axes.figure, drawn)]


def _segments(x1, y1, x2, y2, pieces: int = 8):
    for i in range(pieces):
        a, b = i / pieces, (i + 1) / pieces
        yield (
            x1 + (x2 - x1) * a,
            y1 + (y2 - y1) * a,
            x1 + (x2 - x1) * b,
            y1 + (y2 - y1) * b,
        )


def draw_marks(
    axes,
    marks: Optional[Dict[str, Any]],
    scale: float,
    placer: LabelPlacer,
    colour: str,
    dashed: bool = False,
) -> None:
    """*marks* (the image's pixels) on *axes* in *colour* (a mark's own
    ``colour`` when it has one: the markers picture's status), their labels
    placed by *placer*: every mark is kept clear first, so no label covers
    one."""
    from matplotlib import patheffects
    from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

    if not marks:
        return
    line = max(2.0, placer.width / 300)  # PNG pixels
    width = line * PT
    halo = [patheffects.withStroke(linewidth=width * 2.4, foreground=_HALO)]
    style = "--" if dashed else "-"
    arm = line * 6 / scale  # a point's reach, in the image's pixels
    s = scale

    def plot(xs, ys, c):
        axes.plot(
            xs,
            ys,
            color=c,
            linewidth=width,
            linestyle=style,
            solid_capstyle="round",
            path_effects=halo,
        )

    points = marks.get("points") or []
    areas = marks.get("areas") or []
    circles = marks.get("circles") or []
    lines = marks.get("lines") or []
    for point in points:
        x, y = float(point["x"]), float(point["y"])
        placer.keep_clear_source(x - arm, y - arm, x + arm, y + arm)
    for area in areas:
        x, y = float(area["x"]), float(area["y"])
        w, h = float(area["width"]), float(area["height"])
        for edge in (
            (x, y, x + w, y),
            (x, y + h, x + w, y + h),
            (x, y, x, y + h),
            (x + w, y, x + w, y + h),
        ):
            placer.keep_clear_source(*edge, margin=line)
    for circle in circles:
        x, y, r = float(circle["x"]), float(circle["y"]), float(circle["radius"])
        placer.keep_clear_source(x - r, y - r, x + r, y + r)
    for mark in lines:
        for piece in _segments(mark["x1"], mark["y1"], mark["x2"], mark["y2"]):
            placer.keep_clear_source(*piece, margin=line)

    def of(mark) -> str:
        return mark.get("colour") or colour

    labelled: List[Tuple[Box, str, str]] = []
    for point in points:
        x, y = float(point["x"]), float(point["y"])
        c = of(point)
        if point.get("shape") == "target":
            axes.add_patch(
                Circle(
                    (x, y),
                    arm * 0.6,
                    fill=False,
                    ec=c,
                    lw=width,
                    linestyle=style,
                    path_effects=halo,
                )
            )
            gap = arm * 0.25
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                plot([x + dx * gap, x + dx * arm], [y + dy * gap, y + dy * arm], c)
        else:
            plot([x - arm, x + arm], [y, y], c)
            plot([x, x], [y - arm, y + arm], c)
        anchor = ((x - arm) * s, (y - arm) * s, (x + arm) * s, (y + arm) * s)
        labelled.append((anchor, point.get("label") or "", c))
    for area in areas:
        x, y = float(area["x"]), float(area["y"])
        c = of(area)
        w, h = float(area["width"]), float(area["height"])
        axes.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                fill=False,
                edgecolor=c,
                linewidth=width,
                linestyle=style,
                path_effects=halo,
            )
        )
        labelled.append(
            ((x * s, y * s, (x + w) * s, (y + h) * s), area.get("label") or "", c)
        )
    for circle in circles:
        x, y, r = float(circle["x"]), float(circle["y"]), float(circle["radius"])
        c = of(circle)
        axes.add_patch(
            Circle(
                (x, y),
                r,
                fill=False,
                ec=c,
                lw=width,
                linestyle=style,
                path_effects=halo,
            )
        )
        labelled.append(
            (
                ((x - r) * s, (y - r) * s, (x + r) * s, (y + r) * s),
                circle.get("label") or "",
                c,
            )
        )
    for mark in lines:
        x1, y1, x2, y2 = (float(mark[k]) for k in ("x1", "y1", "x2", "y2"))
        c = of(mark)
        if mark.get("arrow"):
            axes.add_patch(
                FancyArrowPatch(
                    (x1, y1),
                    (x2, y2),
                    arrowstyle=f"-|>,head_length={line * 4},head_width={line * 2}",
                    mutation_scale=1,
                    color=c,
                    linewidth=width,
                    linestyle=style,
                    path_effects=halo,
                )
            )
        else:
            plot([x1, x2], [y1, y2], c)
        if mark.get("measure"):
            # The ruler's end ticks, square to the line.
            length = math.hypot(x2 - x1, y2 - y1) or 1.0
            nx, ny = -(y2 - y1) / length * arm * 0.6, (x2 - x1) / length * arm * 0.6
            for ex, ey in ((x1, y1), (x2, y2)):
                plot([ex - nx, ex + nx], [ey - ny, ey + ny], c)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        else:
            cx, cy = x1, y1  # at the tail: the head stays clear
        anchor = ((cx * s) - line, (cy * s) - line, (cx * s) + line, (cy * s) + line)
        labelled.append((anchor, mark.get("label") or "", c))
    for anchor, label, c in labelled:
        placer.place(anchor, label, c)


def plane_shown(
    marks: Optional[Dict[str, Any]], plane: Optional[int] = None, stack: bool = True
) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
    """``(plane, marks)``: the plane of a *stack* a picture of *marks* shows
    (*plane* when given, else the one every mark with a z is on, else None,
    the max projection) and the marks it draws: on a plane, that plane's and
    those with no z; on the projection or an image with no planes, all, each
    label saying its plane."""
    if plane is None and stack:
        planes = {_plane(m) for kind in (marks or {}).values() for m in kind}
        planes.discard(None)
        plane = planes.pop() if len(planes) == 1 else None
    if not marks:
        return plane, marks
    shown = {}
    for name, kind in marks.items():
        if plane is not None:
            kind = [m for m in kind if _plane(m) in (None, plane)]
        else:
            kind = [_with_plane(m) if "z" in m else m for m in kind]
        if kind:
            shown[name] = kind
    return plane, shown or None


def _plane(mark: Dict[str, Any]) -> Optional[int]:
    """The plane *mark* is drawn on: its z, a fitted fraction to the nearest."""
    return round(mark["z"]) if "z" in mark else None


def _with_plane(mark: Dict[str, Any]) -> Dict[str, Any]:
    """*mark* labelled with its plane, counted from 1 as the viewer counts."""
    where = f"plane {_plane(mark) + 1}"
    label = mark.get("label")
    return {**mark, "label": f"{label} · {where}" if label else where}


def load_marked_image(
    path: str, channels: Optional[Sequence[str]] = None, plane: Optional[int] = None
):
    """``(image, options)``: the image at *path* as the export shows it, a beam
    image or a fluorescence stack's *plane* (0 the first; None the max
    projection), only *channels* (by name) when given, with its scale bar and
    metadata bar, or any other image as it is; ``(None, None)`` when it does
    not read. ValueError for a plane the stack doesn't have."""
    from PIL import Image

    from fibsem.imaging.export import load_export_image, restacked

    if path.lower().endswith((".tif", ".tiff")):
        try:
            image = load_export_image(path)
        except Exception:  # noqa: BLE001 - try it as a plain image
            image = None
        if image is not None:
            options = options_for(image, channels=channels)
            if plane is not None:
                if not image.layers:
                    raise ValueError(f"plane is for a fluorescence stack: {path}")
                options.plane = plane
                image = restacked(image, options)
            return image, options
    try:
        with Image.open(path) as opened:
            rgb = np.asarray(opened.convert("RGB"))
    except Exception:  # noqa: BLE001 - nothing to draw on
        return None, None
    image = ExportImage(rgb=rgb, pixel_size=None, kind="Image")
    return image, ExportOptions(scalebar=False)


def options_for(
    image: ExportImage,
    crosshair: bool = False,
    metadata_bar: bool = True,
    channels: Optional[Sequence[str]] = None,
) -> ExportOptions:
    """The export dialog's defaults for *image*, with the centre *crosshair*,
    the metadata bar when *metadata_bar*, and only *channels* (by name) of a
    fluorescence image shown."""
    options = default_options(image)
    options.crosshair = crosshair
    if not metadata_bar:
        options.fields, options.provenance = (), ()
    if channels:
        names = set(channels)
        options.hidden_channels = [
            i for i, channel in enumerate(image.channels) if channel.name not in names
        ]
    return options


def finish(
    rgb: np.ndarray, image: ExportImage, options: ExportOptions, source_width: int
) -> np.ndarray:
    """*rgb* (*image* drawn on, maybe scaled down from *source_width*) with the
    export's overlays and metadata bar, as *options* ask."""
    shown = copy.copy(image)
    if shown.pixel_size:
        shown.pixel_size = shown.pixel_size * source_width / rgb.shape[1]
    shown.rgb, shown.gray, shown.layers = rgb, None, []
    return render_export(shown, options)


def blocked(
    image: ExportImage, options: ExportOptions, width: int, height: int
) -> List[Box]:
    """What `finish` draws over a *width* x *height* picture of *image*."""
    shown = copy.copy(image)
    if shown.pixel_size:
        shown.pixel_size = shown.pixel_size * image.rgb.shape[1] / width
    return [
        tuple(map(float, box)) for box in overlay_boxes(shown, options, width, height)
    ]


def marked_picture(
    image: ExportImage,
    marks: Optional[Dict[str, Any]],
    colour: str,
    options: ExportOptions,
    max_width: int,
    decorate: Optional[Callable[[Any, Any, float], List[Box]]] = None,
) -> np.ndarray:
    """*image* as *options* show it, at most *max_width* wide, *marks* drawn
    on it in *colour*; *decorate(figure, axes, scale)* draws more first and
    answers the places it took. The overlays and metadata bar under it."""
    data = _display_rgb(image, options)
    figure, axes, scale = canvas(data, max_width)
    width, height = int(figure.bbox.width), int(figure.bbox.height)
    taken = blocked(image, options, width, height)
    if decorate is not None:
        taken += decorate(figure, axes, scale)
    placer = LabelPlacer(axes, scale, taken)
    draw_marks(axes, marks, scale, placer, colour)
    return finish(rgb_of(figure), image, options, data.shape[1])


def scaled_marks(
    marks: Optional[Dict[str, Any]],
    fx: float,
    fy: float,
    dx: float = 0.0,
    dy: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """*marks* moved to another image's pixels: each x to ``x * fx + dx``,
    each y to ``y * fy + dy``."""
    if not marks:
        return marks
    scaled = copy.deepcopy(marks)

    def x(value):
        return value * fx + dx

    def y(value):
        return value * fy + dy

    for point in scaled.get("points", []):
        point["x"], point["y"] = x(point["x"]), y(point["y"])
    for area in scaled.get("areas", []):
        area["x"], area["width"] = x(area["x"]), area["width"] * fx
        area["y"], area["height"] = y(area["y"]), area["height"] * fy
    for circle in scaled.get("circles", []):
        circle["x"], circle["y"] = x(circle["x"]), y(circle["y"])
        circle["radius"] = circle["radius"] * (fx + fy) / 2
    for line in scaled.get("lines", []):
        line["x1"], line["x2"] = x(line["x1"]), x(line["x2"])
        line["y1"], line["y2"] = y(line["y1"]), y(line["y2"])
    return scaled


# A pair's gap between its two pictures, in pixels.
PAIR_GAP = 8


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """*left* and *right* (RGB) in one picture, tops aligned, on black."""
    height = max(left.shape[0], right.shape[0])
    width = left.shape[1] + PAIR_GAP + right.shape[1]
    out = np.zeros((height, width, 3), dtype=np.uint8)
    out[: left.shape[0], : left.shape[1]] = left
    out[: right.shape[0], left.shape[1] + PAIR_GAP :] = right
    return out


def image_stem(path: str) -> str:
    """A file's name without its extensions (``.ome.tiff`` whole)."""
    import os

    name = os.path.basename(path)
    for suffix in (".ome.tiff", ".ome.tif"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return os.path.splitext(name)[0]


def drawn_path(beside: str, name: str) -> str:
    """Where a picture drawn on the image *beside* is kept: ``drawn/<name>.png``
    in its folder."""
    import os

    return os.path.join(os.path.dirname(beside), "drawn", f"{name}.png")


def save_drawn(rgb: np.ndarray, path: str) -> Dict[str, str]:
    """Save the drawn picture *rgb* as the PNG *path*; answers its preview."""
    import base64
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(png_bytes(rgb))
    return {"image_b64_jpeg": base64.b64encode(jpeg_bytes(rgb, 1536)).decode()}


# The widest a drawn picture is saved, each half of a pair too.
DRAWN_WIDTH = 1024


def drawn_picture(
    path: str,
    marks: Optional[Dict[str, Any]],
    colour: str = AGENT_MARK_COLOR,
    crosshair: bool = False,
    metadata_bar: bool = True,
    channels: Optional[Sequence[str]] = None,
    max_width: int = DRAWN_WIDTH,
    plane: Optional[int] = None,
) -> np.ndarray:
    """The image at *path* with *marks* (its pixels) drawn on it, at most
    *max_width* wide: the centre *crosshair*, the scale bar and, when
    *metadata_bar*, the metadata bar, as the export dialog draws them. Raises
    ValueError for a file that is not an image. On a fluorescence stack, the
    *plane* `plane_shown` picks for the marks."""
    plane, marks = plane_shown(marks, plane, is_fluorescence_image(path))
    image, options = load_marked_image(path, channels, plane)
    if image is None:
        raise ValueError(f"{path} is not an image")
    options.crosshair = crosshair
    if not metadata_bar:
        options.fields, options.provenance = (), ()
    return marked_picture(image, marks, colour, options, max_width)


def _field_width(path: str, channels) -> Tuple[Optional[float], Tuple[int, int]]:
    """The image's field width (metres; None unknown) and ``(height, width)``."""
    image, _ = load_marked_image(path, channels)
    if image is None:
        raise ValueError(f"{path} is not an image")
    height, width = image.rgb.shape[:2]
    return (image.pixel_size * width if image.pixel_size else None), (height, width)


def drawn_pair(
    before: str,
    after: str,
    marks: Optional[Dict[str, Any]],
    colour: str = AGENT_MARK_COLOR,
    metadata_bar: bool = True,
    channels: Optional[Sequence[str]] = None,
    before_plane: Optional[int] = None,
    after_plane: Optional[int] = None,
) -> np.ndarray:
    """*before* and *after* side by side at one scale (the wider field at
    most `DRAWN_WIDTH` wide), each with its own bars, *marks* (*before*'s pixels)
    on both: on *after* at the same place on the sample, the two taken
    centred on one point (at the same place of its frame when either does not
    say its pixel size). A stack shows its *before_plane* / *after_plane* (as
    `drawn_picture` picks when None): two planes of one stack side by side."""
    field_before, shape_before = _field_width(before, channels)
    field_after, shape_after = _field_width(after, channels)
    widths = [DRAWN_WIDTH, DRAWN_WIDTH]
    if field_before and field_after:
        fields = (field_before, field_after)
        # Pixels per metre: the wider field at most DRAWN_WIDTH, and neither
        # image drawn larger than it is.
        per_metre = min(
            DRAWN_WIDTH / max(fields),
            shape_before[1] / field_before,
            shape_after[1] / field_after,
        )
        widths = [max(1, round(per_metre * f)) for f in fields]
    if field_before and field_after:
        # The same place on the sample, both images centred on one point.
        f = (field_before / shape_before[1]) / (field_after / shape_after[1])
        moved = scaled_marks(
            marks,
            f,
            f,
            (shape_after[1] - shape_before[1] * f) / 2,
            (shape_after[0] - shape_before[0] * f) / 2,
        )
    else:
        moved = scaled_marks(
            marks,
            shape_after[1] / shape_before[1],
            shape_after[0] / shape_before[0],
        )
    left = drawn_picture(
        before, marks, colour, False, metadata_bar, channels, widths[0], before_plane
    )
    right = drawn_picture(
        after, moved, colour, False, metadata_bar, channels, widths[1], after_plane
    )
    return side_by_side(left, right)
