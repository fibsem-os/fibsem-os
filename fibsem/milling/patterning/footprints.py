"""Milling patterns drawn on an overview of the grid: each pattern's outline,
defined in the FIB image frame at the milling pose, carried onto the
overview through the stage position under each corner, and the gaps between
neighbouring items' outlines on the sample surface.

The conversions are the existing ones: a FIB-frame point to a stage position
through `fibsem.geometry.projection.plane_point_to_stage` (how a click on a
saved image resolves, the stable move's geometry), and a stage position to an
overview pixel through `stage_to_pixel` (its inverse, which agrees with the
markers' `reproject_stage_positions_onto_image2`). A point is taken to lie on
the sample surface through the FIB image's centre: a flat sample.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from fibsem.fm.preview import is_fluorescence_image
from fibsem.fm.structures import FluorescenceImage
from fibsem.geometry.projection import (
    BeamStageProjection,
    image_stage_geometry,
    plane_point_to_stage,
    stage_to_pixel,
    surface_foreshortening,
)
from fibsem.imaging.looks import cropped, looks_for
from fibsem.imaging.marks import PT, legend, load_marked_image, marked_picture
from fibsem.milling.patterning.shapes import rotated_box
from fibsem.structures.image import FibsemImage
from fibsem.structures.settings import (
    BeamType,
    FibsemBitmapSettings,
    FibsemCircleSettings,
    FibsemLineSettings,
    FibsemPolygonSettings,
    FibsemRectangleSettings,
    FibsemStagePosition,
)
from fibsem.theme.tokens import AGENT_MARK_COLOR

# Each kind of footprint's colour on the overview.
KIND_COLOURS = {
    "trench": "yellow",
    "stress relief": "magenta",
    "fiducial": "lime",
    "polishing": "cyan",
}
OTHER_COLOUR = "orange"
# The least margin round the items in a picture: about a grid square, so the
# bars round a site are in it.
MARGIN = 100e-6
_CIRCLE_CORNERS = 24
# A crop narrower than this is enlarged (whole pixels) to be read.
_READABLE_WIDTH = 768
_MOST_ENLARGED = 8


def stage_kind(stage: Any, task_name: str = "") -> str:
    """What *stage* mills, in a word or two: trench, stress relief, fiducial,
    polishing (a trench in a polishing task), else its pattern's name."""
    pattern = str(getattr(getattr(stage, "pattern", None), "name", "") or "")
    if pattern == "Fiducial":
        return "fiducial"
    if pattern == "MicroExpansion":
        return "stress relief"
    if "trench" in pattern.lower():
        polishing = "polish" in f"{task_name} {stage.name}".lower()
        return "polishing" if polishing else "trench"
    return pattern.lower() or "pattern"


def _box_corners(shape) -> np.ndarray:
    # Microscope coordinates run +y up, so the drawing's rotation (by
    # -rotation in the image's +y-down pixels) is +rotation here.
    return rotated_box(
        shape.centre_x, shape.centre_y, shape.width, shape.height, shape.rotation or 0.0
    )


def shape_outline(shape) -> Optional[np.ndarray]:
    """One milled shape's outline, ``(n, 2)`` metres from the FIB image
    centre, +y up (the patterns' own frame); two rows for a line. None for an
    exclusion area or a shape with no outline."""
    if getattr(shape, "is_exclusion", False):
        return None
    if isinstance(shape, (FibsemRectangleSettings, FibsemBitmapSettings)):
        return _box_corners(shape)
    if isinstance(shape, FibsemCircleSettings):
        angles = np.linspace(0, 2 * np.pi, _CIRCLE_CORNERS, endpoint=False)
        return np.stack(
            [
                shape.centre_x + shape.radius * np.cos(angles),
                shape.centre_y + shape.radius * np.sin(angles),
            ],
            axis=1,
        )
    if isinstance(shape, FibsemLineSettings):
        return np.array([[shape.start_x, shape.start_y], [shape.end_x, shape.end_y]])
    if isinstance(shape, FibsemPolygonSettings) and len(shape.vertices) >= 2:
        return np.asarray(shape.vertices, dtype=float)[:, :2]
    return None


def stage_outlines(stage) -> List[np.ndarray]:
    """*stage*'s milled outlines (`shape_outline`)."""
    try:
        shapes = list(stage.define_patterns())
    except Exception:  # noqa: BLE001 - a stage that can't define its patterns
        return []
    return [o for o in (shape_outline(s) for s in shapes) if o is not None]


@dataclass(frozen=True)
class FibFrame:
    """The FIB view the patterns are defined in: its projection, the stage
    position it was at, and where it was read from (an image's path, or
    "milling pose")."""

    projection: Any
    base: FibsemStagePosition
    source: str

    @classmethod
    def from_image(cls, image, source: str) -> Optional["FibFrame"]:
        """The frame an ion image recorded; None when it says too little."""
        geometry = image_stage_geometry(image)
        if geometry is None:
            return None
        projection, _, base = geometry
        if getattr(projection, "beam_type", None) is not BeamType.ION:
            return None
        return cls(projection, base, source)

    @classmethod
    def from_pose(
        cls, position: FibsemStagePosition, geometry, scan_rotation: float
    ) -> "FibFrame":
        """The ion view at a saved pose: the pose's position, with the
        instrument geometry and ion scan rotation recorded elsewhere."""
        projection = BeamStageProjection(geometry, BeamType.ION, scan_rotation)
        return cls(projection, position, "milling pose")

    def to_overview(self, outline: np.ndarray, overview) -> np.ndarray:
        """*outline* (`shape_outline`'s frame) as source pixels of *overview*:
        each corner's stage position on the plane through the frame's centre,
        then where that falls on the overview."""
        pixels = []
        for x, y in outline:
            # The plane runs +y down, as the image's rows do.
            stage = plane_point_to_stage(self.projection, self.base, x, -y)
            pixel = stage_to_pixel(overview, stage)
            if pixel is None:
                raise ValueError("The overview has no stage geometry to draw on.")
            pixels.append(pixel)
        return np.asarray(pixels, dtype=float)


def load_overview(path: str):
    """The saved image at *path*: a fluorescence stack or a beam image."""

    if is_fluorescence_image(path):
        return FluorescenceImage.load(path)

    return FibsemImage.load(path)


def pose_frame(position: FibsemStagePosition, pose, overview) -> Optional[FibFrame]:
    """The ion view at *position* (a milling pose's), with the ion scan
    rotation the pose (a ``MicroscopeState``) or else the *overview*
    recorded, and the overview's instrument geometry; None when neither says."""
    metadata = getattr(overview, "metadata", None)
    geometry = getattr(metadata, "hardware_geometry", None) or getattr(
        metadata, "geometry", None
    )
    rotations = [
        getattr(getattr(state, "ion_beam", None), "scan_rotation", None)
        for state in (pose, getattr(metadata, "microscope_state", None))
    ]
    rotation = next((r for r in rotations if r is not None), None)
    if geometry is None or rotation is None:
        return None
    return FibFrame.from_pose(position, geometry, float(rotation))


@dataclass
class Footprint:
    """One milled shape of an item, on the overview: its kind, its stage and
    its outline in the overview's source pixels."""

    item: str
    kind: str
    stage: str
    pixels: np.ndarray


@dataclass(frozen=True)
class Surface:
    """An overview's pixels as metres along the sample surface, from its
    centre, x right and y down: y divided by how much of a length on the
    surface the view keeps (`surface_foreshortening`)."""

    pixel_size: float
    foreshortening: float
    width: int
    height: int

    @classmethod
    def of(cls, overview) -> "Surface":
        geometry = image_stage_geometry(overview)
        if geometry is None:
            raise ValueError("The overview has no stage geometry to draw on.")
        projection, pixel_size, base = geometry
        height, width = np.asarray(overview.data).shape[-2:]
        return cls(
            pixel_size,
            surface_foreshortening(projection, base) or 1.0,
            int(width),
            int(height),
        )

    def metres(self, pixels: np.ndarray) -> np.ndarray:
        centred = np.asarray(pixels, dtype=float) - [self.width / 2, self.height / 2]
        return centred * [self.pixel_size, self.pixel_size / self.foreshortening]

    def pixels(self, metres: np.ndarray) -> np.ndarray:
        scale = [self.pixel_size, self.pixel_size / self.foreshortening]
        return np.asarray(metres, dtype=float) / scale + [
            self.width / 2,
            self.height / 2,
        ]


# --- gaps between outlines -------------------------------------------------------


def _closest_on_segment(p, a, b) -> np.ndarray:
    ab = b - a
    length = float(ab @ ab)
    t = 0.0 if length == 0 else min(1.0, max(0.0, float((p - a) @ ab) / length))
    return a + t * ab


def _crosses(a, b, c, d) -> bool:
    def side(p, q, r):
        return np.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    return side(a, b, c) != side(a, b, d) and side(c, d, a) != side(c, d, b)


def _inside(point, polygon) -> bool:
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    for (x1, y1), (x2, y2) in zip(polygon, np.roll(polygon, -1, axis=0)):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def _edges(outline: np.ndarray):
    if len(outline) == 2:
        return [(outline[0], outline[1])]
    return list(zip(outline, np.roll(outline, -1, axis=0)))


def gap(a: np.ndarray, b: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """The least distance between outlines *a* and *b* (a polygon, or two
    rows for a line), 0 where they overlap, and the two nearest points."""
    for p in a:
        if _inside(p, b):
            return 0.0, p, p
    for p in b:
        if _inside(p, a):
            return 0.0, p, p
    best = (math.inf, a[0], b[0])
    for p1, p2 in _edges(a):
        for q1, q2 in _edges(b):
            if _crosses(p1, p2, q1, q2):
                return 0.0, p1, p1
            for p, (s1, s2), mine in (
                (p1, (q1, q2), True),
                (p2, (q1, q2), True),
                (q1, (p1, p2), False),
                (q2, (p1, p2), False),
            ):
                q = _closest_on_segment(p, s1, s2)
                distance = float(np.hypot(*(p - q)))
                if distance < best[0]:
                    best = (distance, p, q) if mine else (distance, q, p)
    return best


@dataclass
class Gap:
    """The nearest approach of two items' footprints, on the sample surface."""

    items: Tuple[str, str]
    distance: float  # metres
    kinds: Tuple[str, str]
    points: Tuple[np.ndarray, np.ndarray]  # overview source pixels

    def fact(self) -> Dict[str, Any]:
        return {
            "items": list(self.items),
            "distance_um": round(self.distance * 1e6, 1),
            "between": list(self.kinds),
        }


def item_gaps(footprints: Sequence[Footprint], surface: Surface) -> List[Gap]:
    """Every pair of items' nearest approach, nearest first."""
    by_item: Dict[str, List[Tuple[Footprint, np.ndarray]]] = {}
    for footprint in footprints:
        by_item.setdefault(footprint.item, []).append(
            (footprint, surface.metres(footprint.pixels))
        )
    names = list(by_item)
    gaps = []
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            best = None
            for fa, ma in by_item[first]:
                for fb, mb in by_item[second]:
                    distance, pa, pb = gap(ma, mb)
                    if best is None or distance < best[0]:
                        best = (distance, fa.kind, fb.kind, pa, pb)
            distance, ka, kb, pa, pb = best
            gaps.append(
                Gap(
                    (first, second),
                    distance,
                    (ka, kb),
                    (surface.pixels(pa), surface.pixels(pb)),
                )
            )
    return sorted(gaps, key=lambda g: g.distance)


def item_box(
    footprints: Sequence[Footprint], surface: Surface
) -> Tuple[List[float], List[float]]:
    """The items' footprints' bounding box on the sample surface, metres from
    the overview's centre ``[x_min, y_min, x_max, y_max]``, and in its source
    pixels ``[x, y, width, height]``."""
    pixels = np.concatenate([f.pixels for f in footprints])
    metres = surface.metres(pixels)
    (x0, y0), (x1, y1) = pixels.min(axis=0), pixels.max(axis=0)
    return (
        [float(v) for v in (*metres.min(axis=0), *metres.max(axis=0))],
        [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
    )


def neighbourhood(footprints: Sequence[Footprint], surface: Surface) -> List[float]:
    """The region round *footprints* a picture shows, ``[x, y, width,
    height]`` in source pixels: their box with a margin of half its size, at
    least `MARGIN`, inside the overview."""
    _, (x, y, w, h) = item_box(footprints, surface)
    margin = max(0.5 * max(w, h), MARGIN / surface.pixel_size)
    x0, y0 = max(0.0, x - margin), max(0.0, y - margin)
    x1 = min(float(surface.width), x + w + margin)
    y1 = min(float(surface.height), y + h + margin)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("The items fall outside the overview.")
    return [x0, y0, x1 - x0, y1 - y0]


# --- the picture -------------------------------------------------------------------


def _enlarged(image, factor: int):
    """*image* (`ExportImage`) with each pixel repeated *factor* times each way."""
    if factor <= 1:
        return image

    def grow(data):
        return np.repeat(np.repeat(data, factor, axis=0), factor, axis=1)

    out = copy.copy(image)
    out.rgb = grow(image.rgb)
    out.gray = None if image.gray is None else grow(image.gray)
    out.layers = [
        replace(layer, data=grow(layer.data), _clim_cache=None)
        for layer in image.layers
    ]
    if out.pixel_size:
        out.pixel_size = out.pixel_size / factor
    return out


def overview_picture(
    path: str,
    footprints: Sequence[Footprint],
    box: Sequence[float],
    closest: Optional[Gap] = None,
    labels: Optional[Dict[str, Sequence[float]]] = None,
    max_width: int = 1024,
) -> np.ndarray:
    """The overview at *path*, region *box* (source pixels), with each
    footprint outlined in its kind's colour and a legend, each item's box
    named (*labels*: item -> its box in source pixels), and the *closest*
    gap drawn as a line with its length. RGB with the scale bar and
    metadata bar."""
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Polygon

    image, options = load_marked_image(path)
    if image is None:
        raise ValueError(f"{path} is not an image")
    region = cropped(image, box)
    x0, y0 = max(0, int(round(box[0]))), max(0, int(round(box[1])))
    width = region.rgb.shape[1]
    factor = int(min(_MOST_ENLARGED, max(1, _READABLE_WIDTH // max(1, width))))
    region = _enlarged(region, factor)
    options.looks = looks_for(region, {})

    def shown(points):
        return (np.asarray(points, dtype=float) - [x0, y0]) * factor

    def decorate(figure, axes, scale):
        line = max(1.5, figure.bbox.width / 500)
        kinds = []
        for footprint in footprints:
            colour = KIND_COLOURS.get(footprint.kind, OTHER_COLOUR)
            closed = len(footprint.pixels) > 2
            axes.add_patch(
                Polygon(
                    shown(footprint.pixels),
                    closed=closed,
                    fill=closed,
                    facecolor=to_rgba(colour, 0.25),
                    edgecolor=colour,
                    linewidth=line * PT,
                    zorder=4,
                )
            )
            if footprint.kind not in kinds:
                kinds.append(footprint.kind)
        handles = [
            Patch(
                facecolor=to_rgba(KIND_COLOURS.get(k, OTHER_COLOUR), 0.25),
                edgecolor=KIND_COLOURS.get(k, OTHER_COLOUR),
                linewidth=1.5,
                label=k,
            )
            for k in kinds
        ]
        if closest is not None:
            handles.append(
                Line2D(
                    [],
                    [],
                    color=AGENT_MARK_COLOR,
                    linewidth=1.5,
                    label=f"nearest gap {closest.distance * 1e6:.1f} µm",
                )
            )
        return legend(axes, handles)

    marks: Dict[str, Any] = {"areas": [], "lines": []}
    for name, (bx, by, bw, bh) in (labels or {}).items():
        sx, sy = shown([[bx, by]])[0]
        marks["areas"].append(
            {
                "x": sx,
                "y": sy,
                "width": bw * factor,
                "height": bh * factor,
                "label": name,
            }
        )
    if closest is not None:
        (ax_, ay_), (bx_, by_) = shown(closest.points)
        marks["lines"].append(
            {
                "x1": ax_,
                "y1": ay_,
                "x2": bx_,
                "y2": by_,
                # Its length is in the legend: a label here lands on the cuts.
                "label": "",
            }
        )
    return marked_picture(region, marks, AGENT_MARK_COLOR, options, max_width, decorate)
