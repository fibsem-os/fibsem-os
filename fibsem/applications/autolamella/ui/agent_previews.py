"""What the agent's waiting request or suggestion would do, drawn on the card
where the operator decides (FIB-1404, FIB-1411).

Under Why: the image the values were measured on, every value drawn on it in the
agent's purple. A review's or an item's point of interest on that item's image;
an item edit's alignment area; the patterns a mill would draw, a config edit's as
it would leave them (as they are now in grey) and any reaching past the edge of
the field in red, on the FIB image; where a move would go; a suggested alignment
area, spot-burn points, detected features or point of interest, on the image the
question shows.

Nothing is drawn on the live views: they show what the microscope does, and the
agent's real patterns only once its mill runs (`agent_activity`). The picture
comes down when the request is decided or the suggestion is taken or dropped.

A value on the sample is kept with the frame it was measured on and re-projected
through the stage onto the frame it is drawn on (FIB-1411). Patterns are places
in the beam's field, not on the sample: drawn against the frame's field.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from fibsem.server.request_text import parse_patterns
from fibsem.structures import BeamType, Point
from fibsem.ui.tokens import PURPLE_COLOR, RED_COLOR, WHITE_ICON_COLOR

# What a request places on the sample, by kind.
TARGET_MARK = "target"
POI_MARK = "poi"
AREA_MARK = "area"
ITEM_MARK = "item"
POSE_MARK = "pose"
POSE_NOW_MARK = "pose_now"

# The beam a move tool works in when the agent does not say: as the tools do.
_MOVE_BEAMS = {
    "stage_stable_move": BeamType.ELECTRON,
    "stage_vertical_move": BeamType.ION,
}


def _beam(value: Any, default: BeamType) -> BeamType:
    if value is None:
        return default
    text = str(value).upper()
    if text in ("ION", "2"):
        return BeamType.ION
    if text in ("ELECTRON", "1"):
        return BeamType.ELECTRON
    return default


def _milling_commands(request: Dict[str, Any]):
    """``(command, kwargs)`` for each of the request's milling service commands."""
    for action in request.get("actions", []):
        arguments = action.get("arguments") or {}
        if (
            action.get("tool") == "service_call_command"
            and arguments.get("service") == "milling"
        ):
            yield arguments.get("command"), arguments.get("kwargs") or {}


def request_patterns(request: Dict[str, Any]) -> List:
    """The patterns the request's milling ``prepare`` actions would draw."""
    patterns = []
    for command, kwargs in _milling_commands(request):
        if command == "prepare":
            patterns.extend(parse_patterns(kwargs.get("patterns")))
    return patterns


def request_stages(request: Dict[str, Any]) -> List:
    """The milling stages the request's ``run_stage`` and ``run_task`` actions
    would mill (a task's enabled ones), as the milling view draws a protocol's.
    One that does not parse is left out: the route refuses it."""
    from fibsem.milling.base import FibsemMillingStage
    from fibsem.milling.tasks import FibsemMillingTaskConfig

    stages = []
    for command, kwargs in _milling_commands(request):
        try:
            if command == "run_stage":
                stages.append(FibsemMillingStage.from_dict(dict(kwargs["stage"])))
            elif command == "run_task":
                config = FibsemMillingTaskConfig.from_dict(dict(kwargs["config"]))
                stages.extend(stage for stage in config.stages if stage.enabled)
        except Exception:  # noqa: BLE001 - described, not drawn
            continue
    return stages


def config_edit_stages(
    request: Dict[str, Any], task_config: Callable[[str, str], object]
) -> Tuple[List, List]:
    """``(before, after)``: the milling stages the request's task config edits
    change, as the item's config has them and as the edit would leave them.
    *task_config* gives an item's config for a task (None when there is none);
    the edit is applied to a copy, by the same rules as the route."""
    from copy import deepcopy

    from fibsem.server.config_patch import apply_patch

    before, after = [], []
    for action in request.get("actions", []):
        arguments = action.get("arguments") or {}
        patch = arguments.get("patch") or {}
        if action.get("tool") != "item_update_task_config":
            continue
        tasks = {
            path.split(".")[1]
            for path in patch
            if path.startswith("milling.") and path.count(".") > 1
        }
        if not tasks:
            continue
        config = task_config(arguments.get("item_name"), arguments.get("task_name"))
        if config is None:
            continue
        edited = deepcopy(config)
        try:
            apply_patch(edited, patch)
        except Exception:  # noqa: BLE001 - the route refuses it; nothing to draw
            continue
        for name in sorted(tasks):
            for stages, source in ((before, config), (after, edited)):
                milling = dict(getattr(source, "milling", None) or {}).get(name)
                stages.extend(s for s in getattr(milling, "stages", []) if s.enabled)
    return before, after


def request_outside_field(
    request: Dict[str, Any], aspect: float = 2 / 3, stages: Optional[List] = None
) -> List:
    """The patterns the request would mill past the edge of the field: its
    milling hfw wide and *aspect* of that high (an image's height over width).
    The Demo mills them; a real instrument refuses (FIB-1411). *stages* in
    place of the request's own run_stage and run_task stages."""
    from fibsem.milling.base import FibsemMillingStage
    from fibsem.milling.patterning.utils import outside_field
    from fibsem.structures import FibsemMillingSettings

    fields = []  # (hfw, patterns)
    for command, kwargs in _milling_commands(request):
        if command != "prepare":
            continue
        try:
            hfw = FibsemMillingSettings.from_dict(dict(kwargs["settings"])).hfw
        except Exception:  # noqa: BLE001 - no field to check against
            continue
        fields.append((hfw, parse_patterns(kwargs.get("patterns"))))
    for stage in request_stages(request) if stages is None else stages:
        if isinstance(stage, FibsemMillingStage):
            try:
                fields.append((stage.milling.hfw, stage.define_patterns()))
            except Exception:  # noqa: BLE001 - a stage that cannot define them
                continue
    return [
        pattern
        for hfw, patterns in fields
        for pattern in patterns
        if outside_field(pattern, hfw, hfw * aspect)
    ]


def request_targets(request: Dict[str, Any]) -> List[Tuple[BeamType, Point]]:
    """Where the request's stable and vertical moves would bring to the centre:
    the beam and the offset from the image centre, x right and y up, in metres."""
    targets = []
    for action in request.get("actions", []):
        tool = action.get("tool")
        if tool not in _MOVE_BEAMS:
            continue
        arguments = action.get("arguments") or {}
        try:
            dx, dy = (
                float(arguments.get("dx") or 0.0),
                float(arguments.get("dy") or 0.0),
            )
        except (TypeError, ValueError):
            continue
        beam = _beam(arguments.get("beam_type"), _MOVE_BEAMS[tool])
        targets.append((beam, Point(x=dx, y=dy)))
    return targets


def _frame(controller, beam: BeamType):
    """The frame *beam*'s view shows, when it has a pixel size; else None."""
    image = controller.image(beam) if hasattr(controller, "image") else None
    metadata = getattr(image, "metadata", None)
    if image is None or metadata is None or metadata.pixel_size is None:
        return None
    return image


def _offset_pixel(frame, offset: Point) -> Tuple[float, float]:
    """An offset from *frame*'s centre (metres, x right, y up) as its pixel."""
    from fibsem import conversions

    pixel = conversions.microscope_image_to_image_coordinates(
        offset, frame.data.shape, frame.metadata.pixel_size.x
    )
    return (pixel.x, pixel.y)


def _overview_pixel(image, overview: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """An item_add's point on its overview (``px``, or ``fraction`` of it) as
    the image's pixel; None for no image or no point."""
    if not _has_pixel_size(image) or getattr(image.data, "ndim", 0) != 2:
        return None
    height, width = image.data.shape
    try:
        if overview.get("px") is not None:
            x, y = overview["px"]
            return (float(x), float(y))
        fx, fy = overview["fraction"]
        return (float(fx) * width, float(fy) * height)
    except (KeyError, TypeError, ValueError):
        return None


def _stage_position(value: Dict[str, Any]):
    """``{x, y, z, r, t}`` as the stage tools give it, as a stage position."""
    from fibsem.structures import FibsemStagePosition

    try:
        return FibsemStagePosition.from_dict({"coordinate_system": "RAW", **value})
    except Exception:  # noqa: BLE001 - described, not drawn
        return None


def _load_beam_image(path: str):
    from fibsem.structures import FibsemImage

    return FibsemImage.load(path)


def _agent_image(load: Callable[[str], object], path: Any):
    """The image a request names as its evidence, loaded; None for none, or one
    that is not a beam image with a pixel size."""
    if not path:
        return None
    try:
        image = load(str(path))
    except Exception:  # noqa: BLE001 - shown without it
        return None
    if not _has_pixel_size(image) or getattr(image.data, "ndim", 0) != 2:
        return None
    return image


def _has_pixel_size(image) -> bool:
    size = getattr(getattr(image, "metadata", None), "pixel_size", None)
    return image is not None and size is not None


def _point(value: Any) -> Optional[Point]:
    """``{x, y}`` (metres) as a Point; None when it is not one."""
    try:
        return Point(x=float(value["x"]), y=float(value["y"]))
    except (KeyError, TypeError, ValueError):
        return None


def _area(patch: Dict[str, Any]) -> Optional[List[Tuple[float, float]]]:
    """An item edit's alignment area as its two corners, fractions of the frame
    (origin top-left); None when the edit sets no complete area."""
    try:
        left, top = (
            float(patch["alignment_area.left"]),
            float(patch["alignment_area.top"]),
        )
        width = float(patch["alignment_area.width"])
        height = float(patch["alignment_area.height"])
    except (KeyError, TypeError, ValueError):
        return None
    return [(left, top), (left + width, top + height)]


def _fraction_pixel(frame, fraction: Tuple[float, float]) -> Tuple[float, float]:
    """A fraction of *frame* (origin top-left) as its pixel."""
    height, width = frame.data.shape[-2:]
    return (fraction[0] * width, fraction[1] * height)


# How a mark draws its places: a point of interest or a feature "+", a move
# target a crosshair, a spot burn point a dot.
CROSS, TARGET, DOT = "cross", "target", "dot"
# How the legend shows the rest: patterns to mill, filled; patterns outlined (a
# config edit's as they are now, those past the field); an area, dashed.
FILL, OUTLINE, AREA = "fill", "outline", "area"
# The patterns to mill: see-through, so where they overlap shows darker
# (FIB-1424; Patrick: "solid rectangles", not bounding boxes).
FILL_ALPHA = 70


class _Mark:
    """Places on the sample one overlay draws: points (*shape*), or the two
    corners of an area. Each place is kept with the frame it was measured on, or
    with how to measure it on the first frame the view shows
    (``(None, on_frame)``)."""

    def __init__(
        self,
        beam: BeamType,
        legend: str,
        places,
        area: bool = False,
        shape: str = CROSS,
        before: bool = False,
        title: str = "",
    ):
        self.beam = beam
        self.legend = legend
        self.places: List[Tuple[Optional[object], Any]] = list(places)
        self.area = area
        self.shape = shape
        # Where something is now, drawn as a config edit's patterns as they are.
        self.before = before
        # Whose image the places were measured on ("05-pale-lark"), for the
        # picture's caption.
        self.title = title

    def pixels(self, frame) -> List[Tuple[float, float]]:
        """The places as *frame*'s pixels; one the frame cannot place (None) is
        left out."""
        from fibsem.geometry.projection import reproject_pixel

        measured = []
        for measured_on, value in self.places:
            if measured_on is None:
                measured_on, value = frame, value(frame)
            if value is not None:
                measured.append((measured_on, value))
        self.places = measured
        return [reproject_pixel(on, frame, *value) for on, value in measured]


# The picture on the card: its width at most, and how its marks are drawn.
PICTURE_WIDTH = 1024
# A config edit's patterns as they are now: solid white, seen on any image.
_BEFORE = WHITE_ICON_COLOR


class _Layer:
    """What one colour draws on the card's picture: crosses, crosshairs, dots
    and areas (pixels of the picture's frame), and patterns (metres from the
    field's centre), filled (*fill*, what would be milled) or outlined.
    Areas, which are never milled, are dashed."""

    def __init__(self, colour: str, fill: bool = False) -> None:
        self.colour = colour
        self.fill = fill
        self.crosses: List[Tuple[float, float]] = []
        self.targets: List[Tuple[float, float]] = []
        self.dots: List[Tuple[float, float]] = []
        self.areas: List[List[Tuple[float, float]]] = []
        self.patterns: List = []
        # What it draws, for the picture's legend: (shape, words), shape
        # FILL, OUTLINE, AREA or a mark's.
        self.keys: List[Tuple[str, str]] = []

    @property
    def pattern_shape(self) -> str:
        """How the legend shows this layer's patterns."""
        return FILL if self.fill else OUTLINE

    def key(self, shape: str, text: str, drawn=True) -> None:
        """Name what this colour draws in the picture's legend, when it draws it."""
        if drawn and (shape, text) not in self.keys:
            self.keys.append((shape, text))

    def __bool__(self) -> bool:
        return bool(
            self.crosses or self.targets or self.dots or self.areas or self.patterns
        )

    def points(self, shape: str) -> List[Tuple[float, float]]:
        return {CROSS: self.crosses, TARGET: self.targets, DOT: self.dots}[shape]


def picture_caption(frame, beam: BeamType) -> str:
    """What the picture is: its beam, and when it was taken ("FIB · 07:41")."""
    from datetime import datetime

    word = "SEM" if beam is BeamType.ELECTRON else "FIB"
    state = getattr(frame.metadata, "microscope_state", None)
    stamp = getattr(state, "timestamp", None)
    try:
        when = datetime.fromtimestamp(float(stamp)).strftime("%H:%M")
    except (TypeError, ValueError, OSError, OverflowError):
        return word
    return f"{word} · {when}"


def draw_picture(frame, layers: List[_Layer]):
    """*frame* with *layers* drawn on it, as the card shows it: a QImage at most
    `PICTURE_WIDTH` wide (FIB-1411). Patterns are drawn as the milling view
    draws them, against this frame's field."""
    from PyQt5.QtCore import QLineF, QPointF, Qt
    from PyQt5.QtGui import QColor, QPainter, QPen, QPolygonF

    from fibsem.applications.autolamella.ui.lamella_card_widget import (
        _arr_to_qimage,
    )
    from fibsem.imaging.export import _to_uint8
    from fibsem.milling.patterning.utils import pattern_outline

    height, width = frame.data.shape[:2]
    image = _arr_to_qimage(_to_uint8(frame.data))
    scale = min(1.0, PICTURE_WIDTH / width)
    if scale < 1.0:
        image = image.scaledToWidth(
            int(width * scale),
            Qt.SmoothTransformation,  # type: ignore
        )
    image = image.convertToFormat(image.Format_RGB32)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing)
    # Seen at the Why column's width, a third of this or less.
    line = max(2.0, image.width() / 150)

    def at(pixel) -> QPointF:
        return QPointF(pixel[0] * scale, pixel[1] * scale)

    for layer in layers:
        colour = QColor(layer.colour)
        pen = QPen(colour, line)
        painter.setPen(pen)
        fill = QColor(colour)
        fill.setAlpha(FILL_ALPHA)
        painter.setBrush(fill if layer.fill else Qt.NoBrush)  # type: ignore
        for pattern in layer.patterns:
            for ring in pattern_outline(pattern):
                points = [at(_offset_pixel(frame, Point(x=x, y=y))) for x, y in ring]
                if len(points) == 2:
                    painter.drawLine(QLineF(points[0], points[1]))
                else:
                    painter.drawPolygon(QPolygonF(points))
        painter.setBrush(Qt.NoBrush)  # type: ignore
        pen.setStyle(Qt.DashLine)  # type: ignore
        painter.setPen(pen)
        for (x0, y0), (x1, y1) in layer.areas:
            painter.drawPolygon(
                QPolygonF([at((x0, y0)), at((x1, y0)), at((x1, y1)), at((x0, y1))])
            )
        pen.setStyle(Qt.SolidLine)  # type: ignore
        pen.setWidthF(line * 1.5)
        painter.setPen(pen)
        for shape in (CROSS, TARGET, DOT):
            for pixel in layer.points(shape):
                _draw_mark(painter, shape, at(pixel), 6 * line)
    painter.end()
    return image


def _draw_mark(painter, shape: str, centre, arm: float) -> None:
    """One point, *arm* its reach, in the painter's pen: "+", a crosshair (a
    ring, its lines stopping short of the centre), or a dot."""
    from PyQt5.QtCore import QLineF, QPointF, Qt

    x, y = centre.x(), centre.y()
    if shape == DOT:
        painter.setBrush(painter.pen().color())
        painter.drawEllipse(centre, arm * 0.35, arm * 0.35)
        painter.setBrush(Qt.NoBrush)  # type: ignore
        return
    if shape == TARGET:
        painter.drawEllipse(QPointF(x, y), arm * 0.6, arm * 0.6)
        gap = arm * 0.25
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            painter.drawLine(
                QLineF(x + dx * gap, y + dy * gap, x + dx * arm, y + dy * arm)
            )
        return
    painter.drawLine(QLineF(x - arm, y, x + arm, y))
    painter.drawLine(QLineF(x, y - arm, x, y + arm))


def legend_entries(layers: List[_Layer]) -> List[Tuple[str, str, str]]:
    """What each layer draws, for the picture's legend: ``(colour, shape,
    words)``."""
    return [
        (layer.colour, shape, text) for layer in layers for shape, text in layer.keys
    ]


def draw_legend(painter, entries, font_px: int) -> None:
    """*entries* (`legend_entries`) in a box at the top left of what *painter*
    draws on: a sample of each mark, then its words, as the canvas legend does.
    *font_px* the words' height, in the painter's pixels."""
    from PyQt5.QtCore import QPointF, QRectF, Qt
    from PyQt5.QtGui import QColor, QFont, QFontMetricsF, QPen

    if not entries:
        return
    font = QFont()
    font.setPixelSize(font_px)
    painter.setFont(font)
    metrics = QFontMetricsF(font)
    row = metrics.height() * 1.3
    sample = metrics.height() * 1.2
    pad = metrics.height() * 0.5
    box = QRectF(
        pad,
        pad,
        pad * 3 + sample + max(metrics.horizontalAdvance(e[-1]) for e in entries),
        pad * 2 + row * len(entries),
    )
    painter.setPen(Qt.NoPen)  # type: ignore
    painter.setBrush(QColor(0, 0, 0, 170))
    painter.drawRoundedRect(box, pad / 2, pad / 2)
    painter.setBrush(Qt.NoBrush)  # type: ignore
    for index, (colour, shape, text) in enumerate(entries):
        top = box.top() + pad + row * index
        middle = top + row / 2
        left = box.left() + pad
        pen = QPen(QColor(colour), 1.5)
        if shape == AREA:
            pen.setStyle(Qt.DashLine)  # type: ignore
        painter.setPen(pen)
        if shape in (FILL, OUTLINE, AREA):
            fill = QColor(colour)
            fill.setAlpha(FILL_ALPHA)
            painter.setBrush(fill if shape == FILL else Qt.NoBrush)  # type: ignore
            painter.drawRect(QRectF(left, middle - sample * 0.3, sample, sample * 0.6))
            painter.setBrush(Qt.NoBrush)  # type: ignore
        else:
            _draw_mark(
                painter, shape, QPointF(left + sample / 2, middle), sample * 0.45
            )
        painter.setPen(QColor("#ffffff"))
        painter.drawText(
            QRectF(left + sample + pad, top, box.width(), row),
            int(Qt.AlignVCenter | Qt.AlignLeft),  # type: ignore
            text,
        )


def _stage_patterns(stages) -> List:
    patterns = []
    for stage in stages:
        try:
            patterns.extend(stage.define_patterns())
        except Exception:  # noqa: BLE001 - a stage that cannot define them
            continue
    return patterns


class AgentPreviews:
    """Draws the agent's waiting request and suggestion on the card, on the
    image their values were measured on. Nothing on the views."""

    def __init__(
        self,
        view_controller: Callable[[], object],
        reference_image: Optional[Callable[[Dict[str, Any]], object]] = None,
        task_config: Optional[Callable[[str, str], object]] = None,
        show_picture: Optional[Callable[..., None]] = None,
        item_pose: Optional[Callable[[str, str], object]] = None,
        item_image: Optional[Callable[[str], object]] = None,
        overview_image: Optional[Callable[[str, str], object]] = None,
        load_image: Optional[Callable[[str], object]] = None,
    ) -> None:
        # Asked when there is something to draw: the views are built after this.
        self._view_controller = view_controller
        # The image a review_decide action's values sit on (its arguments), or
        # None: the window reads it from the experiment.
        self._reference_image = reference_image or (lambda arguments: None)
        # An item's task config (item name, task name), for a config edit's
        # patterns; None when there is none.
        self._task_config = task_config or (lambda item, task: None)
        # Where an item's pose puts the stage (item name, pose name): a
        # FibsemStagePosition, or None.
        self._item_pose = item_pose or (lambda item, pose: None)
        # An item's last FIB image, which its point of interest and alignment
        # area are in (item name): a FibsemImage, or None.
        self._item_image = item_image or (lambda item: None)
        # A grid's recorded overview (grid name, file name), for an item added
        # on it; None when there is none.
        self._overview_image = overview_image or (lambda grid, filename: None)
        # A request's own image, by path (the agent's evidence).
        self._load_image = load_image or _load_beam_image
        # The card's picture: (QImage or None, caption, suggestion=bool,
        # legend=legend_entries).
        self._show_picture = show_picture or (lambda *args, **kwargs: None)
        self._request_id: Optional[str] = None
        # The card's pictures, as drawn: (frame, layers), for tests and reading.
        self.request_picture: Optional[Tuple[object, List[_Layer]]] = None
        self.suggestion_picture: Optional[Tuple[object, List[_Layer]]] = None

    # --- requests ---------------------------------------------------------------

    def show_request(self, request: Optional[Dict[str, Any]]) -> None:
        """Draw what *request* would do while it waits; anything else (None, or a
        request being applied or ended) takes the previews down."""
        waiting = request is not None and request.get("status") == "pending"
        request_id = request.get("request_id") if waiting else None
        if request_id == self._request_id:
            return
        self._request_id = request_id
        self._picture(None, None, [], suggestion=False)
        if not waiting:
            return
        controller = self._view_controller()
        if controller is None:
            return
        patterns = request_patterns(request)
        before, after = config_edit_stages(request, self._task_config)
        milled = request_stages(request) + after
        live = _frame(controller, BeamType.ION)
        aspect = live.data.shape[0] / live.data.shape[1] if live is not None else 2 / 3
        outside = request_outside_field(request, aspect, milled)
        missing: List[str] = []
        marks = self._request_marks(request, missing)
        # The agent's own image when it gave one; else the one the values were
        # measured on.
        evidence = _agent_image(self._load_image, request.get("image"))
        if evidence is not None:
            frame = evidence
            beam = frame.metadata.image_settings.beam_type or BeamType.ION
        else:
            frame, beam = self._measured_on(
                controller, marks, bool(patterns or milled or before)
            )
        # The card: the image the values were measured on, every value on it.
        if frame is None:
            if missing:
                # Not the live view: an item's values are in its own frame.
                self._show_picture(None, f"No FIB image of {missing[0]} yet to draw on")
            return
        layers = [
            _Layer(_BEFORE),
            _Layer(PURPLE_COLOR, fill=True),
            _Layer(RED_COLOR),
        ]
        grey, purple, red = layers
        if beam is BeamType.ION:
            grey.patterns = _stage_patterns(before)
            # Past the field, outlined in red over purple.
            purple.patterns = patterns + _stage_patterns(milled)
            red.patterns = outside
            grey.key(OUTLINE, "as now", grey.patterns)
            purple.key(
                FILL, "after the edit" if before else "patterns", purple.patterns
            )
            red.key(OUTLINE, "past the field", red.patterns)
        self._add_marks(frame, [m for m in marks.values() if not m.before], purple)
        self._add_marks(frame, [m for m in marks.values() if m.before], grey)
        title = next((m.title for m in marks.values() if m.title and m.places), "")
        self._picture(
            frame,
            beam,
            layers,
            suggestion=False,
            agent_image=evidence is not None,
            title="" if evidence is not None else title,
        )

    def _measured_on(self, controller, marks: Dict[str, "_Mark"], milling: bool):
        """The frame the request's values were measured on: a review's or an
        item's own image; else the FIB view's, where a mill draws; else the
        view of the beam a move is in. With its beam; None, None for none."""
        for mark in marks.values():
            for measured_on, _value in mark.places:
                if measured_on is not None:
                    return measured_on, mark.beam
        beams = [BeamType.ION] if milling else []
        beams += sorted({m.beam for m in marks.values()}, key=lambda b: b.value)
        if BeamType.ION in beams:
            return _frame(controller, BeamType.ION), BeamType.ION
        if beams:
            return _frame(controller, beams[0]), beams[0]
        return None, None

    def _request_marks(
        self, request: Dict[str, Any], missing: List[str]
    ) -> Dict[str, _Mark]:
        """What *request* places on the sample: move targets, a review's or an
        item edit's point of interest, an item edit's alignment area. An item
        edit whose item has no FIB image to draw on goes in *missing*."""
        marks: Dict[str, _Mark] = {}

        def add(overlay_id, beam, legend, places, area=False, shape=CROSS, **kw):
            mark = marks.setdefault(
                overlay_id, _Mark(beam, legend, [], area, shape, **kw)
            )
            mark.places.extend(places)

        for beam, offset in request_targets(request):
            add(
                TARGET_MARK,
                beam,
                "move target",
                [(None, lambda frame, o=offset: _offset_pixel(frame, o))],
                shape=TARGET,
            )
        for action in request.get("actions", []):
            tool = action.get("tool")
            arguments = action.get("arguments") or {}
            self._item_marks(tool, arguments, add)
            if tool == "review_decide":
                poi = _point((arguments.get("values") or {}).get("poi"))
                image = self._reference_image(arguments) if poi is not None else None
                if poi is not None and _has_pixel_size(image):
                    add(
                        POI_MARK,
                        BeamType.ION,
                        "point of interest",
                        [(image, _offset_pixel(image, poi))],
                        title=str(action.get("what") or ""),
                    )
            elif tool == "item_update":
                # Metres in the item's milling frame, and fractions of it: on
                # the item's own last FIB image, never the live view, which may
                # be somewhere else on the grid (FIB-1424).
                patch = arguments.get("patch") or {}
                poi = _point(patch.get("poi")) or _point(
                    {"x": patch.get("poi.x"), "y": patch.get("poi.y")}
                )
                area = _area(patch)
                if poi is None and area is None:
                    continue
                item = str(arguments.get("item_name") or "")
                image = self._item_image(item)
                if not _has_pixel_size(image) or getattr(image.data, "ndim", 0) != 2:
                    missing.append(item or "the lamella")
                    continue
                if poi is not None:
                    add(
                        POI_MARK,
                        BeamType.ION,
                        "point of interest",
                        [(image, _offset_pixel(image, poi))],
                        title=item,
                    )
                if area is not None:
                    add(
                        AREA_MARK,
                        BeamType.ION,
                        "alignment area",
                        [(image, _fraction_pixel(image, corner)) for corner in area],
                        area=True,
                        title=item,
                    )
        return marks

    def _item_marks(self, tool: str, arguments: Dict[str, Any], add) -> None:
        """Where an item edit puts a lamella: a new one on its overview or where
        the stage is or would be; a pose it moves to; a pose it sets, and that
        pose as it is now. Drawn on the FIB view's frame, through the stage."""
        from fibsem.geometry.projection import stage_to_pixel

        def centre(frame):
            height, width = frame.data.shape[-2:]
            return (width / 2, height / 2)

        def at(position):
            return lambda frame: stage_to_pixel(frame, position)

        item = arguments.get("item_name")
        if tool == "item_add":
            overview = arguments.get("overview")
            position = arguments.get("stage_position")
            if isinstance(overview, dict):
                image = self._overview_image(
                    overview.get("grid_name"), overview.get("filename")
                )
                pixel = _overview_pixel(image, overview)
                if pixel is not None:
                    add(ITEM_MARK, BeamType.ELECTRON, "new lamella", [(image, pixel)])
            elif isinstance(position, dict):
                stage = _stage_position(position)
                if stage is not None:
                    add(ITEM_MARK, BeamType.ION, "new lamella", [(None, at(stage))])
            else:
                add(ITEM_MARK, BeamType.ION, "new lamella", [(None, centre)])
        elif tool == "item_move_to_pose":
            pose = arguments.get("pose")
            stage = self._item_pose(item, pose)
            if stage is not None:
                add(
                    POSE_MARK,
                    BeamType.ION,
                    f"{item} {pose} pose",
                    [(None, at(stage))],
                    shape=TARGET,
                )
        elif tool == "item_set_pose":
            pose = arguments.get("pose_name")
            add(
                POSE_MARK,
                BeamType.ION,
                f"new {pose} pose",
                [(None, centre)],
                shape=TARGET,
            )
            stage = self._item_pose(item, pose)
            if stage is not None:
                add(
                    POSE_NOW_MARK,
                    BeamType.ION,
                    f"{pose} pose now",
                    [(None, at(stage))],
                    shape=TARGET,
                    before=True,
                )

    # --- suggestions ------------------------------------------------------------

    def show_suggestion(self, request: object, value: object) -> None:
        """Draw the agent's suggested value for the pending *request* (a
        question) on the card, on the image the question shows: an alignment
        area, detected features, spot-burn points. None, or a question without
        such a value, takes it down. A suggested point of interest is
        :meth:`show_suggested_point`'s."""
        from fibsem.applications.autolamella.workflows.interaction import (
            ConfirmDetection,
            EditAlignmentArea,
            RunSpotBurn,
        )

        controller = self._view_controller()
        if controller is None or value is None:
            if not isinstance(self.suggestion_picture, _PointPicture):
                self._picture(None, None, [], suggestion=True)
            return
        mark: Optional[_Mark] = None
        if isinstance(request, EditAlignmentArea):
            corners = [
                (value.left, value.top),
                (value.left + value.width, value.top + value.height),
            ]
            mark = _Mark(
                BeamType.ION,
                "alignment area",
                [(None, lambda frame, c=c: _fraction_pixel(frame, c)) for c in corners],
                area=True,
            )
        elif isinstance(request, RunSpotBurn):
            mark = _Mark(
                BeamType.ION,
                "spot burn points",
                [
                    (None, lambda frame, p=p: _fraction_pixel(frame, (p.x, p.y)))
                    for p in value
                ],
                shape=DOT,
            )
        elif isinstance(request, ConfirmDetection):
            image = getattr(request.detection, "fibsem_image", None)
            if _has_pixel_size(image):
                mark = _Mark(
                    image.metadata.image_settings.beam_type or BeamType.ION,
                    "features",
                    [(image, (px.x, px.y)) for px in value.values()],
                )
        if mark is None:
            return
        frame = mark.places[0][0] if mark.places else None
        if frame is None:
            frame = _frame(controller, mark.beam)
        if frame is None:
            return
        layer = _Layer(PURPLE_COLOR)
        self._add_marks(frame, [mark], layer)
        self._picture(frame, mark.beam, [layer], suggestion=True)

    def show_suggested_point(self, pixel: Optional[Tuple[float, float]]) -> None:
        """Draw the agent's suggested point of interest, in the FIB view's
        pixels, on the card. None takes it down."""
        if pixel is None:
            if isinstance(self.suggestion_picture, _PointPicture):
                self._picture(None, None, [], suggestion=True)
            return
        controller = self._view_controller()
        frame = _frame(controller, BeamType.ION) if controller is not None else None
        if frame is not None:
            layer = _Layer(PURPLE_COLOR)
            layer.crosses.append(tuple(pixel))
            layer.key(CROSS, "point of interest")
            self._picture(frame, BeamType.ION, [layer], suggestion=True, point=True)

    # --- drawing ----------------------------------------------------------------

    @staticmethod
    def _add_marks(frame, marks, layer: _Layer) -> None:
        """*marks* re-projected onto *frame*, into *layer*."""
        for mark in marks:
            pixels = mark.pixels(frame)
            if mark.area:
                layer.areas.append(pixels[:2])
                layer.key(AREA, mark.legend)
            else:
                layer.points(mark.shape).extend(pixels)
                layer.key(mark.shape, mark.legend)

    def _picture(
        self,
        frame,
        beam: Optional[BeamType],
        layers: List[_Layer],
        suggestion: bool,
        point: bool = False,
        agent_image: bool = False,
        title: str = "",
    ) -> None:
        drawn = None
        # The agent's own image shows with nothing drawn on it, too.
        if frame is not None and (agent_image or any(layers)):
            drawn = (_PointPicture if point else tuple)((frame, layers))
        if suggestion:
            self.suggestion_picture = drawn
        else:
            self.request_picture = drawn
        if drawn is None:
            self._show_picture(None, "", suggestion=suggestion)
            return
        caption = picture_caption(frame, beam)
        if title:
            caption = f"{title} · {caption}"
        self._show_picture(
            draw_picture(frame, layers),
            f"The agent's image · {caption}" if agent_image else caption,
            suggestion=suggestion,
            legend=legend_entries(layers),
        )


class _PointPicture(tuple):
    """The card's picture of a suggested point of interest: its own, so the
    other suggestions' pictures do not take it down."""
