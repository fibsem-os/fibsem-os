"""What the agent's waiting request or suggestion would do, drawn where the
operator decides (FIB-1404, FIB-1411).

On the card, under Why: the image the values were measured on, every value drawn
on it in the agent's purple. A review's or an item's point of interest on that
item's image; an item edit's alignment area; the patterns a mill would draw, a
config edit's as it would leave them (as they are now in grey) and any reaching
past the edge of the field in red, on the FIB image; where a move would go; a
suggested alignment area, spot-burn points, detected features or point of
interest, on the image the question shows.

On the views, only what is about to happen there: the patterns to mill (and the
red ones past the field) on the FIB view, and where a stable or vertical move
would bring to the centre, a cross on that beam's view; and a suggested point of
interest beside the operator's own marker, where they would move it.

They come down when the request is decided (an approved one draws its real
patterns as it runs) or the suggestion is taken or dropped. Display only: none
of them takes a click.

A value on the sample is kept with the frame it was measured on and re-projected
through the stage onto the frame it is drawn on (FIB-1411); a move target on a
view, onto every new frame the view takes. Patterns are places in the beam's
field, not on the sample: drawn against whatever frame shows.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from fibsem.applications.autolamella.ui.agent_activity import AGENT_LEGEND, DrawnStage
from fibsem.server.request_text import parse_patterns
from fibsem.structures import BeamType, Point
from fibsem.ui.tokens import PURPLE_COLOR, RED_COLOR

# The views' overlays: the request's patterns (and those past the field), its
# move targets, and the suggested point of interest.
PATTERNS_OVERLAY = "agent_request_patterns"
OUTSIDE_OVERLAY = "agent_request_outside"
TARGET_OVERLAY = "agent_request_target"
SUGGESTION_OVERLAY = "agent_suggestion"
# What else a request places on the sample: drawn on the card only.
POI_MARK = "poi"
AREA_MARK = "area"

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


def _cross(overlay_id: str, points, name: str):
    """Purple crosses, named in the view's legend."""
    from fibsem.ui.widgets.canvas.canvas_state import PointsSpec

    points = list(points)
    return PointsSpec(
        id=overlay_id,
        points=points,
        color=PURPLE_COLOR,
        selected_color=PURPLE_COLOR,
        marker="+",
        size=28.0,
        edge_width=3.0,
        legend_label=name,
        # Never armed, so never takes a click: a preview, not a marker to move.
        modal=True,
    )


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


class _Mark:
    """Places on the sample one overlay draws: crosses, or the two corners of an
    area. Each place is kept with the frame it was measured on, or with how to
    measure it on the first frame the view shows (``(None, on_frame)``)."""

    def __init__(self, beam: BeamType, legend: str, places, area: bool = False):
        self.beam = beam
        self.legend = legend
        self.places: List[Tuple[Optional[object], Any]] = list(places)
        self.area = area

    def pixels(self, frame) -> List[Tuple[float, float]]:
        """The places as *frame*'s pixels."""
        from fibsem.geometry.projection import reproject_pixel

        measured = []
        for measured_on, value in self.places:
            if measured_on is None:
                measured_on, value = frame, value(frame)
            measured.append((measured_on, value))
        self.places = measured
        return [reproject_pixel(on, frame, *value) for on, value in measured]


# The picture on the card: its width at most, and how its marks are drawn.
PICTURE_WIDTH = 1024
_GREY = "#9a9a9a"


class _Layer:
    """What one colour draws on the card's picture: crosses and areas (pixels
    of the picture's frame), and patterns (metres from the field's centre)."""

    def __init__(self, colour: str, dashed: bool = True) -> None:
        self.colour = colour
        self.dashed = dashed
        self.crosses: List[Tuple[float, float]] = []
        self.areas: List[List[Tuple[float, float]]] = []
        self.patterns: List = []

    def __bool__(self) -> bool:
        return bool(self.crosses or self.areas or self.patterns)


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
        pen = QPen(QColor(layer.colour), line)
        if layer.dashed:
            pen.setStyle(Qt.DashLine)  # type: ignore
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)  # type: ignore
        for pattern in layer.patterns:
            for ring in pattern_outline(pattern):
                points = [at(_offset_pixel(frame, Point(x=x, y=y))) for x, y in ring]
                if len(points) == 2:
                    painter.drawLine(QLineF(points[0], points[1]))
                else:
                    painter.drawPolygon(QPolygonF(points))
        for (x0, y0), (x1, y1) in layer.areas:
            painter.drawPolygon(
                QPolygonF([at((x0, y0)), at((x1, y0)), at((x1, y1)), at((x0, y1))])
            )
        pen.setStyle(Qt.SolidLine)  # type: ignore
        pen.setWidthF(line * 1.5)
        painter.setPen(pen)
        arm = 6 * line
        for pixel in layer.crosses:
            centre = at(pixel)
            painter.drawLine(
                QLineF(centre.x() - arm, centre.y(), centre.x() + arm, centre.y())
            )
            painter.drawLine(
                QLineF(centre.x(), centre.y() - arm, centre.x(), centre.y() + arm)
            )
    painter.end()
    return image


def _stage_patterns(stages) -> List:
    patterns = []
    for stage in stages:
        try:
            patterns.extend(stage.define_patterns())
        except Exception:  # noqa: BLE001 - a stage that cannot define them
            continue
    return patterns


class AgentPreviews:
    """Draws the agent's waiting request and suggestion: on the card, on the
    image their values were measured on; on the views, what is about to happen
    there (the patterns to mill, where a move goes)."""

    def __init__(
        self,
        view_controller: Callable[[], object],
        reference_image: Optional[Callable[[Dict[str, Any]], object]] = None,
        task_config: Optional[Callable[[str, str], object]] = None,
        show_picture: Optional[Callable[..., None]] = None,
    ) -> None:
        # Asked when there is something to draw: the views are built after this.
        self._view_controller = view_controller
        # The image a review_decide action's values sit on (its arguments), or
        # None: the window reads it from the experiment.
        self._reference_image = reference_image or (lambda arguments: None)
        # An item's task config (item name, task name), for a config edit's
        # patterns; None when there is none.
        self._task_config = task_config or (lambda item, task: None)
        # The card's picture: (QImage or None, caption, suggestion=bool).
        self._show_picture = show_picture or (lambda *args, **kwargs: None)
        self._request_id: Optional[str] = None
        self._drawn: List[Tuple[BeamType, str]] = []
        # Where the waiting request's moves go, re-projected as the views move.
        self._marks: Dict[str, _Mark] = {}
        self._following = None  # the controller whose frame changes are heard
        # The card's pictures, as drawn: (frame, layers), for tests and reading.
        self.request_picture: Optional[Tuple[object, List[_Layer]]] = None
        self.suggestion_picture: Optional[Tuple[object, List[_Layer]]] = None

    @property
    def drawn(self) -> List[Tuple[BeamType, str]]:
        """The overlays drawn on the views now, as (beam, overlay id)."""
        return list(self._drawn)

    # --- requests ---------------------------------------------------------------

    def show_request(self, request: Optional[Dict[str, Any]]) -> None:
        """Draw what *request* would do while it waits; anything else (None, or a
        request being applied or ended) takes the previews down."""
        waiting = request is not None and request.get("status") == "pending"
        request_id = request.get("request_id") if waiting else None
        if request_id == self._request_id:
            return
        self._remove(PATTERNS_OVERLAY, OUTSIDE_OVERLAY, TARGET_OVERLAY)
        self._request_id = request_id
        self._picture(None, None, [], suggestion=False)
        if not waiting:
            return
        controller = self._view_controller()
        if controller is None:
            return
        from fibsem.ui.widgets.canvas.canvas_state import MillingSpec

        patterns = request_patterns(request)
        before, after = config_edit_stages(request, self._task_config)
        milled = request_stages(request) + after
        stages = list(milled)
        if patterns:
            stages.insert(0, DrawnStage(patterns, name="Agent · waiting for you"))
        if stages:
            controller.set_overlay(
                BeamType.ION,
                MillingSpec(
                    id=PATTERNS_OVERLAY,
                    stages=stages,
                    # What a config edit changes, as it is now.
                    background_stages=before,
                    colour=PURPLE_COLOR,
                    dashed=True,
                    legend_loc=AGENT_LEGEND,
                ),
            )
            self._drawn.append((BeamType.ION, PATTERNS_OVERLAY))
        live = _frame(controller, BeamType.ION)
        aspect = live.data.shape[0] / live.data.shape[1] if live is not None else 2 / 3
        outside = request_outside_field(request, aspect, milled)
        if outside:
            controller.set_overlay(
                BeamType.ION,
                MillingSpec(
                    id=OUTSIDE_OVERLAY,
                    stages=[DrawnStage(outside, name="Agent · outside the field")],
                    colour=RED_COLOR,
                    dashed=True,
                    legend_loc=AGENT_LEGEND,
                ),
            )
            self._drawn.append((BeamType.ION, OUTSIDE_OVERLAY))
        marks = self._request_marks(request)
        # Before a target is first drawn, which measures it on the view's frame.
        frame, beam = self._measured_on(controller, marks, bool(stages or outside))
        if TARGET_OVERLAY in marks:
            self._marks[TARGET_OVERLAY] = marks[TARGET_OVERLAY]
            self._follow(controller)
            self._draw_mark(controller, TARGET_OVERLAY)

        # The card: the image the values were measured on, every value on it.
        if frame is None:
            return
        layers = [_Layer(_GREY), _Layer(PURPLE_COLOR), _Layer(RED_COLOR)]
        grey, purple, red = layers
        if beam is BeamType.ION:
            grey.patterns = _stage_patterns(before)
            # Past the field, red over purple.
            purple.patterns = patterns + _stage_patterns(milled)
            red.patterns = outside
        self._add_marks(frame, marks.values(), purple)
        self._picture(frame, beam, layers, suggestion=False)

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

    def _request_marks(self, request: Dict[str, Any]) -> Dict[str, _Mark]:
        """What *request* places on the sample: move targets, a review's or an
        item edit's point of interest, an item edit's alignment area."""
        marks: Dict[str, _Mark] = {}

        def add(overlay_id, beam, legend, places, area=False):
            mark = marks.setdefault(overlay_id, _Mark(beam, legend, [], area))
            mark.places.extend(places)

        for beam, offset in request_targets(request):
            add(
                TARGET_OVERLAY,
                beam,
                "Agent · move target",
                [(None, lambda frame, o=offset: _offset_pixel(frame, o))],
            )
        for action in request.get("actions", []):
            tool = action.get("tool")
            arguments = action.get("arguments") or {}
            if tool == "review_decide":
                poi = _point((arguments.get("values") or {}).get("poi"))
                image = self._reference_image(arguments) if poi is not None else None
                if poi is not None and _has_pixel_size(image):
                    add(
                        POI_MARK,
                        BeamType.ION,
                        "Agent · point of interest",
                        [(image, _offset_pixel(image, poi))],
                    )
            elif tool == "item_update":
                # Metres in the milling frame, and fractions of it: measured on
                # the frame the FIB view shows as the request comes (FIB-1411).
                patch = arguments.get("patch") or {}
                poi = _point(patch.get("poi")) or _point(
                    {"x": patch.get("poi.x"), "y": patch.get("poi.y")}
                )
                if poi is not None:
                    add(
                        POI_MARK,
                        BeamType.ION,
                        "Agent · point of interest",
                        [(None, lambda frame, p=poi: _offset_pixel(frame, p))],
                    )
                area = _area(patch)
                if area is not None:
                    add(
                        AREA_MARK,
                        BeamType.ION,
                        "Agent · alignment area",
                        [
                            (None, lambda frame, c=corner: _fraction_pixel(frame, c))
                            for corner in area
                        ],
                        area=True,
                    )
        return marks

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
                "Agent's suggested area",
                [(None, lambda frame, c=c: _fraction_pixel(frame, c)) for c in corners],
                area=True,
            )
        elif isinstance(request, RunSpotBurn):
            mark = _Mark(
                BeamType.ION,
                "Agent's suggested points",
                [
                    (None, lambda frame, p=p: _fraction_pixel(frame, (p.x, p.y)))
                    for p in value
                ],
            )
        elif isinstance(request, ConfirmDetection):
            image = getattr(request.detection, "fibsem_image", None)
            if _has_pixel_size(image):
                mark = _Mark(
                    image.metadata.image_settings.beam_type or BeamType.ION,
                    "Agent's suggested features",
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
        pixels: on the view, beside the operator's own marker, where they would
        move it; and on the card. None takes it down."""
        self._remove(SUGGESTION_OVERLAY)
        if pixel is None:
            if isinstance(self.suggestion_picture, _PointPicture):
                self._picture(None, None, [], suggestion=True)
            return
        controller = self._view_controller()
        if controller is None:
            return
        controller.set_overlay(
            BeamType.ION,
            _cross(SUGGESTION_OVERLAY, [pixel], "Agent's suggestion"),
        )
        self._drawn.append((BeamType.ION, SUGGESTION_OVERLAY))
        frame = _frame(controller, BeamType.ION)
        if frame is not None:
            layer = _Layer(PURPLE_COLOR)
            layer.crosses.append(tuple(pixel))
            self._picture(frame, BeamType.ION, [layer], suggestion=True, point=True)

    # --- drawing ----------------------------------------------------------------

    @staticmethod
    def _add_marks(frame, marks, layer: _Layer) -> None:
        """*marks* re-projected onto *frame*, into *layer*."""
        for mark in marks:
            pixels = mark.pixels(frame)
            if mark.area:
                layer.areas.append(pixels[:2])
            else:
                layer.crosses.extend(pixels)

    def _picture(
        self,
        frame,
        beam: Optional[BeamType],
        layers: List[_Layer],
        suggestion: bool,
        point: bool = False,
    ) -> None:
        drawn = None
        if frame is not None and any(layers):
            drawn = (_PointPicture if point else tuple)((frame, layers))
        if suggestion:
            self.suggestion_picture = drawn
        else:
            self.request_picture = drawn
        if drawn is None:
            self._show_picture(None, "", suggestion=suggestion)
            return
        self._show_picture(
            draw_picture(frame, layers),
            picture_caption(frame, beam),
            suggestion=suggestion,
        )

    def _follow(self, controller) -> None:
        """Hear *controller*'s new frames, once."""
        signal = getattr(controller, "frame_changed", None)
        if signal is None or self._following is controller:
            return
        signal.connect(self.on_frame_changed)
        self._following = controller

    def on_frame_changed(self, beam: BeamType) -> None:
        """The view for *beam* took another frame: where a move goes moves with
        it."""
        controller = self._view_controller()
        if controller is None:
            return
        for overlay_id, mark in self._marks.items():
            if mark.beam == beam:
                self._draw_mark(controller, overlay_id)

    def _draw_mark(self, controller, overlay_id: str) -> None:
        """*overlay_id*'s places on its beam's view, from the frames they were
        measured on onto the one shown."""
        mark = self._marks[overlay_id]
        frame = _frame(controller, mark.beam)
        if frame is None:
            return
        controller.set_overlay(
            mark.beam, _cross(overlay_id, mark.pixels(frame), mark.legend)
        )
        if (mark.beam, overlay_id) not in self._drawn:
            self._drawn.append((mark.beam, overlay_id))

    def _remove(self, *overlay_ids: str) -> None:
        for overlay_id in overlay_ids:
            self._marks.pop(overlay_id, None)
        keep = [(b, i) for b, i in self._drawn if i not in overlay_ids]
        gone = [(b, i) for b, i in self._drawn if i in overlay_ids]
        self._drawn = keep
        if not gone:
            return
        controller = self._view_controller()
        if controller is None:
            return
        for beam, overlay_id in gone:
            controller.remove_overlay(beam, overlay_id)


class _PointPicture(tuple):
    """The card's picture of a suggested point of interest: its own, so the
    other suggestions' pictures do not take it down."""
