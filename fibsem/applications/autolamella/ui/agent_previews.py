"""What the agent's waiting request or suggestion would do, drawn on the image
(FIB-1404).

While a request waits for the operator, what it would do is drawn in the agent's
purple, dashed, so the operator sees it before they approve:

* the patterns a milling ``prepare`` would draw, and the stages a ``run_stage``
  or ``run_task`` would mill, on the FIB view; any reaching past the edge of
  the field again in red;
* where a stable or vertical move would bring to the centre, a cross on that
  beam's view.

A suggested point of interest is drawn on the FIB view as a cross too, beside the
operator's own marker, until the suggestion goes.

They come down when the request is decided (an approved one draws its real
patterns as it runs) or the suggestion is taken or dropped. Display only: none
of them takes a click.

A move target is a place on the sample, measured on the frame the view showed
when the request came: it is kept with that frame and re-projected through the
stage onto every new frame the view takes (FIB-1411), and drawn once a frame
arrives if none showed yet. Patterns are places in the beam's field, not on the
sample: the view draws them against whatever frame it shows.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from fibsem.applications.autolamella.ui.agent_activity import AGENT_LEGEND, DrawnStage
from fibsem.server.request_text import parse_patterns
from fibsem.structures import BeamType, Point
from fibsem.ui.tokens import PURPLE_COLOR, RED_COLOR

# The overlays' ids: one for the request's patterns, one per beam for its move
# targets, and one for the suggested point.
PATTERNS_OVERLAY = "agent_request_patterns"
OUTSIDE_OVERLAY = "agent_request_outside"
TARGET_OVERLAY = "agent_request_target"
POI_OVERLAY = "agent_request_poi"
AREA_OVERLAY = "agent_request_area"
# What a request places on the sample, re-projected as the views move.
SAMPLE_OVERLAYS = (TARGET_OVERLAY, POI_OVERLAY, AREA_OVERLAY)
SUGGESTION_OVERLAY = "agent_suggestion"

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


def request_outside_field(request: Dict[str, Any], aspect: float = 2 / 3) -> List:
    """The patterns the request would mill past the edge of the field: its
    milling hfw wide and *aspect* of that high (an image's height over width).
    The Demo mills them; a real instrument refuses (FIB-1411)."""
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
    for stage in request_stages(request):
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


def _area_stage(frame, corners, name: str) -> DrawnStage:
    """An area given by two corner pixels of *frame*, as a rectangle the
    milling overlay draws (metres from the frame's centre, y up)."""
    from fibsem.structures import FibsemRectangleSettings

    (x0, y0), (x1, y1) = corners
    height, width = frame.data.shape[-2:]
    size = frame.metadata.pixel_size.x
    rectangle = FibsemRectangleSettings(
        width=abs(x1 - x0) * size,
        height=abs(y1 - y0) * size,
        depth=0.0,
        centre_x=((x0 + x1) / 2 - width / 2) * size,
        centre_y=-((y0 + y1) / 2 - height / 2) * size,
    )
    return DrawnStage([rectangle], name=name)


class AgentPreviews:
    """Draws the agent's waiting request and suggested point on the views."""

    def __init__(
        self,
        view_controller: Callable[[], object],
        reference_image: Optional[Callable[[Dict[str, Any]], object]] = None,
    ) -> None:
        # Asked when there is something to draw: the views are built after this.
        self._view_controller = view_controller
        # The image a review_decide action's values sit on (its arguments), or
        # None: the window reads it from the experiment.
        self._reference_image = reference_image or (lambda arguments: None)
        self._request_id: Optional[str] = None
        self._drawn: List[Tuple[BeamType, str]] = []
        # What the waiting request places on the sample, by overlay id.
        self._marks: Dict[str, _Mark] = {}
        self._following = None  # the controller whose frame changes are heard

    @property
    def drawn(self) -> List[Tuple[BeamType, str]]:
        """The overlays drawn now, as (beam, overlay id)."""
        return list(self._drawn)

    def show_request(self, request: Optional[Dict[str, Any]]) -> None:
        """Draw what *request* would do while it waits; anything else (None, or a
        request being applied or ended) takes the previews down."""
        waiting = request is not None and request.get("status") == "pending"
        request_id = request.get("request_id") if waiting else None
        if request_id == self._request_id:
            return
        self._remove(PATTERNS_OVERLAY, OUTSIDE_OVERLAY, *SAMPLE_OVERLAYS)
        self._request_id = request_id
        if not waiting:
            return
        controller = self._view_controller()
        if controller is None:
            return
        from fibsem.ui.widgets.canvas.canvas_state import MillingSpec

        patterns = request_patterns(request)
        stages = request_stages(request)
        if patterns:
            stages.insert(0, DrawnStage(patterns, name="Agent · waiting for you"))
        if stages:
            controller.set_overlay(
                BeamType.ION,
                MillingSpec(
                    id=PATTERNS_OVERLAY,
                    stages=stages,
                    colour=PURPLE_COLOR,
                    dashed=True,
                    legend_loc=AGENT_LEGEND,
                ),
            )
            self._drawn.append((BeamType.ION, PATTERNS_OVERLAY))
        frame = _frame(controller, BeamType.ION)
        aspect = (
            frame.data.shape[0] / frame.data.shape[1] if frame is not None else 2 / 3
        )
        outside = request_outside_field(request, aspect)
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
        self._marks = self._request_marks(request)
        self._follow(controller)
        for overlay_id in self._marks:
            self._draw_mark(controller, overlay_id)

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
                        POI_OVERLAY,
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
                        POI_OVERLAY,
                        BeamType.ION,
                        "Agent · point of interest",
                        [(None, lambda frame, p=poi: _offset_pixel(frame, p))],
                    )
                area = _area(patch)
                if area is not None:
                    add(
                        AREA_OVERLAY,
                        BeamType.ION,
                        "Agent · alignment area",
                        [
                            (None, lambda frame, c=corner: _fraction_pixel(frame, c))
                            for corner in area
                        ],
                        area=True,
                    )
        return marks

    def show_suggested_point(self, pixel: Optional[Tuple[float, float]]) -> None:
        """Draw the agent's suggested point of interest on the FIB view, in
        pixels; None takes it down."""
        self._remove(SUGGESTION_OVERLAY)
        if pixel is None:
            return
        controller = self._view_controller()
        if controller is None:
            return
        controller.set_overlay(
            BeamType.ION,
            _cross(SUGGESTION_OVERLAY, [pixel], "Agent's suggestion"),
        )
        self._drawn.append((BeamType.ION, SUGGESTION_OVERLAY))

    def _follow(self, controller) -> None:
        """Hear *controller*'s new frames, once."""
        signal = getattr(controller, "frame_changed", None)
        if signal is None or self._following is controller:
            return
        signal.connect(self.on_frame_changed)
        self._following = controller

    def on_frame_changed(self, beam: BeamType) -> None:
        """The view for *beam* took another frame: what sits on the sample moves
        with it."""
        controller = self._view_controller()
        if controller is None:
            return
        for overlay_id, mark in self._marks.items():
            if mark.beam == beam:
                self._draw_mark(controller, overlay_id)

    def _draw_mark(self, controller, overlay_id: str) -> None:
        """*overlay_id*'s places on its beam's view, from the frames they were
        measured on onto the one shown."""
        from fibsem.ui.widgets.canvas.canvas_state import MillingSpec

        mark = self._marks[overlay_id]
        frame = _frame(controller, mark.beam)
        if frame is None:
            return
        pixels = mark.pixels(frame)
        if mark.area:
            spec = MillingSpec(
                id=overlay_id,
                stages=[_area_stage(frame, pixels, mark.legend)],
                colour=PURPLE_COLOR,
                dashed=True,
                legend_loc=AGENT_LEGEND,
            )
        else:
            spec = _cross(overlay_id, pixels, mark.legend)
        controller.set_overlay(mark.beam, spec)
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
