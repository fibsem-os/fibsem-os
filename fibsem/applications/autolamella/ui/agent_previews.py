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


class AgentPreviews:
    """Draws the agent's waiting request and suggested point on the views."""

    def __init__(self, view_controller: Callable[[], object]) -> None:
        # Asked when there is something to draw: the views are built after this.
        self._view_controller = view_controller
        self._request_id: Optional[str] = None
        self._drawn: List[Tuple[BeamType, str]] = []
        # The waiting request's move targets per beam, each with the frame it was
        # measured on (None until the view shows one): (frame, offset or pixel).
        self._targets: Dict[BeamType, List[Tuple[Optional[object], object]]] = {}
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
        self._remove(PATTERNS_OVERLAY, OUTSIDE_OVERLAY, TARGET_OVERLAY)
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
        for beam, offset in request_targets(request):
            self._targets.setdefault(beam, []).append((None, offset))
        self._follow(controller)
        for beam in self._targets:
            self._draw_targets(controller, beam)

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
        if beam in self._targets:
            controller = self._view_controller()
            if controller is not None:
                self._draw_targets(controller, beam)

    def _draw_targets(self, controller, beam: BeamType) -> None:
        """The move targets on *beam*'s view, each from the frame it was measured
        on onto the one shown; a target with no frame yet is measured on this one."""
        from fibsem.geometry.projection import reproject_pixel

        frame = _frame(controller, beam)
        if frame is None:
            return
        targets, points = [], []
        for measured_on, value in self._targets[beam]:
            if measured_on is None:
                measured_on, value = frame, _offset_pixel(frame, value)
            targets.append((measured_on, value))
            points.append(reproject_pixel(measured_on, frame, *value))
        self._targets[beam] = targets
        controller.set_overlay(
            beam, _cross(TARGET_OVERLAY, points, "Agent · move target")
        )
        if (beam, TARGET_OVERLAY) not in self._drawn:
            self._drawn.append((beam, TARGET_OVERLAY))

    def _remove(self, *overlay_ids: str) -> None:
        if TARGET_OVERLAY in overlay_ids:
            self._targets = {}
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
