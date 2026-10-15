"""What the agent's waiting request or suggestion would do, drawn on the image
(FIB-1404).

While a request waits for the operator, what it would do is drawn in the agent's
purple, dashed, so the operator sees it before they approve:

* the patterns a milling ``prepare`` would draw, on the FIB view;
* where a stable or vertical move would bring to the centre, a cross on that
  beam's view.

A suggested point of interest is drawn on the FIB view as a cross too, beside the
operator's own marker, until the suggestion goes.

They come down when the request is decided (an approved one draws its real
patterns as it runs) or the suggestion is taken or dropped. Display only: none
of them takes a click.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from fibsem.applications.autolamella.ui.agent_activity import DrawnStage
from fibsem.server.request_text import parse_patterns
from fibsem.structures import BeamType, Point
from fibsem.ui.tokens import PURPLE_COLOR

# The overlays' ids: one for the request's patterns, one per beam for its move
# targets, and one for the suggested point.
PATTERNS_OVERLAY = "agent_request_patterns"
TARGET_OVERLAY = "agent_request_target"
SUGGESTION_OVERLAY = "agent_suggestion"

# The beam a move tool works in when the agent does not say: as the tools do.
_MOVE_BEAMS = {"stable_move": BeamType.ELECTRON, "vertical_move": BeamType.ION}


def _beam(value: Any, default: BeamType) -> BeamType:
    if value is None:
        return default
    text = str(value).upper()
    if text in ("ION", "2"):
        return BeamType.ION
    if text in ("ELECTRON", "1"):
        return BeamType.ELECTRON
    return default


def request_patterns(request: Dict[str, Any]) -> List:
    """The patterns the request's milling ``prepare`` actions would draw."""
    patterns = []
    for action in request.get("actions", []):
        arguments = action.get("arguments") or {}
        if (
            action.get("tool") != "call_service_command"
            or arguments.get("service") != "milling"
            or arguments.get("command") != "prepare"
        ):
            continue
        kwargs = arguments.get("kwargs") or {}
        patterns.extend(parse_patterns(kwargs.get("patterns")))
    return patterns


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
        size=18.0,
        edge_width=2.0,
        legend_label=name,
        # Never armed, so never takes a click: a preview, not a marker to move.
        modal=True,
    )


class AgentPreviews:
    """Draws the agent's waiting request and suggested point on the views."""

    def __init__(self, view_controller: Callable[[], object]) -> None:
        # Asked when there is something to draw: the views are built after this.
        self._view_controller = view_controller
        self._request_id: Optional[str] = None
        self._drawn: List[Tuple[BeamType, str]] = []

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
        self._remove(PATTERNS_OVERLAY, TARGET_OVERLAY)
        self._request_id = request_id
        if not waiting:
            return
        controller = self._view_controller()
        if controller is None:
            return
        from fibsem.ui.widgets.canvas.canvas_state import MillingSpec

        patterns = request_patterns(request)
        if patterns:
            controller.set_overlay(
                BeamType.ION,
                MillingSpec(
                    id=PATTERNS_OVERLAY,
                    stages=[DrawnStage(patterns, name="Agent · waiting for you")],
                    colour=PURPLE_COLOR,
                    dashed=True,
                ),
            )
            self._drawn.append((BeamType.ION, PATTERNS_OVERLAY))
        by_beam: Dict[BeamType, List[Tuple[float, float]]] = {}
        for beam, offset in request_targets(request):
            pixel = self._pixel(controller, beam, offset)
            if pixel is not None:
                by_beam.setdefault(beam, []).append(pixel)
        for beam, points in by_beam.items():
            controller.set_overlay(
                beam, _cross(TARGET_OVERLAY, points, "Agent · move target")
            )
            self._drawn.append((beam, TARGET_OVERLAY))

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

    @staticmethod
    def _pixel(controller, beam: BeamType, offset: Point):
        from fibsem import conversions

        image = controller.image(beam) if hasattr(controller, "image") else None
        metadata = getattr(image, "metadata", None)
        if image is None or metadata is None or metadata.pixel_size is None:
            return None
        pixel = conversions.microscope_image_to_image_coordinates(
            offset, image.data.shape, metadata.pixel_size.x
        )
        return (pixel.x, pixel.y)

    def _remove(self, *overlay_ids: str) -> None:
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
