"""What the agent's waiting request or suggestion would do, drawn where the
operator decides: the window's card and the chat page (FIB-1404, FIB-1411).
The session draws them, window or none (FIB-1500).

Under Why: the image the values were measured on, every value drawn on it in
the app's own styles, with a legend, an agent chip, a scalebar and the image's
metadata bar. A decision's or an item's point of interest on that item's image;
an item edit's alignment area; the patterns a mill would draw, a config edit's as
it would leave them (the current ones dashed grey) and any reaching past the edge
of the field in red, on the FIB image; where a move would go. A suggestion on a
pending proposal is drawn as the decision it would make.

Nothing is drawn on the live views: they show what the microscope does, and the
agent's real patterns only once its mill runs (`agent_activity`). The picture
comes down when the request is decided, or the suggestion is approved,
declined or dropped.

A value on the sample is kept with the frame it was measured on and re-projected
through the stage onto the frame it is drawn on (FIB-1411). Patterns are places
in the beam's field, not on the sample: drawn against the frame's field.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from fibsem.services.milling import parse_patterns
from fibsem.structures import BeamType, Point
from fibsem.theme.tokens import ERROR_COLOR, ORANGE_COLOR, WHITE_ICON_COLOR
from fibsem.workflows.evidence import frame_path

# What a request places on the sample, by kind.
TARGET_MARK = "target"
POI_MARK = "poi"
AREA_MARK = "area"
ITEM_MARK = "item"
POSE_MARK = "pose"
POSE_NOW_MARK = "pose_now"
SPOT_MARK = "spot"
FEATURE_MARK = "features"

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


# The milling tools that are a service command by another name: their
# arguments are the command's kwargs.
_MILLING_TOOLS = {"milling_run_task": "run_task", "milling_run_stage": "run_stage"}


def _milling_commands(request: Dict[str, Any]):
    """``(command, kwargs)`` for each of the request's milling service commands,
    called as one or through a milling tool."""
    for action in request.get("actions", []):
        arguments = action.get("arguments") or {}
        tool = action.get("tool")
        if tool == "service_call_command" and arguments.get("service") == "milling":
            yield arguments.get("command"), arguments.get("kwargs") or {}
        elif tool in _MILLING_TOOLS:
            yield _MILLING_TOOLS[tool], arguments


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

    from fibsem.workflows.config_patch import apply_patch

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


def decided_passes(
    request: Dict[str, Any], proposal_for: Callable[[Dict[str, Any]], object]
) -> Tuple[List, List, List]:
    """``(milled, kept, points)``: what the request's ``proposal_decide`` calls on
    a loop's pass would do. A run mills ``milled`` (the pass's config as
    decided: the call's values, patched, else the pass's own) or burns
    ``points`` (fractions of the frame); a pass decided without running leaves
    its stages in ``kept``. *proposal_for* gives the pass a call decides (its
    arguments), or None; a call that does not read is left out: the route
    refuses it."""
    from fibsem.workflows.proposals import (
        ACTION_RUN,
        MILLING,
        SPOT_BURN,
        _decode_values,
        patch_values,
    )

    milled, kept, points = [], [], []
    for action in request.get("actions", []):
        if action.get("tool") != "proposal_decide":
            continue
        arguments = action.get("arguments") or {}
        proposal = proposal_for(arguments)
        kind = getattr(proposal, "kind", None)
        if kind not in (MILLING, SPOT_BURN):
            continue
        try:
            values = _decode_values(arguments.get("values") or {}) or dict(
                proposal.values
            )
            if arguments.get("patch"):
                values = patch_values(kind, values, dict(arguments["patch"]))
        except Exception:  # noqa: BLE001 - described, not drawn
            continue
        runs = arguments.get("action") == ACTION_RUN
        if kind == SPOT_BURN:
            if runs:
                settings = values.get("spot_burn")
                points.extend((p.x, p.y) for p in getattr(settings, "coordinates", []))
            continue
        config = values.get("milling_config")
        stages = [s for s in getattr(config, "stages", []) or [] if s.enabled]
        (milled if runs else kept).extend(stages)
    return milled, kept, points


def request_outside_field(
    request: Dict[str, Any], aspect: float = 2 / 3, stages: Optional[List] = None
) -> List:
    """The patterns the request would mill past the edge of the field: its
    milling hfw wide and *aspect* of that high (an image's height over width).
    The Demo mills them; a real instrument refuses (FIB-1411). *stages* in
    place of the request's own run_stage and run_task stages."""
    from fibsem.milling.base import FibsemMillingStage
    from fibsem.milling.patterning.utils import outside_field, stage_outside_field
    from fibsem.structures import FibsemMillingSettings

    outside = []
    for command, kwargs in _milling_commands(request):
        if command != "prepare":
            continue
        try:
            hfw = FibsemMillingSettings.from_dict(dict(kwargs["settings"])).hfw
        except Exception:  # noqa: BLE001 - no field to check against
            continue
        outside.extend(
            pattern
            for pattern in parse_patterns(kwargs.get("patterns"))
            if outside_field(pattern, hfw, hfw * aspect)
        )
    for stage in request_stages(request) if stages is None else stages:
        if isinstance(stage, FibsemMillingStage):
            try:
                outside.extend(stage_outside_field(stage, aspect))
            except Exception:  # noqa: BLE001 - a stage that cannot define them
                continue
    return outside


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


def _frame(frame_of: Callable[[BeamType], object], beam: BeamType):
    """*beam*'s latest frame, when it has a pixel size; else None."""
    image = frame_of(beam)
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
    (origin top-left), dotted ("alignment_area.left") or nested; None when the
    edit sets no complete area."""
    nested = patch.get("alignment_area")
    area = dict(nested) if isinstance(nested, dict) else {}
    for key in ("left", "top", "width", "height"):
        area.setdefault(key, patch.get(f"alignment_area.{key}"))
    try:
        left, top = float(area["left"]), float(area["top"])
        width, height = float(area["width"]), float(area["height"])
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
# How the legend shows the rest: patterns to mill, filled; patterns outlined
# (those past the field), or dashed (a config edit's as they are now); an area,
# dashed.
FILL, OUTLINE, DASHED, AREA = "fill", "outline", "dashed", "area"


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
        item: str = "",
        stage: Optional[object] = None,
        kind: str = "",
    ):
        self.beam = beam
        # What it is (`POI_MARK`, ...), for its colour.
        self.kind = kind
        self.legend = legend
        self.places: List[Tuple[Optional[object], Any]] = list(places)
        self.area = area
        self.shape = shape
        # Where something is now, drawn as a config edit's patterns as they are.
        self.before = before
        # Whose image the places were measured on ("05-pale-lark"), for the
        # picture's caption.
        self.title = title
        # A pose's stage position and its item: drawn on the item's grid
        # overview when the view does not show it.
        self.item = item
        self.stage = stage

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


# The picture on the card: its width at most.
PICTURE_WIDTH = 1024
# The app's colours for what a request places, as its editors draw them: the
# point of interest as the lamella editor, the alignment area as its overlay,
# spot burn points as their editor. A move target has no style of its own.
MARK_COLOURS = {
    TARGET_MARK: WHITE_ICON_COLOR,
    POI_MARK: "magenta",
    AREA_MARK: "limegreen",
    ITEM_MARK: ORANGE_COLOR,
    POSE_MARK: WHITE_ICON_COLOR,
    SPOT_MARK: "cyan",
    FEATURE_MARK: "yellow",
}
# What is there now (a config edit's patterns, a pose): grey, dashed.
CURRENT_COLOUR = "#bfbfbf"
# Patterns past the field: the app's error red.
PAST_FIELD_COLOUR = ERROR_COLOR
# A pass decided without running: outlined, nothing milled.
UNMILLED_COLOUR = CURRENT_COLOUR


class _Layer:
    """What one colour draws on the card's picture: crosses, crosshairs, dots
    and areas (pixels of the picture's frame), and patterns (metres from the
    field's centre), filled (*fill*, what would be milled: each stage in the
    milling view's colour, `names` its stage) or outlined, solid or *dashed*.
    Areas, which are never milled, are dashed."""

    def __init__(self, colour: str, fill: bool = False, dashed: bool = False) -> None:
        self.colour = colour
        self.fill = fill
        self.dashed = dashed
        # Each pattern's stage name, for its colour and the legend.
        self.names: List[str] = []
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
        return FILL if self.fill else DASHED if self.dashed else OUTLINE

    def add_stages(self, stages, name: str = "") -> None:
        """*stages*' patterns, each named for its stage; or, *name* given,
        bare patterns under that name."""
        for stage in stages:
            patterns = _stage_patterns([stage]) if not name else [stage]
            self.patterns.extend(patterns)
            self.names.extend([name or stage.name] * len(patterns))

    def stage_names(self) -> List[str]:
        """The stages it draws, in order, each once."""
        return list(dict.fromkeys(self.names))

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


def _on_frame(frame, corners) -> bool:
    """Whether the box *corners* spans (two pixels) overlaps *frame*."""
    height, width = frame.data.shape[:2]
    (x0, y0), (x1, y1) = corners[0], corners[-1]
    return (
        min(x0, x1) < width
        and max(x0, x1) >= 0
        and min(y0, y1) < height
        and max(y0, y1) >= 0
    )


def _patterns_on_frame(frame, patterns) -> bool:
    """Whether any of *patterns* shows on *frame*, as the picture places them."""
    from fibsem.milling.patterning.utils import pattern_outline

    for pattern in patterns:
        try:
            rings = pattern_outline(pattern)
        except Exception:  # noqa: BLE001 - not drawn either
            continue
        for ring in rings:
            pixels = [_offset_pixel(frame, Point(x=x, y=y)) for x, y in ring]
            if not pixels:
                continue
            xs, ys = [p[0] for p in pixels], [p[1] for p in pixels]
            if _on_frame(frame, [(min(xs), min(ys)), (max(xs), max(ys))]):
                return True
    return False


def _key_past_field(frame, layer: "_Layer") -> None:
    """Key *layer*'s past-the-field patterns, saying so when none is on the
    picture: the view can be narrower than the milling field."""
    if layer.patterns and not _patterns_on_frame(frame, layer.patterns):
        layer.key(OUTLINE, "past the field, off the picture")
    else:
        layer.key(OUTLINE, "past the field", layer.patterns)


def _chip(suggestion: bool) -> str:
    """Whose the picture is, on it."""
    return "Agent's suggestion" if suggestion else "Agent's request"


def stage_colour(index: int) -> str:
    """The milling view's colour for its *index*-th stage."""
    from fibsem.milling.patterning.plotting import COLOURS

    return COLOURS[index % len(COLOURS)]


def legend_entries(layers: List[_Layer]) -> List[Tuple[str, str, str]]:
    """What each layer draws, for the picture's legend: ``(colour, shape,
    words)``. Patterns to mill are named by their stages."""
    entries = []
    for layer in layers:
        for shape, text in layer.keys:
            if shape == FILL and layer.names:
                entries.extend(
                    (stage_colour(i), FILL, name)
                    for i, name in enumerate(layer.stage_names())
                )
            else:
                entries.append((layer.colour, shape, text))
    return entries


def _stage_patterns(stages) -> List:
    patterns = []
    for stage in stages:
        try:
            patterns.extend(stage.define_patterns())
        except Exception:  # noqa: BLE001 - a stage that cannot define them
            continue
    return patterns


class AgentPictures:
    """Draws the agent's waiting request and suggestion on the image their
    values were measured on, for the card and the chat page. Nothing on the
    views."""

    def __init__(
        self,
        frame: Callable[[BeamType], object],
        reference_image: Optional[Callable[[Dict[str, Any]], object]] = None,
        task_config: Optional[Callable[[str, str], object]] = None,
        show_picture: Optional[Callable[..., None]] = None,
        item_pose: Optional[Callable[[str, str], object]] = None,
        item_image: Optional[Callable[[str], object]] = None,
        overview_image: Optional[Callable[[str, str], object]] = None,
        load_image: Optional[Callable[[str], object]] = None,
        item_overview: Optional[Callable[[str], object]] = None,
        pending_proposal: Optional[Callable[[Dict[str, Any]], object]] = None,
    ) -> None:
        # Each beam's latest frame, or None: the session's images.
        self._frame_of = frame
        # The image a proposal_decide action's values sit on (its arguments), or
        # None: the window reads it from the experiment.
        self._reference_image = reference_image or (lambda arguments: None)
        # The pending proposal a proposal_decide action decides (its arguments),
        # or None: for a loop's pass, what it would run.
        self._pending_proposal = pending_proposal or (lambda arguments: None)
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
        # The latest overview of an item's grid (item name), for a pose the
        # view does not show; None when there is none.
        self._item_overview = item_overview or (lambda item: None)
        # A request's own image, by path (the agent's evidence).
        self._load_image = load_image or _load_beam_image
        # The picture: (PNG or None, caption, suggestion=bool).
        self._show_picture = show_picture or (lambda *args, **kwargs: None)
        self._shown_ids: Dict[bool, Optional[str]] = {False: None, True: None}
        # The card's pictures, as drawn: (frame, layers), for tests and reading.
        self.request_picture: Optional[Tuple[object, List[_Layer]]] = None
        self.suggestion_picture: Optional[Tuple[object, List[_Layer]]] = None

    # --- requests ---------------------------------------------------------------

    def show_request(self, request: Optional[Dict[str, Any]]) -> None:
        """Draw what *request* would do while it waits; anything else (None, or a
        request being applied or ended) takes the previews down."""
        self._draw(request, suggestion=False)

    def show_suggestion(self, suggestion: Optional[Dict[str, Any]]) -> None:
        """Draw the agent's suggestion on a pending proposal, given as the
        request that would decide it (``suggestion_request``); None takes it
        down."""
        self._draw(suggestion, suggestion=True)

    def _draw(self, request: Optional[Dict[str, Any]], suggestion: bool) -> None:
        waiting = request is not None and request.get("status") == "pending"
        request_id = request.get("request_id") if waiting else None
        picture = self.suggestion_picture if suggestion else self.request_picture
        # Once drawn, a request's picture stays; one that had nothing to draw on
        # tries again, as the views take images (sim session 2026-10-16: none
        # came until the request was filed again).
        shown = self._shown_ids[suggestion]
        if request_id == shown and (request_id is None or picture is not None):
            return
        again = request_id == shown
        self._shown_ids[suggestion] = request_id
        if not again:
            self._picture(None, None, [], suggestion=suggestion)
        if not waiting:
            return
        cited = request.get("evidence") or {}
        if cited.get("image") and frame_path(cited) is None:
            # An image the agent made: shown as it is, its marks on it.
            self._evidence_only(cited, suggestion)
            return
        patterns = request_patterns(request)
        before, after = config_edit_stages(request, self._task_config)
        passed, kept, _points = decided_passes(request, self._pending_proposal)
        milled = request_stages(request) + after + passed
        live = _frame(self._frame_of, BeamType.ION)
        aspect = live.data.shape[0] / live.data.shape[1] if live is not None else 2 / 3
        outside = request_outside_field(request, aspect, milled)
        missing: List[str] = []
        marks = self._request_marks(request, missing)
        pass_title = ""
        # The agent's own image when it gave one; else the one the values were
        # measured on.
        evidence = _agent_image(self._load_image, request.get("image"))
        if evidence is not None:
            frame = evidence
            beam = frame.metadata.image_settings.beam_type or BeamType.ION
        else:
            frame, beam = self._measured_on(
                marks, bool(patterns or milled or before or kept)
            )
            if passed or kept or not marks:
                # A pass's patterns, or a decision with nothing to mark, on the
                # image it was asked on.
                found = self._pass_frame(request)
                if found is not None:
                    frame, beam, pass_title = found
            if not (patterns or milled or before or kept):
                frame, beam = self._pose_overview(marks, frame, beam)
        # The card: the image the values were measured on, every value on it.
        if frame is None:
            if missing:
                # Not the live view: an item's values are in its own frame.
                self._show_picture(
                    None,
                    f"No FIB image of {missing[0]} yet to draw on",
                    suggestion=suggestion,
                )
            elif marks:
                # Say so, rather than show nothing (headless round 12).
                mark = next(iter(marks.values()))
                beam_name = "SEM" if mark.beam is BeamType.ELECTRON else "FIB"
                self._show_picture(
                    None,
                    f"No {beam_name} image yet to draw the {mark.legend} on",
                    suggestion=suggestion,
                )
            return
        current = _Layer(CURRENT_COLOUR, dashed=True)
        milling = _Layer(stage_colour(0), fill=True)
        past = _Layer(PAST_FIELD_COLOUR)
        layers = [current, milling, past]
        if beam is BeamType.ION:
            current.patterns = _stage_patterns(before)
            # Past the field, outlined in red over the stage's colour.
            milling.add_stages(patterns, name="patterns")
            milling.add_stages(milled)
            past.patterns = outside
            current.key(DASHED, "current", current.patterns)
            if kept:
                # A pass decided without running: nothing is milled.
                unmilled = _Layer(UNMILLED_COLOUR)
                unmilled.patterns = _stage_patterns(kept)
                unmilled.key(OUTLINE, "not milled", unmilled.patterns)
                layers.append(unmilled)
            milling.key(FILL, "patterns", milling.patterns)
            _key_past_field(frame, past)
        layers.extend(
            self._add_marks(frame, [m for m in marks.values() if not m.before])
        )
        for mark in (m for m in marks.values() if m.before):
            self._add_mark(frame, mark, current)
        title = next(
            (m.title for m in marks.values() if m.title and m.places), pass_title
        )
        self._picture(
            frame,
            beam,
            layers,
            suggestion=suggestion,
            agent_image=evidence is not None,
            title="" if evidence is not None else title,
            marks=cited.get("marks") if evidence is not None else None,
        )

    def _evidence_only(self, evidence: Dict[str, Any], suggestion: bool) -> None:
        from fibsem.ai.agent.drawing import evidence_png

        png = evidence_png(evidence, _chip(suggestion))
        drawn = (evidence, []) if png is not None else None
        if suggestion:
            self.suggestion_picture = drawn
        else:
            self.request_picture = drawn
        self._show_picture(png, "The agent's image", suggestion=suggestion)

    def _pass_frame(self, request: Dict[str, Any]):
        """``(image, ION, what)``: the image the first pass a request decides
        was asked on, and what the call names; None when it has none."""
        for action in request.get("actions", []):
            if action.get("tool") != "proposal_decide":
                continue
            image = self._reference_image(action.get("arguments") or {})
            if _has_pixel_size(image):
                return image, BeamType.ION, str(action.get("what") or "")
        return None

    def _measured_on(self, marks: Dict[str, "_Mark"], milling: bool):
        """The frame the request's values were measured on: a decision's or an
        item's own image; else the FIB view's, where a mill draws; else the
        view of the beam a move is in. With its beam; None, None for none."""
        for mark in marks.values():
            for measured_on, _value in mark.places:
                if measured_on is not None:
                    return measured_on, mark.beam
        beams = [BeamType.ION] if milling else []
        beams += sorted({m.beam for m in marks.values()}, key=lambda b: b.value)
        if BeamType.ION in beams:
            return _frame(self._frame_of, BeamType.ION), BeamType.ION
        if beams:
            return _frame(self._frame_of, beams[0]), beams[0]
        return None, None

    def _pose_overview(self, marks: Dict[str, "_Mark"], frame, beam):
        """*frame* and *beam*; or, when *marks* are all poses and the view does
        not show them, the item's grid overview that does, the poses placed on
        it. A view or overview without stage geometry places nothing."""
        from fibsem.geometry.projection import stage_to_pixel

        posed = list(marks.values())
        if not posed or any(m.stage is None for m in posed):
            return frame, beam

        def shows(image) -> bool:
            for mark in posed:
                pixel = stage_to_pixel(image, mark.stage)
                if pixel is None or not _on_frame(image, [pixel, pixel]):
                    return False
            return True

        if frame is not None and shows(frame):
            return frame, beam
        overview = self._item_overview(posed[0].item)
        if overview is None or not shows(overview):
            return frame, beam
        for mark in posed:
            mark.places = [(overview, stage_to_pixel(overview, mark.stage))]
            mark.legend = f"{mark.legend}, on the grid overview"
        settings = getattr(overview.metadata, "image_settings", None)
        return overview, getattr(settings, "beam_type", None) or BeamType.ELECTRON

    def _request_marks(
        self, request: Dict[str, Any], missing: List[str]
    ) -> Dict[str, _Mark]:
        """What *request* places on the sample: move targets, a decision's or an
        item edit's point of interest, an item edit's alignment area. An item
        edit whose item has no FIB image to draw on goes in *missing*."""
        marks: Dict[str, _Mark] = {}

        def add(overlay_id, beam, legend, places, area=False, shape=CROSS, **kw):
            mark = marks.setdefault(
                overlay_id, _Mark(beam, legend, [], area, shape, kind=overlay_id, **kw)
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
            if tool == "proposal_decide":
                self._decision_marks(request, action, arguments, add)
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
                    if poi is None:
                        missing.append(item or "the lamella")
                        continue
                    # A new lamella has no image of its own yet: its point on
                    # the live FIB view, where it was added (FIB-1492).
                    add(
                        POI_MARK,
                        BeamType.ION,
                        "point of interest",
                        [(None, lambda frame, p=poi: _offset_pixel(frame, p))],
                        title=f"{item or 'the lamella'}, on the live view",
                    )
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

    def _decision_marks(self, request, action, arguments, add) -> None:
        """A decision's values on the image its proposal sits on: a point of
        interest, an alignment area, detected features, a spot burn's points."""
        values = arguments.get("values") or {}
        poi = _point(values.get("poi"))
        area = _area(values) if values.get("alignment_area") else None
        features = [
            point
            for feature in values.get("features") or []
            if isinstance(feature, dict) and (point := _point(feature.get("px")))
        ]
        _milled, _kept, points = decided_passes(
            {"actions": [action]}, self._pending_proposal
        )
        if poi is None and area is None and not features and not points:
            return
        image = self._reference_image(arguments)
        if not _has_pixel_size(image):
            return
        title = str(action.get("what") or "")
        if poi is not None:
            add(
                POI_MARK,
                BeamType.ION,
                "point of interest",
                [(image, _offset_pixel(image, poi))],
                title=title,
            )
        if area is not None:
            add(
                AREA_MARK,
                BeamType.ION,
                "alignment area",
                [(image, _fraction_pixel(image, corner)) for corner in area],
                area=True,
                title=title,
            )
        if features:
            add(
                FEATURE_MARK,
                BeamType.ION,
                "features",
                [(image, (p.x, p.y)) for p in features],
                title=title,
            )
        if points:
            add(
                SPOT_MARK,
                BeamType.ION,
                "spot burn points",
                [(image, _fraction_pixel(image, p)) for p in points],
                shape=DOT,
                title=title,
            )

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
                    item=item,
                    stage=stage,
                )
        elif tool == "item_set_pose":
            pose = arguments.get("pose")
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

    # --- drawing ----------------------------------------------------------------

    @classmethod
    def _add_marks(cls, frame, marks) -> List[_Layer]:
        """*marks* re-projected onto *frame*, each in its colour's layer."""
        layers: Dict[str, _Layer] = {}
        for mark in marks:
            colour = MARK_COLOURS.get(mark.kind, WHITE_ICON_COLOR)
            cls._add_mark(frame, mark, layers.setdefault(colour, _Layer(colour)))
        return list(layers.values())

    @staticmethod
    def _add_mark(frame, mark: "_Mark", layer: _Layer) -> None:
        """*mark* re-projected onto *frame*, into *layer*. A mark that falls
        off the picture is not drawn, and its key says so."""
        pixels = mark.pixels(frame)
        if mark.area:
            shown = len(pixels) >= 2 and _on_frame(frame, pixels[:2])
            if shown:
                layer.areas.append(pixels[:2])
            shape = AREA
        else:
            inside = [p for p in pixels if _on_frame(frame, [p, p])]
            layer.points(mark.shape).extend(inside)
            shown = bool(inside) or not pixels
            shape = mark.shape
        layer.key(shape, mark.legend if shown else f"{mark.legend}, off the picture")

    def _picture(
        self,
        frame,
        beam: Optional[BeamType],
        layers: List[_Layer],
        suggestion: bool = False,
        agent_image: bool = False,
        title: str = "",
        marks: Optional[Dict[str, Any]] = None,
    ) -> None:
        drawn = None
        # The agent's own image shows with nothing drawn on it, too.
        if frame is not None and (agent_image or any(layers)):
            drawn = (frame, layers)
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
        from fibsem.ai.agent.drawing import picture_png

        self._show_picture(
            picture_png(frame, layers, marks, chip=_chip(suggestion)),
            f"The agent's image · {caption}" if agent_image else caption,
            suggestion=suggestion,
        )
