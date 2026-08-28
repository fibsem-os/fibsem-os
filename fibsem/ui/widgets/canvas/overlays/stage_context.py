"""Where the stage can go, and where the sample sits, drawn in whatever view is up.

The context an overview is read against: which grid you are on, how far from its centre,
and how much travel is left. None of it depends on what is imaging the sample -- the
numbers come from `microscope._stage`, and turning them into canvas coordinates is the
`StageFrame`'s job, which both overview tabs already have.

They had a copy each, and the copies disagreed (FIB-698). The fluorescence one gated the
travel box on `stage_is_compustage` -- one guard answering two questions, since travel
limits have nothing to do with grids -- sized the grid boundary with `frame.length()` and
drew it as a circle, and handed raw slot positions to `frame.to_canvas`. Each is fixed
here by there being one implementation rather than by anyone fixing it twice.

Free functions over `(microscope, frame)` rather than a mixin: they read nothing else,
hold nothing, and a caller that wants one shape does not want a base class.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from fibsem.structures import FibsemStagePosition
from fibsem.ui.tokens import (
    GRID_BOUNDARY_COLOUR,
    SLOT_COLOUR,
    STAGE_LIMITS_COLOUR,
)
from fibsem.ui.widgets.canvas.overlays.minimap_overlays import (
    GRID_BOUNDARY_RADIUS_M,
    ShapeSpec,
)
from fibsem.ui.widgets.canvas.stage_frame import StageFrame

logger = logging.getLogger(__name__)

# Keys for `CanvasOverlayControls`, named here rather than in either tab: the switch and
# the shape it gates cannot drift apart under a rename, and both tabs offer the same
# three. What each tab draws *besides* these -- saved positions, grid bars -- is its own
# and keeps its keys there.
OVERLAY_LIMITS = "limits"
OVERLAY_BOUNDARIES = "boundaries"
OVERLAY_SLOTS = "slots"

# Label and default for each, so the two tabs cannot offer the same switch under
# different words.
#
# Grid boundaries and holder slots default **off**. Both describe a cryo sample holder --
# where its grids sit, and where its slots are -- so on a system without one they draw a
# holder that is not there, over an overview of a sample that is. Travel limits stay on:
# they are a property of the stage itself, and true on every system.
#
# Off for everyone rather than gated on the stage type, which is the narrower change: the
# switch is one click away in the overlays popover, and a default that read the hardware
# would have to be resolved per-microscope at construction rather than being the module
# constant both tabs share.
CONTEXT_OVERLAY_ENTRIES = (
    (OVERLAY_LIMITS, "Stage travel limits", True),
    (OVERLAY_BOUNDARIES, "Grid boundaries", False),
    (OVERLAY_SLOTS, "Holder slots", False),
)

__all__ = [
    "OVERLAY_LIMITS",
    "OVERLAY_BOUNDARIES",
    "OVERLAY_SLOTS",
    "CONTEXT_OVERLAY_ENTRIES",
    "landmark",
    "canvas_span",
    "holder_slots",
    "slot_landmark",
    "limit_shapes",
    "boundary_shapes",
    "slot_shapes",
    "context_shapes",
]


def landmark(
    frame: StageFrame, x: float, y: float, name: str = ""
) -> FibsemStagePosition:
    """A stage position that is a *place*, in the frame's own pose.

    Grid centre and the corners of the travel envelope are places, not recorded poses,
    so the rotation has to come from somewhere. Taking the frame's means
    `BeamStageProjection` sees no rotation difference and leaves the compucentric
    correction alone -- it is there to flip positions *recorded* half a turn away, and
    firing it on a synthetic landmark would move the travel limits by the instrument's
    compucentric calibration for no reason. Tilt is not read at all (the projection takes
    it from the base), and is carried only so the position is complete.
    """
    origin = frame.origin
    return FibsemStagePosition(name=name, x=x, y=y, z=0.0, r=origin.r, t=origin.t)


def canvas_span(frame: StageFrame, length: float) -> Tuple[float, float]:
    """A length *along the sample surface*, in canvas pixels, per axis.

    Not `frame.length()` twice. That divides by the canvas scale and stops, which is
    right for x and wrong for y: the view foreshortens the surface by a factor that
    changes with the beam and the pose -- 1.00 looking down the surface normal, 0.26 for
    the ion beam at the milling pose. A boundary sized by the scale alone is the same
    size in every view, and therefore right in at most one of them.

    **A surface length, not a stage-axis one.** Stepping stage y with no z is a move
    *through* a tilted surface rather than along it, and inflates every span by
    `1 / cos(pre_tilt)` -- exactly 1.000 at the pre-tilt of 0 that every Arctis and every
    test has, and 1.221 on a 35 degree shuttle, where a grid boundary came out 22% tall
    in the two views it must be a circle in (FIB-657). Stage x needs no such care: the
    tilt is about x, so stage x lies in the surface and a step along it is a step along
    the surface.

    Measured through the frame, not derived from the geometry again, so it cannot
    disagree with where the frame puts a marker. Absolute, because a view can flip an
    axis and a span has no sign.
    """
    ox, _ = frame.to_canvas(landmark(frame, 0.0, 0.0))
    along_x, _ = frame.to_canvas(landmark(frame, length, 0.0))
    span_x = abs(along_x - ox)
    return span_x, span_x * frame.surface_foreshortening()


def holder_slots(microscope) -> List[object]:
    """The sample holder's slots, or nothing if the stage does not describe one."""
    try:
        return list(microscope._stage.holder.slots.values())
    except Exception:
        return []


def slot_landmark(microscope, slot: object) -> Optional[FibsemStagePosition]:
    """Where a holder slot sits, as a position that can be drawn in any view.

    Slots are stored as x/y/z and **nothing else**: `default-sample-holder.yaml` gives
    each one three numbers, and `SampleHolder.load` leaves `r` and `t` as None. Handed to
    `frame.to_canvas` that raises -- and the fluorescence tab did exactly that, so a
    shipped two-slot shuttle would have drawn no slot markers at all. The simulator's
    holder hides it: `_ensure_slots` invents its slot with r=0.

    The missing rotation is the **SEM orientation**, which is the frame the holder file
    is written in. Stamped here rather than assumed away, because it is what makes a slot
    re-expressible: a shuttle's grids sit at x = -5 mm and +5 mm, and after a 180-degree
    rotation the raw coordinate that reaches a given grid is the *other* one. Carrying
    the pose lets `BeamStageProjection` see the difference and apply the compucentric
    flip, exactly as it does for a position recorded at one orientation and drawn at
    another.

    So **not** `landmark`, which takes the frame's own rotation precisely to stop that
    flip firing. A travel-envelope corner is a place in the frame being drawn; a slot is
    a place on the holder, and the holder turns over with the stage.

    No hardware: `get_orientation` is a lookup over the configured orientations. On a
    compustage every orientation shares one rotation, so this is the identity and the
    slots draw where they always did.
    """
    position = getattr(slot, "position", None)
    if position is None:
        return None
    try:
        pose = microscope.get_orientation("SEM")
    except Exception as e:
        logger.debug(f"Could not resolve the holder's reference orientation: {e}")
        return None
    return FibsemStagePosition(
        name=position.name or "",
        x=position.x,
        y=position.y,
        z=position.z or 0.0,
        r=pose.r,
        t=pose.t,
    )


def limit_shapes(microscope, frame: StageFrame) -> List[ShapeSpec]:
    """The stage's travel envelope, wherever limits are configured.

    Not gated on the stage type. The fluorescence copy gated this and the grid boundary
    together on `stage_is_compustage`, so a standard stage lost the travel box along with
    the grid circle -- and travel limits have nothing to do with grids. One guard was
    answering two questions.
    """
    limits = getattr(microscope._stage, "limits", None)
    if not limits:
        return []
    try:
        # Every corner, not a width and a height: the projection can flip either axis,
        # and reading the envelope off the extremes of the projected corners is true
        # whatever it does to them. The box stays axis-aligned -- the terms are a scale
        # per axis and a possible flip of both.
        corners = [
            frame.to_canvas(landmark(frame, x, y))
            for x in (limits["x"].min, limits["x"].max)
            for y in (limits["y"].min, limits["y"].max)
        ]
        xs = [point[0] for point in corners]
        ys = [point[1] for point in corners]
        width, height = max(xs) - min(xs), max(ys) - min(ys)
        box_cx, box_cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    except Exception as e:
        logger.debug(f"Could not draw the stage limits: {e}")
        return []
    return [
        ShapeSpec(
            kind="rect",
            cx=box_cx,
            cy=box_cy,
            width=width,
            height=height,
            color=STAGE_LIMITS_COLOUR,
            label="Stage Limits",
        ),
    ]


def boundary_shapes(microscope, frame: StageFrame) -> List[ShapeSpec]:
    """A grid boundary around every slot the holder carries.

    One per slot rather than one at the stage origin. A grid is 1 mm in radius whatever
    holds it, so what the boundary needs is *where the grids are* -- and the holder
    already says, in the same slot positions `slot_shapes` draws crosshairs at. A
    compustage carries a single slot at the origin, so it goes on drawing the one circle
    it always drew; a multi-grid shuttle gets one each, where before it got a single
    circle at a place no grid is.

    Placed exactly as the crosshair is, through `frame.to_canvas(slot_landmark(...))`, so
    the boundary and the marker at its centre cannot disagree: whatever the frame does to
    that position it does to both. Deliberately *not* re-derived through `landmark` --
    that would make the boundary a synthetic place and the crosshair a recorded one, and
    the two would part company on a stage where the compucentric correction fires.

    A circle on the sample, so an **ellipse** on screen everywhere but the two views
    where the beam looks down the pose it is named after.
    """
    specs: List[ShapeSpec] = []
    for slot in holder_slots(microscope):
        place = slot_landmark(microscope, slot)
        if place is None:
            continue
        try:
            cx, cy = frame.to_canvas(place)
            span_x, span_y = canvas_span(frame, GRID_BOUNDARY_RADIUS_M)
        except Exception as e:
            logger.debug(f"Could not draw a grid boundary: {e}")
            continue
        specs.append(
            ShapeSpec(
                kind="ellipse",
                cx=cx,
                cy=cy,
                width=2 * span_x,
                height=2 * span_y,
                color=GRID_BOUNDARY_COLOUR,
                label="Grid Boundary",
            )
        )
    return specs


def slot_shapes(microscope, frame: StageFrame) -> List[ShapeSpec]:
    """The sample holder's slots, as crosshairs at their configured positions.

    Through `slot_landmark` for the same reason the boundary is, and so the two stay
    concentric by construction rather than by both happening to be right.
    """
    specs: List[ShapeSpec] = []
    for slot in holder_slots(microscope):
        place = slot_landmark(microscope, slot)
        if place is None:
            continue
        try:
            cx, cy = frame.to_canvas(place)
        except Exception as e:
            logger.debug(f"Could not draw a holder slot: {e}")
            continue
        specs.append(
            ShapeSpec(
                kind="crosshair",
                cx=cx,
                cy=cy,
                color=SLOT_COLOUR,
                label=place.name or "",
            )
        )
    return specs


def context_shapes(
    microscope,
    frame: StageFrame,
    limits: bool = True,
    boundaries: bool = True,
    slots: bool = True,
) -> List[ShapeSpec]:
    """All three, in draw order, with each switchable.

    The flags exist because both tabs offer the same switches, so the caller passing
    three booleans is shorter than the caller assembling three lists -- and it puts the
    order in one place, where the boundary is drawn under the crosshair that marks its
    centre rather than over it.
    """
    specs: List[ShapeSpec] = []
    if limits:
        specs.extend(limit_shapes(microscope, frame))
    if boundaries:
        specs.extend(boundary_shapes(microscope, frame))
    if slots:
        specs.extend(slot_shapes(microscope, frame))
    return specs
