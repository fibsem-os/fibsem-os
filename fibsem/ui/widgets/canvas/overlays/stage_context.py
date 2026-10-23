"""Where the stage can go, and where the sample sits, drawn in whatever view is up.

The context an overview is read against: which grid you are on, how far from its centre,
and how much travel is left. None of it depends on what is imaging the sample -- the
numbers come from `microscope.sample_handling`, and turning them into canvas coordinates is the
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
from copy import deepcopy
from typing import List, Optional, Tuple

import numpy as np

from fibsem.geometry.frames import StageModel
from fibsem.geometry.projection import BeamStageProjection, lifted
from fibsem.geometry.stage_places import (
    BEAMS_DEVICE,
    holder_is_calibrated,
    holder_slots,
    slot_landmark,
)
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


def context_overlay_entries(microscope):
    """The overlay switches for this instrument, with the defaults resolved.

    Grid boundaries and holder slots come on when the holder has a calibrated
    slot -- then they draw where a grid is, which is what an overview is read
    against -- and stay off otherwise, where they would draw a holder that is
    not there. The module constant keeps the off defaults for callers with no
    instrument to ask.
    """
    calibrated = holder_is_calibrated(microscope)
    return [
        (
            key,
            label,
            calibrated if key in (OVERLAY_BOUNDARIES, OVERLAY_SLOTS) else shown,
        )
        for key, label, shown in CONTEXT_OVERLAY_ENTRIES
    ]


__all__ = [
    "OVERLAY_LIMITS",
    "OVERLAY_BOUNDARIES",
    "OVERLAY_SLOTS",
    "CONTEXT_OVERLAY_ENTRIES",
    "context_overlay_entries",
    "holder_is_calibrated",
    "landmark",
    "at_click_height",
    "click_height_key",
    "canvas_span",
    "holder_slots",
    "slot_landmark",
    "nearest_slot_landmark",
    "at_device",
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


def _same_pose(
    a: FibsemStagePosition, b: FibsemStagePosition, axes: Tuple[str, ...] = ("r", "t")
) -> bool:
    return all(
        getattr(a, axis) is not None
        and getattr(b, axis) is not None
        and np.isclose(getattr(a, axis), getattr(b, axis), atol=np.deg2rad(1.0))
        for axis in axes
    )


def at_click_height(
    frame: StageFrame,
    place: FibsemStagePosition,
    stage: Optional[FibsemStagePosition],
) -> FibsemStagePosition:
    """*place*, moved to where the canvas draws it so that a click there resolves to its
    x and y: on the sample surface through *stage*, then taken off the stage's lift
    (`lifted`), the inverse of how a click is resolved.

    Without it a place drawn at its own z slides in y wherever the view is tilted, by
    the height times the sine of the view's tilt: hundreds of microns on an Arctis at
    the milling pose. With the stage in another pose, which a click is refused in, the
    frame's origin stands in for it. A beam view only, and *place* only at the frame's
    rotation (a slot carries the SEM pose's tilt, which the projection ignores);
    otherwise *place* is returned as it is.
    """
    if (
        stage is None
        or not isinstance(frame.projection, BeamStageProjection)
        or not _same_pose(place, frame.origin, axes=("r",))
    ):
        return place
    if not _same_pose(stage, frame.origin):
        stage = frame.origin
    model = StageModel.from_geometry(frame.projection.geometry)
    along_y, along_z = model.in_plane_direction(stage.r or 0.0)
    place = deepcopy(place)
    place.z = (stage.z or 0.0) + ((place.y or 0.0) - (stage.y or 0.0)) * (
        along_z / along_y
    )
    return lifted(frame.projection, frame.origin, place, stage, sign=-1.0)


def click_height_key(
    frame: StageFrame, stage: Optional[FibsemStagePosition]
) -> Optional[Tuple[float, ...]]:
    """What `at_click_height` depends on in *stage*, to the nearest 0.1 um: its lift off
    the frame's plane and its pose. Unchanged by a move along the sample surface."""
    if stage is None or not isinstance(frame.projection, BeamStageProjection):
        return None
    lift = lifted(
        frame.projection,
        frame.origin,
        FibsemStagePosition(x=0.0, y=0.0, z=0.0),
        stage,
    )
    return tuple(
        round((value or 0.0) * 1e7)
        for value in (lift.x, lift.y, lift.z, stage.r, stage.t)
    )


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


def nearest_slot_landmark(
    microscope, frame: StageFrame, device: Optional[str] = None
) -> Optional[FibsemStagePosition]:
    """The holder slot drawn nearest the view's origin, placed as `boundary_shapes`
    places it, or None when the holder names no slot that can be drawn.

    The grid a view is looking at: its origin is the first image placed in it.
    """
    try:
        ox, oy = frame.to_canvas(frame.origin)
    except Exception as e:
        logger.debug(f"Could not place the view's origin: {e}")
        return None
    best, best_distance = None, None
    for slot in holder_slots(microscope):
        place = slot_landmark(microscope, slot)
        if place is None:
            continue
        place = at_device(microscope, device, place)
        try:
            cx, cy = frame.to_canvas(place)
        except Exception as e:
            logger.debug(f"Could not place a holder slot: {e}")
            continue
        distance = (cx - ox) ** 2 + (cy - oy) ** 2
        if best_distance is None or distance < best_distance:
            best, best_distance = place, distance
    return best


def at_device(
    microscope, device: Optional[str], place: FibsemStagePosition
) -> FibsemStagePosition:
    """*place*, carried to where *device* sees it.

    Holder slots are written in the beams' frame: x = 0 is the chamber origin, and the
    pose is the SEM one. The beam canvas draws them as they are. The fluorescence
    canvas is the FM's view of the sample, and everything it marks is a fluorescence
    pose -- at an offset FM that is ~48 mm away along x, in the pose the sample is
    imaged in -- so a raw slot handed to it lands 48 mm off the canvas, with the grid
    boundary around it. Keyed on the device the *canvas* is for, not on where the
    stage happens to be: the markers do not move when the stage does, and neither
    should the slot they sit in.

    Converted with `to_device`, the same call a lamella's fluorescence pose is derived
    with, so a lamella marked at a slot's centre draws on its crosshair rather than
    beside it. Nothing here asks about the mounting: a compustage takes the same call
    and gets its flip.

    A place that cannot be converted is drawn where it was written, and said out
    loud -- on an offset mount that is off this canvas altogether.
    """
    if device in (None, BEAMS_DEVICE):
        return place
    try:
        return microscope.to_device(place, device)
    except ValueError as e:
        logger.warning(
            f"Could not carry {place.name or 'a holder place'} to the {device} "
            f"device, so it is drawn where the beams see it: {e}"
        )
        return place


def limit_shapes(
    microscope, frame: StageFrame, stage: Optional[FibsemStagePosition] = None
) -> List[ShapeSpec]:
    """The stage's travel envelope, wherever limits are configured.

    Not gated on the stage type. The fluorescence copy gated this and the grid boundary
    together on `stage_is_compustage`, so a standard stage lost the travel box along with
    the grid circle -- and travel limits have nothing to do with grids. One guard was
    answering two questions.
    """
    device = microscope.stage
    limits = device.position.limits if device is not None else None
    if not limits:
        return []
    try:
        # Every corner, not a width and a height: the projection can flip either axis,
        # and reading the envelope off the extremes of the projected corners is true
        # whatever it does to them. The box stays axis-aligned -- the terms are a scale
        # per axis and a possible flip of both.
        corners = [
            frame.to_canvas(at_click_height(frame, landmark(frame, x, y), stage))
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


def boundary_shapes(
    microscope,
    frame: StageFrame,
    device: Optional[str] = None,
    stage: Optional[FibsemStagePosition] = None,
) -> List[ShapeSpec]:
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
        place = at_device(microscope, device, place)
        try:
            cx, cy = frame.to_canvas(at_click_height(frame, place, stage))
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


def slot_shapes(
    microscope,
    frame: StageFrame,
    device: Optional[str] = None,
    stage: Optional[FibsemStagePosition] = None,
) -> List[ShapeSpec]:
    """The sample holder's slots, as crosshairs at their configured positions.

    Through `slot_landmark` for the same reason the boundary is, and so the two stay
    concentric by construction rather than by both happening to be right.
    """
    specs: List[ShapeSpec] = []
    for slot in holder_slots(microscope):
        place = slot_landmark(microscope, slot)
        if place is None:
            continue
        place = at_device(microscope, device, place)
        try:
            cx, cy = frame.to_canvas(at_click_height(frame, place, stage))
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
    device: Optional[str] = None,
    stage: Optional[FibsemStagePosition] = None,
) -> List[ShapeSpec]:
    """All three, in draw order, with each switchable.

    The flags exist because both tabs offer the same switches, so the caller passing
    three booleans is shorter than the caller assembling three lists -- and it puts the
    order in one place, where the boundary is drawn under the crosshair that marks its
    centre rather than over it.

    `device` names the device the canvas is the view from -- see `at_device`. The
    beam canvas leaves it out; the fluorescence canvas passes "FM".

    `stage` draws each place where a click resolves to it (`at_click_height`).
    """
    specs: List[ShapeSpec] = []
    if limits:
        specs.extend(limit_shapes(microscope, frame, stage))
    if boundaries:
        specs.extend(boundary_shapes(microscope, frame, device, stage))
    if slots:
        specs.extend(slot_shapes(microscope, frame, device, stage))
    return specs
