"""The places on a stage map: the holder's slots and their grids, the travel limits,
the device stations, the lamellae and the stage, in the plane the map is drawn in.

The arithmetic the Qt chamber view's map (`fibsem.ui.widgets.canvas.stage_map`) and
the web app's stage card both draw from, with no Qt in it, so the server can answer it
and the two drawings cannot disagree about where anything is. Nothing here reads the
microscope's hardware: limits, slots, orientations and the geometry are configuration,
and positions are handed in.

**The plane** is the electron beam's at the SEM orientation, as the Overview tab's,
with the scan rotation taken as zero: a map is about the stage, not about how the SEM
scans it. Metres, +x right, +y down the screen, about an origin at x = y = 0 in the SEM
pose whose z the caller fixes once.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from fibsem.geometry.angles import wrap_degrees
from fibsem.geometry.projection import BeamStageProjection, surface_foreshortening
from fibsem.structures.settings import BeamType, FibsemStagePosition

logger = logging.getLogger(__name__)

# The specimen grid's radius in metres, for the boundary drawn around each slot.
#
# A constant only because nothing currently reports it; `SampleGrid.radius` would
# replace it: a holder knows how big its grids are, and a fixed 1 mm is right for the
# common case and wrong for anything else.
GRID_BOUNDARY_RADIUS_M = 1000e-6

# The device holder places are written against.
BEAMS_DEVICE = "FIBSEM"

# The holder, drawn as a square around its slots: their extent padded by this much,
# the grid's 1 mm radius and 2 mm of clearance. Holder files state slots, not an
# outline, so this is schematic, not measured.
HOLDER_PAD_M = 3e-3

# A lamella the stage is within this distance of (in the map's plane) is "here".
HERE_M = 20e-6

Point = Tuple[float, float]


def holder_slots(microscope) -> List[object]:
    """The sample holder's slots, or nothing if the stage does not describe one."""
    try:
        return list(microscope.sample_handling.holder.slots.values())
    except Exception:
        return []


def holder_is_calibrated(microscope) -> bool:
    """Whether the holder has a slot with a trusted position: the one case in
    which the grid boundary and the slot marker describe something real."""
    return any(
        getattr(slot, "is_calibrated", False) for slot in holder_slots(microscope)
    )


def slot_landmark(microscope, slot: object) -> Optional[FibsemStagePosition]:
    """Where a holder slot sits, as a position that can be drawn in any view.

    A slot position may carry x/y/z and **nothing else**: a holder file states three
    numbers per slot and `SampleHolder.load` leaves `r` and `t` as None. Handed to a
    projection that raises.

    The missing rotation is the **SEM orientation**, which is the frame the holder file
    is written in. Stamped here rather than assumed away, because it is what makes a slot
    re-expressible: a shuttle's grids sit at x = -5 mm and +5 mm, and after a 180-degree
    rotation the raw coordinate that reaches a given grid is the *other* one. Carrying
    the pose lets `BeamStageProjection` see the difference and apply the compucentric
    flip, exactly as it does for a position recorded at one orientation and drawn at
    another. On a compustage every orientation shares one rotation, so this is the
    identity.

    No hardware: `get_orientation` is a lookup over the configured orientations.
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


def is_half_turn(r_radians: float, rotation_reference_degrees: float) -> bool:
    """Whether the stage is turned more than a quarter turn from the reference.

    That is what mirrors the side view, so it is read from the rotation itself rather
    than from the orientation's name: a compustage has no rotation axis and reaches
    FIB by tilting, and a stage between orientations still has a rotation to draw.

    A side view can only be honest at the reference or a half turn from it; anywhere
    else the sample tilts out of the plane of the columns. The stage is at one or the
    other in practice, so in between it snaps to the nearer rather than projecting.
    """
    delta = math.degrees(r_radians) - rotation_reference_degrees
    delta = wrap_degrees(delta)
    return abs(delta) > 90.0


class StagePlane:
    """Stage positions on the map's plane, about an origin fixed once.

    Built from the microscope's configuration; None from `build` when it cannot be read.
    *z* is the origin's height: the electron view of a pre-tilted stage moves with z,
    and an origin a few millimetres off in z shifts what is placed through it by a
    comparable amount in y, so the caller fixes it from a real position.
    """

    def __init__(self, projection: BeamStageProjection, origin: FibsemStagePosition):
        self.projection = projection
        self.origin = origin

    @classmethod
    def build(cls, microscope, z: float) -> Optional["StagePlane"]:
        try:
            projection = BeamStageProjection(
                geometry=microscope.hardware_geometry(),
                beam_type=BeamType.ELECTRON,
                scan_rotation=0.0,
            )
            sem = microscope.get_orientation("SEM")
        except Exception:
            logger.debug("The stage map could not read its geometry", exc_info=True)
            return None
        return cls(projection, FibsemStagePosition(x=0.0, y=0.0, z=z, r=sem.r, t=sem.t))

    def point(self, position: FibsemStagePosition) -> Point:
        return self.projection.to_plane(position, self.origin)

    def landmark(self, x: float, y: float) -> FibsemStagePosition:
        """A place in the plane's own pose (a corner of the travel envelope), so the
        projection sees no rotation difference and leaves the compucentric correction
        alone."""
        return FibsemStagePosition(x=x, y=y, z=0.0, r=self.origin.r, t=self.origin.t)

    def surface_scale(self) -> Point:
        """How a length along the sample surface comes out on the plane, per axis:
        1 across, foreshortened down."""
        return 1.0, surface_foreshortening(self.projection, self.origin)


def slot_points(microscope, plane: StagePlane) -> List[Tuple[str, Point]]:
    """The calibrated holder's slots on the plane, named; none on a holder with no
    calibrated slot, where they would draw a holder that is not there."""
    if not holder_is_calibrated(microscope):
        return []
    points = []
    for slot in holder_slots(microscope):
        place = slot_landmark(microscope, slot)
        if place is not None:
            points.append((place.name or getattr(slot, "name", ""), plane.point(place)))
    return points


def holder_square(points: Sequence[Point]) -> Optional[Tuple[Point, float]]:
    """The holder's square on the plane around the slot *points*: its centre and side,
    in metres; None with no slot."""
    if not points:
        return None
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    side = 2 * HOLDER_PAD_M + max(max(xs) - min(xs), max(ys) - min(ys))
    return ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2), side


def device_stations(
    microscope, plane: StagePlane, rotation: Optional[float]
) -> List[Tuple[str, FibsemStagePosition]]:
    """The places the stage travels to for a device other than the beams: an offset
    fluorescence microscope's station, say. Not the beams' own, which is where the
    holder already is, and not a device that sits at the beams' origin (an Arctis FM),
    which has nowhere separate to draw.

    A station is a place in the chamber, so it is stated at the stage's *rotation*: the
    map is the holder's frame, and with the stage turned a half turn the station is on
    the other side of the holder.
    """
    try:
        devices = microscope.system.stage.devices
        beams = microscope.get_device_origin(BEAMS_DEVICE)
    except Exception:
        return []
    if rotation is None:
        rotation = plane.origin.r
    stations = []
    for name in devices:
        if name == BEAMS_DEVICE:
            continue
        try:
            origin = microscope.get_device_origin(name)
        except Exception:
            continue
        x = origin.x if origin.x is not None else beams.x
        y = origin.y if origin.y is not None else beams.y
        if x is None or y is None or (x == beams.x and y == beams.y):
            continue
        stations.append(
            (
                name,
                FibsemStagePosition(
                    x=x, y=y, z=plane.origin.z, r=rotation, t=plane.origin.t
                ),
            )
        )
    return stations


def limit_corners(microscope, plane: StagePlane) -> Optional[Tuple[Point, Point]]:
    """The stage's travel envelope on the plane, as its (left, top) and (right, bottom)
    corners; None where no limits are configured. Every corner is projected, since the
    projection can flip either axis."""
    device = microscope.stage
    limits = device.position.limits if device is not None else None
    if not limits:
        return None
    try:
        corners = [
            plane.point(plane.landmark(x, y))
            for x in (limits["x"].min, limits["x"].max)
            for y in (limits["y"].min, limits["y"].max)
        ]
    except Exception as e:
        logger.debug(f"Could not place the stage limits: {e}")
        return None
    xs, ys = [p[0] for p in corners], [p[1] for p in corners]
    return (min(xs), min(ys)), (max(xs), max(ys))


@dataclass
class MapPlace:
    name: str
    x: float
    y: float


@dataclass
class StageMapPlaces:
    """What a stage map draws, on the plane (see the module's docstring)."""

    # the stage's point; None without a position
    stage: Optional[List[float]] = None
    # the travel envelope as [left, top, right, bottom]
    limits: Optional[List[float]] = None
    slots: List[MapPlace] = field(default_factory=list)
    # a grid boundary's radii on the plane, across and down
    boundary: List[float] = field(default_factory=list)
    # the holder's square, as [centre x, centre y, side]
    holder: Optional[List[float]] = None
    stations: List[MapPlace] = field(default_factory=list)
    positions: List[MapPlace] = field(default_factory=list)
    # the names of the positions the stage is on
    here: List[str] = field(default_factory=list)
    # the slot whose grid the stage is over
    slot: Optional[str] = None


def stage_map(
    microscope,
    stage: Optional[FibsemStagePosition],
    positions: Sequence[FibsemStagePosition] = (),
    z: Optional[float] = None,
) -> Optional[StageMapPlaces]:
    """The map with the stage at *stage* and *positions* marked. The origin's height
    is *z*, else the stage's. None when the geometry cannot be read."""
    if z is None:
        z = (stage.z if stage is not None else None) or 0.0
    plane = StagePlane.build(microscope, z)
    if plane is None:
        return None
    out = StageMapPlaces()
    slots = slot_points(microscope, plane)
    out.slots = [MapPlace(name, x, y) for name, (x, y) in slots]
    across, down = plane.surface_scale()
    out.boundary = [GRID_BOUNDARY_RADIUS_M * across, GRID_BOUNDARY_RADIUS_M * down]
    square = holder_square([p for _, p in slots])
    if square is not None:
        (cx, cy), side = square
        out.holder = [cx, cy, side]
    corners = limit_corners(microscope, plane)
    if corners is not None:
        (left, top), (right, bottom) = corners
        out.limits = [left, top, right, bottom]
    rotation = stage.r if stage is not None else None
    for name, place in device_stations(microscope, plane, rotation):
        x, y = plane.point(place)
        out.stations.append(MapPlace(name, x, y))
    for position in positions:
        if position is None:
            continue
        try:
            x, y = plane.point(position)
        except Exception:
            continue
        out.positions.append(MapPlace(position.name or "", x, y))
    if stage is not None:
        try:
            sx, sy = plane.point(stage)
        except Exception:
            return out
        out.stage = [sx, sy]
        out.here = [
            p.name for p in out.positions if math.dist((p.x, p.y), (sx, sy)) < HERE_M
        ]
        rx, ry = out.boundary
        for place in out.slots:
            if ((place.x - sx) / rx) ** 2 + ((place.y - sy) / ry) ** 2 <= 1:
                out.slot = place.name
                break
    return out
