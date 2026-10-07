"""A map of the stage from above: travel limits, holder slots, lamellae, the stage.

The chamber view's other half. The side view shows how the stage is tilted; this shows
where it is. Both are pictures, with no text of their own beyond slot names.

**Drawn in the Overview tab's frame, by the Overview tab's code.** Placement goes
through a :class:`StageFrame` and the `stage_context` helpers the overview canvases
use, so a slot, a grid boundary or the travel box lands exactly where it does there.
The frame is the electron beam's at the SEM orientation, so the map holds still while
the stage turns: a position recorded half a turn away -- at the FIB orientation on a
stage that rotates to reach it -- is flipped back by the projection's compucentric
correction rather than drawn on the far side of the grid. On a compustage nothing
rotates and nothing is flipped.

The scan rotation is taken as zero rather than read: building the projection the
Overview way reads it from the instrument, which on a Thermo system takes the imaging
channel. A map is about the stage, not about how the SEM scans it.

Nothing here reads the microscope's hardware. Limits, slots, orientations and the
geometry come from its configuration; positions are handed in.

**No holder outline.** A holder's configuration states its slots and nothing about its
shape, so the map draws the slots and the grid boundary around each, and not an
outline it would have to invent.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Sequence, Tuple

from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import QWidget

from fibsem.projection import BeamStageProjection
from fibsem.structures import BeamType, FibsemStagePosition
from fibsem.ui.tokens import (
    CANVAS_BG,
    CURRENT_POSITION_COLOUR,
    GRID_BOUNDARY_COLOUR,
    NEUTRAL_550,
    SAVED_POSITION_COLOUR,
    SELECTED_POSITION_COLOUR,
    SLOT_COLOUR,
    STAGE_LIMITS_COLOUR,
)
from fibsem.ui.widgets.canvas.overlays.minimap_overlays import GRID_BOUNDARY_RADIUS_M
from fibsem.ui.widgets.canvas.overlays.stage_context import (
    boundary_shapes,
    holder_is_calibrated,
    holder_slots,
    landmark,
    limit_shapes,
    slot_landmark,
)
from fibsem.ui.widgets.canvas.stage_frame import StageFrame

logger = logging.getLogger(__name__)

ZOOM_TRAVEL, ZOOM_HOLDER, ZOOM_GRID = 0, 1, 2
ZOOM_NAMES = ("Travel", "Holder", "Grid")

# Room around what a zoom step fits, as a fraction of it.
_MARGIN = 0.12
# The holder step's span when there is one slot or none to fit: a few grids across.
_HOLDER_MIN_SPAN_M = 8e-3
# The grid step's span: the grid and a little around it.
_GRID_SPAN_M = 2.6 * GRID_BOUNDARY_RADIUS_M
# The travel step never shows more than this across: a 200 mm envelope at thumbnail
# size leaves a grid a pixel wide, and the stage rarely goes that far.
_TRAVEL_MAX_SPAN_M = 100e-3
# A lamella the stage is within this distance of (in the map's plane) is "here".
_HERE_M = 20e-6
# The ring drawn round the lamella the stage is on, over the stage cross so it shows.
_HERE_RING_PX = 6.0


class _Viewport:
    """The two calls :class:`StageFrame` makes of a canvas, for a painted map.

    A scale and nothing else, as on `FibsemRealSpaceCanvas`, whose conversions are a
    pixel size with no offset: panning is the axes' business there, and here it is the
    painter's, translated before drawing. An offset folded in here would be folded
    into `StageFrame.length` too, which would then stop being a length.
    """

    def __init__(self, scale: float):
        self._scale = scale

    def metres_to_canvas(self, x: float, y: float) -> Tuple[float, float]:
        return x * self._scale, y * self._scale

    def canvas_to_metres(self, x: float, y: float) -> Tuple[float, float]:
        return x / self._scale, y / self._scale


class StageMap(QWidget):
    """The stage from above, at one of three zoom steps, following the stage."""

    zoom_changed = pyqtSignal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._microscope = None
        self._projection: Optional[BeamStageProjection] = None
        self._origin: Optional[FibsemStagePosition] = None
        self._stage: Optional[FibsemStagePosition] = None
        self._positions: List[FibsemStagePosition] = []
        self.zoom = ZOOM_TRAVEL

    # ── inputs ────────────────────────────────────────────────────────────
    def set_microscope(self, microscope) -> None:
        """Take the configuration the map is drawn from. No-op for the same one."""
        if microscope is self._microscope:
            return
        self._microscope = microscope
        self._projection = None
        self._origin = None
        try:
            self._projection = BeamStageProjection(
                geometry=microscope.hardware_geometry(),
                beam_type=BeamType.ELECTRON,
                scan_rotation=0.0,
            )
            sem = microscope.get_orientation("SEM")
            # Fixed once, as a frame's origin must be. z is taken from the first
            # position the map is given (see `set_stage`): the electron view of a
            # pre-tilted stage moves with z, and an origin a few millimetres off in z
            # would shift everything by a comparable amount in y.
            self._origin = FibsemStagePosition(x=0.0, y=0.0, z=None, r=sem.r, t=sem.t)
        except Exception:
            logger.debug("The stage map could not read its geometry", exc_info=True)
        self.update()

    def set_stage(self, position: Optional[FibsemStagePosition]) -> None:
        if position is None:
            return
        self._stage = position
        if self._origin is not None and self._origin.z is None:
            self._origin.z = position.z or 0.0
        self.update()

    def set_positions(self, positions: Sequence[FibsemStagePosition]) -> None:
        """The lamellae to mark, at grid zoom."""
        self._positions = [p for p in positions if p is not None]
        self.update()

    # ── zoom ──────────────────────────────────────────────────────────────
    def set_zoom(self, zoom: int) -> None:
        zoom = max(ZOOM_TRAVEL, min(ZOOM_GRID, int(zoom)))
        if zoom != self.zoom:
            self.zoom = zoom
            self.zoom_changed.emit(zoom)
            self.update()

    def zoom_in(self) -> None:
        self.set_zoom(self.zoom + 1)

    def zoom_out(self) -> None:
        self.set_zoom(self.zoom - 1)

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.angleDelta().y() > 0:
            self.zoom_in()
        elif event.angleDelta().y() < 0:
            self.zoom_out()
        event.accept()

    # ── geometry ──────────────────────────────────────────────────────────
    def _plane(self, position: FibsemStagePosition) -> Tuple[float, float]:
        return self._projection.to_plane(position, self._origin)

    def _travel_box(self) -> Optional[Tuple[float, float, float, float]]:
        """The travel limits in the plane, as (min x, min y, max x, max y)."""
        limits = getattr(getattr(self._microscope, "_stage", None), "limits", None)
        if not limits or "x" not in limits or "y" not in limits:
            return None
        frame = StageFrame(_Viewport(1.0), self._origin, self._projection)
        points = [
            self._plane(landmark(frame, x, y))
            for x in (limits["x"].min, limits["x"].max)
            for y in (limits["y"].min, limits["y"].max)
        ]
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        return min(xs), min(ys), max(xs), max(ys)

    def _slot_points(self) -> List[Tuple[float, float]]:
        if not holder_is_calibrated(self._microscope):
            return []
        points = []
        for slot in holder_slots(self._microscope):
            place = slot_landmark(self._microscope, slot)
            if place is not None:
                points.append(self._plane(place))
        return points

    def _view(
        self, rect: QRectF, zoom: int
    ) -> Optional[Tuple[Tuple[float, float], float]]:
        """The plane centre and the scale a zoom step shows *rect* at."""
        if self._projection is None or self._origin is None or self._origin.z is None:
            return None
        width, height = max(rect.width(), 1.0), max(rect.height(), 1.0)
        stage = self._plane(self._stage) if self._stage is not None else (0.0, 0.0)
        if zoom == ZOOM_TRAVEL:
            return self._travel_view(width, height, stage)
        if zoom == ZOOM_HOLDER:
            points = self._slot_points()
            span = _HOLDER_MIN_SPAN_M
            if points:
                xs, ys = [p[0] for p in points], [p[1] for p in points]
                extent = max(max(xs) - min(xs), max(ys) - min(ys))
                span = max(span, (extent + 4 * GRID_BOUNDARY_RADIUS_M) * (1 + _MARGIN))
            return stage, min(width, height) / span
        return stage, min(width, height) / _GRID_SPAN_M

    def _travel_view(
        self, width: float, height: float, stage: Tuple[float, float]
    ) -> Tuple[Tuple[float, float], float]:
        """The travel step: the limits, every grid and the stage, at most
        `_TRAVEL_MAX_SPAN_M` across.

        The grids are fitted as well as the limits because a compustage's limits can
        sit inside its grid, and fitting the limits alone cropped the grid. Past the
        cap the view is centred on what it fits, then shifted just enough to keep the
        stage in it.
        """
        points = [stage]
        for x, y in self._slot_points():
            points += [
                (x - GRID_BOUNDARY_RADIUS_M, y - GRID_BOUNDARY_RADIUS_M),
                (x + GRID_BOUNDARY_RADIUS_M, y + GRID_BOUNDARY_RADIUS_M),
            ]
        box = self._travel_box()
        if box is not None:
            points += [(box[0], box[1]), (box[2], box[3])]
        else:
            # No limits configured: room around the slots and the stage instead.
            pad = _HOLDER_MIN_SPAN_M
            points += [(x + d, y + d) for x, y in list(points) for d in (-pad, pad)]
        xs, ys = [p[0] for p in points], [p[1] for p in points]

        span_x = min((max(xs) - min(xs)) * (1 + _MARGIN), _TRAVEL_MAX_SPAN_M)
        span_y = min((max(ys) - min(ys)) * (1 + _MARGIN), _TRAVEL_MAX_SPAN_M)
        scale = min(width / span_x, height / span_y)
        # The cap holds along the view's long side too, not only the side that fits.
        scale = max(scale, max(width, height) / _TRAVEL_MAX_SPAN_M)

        # Then the window that scale actually shows, slid towards the stage until the
        # stage is inside it, short of the edge. A no-op whenever nothing was capped.
        centre = []
        for low, high, at, shown in (
            (min(xs), max(xs), stage[0], width / scale),
            (min(ys), max(ys), stage[1], height / scale),
        ):
            reach = shown / 2 * (1 - _MARGIN)
            centre.append(min(max((low + high) / 2, at - reach), at + reach))
        return (centre[0], centre[1]), scale

    # ── drawing ───────────────────────────────────────────────────────────
    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        try:
            self.paint_map(painter, QRectF(self.rect()), self.zoom, detailed=True)
        finally:
            painter.end()

    def paint_map(
        self, painter: QPainter, rect: QRectF, zoom: int, detailed: bool
    ) -> None:
        """Draw the map into *rect*. *detailed* adds slot names and lamellae, which a
        thumbnail is too small to carry."""
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setClipRect(rect)
        painter.fillRect(rect, QColor(CANVAS_BG))
        view = self._view(rect, zoom)
        if view is None:
            painter.restore()
            return
        (centre_x, centre_y), scale = view
        frame = StageFrame(_Viewport(scale), self._origin, self._projection)
        # Centre the view on the rectangle: the frame's coordinates are a scale only.
        painter.translate(
            rect.center().x() - centre_x * scale, rect.center().y() - centre_y * scale
        )
        try:
            self._paint_limits(painter, frame)
            self._paint_slots(painter, frame, zoom, detailed)
            if detailed and zoom == ZOOM_GRID:
                self._paint_positions(painter, frame)
            self._paint_stage(painter, frame, detailed)
            if detailed and zoom == ZOOM_GRID:
                self._paint_here(painter, frame)
        except Exception:
            logger.debug("The stage map could not be drawn", exc_info=True)
        painter.restore()

    def _paint_limits(self, painter: QPainter, frame: StageFrame) -> None:
        for shape in limit_shapes(self._microscope, frame):
            colour = QColor(STAGE_LIMITS_COLOUR)
            colour.setAlphaF(0.5)
            painter.setPen(QPen(colour, 1, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(
                QRectF(
                    shape.cx - shape.width / 2,
                    shape.cy - shape.height / 2,
                    shape.width,
                    shape.height,
                )
            )

    def _paint_slots(
        self, painter: QPainter, frame: StageFrame, zoom: int, detailed: bool
    ) -> None:
        if not holder_is_calibrated(self._microscope):
            return
        boundary = QColor(GRID_BOUNDARY_COLOUR)
        boundary.setAlphaF(0.55)
        for shape in boundary_shapes(self._microscope, frame):
            painter.setPen(QPen(boundary, 1, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(
                QPointF(shape.cx, shape.cy),
                max(shape.width / 2, 1.5),
                max(shape.height / 2, 1.5),
            )
        painter.setPen(QPen(QColor(SLOT_COLOUR), 1))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        for slot in holder_slots(self._microscope):
            place = slot_landmark(self._microscope, slot)
            if place is None:
                continue
            x, y = frame.to_canvas(place)
            arm = 4.0 if detailed else 2.0
            painter.drawLine(QPointF(x - arm, y), QPointF(x + arm, y))
            painter.drawLine(QPointF(x, y - arm), QPointF(x, y + arm))
            # At holder zoom only: at grid zoom the slot is the one the stage is on,
            # and its name would sit off the edge of the view.
            if detailed and zoom == ZOOM_HOLDER and place.name:
                radius = frame.length(GRID_BOUNDARY_RADIUS_M)
                painter.setPen(QColor(NEUTRAL_550))
                painter.drawText(QPointF(x - radius, y - radius - 6), place.name)
                painter.setPen(QPen(QColor(SLOT_COLOUR), 1))

    def lamellae_here(self) -> List[FibsemStagePosition]:
        """The marked positions the stage is on."""
        if self._stage is None:
            return []
        stage = self._plane(self._stage)
        here = []
        for position in self._positions:
            try:
                if math.dist(self._plane(position), stage) < _HERE_M:
                    here.append(position)
            except Exception:
                continue
        return here

    def _paint_positions(self, painter: QPainter, frame: StageFrame) -> None:
        """Lamellae as hairline crosshairs."""
        colour = QColor(SAVED_POSITION_COLOUR)
        colour.setAlphaF(0.65)
        painter.setPen(QPen(colour, 1))
        for position in self._positions:
            try:
                x, y = frame.to_canvas(position)
            except Exception:
                continue
            painter.drawLine(QPointF(x - 3.5, y), QPointF(x + 3.5, y))
            painter.drawLine(QPointF(x, y - 3.5), QPointF(x, y + 3.5))

    def _paint_here(self, painter: QPainter, frame: StageFrame) -> None:
        """A ring round the lamella the stage is on, drawn over the stage cross: a
        crosshair there was hidden under it."""
        colour = QColor(SELECTED_POSITION_COLOUR)
        colour.setAlphaF(0.9)
        painter.setPen(QPen(colour, 1.2))
        painter.setBrush(Qt.NoBrush)
        for position in self.lamellae_here():
            x, y = frame.to_canvas(position)
            painter.drawEllipse(QPointF(x, y), _HERE_RING_PX, _HERE_RING_PX)

    def _paint_stage(
        self, painter: QPainter, frame: StageFrame, detailed: bool
    ) -> None:
        if self._stage is None:
            return
        x, y = frame.to_canvas(self._stage)
        arm = 7.0 if detailed else 4.0
        colour = QColor(CURRENT_POSITION_COLOUR)
        painter.setPen(QPen(colour, 1.5 if detailed else 1.0))
        painter.drawLine(QPointF(x - arm, y), QPointF(x + arm, y))
        painter.drawLine(QPointF(x, y - arm), QPointF(x, y + arm))
        # The rotation, relative to the frame's: a tick from the centre of the cross.
        if self._stage.r is not None and self._origin.r is not None:
            turn = self._stage.r - self._origin.r
            tick = 16.0 if detailed else 8.0
            colour.setAlphaF(0.8)
            painter.setPen(QPen(colour, 1))
            painter.drawLine(
                QPointF(x, y),
                QPointF(x + math.sin(turn) * tick, y - math.cos(turn) * tick),
            )
