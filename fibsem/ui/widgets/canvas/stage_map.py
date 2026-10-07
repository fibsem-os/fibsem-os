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
from PyQt5.QtGui import QColor, QPainter, QPen, QPolygonF
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
    BEAMS_DEVICE,
    boundary_shapes,
    holder_is_calibrated,
    holder_slots,
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
# Nor less than this: a single grid would otherwise fill the travel step, and it would
# repeat the holder step rather than say where in the chamber the stage is.
_TRAVEL_MIN_SPAN_M = 20e-3
# How far inside the view's edge an off-screen place's arrow sits, in pixels.
_OFFSCREEN_INSET_PX = 12.0
# Arrows that land closer than this share one, with both names.
_MERGE_PX = 30.0
# The diamond marking a device station, in pixels from centre to corner.
_STATION_PX = (6.0, 3.0)  # detailed, thumbnail
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

    def _device_stations(self) -> List[Tuple[str, FibsemStagePosition]]:
        """The places the stage travels to for a device other than the beams: an offset
        fluorescence microscope's station, say. Not the beams' own, which is where the
        holder already is, and not a device that sits at the beams' origin (an Arctis
        FM), which has nowhere separate to draw.

        A station is a place in the chamber, so it is stated at the stage's current
        rotation: the map is the holder's frame, and with the stage turned a half turn
        the station is on the other side of the holder. Drawn at the frame's rotation
        instead, a stage at the station in its FIB pose would sit 100 mm from it.
        """
        try:
            devices = self._microscope.system.stage.devices
            beams = self._microscope.get_device_origin(BEAMS_DEVICE)
        except Exception:
            return []
        rotation = self._stage.r if self._stage is not None else self._origin.r
        stations = []
        for name in devices:
            if name == BEAMS_DEVICE:
                continue
            try:
                origin = self._microscope.get_device_origin(name)
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
                        x=x, y=y, z=self._origin.z, r=rotation, t=self._origin.t
                    ),
                )
            )
        return stations

    def _travel_view(
        self, width: float, height: float, stage: Tuple[float, float]
    ) -> Tuple[Tuple[float, float], float]:
        """The travel step: the places the stage goes -- every grid, every device
        station -- and the stage itself, between `_TRAVEL_MIN_SPAN_M` and
        `_TRAVEL_MAX_SPAN_M` across.

        Not the travel limits. They say how far the stage *could* go, and fitted they
        left half the view as empty chamber; they are still drawn, as an edge where
        they fall inside. A stage out at a load position stretches the view to reach
        it, up to the cap; past the cap the view slides to keep the stage in it.
        """
        points = [stage]
        for x, y in self._slot_points():
            points += [
                (x - GRID_BOUNDARY_RADIUS_M, y - GRID_BOUNDARY_RADIUS_M),
                (x + GRID_BOUNDARY_RADIUS_M, y + GRID_BOUNDARY_RADIUS_M),
            ]
        points += [self._plane(place) for _, place in self._device_stations()]
        xs, ys = [p[0] for p in points], [p[1] for p in points]

        def span(low: float, high: float) -> float:
            fitted = (high - low) * (1 + _MARGIN)
            return min(max(fitted, _TRAVEL_MIN_SPAN_M), _TRAVEL_MAX_SPAN_M)

        span_x, span_y = span(min(xs), max(xs)), span(min(ys), max(ys))
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
        """Draw the map into *rect*. *detailed* is the large map: names, edge arrows
        and lamella crosshairs, which a thumbnail is too small to carry -- it shows the
        lamellae, at grid zoom, as dots."""
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
        shift_x = rect.center().x() - centre_x * scale
        shift_y = rect.center().y() - centre_y * scale
        painter.translate(shift_x, shift_y)
        # The rectangle on screen, in the frame's coordinates.
        visible = rect.translated(-shift_x, -shift_y)
        try:
            self._paint_limits(painter, frame)
            self._paint_stations(painter, frame, zoom, detailed, visible)
            self._paint_slots(painter, frame, zoom, detailed)
            if zoom == ZOOM_GRID:
                self._paint_positions(painter, frame, detailed)
            self._paint_stage(painter, frame, detailed)
            if zoom == ZOOM_GRID:
                self._paint_here(painter, frame, detailed)
            if detailed:
                self._paint_offscreen(painter, rect, zoom, scale, visible)
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

    def _paint_stations(
        self,
        painter: QPainter,
        frame: StageFrame,
        zoom: int,
        detailed: bool,
        visible: QRectF,
    ) -> None:
        """Device stations as diamonds, named on the large map."""
        size = _STATION_PX[0] if detailed else _STATION_PX[1]
        painter.setBrush(Qt.NoBrush)
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        for name, place in self._device_stations():
            x, y = frame.to_canvas(place)
            painter.setPen(QPen(QColor(SLOT_COLOUR), 1))
            painter.drawPolygon(
                QPolygonF(
                    [
                        QPointF(x, y - size),
                        QPointF(x + size, y),
                        QPointF(x, y + size),
                        QPointF(x - size, y),
                    ]
                )
            )
            if detailed and zoom != ZOOM_GRID:
                painter.setPen(QColor(NEUTRAL_550))
                # Right of the diamond, unless that runs off the view.
                text_width = painter.fontMetrics().horizontalAdvance(name)
                left = x + size + 4
                if left + text_width > visible.right() - 4:
                    left = x - size - 4 - text_width
                painter.drawText(QPointF(left, y + 4), name)

    def offscreen_places(
        self, rect: QRectF, zoom: int
    ) -> List[Tuple[str, Tuple[float, float]]]:
        """The named places -- device stations, holder slots -- outside what *zoom*
        shows of *rect*, as (name, point in the plane)."""
        view = self._view(rect, zoom)
        if view is None:
            return []
        (centre_x, centre_y), scale = view
        inset = _OFFSCREEN_INSET_PX / scale
        half_w = rect.width() / 2 / scale - inset
        half_h = rect.height() / 2 / scale - inset
        places = [(name, self._plane(place)) for name, place in self._device_stations()]
        if holder_is_calibrated(self._microscope):
            for slot in holder_slots(self._microscope):
                place = slot_landmark(self._microscope, slot)
                if place is not None:
                    places.append((place.name or slot.name, self._plane(place)))
        return [
            (name, point)
            for name, point in places
            if abs(point[0] - centre_x) > half_w or abs(point[1] - centre_y) > half_h
        ]

    def _paint_offscreen(
        self,
        painter: QPainter,
        rect: QRectF,
        zoom: int,
        scale: float,
        visible: QRectF,
    ) -> None:
        """An arrow on the edge of the view for each place outside it, pointing the
        way to it, with its name: how far the stage has to go is the travel step's
        business, which way is this one's."""
        centre = visible.center()
        half_w = visible.width() / 2 - _OFFSCREEN_INSET_PX
        half_h = visible.height() / 2 - _OFFSCREEN_INSET_PX
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)

        # Where each arrow lands, then one arrow per cluster: two slots off the same
        # side would otherwise draw their arrows and names over each other.
        arrows: List[Tuple[List[str], QPointF, float, float]] = []
        for name, (px, py) in self.offscreen_places(rect, zoom):
            dx, dy = px * scale - centre.x(), py * scale - centre.y()
            length = math.hypot(dx, dy)
            if length == 0:
                continue
            ux, uy = dx / length, dy / length
            # Along the line from the centre, to where it leaves the inset rectangle.
            reach = min(
                half_w / abs(ux) if ux else math.inf,
                half_h / abs(uy) if uy else math.inf,
            )
            tip = QPointF(centre.x() + ux * reach, centre.y() + uy * reach)
            for names, other, _, _ in arrows:
                if math.hypot(tip.x() - other.x(), tip.y() - other.y()) < _MERGE_PX:
                    names.append(name)
                    break
            else:
                arrows.append(([name], tip, ux, uy))

        metrics = painter.fontMetrics()
        for names, tip, ux, uy in arrows:
            back = QPointF(tip.x() - ux * 9, tip.y() - uy * 9)
            side = QPointF(-uy * 5, ux * 5)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(SLOT_COLOUR))
            painter.drawPolygon(QPolygonF([tip, back + side, back - side]))

            # The name inward of the arrow, far enough along the arrow's line that its
            # own width or height clears the arrowhead.
            text = ", ".join(names)
            width, height = metrics.horizontalAdvance(text), metrics.height()
            gap = 14 + abs(ux) * width / 2 + abs(uy) * height / 2
            label = QPointF(tip.x() - ux * gap, tip.y() - uy * gap)
            painter.setPen(QColor(NEUTRAL_550))
            painter.setBrush(Qt.NoBrush)
            painter.drawText(
                QRectF(
                    label.x() - width / 2 - 2, label.y() - height / 2, width + 4, height
                ),
                Qt.AlignCenter,
                text,
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

    def _paint_positions(
        self, painter: QPainter, frame: StageFrame, detailed: bool
    ) -> None:
        """Lamellae as hairline crosshairs, or as dots in a thumbnail."""
        colour = QColor(SAVED_POSITION_COLOUR)
        colour.setAlphaF(0.65 if detailed else 0.8)
        painter.setPen(QPen(colour, 1 if detailed else 1.5))
        for position in self._positions:
            try:
                x, y = frame.to_canvas(position)
            except Exception:
                continue
            if detailed:
                painter.drawLine(QPointF(x - 3.5, y), QPointF(x + 3.5, y))
                painter.drawLine(QPointF(x, y - 3.5), QPointF(x, y + 3.5))
            else:
                painter.drawPoint(QPointF(x, y))

    def _paint_here(self, painter: QPainter, frame: StageFrame, detailed: bool) -> None:
        """A ring round the lamella the stage is on, drawn over the stage cross: a
        crosshair there was hidden under it."""
        colour = QColor(SELECTED_POSITION_COLOUR)
        colour.setAlphaF(0.9)
        painter.setPen(QPen(colour, 1.2 if detailed else 1.0))
        painter.setBrush(Qt.NoBrush)
        radius = _HERE_RING_PX if detailed else _HERE_RING_PX / 2
        for position in self.lamellae_here():
            x, y = frame.to_canvas(position)
            painter.drawEllipse(QPointF(x, y), radius, radius)

    def _paint_stage(
        self, painter: QPainter, frame: StageFrame, detailed: bool
    ) -> None:
        if self._stage is None:
            return
        x, y = frame.to_canvas(self._stage)
        arm = 7.0 if detailed else 4.0
        # No rotation indicator. The map holds the holder still, so the stage has no
        # heading in it; what a rotation changes is which side the ion beam comes
        # from, and that also depends on the tilt (none when the beam is square on).
        # The side view shows the half turn.
        colour = QColor(CURRENT_POSITION_COLOUR)
        painter.setPen(QPen(colour, 1.5 if detailed else 1.0))
        painter.drawLine(QPointF(x - arm, y), QPointF(x + arm, y))
        painter.drawLine(QPointF(x, y - arm), QPointF(x, y + arm))
