"""Drag-to-measure widgets.

Two tools share a common base overlay:

DragDistanceOverlay
    Draw a line; shows diagonal, horizontal and vertical distances.
    Hold Shift = horizontal lock, Ctrl = vertical lock.

RectMeasureOverlay
    Draw a rectangle; shows width, height and area.
    Hold Shift = constrain to square.

Pass ``pixel_size`` (metres per pixel) for SI auto-formatted distances,
or ``scale`` + ``unit`` for manual control.

Usage::

    ruler = DragDistanceOverlay(pixel_size=100e-9, on_measure=my_cb)
    rect  = RectMeasureOverlay(pixel_size=100e-9, on_measure=my_cb)

    class MyWidget(QWidget):
        def mousePressEvent(self, event):
            if event.button() == Qt.LeftButton:
                ruler.start(event.globalPos(), self)
"""

import math
from enum import Enum, auto
from typing import Callable, Optional

from PyQt5 import QtWidgets
from PyQt5.QtCore import QPoint, QRect, Qt
from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath, QPen

# ---------------------------------------------------------------------------
# Shared colours
# ---------------------------------------------------------------------------
_LINE_COLOR = QColor(255, 220, 50, 230)      # yellow  – diagonal / main
_LINE_COLOR_H = QColor(80, 200, 255, 230)    # blue    – horizontal
_LINE_COLOR_V = QColor(120, 255, 120, 230)   # green   – vertical
_RECT_FILL = QColor(255, 220, 50, 30)
_TEXT_BG = QColor(20, 20, 20, 180)
_ENDPOINT_RADIUS = 5


class ConstraintMode(Enum):
    FREE = auto()
    HORIZONTAL = auto()
    VERTICAL = auto()
    SQUARE = auto()


# ---------------------------------------------------------------------------
# Base overlay
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# SI formatting helpers
# ---------------------------------------------------------------------------

def _fmt_distance(metres: float) -> str:
    """Auto-scale a distance in metres to a human-readable SI string."""
    abs_m = abs(metres)
    if abs_m == 0:
        return "0 nm"
    if abs_m < 1e-6:
        return f"{metres * 1e9:.1f} nm"
    if abs_m < 1e-3:
        return f"{metres * 1e6:.2f} µm"
    if abs_m < 1.0:
        return f"{metres * 1e3:.3f} mm"
    return f"{metres:.4f} m"


def _fmt_area(m2: float) -> str:
    """Auto-scale an area in m² to a human-readable SI string."""
    abs_m2 = abs(m2)
    if abs_m2 == 0:
        return "0 nm²"
    if abs_m2 < 1e-12:
        return f"{m2 * 1e18:.1f} nm²"
    if abs_m2 < 1e-6:
        return f"{m2 * 1e12:.2f} µm²"
    if abs_m2 < 1.0:
        return f"{m2 * 1e6:.3f} mm²"
    return f"{m2:.4f} m²"


class _MeasureOverlayBase(QtWidgets.QWidget):
    """Shared boilerplate for all measure overlays.

    Subclasses must implement:
      - ``mouseMoveEvent``
      - ``paintEvent``
      - ``_finish()`` — called on mouse release; should fire the callback.
    """

    def __init__(
        self,
        pixel_size: Optional[float] = None,
        scale: float = 1.0,
        unit: str = "px",
        on_measure: Optional[Callable] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self._pixel_size = pixel_size   # metres per pixel; None = use scale/unit
        self._scale = scale
        self._unit = unit
        self._on_measure = on_measure
        self._start: Optional[QPoint] = None
        self._end: Optional[QPoint] = None
        self._active = False

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setMouseTracking(True)

    def _fmt(self, pixels: float) -> str:
        """Format a pixel distance using pixel_size (SI) or scale/unit."""
        if self._pixel_size is not None:
            return _fmt_distance(pixels * self._pixel_size)
        return f"{pixels * self._scale:.1f} {self._unit}"

    def _fmt_area(self, pixels_sq: float) -> str:
        """Format a pixel area using pixel_size (SI) or scale/unit."""
        if self._pixel_size is not None:
            return _fmt_area(pixels_sq * self._pixel_size ** 2)
        return f"{pixels_sq * self._scale ** 2:.1f} {self._unit}²"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, global_pos: QPoint, source_widget: QtWidgets.QWidget) -> None:
        """Begin a measurement from *global_pos*, overlaying *source_widget*."""
        geo = source_widget.rect()
        top_left = source_widget.mapToGlobal(geo.topLeft())
        self.setGeometry(QRect(top_left, geo.size()))

        self._start = self.mapFromGlobal(global_pos)
        self._end = self._start
        self._active = True
        self._on_start()

        self.show()
        self.grabMouse()
        self.grabKeyboard()
        self.update()

    def _on_start(self) -> None:
        """Hook for subclasses to reset per-gesture state."""

    # ------------------------------------------------------------------
    # Mouse / keyboard events
    # ------------------------------------------------------------------

    def mouseReleaseEvent(self, event) -> None:
        if event.button() in (Qt.RightButton, Qt.LeftButton) and self._active:
            self.releaseMouse()
            self.releaseKeyboard()
            self._active = False
            self.hide()
            if self._start and self._end:
                self._finish()

    # ------------------------------------------------------------------
    # Shared drawing helper
    # ------------------------------------------------------------------

    def _draw_label(
        self,
        painter: QPainter,
        text: str,
        cx: float,
        cy: float,
        color: QColor,
        offset_x: float = 0,
        offset_y: float = 0,
    ) -> None:
        """Draw a pill-shaped label centred at (cx + offset_x, cy + offset_y)."""
        fm = painter.fontMetrics()
        text_w = fm.horizontalAdvance(text) + 12
        text_h = fm.height() + 6
        lx = int(cx + offset_x - text_w / 2)
        ly = int(cy + offset_y - text_h / 2)

        bg = QRect(lx - 2, ly - 2, text_w + 4, text_h + 4)
        path = QPainterPath()
        path.addRoundedRect(bg.x(), bg.y(), bg.width(), bg.height(), 6, 6)
        painter.fillPath(path, _TEXT_BG)
        painter.setPen(QPen(color))
        painter.drawText(QRect(lx, ly, text_w, text_h), Qt.AlignCenter, text)

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    def _finish(self) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Ruler
# ---------------------------------------------------------------------------

class DragDistanceOverlay(_MeasureOverlayBase):
    """Drag-to-measure line overlay.

    Shows the diagonal distance and, in FREE mode, the horizontal and
    vertical components as dashed guide lines.

    Constraints
    -----------
    Shift — horizontal lock
    Ctrl  — vertical lock
    """

    def __init__(
        self,
        pixel_size: Optional[float] = None,
        scale: float = 1.0,
        unit: str = "px",
        constraint: ConstraintMode = ConstraintMode.FREE,
        on_measure: Optional[Callable[[float], None]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(pixel_size=pixel_size, scale=scale, unit=unit,
                         on_measure=on_measure, parent=parent)
        self._default_constraint = constraint
        self._constraint = constraint

    def _on_start(self) -> None:
        self._constraint = self._default_constraint

    # ------------------------------------------------------------------
    # Mouse / keyboard
    # ------------------------------------------------------------------

    def mouseMoveEvent(self, event) -> None:
        if not self._active:
            return
        self._constraint = self._runtime_constraint(event.modifiers())
        self._end = self._apply_constraint(event.pos())
        self.update()

    def keyPressEvent(self, event) -> None:
        if self._active and self._end:
            self._constraint = self._runtime_constraint(event.modifiers())
            self._end = self._apply_constraint(self._end)
            self.update()

    def keyReleaseEvent(self, event) -> None:
        if self._active and self._end:
            self._constraint = self._runtime_constraint(event.modifiers())
            self._end = self._apply_constraint(self._end)
            self.update()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:
        if not self._start or not self._end:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(QFont("Arial", 11, QFont.Bold))

        x1, y1 = self._start.x(), self._start.y()
        x2, y2 = self._end.x(), self._end.y()
        dx, dy = x2 - x1, y2 - y1

        if self._constraint == ConstraintMode.FREE and dx != 0 and dy != 0:
            # Horizontal leg
            painter.setPen(QPen(_LINE_COLOR_H, 1, Qt.DashLine))
            painter.drawLine(x1, y1, x2, y1)
            # Vertical leg
            painter.setPen(QPen(_LINE_COLOR_V, 1, Qt.DashLine))
            painter.drawLine(x2, y1, x2, y2)
            # Right-angle corner dot
            painter.setBrush(_TEXT_BG)
            painter.setPen(Qt.NoPen)
            cr = 3
            painter.drawEllipse(x2 - cr, y1 - cr, cr * 2, cr * 2)
            # Component labels
            self._draw_label(painter, self._fmt(abs(dx)),
                             (x1 + x2) / 2, y1, _LINE_COLOR_H, offset_y=-18)
            side = 18 if x2 >= x1 else -18
            self._draw_label(painter, self._fmt(abs(dy)),
                             x2, (y1 + y2) / 2, _LINE_COLOR_V, offset_x=side)
            line_color = _LINE_COLOR

        elif self._constraint == ConstraintMode.HORIZONTAL:
            painter.setPen(QPen(_LINE_COLOR_H.darker(150), 1, Qt.DashLine))
            painter.drawLine(x2, y1, x2, y1 + dy)
            line_color = _LINE_COLOR_H

        elif self._constraint == ConstraintMode.VERTICAL:
            painter.setPen(QPen(_LINE_COLOR_V.darker(150), 1, Qt.DashLine))
            painter.drawLine(x1, y2, x1 + dx, y2)
            line_color = _LINE_COLOR_V

        else:
            line_color = _LINE_COLOR

        # Main line
        painter.setPen(QPen(line_color, 2))
        painter.drawLine(x1, y1, x2, y2)

        # Endpoint dots
        painter.setBrush(line_color)
        painter.setPen(Qt.NoPen)
        r = _ENDPOINT_RADIUS
        painter.drawEllipse(x1 - r, y1 - r, r * 2, r * 2)
        painter.drawEllipse(x2 - r, y2 - r, r * 2, r * 2)

        # Main distance label perpendicular to line
        label = self._fmt(math.hypot(dx, dy))
        if self._constraint != ConstraintMode.FREE:
            label += f"  [{self._constraint.name[0]}]"
        length = math.hypot(dx, dy) or 1
        nx, ny = -dy / length, dx / length
        if ny > 0:
            nx, ny = -nx, -ny
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        self._draw_label(painter, label, mx + nx * 18, my + ny * 18, line_color)

        painter.end()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _finish(self) -> None:
        dx = self._end.x() - self._start.x()
        dy = self._end.y() - self._start.y()
        dist_px = math.hypot(dx, dy)
        if self._on_measure:
            self._on_measure(dist_px * (self._pixel_size or self._scale))
        else:
            print(f"Distance: {self._fmt(dist_px)}")

    def _apply_constraint(self, pos: QPoint) -> QPoint:
        if self._constraint == ConstraintMode.HORIZONTAL:
            return QPoint(pos.x(), self._start.y())
        if self._constraint == ConstraintMode.VERTICAL:
            return QPoint(self._start.x(), pos.y())
        return pos

    def _runtime_constraint(self, modifiers) -> ConstraintMode:
        if modifiers & Qt.ShiftModifier:
            return ConstraintMode.HORIZONTAL
        if modifiers & Qt.ControlModifier:
            return ConstraintMode.VERTICAL
        return self._default_constraint


# ---------------------------------------------------------------------------
# Rectangle
# ---------------------------------------------------------------------------

class RectMeasureOverlay(_MeasureOverlayBase):
    """Drag-to-measure rectangle overlay.

    Shows width (blue), height (green) and area (yellow) labels.
    Hold Shift to constrain to a square.

    The ``on_measure`` callback receives ``(width, height, area)`` in
    scaled units.
    """

    def __init__(
        self,
        pixel_size: Optional[float] = None,
        scale: float = 1.0,
        unit: str = "px",
        on_measure: Optional[Callable[[float, float, float], None]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(pixel_size=pixel_size, scale=scale, unit=unit,
                         on_measure=on_measure, parent=parent)
        self._square = False

    def _on_start(self) -> None:
        self._square = False

    # ------------------------------------------------------------------
    # Mouse / keyboard
    # ------------------------------------------------------------------

    def mouseMoveEvent(self, event) -> None:
        if not self._active:
            return
        self._square = bool(event.modifiers() & Qt.ShiftModifier)
        self._end = self._apply_square(event.pos())
        self.update()

    def keyPressEvent(self, event) -> None:
        if self._active and self._end:
            self._square = bool(event.modifiers() & Qt.ShiftModifier)
            self._end = self._apply_square(self._end)
            self.update()

    def keyReleaseEvent(self, event) -> None:
        if self._active and self._end:
            self._square = bool(event.modifiers() & Qt.ShiftModifier)
            self._end = self._apply_square(self._end)
            self.update()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:
        if not self._start or not self._end:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(QFont("Arial", 11, QFont.Bold))

        x1, y1 = self._start.x(), self._start.y()
        x2, y2 = self._end.x(), self._end.y()
        dx, dy = x2 - x1, y2 - y1

        rect = QRect(min(x1, x2), min(y1, y2), abs(dx), abs(dy))

        # Semi-transparent fill
        painter.setBrush(_RECT_FILL)
        painter.setPen(Qt.NoPen)
        painter.drawRect(rect)

        # Border
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(_LINE_COLOR, 2))
        painter.drawRect(rect)

        # Corner dots
        painter.setBrush(_LINE_COLOR)
        painter.setPen(Qt.NoPen)
        r = _ENDPOINT_RADIUS
        for cx, cy in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]:
            painter.drawEllipse(cx - r, cy - r, r * 2, r * 2)

        # Width label — centred above the top edge
        top_y = min(y1, y2)
        self._draw_label(painter, self._fmt(abs(dx)),
                         (x1 + x2) / 2, top_y, _LINE_COLOR_H, offset_y=-18)

        # Height label — centred to the right of the right edge
        right_x = max(x1, x2)
        self._draw_label(painter, self._fmt(abs(dy)),
                         right_x, (y1 + y2) / 2, _LINE_COLOR_V, offset_x=18)

        # Area label — centred inside the rectangle (if large enough)
        area_label = self._fmt_area(abs(dx) * abs(dy))
        if self._square:
            area_label += "  [■]"
        if abs(dx) > 80 and abs(dy) > 30:
            self._draw_label(painter, area_label,
                             (x1 + x2) / 2, (y1 + y2) / 2, _LINE_COLOR)
        else:
            # Fallback: place outside, below the bottom edge
            bottom_y = max(y1, y2)
            self._draw_label(painter, area_label,
                             (x1 + x2) / 2, bottom_y, _LINE_COLOR, offset_y=18)

        painter.end()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _finish(self) -> None:
        dx = self._end.x() - self._start.x()
        dy = self._end.y() - self._start.y()
        s = self._pixel_size or self._scale
        w, h = abs(dx) * s, abs(dy) * s
        area = w * h
        if self._on_measure:
            self._on_measure(w, h, area)
        else:
            print(f"Rectangle: {self._fmt(abs(dx))} × {self._fmt(abs(dy))}  area={self._fmt_area(abs(dx) * abs(dy))}")

    def _apply_square(self, pos: QPoint) -> QPoint:
        if not self._square or not self._start:
            return pos
        dx = pos.x() - self._start.x()
        dy = pos.y() - self._start.y()
        side = max(abs(dx), abs(dy))
        return QPoint(
            int(self._start.x() + math.copysign(side, dx)),
            int(self._start.y() + math.copysign(side, dy)),
        )


# ---------------------------------------------------------------------------
# Mixins
# ---------------------------------------------------------------------------

class DragDistanceMixin:
    """Adds left-click drag-to-measure (ruler) to any QWidget."""

    def setup_drag_distance(
        self,
        scale: float = 1.0,
        unit: str = "px",
        constraint: ConstraintMode = ConstraintMode.FREE,
        on_measure: Optional[Callable[[float], None]] = None,
    ) -> None:
        self._drag_distance = DragDistanceOverlay(
            scale=scale, unit=unit, constraint=constraint, on_measure=on_measure
        )

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and hasattr(self, "_drag_distance"):
            self._drag_distance.start(event.globalPos(), self)
        else:
            super().mousePressEvent(event)


class RectMeasureMixin:
    """Adds left-click drag-to-measure (rectangle) to any QWidget."""

    def setup_rect_measure(
        self,
        scale: float = 1.0,
        unit: str = "px",
        on_measure: Optional[Callable[[float, float, float], None]] = None,
    ) -> None:
        self._rect_measure = RectMeasureOverlay(
            scale=scale, unit=unit, on_measure=on_measure
        )

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and hasattr(self, "_rect_measure"):
            self._rect_measure.start(event.globalPos(), self)
        else:
            super().mousePressEvent(event)
