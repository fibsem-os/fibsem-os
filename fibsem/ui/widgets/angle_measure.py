"""Angle measurement overlay.

Two-phase drag interaction:
  Phase 1 — drag from vertex to arm-1 end (left-click + drag, release)
  Phase 2 — drag from vertex to arm-2 end (left-click + drag, release)
             → shows angle arc and value, fires callback

Escape or right-click cancels at any phase.
"""

import math
from typing import Callable, Optional

from PyQt5 import QtWidgets
from PyQt5.QtCore import QPoint, QRect, Qt
from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath, QPen

from fibsem.ui.widgets.drag_distance import _MeasureOverlayBase, _fmt_distance

_COLOR_ARM1  = QColor(80, 200, 255, 220)   # blue
_COLOR_ARM2  = QColor(120, 255, 120, 220)  # green
_COLOR_ARC   = QColor(255, 220, 50, 200)   # yellow
_COLOR_LABEL = QColor(255, 220, 50, 230)
_TEXT_BG     = QColor(20, 20, 20, 180)
_DOT_R = 5
_ARC_R = 40   # arc radius in screen pixels


class AngleMeasureOverlay(_MeasureOverlayBase):
    """Drag twice from the vertex to measure the angle between two arms.

    The ``on_measure`` callback receives the angle in degrees.
    """

    def __init__(
        self,
        pixel_size: Optional[float] = None,
        scale: float = 1.0,
        unit: str = "px",
        on_measure: Optional[Callable[[float], None]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(pixel_size=pixel_size, scale=scale, unit=unit,
                         on_measure=on_measure, parent=parent)
        self._phase = 0          # 0=idle, 1=drawing arm1, 2=drawing arm2
        self._vertex: Optional[QPoint] = None
        self._arm1:   Optional[QPoint] = None
        self._arm2:   Optional[QPoint] = None

    # ------------------------------------------------------------------
    # Public API — override start() for two-phase gesture
    # ------------------------------------------------------------------

    def start(self, global_pos: QPoint, source_widget: QtWidgets.QWidget) -> None:
        from PyQt5.QtCore import QRect
        geo = source_widget.rect()
        top_left = source_widget.mapToGlobal(geo.topLeft())
        self.setGeometry(QRect(top_left, geo.size()))

        self._vertex = self.mapFromGlobal(global_pos)
        self._arm1 = self._vertex
        self._arm2 = None
        self._phase = 1
        self._active = True

        self.show()
        self.grabMouse()
        self.grabKeyboard()
        self.update()

    # ------------------------------------------------------------------
    # Mouse / keyboard
    # ------------------------------------------------------------------

    def mouseMoveEvent(self, event) -> None:
        if not self._active:
            return
        if self._phase == 1:
            self._arm1 = event.pos()
        elif self._phase == 2:
            self._arm2 = event.pos()
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._phase == 2:
            # Confirm second arm on press so the user sees it snap
            self._arm2 = event.pos()
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.RightButton:
            self._cancel()
            return
        if event.button() != Qt.LeftButton or not self._active:
            return

        if self._phase == 1:
            self._arm1 = event.pos()
            self._phase = 2
            self._arm2 = event.pos()
            self.update()
        elif self._phase == 2:
            self._arm2 = event.pos()
            self.releaseMouse()
            self.releaseKeyboard()
            self._active = False
            self._phase = 0
            self.hide()
            self._finish()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self._cancel()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:
        if self._vertex is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(QFont("Arial", 11, QFont.Bold))

        vx, vy = self._vertex.x(), self._vertex.y()

        # --- Arm 1 ---
        if self._arm1:
            painter.setPen(QPen(_COLOR_ARM1, 2))
            painter.drawLine(vx, vy, self._arm1.x(), self._arm1.y())
            self._dot(painter, self._arm1, _COLOR_ARM1)

        # --- Arm 2 ---
        if self._arm2:
            painter.setPen(QPen(_COLOR_ARM2, 2))
            painter.drawLine(vx, vy, self._arm2.x(), self._arm2.y())
            self._dot(painter, self._arm2, _COLOR_ARM2)

        # --- Vertex dot ---
        self._dot(painter, self._vertex, _COLOR_LABEL)

        # --- Phase hint ---
        if self._phase == 1:
            self._draw_label(painter, "Set first arm →", vx, vy,
                             _COLOR_ARM1, offset_y=-22)
        elif self._phase == 2 and self._arm2 is None:
            self._draw_label(painter, "Set second arm →", vx, vy,
                             _COLOR_ARM2, offset_y=-22)

        # --- Arc + angle ---
        if self._arm1 and self._arm2:
            angle = self._compute_angle()
            if angle is not None:
                self._draw_arc(painter, vx, vy, angle)
                mid_angle = self._midpoint_angle()
                lx = vx + math.cos(mid_angle) * (_ARC_R + 22)
                ly = vy - math.sin(mid_angle) * (_ARC_R + 22)
                self._draw_label(painter, f"{angle:.1f}°", lx, ly, _COLOR_LABEL)

        painter.end()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _dot(self, painter: QPainter, pt: QPoint, color: QColor) -> None:
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(pt.x() - _DOT_R, pt.y() - _DOT_R,
                            _DOT_R * 2, _DOT_R * 2)

    def _draw_arc(self, painter: QPainter, vx: int, vy: int, angle: float) -> None:
        start_angle = self._angle_of(self._arm1) * 180 / math.pi
        span = angle if self._signed_angle() >= 0 else -angle

        painter.setBrush(QColor(255, 220, 50, 40))
        painter.setPen(QPen(_COLOR_ARC, 1.5))
        arc_rect = QRect(int(vx - _ARC_R), int(vy - _ARC_R),
                         _ARC_R * 2, _ARC_R * 2)
        # Qt angles: CCW positive, 1/16 degree units
        painter.drawPie(arc_rect,
                        int(start_angle * 16),
                        int(span * 16))

    def _angle_of(self, pt: Optional[QPoint]) -> float:
        if pt is None:
            return 0.0
        dx = pt.x() - self._vertex.x()
        dy = -(pt.y() - self._vertex.y())   # flip Y for screen coords
        return math.atan2(dy, dx)

    def _signed_angle(self) -> float:
        """Signed angle from arm1 to arm2 (CCW positive)."""
        a1 = self._angle_of(self._arm1)
        a2 = self._angle_of(self._arm2)
        diff = a2 - a1
        # Normalise to (-π, π]
        while diff > math.pi:  diff -= 2 * math.pi
        while diff <= -math.pi: diff += 2 * math.pi
        return diff

    def _midpoint_angle(self) -> float:
        a1 = self._angle_of(self._arm1)
        diff = self._signed_angle()
        return a1 + diff / 2

    def _compute_angle(self) -> Optional[float]:
        if self._arm1 is None or self._arm2 is None:
            return None
        return abs(math.degrees(self._signed_angle()))

    def _finish(self) -> None:
        angle = self._compute_angle()
        if angle is None:
            return
        if self._on_measure:
            self._on_measure(angle)
        else:
            print(f"Angle: {angle:.2f}°")

    def _cancel(self) -> None:
        if self._active:
            self.releaseMouse()
            self.releaseKeyboard()
        self._active = False
        self._phase = 0
        self.hide()

    # _finish is called by base mouseReleaseEvent — override to no-op
    # since we handle it manually above
    def _on_start(self) -> None:
        pass
