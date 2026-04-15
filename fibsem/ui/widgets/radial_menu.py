"""Radial (pie) menu overlay — variable number of sectors.

Any number of items are distributed evenly around a circle, starting at
the top (North) and going clockwise.

Usage::

    menu = RadialMenuOverlay(items=[
        ("Ruler",     fn_ruler),
        ("Rectangle", fn_rect),
        ("Angle",     fn_angle),
        ("Profile",   fn_profile),
        ("Pin",       fn_pin),
        ("Clear",     fn_clear),
    ])

    class MyWidget(QWidget):
        def mousePressEvent(self, event):
            if event.button() == Qt.RightButton:
                menu.show_at(event.globalPos())

``QuadMenuOverlay`` is kept as a backward-compatible wrapper that accepts
the original ``callbacks`` / ``labels`` dicts keyed by N / E / S / W.
"""

import math
from typing import Callable, Dict, List, Optional, Tuple

from PyQt5 import QtWidgets
from PyQt5.QtCore import QPoint, QPointF, QRect, Qt
from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath, QPen

# Default colour palette — cycles if there are more items than colours
_PALETTE = [
    QColor(60,  120, 200, 180),   # blue
    QColor(60,  180, 100, 180),   # green
    QColor(200,  80,  80, 180),   # red
    QColor(200, 160,  40, 180),   # amber
    QColor(140,  60, 200, 180),   # purple
    QColor(200, 110,  40, 180),   # orange
    QColor( 40, 180, 180, 180),   # cyan
    QColor(180,  40, 120, 180),   # pink
]
_PALETTE_ACTIVE = [
    QColor( 90, 160, 255, 220),
    QColor( 80, 230, 130, 220),
    QColor(255, 110, 110, 220),
    QColor(255, 210,  60, 220),
    QColor(180, 100, 255, 220),
    QColor(255, 160,  70, 220),
    QColor( 60, 230, 230, 220),
    QColor(240,  80, 160, 220),
]

_SIZE             = 300
_DEAD_ZONE_RADIUS = 38
_OUTER_RADIUS     = _SIZE // 2
_LABEL_RADIUS     = 0.58   # fraction of outer radius


class RadialMenuOverlay(QtWidgets.QWidget):
    """Radial menu with a variable number of evenly-spaced sectors.

    Parameters
    ----------
    items:
        List of ``(label, callback)`` pairs.  Sectors are placed starting
        at the top (North) and going clockwise.
    colors:
        Optional list of ``QColor`` for each sector (normal state).
        Defaults to the built-in palette (cycles if len > 8).
    """

    def __init__(
        self,
        items: List[Tuple[str, Callable]],
        colors: Optional[List[QColor]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self._items  = items            # [(label, callback), ...]
        self._colors = colors
        self._active_index: Optional[int] = None

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setMouseTracking(True)
        self.setFixedSize(_SIZE, _SIZE)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show_at(self, global_pos: QPoint) -> None:
        """Centre the menu on *global_pos* and grab the mouse."""
        self._active_index = None
        top_left = global_pos - QPoint(_SIZE // 2, _SIZE // 2)
        self.move(top_left)
        self.show()
        self.grabMouse()

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def mouseMoveEvent(self, event) -> None:
        idx = self._sector_at(event.pos())
        if idx != self._active_index:
            self._active_index = idx
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.RightButton:
            self.releaseMouse()
            self.hide()
            idx = self._sector_at(event.pos())
            if idx is not None:
                _, cb = self._items[idx]
                if cb is not None:
                    cb()
                else:
                    print(self._items[idx][0])

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:
        if not self._items:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        n   = len(self._items)
        cx  = cy = _SIZE // 2
        rect = QRect(0, 0, _SIZE, _SIZE)
        span_deg = 360.0 / n

        for i, (label, _) in enumerate(self._items):
            # Sector centre angle in Qt degrees (0 = East, CCW positive).
            # We want sector 0 at North, going CW in screen space.
            # North in Qt = 90°.  Each step CW = subtract span_deg.
            center_qt = 90.0 - i * span_deg
            start_qt  = center_qt + span_deg / 2   # Qt CCW, so +half puts start CW of centre
            # We draw CCW spans, so negate span to go CW
            # Actually: drawPie(rect, startAngle*16, spanAngle*16)
            # positive span = CCW.  We want CW sectors → negative span.
            # But let's just use positive span and offset start accordingly.
            start_qt = center_qt - span_deg / 2

            color = self._sector_color(i, active=(i == self._active_index))
            painter.setBrush(color)
            painter.setPen(QPen(QColor(255, 255, 255, 50), 1))
            painter.drawPie(rect,
                            int(start_qt * 16),
                            int(span_deg * 16))

        # Punch out dead-zone
        painter.setCompositionMode(QPainter.CompositionMode_Clear)
        painter.setBrush(Qt.transparent)
        painter.setPen(Qt.NoPen)
        dz = QRect(cx - _DEAD_ZONE_RADIUS, cy - _DEAD_ZONE_RADIUS,
                   _DEAD_ZONE_RADIUS * 2,  _DEAD_ZONE_RADIUS * 2)
        painter.drawEllipse(dz)

        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)

        # Dead-zone ring
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(255, 255, 255, 100), 1.5))
        painter.drawEllipse(dz)

        # Dividing lines between sectors
        painter.setPen(QPen(QColor(255, 255, 255, 70), 1))
        for i in range(n):
            # Angle of the boundary between sector i and i+1 (screen CW from North)
            boundary_screen = i * 2 * math.pi / n   # radians, CW from North
            bx = cx + (_OUTER_RADIUS - 1) * math.sin(boundary_screen)
            by = cy - (_OUTER_RADIUS - 1) * math.cos(boundary_screen)
            painter.drawLine(cx, cy, int(bx), int(by))

        # Outer border
        painter.setPen(QPen(QColor(255, 255, 255, 90), 1.5))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(rect.adjusted(1, 1, -1, -1))

        # Labels
        font = QFont("Arial", max(8, 12 - max(0, n - 4)), QFont.Bold)
        painter.setFont(font)
        fm  = painter.fontMetrics()
        lr  = _OUTER_RADIUS * _LABEL_RADIUS

        for i, (label, _) in enumerate(self._items):
            a = i * 2 * math.pi / n          # screen-CW from North
            lx = cx + lr * math.sin(a)
            ly = cy - lr * math.cos(a)

            text_w = fm.horizontalAdvance(label) + 16
            text_h = fm.height() + 8
            label_rect = QRect(int(lx) - text_w // 2, int(ly) - text_h // 2,
                               text_w, text_h)

            if i == self._active_index:
                painter.setPen(QPen(Qt.white))
            else:
                painter.setPen(QPen(QColor(220, 220, 220, 200)))
            painter.drawText(label_rect, Qt.AlignCenter, label)

        painter.end()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sector_at(self, pos: QPoint) -> Optional[int]:
        """Return the sector index under *pos*, or None if in dead-zone."""
        cx = cy = _SIZE // 2
        dx = pos.x() - cx
        dy = pos.y() - cy

        if math.hypot(dx, dy) < _DEAD_ZONE_RADIUS:
            return None

        # atan2(dx, -dy): angle CW from North in screen space, range (-π, π]
        theta = math.atan2(dx, -dy)
        # Normalise to [0, 2π)
        theta = theta % (2 * math.pi)
        n = len(self._items)
        return int(theta / (2 * math.pi / n)) % n

    def _sector_color(self, index: int, active: bool) -> QColor:
        if self._colors:
            base = self._colors[index % len(self._colors)]
            if active:
                return base.lighter(140)
            return base
        if active:
            return _PALETTE_ACTIVE[index % len(_PALETTE_ACTIVE)]
        return _PALETTE[index % len(_PALETTE)]


# ---------------------------------------------------------------------------
# Backward-compatible quad menu
# ---------------------------------------------------------------------------

_QUAD_ORDER = ["N", "E", "S", "W"]


class QuadMenuOverlay(RadialMenuOverlay):
    """Four-sector radial menu with N / E / S / W dict interface.

    Preserved for backward compatibility.  New code should use
    ``RadialMenuOverlay`` directly.
    """

    def __init__(
        self,
        callbacks: Optional[Dict[str, Callable]] = None,
        labels: Optional[Dict[str, str]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        cbs    = callbacks or {}
        lbls   = labels    or {}
        items  = [
            (lbls.get(k, k), cbs.get(k))
            for k in _QUAD_ORDER
        ]
        super().__init__(items=items, parent=parent)
