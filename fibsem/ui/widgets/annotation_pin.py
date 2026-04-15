"""Annotation pin overlay.

Click on the image to drop a named pin. Pins are drawn as persistent
markers in scene coordinates so they stay fixed when panning/zooming.

Usage::

    pins = AnnotationPinOverlay(view, parent=image_view)
    pins.show()

    # In mousePressEvent:
    if tool == "pin":
        pins.add_pin_at(event.globalPos())

    # Clear all pins:
    pins.clear()
"""

from dataclasses import dataclass, field
from typing import List, Optional

from PyQt5 import QtWidgets
from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath, QPen

_PIN_COLOR   = QColor(255, 80,  80,  230)
_TEXT_BG     = QColor(20,  20,  20,  200)
_TEXT_COLOR  = QColor(240, 240, 240, 230)
_DOT_R = 5
_STEM = 14   # vertical stem below label


@dataclass
class _Pin:
    scene_x: float
    scene_y: float
    label: str


class AnnotationPinOverlay(QtWidgets.QWidget):
    """Transparent overlay that draws annotation pins over a QGraphicsView.

    Pins are stored in scene coordinates so they stay anchored to the
    image content as the view is panned or zoomed.
    """

    def __init__(
        self,
        view: QtWidgets.QGraphicsView,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent or view)
        self._view = view
        self._pins: List[_Pin] = []
        self._counter = 1

        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._refit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_pin_at(self, global_pos) -> None:
        """Add a pin at the given global cursor position."""
        viewport_pos = self._view.viewport().mapFromGlobal(global_pos)
        scene_pos: QPointF = self._view.mapToScene(viewport_pos)

        label, ok = QtWidgets.QInputDialog.getText(
            self._view,
            "Add Pin",
            "Label:",
            text=f"Pin {self._counter}",
        )
        if not ok:
            return

        self._pins.append(_Pin(scene_pos.x(), scene_pos.y(), label or f"Pin {self._counter}"))
        self._counter += 1
        self.update()

    def clear(self) -> None:
        self._pins.clear()
        self._counter = 1
        self.update()

    def refresh(self) -> None:
        """Call after the view is panned/zoomed to redraw pins in new positions."""
        self.update()

    # ------------------------------------------------------------------
    # Resize with parent / view
    # ------------------------------------------------------------------

    def _refit(self) -> None:
        p = self.parent()
        if p:
            self.setGeometry(0, 0, p.width(), p.height())

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:
        if not self._pins:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(QFont("Arial", 9, QFont.Bold))

        for pin in self._pins:
            # Convert scene → viewport coordinates
            vp = self._view.mapFromScene(QPointF(pin.scene_x, pin.scene_y))
            px, py = vp.x(), vp.y()

            # Skip if outside the visible area
            if not self.rect().contains(px, py):
                continue

            fm = painter.fontMetrics()
            text_w = fm.horizontalAdvance(pin.label) + 10
            text_h = fm.height() + 4

            # Label box above the pin
            lx = int(px - text_w / 2)
            ly = int(py - _STEM - text_h)

            # Background pill
            bg = QPainterPath()
            bg.addRoundedRect(lx - 2, ly - 2, text_w + 4, text_h + 4, 4, 4)
            painter.fillPath(bg, _TEXT_BG)

            # Label border
            painter.setPen(QPen(_PIN_COLOR, 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(bg)

            # Label text
            painter.setPen(_TEXT_COLOR)
            painter.drawText(lx, ly, text_w, text_h, Qt.AlignCenter, pin.label)

            # Stem line
            painter.setPen(QPen(_PIN_COLOR, 1.5))
            painter.drawLine(int(px), int(ly + text_h + 2), int(px), int(py - _DOT_R))

            # Dot
            painter.setBrush(_PIN_COLOR)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(int(px - _DOT_R), int(py - _DOT_R),
                                _DOT_R * 2, _DOT_R * 2)

        painter.end()
