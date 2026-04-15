"""Zoom loupe overlay.

A circular magnifying glass that follows the cursor over a QGraphicsView.

Hold the configured key (default: Z) to show the loupe; release to hide.

Usage::

    loupe = ZoomLoupeOverlay(view, pixmap, zoom=4.0, diameter=220)
    loupe.attach(parent_widget)   # installs key handling on parent_widget

Then call ``loupe.on_mouse_move(viewport_pos)`` from the view's mouseMoveEvent.
"""

from typing import Optional

from PyQt5 import QtWidgets
from PyQt5.QtCore import QPoint, QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap

_BORDER_COLOR = QColor(220, 220, 220, 200)
_CROSSHAIR_COLOR = QColor(255, 80, 80, 180)
_SHADOW_COLOR = QColor(0, 0, 0, 80)


class ZoomLoupeOverlay(QtWidgets.QWidget):
    """Circular magnifying loupe that floats over a QGraphicsView.

    Parameters
    ----------
    view:
        The QGraphicsView whose scene content is magnified.
    pixmap:
        The full-resolution source pixmap displayed in the view.
    zoom:
        Magnification factor (e.g. 4.0 = 4×).
    diameter:
        Diameter of the loupe circle in screen pixels.
    activate_key:
        Qt key constant that shows/hides the loupe (default: Qt.Key_Z).
    """

    def __init__(
        self,
        view: QtWidgets.QGraphicsView,
        pixmap: QPixmap,
        zoom: float = 4.0,
        diameter: int = 220,
        activate_key: int = Qt.Key_Z,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self._view = view
        self._pixmap = pixmap
        self._zoom = zoom
        self._diameter = diameter
        self._activate_key = activate_key
        self._active = False
        self._cursor_viewport: Optional[QPoint] = None  # last known viewport pos

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(diameter, diameter)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_pixmap(self, pixmap: QPixmap) -> None:
        """Update the source image (e.g. after zoom/pan changes the displayed image)."""
        self._pixmap = pixmap
        if self._active:
            self.update()

    def on_mouse_move(self, viewport_pos: QPoint) -> None:
        """Call this from the parent view's mouseMoveEvent with the viewport position."""
        self._cursor_viewport = viewport_pos
        if self._active:
            self._reposition(self._view.mapToGlobal(viewport_pos))
            self.update()

    def on_key_press(self, key: int) -> bool:
        """Returns True if the event was consumed."""
        if key == self._activate_key and not self._active:
            self._active = True
            if self._cursor_viewport is not None:
                self._reposition(self._view.mapToGlobal(self._cursor_viewport))
            self.show()
            return True
        return False

    def on_key_release(self, key: int) -> bool:
        """Returns True if the event was consumed."""
        if key == self._activate_key and self._active:
            self._active = False
            self.hide()
            return True
        return False

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:
        if not self._active or self._cursor_viewport is None or self._pixmap.isNull():
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        r = self._diameter / 2
        cx, cy = r, r

        # Compute the source rect in scene (image) coordinates
        scene_pos: QPointF = self._view.mapToScene(self._cursor_viewport)
        half_src = r / self._zoom           # half-size in image pixels
        src_rect = QRectF(
            scene_pos.x() - half_src,
            scene_pos.y() - half_src,
            half_src * 2,
            half_src * 2,
        )

        # Clip painter to circle
        clip = QPainterPath()
        clip.addEllipse(0, 0, self._diameter, self._diameter)
        painter.setClipPath(clip)

        # Draw the magnified crop from the source pixmap
        painter.drawPixmap(
            QRectF(0, 0, self._diameter, self._diameter),
            self._pixmap,
            src_rect,
        )

        painter.setClipping(False)

        # Drop shadow ring
        shadow_pen = QPen(_SHADOW_COLOR, 5)
        painter.setPen(shadow_pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(2, 2, self._diameter - 4, self._diameter - 4)

        # Border ring
        painter.setPen(QPen(_BORDER_COLOR, 2))
        painter.drawEllipse(1, 1, self._diameter - 2, self._diameter - 2)

        # Crosshair at centre
        ch_size = 12
        painter.setPen(QPen(_CROSSHAIR_COLOR, 1.5))
        painter.drawLine(int(cx - ch_size), int(cy), int(cx + ch_size), int(cy))
        painter.drawLine(int(cx), int(cy - ch_size), int(cx), int(cy + ch_size))

        painter.end()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reposition(self, global_cursor: QPoint) -> None:
        """Move the loupe so it sits to the top-right of the cursor."""
        offset = 20
        x = global_cursor.x() + offset
        y = global_cursor.y() - self._diameter - offset
        # Keep on screen
        screen = QtWidgets.QApplication.screenAt(global_cursor)
        if screen:
            sg = screen.geometry()
            if x + self._diameter > sg.right():
                x = global_cursor.x() - self._diameter - offset
            if y < sg.top():
                y = global_cursor.y() + offset
        self.move(x, y)
