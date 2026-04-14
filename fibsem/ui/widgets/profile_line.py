"""Intensity profile line overlay.

Draw a line on a FibsemImage; on release a docked plot widget shows the
1-D pixel intensity along that line.

Usage::

    profile = ProfileLineOverlay(view, image, pixel_size=100e-9)
    profile.plot_widget   # QWidget — embed this somewhere in your layout

    class MyWidget(QWidget):
        def mousePressEvent(self, event):
            if event.button() == Qt.LeftButton and tool == "profile":
                profile.start(event.globalPos(), self)
"""

import math
from typing import Optional

import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import QPoint, QRect, Qt
from PyQt5.QtGui import QColor, QFont, QPainter, QPen

from fibsem.ui.widgets.drag_distance import _MeasureOverlayBase, _fmt_distance

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    _MPL = True
except ImportError:
    _MPL = False

_LINE_COLOR = QColor(255, 140, 0, 230)   # orange
_TEXT_BG    = QColor(20, 20, 20, 180)
_DOT_R = 5


class _ProfilePlotWidget(QtWidgets.QWidget):
    """Embedded matplotlib canvas that shows the intensity profile."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(160)
        self.setMaximumHeight(200)
        self.setStyleSheet("background: #1a1b1e;")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        if _MPL:
            self._fig = Figure(figsize=(4, 1.8), facecolor="#1a1b1e")
            self._ax  = self._fig.add_subplot(111)
            self._canvas = FigureCanvasQTAgg(self._fig)
            layout.addWidget(self._canvas)
            self._style_axes()
        else:
            layout.addWidget(QtWidgets.QLabel("matplotlib not available"))

        self._empty_label = QtWidgets.QLabel(
            "Draw a line on the image to see the intensity profile",
            self,
        )
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self._empty_label)
        if _MPL:
            self._canvas.hide()

    def _style_axes(self):
        ax = self._ax
        self._fig.patch.set_alpha(0)
        ax.set_facecolor("#23242a")
        ax.tick_params(colors="#999", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#444")
        ax.xaxis.label.set_color("#999")
        ax.yaxis.label.set_color("#999")

    def update_profile(
        self,
        distances: np.ndarray,
        intensities: np.ndarray,
        unit: str = "px",
    ) -> None:
        if not _MPL:
            return
        self._empty_label.hide()
        self._canvas.show()

        self._ax.clear()
        self._style_axes()
        self._ax.plot(distances, intensities, color="#ff8c00", linewidth=1.2)
        self._ax.set_xlabel(f"Distance ({unit})", fontsize=8, color="#999")
        self._ax.set_ylabel("Intensity", fontsize=8, color="#999")
        self._ax.set_xlim(distances[0], distances[-1])
        self._ax.set_ylim(0, 255)
        self._fig.tight_layout(pad=0.5)
        self._canvas.draw()


class ProfileLineOverlay(_MeasureOverlayBase):
    """Draw a line; on release show the 1-D intensity profile in a plot widget.

    Parameters
    ----------
    view:
        The QGraphicsView over which the overlay is drawn.
    image_data:
        2-D numpy array of the image (grayscale uint8).
    pixel_size:
        Metres per image pixel (used for the x-axis distance label).
    """

    def __init__(
        self,
        view: QtWidgets.QGraphicsView,
        image_data: Optional[np.ndarray] = None,
        pixel_size: Optional[float] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(pixel_size=pixel_size, parent=parent)
        self._view = view
        self._image_data = image_data

        self.plot_widget = _ProfilePlotWidget()

    def set_image_data(self, data: np.ndarray) -> None:
        self._image_data = data

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def mouseMoveEvent(self, event) -> None:
        if not self._active:
            return
        self._end = event.pos()
        self.update()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape and self._active:
            self.releaseMouse()
            self.releaseKeyboard()
            self._active = False
            self.hide()

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

        # Main line
        painter.setPen(QPen(_LINE_COLOR, 2, Qt.DashLine))
        painter.drawLine(x1, y1, x2, y2)

        # Endpoint dots
        painter.setBrush(_LINE_COLOR)
        painter.setPen(Qt.NoPen)
        for px, py in [(x1, y1), (x2, y2)]:
            painter.drawEllipse(px - _DOT_R, py - _DOT_R,
                                _DOT_R * 2, _DOT_R * 2)

        # Distance label
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy) or 1
        self._draw_label(painter, self._fmt(length),
                         (x1 + x2) / 2, (y1 + y2) / 2,
                         _LINE_COLOR, offset_y=-18)

        painter.end()

    # ------------------------------------------------------------------
    # Profile extraction
    # ------------------------------------------------------------------

    def _finish(self) -> None:
        if self._image_data is None:
            return

        # Map overlay coords → scene coords → image pixel coords
        p1_scene = self._view.mapToScene(self._start)
        p2_scene = self._view.mapToScene(self._end)

        x1, y1 = p1_scene.x(), p1_scene.y()
        x2, y2 = p2_scene.x(), p2_scene.y()

        num_samples = max(int(math.hypot(x2 - x1, y2 - y1)), 2)
        xs = np.linspace(x1, x2, num_samples)
        ys = np.linspace(y1, y2, num_samples)

        h, w = self._image_data.shape[:2]
        xs = np.clip(xs, 0, w - 1).astype(int)
        ys = np.clip(ys, 0, h - 1).astype(int)

        intensities = self._image_data[ys, xs].astype(float)

        # Build distance axis
        pixel_length = math.hypot(x2 - x1, y2 - y1)
        if self._pixel_size is not None:
            real_length = pixel_length * self._pixel_size
            distances = np.linspace(0, real_length, num_samples)
            # Pick best SI unit
            if real_length < 1e-6:
                distances *= 1e9;  unit = "nm"
            elif real_length < 1e-3:
                distances *= 1e6;  unit = "µm"
            else:
                distances *= 1e3;  unit = "mm"
        else:
            distances = np.linspace(0, pixel_length, num_samples)
            unit = "px"

        self.plot_widget.update_profile(distances, intensities, unit=unit)

        if self._on_measure:
            self._on_measure(distances, intensities)
