"""Correlation point-picking on the shared canvas stack (FIB-535).

`FibsemImageCanvas` + :class:`CorrelationPointOverlay`, exposing the same signals
and methods as :class:`ImagePointCanvas` so the eventual swap in
``correlation_tab_widget`` is a construction-site change and nothing more.

**Not wired in yet.** `ImagePointCanvas` remains what the correlation tab uses;
this is built alongside so each piece can be reviewed and tested on its own. The
old canvas is deleted only in the last PR of the series, once all three of its
consumers have moved.

Skeleton scope — image display, points, selection, and the add-menu. Still to
come: legend, per-point labels, `render_to_axes`, and the reprojected-result
overlay (`add_overlay_points` / `clear_overlay`).

What comes free with the shared canvas, and is the reason for the whole exercise:
**contrast and gamma**, which `ImagePointCanvas` has never had — on FM data,
where they are exactly the controls an operator wants while picking points. Also
zoom/pan, the scalebar, the toolbar and the 11 other overlay types.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QAction, QMenu, QVBoxLayout, QWidget

from fibsem.correlation.structures import Coordinate, PointType
from fibsem.structures import FibsemImage
from fibsem.ui.correlation.widgets.correlation_point_overlay import (
    CorrelationPointOverlay,
)
from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas


class CorrelationCanvasWidget(QWidget):
    """Image + draggable correlation points, on the shared canvas.

    Signal shapes are deliberately identical to ``ImagePointCanvas``: the tab
    widget's four handlers connect unchanged when this replaces it.
    """

    point_selected = pyqtSignal(object)  # Coordinate
    point_moved = pyqtSignal(object)  # Coordinate
    point_removed = pyqtSignal(object)  # Coordinate
    point_add_requested = pyqtSignal(float, float, object)  # x, y, PointType

    def __init__(
        self,
        allowed_point_types: Optional[List[PointType]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        # None means "every type"; an empty list means "adding is off" -- the same
        # convention ImagePointCanvas uses, kept so the swap does not silently
        # re-enable adding in the result widget, which passes [].
        self._allowed_types = allowed_point_types

        self.canvas = FibsemImageCanvas()
        self.points = CorrelationPointOverlay()
        self.canvas.add_overlay(self.points)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        # Identity signals straight through; only the add path needs translating,
        # because the overlay reports *where* and this widget decides *what*.
        self.points.coordinate_selected.connect(self.point_selected)
        self.points.coordinate_moved.connect(self.point_moved)
        self.points.coordinate_removed.connect(self.point_removed)
        self.points.add_requested.connect(self._show_add_menu)

    # ── the _CanvasAdapter surface ────────────────────────────────────────

    def set_coordinates(self, coords: List[Coordinate]) -> None:
        self.points.set_coordinates(coords)

    def set_selected(self, coord: Optional[Coordinate]) -> None:
        self.points.set_selected_coordinate(coord)

    def refresh_coordinate(self, coord: Coordinate) -> None:
        self.points.refresh_coordinate(coord)

    # ── image ─────────────────────────────────────────────────────────────

    def set_image(self, image: FibsemImage, cmap: str = "gray") -> None:
        self.canvas.set_image(image, cmap=cmap)

    def update_display(
        self, image: np.ndarray, pixel_size: Optional[float] = None
    ) -> None:
        """Replace the displayed array, keeping zoom/pan across a same-size update.

        ``ImagePointCanvas.update_display`` took an array and nothing else; the
        optional *pixel_size* is the shared canvas's scalebar input, and omitting
        it leaves the current value alone.
        """
        self.canvas.update_display(image, pixel_size=pixel_size)

    def set_pixel_size(self, pixel_size_m: float) -> None:
        self.canvas.pixel_size = pixel_size_m

    def reset_view(self) -> None:
        self.canvas.reset_view()

    # ── add menu ──────────────────────────────────────────────────────────

    def _show_add_menu(self, x: float, y: float) -> None:
        """Ask which PointType to add at *x*, *y*, then report it.

        The overlay cannot do this itself -- it is a QObject, not a widget, so it
        has no parent to hang a QMenu from, and the allowed-type list is per-side
        config the tab widget already owns. No stylesheet: QMenu is styled
        app-wide in napari_style, and ImagePointCanvas overriding that locally is
        the anomaly, not the rule.

        Emits nothing when adding is disabled (``allowed_point_types=[]``) or the
        menu is dismissed -- the point is created by the caller, not here, so the
        Coordinate still gets its z from the adapter as it does today.
        """
        types = self._allowed_types if self._allowed_types is not None else list(PointType)
        if not types:
            return

        menu = QMenu(self)
        for pt in types:
            action = QAction(f"Add {pt.value}", self)
            action.setData((x, y, pt))
            menu.addAction(action)

        chosen = menu.exec_(QCursor.pos())
        if chosen is not None:
            cx, cy, pt = chosen.data()
            self.point_add_requested.emit(cx, cy, pt)
