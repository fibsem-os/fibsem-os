"""Correlation point-picking on the shared canvas stack (FIB-535).

`FibsemImageCanvas` + :class:`PointPicking`. This is what the correlation
tab's FIB panel is; :class:`CorrelationFMCanvasWidget` is its FM counterpart, and
the two share all of their point-picking behaviour.

Replaced `ImagePointCanvas`, a 1030-line canvas correlation owned outright, which
is why several comments here explain a choice by reference to it. That class is
gone; the references are history, not a live dependency.

Everything about *picking* lives in :mod:`point_picking`, shared with the
FM side, which puts the identical behaviour on an `FMCanvasWidget` instead. What
stays here is the half that genuinely differs between the two: how an image gets
onto the canvas.

What comes free with the shared canvas, and is the reason for the whole exercise:
**contrast and gamma**, which `ImagePointCanvas` has never had — on FM data,
where they are exactly the controls an operator wants while picking points. Also
zoom/pan, the scalebar, the toolbar and the 11 other overlay types.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QVBoxLayout, QWidget

from fibsem.correlation.structures import PointType
from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas
from fibsem.ui.widgets.canvas.point_picking import (
    PickingCanvasMixin,
    PointPicking,
)
from fibsem.ui.widgets.point_store import PointStore


class PointCanvasWidget(PickingCanvasMixin, QWidget):
    """Image + draggable correlation points, on the shared canvas.

    The tab widget wires both of its surfaces in one loop, so the four signals
    here and on ``CorrelationFMCanvasWidget`` have to stay identically named.
    """

    point_selected = pyqtSignal(object)  # Coordinate
    point_moved = pyqtSignal(object)  # Coordinate
    point_removed = pyqtSignal(object)  # Coordinate
    point_add_requested = pyqtSignal(float, float, object)  # x, y, PointType

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        allowed_point_types: Optional[List[PointType]] = None,
        store: Optional[PointStore] = None,
        side: Optional[str] = None,
    ) -> None:
        super().__init__(parent)
        self.canvas = FibsemImageCanvas()
        # The shared canvas draws a yellow "+" at image centre by default and the
        # old correlation canvas drew none. Correlation draws SURFACE as an orange
        # "+", so the two are easy to confuse on the one point whose whole job is
        # marking a position -- and it would land in a saved plot. The toolbar
        # toggle still brings it back.
        self.canvas.set_crosshair_visible(False)

        self.picking = PointPicking(
            self.canvas,
            allowed_point_types=allowed_point_types,
            menu_parent=self,
            store=store,
            side=side,
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        # Forwarded rather than left on `.picking`, so the tab widget and its
        # _CanvasAdapter connect to this exactly as they do to ImagePointCanvas.
        self._connect_picking_signals()

    # ── image ─────────────────────────────────────────────────────────────

    def set_image(self, image: np.ndarray, *, cmap: str = "gray") -> None:
        """Display a raw image array, as ``ImagePointCanvas.set_image`` does.

        An array rather than a ``FibsemImage`` because that is what both
        correlation consumers hold here -- a filtered FIB frame, an additive FM
        channel composite -- and neither has a FibsemImage to hand.
        ``self.canvas.set_image`` still takes one, for a caller that does.
        """
        self.canvas.set_array(np.asarray(image), cmap=cmap)

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
