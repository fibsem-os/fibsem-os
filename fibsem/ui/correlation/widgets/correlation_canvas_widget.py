"""Correlation point-picking on the shared canvas stack (FIB-535).

`FibsemImageCanvas` + :class:`CorrelationPicking`. This is what the correlation
tab's FIB panel is; :class:`CorrelationFMCanvasWidget` is its FM counterpart, and
the two share all of their point-picking behaviour.

Replaced `ImagePointCanvas`, a 1030-line canvas correlation owned outright, which
is why several comments here explain a choice by reference to it. That class is
gone; the references are history, not a live dependency.

Everything about *picking* lives in :mod:`correlation_picking`, shared with the
FM side, which puts the identical behaviour on an `FMCanvasWidget` instead. What
stays here is the half that genuinely differs between the two: how an image gets
onto the canvas.

What comes free with the shared canvas, and is the reason for the whole exercise:
**contrast and gamma**, which `ImagePointCanvas` has never had — on FM data,
where they are exactly the controls an operator wants while picking points. Also
zoom/pan, the scalebar, the toolbar and the 11 other overlay types.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QVBoxLayout, QWidget

from fibsem.correlation.structures import Coordinate, PointType
from fibsem.ui.correlation.widgets.correlation_picking import CorrelationPicking
from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas


class CorrelationCanvasWidget(QWidget):
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
    ) -> None:
        super().__init__(parent)
        self.canvas = FibsemImageCanvas()
        # The shared canvas draws a yellow "+" at image centre by default and the
        # old correlation canvas drew none. Correlation draws SURFACE as an orange
        # "+", so the two are easy to confuse on the one point whose whole job is
        # marking a position -- and it would land in a saved plot. The toolbar
        # toggle still brings it back.
        self.canvas.set_crosshair_visible(False)

        self.picking = CorrelationPicking(
            self.canvas,
            allowed_point_types=allowed_point_types,
            menu_parent=self,
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        # Forwarded rather than left on `.picking`, so the tab widget and its
        # _CanvasAdapter connect to this exactly as they do to ImagePointCanvas.
        self.picking.point_selected.connect(self.point_selected)
        self.picking.point_moved.connect(self.point_moved)
        self.picking.point_removed.connect(self.point_removed)
        self.picking.point_add_requested.connect(self.point_add_requested)

    @property
    def points(self):
        """The interactive point overlay."""
        return self.picking.points

    @property
    def results(self):
        """The static result-marker overlay."""
        return self.picking.results

    # ── the _CanvasAdapter surface ────────────────────────────────────────

    def set_coordinates(self, coords: List[Coordinate]) -> None:
        self.picking.set_coordinates(coords)

    def set_selected(self, coord: Optional[Coordinate]) -> None:
        self.picking.set_selected(coord)

    def refresh_coordinate(self, coord: Coordinate) -> None:
        self.picking.refresh_coordinate(coord)

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

    def reset_view(self) -> None:
        self.canvas.reset_view()

    # ── chrome ────────────────────────────────────────────────────────────

    def set_legend_visible(self, visible: bool) -> None:
        """Show or hide the point-type legend (one swatch per type on screen)."""
        self.picking.set_legend_visible(visible)

    def set_labels_visible(self, visible: bool) -> None:
        """Show or hide the per-point names, the result markers' numbers included."""
        self.picking.set_labels_visible(visible)

    def set_scalebar_visible(self, visible: bool) -> None:
        """Show or hide the scalebar."""
        self.canvas.set_scalebar_visible(visible)

    # ── result markers ────────────────────────────────────────────────────

    def add_overlay_points(
        self,
        points: Sequence[Tuple[float, float]],
        *,
        color: str = "#ff0000",
        label_prefix: str = "",
        size: float = 7.0,
        marker: str = "o",
        alpha: float = 1.0,
        show_labels: bool = True,
        hollow: bool = False,
        legend_label: Optional[str] = None,
    ) -> None:
        """Append a group of non-interactive result markers."""
        self.picking.add_overlay_points(
            points,
            color=color,
            label_prefix=label_prefix,
            size=size,
            marker=marker,
            alpha=alpha,
            show_labels=show_labels,
            hollow=hollow,
            legend_label=legend_label,
        )

    def clear_overlay(self) -> None:
        """Remove every marker added via :meth:`add_overlay_points`."""
        self.picking.clear_overlay()
