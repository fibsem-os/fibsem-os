"""Correlation point-picking on the shared FM canvas (FIB-535).

`FMCanvasWidget` + :class:`PointPicking` — the FM half of the migration,
and the counterpart to :class:`PointCanvasWidget` on the FIB side. Both
present the same picking surface; they differ only in how an image reaches the
canvas, which is the one thing that genuinely differs between them.

Replaced `FMImageDisplayWidget`, a 574-line parallel implementation of the FM
canvas. Replacing rather than porting it is what puts the
correlation FM display on the *standard* FM control set — the z-slider, the
per-channel layers popover and the max-projection toolbar button that the quad
view and the FM overview already use — instead of a private second
implementation of all three. The visible trades, agreed 2026-08-10: per-channel
intensity moves from a clip percentage to the layers panel's contrast/gamma, and
the channel rows move from an always-visible pane into the popover.
"""

from __future__ import annotations

from typing import List, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget

from fibsem.correlation.structures import PointType
from fibsem.ui.widgets.canvas.fm_canvas import FMCanvasWidget
from fibsem.ui.widgets.canvas.point_picking import (
    PickingCanvasMixin,
    PointPicking,
)
from fibsem.ui.widgets.point_store import PointStore


class CorrelationFMCanvasWidget(PickingCanvasMixin, FMCanvasWidget):
    """Multi-channel FM display with draggable correlation points on top.

    The tab widget wires both of its surfaces in one loop, so the four signals
    here and on ``PointCanvasWidget`` have to stay identically named.
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
        # Same reasoning as the FIB canvas: correlation draws SURFACE as an orange
        # "+", which the yellow centre "+" is easy to mistake for. Toolbar toggle
        # brings it back.
        self.canvas.set_crosshair_visible(False)
        # Planes, not a projection. FMCanvasWidget defaults to max-projection
        # because the quad view is a viewer; correlation is a picker, and
        # `current_z` has no meaningful answer under a projection — every point
        # picked on it would silently take z=0. FMImageDisplayWidget defaulted
        # the same way, with its Max Projection checkbox unticked.
        self.set_max_projection(False)

        self.picking = PointPicking(
            self.canvas,
            allowed_point_types=allowed_point_types,
            menu_parent=self,
            store=store,
            side=side,
        )
        self._connect_picking_signals()

        # Shift+scroll steps the z-plane, as it did on FMImageDisplayWidget.
        # `canvas_scrolled` rather than a bespoke signal: the shared canvas
        # already broadcasts modified scroll and already declines to zoom under
        # any modifier, and two widgets consume it in exactly this shape --
        # objective_control_widget and beam_settings_widget.
        self.canvas.canvas_scrolled.connect(self._on_canvas_scrolled)
        # Shift+scroll has no visible affordance, so the slider tooltip is the
        # only place it can be discovered. Set here rather than on FMCanvasWidget
        # because only this widget wires it -- the quad view would be advertising
        # something that does nothing.
        self._z_slider.setToolTip(
            "Drag to change Z-slice, or Shift+scroll on the image"
        )

    def set_fm_image(self, image) -> None:
        """Load a stack and start in the middle of it.

        `FMCanvasWidget` starts at plane 0, which is fine for a viewer that opens
        on the max projection. A picker opens on planes, and plane 0 is the edge
        of the volume — usually out of focus, so every operator would scrub to
        the middle before picking anything. `FMImageDisplayWidget` started at
        ``n_z // 2`` for that reason, and the swap onto the shared canvas lost it.
        """
        super().set_fm_image(image)
        if self._z_max > 0:
            self._z_slider.setValue(self._z_max // 2 + self._z_max % 2)

    def _on_canvas_scrolled(self, x, y, direction: int, modifiers) -> None:
        """Shift+scroll → one z-plane. Plain scroll is left to the canvas (zoom).

        Note for macOS: matplotlib's Qt backend reads only the vertical wheel
        delta, and AppKit turns Shift+wheel horizontal on a discrete mouse, so
        this never fires there. Windows and Linux are unaffected; the ‹ › buttons
        and the slider work everywhere. FIB-552, deliberately parked.
        """
        if "Shift" not in modifiers:
            return
        self.step_z(direction)

    # No set_pixel_size override: FMCanvasWidget's also records `_pixel_size`,
    # which the compositing path reads. Shadowing it would drop that quietly.
