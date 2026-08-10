"""Correlation point-picking on the shared FM canvas (FIB-535).

`FMCanvasWidget` + :class:`CorrelationPicking` — the FM half of the migration,
and the counterpart to :class:`CorrelationCanvasWidget` on the FIB side. Both
present the same picking surface; they differ only in how an image reaches the
canvas, which is the one thing that genuinely differs between them.

**Not wired in yet.** `fm_image_display_widget.FMImageDisplayWidget` is still what
the correlation tab uses; this is built alongside so it can be reviewed and
compared side by side first. Both old widgets are deleted in the last PR of the
series.

Replacing `FMImageDisplayWidget` rather than porting it is what puts the
correlation FM display on the *standard* FM control set — the z-slider, the
per-channel layers popover and the max-projection toolbar button that the quad
view and the FM overview already use — instead of a private second
implementation of all three. The visible trades, agreed 2026-08-10: per-channel
intensity moves from a clip percentage to the layers panel's contrast/gamma, and
the channel rows move from an always-visible pane into the popover.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget

from fibsem.correlation.structures import Coordinate, PointType
from fibsem.ui.correlation.widgets.correlation_picking import CorrelationPicking
from fibsem.ui.widgets.canvas.fm_canvas import FMCanvasWidget


class CorrelationFMCanvasWidget(FMCanvasWidget):
    """Multi-channel FM display with draggable correlation points on top.

    Signal shapes are deliberately identical to ``FMImageDisplayWidget``, which is
    in turn identical to ``ImagePointCanvas``: the tab widget wires both of its
    surfaces in one loop, and that loop does not change.
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

        self.picking = CorrelationPicking(
            self.canvas,
            allowed_point_types=allowed_point_types,
            menu_parent=self,
        )
        self.picking.point_selected.connect(self.point_selected)
        self.picking.point_moved.connect(self.point_moved)
        self.picking.point_removed.connect(self.point_removed)
        self.picking.point_add_requested.connect(self.point_add_requested)

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
        self._z_slider.setToolTip("Drag to change Z-slice, or Shift+scroll on the image")

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

    # ── chrome, mirroring CorrelationCanvasWidget ─────────────────────────

    def set_legend_visible(self, visible: bool) -> None:
        self.picking.set_legend_visible(visible)

    def set_labels_visible(self, visible: bool) -> None:
        self.picking.set_labels_visible(visible)

    def set_scalebar_visible(self, visible: bool) -> None:
        self.canvas.set_scalebar_visible(visible)

    # No set_pixel_size override: FMCanvasWidget's also records `_pixel_size`,
    # which the compositing path reads. Shadowing it would drop that quietly.

    def reset_view(self) -> None:
        self.canvas.reset_view()

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
