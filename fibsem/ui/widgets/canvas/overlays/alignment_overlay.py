"""Editable alignment-area overlay for FibsemImageCanvas.

A reusable ``RectOverlay`` that speaks a normalized ``FibsemRectangle``
(``left, top, width, height`` in [0, 1], top-left origin) and emits one when the
user drags/resizes. Used for the image-alignment / drift-correction reduced area
(image widget — editable; milling — read-only; lamella editor later).

The normalized↔pixel conversion and the ``editable`` toggle mirror the
``feat-proj-volume-milling`` branch (``VolumeMillingImageCanvas`` /
``VolumeSEMAcquisitionCanvas``) so the two converge — that branch's per-canvas
alignment logic can drop down to this overlay when it rebases.
"""
from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import pyqtSignal

from fibsem.structures import FibsemRectangle
from fibsem.ui.widgets.canvas.overlays.rect_overlay import RectOverlay

_ALIGNMENT_COLOUR = "limegreen"


class AlignmentAreaOverlay(RectOverlay):
    """RectOverlay that maps to/from a normalized ``FibsemRectangle``.

    Emits :attr:`alignment_area_changed` (a ``FibsemRectangle``) on drag/resize.
    Pass ``editable=False`` for a read-only display (no handles, no mouse capture).
    """

    alignment_area_changed = pyqtSignal(object)  # FibsemRectangle

    def __init__(
        self,
        color: str = _ALIGNMENT_COLOUR,
        editable: bool = True,
        parent=None,
    ) -> None:
        super().__init__(
            color=color,
            facecolor=None,
            alpha=1.0,
            linewidth=2.5,
            linestyle="--",
            resizable=True,
            parent=parent,
        )
        self._interactive = editable
        self._norm_area: Optional[FibsemRectangle] = None
        self._area_visible = False  # desired visibility, preserved across rebuilds
        self.rect_changed.connect(self._on_rect_changed)

    # ── overlay protocol ──────────────────────────────────────────────────

    def on_image_changed(self, width: int, height: int) -> None:
        # super() rebuilds the artists (fresh + visible); re-position then re-apply
        # the desired visibility, otherwise a new image un-hides a cleared overlay.
        super().on_image_changed(width, height)
        if self._norm_area is not None and self._img_w:
            self.set_area(self._norm_area)
        self._apply_visibility()

    # ── public API ────────────────────────────────────────────────────────

    def set_area(self, reduced_area: FibsemRectangle) -> None:
        """Position the rectangle from a normalized ``FibsemRectangle``."""
        self._norm_area = reduced_area
        if self._img_w is None or self._img_h is None:
            return
        self.set_rect(
            reduced_area.left * self._img_w,
            reduced_area.top * self._img_h,
            reduced_area.width * self._img_w,
            reduced_area.height * self._img_h,
        )

    def get_area(self) -> FibsemRectangle:
        """Return the current rectangle as a normalized ``FibsemRectangle``."""
        if self._img_w is None or self._img_h is None:
            return FibsemRectangle()
        d = self.get_rect()
        return FibsemRectangle(
            left=d["x0"] / self._img_w,
            top=d["y0"] / self._img_h,
            width=d["width"] / self._img_w,
            height=d["height"] / self._img_h,
        )

    def set_visible(self, visible: bool) -> None:
        """Show or hide the rectangle (state preserved across image rebuilds)."""
        self._area_visible = visible
        self._apply_visibility()

    def set_editable(self, editable: bool) -> None:
        """Toggle drag/resize (handles hidden + mouse ignored when disabled)."""
        self._interactive = editable
        self._apply_visibility()

    # ── private ───────────────────────────────────────────────────────────

    def _apply_visibility(self) -> None:
        """Apply current visibility: patch follows ``_area_visible``; handles also
        require ``_interactive`` (read-only overlays show no handles)."""
        if self._patch is not None:
            self._patch.set_visible(self._area_visible)
        for h in self._handles.values():
            h.set_visible(self._area_visible and self._interactive)
        if self._canvas is not None:
            self._canvas.draw_idle()

    def _on_rect_changed(self, _d: dict) -> None:
        # rect_changed only fires on user drag/resize (not on programmatic set_rect)
        self._norm_area = self.get_area()
        self.alignment_area_changed.emit(self._norm_area)
