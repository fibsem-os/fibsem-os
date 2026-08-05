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

from typing import TYPE_CHECKING, Optional

from PyQt5.QtCore import pyqtSignal

from fibsem.structures import FibsemRectangle
from fibsem.ui.widgets.canvas.overlays.rect_overlay import RectOverlay

if TYPE_CHECKING:
    from fibsem.ui.widgets.canvas.canvas_base import ContentRect

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

    def on_content_changed(self, rect: "ContentRect") -> None:
        # super() rebuilds the artists (fresh + visible); re-position then re-apply
        # the desired visibility, otherwise a new image un-hides a cleared overlay.
        super().on_content_changed(rect)
        if self._norm_area is not None and not rect.is_empty:
            self.set_area(self._norm_area)
        self._apply_visibility()

    # ── public API ────────────────────────────────────────────────────────

    def set_area(self, reduced_area: FibsemRectangle) -> None:
        """Position the rectangle from a normalized ``FibsemRectangle``.

        A reduced area is defined as fractions *of an image*, so unlike the other
        overlays this one stays image-relative: it is meaningful only on a canvas whose
        content is a single frame, not on one showing many images in stage space.
        """
        self._norm_area = reduced_area
        if self._rect is None or self._rect.is_empty:
            return
        self.set_rect(
            self._rect.x0 + reduced_area.left * self._rect.width,
            self._rect.y0 + reduced_area.top * self._rect.height,
            reduced_area.width * self._rect.width,
            reduced_area.height * self._rect.height,
        )

    def get_area(self) -> FibsemRectangle:
        """Return the current rectangle as a normalized ``FibsemRectangle``."""
        if self._rect is None or self._rect.is_empty:
            return FibsemRectangle()
        d = self.get_rect()
        return FibsemRectangle(
            left=(d["x0"] - self._rect.x0) / self._rect.width,
            top=(d["y0"] - self._rect.y0) / self._rect.height,
            width=d["width"] / self._rect.width,
            height=d["height"] / self._rect.height,
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
