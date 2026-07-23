"""Display-only segmentation-mask overlay for FibsemImageCanvas.

Alpha-blends a label array (per-pixel class indices) over the image using a
per-class colour map (``CLASS_COLORS_RGB`` by default), with the background class
rendered fully transparent. Display-only — captures no mouse events, so it
coexists with pan/zoom and any interactive overlay (e.g. the detection feature
points) on the same canvas.

Lifecycle: add once via ``canvas.add_overlay(overlay)``; draws nothing until
:meth:`set_mask` is called; :meth:`clear` (or ``set_mask(None)``) removes it.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

from fibsem.ui.widgets.canvas.overlays.base import CanvasOverlay

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas

_logger = logging.getLogger(__name__)

_MASK_ALPHA = 0.3
_MASK_ZORDER = 4  # above the base image, below patterns (6) / crosshair (7) / points (8)
_BACKGROUND_CLASS = 0  # rendered transparent


class MaskOverlay(CanvasOverlay):
    """Alpha-blended label-mask overlay (display-only)."""

    def __init__(self) -> None:
        self._ax = None
        self._canvas: Optional[FibsemImageCanvas] = None
        self._artist = None  # AxesImage
        self._mask: Optional[np.ndarray] = None
        self._colors: Optional[Sequence] = None
        self._alpha: float = _MASK_ALPHA

    # ── overlay protocol ──────────────────────────────────────────────────

    def attach(self, ax, canvas: FibsemImageCanvas) -> None:
        self._ax = ax
        self._canvas = canvas

    def detach(self) -> None:
        self._remove_artist()
        self._ax = None
        self._canvas = None

    def on_image_changed(self, width: int, height: int) -> None:
        # ax was cleared + a new image drawn; re-create from the cached mask
        self._remove_artist()
        if self._mask is not None and width > 0:
            self._draw()

    # ── public API ────────────────────────────────────────────────────────

    def set_mask(
        self,
        mask: Optional[np.ndarray],
        colors: Optional[Sequence] = None,
        alpha: float = _MASK_ALPHA,
    ) -> None:
        """Display *mask* (HxW integer class indices).

        ``colors`` is an index→(r, g, b) sequence (0-255); defaults to
        ``CLASS_COLORS_RGB``. The background class (0) is always transparent;
        every other class is drawn at ``alpha``. Pass ``mask=None`` to clear.
        """
        if mask is None:
            self.clear()
            return
        self._mask = np.asarray(mask)
        self._colors = colors
        self._alpha = alpha
        self._remove_artist()
        self._draw()
        if self._canvas is not None:
            self._canvas.draw_idle()

    def clear(self) -> None:
        """Remove the mask (no segmentation → nothing drawn)."""
        self._mask = None
        self._remove_artist()
        if self._canvas is not None:
            self._canvas.draw_idle()

    # ── drawing ───────────────────────────────────────────────────────────

    def _colours_rgb(self) -> Sequence:
        if self._colors is not None:
            return self._colors
        from fibsem.segmentation.config import CLASS_COLORS_RGB
        return CLASS_COLORS_RGB

    def _mask_to_rgba(self, mask: np.ndarray) -> np.ndarray:
        """Map an HxW label array to an HxWx4 float RGBA image via a class LUT.

        Background (class 0) is transparent; other classes use ``_alpha``. Using a
        direct RGBA image (rather than cmap + scalar alpha) keeps per-pixel alpha,
        so the background stays clear instead of dimming the underlying image.
        """
        colours = self._colours_rgb()
        lut = np.zeros((len(colours), 4), dtype=float)
        for i, c in enumerate(colours):
            lut[i, :3] = [v / 255.0 for v in c[:3]]
            lut[i, 3] = 0.0 if i == _BACKGROUND_CLASS else self._alpha
        idx = np.clip(mask.astype(int), 0, len(colours) - 1)
        return lut[idx]

    def _draw(self) -> None:
        if self._ax is None or self._mask is None:
            return
        try:
            rgba = self._mask_to_rgba(self._mask)
        except Exception:
            _logger.exception("MaskOverlay: failed to build RGBA from mask")
            return
        h, w = self._mask.shape[:2]
        self._artist = self._ax.imshow(
            rgba,
            interpolation="nearest",
            origin="upper",
            extent=(-0.5, w - 0.5, h - 0.5, -0.5),
            zorder=_MASK_ZORDER,
        )

    def _remove_artist(self) -> None:
        if self._artist is not None:
            try:
                self._artist.remove()
            except Exception:
                pass
            self._artist = None
