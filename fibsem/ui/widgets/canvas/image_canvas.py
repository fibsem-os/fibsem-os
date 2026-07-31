"""FibsemImageCanvas — a single FibsemImage on a :class:`FibsemCanvasBase`.

Holds exactly one image artist, drawn at the origin with ``extent=(-0.5, w-0.5,
h-0.5, -0.5)``, so canvas coordinates *are* image pixels. All navigation, overlay,
toolbar and chrome behaviour comes from the base class.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from fibsem.structures import FibsemImage
from fibsem.ui.widgets.canvas.canvas_base import (
    _MAX_DISPLAY_PX,
    FibsemCanvasBase,
    _downsample,
)

_logger = logging.getLogger(__name__)


class FibsemImageCanvas(FibsemCanvasBase):
    """Reusable matplotlib canvas for FibsemImage.

    * Scroll-wheel zoom centred on cursor
    * Left-drag pan on empty area
    * Pluggable overlay objects via add_overlay() / remove_overlay()
    * Optional scalebar (auto-populated from FibsemImage.metadata.pixel_size)
    """

    def _content_extent(self) -> Optional[Tuple[float, float, float, float]]:
        """The image artist's extent, or None before any image is set."""
        imgs = self._ax.get_images()
        if not imgs:
            return None
        return imgs[0].get_extent()

    # ── public API ────────────────────────────────────────────────────────

    def set_image(self, image: FibsemImage, cmap: str = "gray") -> None:
        """Display a FibsemImage.  Notifies all registered overlays."""
        pixel_size = None
        try:
            if image.metadata and image.metadata.pixel_size:
                pixel_size = image.metadata.pixel_size.x
        except Exception:
            pass
        self.set_array(image.filtered_data, pixel_size=pixel_size, cmap=cmap)

    def set_array(
        self,
        arr: np.ndarray,
        pixel_size: Optional[float] = None,
        cmap: str = "gray",
    ) -> None:
        """Display a raw 2-D (grayscale) or HxWx3 (RGB) array.

        The lower-level entry point behind :meth:`set_image`, for composites/RGB
        that have no backing ``FibsemImage`` (e.g. the multi-channel FM canvas).
        *pixel_size* (metres/px) drives the scalebar; ``None`` leaves the current
        value unchanged.  Notifies all registered overlays.
        """
        arr = np.asarray(arr)
        h, w = arr.shape[:2]
        # Preserve the current zoom/pan across a same-resolution update — otherwise live
        # acquisition re-frames on every frame. Auto-fit only on the first image or a
        # resolution change; the toolbar "fit to view" (reset_view) still fits on demand.
        had_image = self._img_w is not None
        prev_wh = (self._img_w, self._img_h)
        prev_xlim, prev_ylim = self._ax.get_xlim(), self._ax.get_ylim()
        self._img_w, self._img_h = w, h

        self._ax.cla()
        self._ax.set_facecolor(self._facecolor)
        self._ax.axis("off")
        self._fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        self._display_base = _downsample(arr, _MAX_DISPLAY_PX)
        self._is_gray = arr.ndim == 2
        self._norm = None  # recomputed lazily when contrast is engaged
        extent = (-0.5, w - 0.5, h - 0.5, -0.5)
        kw = dict(
            origin="upper", aspect="equal", interpolation="nearest", extent=extent
        )
        to_show, clim = self._contrast_display()
        if self._is_gray:
            im = self._ax.imshow(to_show, cmap=cmap, **kw)
            if clim is not None:
                im.set_clim(*clim)
        else:
            self._ax.imshow(to_show, **kw)

        if had_image and prev_wh == (w, h):
            self._ax.set_xlim(prev_xlim)  # same resolution -> keep the user's zoom/pan
            self._ax.set_ylim(prev_ylim)
        else:
            self._fit_view()

        # Scalebar
        self._scalebar_artist = None
        if pixel_size and pixel_size > 0:
            self._pixel_size = pixel_size
        self._refresh_scalebar()
        self._refresh_crosshair()
        self._refresh_hint()  # axes was cleared above; restore the remembered hint
        self._refresh_title()  # ditto: restore the remembered title
        self._refresh_info_bar()  # ditto: restore the remembered info bar
        self._refresh_live_badge()  # ditto: keep the LIVE badge across streamed frames
        self._refresh_flash()  # ditto: keep a live flash (e.g. WD scroll) visible across frames
        self._refresh_legend()  # ditto: restore the patch legend

        for overlay in self._overlays:
            try:
                overlay.on_image_changed(w, h)
            except Exception:
                _logger.exception("Overlay on_image_changed failed: %r", overlay)

        self.draw_idle()

    def update_display(self, arr: np.ndarray, pixel_size: Optional[float] = None) -> None:
        """Fast pixel-data swap without resetting overlays.

        Use for z-slice navigation / recomposites where image dimensions don't change.
        *pixel_size* (metres/px) updates the scalebar if it changed — important when the
        same-shape swap actually carries a different scale (e.g. a real overview replacing
        a blank placeholder of matching pixel dimensions). ``None`` leaves it unchanged.
        Falls back to a no-op if no image has been set yet.
        """
        imgs = self._ax.get_images()
        if not imgs:
            return
        self._display_base = _downsample(arr, _MAX_DISPLAY_PX)
        self._is_gray = arr.ndim == 2
        self._norm = None
        to_show, clim = self._contrast_display()
        imgs[0].set_data(to_show)
        if clim is not None:
            imgs[0].set_clim(*clim)
        elif self._is_gray:
            # Rescale to the new data, matching set_array's imshow autoscale. Without
            # this the frame is drawn against the *previous* frame's intensity range:
            # scrubbing back to a dim FM timelapse frame after an intensity drop
            # rendered it near-black. RGB is excluded — imshow doesn't scale it either.
            imgs[0].set_clim(float(to_show.min()), float(to_show.max()))
        if pixel_size and pixel_size > 0 and pixel_size != self._pixel_size:
            self._pixel_size = pixel_size
            self._refresh_scalebar()
        self.draw_idle()

