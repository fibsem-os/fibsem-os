"""An image laid over the canvas through a transform of its own (FIB-1030).

A fluorescence overview on the FIB/SEM Overview tab, to begin with: a picture of the
sample that has to be *placed* against the picture underneath rather than drawn where
its own metadata says, because the two instruments looked at the sample from different
sides and the metadata gets it close, not right. So the image carries a placement --
centre, turn, and the view's squash and mirror -- and is drawn through it, and the
gesture in :class:`~fibsem.ui.widgets.canvas.overlays.transform_overlay.TransformGestureOverlay`
moves and turns it.

Drawn the way the milling overlay draws a rotated bitmap: an ``AxesImage`` on the unit
square, mapped onto the canvas through a transform, so the pixels are never resampled
and a turn costs nothing. Built directly rather than via ``imshow`` -- that would
autoscale the axes to the image and throw away the view.

A stretch-to-fit overlay is not an alignment (FIB-355): the image's size on the canvas
comes from its own pixel size, and the only things the user moves are the four numbers
a similarity has. Anything the view does to it -- the squash of a tilted view, the
mirror of the far side of the grid -- is the geometry's, supplied by the host.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from fibsem.ui.widgets.canvas.overlays.transform_overlay import (
    TransformGestureOverlay,
)

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from PyQt5.QtCore import QObject

logger = logging.getLogger(__name__)

_OUTLINE_COLOUR = "#4dd0e1"
_DEFAULT_OPACITY = 0.6


class ImageOverlay(TransformGestureOverlay):
    """An RGB(A) image on the canvas, placed through a transform.

    :meth:`set_image` gives the pixels and their footprint; :meth:`set_placement` (the
    base class's) says where, turned how, and how the view squashes and mirrors it.
    All in canvas units, converted by the caller, which is the only party that knows
    the frame.
    """

    def __init__(
        self,
        *,
        opacity: float = _DEFAULT_OPACITY,
        zorder: float = 4.0,
        parent: Optional["QObject"] = None,
    ) -> None:
        super().__init__(parent)
        self._data: Optional[np.ndarray] = None
        self._width: float = 0.0  # canvas units, along the image's own axes
        self._height: float = 0.0
        self._opacity: float = float(opacity)
        self._zorder = zorder

    # ── public API ────────────────────────────────────────────────────────

    def set_image(
        self,
        data: np.ndarray,
        width: float,
        height: float,
        centre: Optional[Tuple[float, float]] = None,
        rotation: Optional[float] = None,
        squash: Optional[float] = None,
        mirror: Optional[bool] = None,
    ) -> None:
        """The pixels, as an (H, W, 3|4) array, and the ground they cover.

        `width` and `height` are the image's footprint in canvas units along its own
        axes, before the view squashes it -- its pixel count times its pixel size,
        through the frame. Row 0 is drawn at the top of the footprint, image-fashion.
        The placement arguments are optional here so a caller can swap the pixels
        without moving the image; omitted, the current placement stands.
        """
        data = np.asarray(data)
        if data.ndim != 3 or data.shape[2] not in (3, 4):
            raise ValueError(
                f"expected an (H, W, 3|4) RGB(A) image, got shape {data.shape}"
            )
        if data.shape[2] == 3:
            # Always with an alpha channel: a turned image is resampled into an
            # axis-aligned buffer, and without alpha matplotlib fills the corners
            # outside the footprint with opaque black. With it they are clear.
            opaque = 255 if data.dtype == np.uint8 else 1.0
            data = np.dstack([data, np.full(data.shape[:2], opaque, dtype=data.dtype)])
        self._data = data
        self._width = float(width)
        self._height = float(height)
        self.set_placement(
            centre if centre is not None else (self._centre or (0.0, 0.0)),
            self._rotation if rotation is None else rotation,
            self._squash if squash is None else squash,
            self._mirror if mirror is None else mirror,
        )

    def set_opacity(self, opacity: float) -> None:
        self._opacity = max(0.0, min(1.0, float(opacity)))
        self._redraw()

    @property
    def opacity(self) -> float:
        return self._opacity

    @property
    def footprint(self) -> Tuple[float, float]:
        """(width, height) in canvas units along the image's own axes."""
        return self._width, self._height

    def corners(self):
        """The footprint's four corners on the canvas, top-left first, clockwise."""
        w, h = self._width / 2.0, self._height / 2.0
        return [self.to_canvas(u, v) for u, v in ((-w, -h), (w, -h), (w, h), (-w, h))]

    # ── the body ──────────────────────────────────────────────────────────

    def _body_contains(self, x: float, y: float) -> bool:
        if self._data is None:
            return False
        u, v = self.from_canvas(x, y)
        return abs(u) <= self._width / 2.0 and abs(v) <= self._height / 2.0

    def _draw_body(self, animated: bool) -> list:
        from matplotlib.image import AxesImage
        from matplotlib.patches import Polygon
        from matplotlib.transforms import Affine2D

        if self._data is None or self._width <= 0 or self._height <= 0:
            return []
        # The unit square onto the footprint, centred, then through the body's own
        # map onto the canvas. `origin="lower"` because the canvas's y runs down: v=0
        # is the footprint's top edge there, and row 0 belongs at the top.
        onto_footprint = (
            Affine2D()
            .scale(self._width, self._height)
            .translate(-self._width / 2.0, -self._height / 2.0)
        )
        transform = onto_footprint + self.body_transform() + self._ax.transData
        image = AxesImage(
            self._ax,
            extent=(0, 1, 0, 1),
            origin="lower",
            zorder=self._zorder,
            interpolation="nearest",
        )
        image.set_data(self._data)
        image.set_alpha(self._opacity)
        image.set_transform(transform)
        image.set_clip_path(self._ax.patch)
        image.set_animated(animated)
        self._ax.add_artist(image)
        artists = [image]
        if self.is_editable():
            outline = Polygon(
                self.corners(),
                closed=True,
                facecolor="none",
                edgecolor=_OUTLINE_COLOUR,
                linewidth=1.0,
                linestyle=(0, (4, 3)),
                zorder=self._zorder + 0.5,
                animated=animated,
            )
            self._ax.add_patch(outline)
            artists.append(outline)
        return artists
