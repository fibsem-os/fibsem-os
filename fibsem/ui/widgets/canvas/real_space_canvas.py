"""FibsemRealSpaceCanvas — many images placed where they were acquired.

Where :class:`~fibsem.ui.widgets.canvas.image_canvas.FibsemImageCanvas` shows one image
filling the view, this shows *space*: a backdrop onto which each image is drawn at its
own position and scale. Images acquired at different stage positions line up; images
acquired at different pixel sizes compose at their true relative sizes.

Coordinates
-----------
Positions are supplied in **metres**, in the plane the canvas is showing, with ``+x``
right and ``+y`` down (image convention). Converting a *stage* position into that plane
is the caller's job — see :mod:`fibsem.fm.reprojection` — which keeps this canvas free
of microscope geometry and usable for FM and FIB/SEM alike.

Internally the axes are in **canvas pixels at a reference pixel size**: a position in
metres divided by :attr:`reference_pixel_size`. Metres would be the more natural unit,
but the scalebar, ruler and view margin all assume pixel-ish magnitudes, and overlays
would need converting. Matplotlib does the placement and scaling through each artist's
``extent``, so nothing is resampled and no backing raster is allocated — memory is just
the sum of the images.

The reference pixel size defaults to that of the first image added, so a canvas holding
a single image has axes in that image's own pixels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from fibsem.ui.stylesheets import GRAY_CANVAS_COLOR
from fibsem.ui.widgets.canvas.canvas_base import (
    EMPTY_CONTENT,
    ContentRect,
    FibsemCanvasBase,
    _downsample,
)

_logger = logging.getLogger(__name__)

_DEFAULT_BACKGROUND = GRAY_CANVAS_COLOR  # empty space reads as "nothing acquired here"
_ORIGIN_MARKER_SIZE = 11  # points; fixed on screen, so zoom-independent
# Red, matching the grid boundary this marks the centre of. The current stage
# position is the yellow one -- the pair have to stay distinguishable.
_ORIGIN_MARKER_COLOUR = "#ff5252"

# Pixels kept per placed image. Every artist is redrawn in full on each pan/zoom, so a
# redraw costs the *total* stored pixels — 100 tiles kept at 1024 px square take ~2.7 s,
# against ~0.75 s at 512. And 512 is already more than the display can use for the case
# this canvas exists for: tiles in a 3x3 occupy ~330 screen px each, in a 10x10 ~100 px.
# It only costs detail when zooming deep into one image; see FIB-414 for the real fix.
_DEFAULT_DISPLAY_PX = 512


def _require_displayable(data: np.ndarray) -> None:
    """Reject anything that is not a single displayable picture, before anything mutates.

    This canvas places pictures, not stacks: z is projected and channels are composited
    *before* an image gets here, because neither has a single answer once many images are
    on screen at once. Scrubbing z across tiles that were acquired over different ranges,
    or contrasting one tile independently of its neighbours, are questions the placement
    layer cannot answer — so they belong to whatever owns the channels.

    Without this check matplotlib raises from inside ``imshow`` instead, by which point
    :meth:`FibsemRealSpaceCanvas.add_image` has already dropped the image previously held
    under that key — so a bad call would destroy a good picture.
    """
    if data.ndim == 2:
        return
    if data.ndim == 3 and data.shape[2] in (3, 4):
        return
    raise ValueError(
        f"expected a 2-D image or (H, W, 3|4) RGB(A), got shape {data.shape}. "
        "Project z and composite channels before placing."
    )


@dataclass
class PlacedImage:
    """One image on the canvas, with the extent it was drawn at (canvas pixels)."""

    key: str
    artist: object
    extent: Tuple[float, float, float, float]  # (xmin, xmax, ymax, ymin)


class FibsemRealSpaceCanvas(FibsemCanvasBase):
    """Canvas holding N images, each placed at the position it was acquired.

    * :meth:`add_image` places an image from its centre position and pixel size
    * :meth:`clear_images` empties the canvas without disturbing overlays
    * the view fits the union of everything placed

    Navigation, overlays, toolbar and chrome all come from :class:`FibsemCanvasBase`.
    """

    def __init__(
        self,
        parent=None,
        reference_pixel_size: Optional[float] = None,
        display_max_px: int = _DEFAULT_DISPLAY_PX,
    ):
        super().__init__(parent)
        # Raise this for a canvas holding only a handful of images, where per-image
        # detail matters more than redraw cost. See _DEFAULT_DISPLAY_PX.
        self._display_max_px = int(display_max_px)
        self._placed: Dict[str, PlacedImage] = {}
        self._reference_pixel_size: Optional[float] = reference_pixel_size
        # Optional (width, height, cx, cy) in metres — see set_world_extent.
        self._world_extent_m: Optional[Tuple[float, float, float, float]] = None
        self._auto_key = 0
        # Refit as content grows, so tiles arriving during an acquisition stay in view.
        # Callers that would rather keep the user's zoom can turn this off.
        self.auto_fit: bool = True
        self._fitted_extent: Optional[Tuple[float, float, float, float]] = None

        if reference_pixel_size:
            self._pixel_size = reference_pixel_size  # drives the scalebar

        # Square pixels from the outset, not just once an image happens to arrive.
        # imshow sets this per-artist, so an empty canvas would otherwise be "auto":
        # the axes would stretch to the widget and draw a square grid as a rectangle --
        # a resize alone distorted it 3x. Overlays are drawn in this frame whether or
        # not anything has been acquired, so the frame has to be honest first.
        self._ax.set_aspect("equal", adjustable="box")

        self.set_background_color(_DEFAULT_BACKGROUND)
        # Contrast/gamma acts on a single frame; there is no meaning yet for "the"
        # image here, so don't offer a control that would silently do nothing.
        self.btn_contrast.hide()
        self._reposition_overlay_buttons()

    # ── properties ────────────────────────────────────────────────────────

    @property
    def reference_pixel_size(self) -> Optional[float]:
        """Metres per canvas pixel, or None until the first image sets it."""
        return self._reference_pixel_size

    @property
    def placed_keys(self) -> List[str]:
        """Keys of the images currently placed, in insertion order."""
        return list(self._placed)

    # ── public API ────────────────────────────────────────────────────────

    def add_image(
        self,
        data: np.ndarray,
        centre: Tuple[float, float],
        pixel_size: float,
        key: Optional[str] = None,
        cmap: str = "gray",
        zorder: Optional[float] = None,
        covers: Optional[Tuple[float, float]] = None,
    ) -> str:
        """Place *data* centred on *centre* (metres), at *pixel_size* metres/px.

        Re-using a *key* replaces that image in place rather than stacking a second
        artist on top — which is what a live preview wants as frames arrive. Returns
        the key, so callers that don't supply one can still address the image later.

        Images otherwise draw in the order they were added, so a later one covers an
        earlier one where they overlap. Pass *zorder* when that is the wrong answer —
        a coarse overview belongs *under* the detailed tiles acquired over it,
        regardless of which arrived first.

        The ground an image covers is normally its shape times *pixel_size*. Pass
        *covers* as ``(width, height)`` in metres when the array is a *reduced*
        representation of something larger — a decimated preview, or a composite blended
        at display resolution — so it is placed at the size it represents rather than
        the size it is stored at. *pixel_size* then describes the source, and still
        decides how this image sorts against others by detail.
        """
        data = np.asarray(data)
        _require_displayable(data)
        if not pixel_size or pixel_size <= 0:
            raise ValueError(f"pixel_size must be positive, got {pixel_size!r}")

        if self._reference_pixel_size is None:
            # First image defines the scale, so a lone image maps 1:1 to canvas pixels.
            self._reference_pixel_size = float(pixel_size)
            self._pixel_size = float(pixel_size)

        if key is None:
            key = f"image-{self._auto_key}"
            self._auto_key += 1

        existing = self._placed.pop(key, None)
        if existing is not None:
            self._remove_artist(existing)

        extent = self._extent_for(data.shape, centre, pixel_size, covers)
        shown = _downsample(data, self._display_max_px)
        kw = {} if zorder is None else {"zorder": zorder}
        artist = self._ax.imshow(
            shown,
            origin="upper",
            aspect="equal",
            interpolation="nearest",
            extent=extent,
            cmap=cmap if shown.ndim == 2 else None,
            **kw,
        )
        self._placed[key] = PlacedImage(key=key, artist=artist, extent=extent)

        self._after_content_change()
        return key

    def update_image(self, key: str, data: np.ndarray) -> bool:
        """Swap the pixels of a placed image, keeping its position, scale and z-order.

        The counterpart to :meth:`~FibsemImageCanvas.update_display` for a canvas holding
        many images. Re-adding under the same key also works, but it destroys and
        recreates the artist, which moves it to the top of the draw order — so a live
        preview refreshed that way would climb over tiles it should sit beneath.

        Returns False if *key* was not placed. The shape may change; the image keeps its
        extent, so different-shaped data is stretched over the same ground.
        """
        placed = self._placed.get(key)
        if placed is None:
            return False
        data = np.asarray(data)
        _require_displayable(data)
        placed.artist.set_data(_downsample(data, self._display_max_px))
        self.draw_idle()
        return True

    def remove_image(self, key: str) -> bool:
        """Remove one placed image. Returns False if *key* was not placed."""
        placed = self._placed.pop(key, None)
        if placed is None:
            return False
        self._remove_artist(placed)
        self._after_content_change()
        return True

    def clear_images(self) -> None:
        """Remove every placed image, leaving overlays and chrome attached."""
        if not self._placed:
            return
        for placed in self._placed.values():
            self._remove_artist(placed)
        self._placed.clear()
        self._fitted_extent = None
        self._after_content_change()

    def clear(self) -> None:
        """Clear the canvas entirely (axes are wiped, so drop the placements too)."""
        self._placed.clear()
        self._fitted_extent = None
        super().clear()

    def set_reference_pixel_size(self, pixel_size: float) -> bool:
        """Fix the canvas scale before anything is placed.

        Normally the first image supplies it. Setting it up front lets a caller draw in
        the canvas frame — a planned tile grid, stage markers — while the canvas is still
        empty, which is exactly when a *planned* overlay is most useful.

        Refuses once images are placed: their extents were computed against the current
        scale, so changing it would move everything already drawn. Returns whether it
        was applied.
        """
        if self._placed or not pixel_size or pixel_size <= 0:
            return False
        self._reference_pixel_size = float(pixel_size)
        self._pixel_size = float(pixel_size)
        self._refresh_scalebar()
        self._after_content_change()
        return True

    def set_world_extent(
        self,
        width: Optional[float],
        height: Optional[float] = None,
        centre: Tuple[float, float] = (0.0, 0.0),
        refit: bool = True,
    ) -> None:
        """Always frame at least *width* x *height* metres, centred on *centre*.

        Without this the view hugs whatever has been acquired, which has two costs: the
        framing jumps every time a tile lands, and a run that walks in one direction
        produces a long thin footprint — at which point ``aspect="equal"`` squeezes the
        axes into a sliver of the window, since keeping pixels square leaves nothing else
        to give. Declaring the working area up front keeps the framing stable and gives
        the acquired region somewhere to sit.

        A minimum, not a clip: images outside it still draw, and the view still grows to
        include them. Pass None to go back to fitting the content alone.

        `refit=False` declares the area without moving the camera. The working area is
        also what the zoom limiter measures against, so a caller may need it to keep up
        with something that moves continuously — a planned grid being dragged — where
        re-framing on every step would drag the view along with it. The framing is
        recorded as already-fitted in that case, so the *next* content change does not
        refit on account of this one either.
        """
        if width is None:
            updated = None
        else:
            height = width if height is None else height
            if width <= 0 or height <= 0:
                raise ValueError(f"world extent must be positive, got {width}x{height}")
            updated = (width, height, centre[0], centre[1])

        if updated == self._world_extent_m:
            # Idempotent: re-declaring the same working area is not a change, and
            # refitting anyway would throw away the user's zoom every time a caller
            # restated it — which one doing so on each settings change duly did.
            return

        self._world_extent_m = updated
        if refit:
            self._fitted_extent = None  # force a refit against the new framing
        else:
            # Adopt the new framing as though it had already been fitted. Needed even
            # with images on screen: `_fit_extent` falls back to the working area when
            # nothing is placed, so a bare `_after_content_change` would refit onto the
            # new area on an empty canvas.
            self._fitted_extent = self._fit_extent()
        self._after_content_change()

    @property
    def world_extent(self) -> Optional[Tuple[float, float, float, float]]:
        """The declared working area as (width, height, cx, cy) in metres, or None."""
        return self._world_extent_m

    def metres_to_canvas(self, x: float, y: float) -> Tuple[float, float]:
        """Convert a position in metres to canvas coordinates.

        For placing overlay geometry — markers, a planned grid — in the same frame as
        the images. Returns (0, 0) before a reference pixel size exists.
        """
        ref = self._reference_pixel_size
        if not ref:
            return 0.0, 0.0
        return x / ref, y / ref

    def canvas_to_metres(self, x: float, y: float) -> Tuple[float, float]:
        """Convert canvas coordinates back to metres — e.g. for a click position."""
        ref = self._reference_pixel_size
        if not ref:
            return 0.0, 0.0
        return x * ref, y * ref

    # ── the base's content hooks ──────────────────────────────────────────

    def _content_extent(self) -> Optional[Tuple[float, float, float, float]]:
        """Union of every placed image and the declared working area, or None if empty.

        The world extent is included as a *minimum*, so content outside it still frames.
        """
        extents = [p.extent for p in self._placed.values()]
        world = self._world_extent_canvas()
        if world is not None:
            extents.append(world)
        if not extents:
            return None
        return (
            min(e[0] for e in extents),  # xmin
            max(e[1] for e in extents),  # xmax
            max(e[2] for e in extents),  # ymax (bottom; y runs down)
            min(e[3] for e in extents),  # ymin (top)
        )

    def _image_extent(self) -> Optional[Tuple[float, float, float, float]]:
        """Union of the placed images alone, ignoring any declared working area."""
        extents = [p.extent for p in self._placed.values()]
        if not extents:
            return None
        return (
            min(e[0] for e in extents),
            max(e[1] for e in extents),
            max(e[2] for e in extents),
            min(e[3] for e in extents),
        )

    def _fit_extent(self) -> Optional[Tuple[float, float, float, float]]:
        """Frame the images, not the working area, padded to the widget's shape.

        The working area says how much space the canvas *represents*; fitting to it would
        zoom out past the data whenever the declared area is larger than what has been
        acquired — which is the normal case. Falls back to the working area only when
        there is nothing acquired to look at.
        """
        extent = self._image_extent() or self._content_extent()
        return self._pad_to_widget_aspect(extent)

    def _pad_to_widget_aspect(
        self, extent: Optional[Tuple[float, float, float, float]]
    ) -> Optional[Tuple[float, float, float, float]]:
        """Grow the shorter axis so the framed region has the widget's proportions.

        ``aspect="equal"`` keeps pixels square, and when the framed region is a different
        shape from the widget the only thing left to give is the axes box — which
        collapses to a sliver for a long thin mosaic (a run walking one direction framed
        at 39:1 left an axes 2% of the window wide, with the content squeezed into it).

        Padding the *view* instead means the box always fills the widget and the spare
        room shows as backdrop, which on a real-space canvas is exactly right: empty
        space is a truthful statement that nothing was acquired there.
        """
        if extent is None:
            return None
        xmin, xmax, ymax, ymin = extent
        width, height = xmax - xmin, ymax - ymin
        widget_w, widget_h = self.width(), self.height()
        if width <= 0 or height <= 0 or widget_w <= 0 or widget_h <= 0:
            return extent

        target = widget_w / widget_h
        if width / height < target:  # too tall for the window -> widen
            pad = (height * target - width) / 2.0
            return (xmin - pad, xmax + pad, ymax, ymin)
        pad = (width / target - height) / 2.0  # too wide -> heighten
        return (xmin, xmax, ymax + pad, ymin - pad)

    def _world_extent_canvas(self) -> Optional[Tuple[float, float, float, float]]:
        """The declared working area in canvas pixels, or None if not usable yet.

        Needs a reference pixel size, which the first image supplies unless one was
        given up front — so a world extent set on an empty canvas takes effect as soon
        as there is something to scale it against.
        """
        ref = self._reference_pixel_size
        if self._world_extent_m is None or not ref:
            return None
        width, height, cx, cy = self._world_extent_m
        half_w, half_h = width / 2.0 / ref, height / 2.0 / ref
        cx, cy = cx / ref, cy / ref
        return (cx - half_w, cx + half_w, cy + half_h, cy - half_h)

    def _content_rect(self) -> ContentRect:
        """The union, as the rectangle overlays clamp and anchor to."""
        extent = self._content_extent()
        if extent is None:
            return EMPTY_CONTENT
        xmin, xmax, ymax, ymin = extent
        return ContentRect(xmin, ymin, xmax - xmin, ymax - ymin)

    # ── internals ─────────────────────────────────────────────────────────

    def _extent_for(
        self,
        shape: Tuple[int, ...],
        centre: Tuple[float, float],
        pixel_size: float,
        covers: Optional[Tuple[float, float]] = None,
    ) -> Tuple[float, float, float, float]:
        """Where an image of *shape* centred on *centre* (metres) lands, in canvas px.

        *covers* overrides the ground derived from shape x pixel size, for an array that
        is a reduced representation of something larger.
        """
        ref = self._reference_pixel_size or pixel_size
        if covers is not None:
            half_w = covers[0] / ref / 2.0
            half_h = covers[1] / ref / 2.0
            cx, cy = centre[0] / ref, centre[1] / ref
            return (cx - half_w, cx + half_w, cy + half_h, cy - half_h)
        height, width = shape[0], shape[1]
        scale = pixel_size / ref  # canvas pixels per image pixel
        half_w = width * scale / 2.0
        half_h = height * scale / 2.0
        cx, cy = centre[0] / ref, centre[1] / ref
        # matplotlib extent is (left, right, bottom, top); y runs down, so bottom > top.
        return (cx - half_w, cx + half_w, cy + half_h, cy - half_h)

    def _refresh_crosshair(self):
        """Mark the origin, at a fixed size on screen.

        The base draws at the centre of the content, with arms scaled to it. Neither
        works here: the centre of a union of tiles is wherever they happen to average
        out, and scaling the arms to a declared working area makes the crosshair
        enormous — 5% of 2 mm is 100 um across.

        The origin is the position the caller built the canvas around (the reference
        stage position), so that is the landmark worth drawing. Drawn as a marker rather
        than two lines so it keeps its size on screen through zoom, the way a position
        marker should.
        """
        for a in self._crosshair_artists:
            try:
                a.remove()
            except (ValueError, NotImplementedError):
                pass
        self._crosshair_artists = []
        if not self._crosshair_visible or self._reference_pixel_size is None:
            return
        (marker,) = self._ax.plot(
            [0.0], [0.0],
            marker="+", markersize=_ORIGIN_MARKER_SIZE, markeredgewidth=1.0,
            color=_ORIGIN_MARKER_COLOUR, alpha=0.8, linestyle="None", zorder=7,
        )
        self._crosshair_artists = [marker]

    def _remove_artist(self, placed: PlacedImage) -> None:
        try:
            placed.artist.remove()
        except (ValueError, NotImplementedError, AttributeError):
            pass

    def _plot_empty(self) -> None:
        """No "No image" placeholder — the empty backdrop already says it.

        On a canvas that fills the view with one image, blank axes are ambiguous and the
        text resolves it. Here the black backdrop *is* the statement: it means "nothing
        acquired here", and it stays meaningful once images arrive, since the gaps
        between them carry the same meaning. The text would also sit in the middle of
        the planned tile grid, which is drawn before any acquisition.
        """
        self._ax.set_facecolor(self._facecolor)
        self._ax.axis("off")

    def _after_content_change(self) -> None:
        """Refit if the footprint changed, refresh chrome, and tell the overlays."""
        extent = self._fit_extent()
        if self.auto_fit and extent != self._fitted_extent:
            self._fit_view()
            self._fitted_extent = extent

        self._refresh_scalebar()
        self._refresh_crosshair()
        self._notify_overlays(self._content_rect())
        self.draw_idle()
