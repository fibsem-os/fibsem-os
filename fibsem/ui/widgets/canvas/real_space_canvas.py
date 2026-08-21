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
from dataclasses import dataclass, field
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QTimer

from fibsem.imaging.reduce import downsample
from fibsem.ui.stylesheets import GRAY_CANVAS_COLOR
from fibsem.ui.widgets.canvas.canvas_base import (
    EMPTY_CONTENT,
    ContentRect,
    FibsemCanvasBase,
)

_logger = logging.getLogger(__name__)

_DEFAULT_BACKGROUND = GRAY_CANVAS_COLOR  # empty space reads as "nothing acquired here"

# Pixels handed to matplotlib per placed image — a *drawing* cost, and only that. Every
# artist is redrawn in full on each pan/zoom, so a redraw costs the total drawn pixels
# across every placed image. Measured per redraw on an 8192 px source:
#
#     images on canvas |   512   |  1024   |  2048
#     one              | 13.9 ms | 30.8 ms |  100 ms
#     five             | 42.6 ms |  126 ms |  451 ms
#
# Raised 512 -> 2048 for FIB-658: a mosaic is tens of thousands of pixels across, and 512
# showed ~0.02% of them -- too coarse to read the sample you just spent a run acquiring.
#
# Distinct from how much a *caller* keeps, and that distinction decides what this figure
# costs. A caller holding more than this can hand over a different part of it, or a
# different resolution of it, as the view moves (`DetailSource`); for such a caller this
# is a ceiling rarely approached, since what it is asked for is bounded by the pixels the
# image occupies on screen. The FIB/SEM overview works that way, so the table above is not
# what it pays. The FM canvas does not yet -- it reduces a whole composite to this figure
# and hands it over -- so the five-image column is FM's, and is what FIB-414 removes by
# giving it a source of its own. Drop back to 1024 if it bites before then.
#
# The old figure here ("100 tiles at 1024 px take ~2.7 s") measured one artist *per tile*;
# overviews are placed as a single image per run now, so it no longer describes this
# canvas -- but note the saving from that change is why 512 felt fine, not headroom.
#
# Reducing here is the floor under callers with no source: the reduction box-averages, so
# it costs resolution and nothing else -- a feature smaller than a stored pixel is dimmed
# into its neighbours rather than dropped. What it cannot do is give the detail back on
# zoom, since matplotlib resamples the whole array however little of it is on screen (1%
# of a 5120 px artist costs 321 ms against 335 ms for all of it). Only a source can.
_DEFAULT_DISPLAY_PX = 2048

# How far past the viewport a detail patch reaches, as a fraction of the viewport on each
# side. A pan inside the margin is already drawn, so it costs nothing; one past it waits
# for the next refresh. Small on purpose -- the margin is paid for in resolution, since a
# patch covering 1.5x the viewport spends its budget over 1.5x the ground, and refetching
# is cheap enough (single-digit ms) that buying fewer fetches with blur is a bad trade.
_DETAIL_MARGIN = 0.25

# Coalescing interval for detail refreshes, in ms. Longer than the 32 ms redraw on
# purpose: a pan emits a limit change per mouse move, and re-deriving on each would do
# the work tens of times over for one gesture. What the delay costs is that the instant
# after a large jump is drawn from the previous patch, stretched, until this fires.
_DETAIL_INTERVAL = 120

# How far below the screen's own resolution a patch may fall before it is refetched. Not
# 1.0, because an exact comparison refetches on any zoom whatsoever -- including the
# sub-pixel drift a resize produces -- and a tenth of a screen pixel is not visible.
_DETAIL_TOLERANCE = 0.9


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


class ImageRegion(NamedTuple):
    """A sub-rectangle of one placed image, as fractions of its own footprint.

    ``(0, 1, 0, 1)`` is the whole image, and ``top < bottom`` because this canvas's y
    runs down.

    Fractions rather than metres or array indices, so a source can be asked for part of
    itself without knowing what the canvas draws in, what scale it is at, or how large
    the source's own array is. That is what lets one protocol serve an array held in
    memory and a mosaic still on disk.
    """

    left: float
    right: float
    top: float
    bottom: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    def contains(self, other: "ImageRegion") -> bool:
        """Whether *other* falls wholly inside this one."""
        return (
            self.left <= other.left
            and self.right >= other.right
            and self.top <= other.top
            and self.bottom >= other.bottom
        )


WHOLE_IMAGE = ImageRegion(0.0, 1.0, 0.0, 1.0)

# Asked for part of an image and a pixel budget; answers with the pixels and the part it
# actually gave. The two regions differ because a source snaps outward to whole pixels of
# whatever it holds — returning the region rather than assuming it is what was asked for
# is what keeps a patch drawn exactly over the ground it came from, instead of a fraction
# of a pixel off. Returning None declines, leaving whatever is already drawn alone.
DetailSource = Callable[[ImageRegion, int], Optional[Tuple[np.ndarray, ImageRegion]]]


@dataclass
class PlacedImage:
    """One image on the canvas, with the extent it was drawn at (canvas pixels)."""

    key: str
    artist: object
    # The image's *whole* footprint, which is what the canvas fits and frames against —
    # never the part of it currently in the artist. Keeping the two separate is what
    # lets a detail patch change under the view without the view chasing it.
    extent: Tuple[float, float, float, float]  # (xmin, xmax, ymax, ymin)
    detail: Optional[DetailSource] = None
    drawn: ImageRegion = field(default=WHOLE_IMAGE)
    # The pixel budget the patch in the artist was fetched at -- what was *asked* for,
    # not what came back. A source routinely returns less: `downsample` reduces by whole
    # factors, so a 616 px slice asked for at 256 comes back at 206. Comparing the next
    # request against what arrived rather than against what was requested makes every
    # patch look deficient, and re-asking returns the identical array, forever.
    asked_px: int = 0
    # The source's own content changed under the patch, so what is drawn is out of date
    # however well it fits the view. Set for *every* image by a forced refresh, and
    # cleared only when that image is actually redrawn -- an image off screen when the
    # change happened keeps the flag until it comes back, which is the whole point of it
    # (see `refresh_detail`).
    stale: bool = False


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
        if self._display_max_px <= 0:
            raise ValueError(f"display_max_px must be positive, got {display_max_px!r}")
        self._placed: Dict[str, PlacedImage] = {}
        self._reference_pixel_size: Optional[float] = reference_pixel_size
        # Optional (width, height, cx, cy) in metres — see set_world_extent.
        self._world_extent_m: Optional[Tuple[float, float, float, float]] = None
        self._auto_key = 0
        # Refit as content grows, so tiles arriving during an acquisition stay in view.
        #
        # True until the framing becomes someone's choice rather than this canvas's
        # guess. A pan or a zoom is the user making it theirs (`_view_moved_by_user`);
        # an owner driving the camera itself sets this False directly. Either way the
        # only thing that takes it back is "reset view" -- the button whose whole
        # meaning is "frame it for me again".
        self.auto_fit: bool = True
        self._fitted_extent: Optional[Tuple[float, float, float, float]] = None

        # Detail refresh: coalesced, because the thing that provokes it is a gesture.
        self._detail_timer = QTimer(self)
        self._detail_timer.setSingleShot(True)
        self._detail_timer.setInterval(_DETAIL_INTERVAL)
        self._detail_timer.timeout.connect(self.refresh_detail)
        # Off the limits themselves rather than off the pan and zoom handlers: a refit, a
        # resize, and an owner driving the camera all change what is on screen without
        # going near either handler, and every one of them wants a fresh patch.
        self._ax.callbacks.connect("xlim_changed", self._on_limits_changed)
        self._ax.callbacks.connect("ylim_changed", self._on_limits_changed)

        if reference_pixel_size:
            self._pixel_size = reference_pixel_size  # drives the scalebar

        self._apply_axes_convention()

        self.set_background_color(_DEFAULT_BACKGROUND)
        # Contrast/gamma acts on a single frame; there is no meaning yet for "the"
        # image here, so don't offer a control that would silently do nothing.
        self.btn_contrast.hide()
        # Same argument: nothing on this canvas has a centre of its own to cross-hair,
        # and what is worth marking lives in stage space, which its callers draw.
        self.btn_toggle_crosshair.hide()
        self._reposition_overlay_buttons()

    # ── properties ────────────────────────────────────────────────────────

    @property
    def reference_pixel_size(self) -> Optional[float]:
        """Metres per canvas pixel, or None until the first image sets it."""
        return self._reference_pixel_size

    @property
    def display_max_px(self) -> int:
        """The most pixels this canvas will draw of any one image. See the constant.

        Public because callers legitimately need it: one that reduces before handing an
        image over — to blend at display resolution, or to keep what it placed — wants
        to reduce to the same figure rather than guess at it. Read-only, because the
        extents of everything already placed were computed against the current one.
        """
        return self._display_max_px

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
        detail: Optional[DetailSource] = None,
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

        *detail* offers a caller that holds more than *data* shows. Given one, the canvas
        asks it for the part of the image on screen at the resolution the screen can use,
        and asks again as the view moves; *data* is then the whole-image fallback that is
        drawn until the first patch arrives and whenever the source declines. Without
        one, *data* is all there is and is drawn as-is, which is every caller today.
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
        shown = downsample(data, self._display_max_px)
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
        self._placed[key] = PlacedImage(
            key=key,
            artist=artist,
            extent=extent,
            detail=detail,
            drawn=WHOLE_IMAGE,
            asked_px=int(max(shown.shape[:2])),
        )

        # Framing first: what the screen can show of an image depends on where the camera
        # is, and adding this one may well have moved it.
        self._after_content_change()
        if detail is not None:
            self.refresh_detail()
        return key

    def update_image(self, key: str, data: np.ndarray) -> bool:
        """Swap the pixels of a placed image, keeping its position, scale and z-order.

        The counterpart to :meth:`~FibsemImageCanvas.update_display` for a canvas holding
        many images. Re-adding under the same key also works, but it destroys and
        recreates the artist, which moves it to the top of the draw order — so a live
        preview refreshed that way would climb over tiles it should sit beneath.

        Returns False if *key* was not placed. The shape may change; the image keeps its
        extent, so different-shaped data is stretched over the same ground.

        *data* is the whole image, so this puts the artist back to the whole footprint
        even if a detail patch had it cropped to part of one — otherwise the new pixels
        would be stretched over the old patch's ground. A fresh patch is then asked for,
        since the source has presumably changed underneath too.
        """
        placed = self._placed.get(key)
        if placed is None:
            return False
        data = np.asarray(data)
        _require_displayable(data)
        shown = downsample(data, self._display_max_px)
        placed.artist.set_data(shown)
        if placed.drawn != WHOLE_IMAGE:
            placed.artist.set_extent(placed.extent)
        placed.drawn, placed.asked_px = WHOLE_IMAGE, int(max(shown.shape[:2]))
        # The whole of the current image is in the artist, so nothing is out of date --
        # even if this ran while the image was off screen and no patch follows.
        placed.stale = False
        if placed.detail is not None:
            # Scheduled, not immediate: this is the per-frame call during a live
            # acquisition, and fetching synchronously on each frame is the cost the
            # detail pass exists to avoid.
            self._detail_timer.start()
        self.draw_idle()
        return True

    def set_image_visible(self, key: str, visible: bool) -> bool:
        """Show or hide one placed image, keeping it placed. False if *key* is unknown.

        Distinct from :meth:`remove_image`: the artist, its extent and its draw order
        all survive, so showing it again costs nothing and does not disturb the fit.

        Matplotlib skips a hidden artist when it draws, so this is not only a visual
        change -- a canvas redraws every stored pixel of everything visible, and hiding
        an image takes it out of that cost as surely as removing it would (FIB-414).
        """
        placed = self._placed.get(key)
        if placed is None:
            return False
        placed.artist.set_visible(bool(visible))
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
        # `cla()` takes the aspect and the y direction with it.
        self._apply_axes_convention()

    def _apply_axes_convention(self) -> None:
        """Square pixels and y running *down*, whether or not anything is placed.

        Both are set by `imshow` per artist, so a canvas with no images inherits
        matplotlib's defaults instead: axes that stretch to the widget, and a y axis
        that increases upwards. Neither is survivable here, because overlays are drawn
        in this frame whether or not anything has been acquired.

        The aspect was the visible one first -- a resize alone drew a square lattice as
        a rectangle, 3x out. The y direction is worse and stayed hidden longer: with
        nothing placed the scene is drawn *vertically mirrored*, and flips the right way
        up the moment the first tile lands. Nothing showed it while an empty canvas drew
        nothing; it appeared as soon as the overlays did (FIB-616), first as labels
        sitting under their shapes instead of over them.

        Applied by inverting rather than by setting limits: matplotlib's autoscale
        preserves the axis direction, so this survives every later refit.

        `adjustable="box"` is kept deliberately, though it is what leaves black bands
        when the framing does not match the widget's shape: it gives away the *axes*
        rather than the view. `"datalim"` fills the widget instead, and costs more than
        it saves -- matplotlib then adjusts limits a caller has set, so a declared
        working area drifts and a view asked to hold still does not (three fluorescence
        tests measured it moving 5000 -> 5831). Callers keep the limits they set here,
        and it is the caller's job to frame something the widget's shape can hold:
        `_pad_to_widget_aspect` does exactly that.
        """
        self._ax.set_aspect("equal", adjustable="box")
        if not self._ax.yaxis_inverted():
            self._ax.invert_yaxis()
        # Autoscale off, because `AxesImage.set_extent` moves the camera while it is on:
        # it ends in `set_xlim`/`set_ylim` onto the artist's own extent. That matters
        # here and nowhere else, because the detail pass re-extents an artist on every
        # view change -- so with autoscale on, looking at one image would frame it, and
        # frame it again, and never settle.
        #
        # A no-op in every path that already existed: `set_xlim` clears the flag itself,
        # so the first `_fit_view` turned autoscale off anyway and nothing has been
        # relying on it since. This closes the one window where it is still on -- a fresh
        # or just-cleared axes, which is also where `cla()` puts it back, hence setting it
        # here rather than once in the constructor.
        self._ax.set_autoscale_on(False)

    def set_reference_pixel_size(self, pixel_size: float) -> bool:
        """Fix the canvas scale before anything is placed.

        Normally the first image supplies it. Setting it up front lets a caller draw in
        the canvas frame — a planned tile grid, stage markers — while the canvas is still
        empty, which is exactly when a *planned* overlay is most useful.

        Refuses once images are placed: their extents were computed against the current
        scale, so changing it would move everything already drawn. Returns whether it
        changed anything.

        Asking for the scale it already has changes nothing, and says so rather than
        repainting. It used to repaint: the fluorescence tab sets this on every tile
        grid refresh, which is every motion event of a drag, and until the first image
        landed that was a full canvas repaint per event -- which is also what threw away
        the blitted background the grid was being drawn over (FIB-752). It looked like
        the tab getting *faster* once an image arrived, because that is when this call
        starts refusing.
        """
        if self._placed or not pixel_size or pixel_size <= 0:
            return False
        if self._reference_pixel_size == float(pixel_size):
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

    # ── who owns the camera ───────────────────────────────────────────────

    def _view_moved_by_user(self) -> None:
        """A pan or a zoom hands the framing over. See :attr:`auto_fit`.

        Zoom in on a feature and the next thing placed used to throw the view away and
        refit to everything -- at the two moments you are most likely to be looking
        closely, since a run ends by removing the preview and placing the stitch under
        another key, which is two extent changes back to back (FIB-648).
        """
        self.auto_fit = False

    def reset_view(self) -> None:
        """Frame the content, and take the camera back.

        The one way `auto_fit` comes back on, deliberately: it is the button that means
        "frame it for me", so re-arming is what it was already being pressed for, and
        nothing else can quietly undo a framing someone chose.
        """
        self.auto_fit = True
        super().reset_view()
        # Adopt what was just fitted, so the next content change is measured against
        # this framing rather than against whatever was fitted before the user's zoom.
        self._fitted_extent = self._fit_extent()

    # ── detail: drawing what the view can use, not what the caller holds ──

    def _on_limits_changed(self, _axes=None) -> None:
        """The view moved; whatever is drawn may no longer be the right part of it."""
        if self._placed and any(p.detail is not None for p in self._placed.values()):
            self._detail_timer.start()

    def refresh_detail(self, force: bool = False) -> None:
        """Ask every source for the part of itself now worth drawing.

        Called by the canvas itself whenever the view moves. Public for *force*, which
        is the other way a patch goes stale: the source's own content changed under it —
        a contrast curve, a re-blend, a new frame — which the view knows nothing about,
        so none of the usual tests would notice.

        *force* marks every source's patch stale, not only the ones it can redraw now.
        An image off screen cannot be redrawn — there is no region to fetch — and if the
        flag were not left set, the next pan back to it would find a patch that covers
        the viewport at the right resolution and keep it, styled the way it was before
        the change. That is a restyle silently not applying to part of the canvas, which
        is exactly the failure the caller used *force* to avoid.

        Deliberately *not* a content change: an image's footprint is unchanged by which
        part of it is in the artist, so nothing here refits, and `PlacedImage.extent`
        stays the whole image throughout. That is what keeps this from chasing its own
        tail -- a refit would move the limits, which would schedule another refresh.
        """
        changed = False
        if force:
            for placed in self._placed.values():
                if placed.detail is not None:
                    placed.stale = True
        for placed in list(self._placed.values()):
            if placed.detail is None:
                continue
            if not placed.artist.get_visible():
                # Hiding an image is how a caller takes it out of the redraw cost
                # (`set_image_visible`); fetching patches for it would put it back.
                continue
            # Two regions, and the difference between them is the whole of the margin.
            # What has to be *covered* is the viewport itself; what gets *fetched* reaches
            # past it. Testing coverage against the reaching one would compare a margin
            # with a margin, so every pan of any size would fail it and the margin would
            # buy exactly nothing.
            wanted = self._visible_region(placed)
            if wanted is None:
                continue  # off screen: leave the last patch, it costs nothing to keep
            fetch = self._visible_region(placed, margin=_DETAIL_MARGIN) or wanted
            budget = self._detail_budget(placed, fetch)
            if not placed.stale and not self._needs_detail(
                placed, wanted, fetch, budget
            ):
                continue
            try:
                answer = placed.detail(fetch, budget)
            except Exception as e:
                # A source that raises must not take the canvas with it: this runs off a
                # timer, and PyQt5 aborts the process on an exception out of a slot
                # (FIB-329). What is already drawn stays drawn.
                _logger.debug(f"Detail source for {placed.key!r} failed: {e}")
                continue
            if answer is None:
                continue
            data, got = answer
            if self._draw_patch(placed, data, got, budget):
                changed = True
        if changed:
            self.draw_idle()

    def _visible_region(
        self, placed: PlacedImage, margin: float = 0.0
    ) -> Optional[ImageRegion]:
        """Which part of *placed* the viewport covers, reaching *margin* past it, or None.

        None means none of it: no patch is worth fetching for an image that is not on
        screen, and the one it already has is the right thing to keep in case it comes
        back.
        """
        xmin, xmax, ymax, ymin = placed.extent
        width, height = xmax - xmin, ymax - ymin
        if width <= 0 or height <= 0:
            return None
        (vx0, vx1), (vy0, vy1) = self._ax.get_xlim(), self._ax.get_ylim()
        # y runs down, so the axes' limits arrive bottom-first.
        vy0, vy1 = min(vy0, vy1), max(vy0, vy1)
        vx0, vx1 = min(vx0, vx1), max(vx0, vx1)
        mx, my = margin * (vx1 - vx0), margin * (vy1 - vy0)
        left = max(xmin, vx0 - mx)
        right = min(xmax, vx1 + mx)
        top = max(ymin, vy0 - my)
        bottom = min(ymax, vy1 + my)
        if right <= left or bottom <= top:
            return None
        return ImageRegion(
            (left - xmin) / width,
            (right - xmin) / width,
            (top - ymin) / height,
            (bottom - ymin) / height,
        )

    def _detail_budget(self, placed: PlacedImage, region: ImageRegion) -> int:
        """The most pixels worth fetching for *region*: what the screen can show of it.

        Bounded by the display cap, so this can only ever ask for less than drawing the
        whole image would. That is the property that makes cost track the *window* rather
        than the data -- an image occupying a corner of the canvas is budgeted for a
        corner's worth, however large the thing behind it is.
        """
        xmin, xmax, _, _ = placed.extent
        span = self._ax.get_xlim()
        canvas_span = abs(span[1] - span[0])
        if canvas_span <= 0:
            return self._display_max_px
        # Device pixels the region occupies across the axes, at the current scale.
        across = (
            (xmax - xmin) * region.width / canvas_span * max(1, self._axes_width_px())
        )
        return int(max(1, min(self._display_max_px, np.ceil(across))))

    def _axes_width_px(self) -> int:
        """The axes' width in device pixels — what "as much as the screen can use" is."""
        try:
            return int(self._ax.get_window_extent().width)
        except Exception:  # pragma: no cover - before the first draw
            return self.width()

    def _needs_detail(
        self,
        placed: PlacedImage,
        wanted: ImageRegion,
        fetch: ImageRegion,
        budget: int,
    ) -> bool:
        """Whether fetching *fetch* would beat the patch already in the artist.

        Two ways it would. The view moved somewhere the patch does not cover — *wanted*,
        the bare viewport, so the margin is what absorbs an ordinary pan. Or the patch is
        coarser than what a fetch would now deliver, which is what zooming in produces.

        Asked as "would this be sharper", not as "is this as sharp as the screen wants",
        and the difference matters at the display cap. Past a certain zoom the budget
        clamps, so a patch reaching past the viewport can never resolve the visible part
        at the screen's full resolution — the margin's ground is spending some of the
        cap. Against the absolute standard that is permanently unsatisfied, and every
        view change refetches an identical array forever. Against what a fetch would
        actually give, a clamped budget compares equal and nothing happens.

        Nothing asks whether the patch is *finer* than needed: zooming out grows *wanted*
        past what is drawn, so that case is already the first one.
        """
        if not placed.drawn.contains(wanted):
            return True
        if placed.drawn.width <= 0 or fetch.width <= 0:
            return True
        # Pixels per unit of image, which is comparable across differently-sized regions.
        have = placed.asked_px / placed.drawn.width
        would = budget / fetch.width
        return would > have / _DETAIL_TOLERANCE

    def _draw_patch(
        self, placed: PlacedImage, data: np.ndarray, region: ImageRegion, budget: int
    ) -> bool:
        """Put *data* in the artist over the ground *region* names. False if unusable.

        The extent comes from the region the source says it *gave*, not the one it was
        asked for: a source snaps outward to whole pixels of what it holds, and drawing
        the requested rectangle instead would place the patch a fraction of a pixel off
        its own ground — which on a canvas whose whole point is that images line up
        would show as a seam every time the view moved.
        """
        data = np.asarray(data)
        try:
            _require_displayable(data)
        except ValueError as e:
            _logger.debug(f"Detail source for {placed.key!r} returned {e}")
            return False
        if not (0.0 <= region.left < region.right <= 1.0):
            _logger.debug(f"Detail source for {placed.key!r} gave x outside itself")
            return False
        if not (0.0 <= region.top < region.bottom <= 1.0):
            _logger.debug(f"Detail source for {placed.key!r} gave y outside itself")
            return False

        xmin, xmax, ymax, ymin = placed.extent
        width, height = xmax - xmin, ymax - ymin
        placed.artist.set_data(downsample(data, self._display_max_px))
        placed.artist.set_extent(
            (
                xmin + region.left * width,
                xmin + region.right * width,
                ymin
                + region.bottom * height,  # matplotlib takes (left, right, bottom, top)
                ymin + region.top * height,
            )
        )
        placed.drawn = region
        placed.asked_px = budget
        placed.stale = False
        return True

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
        """Nothing. A real-space canvas has no centre worth marking of its own.

        The base draws at the centre of the content, which here is wherever a union of
        tiles happens to average out. This canvas drew at canvas zero instead -- the
        *origin* -- on the argument that it is the position the caller built the canvas
        around. It is, and that turns out to mean nothing to a user: the origin is the
        stage position of the first image that happened to be placed, so the marker sat
        somewhere arbitrary, moved when the view changed, and was drawn in the one
        colour on the canvas that reads as a warning.

        What is worth marking is in *stage* space -- grid centre, the holder's slots,
        where the stage is now -- and the canvas does not know stage space. Its callers
        do, and they already draw all three through overlays. So this draws nothing and
        the toolbar button that toggled it is hidden, on the same argument as the
        contrast control: do not offer one that would silently do nothing.
        """
        for a in self._crosshair_artists:
            try:
                a.remove()
            except (ValueError, NotImplementedError):
                pass
        self._crosshair_artists = []

    def resizeEvent(self, event) -> None:
        """Re-frame on a resize, because the framing is widget-shaped.

        `_pad_to_widget_aspect` grows the framed region to the widget's proportions, so
        that `aspect="equal"` has nothing left to take out of the axes box. A region
        padded to the shape the widget *was* stops matching the moment it changes, and
        the difference comes straight back off the axes as bands down the sides or
        across the top -- which is what an overview framed before its splitter settled
        looked like.

        A canvas whose camera belongs to someone else is not re-framed behind their
        back, but it is not left letterboxed either: the viewport changed size, so the
        view is grown or shrunk to match and the magnification is what survives.
        """
        super().resizeEvent(event)
        if self.auto_fit:
            self._fitted_extent = None  # the padded extent is stale, whatever it was
            self._fit_view()
        else:
            self._keep_scale_through_resize(event.oldSize(), event.size())
        # Kicked directly as well as through the limits: how many pixels the screen can
        # use is the widget's size, and a resize can leave the limits untouched (the
        # first one, which has no "before" to keep the scale against) while changing it.
        self._on_limits_changed()
        self.draw_idle()

    def _keep_scale_through_resize(self, old, new) -> None:
        """Resize the *viewport*, not the magnification.

        A window that gets taller shows more; it does not zoom. Scaling the limits by
        the same factors as the widget keeps metres-per-pixel exactly where the user
        left it, and keeps the framed region the widget's shape -- so `aspect="equal"`
        still has nothing to take out of the axes box and the canvas goes on filling
        the window. Refitting would do the second thing at the cost of the first, which
        is the framing being taken back; doing nothing does the first at the cost of
        the second, which is black bands down the sides.
        """
        if old is None or old.width() <= 0 or old.height() <= 0:
            return  # the first resize has no "before" to keep the scale against
        kx, ky = new.width() / old.width(), new.height() / old.height()
        if kx <= 0 or ky <= 0:
            return
        (x0, x1), (y0, y1) = self._ax.get_xlim(), self._ax.get_ylim()
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        # Halved spans, signed: the y axis is inverted here, and the sign is what keeps
        # it that way.
        half_w, half_h = (x1 - x0) / 2.0 * kx, (y1 - y0) / 2.0 * ky
        self._ax.set_xlim(cx - half_w, cx + half_w)
        self._ax.set_ylim(cy - half_h, cy + half_h)

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
