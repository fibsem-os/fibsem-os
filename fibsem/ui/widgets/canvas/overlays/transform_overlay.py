"""A body on the canvas you can drag and turn: the gesture, without the body.

Two things on the Overview tab want placing by hand against the picture underneath:
the grid's bars (FIB-608) and, later, a fluorescence overview (FIB-1030). The body is
different, the gesture is the same -- press inside it and drag to move, take the handle
and drag to rotate -- so the gesture lives here once and the body is a subclass.

The body is described in its *own* frame, and drawn through a linear map onto the
canvas: a rotation, and then a squash along canvas y. The squash is the view's, not
the user's. A canvas draws the plane a beam sees, and a beam looking at the sample from
off its normal sees a length along the surface shortened by the cosine of that angle --
1.0 straight down, 0.616 in the crossed views, 0.259 for the ion beam at the milling
pose. The host reads that factor off its frame and hands it in; what the user places is
always the body on the sample, and it lands foreshortened however the view is tilted
(FIB-615). Keeping the squash out of the user's hands is the point: dragged free, it
would be rediscovered by eye every time and slightly wrong every time.

Emits rather than mutating (the convention every interactive overlay here follows): the
host owns where the body is, in whatever units it keeps it in, and re-places on every
emission. Input only while the canvas has made this the active overlay -- outside that
mode a body drawn here is inert, so the click-to-move and pan the canvas already does
are untouched.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, List, Optional, Tuple

from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal

from fibsem.ui.widgets.canvas.overlays.base import CanvasOverlay

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from fibsem.ui.widgets.canvas.canvas_base import ContentRect, FibsemCanvasBase

logger = logging.getLogger(__name__)

# Screen pixels from the body's centre to the rotation handle, and the handle's grab
# radius. Screen rather than data units so the handle is the same size at every zoom.
HANDLE_DISTANCE_PX = 60.0
HANDLE_RADIUS_PX = 9.0
# A press has to travel this far on screen before it is a move rather than a click.
MOVE_DRAG_THRESHOLD_PX = 4.0

_HANDLE_COLOUR = "#4dd0e1"


class TransformGestureOverlay(QObject, CanvasOverlay):
    """Drag to move, handle to rotate, view squash applied on the way to the canvas.

    Subclasses draw the body (:meth:`_draw_body`) in their own frame using
    :meth:`to_canvas`, and say what counts as inside it (:meth:`_body_contains`).
    Everything about the gesture, the handle and the blitting is here.
    """

    moved = pyqtSignal(float, float)  # the body's new centre, canvas coordinates
    rotated = pyqtSignal(float)  # the body's new rotation, degrees, clockwise on screen
    # A move or rotate gesture has ended. For work a host wants to do once, at the end,
    # rather than on every motion event -- anything that repaints the whole canvas
    # belongs here, because during the drag the body is blitted over a held background
    # and a repaint throws that away (FIB-752).
    drag_finished = pyqtSignal()

    def __init__(self, parent: Optional[QObject] = None) -> None:
        QObject.__init__(self, parent)
        self._ax = None
        self._canvas: "FibsemCanvasBase | None" = None
        self._artists: list = []
        self._rect: Optional["ContentRect"] = None
        self._cids: List[int] = []
        self._visible: bool = False

        self._centre: Optional[Tuple[float, float]] = None
        self._rotation: float = 0.0  # degrees, clockwise on screen
        self._squash: float = 1.0  # canvas y per body y, the view's foreshortening
        # Whether the body is seen from behind. A view from the other side of the
        # grid mirrors everything on it, and a mirror is not a rotation: the known
        # geometry supplies it, the user never drags it.
        self._mirror: bool = False

        # A press inside the body, held until it travels far enough to be a move:
        # (press_px_x, press_px_y, press_x, press_y, centre_x, centre_y).
        self._move_start: Optional[Tuple[float, float, float, float, float, float]] = (
            None
        )
        self._move_active: bool = False
        # A press on the handle: (angle at press, rotation at press).
        self._rotate_start: Optional[Tuple[float, float]] = None
        self._rotate_active: bool = False
        self._cursor_set: bool = False
        self._handles_drawn: bool = False
        # The canvas without the body on it, captured once when a drag starts; see
        # `_blit_frame`.
        self._blit_bg = None

    # ── overlay protocol ──────────────────────────────────────────────────

    def attach(self, ax, canvas: "FibsemCanvasBase") -> None:
        self._ax = ax
        self._canvas = canvas
        self._cids = [
            canvas.mpl_connect("button_press_event", self._on_press),
            canvas.mpl_connect("motion_notify_event", self._on_motion),
            canvas.mpl_connect("button_release_event", self._on_release),
            canvas.mpl_connect("draw_event", self._on_canvas_drawn),
        ]

    def detach(self) -> None:
        self._move_start = None
        self._move_active = False
        self._rotate_start = None
        self._rotate_active = False
        self._blit_bg = None
        if self._canvas is not None:
            for cid in self._cids:
                self._canvas.mpl_disconnect(cid)
            self._cids = []
            if self._cursor_set:
                self._canvas.unsetCursor()
                self._cursor_set = False
        self._remove_artists()
        self._ax = None
        self._canvas = None

    def on_content_changed(self, rect: "ContentRect") -> None:
        self._rect = rect
        # Captured against the old bounds, so it no longer describes what is behind
        # the body: the next drag frame recaptures.
        self._blit_bg = None
        self._remove_artists()
        if self._visible and not rect.is_empty:
            self._draw()

    # ── placement ─────────────────────────────────────────────────────────

    def set_placement(
        self,
        centre: Tuple[float, float],
        rotation: float = 0.0,
        squash: float = 1.0,
        mirror: bool = False,
    ) -> None:
        """Where the body sits, in canvas coordinates, and how the view squashes it.

        `rotation` is degrees, clockwise on screen. `squash` is canvas y per body y:
        the view's surface foreshortening, 1.0 for a beam looking straight down.
        `mirror` flips the body's own x before it is turned: the view sees it from
        the other side of the sample.
        """
        self._centre = (float(centre[0]), float(centre[1]))
        self._rotation = float(rotation)
        self._squash = float(squash) if squash else 1.0
        self._mirror = bool(mirror)
        self._redraw()

    def set_visible(self, visible: bool) -> None:
        """Show or hide the body without discarding its placement."""
        visible = bool(visible)
        if visible == self._visible:
            return
        self._visible = visible
        self._redraw()

    @property
    def is_visible(self) -> bool:
        return self._visible

    @property
    def centre(self) -> Optional[Tuple[float, float]]:
        return self._centre

    @property
    def rotation(self) -> float:
        return self._rotation

    @property
    def squash(self) -> float:
        return self._squash

    @property
    def mirror(self) -> bool:
        return self._mirror

    @property
    def is_dragging(self) -> bool:
        """True while a move or a rotate is in flight."""
        return self._move_active or self._rotate_active

    def is_editable(self) -> bool:
        """Whether this overlay owns input: the canvas has made it the active one.

        Modal, deliberately -- a body that could be dragged whenever nothing else was
        active would take every press on the canvas away from click-to-move.
        """
        return (
            self._canvas is not None
            and getattr(self._canvas, "_active_overlay", None) is self
        )

    # ── the body's frame ──────────────────────────────────────────────────

    def to_canvas(self, u: float, v: float) -> Tuple[float, float]:
        """A point in the body's frame, on the canvas: mirror, rotate, squash, offset."""
        cx, cy = self._centre if self._centre is not None else (0.0, 0.0)
        if self._mirror:
            u = -u
        c, s = self._cos_sin()
        x = c * u - s * v
        y = s * u + c * v
        return cx + x, cy + y * self._squash

    def from_canvas(self, x: float, y: float) -> Tuple[float, float]:
        """A canvas point in the body's frame: the exact inverse of :meth:`to_canvas`."""
        cx, cy = self._centre if self._centre is not None else (0.0, 0.0)
        dx, dy = x - cx, (y - cy) / self._squash
        c, s = self._cos_sin()
        u, v = c * dx + s * dy, -s * dx + c * dy
        return (-u if self._mirror else u), v

    def body_transform(self):
        """The body's frame onto the canvas, as a matplotlib transform.

        The same map as :meth:`to_canvas`, for artists that carry a transform rather
        than points -- an image drawn on the unit square, say. Composed in the order
        the points go through: mirror, rotate, squash, offset.
        """
        from matplotlib.transforms import Affine2D

        cx, cy = self._centre if self._centre is not None else (0.0, 0.0)
        return (
            Affine2D()
            .scale(-1.0 if self._mirror else 1.0, 1.0)
            .rotate_deg(self._rotation)
            .scale(1.0, self._squash)
            .translate(cx, cy)
        )

    def _cos_sin(self) -> Tuple[float, float]:
        angle = math.radians(self._rotation)
        return math.cos(angle), math.sin(angle)

    def stretch_of(self, u: float, v: float) -> float:
        """How long a unit vector in the body's frame comes out on the canvas.

        A bar's *width* is measured across it, so a bar drawn as a wide line needs its
        width scaled by the stretch of its normal -- one lying along the squashed
        direction is drawn thinner, as the view really shows it.
        """
        if self._mirror:
            u = -u
        c, s = self._cos_sin()
        x = c * u - s * v
        y = (s * u + c * v) * self._squash
        return math.hypot(x, y)

    # ── subclass hooks ────────────────────────────────────────────────────

    def _draw_body(self, animated: bool) -> list:
        """Create the body's artists on `self._ax` and return them."""
        raise NotImplementedError

    def _body_contains(self, x: float, y: float) -> bool:
        """Whether a canvas point is inside the body. The whole canvas by default."""
        return True

    # ── drawing ───────────────────────────────────────────────────────────

    def _redraw(self) -> None:
        self._remove_artists()
        if self._visible:
            self._draw()
        if self._canvas is not None:
            self._canvas.draw_idle()

    def _draw(self) -> None:
        if self._ax is None or self._centre is None or self._rect is None:
            return
        blitting = self._can_blit()
        try:
            self._artists.extend(self._draw_body(blitting))
        except Exception as e:  # noqa: BLE001 - a body that cannot draw is not fatal
            logger.debug(f"Could not draw the body: {e}")
        editable = self.is_editable()
        if editable:
            self._artists.extend(self._draw_handles(blitting))
        self._handles_drawn = editable
        if blitting and self._canvas is not None:
            self._blit_frame()

    def _remove_artists(self) -> None:
        for artist in self._artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._artists.clear()
        self._handles_drawn = False

    def _pixels_per_unit(self) -> float:
        """Screen pixels per canvas unit, or 0 if the axes cannot be measured yet."""
        try:
            bbox = self._ax.get_window_extent()
            xmin, xmax = self._ax.get_xlim()
            if xmax == xmin or bbox.width <= 0:
                return 0.0
            return bbox.width / abs(xmax - xmin)
        except Exception as e:
            logger.debug(f"Could not measure the axes: {e}")
            return 0.0

    def _points_for(self, width_units: float) -> float:
        """A width in canvas units as a line width in points, so it tracks the zoom."""
        per_unit = self._pixels_per_unit()
        if per_unit <= 0:
            return 1.0
        dpi = self._ax.figure.dpi or 72.0
        return max(0.5, width_units * per_unit * 72.0 / dpi)

    def _handle_offset_px(self) -> Optional[Tuple[float, float]]:
        """The rotation handle's offset from the centre, in display pixels.

        Up the body's own axis, a fixed distance on *screen*: the handle is a control,
        not a feature of the sample, so it must stay the same size and reach at every
        zoom -- and the canvas zooms without telling an overlay, so anything sized in
        data units at draw time was wrong a scroll later.
        """
        if self._centre is None or self._ax is None:
            return None
        cx, cy = self._centre
        try:
            (x0, y0), (x1, y1) = self._ax.transData.transform(
                [(cx, cy), self.to_canvas(0.0, -1.0)]
            )
        except Exception as e:
            logger.debug(f"Could not place the handle: {e}")
            return None
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length == 0:
            return None
        return dx / length * HANDLE_DISTANCE_PX, dy / length * HANDLE_DISTANCE_PX

    def _handle_position(self) -> Optional[Tuple[float, float]]:
        """Where the rotation handle sits, in canvas coordinates, right now."""
        offset = self._handle_offset_px()
        if offset is None:
            return None
        cx, cy = self._centre
        try:
            x0, y0 = self._ax.transData.transform((cx, cy))
            x, y = self._ax.transData.inverted().transform(
                (x0 + offset[0], y0 + offset[1])
            )
        except Exception as e:
            logger.debug(f"Could not place the handle: {e}")
            return None
        return float(x), float(y)

    def _draw_handles(self, animated: bool) -> list:
        from matplotlib.transforms import ScaledTranslation

        offset = self._handle_offset_px()
        if offset is None:
            return []
        cx, cy = self._centre
        fig = self._ax.figure
        dpi = fig.dpi or 72.0
        points = 72.0 / dpi  # one display pixel, in points
        artists = []
        # The stalk: from the centre out to the handle, as a pixel offset so it keeps
        # its length through a zoom. An annotation with no text is the one artist
        # that draws a line between a data point and a screen offset from it.
        stalk = self._ax.annotate(
            "",
            xy=(cx, cy),
            xycoords="data",
            xytext=offset,
            textcoords="offset pixels",
            arrowprops=dict(
                arrowstyle="-",
                color=_HANDLE_COLOUR,
                linestyle=(0, (3, 3)),
                linewidth=1.0,
                shrinkA=0,
                shrinkB=0,
            ),
            annotation_clip=False,
            zorder=30,
            animated=animated,
        )
        artists.append(stalk)
        # Markers, sized in points, so they are the same size at every zoom.
        (ring,) = self._ax.plot(
            [cx],
            [cy],
            marker="o",
            markersize=2 * HANDLE_RADIUS_PX * 0.7 * points,
            markerfacecolor="none",
            markeredgecolor=_HANDLE_COLOUR,
            markeredgewidth=2.0,
            linestyle="none",
            zorder=31,
            animated=animated,
        )
        artists.append(ring)
        (knob,) = self._ax.plot(
            [cx],
            [cy],
            marker="o",
            markersize=2 * HANDLE_RADIUS_PX * points,
            color=_HANDLE_COLOUR,
            linestyle="none",
            transform=self._ax.transData
            + ScaledTranslation(offset[0] / dpi, offset[1] / dpi, fig.dpi_scale_trans),
            zorder=31,
            animated=animated,
        )
        artists.append(knob)
        return artists

    # ── blitting, as the tile grid does it ────────────────────────────────

    def _can_blit(self) -> bool:
        if self._canvas is None or self._ax is None or not self.is_dragging:
            return False
        return all(
            hasattr(self._canvas, name)
            for name in ("draw", "copy_from_bbox", "restore_region", "blit")
        )

    def _blit_frame(self) -> bool:
        try:
            if self._blit_bg is None:
                self._canvas.draw()
                self._blit_bg = self._canvas.copy_from_bbox(self._ax.bbox)
            self._canvas.restore_region(self._blit_bg)
            for artist in self._artists:
                self._ax.draw_artist(artist)
            self._canvas.blit(self._ax.bbox)
            return True
        except Exception as e:  # pragma: no cover - a backend that cannot blit
            logger.debug(f"Could not blit: {e}")
            self._blit_bg = None
            return False

    def _on_canvas_drawn(self, event) -> None:
        """Keep up with a repaint: put the body back mid-drag, or catch a mode change.

        Mid-drag, the same restore the tile grid does (and, as there, no `blit()` from
        inside a paint). Outside a drag this is the one place that notices the canvas
        switching this overlay in or out of its active mode -- the canvas only asks for
        a redraw, and a plain redraw does not rebuild artists -- so the handles are put
        up or taken down with a deferred redraw of our own.
        """
        if self._blit_bg is not None and self.is_dragging:
            try:
                self._blit_bg = self._canvas.copy_from_bbox(self._ax.bbox)
                for artist in self._artists:
                    self._ax.draw_artist(artist)
            except Exception as e:  # pragma: no cover
                logger.debug(f"Could not restore after a repaint: {e}")
                self._blit_bg = None
            return
        if self._visible and self._handles_drawn != self.is_editable():
            QTimer.singleShot(0, self._redraw)

    def _end_blit(self) -> None:
        if self._blit_bg is None:
            return
        self._blit_bg = None
        self._redraw()

    # ── the gesture ───────────────────────────────────────────────────────

    def _near_handle(self, x: float, y: float) -> bool:
        handle = self._handle_position()
        per_unit = self._pixels_per_unit()
        if handle is None or per_unit <= 0:
            return False
        radius = HANDLE_RADIUS_PX / per_unit
        return math.hypot(x - handle[0], y - handle[1]) <= radius

    def _angle_at(self, x: float, y: float) -> float:
        """The pointer's angle about the centre, in the body's unsquashed frame."""
        cx, cy = self._centre
        return math.degrees(math.atan2((y - cy) / self._squash, x - cx))

    def _claim(self) -> None:
        if self._canvas is not None:
            self._canvas._overlay_consuming_event = True

    def _on_press(self, event) -> None:
        if event.button != 1 or event.inaxes is not self._ax or event.xdata is None:
            return
        if not self._visible or self._centre is None or not self.is_editable():
            return
        if getattr(event, "dblclick", False):
            return
        if self._near_handle(event.xdata, event.ydata):
            self._rotate_start = (
                self._angle_at(event.xdata, event.ydata),
                self._rotation,
            )
            self._claim()
            return
        if not self._body_contains(event.xdata, event.ydata):
            return
        self._move_start = (
            event.x,
            event.y,
            event.xdata,
            event.ydata,
            self._centre[0],
            self._centre[1],
        )
        self._claim()

    def _on_motion(self, event) -> None:
        if self._rotate_start is not None:
            self._drag_rotate(event)
            return
        if self._move_start is not None:
            self._drag_move(event)
            return
        self._update_cursor(event)

    def _drag_move(self, event) -> None:
        press_x, press_y, x0, y0, cx0, cy0 = self._move_start
        if event.inaxes is not self._ax or event.xdata is None or event.ydata is None:
            return
        if not self._move_active:
            travelled = (event.x - press_x) ** 2 + (event.y - press_y) ** 2
            if travelled < MOVE_DRAG_THRESHOLD_PX**2:
                return
            self._move_active = True
        # Absolute, measured from where the centre was at press, so the host's
        # rounding on each step cannot accumulate.
        self.moved.emit(cx0 + (event.xdata - x0), cy0 + (event.ydata - y0))

    def _drag_rotate(self, event) -> None:
        angle0, rotation0 = self._rotate_start
        if event.inaxes is not self._ax or event.xdata is None or event.ydata is None:
            return
        if self._centre is None:
            return
        self._rotate_active = True
        delta = self._angle_at(event.xdata, event.ydata) - angle0
        delta = (delta + 180.0) % 360.0 - 180.0
        self.rotated.emit(rotation0 + delta)

    def _on_release(self, event) -> None:
        was_dragged = self._move_active or self._rotate_active
        self._move_start = None
        self._move_active = False
        self._rotate_start = None
        self._rotate_active = False
        # Before anything below can redraw: `_end_blit` reads `is_dragging`, and it is
        # that transition it exists to catch.
        self._end_blit()
        if was_dragged:
            self.drag_finished.emit()
        self._update_cursor(event)

    def _update_cursor(self, event) -> None:
        if self._canvas is None:
            return
        cursor = None
        if (
            self._visible
            and self._centre is not None
            and self.is_editable()
            and event.inaxes is self._ax
            and event.xdata is not None
        ):
            if self._near_handle(event.xdata, event.ydata):
                cursor = Qt.CrossCursor
            elif self._body_contains(event.xdata, event.ydata):
                cursor = Qt.SizeAllCursor
        if cursor is not None:
            self._canvas.setCursor(cursor)
            self._cursor_set = True
        elif self._cursor_set:
            self._canvas.unsetCursor()
            self._cursor_set = False
