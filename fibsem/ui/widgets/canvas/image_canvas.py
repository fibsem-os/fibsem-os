"""FibsemImageCanvas — reusable matplotlib image canvas with pluggable overlays.

Zoom: scroll wheel centred on cursor.
Pan: left-drag on empty canvas area.

Overlays implement a simple duck-typed protocol::

    class MyOverlay:
        def attach(self, ax, canvas: FibsemImageCanvas) -> None: ...
        def detach(self) -> None: ...
        def on_image_changed(self, width: int, height: int) -> None: ...

Overlays that need Qt signals extend QObject directly.  An overlay that wants
to suppress canvas pan/zoom during a drag sets ``canvas._overlay_consuming_event = True``
on button-press; the canvas clears the flag automatically on button-release.

The overlay classes themselves live in the :mod:`fibsem.ui.widgets.canvas.overlays`
package (``CanvasOverlay`` base + ``PointsOverlay`` / ``PointOverlay`` /
``RectOverlay`` / ``RulerOverlay`` / ``PatternOverlay`` / ``ScanDirectionArrowOverlay``
and the milling / mask / alignment / minimap overlays).
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import QSize, QTimer, Qt, pyqtSignal
from PyQt5.QtWidgets import QApplication, QPushButton, QSizePolicy
from superqt import QIconifyIcon

from fibsem.structures import FibsemImage
from fibsem.ui.stylesheets import CANVAS_BG as _BG, PRIMARY_ACCENT as _ACCENT
from fibsem.ui.widgets.canvas.contrast_gamma_control import ContrastGammaControl

if TYPE_CHECKING:
    from fibsem.ui.widgets.canvas.overlays.base import CanvasOverlay
    from fibsem.ui.widgets.canvas.overlays.ruler_overlay import RulerOverlay

_logger = logging.getLogger(__name__)

_MAX_DISPLAY_PX = 2048
_ZOOM_FACTOR = 1.15
_REDRAW_INTERVAL = 32  # ms (~60 fps)

_OVERLAY_BTN_STYLE = (
    "QPushButton { background: rgba(40,41,48,180); border: 1px solid #555;"
    " border-radius: 3px; padding: 0px; }"
    "QPushButton:hover { background: rgba(74,74,74,200); }"
    "QPushButton:pressed { background: rgba(30,30,30,220); }"
    "QPushButton:checked { background: rgba(90,92,100,200); border-color: #FFFFFF; }"
)
_OVERLAY_ICON_SIZE = QSize(14, 14)
_OVERLAY_BTN_SIZE = 22
_OVERLAY_MARGIN = 4
_OVERLAY_GAP = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _downsample(arr: np.ndarray, max_px: int) -> np.ndarray:
    h, w = arr.shape[:2]
    if h <= max_px and w <= max_px:
        return arr
    stride = max(1, math.ceil(max(h, w) / max_px))
    return arr[::stride, ::stride] if arr.ndim == 2 else arr[::stride, ::stride, :]


# Keyboard modifiers mapped to napari-style strings, so handler bodies that
# branch on ``"Alt" in modifiers`` port across from the napari callbacks verbatim.
_QT_MODIFIER_MAP = (
    (Qt.AltModifier, "Alt"),
    (Qt.ShiftModifier, "Shift"),
    (Qt.ControlModifier, "Control"),
    (Qt.MetaModifier, "Meta"),
)


def _modifiers_from_event(event) -> Tuple[str, ...]:
    """Active keyboard modifiers as napari-style strings, e.g. ``("Alt",)``.

    Reads the underlying Qt event (``event.guiEvent``) — the Qt modifier state is
    the reliable source in an embedded canvas, whereas matplotlib's
    ``MouseEvent.key`` depends on canvas keyboard focus.  Falls back to the
    application-wide modifier state when no Qt event is attached.
    """
    gui = getattr(event, "guiEvent", None)
    mods = gui.modifiers() if gui is not None else QApplication.keyboardModifiers()
    return tuple(name for flag, name in _QT_MODIFIER_MAP if mods & flag)


# ---------------------------------------------------------------------------
# FibsemImageCanvas
# ---------------------------------------------------------------------------


class FibsemImageCanvas(FigureCanvasQTAgg):
    """Reusable matplotlib canvas for FibsemImage.

    * Scroll-wheel zoom centred on cursor
    * Left-drag pan on empty area
    * Pluggable overlay objects via add_overlay() / remove_overlay()
    * Optional scalebar (auto-populated from FibsemImage.metadata.pixel_size)
    """

    # Trailing ``object`` is a tuple of napari-style modifier strings, e.g. ("Alt",).
    canvas_clicked = pyqtSignal(float, float, object)  # left single-click (x, y) px, mods
    canvas_double_clicked = pyqtSignal(float, float, object)  # left double-click (x, y) px, mods
    canvas_right_clicked = pyqtSignal(float, float, object)  # right single-click (x, y) px, mods
    canvas_scrolled = pyqtSignal(float, float, int, object)  # (x, y) px, dir +1/-1, mods

    def __init__(self, parent=None):
        self._fig = Figure(facecolor=_BG)
        # Axes + figure background; overridable via set_background_color (the minimap
        # uses black). The label/hint bboxes keep their own colours.
        self._facecolor = _BG
        # Extra empty space around the image when fitting the view, as a fraction of the
        # image size per side (0 = tight to the image; set via set_view_margin). Lets
        # overlays that extend past the image (stage limits, grid boundary) stay visible.
        self._view_margin = 0.0
        super().__init__(self._fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._ax = self._fig.add_subplot(111)
        self._ax.set_facecolor(self._facecolor)
        self._ax.axis("off")
        self._fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        self._img_w: Optional[int] = None
        self._img_h: Optional[int] = None
        self._overlays: List[CanvasOverlay] = []
        self._pan_start: Optional[Tuple] = None
        # Modifiers captured at left-press, emitted with canvas_clicked on release
        self._press_modifiers: Tuple[str, ...] = ()

        # Overlays set this True on press to suppress canvas pan
        self._overlay_consuming_event: bool = False

        # Active-overlay input gating. None = default "Move" (full navigation +
        # stage movement + milling menu). When set, that overlay owns input and
        # the canvas suppresses its semantic click signals; see the design doc's
        # active-overlay model. _mode_overlay/_mode_label back the toolbar toggle.
        self._active_overlay = None
        self._mode_overlay = None
        self._mode_label: str = ""

        self._pixel_size: Optional[float] = None
        self._scalebar_artist = None
        self._scalebar_visible: bool = True
        self._crosshair_visible: bool = True
        self._crosshair_artists: list = []
        self._hint_artist = None  # transient top-left instruction hint
        self._hint_text: Optional[str] = None  # remembered so it survives set_image
        self._info_artist = None  # bottom-left microscope-state info bar
        self._info_text: Optional[str] = None  # remembered so it survives set_image

        # Transient top-centre flash message (e.g. "WD 4.001 mm" on Shift+scroll); auto-clears
        self._flash_artist = None
        self._flash_text: Optional[str] = None
        self._flash_timer = QTimer(self)
        self._flash_timer.setSingleShot(True)
        self._flash_timer.timeout.connect(self._clear_flash)

        # Optional patch legend (list of (color, label)); re-applied across image changes.
        self._legend_artist = None
        self._legend_entries: Optional[List[Tuple[str, str]]] = None
        self._legend_loc: str = "upper right"

        # Drag-to-measure ruler (lazily created on first toggle; see toggle_ruler).
        # _ruler_prev_active restores whatever overlay owned input before measuring.
        self._ruler_overlay: Optional["RulerOverlay"] = None
        self._ruler_prev_active = None

        # Contrast / gamma (display-only; applied to the downsampled grayscale frame)
        self._display_base: Optional[np.ndarray] = None
        self._norm: Optional[np.ndarray] = None  # normalized base, computed lazily
        self._is_gray: bool = True

        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(_REDRAW_INTERVAL)
        self._redraw_timer.timeout.connect(self.draw_idle)

        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("motion_notify_event", self._on_motion)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("scroll_event", self._on_scroll)
        self.mpl_connect("resize_event", lambda _: self.draw_idle())

        # Overlay buttons (parented to self; repositioned in resizeEvent)
        self._overlay_buttons: List[QPushButton] = []
        # Toolbar-group visibility (quad-view "only the selected view shows its toolbar").
        # _toolbar_hidden snapshots the buttons hidden by the group toggle so they restore
        # exactly (buttons hidden for other reasons — e.g. FM hides btn_contrast — stay hidden).
        self._toolbar_visible: bool = True
        self._toolbar_hidden: List[QPushButton] = []
        self.btn_reset_view = self._add_overlay_button(
            "mdi:fit-to-screen-outline", "Reset view", self.reset_view
        )
        self.btn_toggle_scalebar = self._add_overlay_button(
            "mdi:arrow-expand-horizontal", "Hide scalebar", self.toggle_scalebar, checkable=True
        )
        self.btn_toggle_scalebar.setChecked(True)
        self.btn_toggle_crosshair = self._add_overlay_button(
            "mdi:crosshairs", "Hide crosshair", self.toggle_crosshair, checkable=True
        )
        self.btn_toggle_crosshair.setChecked(True)
        self.btn_contrast = self._add_overlay_button(
            "mdi:contrast-box", "Contrast / Gamma", self.toggle_contrast, checkable=True
        )
        self.btn_toggle_ruler = self._add_overlay_button(
            "mdi:ruler", "Measure (ruler)", self.toggle_ruler, checkable=True
        )
        # Contextual mode toggle — shown only while an overlay owns input
        # (enter_overlay_mode). Checked = active; unchecking returns to Move.
        self.btn_mode = self._add_overlay_button(
            "mdi:cursor-default-click", "", self._on_mode_button_clicked, checkable=True
        )
        self.btn_mode.hide()

        # Floating contrast / gamma popover, anchored under btn_contrast
        self._contrast = ContrastGammaControl(self)
        self._contrast.changed.connect(self._apply_contrast)

        self._plot_empty()

    # ── properties ────────────────────────────────────────────────────────

    @property
    def img_width(self) -> Optional[int]:
        return self._img_w

    @property
    def img_height(self) -> Optional[int]:
        return self._img_h

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

        self._fit_view()

        # Scalebar
        self._scalebar_artist = None
        if pixel_size and pixel_size > 0:
            self._pixel_size = pixel_size
        self._refresh_scalebar()
        self._refresh_crosshair()
        self._refresh_hint()  # axes was cleared above; restore the remembered hint
        self._refresh_info_bar()  # ditto: restore the remembered info bar
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
        if pixel_size and pixel_size > 0 and pixel_size != self._pixel_size:
            self._pixel_size = pixel_size
            self._refresh_scalebar()
        self.draw_idle()

    def set_crosshair_visible(self, visible: bool) -> None:
        """Show or hide the yellow crosshair centred on the image."""
        self._crosshair_visible = visible
        self._refresh_crosshair()
        self.draw_idle()

    def set_hint(self, text: Optional[str]) -> None:
        """Show a small instruction hint in the top-left corner, or hide with None.

        Drawn in axes-fraction coords so it stays fixed through zoom/pan.  The text
        is remembered and re-applied after each image change (``set_image`` clears
        the axes), so the hint is not silently dropped by a new acquisition.
        """
        self._hint_text = text or None
        self._refresh_hint()
        self.draw_idle()

    def _refresh_hint(self) -> None:
        """(Re)create the hint artist from the cached text, or remove it."""
        if self._hint_artist is not None:
            try:
                self._hint_artist.remove()
            except Exception:
                pass
            self._hint_artist = None
        if self._hint_text:
            self._hint_artist = self._ax.text(
                0.012, 0.985, self._hint_text,
                transform=self._ax.transAxes, ha="left", va="top",
                fontsize=8, color="#1a1a1a", zorder=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#e6e6e6",
                          edgecolor="none", alpha=0.85),
            )

    def set_info_text(self, text: Optional[str]) -> None:
        """Show a small, muted info bar in the bottom-left, or hide with None/''.

        Remembered + re-applied after each image change (like the hint). Driven by
        the controller from the canvas-state model — microscope state, not image."""
        self._info_text = text or None
        self._refresh_info_bar()
        self.draw_idle()

    def _refresh_info_bar(self) -> None:
        """(Re)create the info artist from the cached text, or remove it."""
        if self._info_artist is not None:
            try:
                self._info_artist.remove()
            except Exception:
                pass
            self._info_artist = None
        if self._info_text:
            self._info_artist = self._ax.text(
                0.012, 0.015, self._info_text,
                transform=self._ax.transAxes, ha="left", va="bottom",
                fontsize=6.5, color="#e8e8e8", zorder=11,
                bbox=dict(boxstyle="round,pad=0.25", facecolor=_BG,
                          edgecolor="none", alpha=0.55),
            )

    def flash_message(self, text: str, duration_ms: int = 1200) -> None:
        """Show a brief top-centre status message that auto-clears after *duration_ms*.

        Repeated calls refresh the text and restart the timer, so it stays visible during a
        burst (e.g. Shift+scroll working-distance nudges) and fades shortly after the last
        event. Independent of :meth:`set_hint` / :meth:`set_info_text` — transient, not
        remembered across image changes."""
        self._flash_text = text or None
        self._refresh_flash()
        self.draw_idle()
        if self._flash_text:
            self._flash_timer.start(duration_ms)

    def _refresh_flash(self) -> None:
        """(Re)create the flash artist from the cached text, or remove it."""
        if self._flash_artist is not None:
            try:
                self._flash_artist.remove()
            except Exception:
                pass
            self._flash_artist = None
        if self._flash_text:
            self._flash_artist = self._ax.text(
                0.5, 0.975, self._flash_text,
                transform=self._ax.transAxes, ha="center", va="top",
                fontsize=9, color="#e8e8e8", zorder=12,
                bbox=dict(boxstyle="round,pad=0.35", facecolor=_BG,
                          edgecolor=_ACCENT, linewidth=1.0, alpha=0.85),
            )

    def _clear_flash(self) -> None:
        self._flash_text = None
        self._refresh_flash()
        self.draw_idle()

    def set_legend(self, entries, loc: str = "upper right") -> None:
        """Show a small patch legend, or clear it with None / an empty list.

        *entries* is a sequence of ``(color, label)`` pairs, each drawn as a filled
        swatch. Remembered and re-applied after every image change (like the hint /
        info bar), so a new frame doesn't silently drop it."""
        self._legend_entries = list(entries) if entries else None
        self._legend_loc = loc
        self._refresh_legend()
        self.draw_idle()

    def _refresh_legend(self) -> None:
        """(Re)create the legend artist from the cached entries, or remove it."""
        if self._legend_artist is not None:
            try:
                self._legend_artist.remove()
            except Exception:
                pass
            self._legend_artist = None
        if not self._legend_entries:
            return
        import matplotlib.patches as mpatches
        from matplotlib.legend import Legend

        labels = [label for _, label in self._legend_entries]
        handles = [
            mpatches.Patch(facecolor=color, edgecolor="white", label=label)
            for color, label in self._legend_entries
        ]
        # Build the Legend directly (not ax.legend) so it doesn't replace an overlay's
        # own legend (e.g. milling stages); styled like the point/milling legends.
        leg = Legend(
            self._ax, handles, labels, loc=self._legend_loc,
            fontsize=7, facecolor=_BG, edgecolor="#555555",
            labelcolor="#d1d2d4", framealpha=0.85,
        )
        leg.set_zorder(10)
        self._ax.add_artist(leg)
        self._legend_artist = leg

    def clear(self) -> None:
        """Clear the image and show placeholder text."""
        self._img_w = self._img_h = None
        self._display_base = None
        self._norm = None
        self._ax.cla()
        self._scalebar_artist = None
        self._crosshair_artists = []
        self._hint_artist = None  # removed by cla(); drop the cached text too
        self._hint_text = None
        self._info_artist = None
        self._info_text = None
        self._flash_artist = None
        self._flash_text = None
        self._flash_timer.stop()
        self._legend_artist = None  # removed by cla(); drop the cached entries too
        self._legend_entries = None
        self._plot_empty()
        for overlay in self._overlays:
            try:
                overlay.on_image_changed(0, 0)
            except Exception:
                pass
        self.draw_idle()

    # ── overlay buttons ───────────────────────────────────────────────────

    def add_toolbar_button(
        self, icon_name: str, tooltip: str, callback, checkable: bool = False
    ) -> QPushButton:
        """Public: add a custom button to the canvas's top-right toolbar (e.g. an
        FM-layers control owned by a wrapper). Returns the QPushButton."""
        return self._add_overlay_button(icon_name, tooltip, callback, checkable)

    def _add_overlay_button(
        self,
        icon_name: str,
        tooltip: str,
        callback,
        checkable: bool = False,
    ) -> QPushButton:
        """Create an overlay button parented to this canvas and register it.

        Buttons are stacked right-to-left in the top-right corner and
        repositioned automatically on resize.  Returns the button.
        """
        btn = QPushButton(self)
        btn.setIcon(QIconifyIcon(icon_name, color="#aaaaaa"))
        btn.setIconSize(_OVERLAY_ICON_SIZE)
        btn.setFixedSize(_OVERLAY_BTN_SIZE, _OVERLAY_BTN_SIZE)
        btn.setToolTip(tooltip)
        btn.setCheckable(checkable)
        btn.setStyleSheet(_OVERLAY_BTN_STYLE)
        btn.clicked.connect(callback)
        btn.raise_()
        self._overlay_buttons.append(btn)
        self._reposition_overlay_buttons()
        return btn

    def _reposition_overlay_buttons(self) -> None:
        """Place overlay buttons right-to-left in the top-right corner."""
        x = self.width() - _OVERLAY_MARGIN
        for btn in self._overlay_buttons:
            if btn.isHidden():  # contextual buttons (e.g. mode toggle) reserve no slot
                continue
            x -= btn.width()
            btn.move(x, _OVERLAY_MARGIN)
            x -= _OVERLAY_GAP

    def set_toolbar_visible(self, visible: bool) -> None:
        """Show or hide this canvas's top-right toolbar buttons as a group.

        Used by the quad view so only the selected canvas shows its toolbar. The
        contextual mode toggle (:attr:`btn_mode`) is exempt — it follows overlay-mode
        state, not selection — so an in-progress edit on a non-selected canvas keeps its
        toggle. Buttons already hidden for other reasons (e.g. the FM canvas hides
        ``btn_contrast``) stay hidden when the group is shown again.
        """
        if visible == self._toolbar_visible:
            return
        self._toolbar_visible = visible
        if not visible:
            self._toolbar_hidden = [
                b for b in self._overlay_buttons
                if b is not self.btn_mode and not b.isHidden()
            ]
            for b in self._toolbar_hidden:
                b.hide()
        else:
            for b in self._toolbar_hidden:
                b.show()
            self._toolbar_hidden = []
        self._reposition_overlay_buttons()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._reposition_overlay_buttons()
        contrast = getattr(self, "_contrast", None)
        if contrast is not None and contrast.isVisible():
            contrast.reposition()

    def _fit_view(self) -> None:
        """Set the view to the image extent expanded by ``_view_margin`` on each side."""
        imgs = self._ax.get_images()
        if not imgs:
            return
        xmin, xmax, ybot, ytop = imgs[0].get_extent()  # (xmin, xmax, ymax, ymin)
        mx = self._view_margin * (xmax - xmin)
        my = self._view_margin * abs(ybot - ytop)
        self._ax.set_xlim(xmin - mx, xmax + mx)
        self._ax.set_ylim(ybot + my, ytop - my)  # y-axis stays inverted (origin upper)

    def set_view_margin(self, frac: float) -> None:
        """Empty space kept around the image when fitting the view, as a fraction of the
        image size per side (0 = tight, 0.5 = 2x the image extent). Also keeps overlays
        that extend beyond the image (e.g. stage limits) visible."""
        self._view_margin = max(0.0, float(frac))
        self._fit_view()
        self._schedule_redraw()

    def set_background_color(self, color: str) -> None:
        """Set the axes + figure background colour (the area around the image)."""
        self._facecolor = color
        self._fig.set_facecolor(color)
        self._ax.set_facecolor(color)
        self._schedule_redraw()

    def reset_view(self) -> None:
        """Fit the view to the image extent (plus any view margin)."""
        self._fit_view()
        self._schedule_redraw()

    def add_overlay(self, overlay: CanvasOverlay) -> None:
        """Register an overlay and attach it to the current axes."""
        self._overlays.append(overlay)
        overlay.attach(self._ax, self)
        if self._img_w is not None:
            try:
                overlay.on_image_changed(self._img_w, self._img_h)
            except Exception:
                _logger.exception("Overlay on_image_changed failed: %r", overlay)
        self.draw_idle()

    def remove_overlay(self, overlay: CanvasOverlay) -> None:
        if overlay in self._overlays:
            self.exit_overlay_mode(overlay)  # no-op unless it owns the mode
            if overlay is self._active_overlay:  # active without a toolbar mode
                self._active_overlay = None
            try:
                overlay.detach()
            except Exception:
                _logger.exception("Overlay detach failed: %r", overlay)
            self._overlays.remove(overlay)
            self.draw_idle()

    def clear_overlays(self) -> None:
        for o in list(self._overlays):
            self.remove_overlay(o)

    # ── active-overlay input gating ───────────────────────────────────────

    @property
    def active_overlay(self):
        """The overlay currently owning input, or None (default 'Move' mode)."""
        return self._active_overlay

    def set_active_overlay(self, overlay) -> None:
        """Make *overlay* the sole input handler on this canvas (None = Move).

        While set, the canvas suppresses its semantic click signals
        (``canvas_clicked`` / ``canvas_double_clicked`` / ``canvas_right_clicked``),
        so stage movement and the milling menu stand down and other interactive
        overlays stand down; pan / zoom / scroll stay live. This is the low-level
        primitive — :meth:`enter_overlay_mode` wraps it with the toolbar toggle.
        """
        self._active_overlay = overlay
        self.draw_idle()

    def _overlay_input_allowed(self, overlay) -> bool:
        """True if *overlay* may handle input now (nothing active, or it's active)."""
        return self._active_overlay is None or self._active_overlay is overlay

    def enter_overlay_mode(
        self, overlay, label: str, icon: str = "mdi:cursor-default-click"
    ) -> None:
        """Activate *overlay* and show the contextual toolbar toggle (checked).

        Checked = the overlay owns input; unchecking returns to Move (re-enables
        stage movement); re-checking re-activates. Call :meth:`exit_overlay_mode`
        when the workflow step ends.
        """
        self._mode_overlay = overlay
        self._mode_label = label
        self.btn_mode.setIcon(QIconifyIcon(icon, color="#aaaaaa"))
        self.btn_mode.setToolTip(f"{label} active — click to enable Move")
        self.btn_mode.setChecked(True)
        self.btn_mode.show()
        self._reposition_overlay_buttons()
        self.set_active_overlay(overlay)

    def exit_overlay_mode(self, overlay=None) -> None:
        """Deactivate the overlay mode and hide the toolbar toggle (idempotent).

        Pass *overlay* to scope the exit — it's a no-op unless that overlay owns
        the current mode, so one caller can't tear down another's mode (POI and
        alignment editing share the FIB canvas). ``None`` forces an exit.
        """
        if overlay is not None and overlay is not self._mode_overlay:
            return
        self._mode_overlay = None
        self.btn_mode.setChecked(False)
        self.btn_mode.hide()
        self._reposition_overlay_buttons()
        self.set_active_overlay(None)

    def _on_mode_button_clicked(self) -> None:
        """Toolbar toggle: flip between the bound overlay and Move (no teardown)."""
        if self._mode_overlay is None:
            return
        if self.btn_mode.isChecked():
            self.btn_mode.setToolTip(f"{self._mode_label} active — click to enable Move")
            self.set_active_overlay(self._mode_overlay)
        else:
            self.btn_mode.setToolTip(f"Click to resume {self._mode_label}")
            self.set_active_overlay(None)

    # ── internals ─────────────────────────────────────────────────────────

    def _plot_empty(self):
        self._ax.set_facecolor(self._facecolor)
        self._ax.axis("off")
        self._ax.text(
            0.5,
            0.5,
            "No image",
            ha="center",
            va="center",
            transform=self._ax.transAxes,
            fontsize=11,
            color="#bbbbbb",
        )

    def toggle_scalebar(self) -> None:
        """Show or hide the scalebar and update the button tooltip."""
        self._scalebar_visible = not self._scalebar_visible
        self.btn_toggle_scalebar.setChecked(self._scalebar_visible)
        self.btn_toggle_scalebar.setToolTip(
            "Hide scalebar" if self._scalebar_visible else "Show scalebar"
        )
        self._refresh_scalebar()
        self.draw_idle()

    def toggle_crosshair(self) -> None:
        """Show or hide the crosshair and update the button tooltip."""
        self.set_crosshair_visible(not self._crosshair_visible)
        self.btn_toggle_crosshair.setChecked(self._crosshair_visible)
        self.btn_toggle_crosshair.setToolTip(
            "Hide crosshair" if self._crosshair_visible else "Show crosshair"
        )

    def toggle_ruler(self) -> None:
        """Toggle the drag-to-measure ruler (a generic canvas tool).

        While on, the ruler owns input (so a stray double-click/right-click
        doesn't move the stage or open the milling menu); the previously active
        overlay, if any, is restored when the ruler is turned off.
        """
        if self.btn_toggle_ruler.isChecked():
            if self._ruler_overlay is None:
                from fibsem.ui.widgets.canvas.overlays.ruler_overlay import RulerOverlay

                self._ruler_overlay = RulerOverlay()
                self.add_overlay(self._ruler_overlay)
            self._ruler_prev_active = self._active_overlay
            self._ruler_overlay.set_visible(True)
            self.set_active_overlay(self._ruler_overlay)
            self.btn_toggle_ruler.setToolTip("Hide ruler")
        else:
            if self._ruler_overlay is not None:
                self._ruler_overlay.set_visible(False)
            self.set_active_overlay(self._ruler_prev_active)
            self._ruler_prev_active = None
            self.btn_toggle_ruler.setToolTip("Measure (ruler)")

    # ── contrast / gamma ──────────────────────────────────────────────────

    def toggle_contrast(self) -> None:
        """Show or hide the floating contrast / gamma popover."""
        self._contrast.set_open(self.btn_contrast.isChecked(), self.btn_contrast)

    def _contrast_display(self) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float]]]:
        """Return (array_to_show, clim) for the current contrast state.

        When the control is at its defaults (or the image is RGB) the raw
        downsampled frame is returned with ``clim=None`` — i.e. no change.
        """
        base = self._display_base
        if base is None:
            return None, None
        if self._is_gray and not self._contrast.is_default():
            if self._norm is None:
                self._norm = ContrastGammaControl.normalize(base)
            return self._contrast.apply(self._norm), (0.0, 1.0)
        return base, None

    def _apply_contrast(self) -> None:
        """Re-apply contrast/gamma to the live image without a full redraw."""
        imgs = self._ax.get_images()
        if not imgs:
            return
        to_show, clim = self._contrast_display()
        if to_show is None:
            return
        im = imgs[0]
        im.set_data(to_show)
        if clim is not None:
            im.set_clim(*clim)
        elif self._is_gray and self._display_base is not None:
            # back to default → restore the raw intensity range
            im.set_clim(float(self._display_base.min()), float(self._display_base.max()))
        self.draw_idle()

    def _refresh_scalebar(self):
        if self._scalebar_artist is not None:
            try:
                self._scalebar_artist.remove()
            except (ValueError, NotImplementedError):
                pass
            self._scalebar_artist = None
        if self._pixel_size is not None and self._scalebar_visible:
            try:
                from matplotlib_scalebar.scalebar import ScaleBar

                self._scalebar_artist = ScaleBar(
                    dx=self._pixel_size,
                    color="white",
                    box_color=_BG,
                    box_alpha=0.6,
                    location="lower right",
                )
                self._ax.add_artist(self._scalebar_artist)
            except Exception:
                pass

    def _refresh_crosshair(self):
        for a in self._crosshair_artists:
            try:
                a.remove()
            except (ValueError, NotImplementedError):
                pass
        self._crosshair_artists = []
        if not self._crosshair_visible or self._img_w is None:
            return
        cx, cy = self._img_w / 2.0, self._img_h / 2.0
        # Size both arms from the longest dimension so the crosshair stays square
        # (axes use aspect="equal", so equal data-unit arms are equal on screen).
        half = max(self._img_w, self._img_h) * 0.05 / 2
        kw = dict(color="yellow", linewidth=1, alpha=0.8, zorder=7)
        (h_line,) = self._ax.plot([cx - half, cx + half], [cy, cy], **kw)
        (v_line,) = self._ax.plot([cx, cx], [cy - half, cy + half], **kw)
        self._crosshair_artists = [h_line, v_line]

    def _schedule_redraw(self):
        if not self._redraw_timer.isActive():
            self._redraw_timer.start()

    # ── mouse events ──────────────────────────────────────────────────────

    def _on_press(self, event):
        if event.inaxes is not self._ax or event.xdata is None:
            return
        if self._img_w is None:
            # No image: axes span the default [0,1], so xdata/ydata are not image pixels.
            # Suppress clicks/pan so a stray double-click can't drive a stage move to (~0,0).
            return
        mods = _modifiers_from_event(event)
        if event.dblclick:
            # active overlay owns input → suppress stage-move double-click
            if event.button == 1 and self._active_overlay is None:
                self.canvas_double_clicked.emit(event.xdata, event.ydata, mods)
            return  # don't start a pan on double-click
        if event.button == 3:
            # active overlay owns input → suppress the right-click (milling) menu
            if self._active_overlay is None:
                self.canvas_right_clicked.emit(event.xdata, event.ydata, mods)
            return
        if event.button != 1:
            return
        # Capture now; canvas_clicked fires on release (after the drag-distance test)
        self._press_modifiers = mods
        inv = self._ax.transData.inverted()
        self._pan_start = (
            event.x,
            event.y,
            self._ax.get_xlim(),
            self._ax.get_ylim(),
            inv,
        )

    def _on_motion(self, event):
        # Overlay in drag mode — cancel any pending pan
        if self._overlay_consuming_event:
            self._pan_start = None
            return
        if self._pan_start is None:
            return
        if event.x is None or event.y is None:
            return
        sx0, sy0, xlim0, ylim0, inv0 = self._pan_start
        x0, y0 = inv0.transform((sx0, sy0))
        x1, y1 = inv0.transform((event.x, event.y))
        dx, dy = x1 - x0, y1 - y0
        self._ax.set_xlim(xlim0[0] - dx, xlim0[1] - dx)
        self._ax.set_ylim(ylim0[0] - dy, ylim0[1] - dy)
        self._schedule_redraw()

    def _on_release(self, event):
        was_consuming = self._overlay_consuming_event
        self._overlay_consuming_event = False
        if event.button == 1 and self._pan_start is not None:
            sx0, sy0, *_ = self._pan_start
            dist = ((event.x - sx0) ** 2 + (event.y - sy0) ** 2) ** 0.5
            if (
                dist < 3
                and not was_consuming
                and self._active_overlay is None  # active overlay owns the click
                and event.xdata is not None
                and event.ydata is not None
            ):
                self.canvas_clicked.emit(event.xdata, event.ydata, self._press_modifiers)
        self._pan_start = None

    def _on_scroll(self, event):
        if event.inaxes is not self._ax or event.xdata is None:
            return
        direction = 1 if event.button == "up" else -1
        mods = _modifiers_from_event(event)
        self.canvas_scrolled.emit(event.xdata, event.ydata, direction, mods)
        if mods:
            # modified scroll (e.g. Shift+scroll → objective) is claimed by a
            # consumer via canvas_scrolled; don't also zoom
            return
        factor = 1.0 / _ZOOM_FACTOR if direction == 1 else _ZOOM_FACTOR
        cx, cy = event.xdata, event.ydata
        xlim = self._ax.get_xlim()
        ylim = self._ax.get_ylim()
        self._ax.set_xlim(cx + (xlim[0] - cx) * factor, cx + (xlim[1] - cx) * factor)
        self._ax.set_ylim(cy + (ylim[0] - cy) * factor, cy + (ylim[1] - cy) * factor)
        self._schedule_redraw()
