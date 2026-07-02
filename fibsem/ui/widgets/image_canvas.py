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

Classes
-------
CanvasOverlay       — plain base (no-op hooks; sub-class or duck-type)
FibsemImageCanvas   — the canvas
PointsOverlay       — static scatter markers with labels
RectOverlay         — configurable rectangle (drag-only or drag+resize)
PatternOverlay      — milling shape patches (placeholder, coords in pixel space)
"""

from __future__ import annotations

import logging
import math
from typing import Any, List, Optional, Tuple

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle as MplRectangle
from PyQt5.QtCore import QObject, QSize, QTimer, Qt, pyqtSignal
from PyQt5.QtWidgets import QApplication, QPushButton, QSizePolicy
from superqt import QIconifyIcon

from fibsem.structures import FibsemImage
from fibsem.ui.widgets.contrast_gamma_control import ContrastGammaControl

_logger = logging.getLogger(__name__)

_MAX_DISPLAY_PX = 2048
_ZOOM_FACTOR = 1.15
_REDRAW_INTERVAL = 32  # ms (~60 fps)
_BG = "#1e2124"
_ACCENT = "#3a6ea5"  # primary accent (matches the quad-view selection border)

_RECT_FRAC = 0.25
_RECT_OFFSET = (1.0 - _RECT_FRAC) / 2.0

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


def _default_extents(H: int, W: int) -> Tuple[float, float, float, float]:
    """Return (x0, x1, y0, y1) for a centred 25 % box."""
    rw, rh = W * _RECT_FRAC, H * _RECT_FRAC
    x0, y0 = W * _RECT_OFFSET, H * _RECT_OFFSET
    return x0, x0 + rw, y0, y0 + rh


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
# Overlay base
# ---------------------------------------------------------------------------


class CanvasOverlay:
    """No-op base for canvas overlays.  Sub-class or use duck-typing."""

    def attach(self, ax, canvas: "FibsemImageCanvas") -> None:
        """Called once when the overlay is added.  Create artists / connect events."""

    def detach(self) -> None:
        """Remove all artists and disconnect mpl events."""

    def on_image_changed(self, width: int, height: int) -> None:
        """Called after ax.cla() + new image drawn.  Re-create artists here."""


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


# ---------------------------------------------------------------------------
# PointsOverlay — static scatter markers with optional labels
# ---------------------------------------------------------------------------


class PointsOverlay(CanvasOverlay):
    """Non-interactive scatter points.  Call set_points() to update."""

    def __init__(
        self,
        points: List[Tuple[float, float]] = (),
        color: str = "white",
        marker: str = "o",
        size: int = 8,
        label_prefix: str = "",
    ):
        self._points = list(points)
        self._color = color
        self._marker = marker
        self._size = size
        self._label_prefix = label_prefix
        self._ax = None
        self._canvas = None
        self._artists: list = []

    def attach(self, ax, canvas: FibsemImageCanvas) -> None:
        self._ax = ax
        self._canvas = canvas

    def detach(self) -> None:
        self._remove_artists()
        self._ax = None
        self._canvas = None

    def on_image_changed(self, width: int, height: int) -> None:
        self._remove_artists()
        if width > 0:
            self._draw()

    def set_points(self, points: List[Tuple[float, float]]) -> None:
        self._points = list(points)
        self._remove_artists()
        self._draw()
        if self._canvas is not None:
            self._canvas.draw_idle()

    # ── private ──

    def _remove_artists(self):
        for a in self._artists:
            try:
                a.remove()
            except Exception:
                pass
        self._artists.clear()

    def _draw(self):
        if self._ax is None:
            return
        for i, (x, y) in enumerate(self._points, 1):
            (line,) = self._ax.plot(
                x,
                y,
                marker=self._marker,
                markersize=self._size,
                color=self._color,
                markeredgecolor="white",
                markeredgewidth=0.8,
                linestyle="none",
                zorder=8,
            )
            self._artists.append(line)
            if self._label_prefix:
                ann = self._ax.annotate(
                    f"{self._label_prefix}{i}",
                    xy=(x, y),
                    xytext=(6, 4),
                    textcoords="offset points",
                    color=self._color,
                    fontsize=8,
                    zorder=9,
                )
                self._artists.append(ann)


# ---------------------------------------------------------------------------
# RectOverlay — configurable drag / drag+resize rectangle
# ---------------------------------------------------------------------------

_HANDLE_RADIUS_PX = 8  # screen-space hit radius for corner handles
# Corner handles: name → (x_fraction, y_fraction) within the rect
_CORNERS = {"tl": (0.0, 0.0), "tr": (1.0, 0.0), "bl": (0.0, 1.0), "br": (1.0, 1.0)}


class RectOverlay(QObject):
    """Rectangle overlay using ``MplRectangle`` + manual mouse handling.

    Both modes share the same drag/release logic.  ``resizable=True`` additionally
    draws four corner handles and supports resize by dragging them.

    Parameters
    ----------
    color : str
        Edge colour (and handle colour).
    facecolor : str | None
        Fill colour.  ``None`` → transparent.
    alpha : float
        Opacity of edge/fill.
    linewidth : int
        Edge linewidth in points.
    linestyle : str
        Matplotlib linestyle string (``"solid"``, ``"--"``, etc.)
    resizable : bool
        ``True`` → drag + four corner resize handles.
        ``False`` → drag only.

    Examples
    --------
    FIB — yellow filled, drag-only::

        RectOverlay(color="yellow", facecolor="yellow", alpha=0.5, resizable=False)

    FM — white dotted, drag + resize::

        RectOverlay(color="white", facecolor=None, linestyle="--", resizable=True)
    """

    rect_changed = pyqtSignal(dict)  # {x0,y0,x1,y1,cx,cy,width,height} pixels

    def __init__(
        self,
        color: str = "yellow",
        facecolor: Optional[str] = None,
        alpha: float = 0.5,
        linewidth: int = 2,
        linestyle: str = "solid",
        resizable: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self._color = color
        self._facecolor = facecolor if facecolor is not None else "none"
        self._alpha = alpha
        self._linewidth = linewidth
        self._linestyle = linestyle
        self._resizable = resizable

        self._ax = None
        self._canvas: Optional[FibsemImageCanvas] = None
        self._img_w: Optional[int] = None
        self._img_h: Optional[int] = None

        # Rect state in data coords
        self._x0 = self._y0 = self._x1 = self._y1 = 0.0
        self._saved: Optional[Tuple[float, float, float, float]] = None  # x0,y0,w,h

        # Artists
        self._patch: Optional[MplRectangle] = None
        self._handles: dict = {}  # name → Line2D marker

        # Drag state: None | "move" | "tl" | "tr" | "bl" | "br"
        self._drag_mode: Optional[str] = None
        self._drag_start_data: Optional[Tuple[float, float]] = None
        self._drag_start_rect: Optional[Tuple[float, float, float, float]] = None
        self._blit_bg = None  # background region captured at drag start

        self._interactive: bool = True
        self._cids: List[int] = []

    # ── overlay protocol ──────────────────────────────────────────────────

    def attach(self, ax, canvas: FibsemImageCanvas) -> None:
        self._ax = ax
        self._canvas = canvas
        self._cids = [
            canvas.mpl_connect("button_press_event", self._on_press),
            canvas.mpl_connect("motion_notify_event", self._on_motion),
            canvas.mpl_connect("button_release_event", self._on_release),
        ]

    def detach(self) -> None:
        if self._canvas is not None:
            for cid in self._cids:
                try:
                    self._canvas.mpl_disconnect(cid)
                except Exception:
                    pass
        self._cids = []
        self._remove_artists()
        self._ax = None
        self._canvas = None

    def on_image_changed(self, width: int, height: int) -> None:
        self._img_w, self._img_h = width, height
        if self._ax is None or width == 0 or height == 0:
            return
        self._rebuild()

    # ── public helpers ────────────────────────────────────────────────────

    def get_rect(self) -> dict:
        return _xywh_to_dict(
            self._x0, self._y0, self._x1 - self._x0, self._y1 - self._y0
        )

    def set_rect(self, x0: float, y0: float, width: float, height: float) -> None:
        self._x0, self._y0 = x0, y0
        self._x1, self._y1 = x0 + width, y0 + height
        self._saved = (x0, y0, width, height)
        self._update_artists()
        if self._canvas is not None:
            self._canvas.draw_idle()

    # ── build / teardown ──────────────────────────────────────────────────

    def _remove_artists(self):
        if self._patch is not None:
            try:
                self._patch.remove()
            except Exception:
                pass
            self._patch = None
        for h in self._handles.values():
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()

    def _rebuild(self):
        self._remove_artists()

        if self._saved is not None:
            x0, y0, w, h = self._saved
        else:
            ex = _default_extents(self._img_h, self._img_w)
            x0, y0, w, h = ex[0], ex[2], ex[1] - ex[0], ex[3] - ex[2]
            self._saved = (x0, y0, w, h)

        self._x0, self._y0 = x0, y0
        self._x1, self._y1 = x0 + w, y0 + h

        self._patch = MplRectangle(
            (self._x0, self._y0),
            w,
            h,
            linewidth=self._linewidth,
            edgecolor=self._color,
            facecolor=self._facecolor,
            linestyle=self._linestyle,
            alpha=self._alpha,
            zorder=5,
        )
        self._ax.add_patch(self._patch)

        if self._resizable:
            for name in _CORNERS:
                hx, hy = self._handle_pos(name)
                (line,) = self._ax.plot(
                    hx,
                    hy,
                    "s",
                    markersize=7,
                    color=self._color,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    zorder=6,
                    visible=self._interactive,
                )
                self._handles[name] = line

    def _update_artists(self):
        if self._patch is not None:
            self._patch.set_xy((self._x0, self._y0))
            self._patch.set_width(self._x1 - self._x0)
            self._patch.set_height(self._y1 - self._y0)
        for name, line in self._handles.items():
            hx, hy = self._handle_pos(name)
            line.set_xdata([hx])
            line.set_ydata([hy])

    def _handle_pos(self, name: str) -> Tuple[float, float]:
        fx, fy = _CORNERS[name]
        return self._x0 + fx * (self._x1 - self._x0), self._y0 + fy * (
            self._y1 - self._y0
        )

    # ── hit testing ───────────────────────────────────────────────────────

    def _hit_handle(self, event) -> Optional[str]:
        """Return corner name if click is within _HANDLE_RADIUS_PX of a handle."""
        trans = self._ax.transData
        for name, line in self._handles.items():
            hx, hy = self._handle_pos(name)
            sx, sy = trans.transform((hx, hy))
            if ((event.x - sx) ** 2 + (event.y - sy) ** 2) ** 0.5 < _HANDLE_RADIUS_PX:
                return name
        return None

    # ── mouse events ──────────────────────────────────────────────────────

    def _on_press(self, event):
        if not self._interactive:
            return
        # Another overlay owns input on this canvas
        if self._canvas is not None and not self._canvas._overlay_input_allowed(self):
            return
        if event.inaxes is not self._ax or event.button != 1:
            return
        if self._patch is None or event.xdata is None:
            return
        # Another overlay already claimed this event
        if self._canvas._overlay_consuming_event:
            return

        # Check corner handles first (resizable only)
        if self._resizable:
            hit = self._hit_handle(event)
            if hit is not None:
                self._start_drag(hit, event)
                return

        # Check rect body
        contains, _ = self._patch.contains(event)
        if contains:
            self._start_drag("move", event)

    def _set_animated(self, val: bool):
        if self._patch is not None:
            self._patch.set_animated(val)
        for h in self._handles.values():
            h.set_animated(val)

    def _blit(self):
        """Restore background and redraw only the overlay artists."""
        if self._blit_bg is None:
            self._canvas.draw_idle()
            return
        self._canvas.restore_region(self._blit_bg)
        if self._patch is not None:
            self._ax.draw_artist(self._patch)
        for h in self._handles.values():
            self._ax.draw_artist(h)
        self._canvas.blit(self._ax.bbox)

    def _start_drag(self, mode: str, event):
        self._drag_mode = mode
        self._drag_start_data = (event.xdata, event.ydata)
        self._drag_start_rect = (self._x0, self._y0, self._x1, self._y1)
        self._canvas._overlay_consuming_event = True
        # Mark artists animated so they're excluded from the background draw
        self._set_animated(True)
        self._canvas.draw()
        self._blit_bg = self._canvas.copy_from_bbox(self._ax.bbox)

    def _on_motion(self, event):
        if self._drag_mode is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        dx = event.xdata - self._drag_start_data[0]
        dy = event.ydata - self._drag_start_data[1]
        rx0, ry0, rx1, ry1 = self._drag_start_rect
        W, H = self._img_w, self._img_h

        if self._drag_mode == "move":
            w, h = rx1 - rx0, ry1 - ry0
            self._x0 = max(0.0, min(rx0 + dx, W - w))
            self._y0 = max(0.0, min(ry0 + dy, H - h))
            self._x1 = self._x0 + w
            self._y1 = self._y0 + h
        elif self._drag_mode == "tl":
            self._x0 = max(0.0, min(rx0 + dx, self._x1 - 1))
            self._y0 = max(0.0, min(ry0 + dy, self._y1 - 1))
        elif self._drag_mode == "tr":
            self._x1 = max(self._x0 + 1, min(rx1 + dx, W))
            self._y0 = max(0.0, min(ry0 + dy, self._y1 - 1))
        elif self._drag_mode == "bl":
            self._x0 = max(0.0, min(rx0 + dx, self._x1 - 1))
            self._y1 = max(self._y0 + 1, min(ry1 + dy, H))
        elif self._drag_mode == "br":
            self._x1 = max(self._x0 + 1, min(rx1 + dx, W))
            self._y1 = max(self._y0 + 1, min(ry1 + dy, H))

        self._update_artists()
        self._blit()

    def _on_release(self, event):
        if self._canvas is not None:
            self._canvas._overlay_consuming_event = False
        if self._drag_mode is not None:
            self._drag_mode = None
            self._saved = (self._x0, self._y0, self._x1 - self._x0, self._y1 - self._y0)
            # Restore normal rendering
            self._set_animated(False)
            self._blit_bg = None
            self._canvas.draw_idle()
            self.rect_changed.emit(self.get_rect())


def _xywh_to_dict(x: float, y: float, w: float, h: float) -> dict:
    return {
        "x0": round(x),
        "y0": round(y),
        "x1": round(x + w),
        "y1": round(y + h),
        "cx": round(x + w / 2),
        "cy": round(y + h / 2),
        "width": round(w),
        "height": round(h),
    }


# ---------------------------------------------------------------------------
# RulerOverlay — drag-to-measure line (distance in SI units)
# ---------------------------------------------------------------------------

_RULER_PICK_PX = 10  # screen-space hit radius for ruler endpoints / line body


def _clampf(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _format_distance(metres: float) -> str:
    """Auto-scale a distance in metres to a short SI string."""
    a = abs(metres)
    if a == 0:
        return "0 nm"
    if a < 1e-6:
        return f"{metres * 1e9:.1f} nm"
    if a < 1e-3:
        return f"{metres * 1e6:.2f} µm"
    if a < 1.0:
        return f"{metres * 1e3:.3f} mm"
    return f"{metres:.4f} m"


class RulerOverlay(CanvasOverlay):
    """Two-endpoint measure line on a :class:`FibsemImageCanvas`.

    Drag either endpoint — or the line body — to measure; the label shows the
    distance using the canvas pixel size (SI-formatted), or pixels when the
    pixel size is unknown.  Endpoints live in data (image-pixel) coordinates, so
    the ruler survives zoom / pan and image swaps.

    Inert until :meth:`set_visible(True)` (driven by the canvas ruler button):
    it builds no artists and captures no input while hidden, so canvases that
    never enable it are unaffected.
    """

    def __init__(self, color: str = "#ffd23f"):
        self._color = color
        self._ax = None
        self._canvas: Optional["FibsemImageCanvas"] = None
        self._img_w: Optional[int] = None
        self._img_h: Optional[int] = None
        self._pixel_size: Optional[float] = None

        # endpoints in data coords (None until seeded)
        self._p1: Optional[List[float]] = None
        self._p2: Optional[List[float]] = None
        self._visible: bool = False

        # artists
        self._line = None   # Line2D segment
        self._dots = None   # Line2D endpoint markers
        self._label = None  # Annotation (distance)

        # drag state: None | "p1" | "p2" | "line"
        self._drag: Optional[str] = None
        self._drag_start_data: Optional[Tuple[float, float]] = None
        self._drag_start_pts = None
        self._blit_bg = None
        self._cids: List[int] = []

    # ── overlay protocol ──────────────────────────────────────────────────

    def attach(self, ax, canvas: "FibsemImageCanvas") -> None:
        self._ax = ax
        self._canvas = canvas
        self._cids = [
            canvas.mpl_connect("button_press_event", self._on_press),
            canvas.mpl_connect("motion_notify_event", self._on_motion),
            canvas.mpl_connect("button_release_event", self._on_release),
        ]

    def detach(self) -> None:
        if self._canvas is not None:
            for cid in self._cids:
                try:
                    self._canvas.mpl_disconnect(cid)
                except Exception:
                    pass
        self._cids = []
        self._remove_artists()
        self._ax = None
        self._canvas = None

    def on_image_changed(self, width: int, height: int) -> None:
        self._img_w, self._img_h = width, height
        if self._canvas is not None:
            self._pixel_size = self._canvas._pixel_size
        if self._ax is None or width == 0 or height == 0 or not self._visible:
            self._remove_artists()  # inert while hidden / between images
            return
        if self._p1 is None or self._p2 is None:
            self._seed_default()
        else:
            self._clamp()
        self._rebuild()

    # ── public ────────────────────────────────────────────────────────────

    def set_visible(self, visible: bool) -> None:
        """Show/hide the ruler.  Endpoints persist while hidden."""
        self._visible = visible
        if not visible:
            self._remove_artists()
        elif self._img_w:
            if self._p1 is None or self._p2 is None:
                self._seed_default()
            self._rebuild()
        if self._canvas is not None:
            self._canvas.draw_idle()

    def measurement(self) -> Optional[float]:
        """Current distance in metres (or pixels when no pixel size), or None."""
        if self._p1 is None or self._p2 is None:
            return None
        d = math.hypot(self._p2[0] - self._p1[0], self._p2[1] - self._p1[1])
        return d * self._pixel_size if self._pixel_size else d

    # ── build / teardown ──────────────────────────────────────────────────

    def _seed_default(self) -> None:
        if not self._img_w or not self._img_h:
            return
        cx, cy = self._img_w / 2.0, self._img_h / 2.0
        half = self._img_w * 0.125
        self._p1 = [cx - half, cy]
        self._p2 = [cx + half, cy]

    def _clamp(self) -> None:
        if self._img_w is None:
            return
        for p in (self._p1, self._p2):
            p[0] = _clampf(p[0], 0.0, self._img_w)
            p[1] = _clampf(p[1], 0.0, self._img_h)

    def _remove_artists(self) -> None:
        for a in (self._line, self._dots, self._label):
            if a is not None:
                try:
                    a.remove()
                except Exception:
                    pass
        self._line = self._dots = self._label = None

    def _xs_ys(self) -> Tuple[List[float], List[float]]:
        return [self._p1[0], self._p2[0]], [self._p1[1], self._p2[1]]

    def _rebuild(self) -> None:
        self._remove_artists()
        if self._p1 is None or self._p2 is None:
            return
        xs, ys = self._xs_ys()
        (self._line,) = self._ax.plot(
            xs, ys, color=self._color, linewidth=1.6,
            solid_capstyle="round", zorder=8,
        )
        (self._dots,) = self._ax.plot(
            xs, ys, linestyle="none", marker="o", markersize=6,
            markerfacecolor=self._color, markeredgecolor="white",
            markeredgewidth=0.8, zorder=9,
        )
        mx, my = (xs[0] + xs[1]) / 2.0, (ys[0] + ys[1]) / 2.0
        self._label = self._ax.annotate(
            self._text(), xy=(mx, my), xytext=(0, 9),
            textcoords="offset points", ha="center", va="bottom",
            fontsize=8, color="#ffffff", zorder=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=_BG,
                      edgecolor=self._color, alpha=0.8, linewidth=0.8),
        )

    def _text(self) -> str:
        d = math.hypot(self._p2[0] - self._p1[0], self._p2[1] - self._p1[1])
        if self._pixel_size:
            return _format_distance(d * self._pixel_size)
        return f"{d:.0f} px"

    def _update_artists(self) -> None:
        xs, ys = self._xs_ys()
        self._line.set_data(xs, ys)
        self._dots.set_data(xs, ys)
        self._label.xy = ((xs[0] + xs[1]) / 2.0, (ys[0] + ys[1]) / 2.0)
        self._label.set_text(self._text())

    # ── hit testing (screen space) ────────────────────────────────────────

    def _screen(self, p: List[float]) -> Tuple[float, float]:
        return self._ax.transData.transform((p[0], p[1]))

    def _hit(self, event) -> Optional[str]:
        for name, p in (("p1", self._p1), ("p2", self._p2)):
            sx, sy = self._screen(p)
            if math.hypot(event.x - sx, event.y - sy) <= _RULER_PICK_PX:
                return name
        x1, y1 = self._screen(self._p1)
        x2, y2 = self._screen(self._p2)
        if self._seg_dist(event.x, event.y, x1, y1, x2, y2) <= _RULER_PICK_PX * 0.6:
            return "line"
        return None

    @staticmethod
    def _seg_dist(px, py, x1, y1, x2, y2) -> float:
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            return math.hypot(px - x1, py - y1)
        t = _clampf(((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy), 0.0, 1.0)
        return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))

    # ── mouse events ──────────────────────────────────────────────────────

    def _on_press(self, event) -> None:
        if not self._visible or self._line is None:
            return
        if self._canvas is not None and not self._canvas._overlay_input_allowed(self):
            return
        if event.inaxes is not self._ax or event.button != 1 or event.xdata is None:
            return
        if self._canvas._overlay_consuming_event:
            return
        hit = self._hit(event)
        if hit is None:
            return
        self._drag = hit
        self._drag_start_data = (event.xdata, event.ydata)
        self._drag_start_pts = (list(self._p1), list(self._p2))
        self._canvas._overlay_consuming_event = True
        self._set_animated(True)
        self._canvas.draw()
        self._blit_bg = self._canvas.copy_from_bbox(self._ax.bbox)

    def _on_motion(self, event) -> None:
        if self._drag is None or event.xdata is None or event.ydata is None:
            return
        dx = event.xdata - self._drag_start_data[0]
        dy = event.ydata - self._drag_start_data[1]
        p1, p2 = self._drag_start_pts
        W, H = self._img_w, self._img_h
        if self._drag == "p1":
            self._p1 = [_clampf(p1[0] + dx, 0, W), _clampf(p1[1] + dy, 0, H)]
        elif self._drag == "p2":
            self._p2 = [_clampf(p2[0] + dx, 0, W), _clampf(p2[1] + dy, 0, H)]
        else:  # move the whole line, clamped so both endpoints stay in bounds
            minx, maxx = min(p1[0], p2[0]), max(p1[0], p2[0])
            miny, maxy = min(p1[1], p2[1]), max(p1[1], p2[1])
            dx = _clampf(dx, -minx, W - maxx)
            dy = _clampf(dy, -miny, H - maxy)
            self._p1 = [p1[0] + dx, p1[1] + dy]
            self._p2 = [p2[0] + dx, p2[1] + dy]
        self._update_artists()
        self._blit()

    def _on_release(self, event) -> None:
        if self._canvas is not None:
            self._canvas._overlay_consuming_event = False
        if self._drag is not None:
            self._drag = None
            self._set_animated(False)
            self._blit_bg = None
            if self._canvas is not None:
                self._canvas.draw_idle()

    # ── blitting ──────────────────────────────────────────────────────────

    def _set_animated(self, val: bool) -> None:
        for a in (self._line, self._dots, self._label):
            if a is not None:
                a.set_animated(val)

    def _blit(self) -> None:
        if self._blit_bg is None:
            self._canvas.draw_idle()
            return
        self._canvas.restore_region(self._blit_bg)
        for a in (self._line, self._dots, self._label):
            if a is not None:
                self._ax.draw_artist(a)
        self._canvas.blit(self._ax.bbox)


# ---------------------------------------------------------------------------
# PointOverlay — interactive scatter points (select / drag / delete)
# ---------------------------------------------------------------------------

_PICK_RADIUS_PX = 12  # screen-space hit radius for point picking


class PointOverlay(QObject):
    """Interactive points overlay.

    * Left-click a point → selects it (highlighted colour + larger marker)
    * Left-click empty area → deselects
    * Drag a selected point → moves it, clamped to image bounds (blitted)
    * Right-click empty area → adds a new point (when ``add_on_right_click=True``)
    * Delete / Backspace → removes the selected point

    Parameters
    ----------
    color : str
        Default point colour.
    selected_color : str
        Colour when a point is selected.
    marker : str
        Matplotlib marker style.
    size : float
        Marker size in points (selected markers are drawn at ``size * 1.4``).
    label_prefix : str
        If non-empty, each point gets an annotation ``label_prefix + (index+1)``.
    add_on_right_click : bool
        If True (default), right-clicking adds a new point.
    removable : bool
        If True (default), Delete/Backspace removes the selected point.
    modal : bool
        If True, the overlay handles input *only* while it is the canvas's active
        overlay (e.g. spot burn — inert in Move mode). If False (default), it also
        responds when no overlay is active (always-on, backward-compatible).
    """

    point_added = pyqtSignal(int, float, float)  # index, x, y
    point_selected = pyqtSignal(int, float, float)  # index, x, y
    point_dragging = pyqtSignal(int, float, float)  # index, x, y  (each motion step)
    point_moved = pyqtSignal(int, float, float)  # index, x, y  (on release)
    point_removed = pyqtSignal(int)  # index (before removal)

    def __init__(
        self,
        color: str = "cyan",
        selected_color: str = "yellow",
        marker: str = "o",
        size: float = 10.0,
        label_prefix: str = "",
        add_on_right_click: bool = True,
        removable: bool = True,
        modal: bool = False,
        edge_width: Optional[float] = None,
        legend_label: Optional[str] = None,
        numbered: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self._color = color
        self._selected_color = selected_color
        self._marker = marker
        self._size = size
        self._label_prefix = label_prefix
        self._add_on_right_click = add_on_right_click
        self._removable = removable
        self._modal = modal
        self._edge_width = edge_width  # override the default marker edge width if set
        self._legend_label = legend_label  # opt-in patch legend for this overlay
        self._legend = None
        self._numbered = numbered  # annotate each point with its 1-based index
        self._visible = True  # toggled by set_visible (points kept, artists hidden)

        self._ax = None
        self._canvas: Optional[FibsemImageCanvas] = None
        self._img_w: Optional[int] = None
        self._img_h: Optional[int] = None

        self._points: List[List[float]] = []  # [[x, y], ...]  mutable for drag
        self._artists: List = []  # Line2D per point (index-aligned)
        self._anns: List = []  # Annotation per point (or None)
        # Optional per-point overrides (index-aligned), else the global style is used
        self._point_colors: Optional[List[str]] = None
        self._point_labels: Optional[List[str]] = None

        self._selected: Optional[int] = None
        self._drag_idx: Optional[int] = None
        self._drag_offset: Tuple[float, float] = (0.0, 0.0)
        self._drag_start_xy: Tuple[float, float] = (0.0, 0.0)
        self._blit_bg = None

        self._cids: List[int] = []

    # ── overlay protocol ──────────────────────────────────────────────────

    def attach(self, ax, canvas: "FibsemImageCanvas") -> None:
        self._ax = ax
        self._canvas = canvas
        self._cids = [
            canvas.mpl_connect("button_press_event", self._on_press),
            canvas.mpl_connect("motion_notify_event", self._on_motion),
            canvas.mpl_connect("button_release_event", self._on_release),
            canvas.mpl_connect("key_press_event", self._on_key),
        ]

    def detach(self) -> None:
        if self._canvas is not None:
            for cid in self._cids:
                try:
                    self._canvas.mpl_disconnect(cid)
                except Exception:
                    pass
        self._cids = []
        self._remove_all_artists()
        self._ax = None
        self._canvas = None

    def on_image_changed(self, width: int, height: int) -> None:
        self._img_w, self._img_h = width, height
        self._remove_all_artists()
        if width > 0 and self._ax is not None:
            self._draw_all()

    # ── public API ────────────────────────────────────────────────────────

    def set_points(
        self,
        points: List[Tuple[float, float]],
        colors: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
    ) -> None:
        """Replace all points.

        ``colors`` / ``labels``, when given, are index-aligned per-point overrides
        (e.g. one colour + name per detection feature); otherwise the global
        ``color`` / ``label_prefix`` style is used.
        """
        self._points = [[float(x), float(y)] for x, y in points]
        self._point_colors = list(colors) if colors is not None else None
        self._point_labels = list(labels) if labels is not None else None
        self._selected = None
        self._remove_all_artists()
        if self._ax is not None and self._img_w:
            self._draw_all()
        if self._canvas is not None:
            self._canvas.draw_idle()

    def add_point(self, x: float, y: float) -> int:
        """Append a point and return its index."""
        idx = len(self._points)
        self._points.append([float(x), float(y)])
        if self._ax is not None:
            self._append_artist(idx)
        if self._canvas is not None:
            self._canvas.draw_idle()
        return idx

    def remove_point(self, index: int) -> None:
        """Remove the point at *index*."""
        if index < 0 or index >= len(self._points):
            return
        self.point_removed.emit(index)
        for lst in (self._artists, self._anns):
            a = lst.pop(index)
            if a is not None:
                try:
                    a.remove()
                except Exception:
                    pass
        self._points.pop(index)
        if self._selected == index:
            self._selected = None
        elif self._selected is not None and self._selected > index:
            self._selected -= 1
        if self._label_prefix or self._numbered:
            self._refresh_ann_text()
        if self._canvas is not None:
            self._canvas.draw_idle()

    def clear_points(self) -> None:
        self._selected = None
        self._remove_all_artists()
        self._points.clear()
        if self._canvas is not None:
            self._canvas.draw_idle()

    def get_points(self) -> List[Tuple[float, float]]:
        return [(p[0], p[1]) for p in self._points]

    def set_visible(self, visible: bool) -> None:
        """Show or hide all markers/labels without discarding the points.

        State is remembered and re-applied across image rebuilds (a hidden
        overlay stays hidden when a new image arrives).
        """
        self._visible = visible
        for a in self._artists + self._anns:
            if a is not None:
                a.set_visible(visible)
        self._draw_legend()  # add/remove the legend to match visibility
        if self._canvas is not None:
            self._canvas.draw_idle()

    def set_selected(self, index: Optional[int]) -> None:
        """Programmatically select a point (e.g. from a synced table).

        Silent — does not emit ``point_selected`` — so it will not loop back onto a
        producer that is driving the selection. Pass ``None`` (or an out-of-range
        index) to clear the selection.
        """
        n = len(self._points)
        idx = index if (index is not None and 0 <= index < n) else None
        if idx == self._selected:
            return
        prev = self._selected
        self._selected = idx
        if prev is not None:
            self._update_artist_appearance(prev)
        if idx is not None:
            self._update_artist_appearance(idx)
        if self._canvas is not None:
            self._canvas.draw_idle()

    # ── private: artists ──────────────────────────────────────────────────

    def _input_allowed(self) -> bool:
        """Whether this overlay may handle input now (modal-aware).

        Modal overlays respond only while they are the canvas's active overlay;
        non-modal overlays also respond when nothing is active (default).
        """
        if self._canvas is None:
            return True
        if self._modal:
            return self._canvas._active_overlay is self
        return self._canvas._overlay_input_allowed(self)

    def _remove_all_artists(self):
        for lst in (self._artists, self._anns):
            for a in lst:
                if a is not None:
                    try:
                        a.remove()
                    except Exception:
                        pass
            lst.clear()
        self._remove_legend()

    def _draw_all(self):
        for idx in range(len(self._points)):
            self._append_artist(idx)
        self._draw_legend()

    def _remove_legend(self) -> None:
        if self._legend is not None:
            try:
                self._legend.remove()
            except Exception:
                pass
            self._legend = None

    def _draw_legend(self) -> None:
        """Opt-in patch legend (top-left), styled like the milling-stage legend."""
        self._remove_legend()
        if (
            not self._legend_label
            or self._ax is None
            or not self._points
            or not self._visible
        ):
            return
        import matplotlib.patches as mpatches
        from matplotlib.legend import Legend

        handle = mpatches.Patch(
            facecolor=self._color, edgecolor="white", label=self._legend_label
        )
        # build the Legend directly (not ax.legend()) so it doesn't replace another
        # overlay's primary legend (e.g. the milling stages, top-right)
        leg = Legend(
            self._ax,
            [handle],
            [self._legend_label],
            loc="upper left",
            fontsize=8,
            facecolor="#1e2124",
            edgecolor="#555555",
            labelcolor="#d1d2d4",
            framealpha=0.85,
        )
        leg.set_zorder(10)
        self._ax.add_artist(leg)
        self._legend = leg

    def _marker_edge(self, color: str, selected: bool):
        """Edge colour/width for the marker. Unfilled markers (+, x, ...) are drawn
        in their edge colour, so they take the point colour and a thicker line;
        filled markers (o, s, ...) keep a thin white outline for contrast.

        ``edge_width`` (if set) overrides the normal-state width; the selected state
        adds a fixed bump, so backward-compatible defaults are preserved when unset.
        """
        from matplotlib.lines import Line2D
        if self._marker in Line2D.filled_markers:
            base = self._edge_width if self._edge_width is not None else 0.8
            return "white", (base + 1.2 if selected else base)
        base = self._edge_width if self._edge_width is not None else 2.0
        return color, (base + 0.8 if selected else base)

    def _point_color(self, idx: int, selected: bool) -> str:
        """Per-point colour override if set, else the global selected/normal colour.
        (Per-point points keep their own colour even when selected — size + edge
        convey the selection instead.)"""
        if self._point_colors is not None and idx < len(self._point_colors):
            return self._point_colors[idx]
        return self._selected_color if selected else self._color

    def _point_label(self, idx: int) -> Optional[str]:
        """Per-point label override if set, else ``label_prefix + (idx+1)``, else the
        bare 1-based index when ``numbered``, else None."""
        if self._point_labels is not None and idx < len(self._point_labels):
            return self._point_labels[idx]
        if self._label_prefix:
            return f"{self._label_prefix}{idx + 1}"
        if self._numbered:
            return str(idx + 1)
        return None

    def _append_artist(self, idx: int):
        if self._ax is None:
            return
        x, y = self._points[idx]
        selected = idx == self._selected
        color = self._point_color(idx, selected)
        ms = self._size * 1.4 if selected else self._size
        edge_color, mew = self._marker_edge(color, selected)
        (line,) = self._ax.plot(
            x,
            y,
            marker=self._marker,
            markersize=ms,
            color=color,
            markeredgecolor=edge_color,
            markeredgewidth=mew,
            linestyle="none",
            zorder=8,
            animated=False,
            visible=self._visible,
        )
        self._artists.append(line)
        ann = None
        label = self._point_label(idx)
        if label is not None:
            ann = self._ax.annotate(
                label,
                xy=(x, y),
                xytext=(6, 4),
                textcoords="offset points",
                color=color,
                fontsize=8,
                zorder=9,
                animated=False,
                visible=self._visible,
            )
        self._anns.append(ann)

    def _update_artist_appearance(self, idx: int):
        if idx >= len(self._artists):
            return
        selected = idx == self._selected
        color = self._point_color(idx, selected)
        ms = self._size * 1.4 if selected else self._size
        edge_color, mew = self._marker_edge(color, selected)
        line = self._artists[idx]
        line.set_color(color)
        line.set_markersize(ms)
        line.set_markeredgecolor(edge_color)
        line.set_markeredgewidth(mew)
        ann = self._anns[idx] if idx < len(self._anns) else None
        if ann is not None:
            ann.set_color(color)

    def _update_artist_position(self, idx: int):
        if idx >= len(self._artists):
            return
        x, y = self._points[idx]
        self._artists[idx].set_xdata([x])
        self._artists[idx].set_ydata([y])
        ann = self._anns[idx] if idx < len(self._anns) else None
        if ann is not None:
            ann.xy = (x, y)

    def _refresh_ann_text(self):
        for idx, ann in enumerate(self._anns):
            if ann is not None:
                label = self._point_label(idx)
                if label is not None:
                    ann.set_text(label)

    # ── hit testing ───────────────────────────────────────────────────────

    def _hit_point(self, event) -> Optional[int]:
        if not self._points or self._ax is None:
            return None
        trans = self._ax.transData
        best_idx, best_dist = None, _PICK_RADIUS_PX
        for i, (px, py) in enumerate(self._points):
            sx, sy = trans.transform((px, py))
            d = ((event.x - sx) ** 2 + (event.y - sy) ** 2) ** 0.5
            if d < best_dist:
                best_dist, best_idx = d, i
        return best_idx

    # ── blit helpers ──────────────────────────────────────────────────────

    def _start_drag(self, idx: int, event):
        if self._canvas is None or self._ax is None:
            return
        self._drag_idx = idx
        px, py = self._points[idx]
        self._drag_offset = (event.xdata - px, event.ydata - py)
        self._drag_start_xy = (px, py)  # so a no-move select-click skips point_moved
        self._canvas._overlay_consuming_event = True
        self._artists[idx].set_animated(True)
        ann = self._anns[idx] if idx < len(self._anns) else None
        if ann is not None:
            ann.set_animated(True)
        self._canvas.draw()
        self._blit_bg = self._canvas.copy_from_bbox(self._ax.bbox)

    def _blit(self):
        if self._canvas is None or self._ax is None:
            return
        if self._blit_bg is None or self._drag_idx is None:
            self._canvas.draw_idle()
            return
        self._canvas.restore_region(self._blit_bg)
        self._ax.draw_artist(self._artists[self._drag_idx])
        ann = self._anns[self._drag_idx] if self._drag_idx < len(self._anns) else None
        if ann is not None:
            self._ax.draw_artist(ann)
        self._canvas.blit(self._ax.bbox)

    # ── mouse / key events ────────────────────────────────────────────────

    def _on_press(self, event):
        if self._canvas is None or self._ax is None:
            return
        if not self._input_allowed():  # another overlay owns input (modal-aware)
            return
        if event.inaxes is not self._ax or event.xdata is None or event.dblclick:
            return
        if self._canvas._overlay_consuming_event:
            return

        if event.button == 3:  # right-click → add a new point
            if not self._add_on_right_click:
                return
            x = max(0.0, min(event.xdata, (self._img_w or 1) - 1))
            y = max(0.0, min(event.ydata, (self._img_h or 1) - 1))
            idx = self.add_point(x, y)
            old_sel = self._selected
            self._selected = idx
            if old_sel is not None:
                self._update_artist_appearance(old_sel)
            self._update_artist_appearance(idx)
            self.point_added.emit(idx, x, y)
            self._canvas.draw_idle()
            return

        if event.button != 1:
            return

        hit = self._hit_point(event)
        if hit is not None:
            old_sel = self._selected
            self._selected = hit
            if old_sel is not None and old_sel != hit:
                self._update_artist_appearance(old_sel)
            self._update_artist_appearance(hit)
            self.point_selected.emit(hit, self._points[hit][0], self._points[hit][1])
            self._start_drag(hit, event)
        elif self._selected is not None:
            # left-click empty → deselect
            old_sel = self._selected
            self._selected = None
            self._update_artist_appearance(old_sel)
            self._canvas.draw_idle()

    def _on_motion(self, event):
        if self._drag_idx is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        W = self._img_w or 1
        H = self._img_h or 1
        x = max(0.0, min(event.xdata - self._drag_offset[0], W - 1))
        y = max(0.0, min(event.ydata - self._drag_offset[1], H - 1))
        self._points[self._drag_idx] = [x, y]
        self._update_artist_position(self._drag_idx)
        self.point_dragging.emit(self._drag_idx, x, y)
        self._blit()

    def _on_release(self, event):
        if self._canvas is None:
            return
        self._canvas._overlay_consuming_event = False
        if self._drag_idx is not None:
            idx = self._drag_idx
            self._drag_idx = None
            self._blit_bg = None
            self._artists[idx].set_animated(False)
            ann = self._anns[idx] if idx < len(self._anns) else None
            if ann is not None:
                ann.set_animated(False)
            # Only a real move emits point_moved (a select-click without a drag
            # leaves the position unchanged; point_selected already covered it).
            if tuple(self._points[idx]) != self._drag_start_xy:
                self.point_moved.emit(idx, self._points[idx][0], self._points[idx][1])
            self._canvas.draw_idle()

    def _on_key(self, event):
        if not self._input_allowed():  # another overlay owns input (modal-aware)
            return
        if not self._removable:
            return
        if event.key in ("delete", "backspace") and self._selected is not None:
            self.remove_point(self._selected)


# ---------------------------------------------------------------------------
# PatternOverlay — milling shape patches (pixel-space coordinates)
# ---------------------------------------------------------------------------


class PatternOverlay(CanvasOverlay):
    """Renders milling pattern shapes as matplotlib patches.

    Coordinates must be in image pixel space (caller handles unit conversion).
    Supported pattern attributes:
      - Rectangle / Bitmap : ``centre_x, centre_y, width, height``
      - Circle             : ``centre_x, centre_y, radius``
      - Line               : ``start_x, start_y, end_x, end_y``
      - Polygon            : ``vertices`` (N×2 array)
    """

    def __init__(self, patterns=(), color: str = "cyan", alpha: float = 0.4):
        self._patterns = list(patterns)
        self._color = color
        self._alpha = alpha
        self._ax = None
        self._canvas = None
        self._artists: list = []

    def attach(self, ax, canvas: FibsemImageCanvas) -> None:
        self._ax = ax
        self._canvas = canvas

    def detach(self) -> None:
        self._remove_artists()
        self._ax = None
        self._canvas = None

    def on_image_changed(self, width: int, height: int) -> None:
        self._remove_artists()
        if width > 0:
            self._draw()

    def set_patterns(self, patterns) -> None:
        self._patterns = list(patterns)
        self._remove_artists()
        self._draw()
        if self._canvas is not None:
            self._canvas.draw_idle()

    # ── private ──

    def _remove_artists(self):
        for a in self._artists:
            try:
                a.remove()
            except Exception:
                pass
        self._artists.clear()

    def _draw(self):
        if self._ax is None:
            return
        import matplotlib.lines as mlines
        import matplotlib.patches as mpatches

        kw = dict(
            edgecolor=self._color,
            facecolor=self._color,
            alpha=self._alpha,
            linewidth=1.5,
            zorder=6,
        )
        for pat in self._patterns:
            try:
                artist = _pattern_to_artist(pat, kw, mpatches, mlines)
                if artist is not None:
                    self._ax.add_artist(artist)
                    self._artists.append(artist)
            except Exception:
                _logger.exception("PatternOverlay: failed to render %r", pat)


def _pattern_to_artist(pat, kw, mpatches, mlines):
    name = type(pat).__name__
    if any(k in name for k in ("Rectangle", "Bitmap")):
        return mpatches.Rectangle(
            (pat.centre_x - pat.width / 2, pat.centre_y - pat.height / 2),
            pat.width,
            pat.height,
            **kw,
        )
    if "Circle" in name:
        return mpatches.Circle(
            (pat.centre_x, pat.centre_y),
            pat.radius,
            **{**kw, "facecolor": "none"},
        )
    if "Line" in name:
        return mlines.Line2D(
            [pat.start_x, pat.end_x],
            [pat.start_y, pat.end_y],
            color=kw["edgecolor"],
            linewidth=2,
            alpha=kw["alpha"],
            zorder=kw["zorder"],
        )
    if "Polygon" in name and hasattr(pat, "vertices"):
        return mpatches.Polygon(pat.vertices, closed=True, **kw)
    return None


class ScanDirectionArrowOverlay(CanvasOverlay):
    """Draws a yellow arrow indicating the milling scan direction.

    Only ``"TopToBottom"`` and ``"BottomToTop"`` are supported; all other
    values result in no arrow being shown.

    Call :meth:`set_arrow` with the pattern bounding box in pixel coordinates
    to update the arrow, or :meth:`clear` to hide it.
    """

    _SUPPORTED = {"TopToBottom", "BottomToTop"}

    def __init__(self, color: str = "yellow"):
        self._color = color
        self._ax = None
        self._canvas = None
        self._artist = None
        # (x_start, y_start, x_end, y_end) in pixel coords
        self._params: Optional[tuple] = None

    def attach(self, ax, canvas: "FibsemImageCanvas") -> None:
        self._ax = ax
        self._canvas = canvas

    def detach(self) -> None:
        self._remove_artist()
        self._ax = None
        self._canvas = None

    def on_image_changed(self, width: int, _height: int) -> None:
        self._remove_artist()
        if width > 0 and self._params is not None:
            self._draw()

    def set_arrow(self, cx: float, cy: float, h_px: float, scan_direction: str) -> None:
        """Position the arrow based on pattern centre and height.

        Args:
            cx: pattern centre x in pixel coords
            cy: pattern centre y in pixel coords
            h_px: pattern height in pixels
            scan_direction: ``"TopToBottom"`` or ``"BottomToTop"``
        """
        if scan_direction not in self._SUPPORTED:
            self.clear()
            return
        margin = h_px * 0.15
        if scan_direction == "TopToBottom":
            # Arrow from near top edge → near bottom edge (↓ in image coords)
            x_s, y_s = cx, cy - h_px / 2 + margin
            x_e, y_e = cx, cy + h_px / 2 - margin
        else:  # BottomToTop
            # Arrow from near bottom edge → near top edge (↑ in image coords)
            x_s, y_s = cx, cy + h_px / 2 - margin
            x_e, y_e = cx, cy - h_px / 2 + margin
        self._params = (x_s, y_s, x_e, y_e)
        self._remove_artist()
        self._draw()
        if self._canvas is not None:
            self._canvas.draw_idle()

    def clear(self) -> None:
        """Hide the arrow."""
        self._params = None
        self._remove_artist()
        if self._canvas is not None:
            self._canvas.draw_idle()

    # ── private ──

    def _remove_artist(self):
        if self._artist is not None:
            try:
                self._artist.remove()
            except Exception:
                pass
            self._artist = None

    def _draw(self):
        if self._ax is None or self._params is None:
            return
        x_s, y_s, x_e, y_e = self._params
        self._artist = self._ax.annotate(
            "",
            xy=(x_e, y_e),
            xytext=(x_s, y_s),
            arrowprops=dict(
                arrowstyle="-|>",
                color=self._color,
                lw=2.0,
                mutation_scale=18,
            ),
            zorder=10,
        )
