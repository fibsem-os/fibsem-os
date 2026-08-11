"""FibsemCanvasBase — the modality-agnostic half of the fibsem matplotlib canvas.

Everything here is independent of *what* is drawn: scroll-wheel zoom, left-drag pan,
the pluggable overlay registry and its input gating, the top-right toolbar, the chrome
(hint / title / info bar / LIVE badge / flash / legend), the scalebar and crosshair,
contrast + gamma, and view fitting.

Subclasses own the content and answer one question — :meth:`_content_extent`, the
rectangle the view fits to. :class:`~fibsem.ui.widgets.canvas.image_canvas.FibsemImageCanvas`
answers it with a single image at the origin.

Zoom: scroll wheel centred on cursor.
Pan: left-drag on empty canvas area.

Overlays implement a simple duck-typed protocol::

    class MyOverlay:
        def attach(self, ax, canvas: FibsemCanvasBase) -> None: ...
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
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import QSize, QTimer, Qt, pyqtSignal
from PyQt5.QtWidgets import QApplication, QPushButton, QSizePolicy

from fibsem.ui.icon import fibsem_icon
from fibsem.ui.stylesheets import CANVAS_BG as _BG, PRIMARY_ACCENT as _ACCENT
from fibsem.ui.widgets.canvas.contrast_gamma_control import ContrastGammaControl
from fibsem.ui.tokens import (
    GRAY_WHITE_COLOR,
    NEUTRAL_400,
    NEUTRAL_450,
    NEUTRAL_900,
    WHITE_ICON_COLOR,
)

if TYPE_CHECKING:
    from fibsem.ui.widgets.canvas.overlays.base import CanvasOverlay
    from fibsem.ui.widgets.canvas.overlays.ruler_overlay import RulerOverlay

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContentRect:
    """The region overlays clamp and anchor to, in canvas coordinates.

    For a canvas holding one image this is ``(0, 0, width, height)`` — the image's
    pixel grid, so ``x0``/``y0`` are zero and the arithmetic is the same as the
    ``(width, height)`` pair this replaced. For a canvas holding many images placed
    in stage space it is their union, whose origin is *not* the coordinate origin.
    Overlays should therefore anchor with :attr:`cx` / :attr:`cy` and clamp against
    :attr:`x0` / :attr:`x1` rather than assuming content starts at (0, 0).

    Distinct from the canvas's matplotlib extent (``_content_extent``), which carries
    imshow's half-pixel offset and exists only to fit the view.
    """

    x0: float
    y0: float
    width: float
    height: float

    @property
    def x1(self) -> float:
        return self.x0 + self.width

    @property
    def y1(self) -> float:
        return self.y0 + self.height

    @property
    def cx(self) -> float:
        """Horizontal centre — where an overlay with no better anchor should sit."""
        return self.x0 + self.width / 2.0

    @property
    def cy(self) -> float:
        return self.y0 + self.height / 2.0

    @property
    def is_empty(self) -> bool:
        return self.width <= 0 or self.height <= 0


EMPTY_CONTENT = ContentRect(0.0, 0.0, 0.0, 0.0)

_LIVE_BADGE_BG = "#2e7d32"  # dark green behind the white "● LIVE" badge
_MAX_DISPLAY_PX = 2048
_ZOOM_FACTOR = 1.15
# How far the view may travel from the content, as a multiple of its longest side.
# Out: enough to see a mosaic in the context around it, not enough to lose it. In:
# enough to inspect single pixels, not enough to land between them.
MAX_ZOOM_OUT = 20.0
MAX_ZOOM_IN = 200.0
_REDRAW_INTERVAL = 32  # ms (~60 fps)

_OVERLAY_BTN_STYLE = (
    "QPushButton { background: rgba(40,41,48,180); border: 1px solid #555;"
    " border-radius: 3px; padding: 0px; }"
    "QPushButton:hover { background: rgba(74,74,74,200); }"
    "QPushButton:pressed { background: rgba(30,30,30,220); }"
    f"QPushButton:checked {{ background: rgba(90,92,100,200); border-color: {GRAY_WHITE_COLOR}; }}"
)
_OVERLAY_ICON_SIZE = QSize(14, 14)
_OVERLAY_BTN_SIZE = 22
_OVERLAY_MARGIN = 4
_OVERLAY_GAP = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
# FibsemCanvasBase
# ---------------------------------------------------------------------------


class FibsemCanvasBase(FigureCanvasQTAgg):
    """Content-agnostic matplotlib canvas: navigation, overlays, toolbar and chrome.

    * Scroll-wheel zoom centred on cursor
    * Left-drag pan on empty area
    * Pluggable overlay objects via add_overlay() / remove_overlay()
    * Optional scalebar, crosshair, contrast/gamma and text chrome

    Subclass and override :meth:`_content_extent` to supply the drawn content.
    """

    # Trailing ``object`` is a tuple of napari-style modifier strings, e.g. ("Alt",).
    canvas_clicked = pyqtSignal(float, float, object)  # left single-click (x, y) px, mods
    canvas_double_clicked = pyqtSignal(float, float, object)  # left double-click (x, y) px, mods
    canvas_right_clicked = pyqtSignal(float, float, object)  # right single-click (x, y) px, mods
    canvas_scrolled = pyqtSignal(float, float, int, object)  # (x, y) px, dir +1/-1, mods
    # Where the cursor is, in canvas coordinates, or (None, None) once it leaves the
    # axes -- so a readout can blank rather than freeze on the last point it saw.
    # Typed `object` for exactly that: pyqtSignal(float, float) cannot carry None.
    cursor_moved = pyqtSignal(object, object)

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
        self._title_artist = None  # top-centre image caption (e.g. FM z-slice)
        self._title_text: Optional[str] = None  # remembered so it survives set_image
        self._info_artist = None  # bottom-left microscope-state info bar
        self._info_text: Optional[str] = None  # remembered so it survives set_image
        self._live_artist = None  # top-right "LIVE" badge during live acquisition
        self._live_on: bool = False  # remembered so it survives set_image

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
        self._ruler_overlay: Optional["RulerOverlay"] = None

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
        # A motion event with `inaxes` unset covers most exits, but a fast flick off
        # the widget can skip one and leave a readout showing a stale point.
        self.mpl_connect("figure_leave_event", lambda _: self.cursor_moved.emit(None, None))

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

    @property
    def pixel_size(self) -> Optional[float]:
        """Metres per pixel of the displayed image, or None if not known.

        Companion to :attr:`img_width` / :attr:`img_height` — together they let a
        caller convert canvas pixel coordinates to physical units. Normally set from
        the *pixel_size* argument to :meth:`set_array` / :meth:`update_display`.
        """
        return self._pixel_size

    @pixel_size.setter
    def pixel_size(self, value: Optional[float]) -> None:
        """Rescale the scalebar to a new metres/px, for a caller that owns the scale
        independently of the pixel data (e.g. the FM canvas, which composites layers
        and sets the scale separately from the RGB frame).

        Note the None semantics differ from the *pixel_size* argument of
        :meth:`set_array` / :meth:`update_display`, where None means "leave unchanged":
        assigning None here is an explicit "no longer known" and drops the scalebar.
        """
        self._pixel_size = value
        self._refresh_scalebar()
        self.draw_idle()

    # ── public API ────────────────────────────────────────────────────────

    def set_crosshair_visible(self, visible: bool) -> None:
        """Show or hide the yellow crosshair centred on the image.

        Syncs the toolbar button here rather than only in :meth:`toggle_crosshair`,
        so a host that overrides the default right after constructing its canvas —
        the correlation canvases turn the crosshair off — does not leave a checked
        button whose tooltip offers to hide something that is not drawn.
        """
        self._crosshair_visible = visible
        self._sync_toggle_button(self.btn_toggle_crosshair, visible, "crosshair")
        self._refresh_crosshair()
        self.draw_idle()

    def set_scalebar_visible(self, visible: bool) -> None:
        """Show or hide the scalebar. Mirrors :meth:`set_crosshair_visible`.

        Present because ``ImagePointCanvas`` has it and the correlation tab calls
        it; the shared canvas previously offered only the toggle, which cannot
        express "off" without knowing what it is currently set to.
        """
        self._scalebar_visible = visible
        self._sync_toggle_button(self.btn_toggle_scalebar, visible, "scalebar")
        self._refresh_scalebar()
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
                fontsize=8, color=NEUTRAL_900, zorder=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#e6e6e6",
                          edgecolor="none", alpha=0.85),
            )

    def set_title(self, text: Optional[str]) -> None:
        """Show a caption centred at the top of the image, or hide with None/''.

        For labelling what the frame *is* (e.g. an FM z-slice index), as opposed to
        the instruction hint (top-left) or the microscope-state info bar (bottom-left).
        Remembered + re-applied after each image change, like the hint.

        Deliberately drawn inside the axes rather than via ``Axes.set_title``: the
        figure is laid out edge-to-edge (``subplots_adjust(top=1)``), so a real axes
        title lands above the figure and is clipped whenever the pane is wider in
        aspect than the image — e.g. any square frame in a wide pane.

        Shares the top-centre slot with :meth:`flash_message`, which is drawn above it
        for the second or so it is on screen.
        """
        self._title_text = text or None
        self._refresh_title()
        self.draw_idle()

    def _refresh_title(self) -> None:
        """(Re)create the title artist from the cached text, or remove it."""
        if self._title_artist is not None:
            try:
                self._title_artist.remove()
            except Exception:
                pass
            self._title_artist = None
        if self._title_text:
            self._title_artist = self._ax.text(
                0.5, 0.985, self._title_text,
                transform=self._ax.transAxes, ha="center", va="top",
                fontsize=10, color=WHITE_ICON_COLOR, zorder=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=_BG,
                          edgecolor="none", alpha=0.55),
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

    def set_live_badge(self, on: bool) -> None:
        """Show/hide a green "● LIVE" badge in the top-right during live acquisition.

        Remembered + re-applied after each image change (like the info bar), so it stays put as
        live frames stream in."""
        self._live_on = bool(on)
        self._refresh_live_badge()
        self.draw_idle()

    def _refresh_live_badge(self) -> None:
        """(Re)create the LIVE badge artist, or remove it."""
        if self._live_artist is not None:
            try:
                self._live_artist.remove()
            except Exception:
                pass
            self._live_artist = None
        if self._live_on:
            self._live_artist = self._ax.text(
                0.988, 0.985, "● LIVE",
                transform=self._ax.transAxes, ha="right", va="top",
                fontsize=7, color=WHITE_ICON_COLOR, zorder=12, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=_LIVE_BADGE_BG,
                          edgecolor="none", alpha=0.9),
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
        """(Re)create the legend artist from the cached entries, or remove it.

        Each entry is ``(color, label)`` (a filled swatch) or ``(color, label, marker)``
        (a marker glyph, e.g. ``"+"`` for a crosshair / point of interest)."""
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
        from matplotlib.lines import Line2D

        labels, handles = [], []
        for entry in self._legend_entries:
            color, label = entry[0], entry[1]
            marker = entry[2] if len(entry) > 2 else None
            labels.append(label)
            if marker:
                handles.append(Line2D(
                    [], [], linestyle="None", marker=marker, markersize=9,
                    color=color, markeredgewidth=1.6, label=label,
                ))
            else:
                handles.append(mpatches.Patch(facecolor=color, edgecolor="white", label=label))
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
        self._title_artist = None
        self._title_text = None
        self._info_artist = None
        self._info_text = None
        self._live_artist = None
        self._live_on = False
        self._flash_artist = None
        self._flash_text = None
        self._flash_timer.stop()
        self._legend_artist = None  # removed by cla(); drop the cached entries too
        self._legend_entries = None
        self._plot_empty()
        self._notify_overlays(EMPTY_CONTENT)
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
        btn.setIcon(fibsem_icon(icon_name, color=NEUTRAL_450))
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

    def _content_rect(self) -> ContentRect:
        """The rectangle overlays clamp and anchor to.

        Defaults to the content size recorded by the subclass, anchored at the origin
        — correct for a single image. A canvas that places content away from the
        origin (e.g. images positioned in stage space) overrides this.
        """
        if self._img_w is None or self._img_h is None:
            return EMPTY_CONTENT
        return ContentRect(0.0, 0.0, self._img_w, self._img_h)

    def _notify_overlay(self, overlay, rect: ContentRect) -> None:
        """Tell one overlay the content rectangle changed.

        Prefers the rect-based ``on_content_changed``; falls back to the older
        ``on_image_changed(width, height)`` so a duck-typed overlay written against
        the previous protocol keeps working untouched.
        """
        handler = getattr(overlay, "on_content_changed", None)
        if handler is None:
            overlay.on_image_changed(rect.width, rect.height)
        else:
            handler(rect)

    def _notify_overlays(self, rect: ContentRect) -> None:
        """Broadcast a content change to every registered overlay."""
        for overlay in self._overlays:
            try:
                self._notify_overlay(overlay, rect)
            except Exception:
                _logger.exception("Overlay content update failed: %r", overlay)

    def _content_extent(self) -> Optional[Tuple[float, float, float, float]]:
        """Extent of the drawn content as ``(xmin, xmax, ymax, ymin)``, or None if empty.

        The seam between this base and its subclasses: the base fits the view to
        whatever rectangle this returns, without knowing whether the content is one
        image at the origin or many placed in stage space.
        """
        return None

    def _fit_extent(self) -> Optional[Tuple[float, float, float, float]]:
        """What "fit to view" frames, as ``(xmin, xmax, ymax, ymin)`` or None.

        Separate from :meth:`_content_extent` because the two answer different questions:
        that one is *the space the canvas represents* (what overlays clamp and anchor to),
        this one is *the thing worth looking at*. They coincide unless a canvas declares
        space it has not filled.
        """
        return self._content_extent()

    def _fit_view(self) -> None:
        """Set the view to the fit extent expanded by ``_view_margin`` on each side.

        Never while an overlay owns the gesture. Dragging something on the canvas is the
        user manipulating a thing in the scene, not asking for the scene to be re-framed
        — and a refit mid-drag moves the whole view out from under the pointer, which
        reads as the view snapping onto whatever is being dragged.

        Guarded here rather than only at the callers because a drag reaches the view
        through several of them — the overlay's own margin, a re-declared working area,
        any content change the drag provokes — and each one is a separate chance to get
        it wrong. The flag is the canvas's own record of who owns the gesture, cleared
        on release along with the pan it already suppresses.
        """
        if self._overlay_consuming_event:
            return
        extent = self._fit_extent()
        if extent is None:
            return
        xmin, xmax, ybot, ytop = extent  # (xmin, xmax, ymax, ymin)
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
                self._notify_overlay(overlay, self._content_rect())
            except Exception:
                _logger.exception("Overlay content update failed: %r", overlay)
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
        self.btn_mode.setIcon(fibsem_icon(icon, color=NEUTRAL_450))
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
            color=NEUTRAL_400,
        )

    def toggle_scalebar(self) -> None:
        """Flip the scalebar, from its toolbar button."""
        self.set_scalebar_visible(not self._scalebar_visible)

    def toggle_crosshair(self) -> None:
        """Flip the crosshair, from its toolbar button."""
        self.set_crosshair_visible(not self._crosshair_visible)

    def _sync_toggle_button(self, button, shown: bool, noun: str) -> None:
        """Keep a show/hide toolbar button in step with the state it reports."""
        button.setChecked(shown)
        button.setToolTip(f"Hide {noun}" if shown else f"Show {noun}")

    def toggle_ruler(self) -> None:
        """Toggle the drag-to-measure ruler (a generic canvas tool).

        While on, the ruler owns input (so a stray double-click/right-click
        doesn't move the stage or open the milling menu); when turned off, input
        returns to whatever the current overlay *mode* dictates.
        """
        if self.btn_toggle_ruler.isChecked():
            if self._ruler_overlay is None:
                from fibsem.ui.widgets.canvas.overlays.ruler_overlay import RulerOverlay

                self._ruler_overlay = RulerOverlay()
                self.add_overlay(self._ruler_overlay)
            self._ruler_overlay.set_visible(True)
            self.set_active_overlay(self._ruler_overlay)
            self.btn_toggle_ruler.setToolTip("Hide ruler")
        else:
            if self._ruler_overlay is not None:
                self._ruler_overlay.set_visible(False)
            # Derive the restore target from the live mode state rather than a snapshot
            # taken when the ruler was switched on: that snapshot goes stale if an overlay
            # is armed / exited / removed while measuring — restoring it could undo a new
            # arm, or re-activate a now-detached overlay and suppress input permanently.
            restore = self._mode_overlay if self.btn_mode.isChecked() else None
            self.set_active_overlay(restore)
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
        rect = self._content_rect()
        if not self._crosshair_visible or rect.is_empty:
            return
        # Centre of the content, wherever that is — for a single image at the origin
        # this is (w/2, h/2) as before; for content placed in stage space it follows.
        cx, cy = rect.cx, rect.cy
        # Size both arms from the longest dimension so the crosshair stays square
        # (axes use aspect="equal", so equal data-unit arms are equal on screen).
        half = max(rect.width, rect.height) * 0.05 / 2
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
        if self._content_rect().is_empty:
            # No content: axes span the default [0,1], so xdata/ydata are not canvas
            # coordinates. Suppress clicks/pan so a stray double-click can't drive a
            # stage move to (~0,0).
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
        # Before every early return below: a readout of where the cursor is should not
        # go quiet because a pan happens to be in progress or an overlay owns the drag.
        if event.inaxes is self._ax and event.xdata is not None:
            self.cursor_moved.emit(event.xdata, event.ydata)
        else:
            self.cursor_moved.emit(None, None)
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
        if not self._zoom_allowed(abs(xlim[1] - xlim[0]) * factor):
            return
        self._ax.set_xlim(cx + (xlim[0] - cx) * factor, cx + (xlim[1] - cx) * factor)
        self._ax.set_ylim(cy + (ylim[0] - cy) * factor, cy + (ylim[1] - cy) * factor)
        self._schedule_redraw()

    def _zoom_allowed(self, span: float) -> bool:
        """Whether the view may show *span* data units across.

        Bounded relative to the content, because unbounded scrolling ends up somewhere
        useless in one direction and degenerate in the other: far enough out and the
        content is a speck in an empty field with no way to tell which way to go back;
        far enough in and you are looking between two pixels. Both are easy to reach by
        accident on a trackpad, and neither is recoverable except through "reset view".
        """
        extent = self._content_extent()
        if extent is None:
            return True  # nothing to be relative to
        content = max(abs(extent[1] - extent[0]), abs(extent[2] - extent[3]))
        if content <= 0:
            return True
        return content / MAX_ZOOM_IN <= span <= content * MAX_ZOOM_OUT
