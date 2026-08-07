"""Dedicated multi-channel fluorescence canvas for the quad-view FM panel.

``FMCanvasWidget`` wraps a generic :class:`FibsemImageCanvas` and adds the
FM-specific bits the bare canvas should not carry: a per-channel layer model, the
additive colour composite (:mod:`fm_composite`), and a toolbar **layers** popover
(:class:`FMLayersPanel`) for per-channel visibility / colormap / opacity / gamma /
contrast — the napari layer-controls equivalent, on matplotlib.

Display only (Phase 6a); FM canvas interactions (position select, relative move)
are Phase 6b.
"""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PyQt5.QtCore import Qt, QPoint, QSize, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton, QSlider, QToolButton,
    QVBoxLayout, QWidget,
)
from superqt import QRangeSlider

from fibsem.ui.icon import fibsem_icon
from fibsem.ui.tokens import (
    ACCENT_COLOR,
    BORDER_COLOR,
    DISABLED_TEXT_COLOR,
    GRAY_FOREGROUND_COLOR,
    GRAY_HIGHLIGHT_COLOR,
    GRAY_PRIMARY_COLOR,
    PANEL_COLOR,
    PRIMARY_ACCENT,
    ROW_ALT_COLOR,
    SURFACE_COLOR,
    TEXT_COLOR,
    TEXT_MUTED_COLOR,
    TEXT_STRONG_COLOR,
)
from fibsem.ui.widgets.canvas.fm_composite import (
    AVAILABLE_COLORS, FMLayer, auto_clim, composite_fm_layers, to_rgba,
)
from fibsem.ui.widgets.canvas.canvas_base import FibsemCanvasBase, _downsample
from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas
from fibsem.ui.widgets.canvas.real_space_canvas import FibsemRealSpaceCanvas

if TYPE_CHECKING:
    from fibsem.fm.structures import FluorescenceImage

_ACCENT = ACCENT_COLOR

# The checked Auto pill needs a dark tinted-blue *surface*, which the palette has
# no token for. Deriving it as a low-alpha wash of the accent means a retune of
# ACCENT_COLOR carries the pill with it; a hand-picked hex would silently fall
# out of step.
_ACCENT_WASH = "rgba({}, {}, {}, 0.16)".format(
    *(int(ACCENT_COLOR[i:i + 2], 16) for i in (1, 3, 5))
)

# napari-style dark theme for the floating layers panel. An f-string, so every
# literal QSS brace is doubled -- Qt discards a malformed rule silently, so a
# missed pair shows up as an unstyled widget rather than an error.
_PANEL_QSS = f"""
QFrame#fmPanel {{ background: {SURFACE_COLOR}; border: 1px solid {BORDER_COLOR}; border-radius: 10px; }}
QLabel {{ color: {TEXT_STRONG_COLOR}; background: transparent; }}
#panelTitle {{ color: {TEXT_MUTED_COLOR}; font-size: 12px; font-weight: 500; letter-spacing: 1px; }}
QFrame#divider {{ background: {BORDER_COLOR}; border: none; }}
#selName {{ color: {TEXT_STRONG_COLOR}; font-size: 13px; font-weight: 500; }}
#selTag {{ color: {TEXT_MUTED_COLOR}; font-size: 11px; }}
#ctrlLbl {{ color: {TEXT_MUTED_COLOR}; font-size: 12px; }}
#valLbl {{ color: {TEXT_COLOR}; font-size: 12px; }}
#valSm {{ color: {GRAY_HIGHLIGHT_COLOR}; font-size: 11px; }}
QFrame#channelRow {{ border-radius: 7px; background: transparent; }}
QFrame#channelRow:hover {{ background: {ROW_ALT_COLOR}; }}
QFrame#channelRow[selected="true"] {{ background: {BORDER_COLOR}; }}
QToolButton#eyeBtn {{ border: none; background: transparent; padding: 0; }}
#chName {{ color: {TEXT_STRONG_COLOR}; font-size: 13px; }}
QComboBox {{ background: {PANEL_COLOR}; color: {TEXT_STRONG_COLOR}; border: 1px solid {BORDER_COLOR};
            border-radius: 6px; padding: 4px 8px; font-size: 12px; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{ background: {PANEL_COLOR}; color: {TEXT_STRONG_COLOR};
            border: 1px solid {BORDER_COLOR}; selection-background-color: {BORDER_COLOR}; outline: none; }}
QPushButton#autoPill {{ color: {ACCENT_COLOR}; background: {_ACCENT_WASH}; border: 1px solid {PRIMARY_ACCENT};
            border-radius: 11px; padding: 3px 11px; font-size: 11px; }}
QPushButton#autoPill:!checked {{ color: {TEXT_MUTED_COLOR}; background: transparent; border: 1px solid {BORDER_COLOR}; }}
QPushButton#resetBtn {{ background: transparent; color: {TEXT_COLOR}; border: 1px solid {BORDER_COLOR};
            border-radius: 6px; padding: 8px; font-size: 12px; }}
QPushButton#resetBtn:hover {{ background: {BORDER_COLOR}; }}
QSlider {{ background: transparent; }}
QSlider::groove:horizontal {{ height: 4px; background: {BORDER_COLOR}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT_COLOR}; border-radius: 2px; }}
QSlider::add-page:horizontal {{ background: {BORDER_COLOR}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 13px; height: 13px; margin: -5px 0; border-radius: 7px;
            background: {TEXT_STRONG_COLOR}; border: 1px solid {ACCENT_COLOR}; }}
QSlider::handle:horizontal:disabled {{ background: {GRAY_PRIMARY_COLOR}; border-color: {GRAY_PRIMARY_COLOR}; }}
QSlider::sub-page:horizontal:disabled {{ background: {GRAY_FOREGROUND_COLOR}; }}
"""


def _color_icon(color: str, size: int = 14) -> QIcon:
    px = QPixmap(size, size)
    px.fill(QColor("white" if color == "gray" else color))
    return QIcon(px)


def _chip_css(color: str) -> str:
    return "background: %s; border-radius: 3px;" % ("white" if color == "gray" else color)


class _ContrastSlider(QRangeSlider):
    """Dual-handle range slider for per-channel contrast limits (min + max on one
    track). Paints its own handles so they stay visible (mirrors the correlation
    widget's ``_ClipSlider``), and dims when disabled (auto contrast)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._style.brush_active = _ACCENT
        self._style.brush_inactive = BORDER_COLOR

    def _draw_handle(self, painter, opt) -> None:
        on = self.isEnabled()
        self._style.brush_active = _ACCENT if on else GRAY_FOREGROUND_COLOR
        if self._should_draw_bar:
            self._drawBar(painter, opt)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(TEXT_STRONG_COLOR if on else TEXT_MUTED_COLOR))
        for i in range(len(self.value())):
            rect = self._handleRect(i).adjusted(1, 1, -1, -1)
            painter.drawEllipse(rect)
        painter.restore()


class _ChannelRow(QFrame):
    """One channel: eye visibility toggle + colour chip + name, click to select."""

    selected = pyqtSignal(int)
    visibility_changed = pyqtSignal(int, bool)

    def __init__(self, index: int, layer: FMLayer, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._index = index
        self.setObjectName("channelRow")
        self.setProperty("selected", False)
        self.setCursor(Qt.PointingHandCursor)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 7, 10, 7)
        lay.setSpacing(11)

        self.eye = QToolButton()
        self.eye.setObjectName("eyeBtn")
        self.eye.setCheckable(True)
        self.eye.setChecked(layer.visible)
        self.eye.setCursor(Qt.PointingHandCursor)
        self.eye.setIconSize(QSize(16, 16))
        self._refresh_eye(layer.visible)
        self.eye.toggled.connect(self._on_eye)

        self.chip = QLabel()
        self.chip.setFixedSize(13, 13)
        self.chip.setStyleSheet(_chip_css(layer.color))

        self.name = QLabel(layer.name)
        self.name.setObjectName("chName")

        lay.addWidget(self.eye)
        lay.addWidget(self.chip)
        lay.addWidget(self.name, 1)
        self._dim(not layer.visible)

    def _refresh_eye(self, visible: bool) -> None:
        icon = "mdi:eye" if visible else "mdi:eye-off-outline"
        self.eye.setIcon(
            fibsem_icon(icon, color=TEXT_COLOR if visible else DISABLED_TEXT_COLOR)
        )

    def _dim(self, dim: bool) -> None:
        self.name.setStyleSheet(
            f"color: {DISABLED_TEXT_COLOR};" if dim else f"color: {TEXT_STRONG_COLOR};"
        )

    def _on_eye(self, checked: bool) -> None:
        self._refresh_eye(checked)
        self._dim(not checked)
        self.visibility_changed.emit(self._index, checked)

    def set_color(self, color: str) -> None:
        self.chip.setStyleSheet(_chip_css(color))

    def set_selected(self, sel: bool) -> None:
        self.setProperty("selected", sel)
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, event) -> None:
        self.selected.emit(self._index)
        super().mousePressEvent(event)


class FMCanvasWidget(QWidget):
    """FibsemImageCanvas + per-channel FM layer model + composite + layers popover.

    The channel model, the layers popover and the z-scrubbing are independent of *how*
    the blended frame is shown, so a subclass can swap the canvas (:meth:`_make_canvas`)
    and how the composite reaches it (:meth:`_show_composite`) while keeping all of it —
    see :class:`FMRealSpaceCanvasWidget`.
    """

    def _make_canvas(self) -> FibsemCanvasBase:
        """The canvas this widget composites onto."""
        return FibsemImageCanvas()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.canvas = self._make_canvas()
        self._layers: List[FMLayer] = []
        self._pixel_size: Optional[float] = None
        self._canvas_shape: Optional[Tuple[int, int]] = None  # drawn-axes shape
        self._shape: Optional[Tuple[int, int]] = None  # composite target shape (last upserted layer)
        # z-stack scrubbing: keep the full ZYX stack per channel + display state
        self._stacks: Dict[str, np.ndarray] = {}   # channel name -> ZYX stack
        self._mip_clim: Dict[str, Tuple[float, float]] = {}  # fixed clim while scrubbing
        self._z_index: int = 0
        self._z_max: int = 0                        # nz - 1
        self._max_projection: bool = True           # default: MIP (behaviour-preserving)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)

        # z-slider (shown only for a multi-plane stack with max-projection off)
        self._z_row = QWidget()
        zrl = QHBoxLayout(self._z_row)
        zrl.setContentsMargins(6, 2, 6, 4)
        zrl.setSpacing(6)
        zrl.addWidget(QLabel("z"))
        self._z_slider = QSlider(Qt.Horizontal)
        self._z_slider.setMinimum(0)
        self._z_slider.valueChanged.connect(self._on_z_changed)
        zrl.addWidget(self._z_slider, 1)
        self._z_label = QLabel("1/1")
        zrl.addWidget(self._z_label)
        self._z_row.setVisible(False)
        lay.addWidget(self._z_row)

        self._btn_layers = self.canvas.add_toolbar_button(
            "mdi:layers", "FM channels", self._toggle_layers_panel, checkable=True
        )
        # max-projection toggle (checked = MIP; uncheck to scrub the z-slider)
        self._btn_mip = self.canvas.add_toolbar_button(
            "mdi:arrow-collapse-vertical", "Max projection", self._on_mip_button, checkable=True
        )
        self._btn_mip.setChecked(True)
        self._btn_mip.setVisible(False)  # shown only once a multi-plane z-stack is loaded
        # FM contrast/gamma is per-channel (layers popover); the canvas's built-in
        # grayscale contrast button does nothing on the RGB composite — hide it.
        self.canvas.btn_contrast.hide()
        self.canvas._reposition_overlay_buttons()

        self._panel = FMLayersPanel(self)
        self._panel.changed.connect(self._recomposite)
        self._panel.hide()

    # ── public API ────────────────────────────────────────────────────────

    def _upsert_layer(self, name: str, data: np.ndarray, color: Optional[str]) -> FMLayer:
        """Create or update a channel's layer (2-D data + colour); display props preserved.

        Rebuilds the panel list only when a channel is added — a data-only update (live
        acquisition / z-scrub) must NOT reset the per-channel controls."""
        layer = next((l for l in self._layers if l.name == name), None)
        is_new = layer is None
        if is_new:
            layer = FMLayer(name=name, color=color or "gray")
            self._layers.append(layer)
        layer.data = np.asarray(data)
        if color is not None:
            layer.color = color
        self._shape = layer.data.shape[:2]
        if is_new:
            self._panel.set_layers(self._layers)
        return layer

    def retain_channels(self, names: Sequence[str]) -> None:
        """Drop the layers for channels no longer being displayed.

        `_upsert_layer` only ever adds, which is right for the live path: channels
        arrive one frame at a time and each is a partial update. But a caller showing a
        whole *image* names every channel it has, and one that is absent has to go --
        otherwise its layer stays behind still holding the previous image's pixels, and
        those get blended into the new one. Acquiring a two-channel overview and then a
        one-channel overview drew the second with the first's second channel, which was
        then recorded as the second's own, so every later contrast change faithfully
        redrew the ghost.
        """
        keep = set(names)
        if all(layer.name in keep for layer in self._layers):
            return

        survivors = []
        for layer in self._layers:
            if layer.name in keep:
                survivors.append(layer)
                continue
            self._stacks.pop(layer.name, None)
            self._mip_clim.pop(layer.name, None)
            if self._channel_still_shown(layer.name):
                # Something else on the canvas still has this channel, so its control
                # has to stay for that image to be styled through. Only the pixels go.
                layer.data = None
                layer._clim_cache = None
                survivors.append(layer)

        self._layers = survivors
        self._panel.set_layers(self._layers)

    def _channel_still_shown(self, name: str) -> bool:
        """Whether a channel absent from the current image is displayed anywhere else.

        False here: this canvas shows one image at a time, so a channel it does not
        have is a channel nothing has. Overridden where several images share the
        controls.
        """
        return False

    def set_channel(self, name: str, data: np.ndarray, color: Optional[str] = None) -> None:
        """Upsert a channel's 2-D image (live path — no z-stack); display props preserved."""
        had_stack = self._stacks.pop(name, None) is not None
        self._mip_clim.pop(name, None)
        layer = self._upsert_layer(name, data, color)
        if had_stack:  # leaving z-scrub for a live 2-D frame → auto-contrast again
            layer.autocontrast = True
            layer.clim = None
            self._refresh_z_controls()  # a stack went away → maybe hide the z-controls
        self._recomposite()

    def set_fm_image(self, image: "FluorescenceImage") -> None:
        """Composite every channel of an acquired ``FluorescenceImage`` (CZYX), keeping the
        z-stacks so the z-slider / max-projection toggle can scrub them.

        Pixel size and per-channel name/colour come from the image metadata; channels
        upsert by name and blend additively (parity with the napari main tab). Defaults to
        the max-intensity projection.
        """
        md = image.metadata
        self.set_pixel_size(getattr(md, "pixel_size_x", None))
        data = np.asarray(image.data)
        self._stacks = {}
        self._mip_clim = {}
        self._z_index = 0
        for ci, channel in enumerate(md.channels):
            if data.ndim >= 4:        # CZYX
                stack = np.asarray(data[ci])
            elif data.ndim == 3:      # ZYX (single channel)
                stack = np.asarray(data)
            else:                      # 2-D single plane
                stack = np.asarray(data)[None]
            self._stacks[channel.name] = stack
            self._upsert_layer(channel.name, stack.max(axis=0), channel.color)
        # After the upserts, so the panel is rebuilt once and from the final set.
        self.retain_channels([channel.name for channel in md.channels])
        self._refresh_z_controls()
        self._apply_z_mode()

    def set_layers(self, layers: List[FMLayer]) -> None:
        self._layers = list(layers)
        for l in self._layers:
            if l.data is not None:
                self._shape = l.data.shape[:2]
        self._panel.set_layers(self._layers)
        self._recomposite()

    def set_pixel_size(self, pixel_size: Optional[float]) -> None:
        self._pixel_size = pixel_size
        if pixel_size and self._canvas_shape is not None:
            self.canvas.pixel_size = pixel_size  # rescales the scalebar + redraws

    def clear(self) -> None:
        self._layers = []
        self._canvas_shape = None
        self._stacks = {}
        self._mip_clim = {}
        self._z_index = 0
        self._refresh_z_controls()  # hides the z-controls (no stacks)
        self._panel.set_layers(self._layers)
        self.canvas.clear()

    @property
    def layers(self) -> List[FMLayer]:
        return self._layers

    # ── display ───────────────────────────────────────────────────────────

    def _composite_inputs(self) -> Tuple[List[FMLayer], Optional[Tuple[int, int]]]:
        """The layers to blend and the shape to blend them at.

        A subclass that shows the composite smaller than it is can substitute reduced
        planes here, so the blend costs what is displayed rather than what was acquired.
        """
        return self._layers, self._shape

    def _recomposite(self) -> None:
        layers, shape = self._composite_inputs()
        rgb = composite_fm_layers(layers, shape)
        if rgb is None:
            return
        h, w = rgb.shape[:2]
        reshaped = self._canvas_shape != (h, w)
        self._canvas_shape = (h, w)
        self._show_composite(rgb, reshaped)

    def _show_composite(self, rgb: np.ndarray, reshaped: bool) -> None:
        """Put the blended RGB frame on the canvas.

        The only place compositing touches the canvas, and so the seam a subclass
        overrides to display the same frame differently — placed in stage space rather
        than filling the view. *reshaped* is True when the frame's pixel dimensions
        changed, which is the expensive path; the steady state swaps pixels only.
        """
        if reshaped:
            # (re)build axes + scalebar at this shape and show the composite
            # directly — no fake single-channel FibsemImage needed
            self.canvas.set_array(rgb, pixel_size=self._pixel_size)
        else:
            self.canvas.update_display(rgb)  # steady state: swap pixels only

    # ── z-stack scrubbing ──────────────────────────────────────────────────

    def _refresh_z_controls(self) -> None:
        """Recompute the z-range from the current stacks and show/hide the z-controls.

        The Max-projection toggle + z-slider appear only when a multi-plane stack is
        present — so a single-frame (live microscope) FM canvas shows neither."""
        nz = max((s.shape[0] for s in self._stacks.values()), default=1)
        self._z_max = max(0, nz - 1)
        self._z_index = min(self._z_index, self._z_max)
        self._z_slider.blockSignals(True)
        self._z_slider.setMaximum(self._z_max)
        self._z_slider.setValue(self._z_index)
        self._z_slider.blockSignals(False)
        self._btn_mip.setVisible(self._z_max > 0)
        self.canvas._reposition_overlay_buttons()
        self._update_z_visibility()

    def _update_z_visibility(self) -> None:
        """The z-slider shows only for a multi-plane stack with max-projection off."""
        show = self._z_max > 0 and not self._max_projection
        self._z_row.setVisible(show)
        if show:
            self._z_label.setText(f"{self._z_index + 1}/{self._z_max + 1}")

    def _plane_for(self, stack: np.ndarray) -> np.ndarray:
        """The 2-D plane to display for a ZYX *stack* under the current mode."""
        if self._max_projection or stack.shape[0] <= 1:
            return stack.max(axis=0)
        return stack[min(self._z_index, stack.shape[0] - 1)]

    def _apply_z_mode(self) -> None:
        """Set each stacked channel's plane + contrast for the current mode, then composite.

        MIP → auto-contrast per the projection; scrub → a fixed clim (computed once off the
        MIP) held across planes, so brightness doesn't jump. Live 2-D channels (no stack)
        are left untouched."""
        for layer in self._layers:
            stack = self._stacks.get(layer.name)
            if stack is None:
                continue
            if self._max_projection or stack.shape[0] <= 1:
                # single plane (incl. every live frame) or an explicit MIP: default to
                # auto-contrast UNLESS the user chose manual. Keying off layer.manual (an
                # explicit user choice) rather than layer.autocontrast (which the z-scrub
                # branch below also toggles) is what lets a MIP toggle restore auto while a
                # live feed — every frame lands here — keeps the user's manual contrast.
                if not layer.manual:
                    layer.autocontrast = True
                    layer.clim = None
            else:
                clim = self._mip_clim.get(layer.name)
                if clim is None:
                    clim = auto_clim(stack.max(axis=0))
                    self._mip_clim[layer.name] = clim
                layer.autocontrast = False
                layer.clim = clim
            layer.data = self._plane_for(stack)
        self._recomposite()

    def _on_z_changed(self, value: int) -> None:
        self._z_index = value
        self._z_label.setText(f"{value + 1}/{self._z_max + 1}")
        self._apply_z_mode()

    def _on_max_projection_toggled(self, on: bool) -> None:
        self._max_projection = bool(on)
        self._update_z_visibility()
        self._apply_z_mode()

    def _on_mip_button(self) -> None:
        """Toolbar toggle → max-projection on/off (checked = MIP)."""
        self._on_max_projection_toggled(self._btn_mip.isChecked())

    # ── layers popover ────────────────────────────────────────────────────

    def _toggle_layers_panel(self) -> None:
        if self._btn_layers.isChecked():
            self._panel.set_layers(self._layers)
            self._position_panel()
            self._panel.show()
            self._panel.raise_()
        else:
            self._panel.hide()

    def _position_panel(self) -> None:
        self._panel.adjustSize()
        # top-level window → anchor near the canvas top-right in global coordinates
        anchor = self.canvas.mapToGlobal(QPoint(self.canvas.width() - 8, 44))
        self._panel.move(anchor.x() - self._panel.width(), anchor.y())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._panel.isVisible():
            self._position_panel()


class FMRealSpaceCanvasWidget(FMCanvasWidget):
    """FM channel model and layer controls, composited onto a real-space canvas.

    Identical to :class:`FMCanvasWidget` above the display: the same channel stacks, the
    same layers popover, the same z-scrubbing. The difference is where the blended frame
    lands — placed at the stage position it was acquired at, on a black stage-space
    backdrop, rather than filling the view.

    That placement is what lets overlays work in stage coordinates: a planned tile grid
    can be drawn before anything is acquired, and position markers do not depend on there
    being an image to reproject onto.

    Layers are shared across everything placed, which is right for an overview — a mosaic
    should render uniformly, and per-image contrast would emphasise the seams rather than
    hide them. See FIB-415 for the per-image case.

    **The canvas is not FM-only.** It knows nothing about modality: ``add_image`` takes
    any 2-D or RGB array with a position and a pixel size, so a caller can place FIB or
    SEM images alongside the composite and they will register with it in stage space::

        widget.canvas.add_image(sem_data, centre=(x, y), pixel_size=ps, zorder=-1)

    This widget only manages the one key it composites into (:attr:`COMPOSITE_KEY`), so
    anything else placed is left alone by a layer change. What is FM-specific here is the
    *controls* — channels, blending, the layers popover — not the display.
    """

    COMPOSITE_KEY = "fm-composite"

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._placement: Tuple[float, float] = (0.0, 0.0)
        self._composite_key: str = self.COMPOSITE_KEY
        # Channel planes of everything placed, so a layer change can re-render images
        # composited long ago. Only the newest is in `_layers`; the rest are here.
        self._held: Dict[str, Dict[str, np.ndarray]] = {}

    def _make_canvas(self) -> FibsemCanvasBase:
        return FibsemRealSpaceCanvas()

    def _channel_still_shown(self, name: str) -> bool:
        """True while any placed image has this channel.

        Several images share one set of controls here, so a channel dropped from the
        newest is not dropped from the canvas -- an earlier overview may well have it.
        Removing the control would leave that overview with a channel nothing can
        style, and `_restyle_others` would quietly stop drawing it: acquire a
        one-channel overview and the two-channel one beside it loses a colour.
        """
        return any(name in planes for planes in self._held.values())

    # ── placement ─────────────────────────────────────────────────────────

    def set_composite_key(self, key: str) -> None:
        """Choose which placed image subsequent channel data composites into.

        Re-using a key replaces that image; a new key adds one. So a second overview
        acquired somewhere else joins the first on the canvas rather than replacing it,
        which is the reason for placing things in stage space at all.
        """
        self._composite_key = key

    def placed_overviews(self) -> List[str]:
        """Keys of everything this widget has composited, oldest first."""
        return list(self._held)

    def forget_overview(self, key: str) -> bool:
        """Drop the channel planes held for one image. False if none were held.

        The counterpart to the canvas's `remove_image`, and it has to be called with it:
        the planes are what `_restyle_others` re-renders from, so keeping them for an
        image no longer placed leaks the pixels and leaves `_channel_still_shown`
        answering for a channel nothing displays.
        """
        return self._held.pop(key, None) is not None

    def clear_overviews(self) -> None:
        """Drop every composited image, leaving overlays and the channel model alone."""
        for key in list(self._held):
            self.canvas.remove_image(key)
        self._held.clear()

    def set_placement(self, centre: Tuple[float, float]) -> None:
        """Where the composite sits, in metres from the canvas origin.

        Set before the frame it applies to. The caller projects a stage position into
        the display plane (see :mod:`fibsem.fm.reprojection`); this widget only carries
        the result through to the canvas.
        """
        self._placement = (float(centre[0]), float(centre[1]))

    def set_world_extent(self, width: Optional[float], height: Optional[float] = None,
                         centre: Tuple[float, float] = (0.0, 0.0),
                         refit: bool = True) -> None:
        """Declare the working area the canvas represents — see the canvas method."""
        self.canvas.set_world_extent(width, height, centre, refit)

    # ── display ───────────────────────────────────────────────────────────

    def _show_composite(self, rgb: np.ndarray, reshaped: bool) -> None:
        """Place the composite, rather than letting it fill the view.

        A reshaped frame has to be re-placed, since its extent depends on its pixel
        dimensions; otherwise the pixels are swapped in place, which keeps the artist and
        so its position and draw order.
        """
        if not self._pixel_size:
            return  # nothing to scale against; the caller has not said how big a pixel is
        key = self._composite_key
        # The composite is blended at display resolution, so it has fewer pixels than
        # the data it represents while covering the same ground. Placement is by pixel
        # size, so the reduction has to be spent there -- otherwise a mosaic reduced 10x
        # is placed a tenth of its size, which is what happened when the blend first
        # moved to display resolution and the pixel size did not follow.
        # The composite is blended at display resolution, so it holds fewer pixels than
        # the data it represents while covering the same ground. Say so explicitly
        # rather than letting the canvas infer the ground from shape x pixel size --
        # inferring placed a mosaic reduced 10x at a tenth of its size.
        covers = None
        if self._shape:
            covers = (self._shape[1] * self._pixel_size, self._shape[0] * self._pixel_size)
        # Keep this image's planes, so a later layer change can re-render it once its
        # channels are no longer the ones in `_layers`.
        self._held[key] = {
            layer.name: layer.data for layer in self._layers if layer.data is not None
        }
        # Placed over other images rather than over a background, so it goes on as
        # colour plus coverage: an opaque frame hides whatever it covers, including
        # where it holds nothing itself. That is what painted a sparse overview's
        # unacquired tiles black over the full one beneath it (FIB-519).
        placed = to_rgba(rgb)
        if reshaped or key not in self.canvas.placed_keys:
            self.canvas.add_image(
                placed,
                centre=self._placement,
                pixel_size=self._pixel_size,
                key=key,
                covers=covers,
                # Ordered by the *acquired* pixel size: how much detail an image holds
                # is a property of the data, not of the blend.
                zorder=self._detail_zorder(self._pixel_size),
            )
        else:
            self.canvas.update_image(key, placed)
        self._restyle_others(key)

    def _reduce(self, plane: np.ndarray) -> np.ndarray:
        """A plane at the resolution the canvas will actually store.

        `_downsample` strides, so this is a view rather than a copy — the reduction
        itself costs nothing, and the blend that follows costs what is displayed.
        """
        return _downsample(np.asarray(plane), self.canvas._display_max_px)

    def _composite_inputs(self) -> Tuple[List[FMLayer], Optional[Tuple[int, int]]]:
        """Blend at display resolution, not acquisition resolution.

        The canvas decimates whatever it is handed, so blending a full mosaic first
        computes ~99% of a result that is then thrown away — 153 ms per layer change for
        a 1x5 overview, against 2.6 ms reduced, and it is the layer sliders that fire it
        continuously. Blending the reduced planes is the same picture for the cost of the
        pixels that survive.
        """
        reduced = [
            replace(layer, data=self._reduce(layer.data), _clim_cache=None)
            if layer.data is not None else layer
            for layer in self._layers
        ]
        first = next((layer.data for layer in reduced if layer.data is not None), None)
        return reduced, (first.shape[:2] if first is not None else self._shape)

    @staticmethod
    def _detail_zorder(pixel_size: float) -> float:
        """Draw finer images over coarser ones, whatever order they arrived in.

        Arrival order is the wrong answer here. A wide, coarse overview acquired *after*
        a detailed one covers the same ground, so by default it would hide the better
        data underneath it — and the more detail an image has, the more it deserves to
        be on top. Ordering by pixel size instead means a coarse survey always sits
        behind whatever was taken at higher resolution over it.
        """
        return -pixel_size * 1e9  # nanometres/px, negated: smaller pixels draw on top

    def _restyle_others(self, current: str) -> None:
        """Re-render everything except *current* under the present layer settings.

        Layers are display state shared by everything placed; the pixels are not. So a
        colour or contrast change has to be applied to each held image's own planes,
        which is why they are kept. Without this, changing a channel's colour would
        recolour the newest overview and leave the others as they were — the same data
        drawn two different ways, side by side.
        """
        for key, planes in self._held.items():
            if key == current or key not in self.canvas.placed_keys:
                continue
            layers = [
                replace(layer, data=self._reduce(planes[layer.name]), _clim_cache=None)
                for layer in self._layers
                if layer.name in planes
            ]
            if not layers:
                continue
            shape = layers[0].data.shape[:2]
            rgb = composite_fm_layers(layers, shape)
            if rgb is not None:
                # Same transform as the image that triggered this: these are placed
                # over each other too, and re-rendering one opaque would put back the
                # occlusion `_show_composite` just avoided.
                self.canvas.update_image(key, to_rgba(rgb))

    def set_pixel_size(self, pixel_size: Optional[float]) -> None:
        """Record the scale without touching the canvas's own.

        The base assigns `canvas.pixel_size` to drive the scalebar, but here the canvas
        takes its scale from the placed image instead — assigning it directly would set a
        scalebar that disagrees with what is drawn.
        """
        self._pixel_size = pixel_size


class FMLayersPanel(QFrame):
    """Floating per-channel controls — a dark channel list (eye toggle + colour chip
    + name) over a detail panel (colormap / opacity / gamma / contrast) for the
    selected channel. Emits :attr:`changed` whenever an edit needs a re-composite."""

    changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        # Float as a separate top-level tool window: the panel overlays the
        # matplotlib canvas, and as a child widget its native sliders were forced
        # to repaint (and flicker) on every canvas redraw during a slider drag.
        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setObjectName("fmPanel")
        self.setStyleSheet(_PANEL_QSS)
        self.setFixedWidth(268)

        self._layers: List[FMLayer] = []
        self._rows: List[_ChannelRow] = []
        self._selected: int = 0
        self._updating = False  # guard so programmatic updates don't emit changed
        self._data_lo: float = 0.0
        self._data_span: float = 1.0

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 16)
        root.setSpacing(0)

        # header
        header = QHBoxLayout(); header.setSpacing(8)
        hicon = QLabel()
        hicon.setPixmap(
            fibsem_icon("mdi:layers-triple-outline", color=TEXT_MUTED_COLOR).pixmap(QSize(16, 16))
        )
        title = QLabel("FM CHANNELS"); title.setObjectName("panelTitle")
        header.addWidget(hicon); header.addWidget(title); header.addStretch()
        root.addLayout(header)
        root.addSpacing(12)

        # channel list
        self._list_box = QVBoxLayout(); self._list_box.setSpacing(2)
        root.addLayout(self._list_box)
        root.addSpacing(12)

        div = QFrame(); div.setObjectName("divider"); div.setFixedHeight(1)
        root.addWidget(div)
        root.addSpacing(12)

        # detail header (selected channel)
        dh = QHBoxLayout(); dh.setSpacing(8)
        self.sel_chip = QLabel(); self.sel_chip.setFixedSize(11, 11)
        self.sel_name = QLabel("—"); self.sel_name.setObjectName("selName")
        sel_tag = QLabel("selected"); sel_tag.setObjectName("selTag")
        dh.addWidget(self.sel_chip); dh.addWidget(self.sel_name); dh.addStretch(); dh.addWidget(sel_tag)
        root.addLayout(dh)
        root.addSpacing(14)

        # colormap
        cm_row = QHBoxLayout()
        cm_lbl = QLabel("Colormap"); cm_lbl.setObjectName("ctrlLbl")
        self.colormap = QComboBox(); self.colormap.setFixedWidth(132)
        for c in AVAILABLE_COLORS:
            self.colormap.addItem(_color_icon(c), c)
        self.colormap.currentTextChanged.connect(self._on_colormap)
        cm_row.addWidget(cm_lbl); cm_row.addStretch(); cm_row.addWidget(self.colormap)
        root.addLayout(cm_row)
        root.addSpacing(13)

        # opacity + gamma sliders (label + value, then track)
        self.opacity, self.opacity_val = self._slider_row(root, "Opacity", 0, 100, 100)
        self.opacity.valueChanged.connect(self._on_opacity)
        self.gamma, self.gamma_val = self._slider_row(root, "Gamma", 10, 300, 100)
        self.gamma.valueChanged.connect(self._on_gamma)

        # contrast header (label + Auto pill)
        ch = QHBoxLayout()
        ct_lbl = QLabel("Contrast"); ct_lbl.setObjectName("ctrlLbl")
        self.autocontrast_cb = QPushButton("Auto"); self.autocontrast_cb.setObjectName("autoPill")
        self.autocontrast_cb.setCheckable(True); self.autocontrast_cb.setChecked(True)
        self.autocontrast_cb.setCursor(Qt.PointingHandCursor)
        self.autocontrast_cb.toggled.connect(self._on_autocontrast)
        ch.addWidget(ct_lbl); ch.addStretch(); ch.addWidget(self.autocontrast_cb)
        root.addLayout(ch)
        root.addSpacing(8)

        self.contrast = _ContrastSlider(Qt.Horizontal)
        self.contrast.setRange(0, 1000); self.contrast.setValue((0, 1000))
        self.contrast.valueChanged.connect(self._on_contrast)
        root.addWidget(self.contrast)
        root.addSpacing(8)

        cv = QHBoxLayout()
        self.cmin_val = QLabel("0"); self.cmin_val.setObjectName("valSm")
        self.cmax_val = QLabel("0"); self.cmax_val.setObjectName("valSm")
        cv.addWidget(self.cmin_val); cv.addStretch(); cv.addWidget(self.cmax_val)
        root.addLayout(cv)
        root.addSpacing(14)

        self.btn_reset = QPushButton("Reset adjustments"); self.btn_reset.setObjectName("resetBtn")
        self.btn_reset.setCursor(Qt.PointingHandCursor)
        self.btn_reset.clicked.connect(self._on_reset)
        root.addWidget(self.btn_reset)

    def _slider_row(self, root, label: str, lo: int, hi: int, val: int):
        head = QHBoxLayout()
        lbl = QLabel(label); lbl.setObjectName("ctrlLbl")
        valw = QLabel(); valw.setObjectName("valLbl")
        head.addWidget(lbl); head.addStretch(); head.addWidget(valw)
        root.addLayout(head)
        root.addSpacing(7)
        s = QSlider(Qt.Horizontal); s.setRange(lo, hi); s.setValue(val)
        root.addWidget(s)
        root.addSpacing(13)
        return s, valw

    # ── populate / select ─────────────────────────────────────────────────

    def set_layers(self, layers: List[FMLayer]) -> None:
        self._layers = layers
        self._updating = True
        prev = self._selected
        for row in self._rows:
            self._list_box.removeWidget(row)
            row.setParent(None)
            row.deleteLater()
        self._rows = []
        for i, layer in enumerate(layers):
            row = _ChannelRow(i, layer)
            row.selected.connect(self._select_row)
            row.visibility_changed.connect(self._on_visibility)
            self._list_box.addWidget(row)
            self._rows.append(row)
        self._selected = prev if 0 <= prev < len(layers) else 0
        self._updating = False
        self._refresh_selection()
        self._sync_detail()

    def _select_row(self, index: int) -> None:
        if index == self._selected or not (0 <= index < len(self._layers)):
            return
        self._selected = index
        self._refresh_selection()
        self._sync_detail()

    def _refresh_selection(self) -> None:
        for i, row in enumerate(self._rows):
            row.set_selected(i == self._selected)

    def _current(self) -> Optional[FMLayer]:
        return self._layers[self._selected] if 0 <= self._selected < len(self._layers) else None

    def _sync_detail(self) -> None:
        layer = self._current()
        prev_updating = self._updating  # save/restore so a caller's guard survives
        self._updating = True
        if layer is None:
            self.sel_name.setText("—")
            self.sel_chip.setStyleSheet("background: transparent;")
        else:
            self.sel_chip.setStyleSheet(_chip_css(layer.color))
            self.sel_name.setText(layer.name)
            self.colormap.setCurrentText(layer.color)
            self.opacity.setValue(int(layer.opacity * 100))
            self.opacity_val.setText("%d%%" % int(layer.opacity * 100))
            self.gamma.setValue(int(layer.gamma * 100))
            self.gamma_val.setText("%.2f" % layer.gamma)
            self.autocontrast_cb.setChecked(layer.autocontrast)
            self.contrast.setEnabled(not layer.autocontrast)  # manual edits only when off
            if layer.data is not None:
                d = np.asarray(layer.data, dtype=np.float32)
                lo_d, hi_d = float(d.min()), float(d.max())
                span = max(hi_d - lo_d, 1.0)
                if layer.autocontrast or layer.clim is None:
                    clo, chi = auto_clim(d)
                else:
                    clo, chi = layer.clim
                self._data_lo, self._data_span = lo_d, span
                self.contrast.setValue((
                    int((clo - lo_d) / span * 1000),
                    int((chi - lo_d) / span * 1000),
                ))
                self.cmin_val.setText("%d" % round(clo))
                self.cmax_val.setText("%d" % round(chi))
        self._updating = prev_updating

    # ── edits ─────────────────────────────────────────────────────────────

    def _on_visibility(self, index: int, visible: bool) -> None:
        if self._updating or not (0 <= index < len(self._layers)):
            return
        self._layers[index].visible = visible
        self.changed.emit()

    def _on_colormap(self, color: str) -> None:
        layer = self._current()
        if self._updating or layer is None or not color:
            return
        layer.color = color
        self.sel_chip.setStyleSheet(_chip_css(color))
        if 0 <= self._selected < len(self._rows):
            self._rows[self._selected].set_color(color)
        self.changed.emit()

    def _on_opacity(self, value: int) -> None:
        self.opacity_val.setText("%d%%" % value)
        layer = self._current()
        if self._updating or layer is None:
            return
        layer.opacity = value / 100.0
        self.changed.emit()

    def _on_gamma(self, value: int) -> None:
        self.gamma_val.setText("%.2f" % (value / 100.0))
        layer = self._current()
        if self._updating or layer is None:
            return
        layer.gamma = value / 100.0
        self.changed.emit()

    def _on_reset(self) -> None:
        """Reset the selected channel's display adjustments (opacity / gamma /
        contrast → auto); colour + visibility are kept."""
        layer = self._current()
        if layer is None:
            return
        layer.opacity = 1.0
        layer.gamma = 1.0
        layer.autocontrast = True
        layer.clim = None
        self._sync_detail()
        self.changed.emit()

    def _on_autocontrast(self, checked: bool) -> None:
        layer = self._current()
        if self._updating or layer is None:
            return
        layer.autocontrast = checked
        layer.manual = not checked  # explicit user choice — survives live frames / MIP toggles
        if not checked and layer.clim is None and layer.data is not None:
            # seed manual limits from the current auto values so the image doesn't jump
            layer.clim = auto_clim(np.asarray(layer.data, dtype=np.float32))
        self._sync_detail()  # refresh slider value + enabled state
        self.changed.emit()

    def _on_contrast(self) -> None:
        layer = self._current()
        lo_n, hi_n = self.contrast.value()
        lo = self._data_lo + lo_n / 1000.0 * self._data_span
        hi = self._data_lo + hi_n / 1000.0 * self._data_span
        self.cmin_val.setText("%d" % round(lo))
        self.cmax_val.setText("%d" % round(hi))
        if self._updating or layer is None:
            return
        if hi > lo:
            layer.clim = (lo, hi)
            self.changed.emit()
