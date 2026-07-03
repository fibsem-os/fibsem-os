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

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import Qt, QPoint, QSize, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton, QSlider, QToolButton,
    QVBoxLayout, QWidget,
)
from superqt import QIconifyIcon, QRangeSlider

from fibsem.ui.widgets.canvas.fm_composite import (
    AVAILABLE_COLORS, FMLayer, auto_clim, composite_fm_layers,
)
from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas

if TYPE_CHECKING:
    from fibsem.fm.structures import FluorescenceImage

_PANEL_BG = "#2c3036"
_ACCENT = "#5b9bd5"

# napari-style dark theme for the floating layers panel
_PANEL_QSS = """
QFrame#fmPanel { background: #2c3036; border: 1px solid #3f444b; border-radius: 10px; }
QLabel { color: #e4e6e9; background: transparent; }
#panelTitle { color: #9aa0a6; font-size: 12px; font-weight: 500; letter-spacing: 1px; }
QFrame#divider { background: #3a3f46; border: none; }
#selName { color: #e4e6e9; font-size: 13px; font-weight: 500; }
#selTag { color: #80858b; font-size: 11px; }
#ctrlLbl { color: #9aa0a6; font-size: 12px; }
#valLbl { color: #d2d5d9; font-size: 12px; }
#valSm { color: #70757b; font-size: 11px; }
QFrame#channelRow { border-radius: 7px; background: transparent; }
QFrame#channelRow:hover { background: #32373e; }
QFrame#channelRow[selected="true"] { background: #363d46; }
QToolButton#eyeBtn { border: none; background: transparent; padding: 0; }
#chName { color: #e4e6e9; font-size: 13px; }
QComboBox { background: #23262b; color: #e4e6e9; border: 1px solid #3f444b;
            border-radius: 6px; padding: 4px 8px; font-size: 12px; }
QComboBox::drop-down { border: none; width: 18px; }
QComboBox QAbstractItemView { background: #23262b; color: #e4e6e9;
            border: 1px solid #3f444b; selection-background-color: #363d46; outline: none; }
QPushButton#autoPill { color: #9bcdf6; background: #243140; border: 1px solid #2e4a61;
            border-radius: 11px; padding: 3px 11px; font-size: 11px; }
QPushButton#autoPill:!checked { color: #9aa0a6; background: transparent; border: 1px solid #3f444b; }
QPushButton#resetBtn { background: transparent; color: #cdd0d4; border: 1px solid #3f444b;
            border-radius: 6px; padding: 8px; font-size: 12px; }
QPushButton#resetBtn:hover { background: #363d46; }
QSlider { background: transparent; }
QSlider::groove:horizontal { height: 4px; background: #3a3f46; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #5b9bd5; border-radius: 2px; }
QSlider::add-page:horizontal { background: #3a3f46; border-radius: 2px; }
QSlider::handle:horizontal { width: 13px; height: 13px; margin: -5px 0; border-radius: 7px;
            background: #eceef0; border: 1px solid #5b9bd5; }
QSlider::handle:horizontal:disabled { background: #5a6068; border-color: #5a6068; }
QSlider::sub-page:horizontal:disabled { background: #4a5058; }
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
        self._style.brush_inactive = "#3a3f46"

    def _draw_handle(self, painter, opt) -> None:
        on = self.isEnabled()
        self._style.brush_active = _ACCENT if on else "#46505d"
        if self._should_draw_bar:
            self._drawBar(painter, opt)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#eceef0" if on else "#8a8f95"))
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
        self.eye.setIcon(QIconifyIcon(icon, color="#d2d5d9" if visible else "#70757b"))

    def _dim(self, dim: bool) -> None:
        self.name.setStyleSheet("color: #70757b;" if dim else "color: #e4e6e9;")

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
    """FibsemImageCanvas + per-channel FM layer model + composite + layers popover."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.canvas = FibsemImageCanvas()
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
            self.canvas._pixel_size = pixel_size
            self.canvas._refresh_scalebar()
            self.canvas.draw_idle()

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

    def _recomposite(self) -> None:
        shape = self._shape
        rgb = composite_fm_layers(self._layers, shape)
        if rgb is None:
            return
        h, w = rgb.shape[:2]
        if self._canvas_shape != (h, w):
            # (re)build axes + scalebar at this shape and show the composite
            # directly — no fake single-channel FibsemImage needed
            self._canvas_shape = (h, w)
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
        hicon.setPixmap(QIconifyIcon("mdi:layers-triple-outline", color="#9aa0a6").pixmap(QSize(16, 16)))
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
