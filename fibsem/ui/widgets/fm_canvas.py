"""Dedicated multi-channel fluorescence canvas for the quad-view FM panel.

``FMCanvasWidget`` wraps a generic :class:`FibsemImageCanvas` and adds the
FM-specific bits the bare canvas should not carry: a per-channel layer model, the
additive colour composite (:mod:`fm_composite`), and a toolbar **layers** popover
(:class:`FMLayersPanel`) for per-channel visibility / colormap / opacity /
contrast — the napari layer-controls equivalent, on matplotlib.

Display only (Phase 6a); FM canvas interactions (position select, relative move)
are Phase 6b.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import Qt, QPoint, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QPushButton, QSlider, QVBoxLayout, QWidget,
)
from superqt import QRangeSlider

from fibsem.structures import FibsemImage
from fibsem.ui.widgets.fm_composite import (
    AVAILABLE_COLORS, FMLayer, auto_clim, composite_fm_layers,
)
from fibsem.ui.widgets.image_canvas import FibsemImageCanvas

_PANEL_BG = "#2b2f33"


def _color_icon(color: str, size: int = 14) -> QIcon:
    px = QPixmap(size, size)
    px.fill(QColor("white" if color == "gray" else color))
    return QIcon(px)


_HANDLE_COLOR = QColor("#c7c9cc")


class _ContrastSlider(QRangeSlider):
    """Dual-handle range slider for per-channel contrast limits (min + max on one
    track). Paints its own handles so they stay visible (mirrors the correlation
    widget's ``_ClipSlider``)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._style.brush_active = "#4a5168"
        self._style.brush_inactive = "#4a5168"

    def _draw_handle(self, painter, opt) -> None:
        if self._should_draw_bar:
            self._drawBar(painter, opt)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_HANDLE_COLOR)
        for i in range(len(self.value())):
            rect = self._handleRect(i).adjusted(1, 1, -1, -1)
            painter.drawEllipse(rect)
        painter.restore()


class FMCanvasWidget(QWidget):
    """FibsemImageCanvas + per-channel FM layer model + composite + layers popover."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.canvas = FibsemImageCanvas()
        self._layers: List[FMLayer] = []
        self._pixel_size: Optional[float] = None
        self._canvas_shape: Optional[Tuple[int, int]] = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)

        self._btn_layers = self.canvas.add_toolbar_button(
            "mdi:layers", "FM channels", self._toggle_layers_panel, checkable=True
        )
        # FM contrast/gamma is per-channel (layers popover); the canvas's built-in
        # grayscale contrast button does nothing on the RGB composite — hide it.
        self.canvas.btn_contrast.hide()
        self.canvas._reposition_overlay_buttons()

        self._panel = FMLayersPanel(self)
        self._panel.changed.connect(self._recomposite)
        self._panel.hide()

    # ── public API ────────────────────────────────────────────────────────

    def set_channel(self, name: str, data: np.ndarray, color: Optional[str] = None) -> None:
        """Upsert a channel's image (and colour); display props are preserved."""
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
            # rebuild the panel list only when a channel is added — a data-only
            # update (live acquisition) must NOT reset the per-channel controls.
            self._panel.set_layers(self._layers)
        self._recomposite()

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
        self._panel.set_layers(self._layers)
        self.canvas.clear()

    @property
    def layers(self) -> List[FMLayer]:
        return self._layers

    # ── display ───────────────────────────────────────────────────────────

    def _recomposite(self) -> None:
        shape = getattr(self, "_shape", None)
        rgb = composite_fm_layers(self._layers, shape)
        if rgb is None:
            return
        h, w = rgb.shape[:2]
        if self._canvas_shape != (h, w):
            # set up axes + scalebar at this shape with a 2D placeholder, then swap
            # in the RGB (a colour composite isn't a valid single-channel FibsemImage)
            self._canvas_shape = (h, w)
            self.canvas.set_image(FibsemImage(data=np.zeros((h, w), dtype=np.uint8)))
            if self._pixel_size:
                self.canvas._pixel_size = self._pixel_size
                self.canvas._refresh_scalebar()
        self.canvas.update_display(rgb)

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
    """Floating per-channel controls: a channel list (visibility) + a detail panel
    (colormap / opacity / contrast) for the selected channel. Emits :attr:`changed`
    whenever an edit should trigger a re-composite."""

    changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        # Float as a separate top-level tool window: the panel overlays the
        # matplotlib canvas, and as a child widget its native sliders were forced
        # to repaint (and flicker) on every canvas redraw during a slider drag.
        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._layers: List[FMLayer] = []
        self.setStyleSheet(
            "FMLayersPanel { background: %s; border: 1px solid #555; "
            "border-radius: 6px; } QLabel { color: #d1d2d4; }" % _PANEL_BG
        )
        self.setFixedWidth(248)
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
        root.addWidget(QLabel("FM Channels"))

        self.list = QListWidget()
        self.list.setMaximumHeight(110)
        self.list.currentRowChanged.connect(self._on_row_changed)
        self.list.itemChanged.connect(self._on_item_changed)
        root.addWidget(self.list)

        # detail controls for the selected channel
        self.colormap = QComboBox()
        for c in AVAILABLE_COLORS:
            self.colormap.addItem(_color_icon(c), c)
        self.colormap.currentTextChanged.connect(self._on_colormap)
        cm_row = QHBoxLayout(); cm_row.addWidget(QLabel("Colormap")); cm_row.addWidget(self.colormap, 1)
        root.addLayout(cm_row)

        self.opacity = QSlider(Qt.Horizontal)
        self.opacity.setRange(0, 100); self.opacity.setValue(100)
        self.opacity.valueChanged.connect(self._on_opacity)
        op_row = QHBoxLayout(); op_row.addWidget(QLabel("Opacity")); op_row.addWidget(self.opacity, 1)
        root.addLayout(op_row)

        self.gamma = QSlider(Qt.Horizontal)
        self.gamma.setRange(10, 300); self.gamma.setValue(100)  # gamma = value / 100 (0.1–3.0)
        self.gamma.valueChanged.connect(self._on_gamma)
        gm_row = QHBoxLayout(); gm_row.addWidget(QLabel("Gamma")); gm_row.addWidget(self.gamma, 1)
        root.addLayout(gm_row)

        ct_row = QHBoxLayout()
        ct_row.addWidget(QLabel("Contrast (min / max)"))
        ct_row.addStretch()
        self.autocontrast_cb = QCheckBox("Auto")
        self.autocontrast_cb.setChecked(True)
        self.autocontrast_cb.toggled.connect(self._on_autocontrast)
        ct_row.addWidget(self.autocontrast_cb)
        root.addLayout(ct_row)
        # NOTE: per-channel histogram shelved for now (sizing + doesn't track clim) — revisit.
        self.contrast = _ContrastSlider(Qt.Horizontal)  # one track, two handles
        self.contrast.setRange(0, 1000)
        self.contrast.setValue((0, 1000))
        self.contrast.valueChanged.connect(self._on_contrast)
        root.addWidget(self.contrast)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setToolTip("Reset this channel's opacity / gamma / contrast")
        self.btn_reset.clicked.connect(self._on_reset)
        root.addWidget(self.btn_reset)

        self._updating = False  # guard so programmatic updates don't emit changed

    # ── populate / select ─────────────────────────────────────────────────

    def set_layers(self, layers: List[FMLayer]) -> None:
        self._layers = layers
        self._updating = True
        prev = self.list.currentRow()  # preserve the selected channel across rebuilds
        self.list.clear()
        for layer in layers:
            item = QListWidgetItem(layer.name)
            item.setIcon(_color_icon(layer.color))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if layer.visible else Qt.Unchecked)
            self.list.addItem(item)
        if layers:
            self.list.setCurrentRow(prev if 0 <= prev < len(layers) else 0)
        self._updating = False
        self._sync_detail()

    def _current(self) -> Optional[FMLayer]:
        i = self.list.currentRow()
        return self._layers[i] if 0 <= i < len(self._layers) else None

    def _sync_detail(self) -> None:
        layer = self._current()
        prev_updating = self._updating  # save/restore so a caller's guard survives
        self._updating = True
        if layer is not None:
            self.colormap.setCurrentText(layer.color)
            self.opacity.setValue(int(layer.opacity * 100))
            self.gamma.setValue(int(layer.gamma * 100))
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
        self._updating = prev_updating

    # ── edits ─────────────────────────────────────────────────────────────

    def _on_row_changed(self, _row: int) -> None:
        if self._updating:  # ignore selection churn during a list rebuild
            return
        self._sync_detail()

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        if self._updating:
            return
        i = self.list.row(item)
        if 0 <= i < len(self._layers):
            self._layers[i].visible = item.checkState() == Qt.Checked
            self.changed.emit()

    def _on_colormap(self, color: str) -> None:
        layer = self._current()
        if self._updating or layer is None or not color:
            return
        layer.color = color
        item = self.list.currentItem()
        if item is not None:
            item.setIcon(_color_icon(color))
        self.changed.emit()

    def _on_opacity(self, value: int) -> None:
        layer = self._current()
        if self._updating or layer is None:
            return
        layer.opacity = value / 100.0
        self.changed.emit()

    def _on_gamma(self, value: int) -> None:
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
        if self._updating or layer is None or not hasattr(self, "_data_lo"):
            return
        lo_n, hi_n = self.contrast.value()
        lo = self._data_lo + lo_n / 1000.0 * self._data_span
        hi = self._data_lo + hi_n / 1000.0 * self._data_span
        if hi > lo:
            layer.clim = (lo, hi)
            self.changed.emit()
