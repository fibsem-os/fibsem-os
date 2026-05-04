"""FMImageDisplayWidget — ImagePointCanvas wrapper for multi-channel FM z-stacks.

Adds three controls below the canvas:
  - Per-channel visibility checkboxes (one row per channel, colored swatch + name)
  - Z-slice slider with label (hidden when image has only one z-plane)
  - Max Intensity Projection checkbox (disables slider when active)

All ImagePointCanvas signals are forwarded unchanged.

Usage
-----
    widget = FMImageDisplayWidget()
    widget.set_fm_image(fluorescence_image)
    widget.set_coordinates(coords)
    widget.point_moved.connect(my_handler)
"""
from __future__ import annotations

from typing import List, Optional

import matplotlib.colors
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from superqt import QRangeSlider

from fibsem.ui.widgets.custom_widgets import IconToolButton

from fibsem.correlation.structures import Coordinate, PointType
from fibsem.correlation.ui.widgets.image_point_canvas import ImagePointCanvas
from fibsem.fm.structures import FluorescenceImage

# Fallback palette when channel metadata has no color field
_DEFAULT_COLORS = ["cyan", "magenta", "yellow", "green", "red", "blue", "gray"]

_SWATCH_SIZE   = 14   # px
_CTRL_HEIGHT   = 28   # compact control row height


def _normalize_clipped(arr: np.ndarray, lo_pct: float, hi_pct: float) -> np.ndarray:
    """Normalise a 2-D array to float32 [0, 1] with percentile clipping."""
    arr = arr.astype(np.float32)
    lo = float(np.percentile(arr, lo_pct))
    hi = float(np.percentile(arr, hi_pct))
    if hi > lo:
        return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return np.zeros_like(arr)


# ---------------------------------------------------------------------------
# Range slider with explicitly painted handles (bypasses stylesheet issues)
# ---------------------------------------------------------------------------

_HANDLE_COLOR = QColor("#4a5168")   # Napari hover grey-blue


class _ClipSlider(QRangeSlider):
    """QRangeSlider that paints its own handles so they're always visible."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Color the bar (selected range) with the same Napari grey-blue
        self._style.brush_active   = "#4a5168"
        self._style.brush_inactive = "#4a5168"

    def _draw_handle(self, painter, opt) -> None:
        """Replace Qt handle drawing entirely — draws bar then our own circles."""
        if self._should_draw_bar:
            self._drawBar(painter, opt)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_HANDLE_COLOR)
        for i in range(len(self.value())):
            rect = self._handleRect(i)
            rect = rect.adjusted(1, 1, -1, -1)
            painter.drawEllipse(rect)
        painter.restore()


# ---------------------------------------------------------------------------
# Per-channel row
# ---------------------------------------------------------------------------

# Colors available in the channel color picker menu
_COLOR_MENU_CHOICES = [
    ("Cyan",    "cyan"),
    ("Magenta", "magenta"),
    ("Yellow",  "yellow"),
    ("Green",   "green"),
    ("Red",     "red"),
    ("Blue",    "blue"),
    ("White",   "white"),
    ("Gray",    "gray"),
]


class _ChannelRow(QWidget):
    """Eye toggle + color button + name label + clip slider for one channel."""

    visibility_changed = pyqtSignal(int, bool)   # (channel_index, visible)
    clip_changed       = pyqtSignal()
    color_changed      = pyqtSignal()

    def __init__(self, index: int, name: str, color: str,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._index = index
        self._color = color

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        # Visibility toggle (left)
        self._btn_vis = IconToolButton(
            icon="mdi:eye-off",
            checked_icon="mdi:eye",
            tooltip="Show channel",
            checked_tooltip="Hide channel",
            checkable=True,
            checked=True,
            size=20,
        )
        self._btn_vis.toggled.connect(self._on_state)
        layout.addWidget(self._btn_vis)

        # Color swatch button — opens color-picker menu
        self._swatch_btn = QToolButton()
        self._swatch_btn.setFixedSize(_SWATCH_SIZE + 4, _SWATCH_SIZE + 4)
        self._swatch_btn.setStyleSheet("border: none; padding: 0;")
        self._swatch_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._color_menu = QMenu(self)
        for label, css_color in _COLOR_MENU_CHOICES:
            action = self._color_menu.addAction(label)
            action.setData(css_color)
            px = QPixmap(12, 12)
            px.fill(QColor(css_color))
            action.setIcon(_pixmap_icon(px))
        self._color_menu.triggered.connect(self._on_color_action)
        self._swatch_btn.setMenu(self._color_menu)
        self._update_swatch()
        layout.addWidget(self._swatch_btn)

        # Channel name
        lbl = QLabel(name)
        lbl.setStyleSheet("color: #d0d0d0; font-size: 11px;")
        lbl.setFixedWidth(180)
        lbl.setToolTip(name)
        lbl.setTextFormat(Qt.TextFormat.PlainText)
        lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(lbl)

        # Per-channel clip range slider
        self._clip_slider = _ClipSlider(Qt.Orientation.Horizontal)
        self._clip_slider.setRange(0, 100)
        self._clip_slider.setValue((0, 100))
        self._clip_slider.setMinimumWidth(40)
        self._clip_slider.setMaximumWidth(150)
        self._clip_slider.valueChanged.connect(lambda _: self.clip_changed.emit())
        layout.addWidget(self._clip_slider)
        layout.addStretch()

    @property
    def is_visible(self) -> bool:
        return self._btn_vis.isChecked()

    @property
    def clip_value(self):
        """Return (lo_pct, hi_pct) tuple from the clip slider."""
        return self._clip_slider.value()

    @property
    def color(self) -> str:
        return self._color

    def _update_swatch(self) -> None:
        px = QPixmap(_SWATCH_SIZE, _SWATCH_SIZE)
        px.fill(QColor(self._color))
        self._swatch_btn.setIcon(_pixmap_icon(px))
        self._swatch_btn.setIconSize(px.size())

    def _on_color_action(self, action) -> None:
        self._color = action.data()
        self._update_swatch()
        self.color_changed.emit()

    def _on_state(self, checked: bool) -> None:
        self.visibility_changed.emit(self._index, checked)


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class FMImageDisplayWidget(QWidget):
    """ImagePointCanvas + FM channel / z-slice / MIP controls.

    Layout
    ------
    ┌────────────────────────────────────────┐
    │          ImagePointCanvas              │  ← expanding
    ├────────────────────────────────────────┤
    │ [☑ Max Projection]  Z: [──slider──] n/N│  ← hidden if n_z == 1
    ├────────────────────────────────────────┤
    │  [■ CH0 ☑]  [■ CH1 ☑]  [■ CH2 ☑]     │
    └────────────────────────────────────────┘
    """

    # Forward all canvas signals
    point_selected      = pyqtSignal(object)               # Coordinate
    point_moved         = pyqtSignal(object)               # Coordinate
    point_removed       = pyqtSignal(object)               # Coordinate
    canvas_clicked      = pyqtSignal(float, float)
    point_add_requested = pyqtSignal(float, float, object) # x, y, PointType

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        allowed_point_types: Optional[list] = None,
    ) -> None:
        super().__init__(parent)

        self._fm_image: Optional[FluorescenceImage] = None
        self._channel_rows: List[_ChannelRow] = []
        self._first_render = True
        self._allowed_point_types = allowed_point_types

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Canvas
        self.canvas = ImagePointCanvas(allowed_point_types=self._allowed_point_types)
        layout.addWidget(self.canvas, stretch=1)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.HLine)
        sep1.setStyleSheet("color: #3a3d42;")
        layout.addWidget(sep1)

        # Z / MIP row
        self._z_row = QWidget()
        self._z_row.setFixedHeight(_CTRL_HEIGHT)
        self._z_row.setStyleSheet("background: #1e2124;")
        z_layout = QHBoxLayout(self._z_row)
        z_layout.setContentsMargins(8, 0, 8, 0)
        z_layout.setSpacing(8)

        self._mip_check = QCheckBox("Max Projection")
        self._mip_check.setStyleSheet("color: #d0d0d0; font-size: 11px;")
        z_layout.addWidget(self._mip_check)

        z_layout.addWidget(_sep_label())

        z_layout.addWidget(QLabel("Z:"))

        self._z_slider = QSlider(Qt.Orientation.Horizontal)
        self._z_slider.setMinimum(0)
        self._z_slider.setMaximum(0)
        self._z_slider.setSingleStep(1)
        self._z_slider.setPageStep(1)
        z_layout.addWidget(self._z_slider, stretch=1)

        self._z_label = QLabel("1 / 1")
        self._z_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self._z_label.setFixedWidth(48)
        z_layout.addWidget(self._z_label)

        layout.addWidget(self._z_row)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("color: #3a3d42;")
        layout.addWidget(sep2)

        # Channel rows (scrollable, vertical stack)
        self._ch_scroll = QScrollArea()
        self._ch_scroll.setWidgetResizable(True)
        self._ch_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._ch_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._ch_scroll.setMaximumHeight(120)
        self._ch_scroll.setStyleSheet("background: #1e2124; border: none;")

        self._ch_container = QWidget()
        self._ch_container.setStyleSheet("background: #1e2124;")
        self._ch_layout = QVBoxLayout(self._ch_container)
        self._ch_layout.setContentsMargins(4, 2, 4, 2)
        self._ch_layout.setSpacing(2)
        self._ch_scroll.setWidget(self._ch_container)
        layout.addWidget(self._ch_scroll)

        # Connect controls
        self._z_slider.valueChanged.connect(self._on_z_changed)
        self._mip_check.stateChanged.connect(self._on_mip_changed)

        # Forward canvas signals
        self.canvas.point_selected.connect(self.point_selected)
        self.canvas.point_moved.connect(self.point_moved)
        self.canvas.point_removed.connect(self.point_removed)
        self.canvas.canvas_clicked.connect(self.canvas_clicked)
        self.canvas.point_add_requested.connect(self.point_add_requested)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_fm_image(self, fm_image: FluorescenceImage) -> None:
        """Load an FM image, rebuild channel rows, render initial composite."""
        self._fm_image = fm_image
        self._first_render = True
        self._build_channel_rows()
        self._update_z_controls()
        self._render()

    def set_coordinates(self, coords: List[Coordinate]) -> None:
        self.canvas.set_coordinates(coords)

    def set_selected(self, coord: Optional[Coordinate]) -> None:
        self.canvas.set_selected(coord)

    def refresh_coordinate(self, coord: Coordinate) -> None:
        self.canvas.refresh_coordinate(coord)

    def reset_view(self) -> None:
        self.canvas.reset_view()

    @property
    def current_z(self) -> int:
        """Current z-slice index (0-based)."""
        return self._z_slider.value()

    # ------------------------------------------------------------------
    # Channel row management
    # ------------------------------------------------------------------

    def _build_channel_rows(self) -> None:
        """Remove old rows and create one _ChannelRow per channel."""
        while self._ch_layout.count() > 0:
            item = self._ch_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._channel_rows.clear()

        if self._fm_image is None:
            return

        n_channels = self._fm_image.data.shape[0]
        meta_channels = (self._fm_image.metadata.channels
                         if self._fm_image.metadata.channels else [])

        for i in range(n_channels):
            if i < len(meta_channels):
                name  = meta_channels[i].name or f"CH {i}"
                color = meta_channels[i].color or _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
            else:
                name  = f"CH {i}"
                color = _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]

            row = _ChannelRow(i, name, color)
            row.visibility_changed.connect(self._on_visibility_changed)
            row.clip_changed.connect(self._render)
            row.color_changed.connect(self._render)
            self._ch_layout.insertWidget(i, row)
            self._channel_rows.append(row)

    # ------------------------------------------------------------------
    # Z / MIP controls
    # ------------------------------------------------------------------

    def _update_z_controls(self) -> None:
        if self._fm_image is None:
            self._z_row.setVisible(False)
            return

        n_z = self._fm_image.data.shape[1]
        if n_z <= 1:
            self._z_row.setVisible(False)
            return

        self._z_row.setVisible(True)
        self._z_slider.blockSignals(True)
        self._z_slider.setMaximum(n_z - 1)
        self._z_slider.setValue(n_z // 2)
        self._z_slider.blockSignals(False)
        self._update_z_label()

    def _update_z_label(self) -> None:
        if self._fm_image is None:
            return
        n_z = self._fm_image.data.shape[1]
        z   = self._z_slider.value()
        self._z_label.setText(f"{z} / {n_z - 1}")

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self) -> None:
        composite = self._composite()
        if self._first_render:
            self.canvas.set_image(composite)
            self._first_render = False
        else:
            self.canvas.update_display(composite)

    def _composite(self) -> np.ndarray:
        """Additive RGB blend of visible channels at current z or MIP."""
        if self._fm_image is None:
            return np.zeros((64, 64, 3), dtype=np.float32)

        use_mip = self._mip_check.isChecked()
        z       = self._z_slider.value()
        h, w    = self._fm_image.data.shape[2:]
        out     = np.zeros((h, w, 3), dtype=np.float32)

        for i, row in enumerate(self._channel_rows):
            if not row.is_visible:
                continue

            ch_data = self._fm_image.data[i]          # (Z, Y, X)
            plane   = ch_data.max(axis=0) if use_mip else ch_data[z]
            lo_pct, hi_pct = row.clip_value
            plane   = _normalize_clipped(plane, lo_pct, hi_pct)

            try:
                r, g, b = matplotlib.colors.to_rgb(row.color)
            except ValueError:
                r, g, b = 1.0, 1.0, 1.0

            out[..., 0] += plane * r
            out[..., 1] += plane * g
            out[..., 2] += plane * b

        return np.clip(out, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_z_changed(self) -> None:
        self._update_z_label()
        if not self._mip_check.isChecked():
            self._render()

    def _on_mip_changed(self) -> None:
        enabled = not self._mip_check.isChecked()
        self._z_slider.setEnabled(enabled)
        self._render()

    def _on_visibility_changed(self, _index: int, _visible: bool) -> None:
        self._render()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pixmap_icon(px: QPixmap) -> QIcon:
    return QIcon(px)


def _sep_label() -> QFrame:
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.VLine)
    sep.setStyleSheet("color: #3a3d42;")
    sep.setFixedHeight(18)
    return sep
