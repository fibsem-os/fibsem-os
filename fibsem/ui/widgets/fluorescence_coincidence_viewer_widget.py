"""Integrated Fluorescence Coincidence Viewer Widget.

Four-quadrant layout:
  FIB canvas  | FM canvas  | Tab widget (rowspan 2)
  ------------|------------|
  Line plot   | Info panel |
  Bottom bar (milling controls, spanning all columns)

FIB and FM canvases use ``FibsemImageCanvas`` with ``RectOverlay`` for the
interactive rectangles:

* FIB — yellow filled rectangle, drag-only
* FM  — white dotted rectangle, drag + resize

The tab widget (right column) holds four tabs:
  1. Lamella  — LamellaNameListWidget for selection and navigation
  2. FIB      — FibsemBeamWidget (ION beam settings)
  3. Milling  — MillingTaskViewerWidget (config, no internal run buttons)
  4. FM       — FMControlWidget (fluorescence microscope control)

The bottom bar shows the selected lamella name, start/pause/stop buttons,
and two progress bars driven by ``microscope.milling_progress_signal``.
"""

import datetime
import logging
import math
import time
import threading
from collections import deque
from datetime import timedelta
from pprint import pformat
from typing import TYPE_CHECKING, Optional

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from superqt import ensure_main_thread, QDoubleSlider
from fibsem.ui.icon import fibsem_icon

from fibsem import conversions
from fibsem.autofunctions.gamma import apply_gamma
from fibsem.constants import METRE_TO_MICRON, MICRON_TO_METRE
from fibsem.fm.structures import CameraImageTransform, FluorescenceImage
from fibsem.milling.strategy.coincidence import CoincidenceMillingStrategy
from fibsem.structures import BeamType, FibsemImage, Point
from fibsem.ui import notification_service, stylesheets
from fibsem.ui.fm.widgets import LinePlotWidget
from fibsem.ui.widgets.custom_widgets import LamellaNameListWidget, TitledPanel
from fibsem.ui.widgets.selected_lamella_widget import SelectedLamellaWidget
from fibsem.ui.widgets.image_canvas import (
    FibsemImageCanvas,
    RectOverlay,
    ScanDirectionArrowOverlay,
)

if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import Experiment, Lamella
    from fibsem.microscope import FibsemMicroscope
    from fibsem.milling.tasks import FibsemMillingTaskConfig

_BG = "#262930"
_HEADER_BG = "#1e2124"

COINCIDENCE_BORDER_STYLESHEET = """
    QFrame#coincidence_border_frame[borderState="idle"]       { border: 4px solid #262930; }
    QFrame#coincidence_border_frame[borderState="automated"]  { border: 4px solid #4caf50; }
    QFrame#coincidence_border_frame[borderState="supervised"] { border: 4px solid #007ACC; }
    QFrame#coincidence_border_frame[borderState="waiting"]    { border: 4px solid #ff9800; }
    QFrame#coincidence_border_frame[borderState="finished"]   { border: 4px solid #4caf50; }
    QFrame#coincidence_border_frame[borderState="stopped"]    { border: 4px solid #99121F; }
"""


def _fmt_timestamp(ts: float) -> str:
    """Format a Unix timestamp as h:MM:SSam/pm (e.g. '6:50:20PM')."""
    dt = datetime.datetime.fromtimestamp(ts)
    h = dt.hour % 12 or 12
    ampm = "AM" if dt.hour < 12 else "PM"
    return f"{h}:{dt.strftime('%M:%S')}{ampm}"


_HISTOGRAM_PANEL_STYLE = (
    "QFrame { background: rgba(30,33,36,230); border: 1px solid #555;"
    " border-radius: 4px; }"
    "QLabel { color: #d1d2d4; font-size: 10px; background: transparent; border: none; }"
    "QPushButton { background: rgba(60,63,70,200); border: 1px solid #666;"
    " border-radius: 3px; color: #d1d2d4; font-size: 10px; padding: 2px 8px; }"
    "QPushButton:hover { background: rgba(80,83,90,220); }"
)


def _build_histogram_panel(canvas, on_min, on_max, on_gamma, on_reset):
    """Create a floating histogram/contrast panel parented to *canvas*.

    Returns (panel, sld_min, lbl_min, sld_max, lbl_max, sld_gamma, lbl_gamma).
    """
    panel = QFrame(canvas)
    panel.setStyleSheet(_HISTOGRAM_PANEL_STYLE)
    panel.setFixedWidth(240)

    outer = QVBoxLayout(panel)
    outer.setContentsMargins(8, 6, 8, 6)
    outer.setSpacing(4)

    title = QLabel("Contrast / Gamma")
    title.setStyleSheet(
        "color: #aaa; font-size: 10px; font-weight: bold;"
        " background: transparent; border: none;"
    )
    outer.addWidget(title)

    form = QFormLayout()
    form.setContentsMargins(0, 0, 0, 0)
    form.setSpacing(3)
    form.setLabelAlignment(Qt.AlignRight)  # type: ignore[attr-defined]

    def _make_row(lo, hi, default, step):
        slider = QDoubleSlider(Qt.Horizontal)  # type: ignore[attr-defined]
        slider.setRange(lo, hi)
        slider.setSingleStep(step)
        slider.setValue(default)
        lbl = QLabel(f"{default:.2f}")
        lbl.setFixedWidth(32)
        lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)  # type: ignore[attr-defined]
        row = QWidget()
        row.setStyleSheet("background: transparent; border: none;")
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(4)
        rl.addWidget(slider)
        rl.addWidget(lbl)
        return slider, lbl, row

    sld_min, lbl_min, row_min = _make_row(0.0, 1.0, 0.0, 0.01)
    sld_max, lbl_max, row_max = _make_row(0.0, 1.0, 1.0, 0.01)
    sld_gamma, lbl_gamma, row_gamma = _make_row(0.1, 3.0, 1.0, 0.05)

    form.addRow(QLabel("Min"), row_min)
    form.addRow(QLabel("Max"), row_max)
    form.addRow(QLabel("Gamma"), row_gamma)
    outer.addLayout(form)

    btn_reset = QPushButton("Reset")
    btn_reset.clicked.connect(on_reset)
    outer.addWidget(btn_reset, alignment=Qt.AlignRight)  # type: ignore[attr-defined]

    sld_min.valueChanged.connect(on_min)
    sld_max.valueChanged.connect(on_max)
    sld_gamma.valueChanged.connect(on_gamma)

    panel.setVisible(False)
    panel.adjustSize()
    return panel, sld_min, lbl_min, sld_max, lbl_max, sld_gamma, lbl_gamma


# ---------------------------------------------------------------------------
# FIB quadrant — FibsemImageCanvas + yellow drag-only RectOverlay
# ---------------------------------------------------------------------------


class _FibImageCanvas(QWidget):
    """FibsemImage viewer with a draggable yellow rectangle."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._contrast_min: float = 0.0
        self._contrast_max: float = 1.0
        self._gamma: float = 1.0
        self._raw_frame: Optional[np.ndarray] = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)

        self.canvas = FibsemImageCanvas()
        self.rect_overlay = RectOverlay(
            color="yellow",
            facecolor="yellow",
            alpha=0.5,
            linewidth=2,
            linestyle="solid",
            resizable=False,
        )
        self.canvas.add_overlay(self.rect_overlay)

        self.arrow_overlay = ScanDirectionArrowOverlay(color="yellow")
        self.canvas.add_overlay(self.arrow_overlay)
        self.canvas.set_crosshair_visible(True)

        layout.addWidget(self.canvas)

        self.btn_histogram = self.canvas._add_overlay_button(
            "mdi:contrast-box", "Histogram Controls", self._toggle_histogram_panel, checkable=True
        )
        (
            self._histogram_panel,
            self._sld_min, self._lbl_min,
            self._sld_max, self._lbl_max,
            self._sld_gamma, self._lbl_gamma,
        ) = _build_histogram_panel(
            self.canvas,
            self._on_min_changed,
            self._on_max_changed,
            self._on_gamma_changed,
            self._reset_histogram_controls,
        )

    def _on_min_changed(self, val: float) -> None:
        if val >= self._contrast_max:
            val = max(0.0, self._contrast_max - 0.01)
            self._sld_min.setValue(val)
        self._contrast_min = val
        self._lbl_min.setText(f"{val:.2f}")
        self._refresh_display()

    def _on_max_changed(self, val: float) -> None:
        if val <= self._contrast_min:
            val = min(1.0, self._contrast_min + 0.01)
            self._sld_max.setValue(val)
        self._contrast_max = val
        self._lbl_max.setText(f"{val:.2f}")
        self._refresh_display()

    def _on_gamma_changed(self, val: float) -> None:
        self._gamma = val
        self._lbl_gamma.setText(f"{val:.2f}")
        self._refresh_display()

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        fmin, fmax = float(frame.min()), float(frame.max())
        if fmax > fmin:
            norm = (frame.astype(np.float32) - fmin) / (fmax - fmin)
        else:
            norm = np.zeros_like(frame, dtype=np.float32)
        lo, hi = self._contrast_min, self._contrast_max
        norm = np.clip(norm, lo, hi)
        if hi > lo:
            norm = (norm - lo) / (hi - lo)
        if self._gamma != 1.0:
            norm = apply_gamma(norm, self._gamma)
        return norm

    def _refresh_display(self) -> None:
        if self._raw_frame is not None:
            processed = self._process_frame(self._raw_frame)
            self.canvas.update_display(processed)
            imgs = self.canvas._ax.get_images()
            if imgs:
                imgs[0].set_clim(0.0, 1.0)
            self.canvas.draw_idle()

    def _reset_histogram_controls(self) -> None:
        self._contrast_min = 0.0
        self._contrast_max = 1.0
        self._gamma = 1.0
        self._sld_min.setValue(0.0)
        self._sld_max.setValue(1.0)
        self._sld_gamma.setValue(1.0)
        self._refresh_display()

    def _toggle_histogram_panel(self) -> None:
        visible = not self._histogram_panel.isVisible()
        self._histogram_panel.setVisible(visible)
        if visible:
            self._position_histogram_panel()
            self._histogram_panel.raise_()

    def _position_histogram_panel(self) -> None:
        btn = self.btn_histogram
        panel = self._histogram_panel
        panel.adjustSize()
        x = self.canvas.width() - panel.width() - 4
        y = btn.y() + btn.height() + 4
        panel.move(x, y)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._histogram_panel.isVisible():
            self._position_histogram_panel()

    def set_image(self, image: FibsemImage):
        self._raw_frame = image.data
        self.canvas.set_image(image)
        processed = self._process_frame(image.data)
        self.canvas.update_display(processed)
        imgs = self.canvas._ax.get_images()
        if imgs:
            imgs[0].set_clim(0.0, 1.0)

    def set_scan_direction(
        self, cx: float, cy: float, h_px: float, scan_direction: str
    ) -> None:
        """Update the scan direction arrow. Pass scan_direction="" to hide."""
        self.arrow_overlay.set_arrow(cx, cy, h_px, scan_direction)

    def clear(self):
        self._raw_frame = None
        self.canvas.clear()


# ---------------------------------------------------------------------------
# FM quadrant — FibsemImageCanvas + white dotted resizable RectOverlay
#               wrapped in a QWidget to include the z-slice controls
# ---------------------------------------------------------------------------


class _FmImageCanvas(QWidget):
    """FluorescenceImage viewer with z-slice navigation and a resizable rectangle."""

    scrub_requested = pyqtSignal(int)
    reset_live_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: Optional[FluorescenceImage] = None
        self._data: Optional[np.ndarray] = None
        self._img_shape: Optional[tuple] = None  # (H, W)
        self._contrast_min: float = 0.0
        self._contrast_max: float = 1.0
        self._gamma: float = 1.0
        self._raw_frame: Optional[np.ndarray] = None
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)

        self.canvas = FibsemImageCanvas()
        self.rect_overlay = RectOverlay(
            color="yellow",  # yellow reads against bright/white reflection images
            facecolor=None,
            alpha=0.9,
            linewidth=2,
            linestyle="--",
            resizable=True,
        )
        self.canvas.add_overlay(self.rect_overlay)
        self.canvas.set_crosshair_visible(True)
        layout.addWidget(self.canvas, 1)

        # Histogram controls button (4th from right, after reset/scalebar/crosshair)
        self.btn_histogram = self.canvas._add_overlay_button(
            "mdi:contrast-box", "Histogram Controls", self._toggle_histogram_panel, checkable=True
        )
        (
            self._histogram_panel,
            self._sld_min, self._lbl_min,
            self._sld_max, self._lbl_max,
            self._sld_gamma, self._lbl_gamma,
        ) = _build_histogram_panel(
            self.canvas,
            self._on_min_changed,
            self._on_max_changed,
            self._on_gamma_changed,
            self._reset_histogram_controls,
        )

        # Timelapse scrubber row (hidden until frames accumulate)
        scrubber_row = QHBoxLayout()
        scrubber_row.setContentsMargins(4, 0, 4, 0)
        self.time_slider = QSlider(Qt.Horizontal)  # type: ignore[attr-defined]
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(0)
        self.time_slider.setSingleStep(1)
        self.time_slider.setEnabled(False)
        self.frame_label = QLabel("00:00:00 (0/0)")
        self.frame_label.setFixedWidth(110)
        self.frame_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)  # type: ignore[attr-defined]
        self.frame_label.setStyleSheet("color: #d1d2d4; font-size: 10px;")
        self.btn_live = QPushButton("Live")
        self.btn_live.setFixedWidth(44)
        self.btn_live.setToolTip("Return to live view")
        self.btn_live.setStyleSheet(
            "QPushButton { background: #2e7d32; color: white; font-size: 10px; "
            "padding: 1px 4px; border-radius: 3px; }"
            "QPushButton:hover { background: #388e3c; }"
        )
        scrubber_row.addWidget(self.time_slider)
        scrubber_row.addWidget(self.frame_label)
        scrubber_row.addWidget(self.btn_live)
        self.scrubber_widget = QWidget()
        self.scrubber_widget.setLayout(scrubber_row)
        self.scrubber_widget.setVisible(False)
        layout.addWidget(self.scrubber_widget)

    def _connect_signals(self):
        self.time_slider.valueChanged.connect(self._on_scrub)
        self.btn_live.clicked.connect(self.reset_live_requested.emit)

    def _on_min_changed(self, val: float) -> None:
        if val >= self._contrast_max:
            val = max(0.0, self._contrast_max - 0.01)
            self._sld_min.setValue(val)
        self._contrast_min = val
        self._lbl_min.setText(f"{val:.2f}")
        self._refresh_display()

    def _on_max_changed(self, val: float) -> None:
        if val <= self._contrast_min:
            val = min(1.0, self._contrast_min + 0.01)
            self._sld_max.setValue(val)
        self._contrast_max = val
        self._lbl_max.setText(f"{val:.2f}")
        self._refresh_display()

    def _on_gamma_changed(self, val: float) -> None:
        self._gamma = val
        self._lbl_gamma.setText(f"{val:.2f}")
        self._refresh_display()

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Normalize frame, apply contrast limits and gamma; returns float32 [0, 1]."""
        fmin, fmax = float(frame.min()), float(frame.max())
        if fmax > fmin:
            norm = (frame.astype(np.float32) - fmin) / (fmax - fmin)
        else:
            norm = np.zeros_like(frame, dtype=np.float32)
        lo, hi = self._contrast_min, self._contrast_max
        norm = np.clip(norm, lo, hi)
        if hi > lo:
            norm = (norm - lo) / (hi - lo)
        if self._gamma != 1.0:
            norm = apply_gamma(norm, self._gamma)
        return norm

    def _refresh_display(self) -> None:
        if self._raw_frame is not None:
            processed = self._process_frame(self._raw_frame)
            self.canvas.update_display(processed)
            imgs = self.canvas._ax.get_images()
            if imgs:
                imgs[0].set_clim(0.0, 1.0)
            self.canvas.draw_idle()

    def _reset_histogram_controls(self) -> None:
        self._contrast_min = 0.0
        self._contrast_max = 1.0
        self._gamma = 1.0
        self._sld_min.setValue(0.0)
        self._sld_max.setValue(1.0)
        self._sld_gamma.setValue(1.0)
        self._refresh_display()

    def _toggle_histogram_panel(self) -> None:
        visible = not self._histogram_panel.isVisible()
        self._histogram_panel.setVisible(visible)
        if visible:
            self._position_histogram_panel()
            self._histogram_panel.raise_()

    def _position_histogram_panel(self) -> None:
        btn = self.btn_histogram
        panel = self._histogram_panel
        panel.adjustSize()
        x = self.canvas.width() - panel.width() - 4
        y = btn.y() + btn.height() + 4
        panel.move(x, y)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._histogram_panel.isVisible():
            self._position_histogram_panel()

    def _on_scrub(self, idx: int) -> None:
        self.scrub_requested.emit(idx)

    def set_timelapse_length(self, n: int) -> None:
        """Update the scrubber range and visibility."""
        self.time_slider.setMaximum(max(0, n - 1))
        self.time_slider.setEnabled(n > 1)
        self.scrubber_widget.setVisible(n > 0)

    def display_timelapse_frame(
        self, arr: np.ndarray, idx: int, total: int, ts: float = 0.0
    ) -> None:
        """Show a pre-processed timelapse frame without disrupting the live canvas state."""
        processed = self._process_frame(arr)
        self.canvas.update_display(processed)
        imgs = self.canvas._ax.get_images()
        if imgs:
            imgs[0].set_clim(0.0, 1.0)
        self.frame_label.setText(f"{_fmt_timestamp(ts)} ({idx}/{total - 1})")

    def set_image(self, image: FluorescenceImage):
        self._image = image
        data = image.data
        # Normalise to CZYX regardless of incoming shape
        if data.ndim == 5:
            data = data[0]  # TCZYX → CZYX
        if data.ndim == 2:
            data = data[None, None]  # YX → 1,1,Y,X
        elif data.ndim == 3:
            data = data[:, None]  # CYX → C,1,Y,X
        self._data = data
        H, W = data.shape[2], data.shape[3]
        # Only reset axes/overlays when the image shape changes (e.g. first
        # image or resolution change). For live updates with the same shape,
        # just swap the pixel data so the rectangle overlay is undisturbed.
        shape_changed = self._img_shape != (H, W)
        self._img_shape = (H, W)
        self._show_slice(0, reset_overlays=shape_changed)

    def _show_slice(self, z: int, *, reset_overlays: bool):
        if self._image is None or self._data is None:
            return
        try:
            frame = np.max(self._data[:, z, :, :], axis=0)  # (Y, X)
            self._raw_frame = frame
            processed = self._process_frame(frame)
            nz = self._data.shape[1]

            def _apply_processed(p: np.ndarray) -> None:
                self.canvas.update_display(p)
                _imgs = self.canvas._ax.get_images()
                if _imgs:
                    _imgs[0].set_clim(0.0, 1.0)

            if reset_overlays:
                # FibsemImage requires uint8/uint16 — use the raw frame for
                # overlay/scalebar setup, then swap in the processed display data.
                wrapped = FibsemImage(
                    data=frame,
                    metadata=self._image.metadata
                    if hasattr(self._image, "metadata")
                    else None,
                )
                self.canvas.set_image(wrapped)
                _apply_processed(processed)
                # FluorescenceImageMetadata uses pixel_size_x, not pixel_size.x —
                # set _pixel_size directly so the scalebar is rendered.
                px = getattr(
                    getattr(self._image, "metadata", None), "pixel_size_x", None
                )
                if px:
                    self.canvas._pixel_size = px
                    self.canvas._refresh_scalebar()
                self.canvas._ax.set_title(
                    f"FM  z={z}/{nz - 1}", color="white", fontsize=10
                )
                self.canvas.draw_idle()
            else:
                _apply_processed(processed)
                self.canvas._ax.set_title(
                    f"FM  z={z}/{nz - 1}", color="white", fontsize=10
                )
                self.canvas.draw_idle()
        except Exception:
            logging.exception("Error plotting FM z-slice")

    def clear(self):
        self._image = None
        self._data = None
        self._img_shape = None
        self._raw_frame = None
        self.canvas.clear()


# ---------------------------------------------------------------------------
# Info panel
# ---------------------------------------------------------------------------


class _InfoWidget(QWidget):
    """Displays rectangle info (px + physical units), intensity stats, and live run metrics."""

    _LABEL_STYLE = "color: #d1d2d4; font-family: arial; font-size: 10px;"
    _HEADER_STYLE = "color: #aaa; font-weight: bold; font-size: 10px;"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fib_pixel_size: Optional[float] = None
        self._fm_pixel_size: Optional[float] = None
        # run-metrics accumulators
        self._frame_count: int = 0
        self._drop_count: int = 0
        self._first_drop_elapsed: Optional[float] = None
        self._recent_ts: deque = deque(maxlen=10)
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(0)

        cols = QHBoxLayout()
        cols.setContentsMargins(0, 0, 0, 0)
        cols.setSpacing(8)
        outer.addLayout(cols)
        outer.addStretch()

        # ── Left column: Rectangle Info ───────────────────────────────────
        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(4)
        cols.addLayout(left, 1)

        # ── Right column: Stats ───────────────────────────────────────────
        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(4)
        cols.addLayout(right, 1)

        def _section(col: QVBoxLayout, title: str) -> QLabel:
            header = QLabel(title)
            header.setStyleSheet(self._HEADER_STYLE)
            sep = QFrame()
            sep.setFrameShape(QFrame.HLine)
            sep.setStyleSheet("color: #3a3d42;")
            lbl = QLabel("—")
            lbl.setStyleSheet(self._LABEL_STYLE)
            lbl.setWordWrap(True)
            col.addWidget(header)
            col.addWidget(sep)
            col.addWidget(lbl)
            col.addSpacing(6)
            return lbl

        self._fib_label = _section(left, "FIB Rectangle")
        self._fm_label = _section(left, "FM Rectangle")
        left.addStretch()

        self._intensity_label = _section(right, "Intensity Stats")

        # Run metrics — hidden until milling starts
        self._run_metrics_container = QWidget()
        rm_layout = QVBoxLayout(self._run_metrics_container)
        rm_layout.setContentsMargins(0, 0, 0, 0)
        rm_layout.setSpacing(4)
        rm_header = QLabel("Run Metrics")
        rm_header.setStyleSheet(self._HEADER_STYLE)
        rm_sep = QFrame()
        rm_sep.setFrameShape(QFrame.HLine)
        rm_sep.setStyleSheet("color: #3a3d42;")
        self._run_metrics_label = QLabel("—")
        self._run_metrics_label.setStyleSheet(self._LABEL_STYLE)
        self._run_metrics_label.setWordWrap(True)
        rm_layout.addWidget(rm_header)
        rm_layout.addWidget(rm_sep)
        rm_layout.addWidget(self._run_metrics_label)
        self._run_metrics_container.setVisible(False)
        right.addWidget(self._run_metrics_container)
        right.addStretch()

    # ------------------------------------------------------------------
    # Pixel-size injection
    # ------------------------------------------------------------------

    def set_fib_pixel_size(self, px: Optional[float]) -> None:
        self._fib_pixel_size = px

    def set_fm_pixel_size(self, px: Optional[float]) -> None:
        self._fm_pixel_size = px

    # ------------------------------------------------------------------
    # Rectangle info
    # ------------------------------------------------------------------

    @staticmethod
    def _format(info: dict, pixel_size: Optional[float] = None) -> str:
        lines = [
            f"Top-left : ({info['x0']}, {info['y0']}) px",
            f"Bot-right: ({info['x1']}, {info['y1']}) px",
            f"Centre   : ({info['cx']}, {info['cy']}) px",
            f"Size     : {info['width']} × {info['height']} px",
        ]
        if pixel_size and pixel_size > 0:
            w_um = info["width"] * pixel_size * 1e6
            h_um = info["height"] * pixel_size * 1e6
            lines.append(f"Size     : {w_um:.2f} × {h_um:.2f} µm")
            lines.append(f"Area     : {w_um * h_um:.2f} µm²")
        return "\n".join(lines)

    def update_fib(self, info: dict) -> None:
        self._fib_label.setText(self._format(info, self._fib_pixel_size))

    def update_fm(self, info: dict) -> None:
        self._fm_label.setText(self._format(info, self._fm_pixel_size))

    # ------------------------------------------------------------------
    # Intensity stats
    # ------------------------------------------------------------------

    def update_stats(self, stats: dict) -> None:
        warmup = stats.get("warmup_complete", False)
        value = stats.get("value", 0.0)
        rolling = stats.get("rolling_mean", 0.0)
        peak = stats.get("peak_rolling_mean", 0.0)
        threshold = stats.get("threshold_value", 0.0)

        if not warmup:
            text = f"Current : {value:.2f}\nWarmup in progress…"
        else:
            drop_pct = ((peak - rolling) / peak * 100) if peak > 0 else 0.0
            text = (
                f"Current  : {value:.2f}\n"
                f"Rolling  : {rolling:.2f}\n"
                f"Peak     : {peak:.2f}\n"
                f"Threshold: {threshold:.2f}\n"
                f"Drop     : {drop_pct:.1f}%"
            )
        self._intensity_label.setText(text)

    # ------------------------------------------------------------------
    # Live run metrics
    # ------------------------------------------------------------------

    def show_run_metrics(self, visible: bool) -> None:
        self._run_metrics_container.setVisible(visible)

    def update_run_metrics(self, stats: dict) -> None:
        self._frame_count += 1
        ts = stats.get("timestamp", time.time())
        self._recent_ts.append(ts)

        if stats.get("drop_detected"):
            self._drop_count += 1
            if self._first_drop_elapsed is None:
                self._first_drop_elapsed = stats.get("elapsed_time", 0.0)

        fps = 0.0
        if len(self._recent_ts) >= 2:
            dt = self._recent_ts[-1] - self._recent_ts[0]
            if dt > 0:
                fps = (len(self._recent_ts) - 1) / dt

        elapsed = stats.get("elapsed_time", 0.0)
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        first_drop_str = (
            f"{self._first_drop_elapsed:.1f}s"
            if self._first_drop_elapsed is not None
            else "—"
        )

        self._run_metrics_label.setText(
            f"Elapsed  : {elapsed_str}\n"
            f"Frames   : {self._frame_count}  ({fps:.1f} fps)\n"
            f"Drops    : {self._drop_count}\n"
            f"1st drop : {first_drop_str}"
        )

    # ------------------------------------------------------------------

    def clear_stats(self) -> None:
        self._intensity_label.setText("—")
        self._frame_count = 0
        self._drop_count = 0
        self._first_drop_elapsed = None
        self._recent_ts.clear()
        self._run_metrics_label.setText("—")
        self._run_metrics_container.setVisible(True)


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------


class FluorescenceCoincidenceViewerWidget(QWidget):
    """Four-quadrant viewer + control tabs for FIB/FM coincidence milling.

    Layout::

        FIB canvas  | FM canvas  | QTabWidget (rowspan 2)
        Line plot   | Info panel |
        Bottom bar (milling controls, spanning all columns)

    Tabs:
        1. Lamella  — LamellaNameListWidget (select + move-to)
        2. FIB      — FibsemBeamWidget (ION)
        3. Milling  — MillingTaskViewerWidget (config only)
        4. FM       — FMControlWidget (fluorescence microscope)

    Public attributes:
        fib_canvas              (_FibImageCanvas)
        fm_canvas               (_FmImageCanvas)
        line_plot_widget        (LinePlotWidget)
        tab_widget              (QTabWidget)
        lamella_list_widget     (LamellaNameListWidget)
        milling_viewer_widget   (MillingTaskViewerWidget | None)
        fm_control_widget       (FMControlWidget | None)
        fib_beam_widget         (FibsemBeamWidget | None)

    Signals:
        lamella_selected_signal(object)  — emits Lamella on selection change
        milling_started_signal()
        milling_finished_signal()
    """

    lamella_selected_signal = pyqtSignal(object)
    milling_started_signal = pyqtSignal()
    milling_finished_signal = pyqtSignal()

    # internal signals for thread-safe canvas updates
    _fib_image_ready = pyqtSignal(object)
    _fm_image_ready = pyqtSignal(object)
    _fib_acquire_done = pyqtSignal()
    _fm_acquire_done = pyqtSignal()
    _intensity_ready = pyqtSignal(float)

    def __init__(
        self,
        microscope: "FibsemMicroscope",
        experiment: "Experiment",
        viewer=None,
        parent=None,
    ):
        super().__init__(parent)
        self.microscope = microscope
        self.experiment = experiment
        self.viewer = viewer
        self._selected_lamella: Optional["Lamella"] = None
        self._active_strategies: list["CoincidenceMillingStrategy"] = []
        self.selected_view: int = 0  # 0 = FM, 1 = FIB

        # Timelapse accumulation state
        self._timelapse_frames: list[np.ndarray] = []
        self._timelapse_timestamps: list[float] = []
        self._last_timelapse_time: float = 0.0
        self._is_scrubbing: bool = False
        self._is_milling_active: bool = False

        # Optional sub-widgets (created only when microscope/fm is available)
        self.fib_beam_widget = None
        self.milling_viewer_widget = None
        self.fm_control_widget = None

        self._setup_ui()
        self._connect_signals()

        self.set_experiment(experiment)

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        # ── Border frame — wraps splitter content, driven by _set_border_state ──
        self._border_frame = QFrame()
        self._border_frame.setObjectName("coincidence_border_frame")
        self._border_frame.setProperty("borderState", "idle")
        self._border_state: str = "idle"
        self._border_enabled: bool = True
        _frame_layout = QVBoxLayout(self._border_frame)
        _frame_layout.setContentsMargins(0, 0, 0, 0)
        _frame_layout.setSpacing(0)
        outer.addWidget(self._border_frame, 1)

        # ── Splitter: left (4-quadrant grid) | right (tab widget) ─────────
        self.splitter = QSplitter(Qt.Horizontal)  # type: ignore[attr-defined]
        self.splitter.setChildrenCollapsible(False)
        _frame_layout.addWidget(self.splitter)

        # Left side — 2×2 QSplitter grid
        self.fib_canvas = _FibImageCanvas()
        self.fm_canvas = _FmImageCanvas()

        # Thin QFrame wrappers used solely for the selection border.
        # QFrame avoids the QWidget { } cascade from NAPARI_STYLE.
        self._fm_frame = self._make_canvas_frame(self.fm_canvas, "fm_canvas_frame")
        self._fib_frame = self._make_canvas_frame(self.fib_canvas, "fib_canvas_frame")

        top_splitter = QSplitter(Qt.Horizontal)  # type: ignore[attr-defined]
        top_splitter.setChildrenCollapsible(False)
        top_splitter.addWidget(self._fm_frame)
        top_splitter.addWidget(self._fib_frame)
        top_splitter.setStretchFactor(0, 1)
        top_splitter.setStretchFactor(1, 1)

        self.line_plot_widget = LinePlotWidget(max_length=5000, rolling_mean_window=25)
        line_panel = TitledPanel(
            "Intensity", content=self.line_plot_widget, collapsible=False
        )
        self._info_widget = _InfoWidget()
        info_panel = TitledPanel("Info", content=self._info_widget, collapsible=False)

        bottom_splitter = QSplitter(Qt.Horizontal)  # type: ignore[attr-defined]
        bottom_splitter.setChildrenCollapsible(False)
        bottom_splitter.addWidget(line_panel)
        bottom_splitter.addWidget(info_panel)
        bottom_splitter.setStretchFactor(0, 1)
        bottom_splitter.setStretchFactor(1, 1)

        left = QSplitter(Qt.Vertical)  # type: ignore[attr-defined]
        left.setChildrenCollapsible(False)
        left.addWidget(top_splitter)
        left.addWidget(bottom_splitter)
        left.setStretchFactor(0, 1)
        left.setStretchFactor(1, 1)

        self.splitter.addWidget(left)

        # Right side — tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setMinimumWidth(200)
        self.tab_widget.addTab(self._build_lamella_tab(), "Experiment")
        self.tab_widget.addTab(self._build_fib_tab(), "FIB")
        self.tab_widget.addTab(self._build_milling_tab(), "Milling")
        self.tab_widget.addTab(self._build_fm_tab(), "Fluorescence")
        self.splitter.addWidget(self.tab_widget)

        # Default splitter proportions: ~75% left, ~25% right
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 1)

        # ── Bottom bar ────────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #3a3d42;")
        outer.addWidget(sep)

        outer.addWidget(self._build_bottom_bar())

        self.setStyleSheet(stylesheets.NAPARI_STYLE + COINCIDENCE_BORDER_STYLESHEET)

        # Apply initial selected-view border (FM selected by default)
        self._set_selected_view(self.selected_view)

    # ------------------------------------------------------------------
    # Tab builders
    # ------------------------------------------------------------------

    def _build_lamella_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.lamella_list_widget = LamellaNameListWidget()
        self.lamella_list_widget.enable_move_to_action(True)
        self.lamella_list_widget.enable_defect_button(True)
        layout.addWidget(self.lamella_list_widget)

        # per-lamella objective position + poses for the selected lamella
        self.selected_lamella_widget = SelectedLamellaWidget()
        self.selected_lamella_widget.objective_position_changed.connect(
            self._on_slw_objective_position_changed
        )
        self.selected_lamella_widget.use_current_objective_requested.connect(
            self._on_slw_use_current_objective
        )
        self.selected_lamella_widget.apply_objective_to_all_requested.connect(
            self._on_slw_apply_objective_to_all
        )
        self.selected_lamella_widget.move_objective_requested.connect(
            self._on_slw_move_objective
        )
        self.selected_lamella_widget.pose_move_to_requested.connect(
            self._on_slw_pose_move_to
        )
        self.selected_lamella_widget.pose_update_requested.connect(
            self._on_slw_pose_update
        )
        layout.addWidget(self.selected_lamella_widget)
        layout.addStretch()
        return container

    def _build_fib_tab(self) -> QWidget:
        if self.microscope is None:
            return self._placeholder("No microscope connected")

        from fibsem.ui.widgets.beam_widget import FibsemBeamWidget

        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Beam + detector settings, pre-loaded from microscope
        self.fib_beam_widget = FibsemBeamWidget(
            microscope=self.microscope, beam_type=BeamType.ION
        )
        self.fib_beam_widget.sync_from_microscope()
        layout.addWidget(self.fib_beam_widget)

        # AutoContrast / AutoFocus row
        auto_row = QHBoxLayout()
        auto_row.setSpacing(4)
        self.btn_autocontrast_fib = QPushButton("AutoContrast")
        self.btn_autocontrast_fib.setIcon(
            fibsem_icon("mdi:contrast-circle", color=stylesheets.GRAY_ICON_COLOR)
        )
        self.btn_autocontrast_fib.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.btn_autofocus_fib = QPushButton("AutoFocus")
        self.btn_autofocus_fib.setIcon(
            fibsem_icon(
                "mdi:image-filter-center-focus", color=stylesheets.GRAY_ICON_COLOR
            )
        )
        self.btn_autofocus_fib.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        auto_row.addWidget(self.btn_autocontrast_fib)
        auto_row.addWidget(self.btn_autofocus_fib)
        layout.addLayout(auto_row)

        # Acquire button
        self.btn_acquire_fib = QPushButton("Acquire FIB Image")
        self.btn_acquire_fib.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        layout.addWidget(self.btn_acquire_fib)
        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(scroll_content)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # type: ignore[attr-defined]
        return scroll

    def _build_milling_tab(self) -> QWidget:
        if self.microscope is None:
            return self._placeholder("No microscope connected")

        from fibsem.ui.widgets.milling_task_viewer_widget import MillingTaskViewerWidget
        from fibsem.ui.widgets.fluorescence_coincidence_widget import (
            milling_task_config as DEFAULT_MILLING_TASK_CONFIG,
        )

        self.milling_viewer_widget = MillingTaskViewerWidget(
            microscope=self.microscope,
            viewer=self.viewer,
            milling_task_config=DEFAULT_MILLING_TASK_CONFIG,
            milling_enabled=False,
            parent=self,
        )
        self.milling_viewer_widget.set_parameters_visible(False)
        self.milling_viewer_widget.set_alignment_visible(False)
        self.milling_viewer_widget.set_acquisition_visible(False)
        return self.milling_viewer_widget

    def _build_fm_tab(self) -> QWidget:
        if self.microscope is None or self.microscope.fm is None:
            return self._placeholder("No fluorescence microscope connected")

        from fibsem.fm.structures import ChannelSettings
        from fibsem.ui.fm.widgets import (
            CameraWidget,
            FluorescenceMultiChannelWidget,
            ObjectiveControlWidget,
        )

        fm = self.microscope.fm

        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Objective control
        from fibsem.ui.widgets.custom_widgets import IconToolButton

        self.fm_objective_widget = ObjectiveControlWidget(fm=fm)
        btn_refresh_objective = IconToolButton(
            icon="mdi:refresh", tooltip="Refresh objective position"
        )
        btn_refresh_objective.clicked.connect(
            lambda: self.fm_objective_widget.update_objective_position_labels(None)
        )
        objective_panel = TitledPanel(
            "Objective", content=self.fm_objective_widget, collapsible=True
        )
        objective_panel.add_header_widget(btn_refresh_objective)
        objective_panel.collapse()
        layout.addWidget(objective_panel)

        # Camera settings (collapsed by default)
        self.fm_camera_widget = CameraWidget(fm=fm)
        camera_panel = TitledPanel(
            "Camera", content=self.fm_camera_widget, collapsible=True
        )
        camera_panel.collapse()
        layout.addWidget(camera_panel)

        # Channel settings
        self.fm_channel_widget = FluorescenceMultiChannelWidget(
            fm=fm,
            channel_settings=[ChannelSettings()],
        )
        channel_panel = TitledPanel(
            "Channel", content=self.fm_channel_widget, collapsible=True
        )
        channel_panel.expand()
        layout.addWidget(channel_panel)

        # Acquire buttons
        self.btn_acquire_fm = QPushButton("Acquire Fluorescence Image")
        self.btn_acquire_fm.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.btn_toggle_fm_acquisition = QPushButton("Start Acquisition")
        self.btn_toggle_fm_acquisition.setStyleSheet(
            stylesheets.SECONDARY_BUTTON_STYLESHEET
        )
        layout.addWidget(self.btn_acquire_fm)
        layout.addWidget(self.btn_toggle_fm_acquisition)
        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(scroll_content)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # type: ignore[attr-defined]
        return scroll

    @staticmethod
    def _placeholder(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        lbl.setStyleSheet("color: #888; font-style: italic; padding: 16px;")
        return lbl

    # ------------------------------------------------------------------
    # Bottom bar
    # ------------------------------------------------------------------

    def _build_bottom_bar(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)

        self.label_selected_lamella = QLabel("Lamella: None")
        self.label_selected_lamella.setStyleSheet("color: #d1d2d4; font-weight: bold;")
        layout.addWidget(self.label_selected_lamella)

        layout.addStretch()

        self.progressBar_stages = QProgressBar()
        self.progressBar_stages.setStyleSheet(
            stylesheets.MILLING_PROGRESS_BAR_STYLESHEET
        )
        self.progressBar_stages.setFixedHeight(24)
        self.progressBar_stages.setFixedWidth(180)
        self.progressBar_stages.setVisible(False)

        self.progressBar_stage = QProgressBar()
        self.progressBar_stage.setStyleSheet(
            stylesheets.MILLING_PROGRESS_BAR_STYLESHEET
        )
        self.progressBar_stage.setFixedHeight(24)
        self.progressBar_stage.setFixedWidth(180)
        self.progressBar_stage.setVisible(False)

        self.btn_milling = QPushButton("Start Milling")
        self.btn_milling.setIcon(
            fibsem_icon("mdi:play-circle", color=stylesheets.GRAY_ICON_COLOR)
        )
        self.btn_milling.setStyleSheet(stylesheets.RUN_WORKFLOW_BUTTON_STYLESHEET)

        self.btn_pause_milling = QPushButton("Pause Milling")
        self.btn_pause_milling.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.btn_pause_milling.setVisible(False)

        self.label_threshold_chip = QPushButton("Threshold Reached")
        self.label_threshold_chip.setIcon(
            fibsem_icon("mdi:alert-circle", color=stylesheets.GRAY_ICON_COLOR)
        )
        self.label_threshold_chip.setStyleSheet(
            stylesheets.USER_ATTENTION_BUTTON_STYLESHEET
        )
        self.label_threshold_chip.setVisible(False)

        self.btn_pause_acquisition = QPushButton("Pause Acquisition")
        self.btn_pause_acquisition.setStyleSheet(
            stylesheets.SECONDARY_BUTTON_STYLESHEET
        )
        self.btn_pause_acquisition.setVisible(False)

        # supervision toggle, styled like the main-UI supervised status button
        self._supervised: bool = True
        self.btn_supervised = QPushButton("Supervised")
        self.btn_supervised.setCursor(Qt.PointingHandCursor)  # type: ignore[attr-defined]
        self.btn_supervised.clicked.connect(self._on_supervised_clicked)
        self._update_supervised_button()

        self.spin_drop_threshold = QSpinBox()
        self.spin_drop_threshold.setRange(5, 90)
        self.spin_drop_threshold.setValue(40)  # % drop (40% == retained fraction 0.6)
        self.spin_drop_threshold.setPrefix("Stop at ")
        self.spin_drop_threshold.setSuffix("% drop")
        self.spin_drop_threshold.setToolTip(
            "Unsupervised auto-stop fires when the rolling mean drops by this "
            "fraction below its peak."
        )

        # push live changes to any running strategies (not just at milling start)
        self.spin_drop_threshold.valueChanged.connect(self._on_drop_threshold_changed)

        layout.addWidget(self.label_threshold_chip)
        layout.addWidget(self.progressBar_stages)
        layout.addWidget(self.progressBar_stage)
        layout.addWidget(self.btn_supervised)
        layout.addWidget(self.spin_drop_threshold)
        layout.addWidget(self.btn_pause_milling)
        layout.addWidget(self.btn_pause_acquisition)
        layout.addWidget(self.btn_milling)

        return bar

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self):
        if self.microscope is None:
            return
        if self.microscope.fm is None:
            return
        # Canvas click → selected view
        self.fib_canvas.canvas.canvas_clicked.connect(
            lambda *_: self._set_selected_view(1)
        )
        self.fm_canvas.canvas.canvas_clicked.connect(
            lambda *_: self._set_selected_view(0)
        )

        # Canvas double-click → stage movement
        self.fib_canvas.canvas.canvas_double_clicked.connect(
            self._on_fib_double_clicked
        )
        self.fm_canvas.canvas.canvas_double_clicked.connect(self._on_fm_double_clicked)

        # Canvas → info panel + strategy bbox
        self.fib_canvas.rect_overlay.rect_changed.connect(self._info_widget.update_fib)
        self.fm_canvas.rect_overlay.rect_changed.connect(self._info_widget.update_fm)
        self.fm_canvas.rect_overlay.rect_changed.connect(self._on_fm_rect_changed)

        # Strategy intensity → line plot (thread-safe: psygnal → pyqtSignal → main thread)
        self._intensity_ready.connect(self.line_plot_widget.append_value)

        # Timelapse scrubber
        self.fm_canvas.scrub_requested.connect(self._on_fm_scrub)
        self.fm_canvas.reset_live_requested.connect(self._on_reset_live_view)

        # Thread-safe canvas updates
        self._fib_image_ready.connect(self.set_fib_image)
        self._fm_image_ready.connect(self.set_fm_image)
        self._fib_acquire_done.connect(self._on_fib_acquire_done)
        self._fm_acquire_done.connect(self._on_fm_acquire_done)

        # Lamella list
        self.lamella_list_widget.lamella_selected.connect(self._on_lamella_selected)
        self.lamella_list_widget.move_to_requested.connect(self._on_move_to_lamella)

        # FIB acquire / auto-function buttons
        self.btn_acquire_fib.clicked.connect(self._acquire_fib_image)
        self.btn_autocontrast_fib.clicked.connect(self._run_fib_autocontrast)
        self.btn_autofocus_fib.clicked.connect(self._run_fib_autofocus)

        # FM acquire buttons
        self.btn_acquire_fm.clicked.connect(self._acquire_fm_image)
        self.btn_toggle_fm_acquisition.clicked.connect(self._toggle_fm_acquisition)
        self.fm_channel_widget.channel_field_changed.connect(
            self._on_channel_field_changed
        )

        # Bottom bar buttons
        self.btn_milling.clicked.connect(self._toggle_milling)
        self.btn_pause_acquisition.clicked.connect(self._toggle_fm_acquisition_pause)
        if self.milling_viewer_widget is not None:
            self.btn_pause_milling.clicked.connect(
                self.milling_viewer_widget.milling_widget.pause_resume_milling
            )
            # finalize the viewer only when milling is *fully* complete (after the
            # post-stop final image) — the progress "finished" state fires too early
            self.milling_viewer_widget.milling_widget.milling_completed_signal.connect(
                self._finalize_milling_ui
            )
            # FIB rect ↔ milling pattern sync
            self.fib_canvas.rect_overlay.rect_changed.connect(self._on_fib_rect_changed)
            sw = self.milling_viewer_widget.config_widget.milling_stages_widget
            sw._list.stage_selected.connect(self._update_fib_rect_from_pattern)
            self.milling_viewer_widget.config_widget.settings_changed.connect(
                self._update_fib_rect_from_pattern
            )

        # Live FM acquisition → FM canvas
        self.microscope.fm.acquisition_signal.connect(self._on_fm_live_image)

        # FIB acquisition signal (e.g. post-milling) → FIB canvas
        self.microscope.fib_acquisition_signal.connect(
            lambda image: self._fib_image_ready.emit(image)
        )

        # Milling progress
        self.microscope.milling_progress_signal.connect(self._on_milling_progress)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_lamella_selected(self, lamella: Optional["Lamella"]):
        self._selected_lamella = lamella
        name = lamella.name if lamella is not None else "None"
        self.label_selected_lamella.setText(f"Lamella: {name}")
        if getattr(self, "selected_lamella_widget", None) is not None:
            self.selected_lamella_widget.set_lamella(lamella)
        self._reset_timelapse()
        self.lamella_selected_signal.emit(lamella)

    # ------------------------------------------------------------------
    # Selected-lamella widget handlers (objective + poses), self-contained
    # ------------------------------------------------------------------

    def _on_slw_objective_position_changed(self, value_um: float):
        """Store the edited objective position on the selected lamella."""
        lamella = self._selected_lamella
        if lamella is None or lamella.fluorescence_pose is None:
            return
        lamella.fluorescence_pose.objective_position = value_um * MICRON_TO_METRE
        if self.experiment is not None:
            self.experiment.save()

    def _on_slw_use_current_objective(self):
        """Read the current FM objective position and store it on the lamella."""
        if self.microscope is None or self.microscope.fm is None:
            notification_service.show_toast("No microscope connected.", "warning")
            return
        lamella = self._selected_lamella
        if lamella is None or lamella.fluorescence_pose is None:
            notification_service.show_toast("No lamella selected.", "warning")
            return
        obj = self.microscope.fm.objective
        value_m = obj.position if obj.state == "Inserted" else obj.focus_position
        if value_m is None:
            notification_service.show_toast("Objective position unavailable.", "warning")
            return
        lamella.fluorescence_pose.objective_position = value_m
        if self.experiment is not None:
            self.experiment.save()
        self.selected_lamella_widget.set_lamella(lamella)
        notification_service.show_toast(
            f"Set objective position to {value_m * METRE_TO_MICRON:.1f} µm "
            f"for {lamella.name}.",
            "info",
        )

    def _on_slw_apply_objective_to_all(self):
        """Copy the selected lamella's objective position to all lamella with an FM pose."""
        if self.experiment is None:
            return
        value_um = self.selected_lamella_widget.objective_value_um()
        value_m = value_um * MICRON_TO_METRE
        count = 0
        for lamella in self.experiment.positions:
            if lamella.fluorescence_pose is not None:
                lamella.fluorescence_pose.objective_position = value_m
                count += 1
        if count:
            self.experiment.save()
            notification_service.show_toast(
                f"Applied objective position ({value_um:.1f} µm) to {count} lamella.",
                "info",
            )

    def _on_slw_move_objective(self):
        """Move the FM objective to the selected lamella's stored objective position."""
        if self.microscope is None or self.microscope.fm is None:
            notification_service.show_toast("No microscope connected.", "warning")
            return
        lamella = self._selected_lamella
        if lamella is None or lamella.fluorescence_pose is None:
            notification_service.show_toast("No lamella selected.", "warning")
            return
        objective_position = lamella.fluorescence_pose.objective_position
        if objective_position is None:
            notification_service.show_toast(
                f"{lamella.name} has no stored objective position.", "warning"
            )
            return
        obj = self.microscope.fm.objective
        if obj.state != "Inserted":
            notification_service.show_toast(
                "Insert the objective before moving to a stored position.", "warning"
            )
            return

        ret = QMessageBox.question(
            self,
            "Move Objective",
            f"Move objective to {objective_position * METRE_TO_MICRON:.1f} µm "
            f"for {lamella.name}?",
            QMessageBox.Yes | QMessageBox.No,  # type: ignore[attr-defined]
        )
        if ret != QMessageBox.Yes:  # type: ignore[attr-defined]
            return

        def _move():
            try:
                obj.move_absolute(objective_position)
            except Exception:
                logging.exception("Error moving objective to stored position")

        threading.Thread(target=_move, daemon=True).start()

    def _on_slw_pose_move_to(self, pose_name: str):
        """Move the stage to the given pose of the selected lamella."""
        lamella = self._selected_lamella
        if lamella is None or self.microscope is None:
            return
        pose = lamella.poses.get(pose_name)
        if pose is None or pose.stage_position is None:
            notification_service.show_toast(
                f"Pose '{pose_name}' has no stage position.", "warning"
            )
            return

        ret = QMessageBox.question(
            self,
            "Move to Pose",
            f"Move to pose '{pose_name}' for {lamella.name}?\n{pose.stage_position.pretty}",
            QMessageBox.Yes | QMessageBox.No,  # type: ignore[attr-defined]
        )
        if ret != QMessageBox.Yes:  # type: ignore[attr-defined]
            return

        stage_position = pose.stage_position

        def _move():
            try:
                self.microscope.safe_absolute_stage_movement(stage_position)
            except Exception:
                logging.exception("Error moving to pose position")

        threading.Thread(target=_move, daemon=True).start()

    def _on_slw_pose_update(self, pose_name: str):
        """Set the current stage position as the given pose of the selected lamella."""
        if self.microscope is None:
            notification_service.show_toast("No microscope connected.", "warning")
            return
        lamella = self._selected_lamella
        if lamella is None or pose_name == "":
            notification_service.show_toast("No lamella selected.", "warning")
            return
        state = self.microscope.get_microscope_state()
        if state is None or state.stage_position is None:
            notification_service.show_toast("Failed to get microscope state.", "warning")
            return

        ret = QMessageBox.question(
            self,
            "Set Pose",
            f"Set current position as pose '{pose_name}' for {lamella.name}?\n"
            f"{state.stage_position.pretty}",
            QMessageBox.Yes | QMessageBox.No,  # type: ignore[attr-defined]
        )
        if ret != QMessageBox.Yes:  # type: ignore[attr-defined]
            return

        # preserve the existing pose's objective position (state does not capture it)
        existing_pose = lamella.poses.get(pose_name)
        if existing_pose is not None and existing_pose.objective_position is not None:
            state.objective_position = existing_pose.objective_position

        lamella.poses[pose_name] = state
        if self.experiment is not None:
            self.experiment.save()
        self.selected_lamella_widget.refresh_pose(
            pose_name, state.stage_position.pretty
        )

    def _on_move_to_lamella(self, lamella: Optional["Lamella"]):
        if lamella is None or self.microscope is None:
            return

        ret = QMessageBox.question(
            self,
            "Move to Lamella Position",
            f"Move to position of Lamella {lamella.name}?\n{lamella.stage_position.pretty}",
            QMessageBox.Yes | QMessageBox.No,  # type: ignore[attr-defined]
        )
        if ret != QMessageBox.Yes:  # type: ignore[attr-defined]
            return

        def _move():
            try:
                self.microscope.safe_absolute_stage_movement(lamella.stage_position)
            except Exception:
                logging.exception("Error moving to lamella position")

        threading.Thread(target=_move, daemon=True).start()

    def _acquire_fib_image(self):
        """Acquire a FIB image using current microscope settings and display it."""
        self._set_fib_buttons_enabled(False)
        self.btn_acquire_fib.setText("Acquiring…")

        def _worker():
            try:
                image = self.microscope.acquire_image(beam_type=BeamType.ION)
                self._fib_image_ready.emit(image)
            except Exception:
                logging.exception("Error acquiring FIB image")
            finally:
                self._fib_acquire_done.emit()

        threading.Thread(target=_worker, daemon=True).start()

    def _acquire_fm_image(self):
        """Acquire a single FM image in a background thread and display it."""
        from fibsem.fm.acquisition import acquire_image as fm_acquire_image

        channel = self.fm_channel_widget.selected_channel
        if channel is None:
            QMessageBox.warning(
                self, "No Channel", "Select an FM channel before acquiring."
            )
            return

        self.btn_acquire_fm.setEnabled(False)
        self.btn_acquire_fm.setText("Acquiring…")

        def _worker():
            try:
                image = fm_acquire_image(
                    microscope=self.microscope.fm, channel_settings=channel
                )
                if image is not None:
                    self._fm_image_ready.emit(image)
            except Exception:
                logging.exception("Error acquiring FM image")
            finally:
                self._fm_acquire_done.emit()

        threading.Thread(target=_worker, daemon=True).start()

    def _set_fib_buttons_enabled(self, enabled: bool) -> None:
        for btn in [
            self.btn_acquire_fib,
            self.btn_autocontrast_fib,
            self.btn_autofocus_fib,
        ]:
            btn.setEnabled(enabled)

    def _run_fib_autocontrast(self) -> None:
        from fibsem.structures import FibsemRectangle

        self._set_fib_buttons_enabled(False)
        self.btn_autocontrast_fib.setText("Running…")

        def _worker():
            try:
                self.microscope.autocontrast(
                    BeamType.ION,
                    reduced_area=FibsemRectangle(
                        left=0.25, top=0.25, width=0.5, height=0.5
                    ),
                )
                image = self.microscope.acquire_image(beam_type=BeamType.ION)
                self._fib_image_ready.emit(image)
            except Exception:
                logging.exception("Error running FIB autocontrast")
            finally:
                self._fib_acquire_done.emit()
                self.btn_autocontrast_fib.setText("AutoContrast")

        threading.Thread(target=_worker, daemon=True).start()

    def _run_fib_autofocus(self) -> None:
        from fibsem.structures import FibsemRectangle

        self._set_fib_buttons_enabled(False)
        self.btn_autofocus_fib.setText("Running…")

        def _worker():
            try:
                self.microscope.auto_focus(
                    BeamType.ION,
                    reduced_area=FibsemRectangle(
                        left=0.25, top=0.25, width=0.5, height=0.5
                    ),
                )
                image = self.microscope.acquire_image(beam_type=BeamType.ION)
                self._fib_image_ready.emit(image)
            except Exception:
                logging.exception("Error running FIB autofocus")
            finally:
                self._fib_acquire_done.emit()
                self.btn_autofocus_fib.setText("AutoFocus")

        threading.Thread(target=_worker, daemon=True).start()

    def _on_fib_acquire_done(self):
        self._set_fib_buttons_enabled(True)
        self.btn_acquire_fib.setText("Acquire FIB Image")

    def _on_fm_acquire_done(self):
        self.btn_acquire_fm.setEnabled(True)
        self.btn_acquire_fm.setText("Acquire Fluorescence Image")

    def _on_fm_live_image(self, image):
        """Slot for live FM acquisition — routes each frame to the FM canvas and line plot."""
        if not self._is_scrubbing:
            self._fm_image_ready.emit(image)
        self._compute_and_emit_intensity(image)
        self._maybe_accumulate_timelapse(image)

    def _compute_and_emit_intensity(self, image) -> None:
        """Crop to the FM rectangle overlay, compute mean intensity, emit to line plot."""
        try:
            data = image.data
            if data is None:
                return
            # Flatten to 2-D (max-project channels/z if needed)
            while data.ndim > 2:
                data = data.max(axis=0)
            shape = self.fm_canvas._img_shape
            if shape is not None:
                H, W = shape
                info = self.fm_canvas.rect_overlay.get_rect()
                x0 = int(info["x0"])
                y0 = int(info["y0"])
                x1 = int(info["x1"])
                y1 = int(info["y1"])
                if x1 > x0 and y1 > y0:
                    data = data[y0:y1, x0:x1]
            mean_val = float(np.mean(data))
            if not np.isnan(mean_val):
                self._intensity_ready.emit(mean_val)
        except Exception:
            logging.exception("Error computing FM intensity")

    # ------------------------------------------------------------------
    # Timelapse accumulation
    # ------------------------------------------------------------------

    TIMELAPSE_INTERVAL: float = 10.0  # seconds between stored frames
    TIMELAPSE_MAX_PX: int = 512  # downsample target for stored frames

    def _maybe_accumulate_timelapse(self, image: FluorescenceImage) -> None:
        """Accumulate one frame into the timelapse buffer every TIMELAPSE_INTERVAL seconds."""
        now = time.time()
        if now - self._last_timelapse_time < self.TIMELAPSE_INTERVAL:
            return
        self._last_timelapse_time = now

        try:
            data = image.data
            # Normalise to CZYX
            if data.ndim == 5:
                data = data[0]  # TCZYX → CZYX
            if data.ndim == 2:
                data = data[None, None]  # YX → 1,1,Y,X
            elif data.ndim == 3:
                data = data[:, None]  # CYX → C,1,Y,X
            # Max-project C and Z → 2D
            frame = np.max(data, axis=(0, 1)).astype(np.float32)

            # Stride-based downsample
            h, w = frame.shape
            max_px = self.TIMELAPSE_MAX_PX
            if h > max_px or w > max_px:
                stride = max(1, math.ceil(max(h, w) / max_px))
                frame = frame[::stride, ::stride]

            self._timelapse_frames.append(frame)
            self._timelapse_timestamps.append(now)

            n = len(self._timelapse_frames)
            self.fm_canvas.set_timelapse_length(n)
            # Advance slider to latest without triggering scrub (live display stays untouched)
            self.fm_canvas.time_slider.blockSignals(True)
            self.fm_canvas.time_slider.setValue(n - 1)
            self.fm_canvas.time_slider.blockSignals(False)
            self.fm_canvas.frame_label.setText(
                f"{_fmt_timestamp(now)} ({n - 1}/{n - 1})"
            )
        except Exception:
            logging.exception("Error accumulating timelapse frame")

    def _on_fm_scrub(self, idx: int) -> None:
        """Display the timelapse frame at *idx* and toggle scrubbing mode."""
        if not self._timelapse_frames:
            return
        total = len(self._timelapse_frames)
        idx = max(0, min(idx, total - 1))
        self._is_scrubbing = idx < total - 1
        ts = self._timelapse_timestamps[idx]
        self.fm_canvas.display_timelapse_frame(
            self._timelapse_frames[idx], idx, total, ts
        )
        self.line_plot_widget.set_scrub_timestamp(ts)

    def _on_reset_live_view(self) -> None:
        """Return to live view — resume displaying incoming FM frames."""
        self._is_scrubbing = False
        self.line_plot_widget.set_scrub_timestamp(None)
        n = len(self._timelapse_frames)
        if n > 0:
            self.fm_canvas.time_slider.blockSignals(True)
            self.fm_canvas.time_slider.setValue(n - 1)
            self.fm_canvas.time_slider.blockSignals(False)
            self.fm_canvas.frame_label.setText(
                f"{_fmt_timestamp(self._timelapse_timestamps[-1])} ({n - 1}/{n - 1})"
            )

    def _reset_timelapse(self) -> None:
        """Clear the timelapse buffer and hide the scrubber."""
        self._timelapse_frames.clear()
        self._timelapse_timestamps.clear()
        self._last_timelapse_time = 0.0
        self._is_scrubbing = False
        self.fm_canvas.set_timelapse_length(0)
        self.line_plot_widget.set_scrub_timestamp(None)
        self.line_plot_widget.update_stats_lines(0.0, 0.0, False)
        self._info_widget.clear_stats()
        self.label_threshold_chip.setVisible(False)

    @ensure_main_thread
    def _on_intensity_stats(self, stats: dict) -> None:
        """Update line plot, stats lines, info panel, and drop chip from live strategy stats."""
        self.line_plot_widget.append_value(stats["value"])
        self.line_plot_widget.update_stats_lines(
            peak_rolling_mean=stats.get("peak_rolling_mean", 0.0),
            threshold_value=stats.get("threshold_value", 0.0),
            warmup_complete=stats.get("warmup_complete", False),
        )
        self._info_widget.update_stats(stats)
        self._info_widget.update_run_metrics(stats)
        if stats.get("drop_detected"):
            self._set_border_state("waiting")
            drop_pct = (1.0 - stats.get("drop_fraction", 1.0)) * 100
            self.label_threshold_chip.setText(f"Intensity Drop: {drop_pct:.0f}%")
            self.label_threshold_chip.setVisible(True)

    def _on_fm_rect_changed(self, info: dict):
        """Push the current FM rectangle to any active coincidence strategies as a bbox."""
        if not self._active_strategies:
            return
        shape = self.fm_canvas._img_shape
        if shape is None:
            return
        from fibsem.structures import FibsemRectangle

        H, W = shape
        w, h = info["width"], info["height"]
        # Zero-size rect (overlay not yet drawn) → None means "use full image"
        rect = (
            FibsemRectangle(
                left=info["x0"] / W,
                top=info["y0"] / H,
                width=w / W,
                height=h / H,
            )
            if w > 0 and h > 0
            else None
        )
        for strategy in self._active_strategies:
            try:
                strategy._on_bbox_update(rect)
            except Exception:
                logging.exception("Error updating strategy bbox")

    def _connect_coincidence_strategies(
        self, milling_task_config: "FibsemMillingTaskConfig"
    ) -> None:
        """Find CoincidenceMillingStrategy instances and seed their bbox from the FM rectangle."""
        from fibsem.milling.strategy.coincidence import CoincidenceMillingStrategy

        self._active_strategies: list[CoincidenceMillingStrategy] = [
            stage.strategy
            for stage in milling_task_config.enabled_stages
            if isinstance(stage.strategy, CoincidenceMillingStrategy)
        ]

        # Seed bbox from the current FM rectangle position if image is loaded
        if self._active_strategies and self.fm_canvas._img_shape is not None:
            try:
                info = self.fm_canvas.rect_overlay.get_rect()
                if info:
                    self._on_fm_rect_changed(info)
            except Exception:
                pass

        # Seed the status-bar controls FROM the strategy config (single source of
        # truth), then connect the live stats signal. Live edits push back below.
        self._seed_controls_from_strategy()
        for strategy in self._active_strategies:
            strategy.intensity_stats_signal.connect(self._on_intensity_stats)

    def _seed_controls_from_strategy(self) -> None:
        """Mirror the first active strategy's config into the status-bar controls.

        Keeps ``strategy.config`` authoritative: the status bar reflects the value
        set in the strategy section rather than overwriting it with its own default.
        """
        if not self._active_strategies:
            return
        config = self._active_strategies[0].config
        self._supervised = bool(config.supervised)
        self._update_supervised_button()
        self.spin_drop_threshold.blockSignals(True)
        self.spin_drop_threshold.setValue(
            int(round(config.intensity_drop_fraction * 100))
        )
        self.spin_drop_threshold.blockSignals(False)

    def _update_supervised_button(self) -> None:
        """Style the supervision button to match the current mode (like the main UI)."""
        if self._supervised:
            self.btn_supervised.setText("Supervised")
            self.btn_supervised.setIcon(
                fibsem_icon("mdi:account-hard-hat", color="white")
            )
            self.btn_supervised.setStyleSheet(
                stylesheets.SUPERVISION_STATUS_SUPERVISED_STYLESHEET
            )
            self.btn_supervised.setToolTip(
                "Supervised: stop milling manually. Click to switch to unsupervised "
                "(auto-stop on intensity drop)."
            )
        else:
            self.btn_supervised.setText("Automated")
            self.btn_supervised.setIcon(
                fibsem_icon("mdi:lightning-bolt", color="white")
            )
            self.btn_supervised.setStyleSheet(
                stylesheets.SUPERVISION_STATUS_AUTOMATED_STYLESHEET
            )
            self.btn_supervised.setToolTip(
                "Unsupervised: automatically stops on an intensity drop. "
                "Click to switch to supervised."
            )

    def _on_supervised_clicked(self) -> None:
        """Toggle supervision mode from the button."""
        self._set_supervised(not self._supervised)

    def _set_supervised(self, supervised: bool) -> None:
        """Apply the supervision mode to the button, active strategies, and border."""
        self._supervised = supervised
        self._update_supervised_button()
        self._on_supervised_toggled(supervised)
        if self._is_milling_active:
            self._set_border_state("supervised" if supervised else "automated")

    def _on_supervised_toggled(self, supervised: bool) -> None:
        """Apply the supervision preference to any active strategies (live)."""
        for strategy in self._active_strategies:
            strategy.config.supervised = supervised

    def _on_drop_threshold_changed(self, pct: int) -> None:
        """Apply the drop-fraction threshold to any active strategies (live)."""
        drop_fraction = pct / 100.0
        for strategy in self._active_strategies:
            strategy.config.intensity_drop_fraction = drop_fraction

    def _toggle_fm_acquisition(self):
        """Start or stop continuous FM acquisition, updating the FM canvas on each frame."""
        if self.microscope is None or self.microscope.fm is None:
            return
        fm = self.microscope.fm
        if fm.is_acquiring:
            fm.stop_acquisition()
            self.btn_toggle_fm_acquisition.setText("Start Acquisition")
            self.btn_toggle_fm_acquisition.setStyleSheet(
                stylesheets.SECONDARY_BUTTON_STYLESHEET
            )
        else:
            selected_channel_settings = self.fm_channel_widget.selected_channel
            if selected_channel_settings is None:
                QMessageBox.warning(
                    self,
                    "No Channel",
                    "Select an FM channel before starting acquisition.",
                )
                return
            fm.start_acquisition(channel_settings=selected_channel_settings)
            self.btn_toggle_fm_acquisition.setText("Stop Acquisition")
            self.btn_toggle_fm_acquisition.setStyleSheet(
                stylesheets.STOP_WORKFLOW_BUTTON_STYLESHEET
            )

    def _toggle_fm_acquisition_pause(self):
        """Pause or resume FM acquisition during milling."""
        if self.microscope is None or self.microscope.fm is None:
            return
        fm = self.microscope.fm
        if fm.is_acquiring:
            fm.stop_acquisition()
            self.btn_pause_acquisition.setText("Resume Acquisition")
            self.btn_pause_acquisition.setStyleSheet(
                stylesheets.RUN_WORKFLOW_BUTTON_STYLESHEET
            )
        else:
            fm.start_acquisition()
            self.btn_pause_acquisition.setText("Pause Acquisition")
            self.btn_pause_acquisition.setStyleSheet(
                stylesheets.SECONDARY_BUTTON_STYLESHEET
            )

    def _on_channel_field_changed(self, channel, field: str, value) -> None:
        """Update a single FM parameter live during acquisition (mirrors FMControlWidget)."""
        if self.microscope is None or self.microscope.fm is None:
            return
        fm = self.microscope.fm
        if not fm.is_acquiring:
            return
        if channel is not self.fm_channel_widget.selected_channel:
            return
        logging.info(f"Channel field changed: {field} -> {value}")
        if field == "excitation_wavelength":
            fm.filter_set.excitation_wavelength = value
        elif field == "emission_wavelength":
            fm.filter_set.emission_wavelength = value
        elif field == "exposure_time":
            fm.set_exposure_time(value)
        elif field == "gain":
            fm.set_gain(value)
        elif field == "power":
            fm.set_power(value)
        elif field == "color":
            fm.set_channel_color(value)

    def _toggle_milling(self):
        """Start milling if idle, stop if running."""
        if self.btn_milling.text() == "Stop Milling":
            self._set_border_state("stopped")
            if self.milling_viewer_widget is not None:
                self.milling_viewer_widget.milling_widget.stop_milling()
        else:
            self._run_milling()

    def _run_milling(self):
        """Validate, confirm, then start the milling task."""
        if self._selected_lamella is None:
            QMessageBox.critical(
                self,
                "Error",
                "No lamella selected. Please select a lamella before starting milling.",
            )
            return

        if self.milling_viewer_widget is None:
            QMessageBox.critical(self, "Error", "Milling widget not initialized.")
            return

        # Remap acquisition path to the selected lamella's directory
        acq_widget = self.milling_viewer_widget.config_widget.acquisition_widget
        acq_widget.image_settings_widget.path_edit.setText(
            str(self._selected_lamella.path)
        )
        acq_widget._emit_settings_changed()

        milling_task_config = self.milling_viewer_widget.get_config()

        # Override field of view with the live FIB field of view
        if self.microscope is not None:
            milling_task_config.field_of_view = self.microscope.get_field_of_view(
                beam_type=BeamType.ION
            )

        if not milling_task_config.enabled_stages:
            QMessageBox.critical(self, "Error", "No enabled milling stages configured.")
            return

        # FM channel check (only required when FM is connected)
        selected_channel_settings = None
        if self.microscope is not None and self.microscope.fm is not None:
            if self.fm_control_widget is not None:
                selected_channel_settings = (
                    self.fm_control_widget.channelSettingsWidget.selected_channel
                )
                if selected_channel_settings is None:
                    QMessageBox.critical(
                        self,
                        "Error",
                        "No FM channel selected. Please select a channel in the FM tab.",
                    )
                    return

        # Build confirmation summary
        lines = [f"Lamella: {self._selected_lamella.name}"]
        if selected_channel_settings is not None:
            lines.append(f"Channel: {selected_channel_settings.pretty}")
        lines.append(f"Field of View: {milling_task_config.field_of_view * 1e6:.1f} µm")
        stage_names = [s.pretty_name for s in milling_task_config.stages if s.enabled]
        if stage_names:
            lines.append("Stages: " + ", ".join(stage_names))

        dlg = QMessageBox(self)
        dlg.setWindowTitle("Confirm Coincidence Milling")
        dlg.setIcon(QMessageBox.Question)
        dlg.setText("Review milling parameters before starting.")
        dlg.setInformativeText("\n".join(lines))
        dlg.setDetailedText(
            pformat(milling_task_config.to_dict(), width=80, compact=True)
        )
        dlg.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        dlg.setDefaultButton(QMessageBox.Yes)
        if dlg.exec_() != QMessageBox.Yes:
            return

        if selected_channel_settings is not None and self.microscope is not None:
            self.microscope.fm.set_channel(selected_channel_settings)

        # Wire coincidence strategies using the same config object passed to run_milling
        self._connect_coincidence_strategies(milling_task_config)
        self._reset_timelapse()

        self._is_milling_active = True
        self._set_widgets_enabled(False)
        self.milling_started_signal.emit()
        self.milling_viewer_widget.milling_widget.run_milling(
            config=milling_task_config
        )

    @ensure_main_thread
    def _on_milling_progress(self, progress: dict):
        progress_info: dict = progress.get("progress", {})
        state = progress_info.get("state")

        if state == "start":
            self._set_border_state(
                "supervised" if self._supervised else "automated"
            )
            current_stage = progress_info.get("current_stage", 0)
            total_stages = progress_info.get("total_stages", 1)
            msg = progress.get("msg", "Preparing...")
            self.progressBar_stage.setRange(0, 100)
            self.progressBar_stage.setValue(0)
            self.progressBar_stage.setFormat(msg)
            self.progressBar_stage.setVisible(True)
            self.progressBar_stages.setRange(0, 100)
            self.progressBar_stages.setValue(
                int((current_stage + 1) / total_stages * 100)
            )
            self.progressBar_stages.setFormat(
                f"Stage {current_stage + 1}/{total_stages}"
            )
            self.progressBar_stages.setVisible(True)
            self.btn_milling.setText("Stop Milling")
            self.btn_milling.setIcon(
                fibsem_icon("mdi:stop-circle", color=stylesheets.GRAY_ICON_COLOR)
            )
            self.btn_milling.setStyleSheet(stylesheets.STOP_WORKFLOW_BUTTON_STYLESHEET)
            self.btn_pause_milling.setVisible(True)
            self.btn_pause_acquisition.setVisible(True)

        elif state == "update":
            remaining = progress_info.get("remaining_time")
            estimated = progress_info.get("estimated_time")
            if remaining is not None and estimated is not None and estimated > 0:
                pct = int((1 - remaining / estimated) * 100)
                self.progressBar_stage.setValue(pct)
                from fibsem.utils import format_duration

                self.progressBar_stage.setFormat(
                    f"{format_duration(remaining)} remaining"
                )

        # NOTE: no "finished" handling here. The progress "finished" state fires
        # before finish_milling + the post-stop final image, so the viewer is kept
        # frozen until the milling widget reports true completion — see
        # _finalize_milling_ui (wired to milling_completed_signal).

    @ensure_main_thread
    def _finalize_milling_ui(self) -> None:
        """Reset the viewer once milling is fully complete (final image acquired).

        Driven by ``FibsemMillingWidget.milling_completed_signal`` rather than the
        progress "finished" state, so the stage/controls stay locked until the
        post-stop final image has actually landed.
        """
        self._set_border_state("idle")
        self.progressBar_stage.setVisible(False)
        self.progressBar_stages.setVisible(False)
        self.btn_milling.setText("Start Milling")
        self.btn_milling.setIcon(
            fibsem_icon("mdi:play-circle", color=stylesheets.GRAY_ICON_COLOR)
        )
        self.btn_milling.setStyleSheet(stylesheets.RUN_WORKFLOW_BUTTON_STYLESHEET)
        self.btn_pause_milling.setVisible(False)
        self.btn_pause_acquisition.setVisible(False)
        self.btn_pause_acquisition.setText("Pause Acquisition")
        self.btn_pause_acquisition.setStyleSheet(
            stylesheets.SECONDARY_BUTTON_STYLESHEET
        )
        self._is_milling_active = False
        self._set_widgets_enabled(True)
        self.label_threshold_chip.setVisible(False)
        self.milling_finished_signal.emit()

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # FIB rect ↔ milling pattern
    # ------------------------------------------------------------------

    def _get_selected_stage(self):
        """Return the currently selected (or first enabled) milling stage, or None."""
        if self.milling_viewer_widget is None:
            return None
        lst = self.milling_viewer_widget.config_widget.milling_stages_widget._list
        if lst._selected_stage is not None:
            return lst._selected_stage
        return next((s for s in lst.get_stages() if s.enabled), None)

    def _update_fib_rect_from_pattern(self, *_) -> None:
        """Snap the FIB rect overlay to the selected stage's pattern position and size."""
        stage = self._get_selected_stage()
        if stage is None:
            return
        pixel_size = self.fib_canvas.canvas._pixel_size
        img_w = self.fib_canvas.canvas.img_width
        img_h = self.fib_canvas.canvas.img_height
        if not pixel_size or not img_w or not img_h:
            return
        from fibsem.structures import FibsemRectangleSettings

        patterns = stage.define_patterns()
        rect_pattern = next(
            (p for p in patterns if isinstance(p, FibsemRectangleSettings)), None
        )
        if rect_pattern is None:
            return
        cx_px = img_w / 2 + rect_pattern.centre_x / pixel_size
        cy_px = img_h / 2 - rect_pattern.centre_y / pixel_size
        w_px = rect_pattern.width / pixel_size
        h_px = rect_pattern.height / pixel_size
        self.fib_canvas.rect_overlay.set_rect(
            cx_px - w_px / 2, cy_px - h_px / 2, w_px, h_px
        )
        self.fib_canvas.set_scan_direction(
            cx_px, cy_px, h_px, getattr(rect_pattern, "scan_direction", "")
        )

    def _on_fib_rect_changed(self, info: dict) -> None:
        """Translate a FIB rect drag into a pattern position update via _move_patterns."""
        if self.milling_viewer_widget is None:
            return
        pixel_size = self.fib_canvas.canvas._pixel_size
        img_w = self.fib_canvas.canvas.img_width
        img_h = self.fib_canvas.canvas.img_height
        if not pixel_size or not img_w or not img_h:
            return
        cx_m = (info["cx"] - img_w / 2) * pixel_size
        cy_m = (info["cy"] - img_h / 2) * -pixel_size  # Y axis is flipped
        self.milling_viewer_widget._move_patterns(Point(cx_m, cy_m), move_all=False)
        self._update_fib_rect_from_pattern()

    # ------------------------------------------------------------------
    # Widget enable/disable + double-click stage movement
    # ------------------------------------------------------------------

    @staticmethod
    def _make_canvas_frame(canvas: QWidget, name: str) -> QFrame:
        """Wrap a canvas in a thin QFrame used for the selection border."""
        frame = QFrame()
        frame.setObjectName(name)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)
        layout.addWidget(canvas)
        return frame

    _BORDER_STATES = [
        "idle",
        "automated",
        "supervised",
        "waiting",
        "finished",
        "stopped",
    ]

    def set_border_enabled(self, enabled: bool) -> None:
        """Enable or disable the workflow border state animation.

        When disabled the border is always shown as 'idle' (invisible).
        """
        self._border_enabled = enabled
        self._apply_border(self._border_state)

    def _set_border_state(self, state: str) -> None:
        """Record *state* and repaint — no-op if border is disabled."""
        self._border_state = state
        self._apply_border(state)

    def _apply_border(self, state: str) -> None:
        """Push *state* (or 'idle' when disabled) to the QFrame property."""
        effective = state if self._border_enabled else "idle"
        self._border_frame.setProperty("borderState", effective)
        style = self._border_frame.style()
        if style:
            style.unpolish(self._border_frame)
            style.polish(self._border_frame)
        self._border_frame.update()

    def _set_selected_view(self, index: int) -> None:
        """Set the active canvas (0 = FM, 1 = FIB) and update border highlights."""
        self.selected_view = index
        _sel = (
            "QFrame#fm_canvas_frame, QFrame#fib_canvas_frame"
            " { border: 1px solid transparent; }"
        )
        _fm = f"QFrame#fm_canvas_frame  {{ border: 1px solid {stylesheets.PRIMARY_COLOR}; }}"
        _fib = f"QFrame#fib_canvas_frame {{ border: 1px solid {stylesheets.PRIMARY_COLOR}; }}"
        self._fm_frame.setStyleSheet(_fm if index == 0 else _sel)
        self._fib_frame.setStyleSheet(_fib if index == 1 else _sel)

    def _set_widgets_enabled(self, enabled: bool) -> None:
        """Enable or disable interactive widgets. FM control stays enabled during milling."""
        for w in [
            self.fib_beam_widget,
            getattr(self, "btn_acquire_fib", None),
            getattr(self, "btn_autocontrast_fib", None),
            getattr(self, "btn_autofocus_fib", None),
            self.milling_viewer_widget.config_widget
            if self.milling_viewer_widget
            else None,
            self.lamella_list_widget,
        ]:
            if w is not None:
                w.setEnabled(enabled)

    def _on_fib_double_clicked(self, x: float, y: float) -> None:
        """Stable-move the stage to the double-clicked position on the FIB canvas."""
        if self._is_milling_active:
            return
        pixel_size = self.fib_canvas.canvas._pixel_size
        img_w = self.fib_canvas.canvas.img_width
        img_h = self.fib_canvas.canvas.img_height
        if not pixel_size or not img_w or not img_h:
            return
        dx = (x - img_w / 2) * pixel_size
        dy = -(y - img_h / 2) * pixel_size
        threading.Thread(
            target=lambda: self.microscope.stable_move(
                dx=dx, dy=dy, beam_type=BeamType.ION
            ),
            daemon=True,
        ).start()

    def _on_fm_double_clicked(self, x: float, y: float) -> None:
        """Stable-move the stage to the double-clicked position on the FM canvas."""
        if self._is_milling_active:
            return
        fm_image = self.fm_canvas._image
        if fm_image is None:
            return
        pixelsize = (
            getattr(fm_image.metadata, "pixel_size_x", None)
            if fm_image.metadata
            else None
        )
        if not pixelsize:
            return
        if not self.microscope.fm.has_valid_orientation():
            logging.info(f"Stage must be in a valid FM orientation to move via FM image (current: {self.microscope.get_stage_orientation()})")
            return
        image_shape = self.fm_canvas._img_shape
        if image_shape is None:
            return
        point = conversions.image_to_microscope_image_coordinates2(
            coord=Point(x=x, y=y),
            image_shape=image_shape,
            pixelsize=pixelsize,
        )
        px, py = point[0], -point[1]  # Y-inversion (mirrors FMControlWidget)
        transform = self.microscope.fm._transform
        if transform is CameraImageTransform.FLIP_X:
            px = -px
        elif transform is CameraImageTransform.FLIP_Y:
            py = -py
        elif transform is CameraImageTransform.FLIP_XY:
            px, py = -px, -py
        elif transform is CameraImageTransform.ROTATE_90_CW:
            px, py = py, -px
        elif transform is CameraImageTransform.ROTATE_90_CCW:
            px, py = -py, px
        elif transform is CameraImageTransform.ROTATE_180:
            px, py = -px, -py
        threading.Thread(
            target=lambda: self.microscope.stable_move(
                dx=px, dy=py, beam_type=BeamType.ELECTRON
            ),
            daemon=True,
        ).start()

    # Public API
    # ------------------------------------------------------------------

    def set_fib_image(self, image: FibsemImage):
        """Display a FIB image in the FIB canvas."""
        self.fib_canvas.set_image(image)
        if self.milling_viewer_widget is not None:
            self.milling_viewer_widget._fib_image = image
        try:
            self._info_widget.set_fib_pixel_size(image.metadata.pixel_size.x)
        except Exception:
            pass
        self._update_fib_rect_from_pattern()

    def set_fm_image(self, image: FluorescenceImage):
        """Display a fluorescence image in the FM canvas."""
        self.fm_canvas.set_image(image)
        try:
            self._info_widget.set_fm_pixel_size(image.metadata.pixel_size_x)
        except Exception:
            pass

    def set_experiment(self, experiment: "Experiment"):
        """Populate the lamella list from *experiment.positions*."""
        self.experiment = experiment
        if experiment is not None:
            self.lamella_list_widget.set_lamella(experiment.positions)
            # set_lamella blocks signals — manually fire selection for the first row
            self._on_lamella_selected(self.lamella_list_widget.selected_lamella)


# ---------------------------------------------------------------------------
# Dialog helper
# ---------------------------------------------------------------------------


def open_coincidence_viewer_dialog(microscope, experiment, viewer=None, parent=None):
    """Open FluorescenceCoincidenceViewerWidget in a non-modal QDialog.

    Returns the dialog (caller should keep a reference to prevent GC).
    """
    from PyQt5.QtWidgets import QDialog, QVBoxLayout

    dlg = QDialog(parent)
    dlg.setWindowTitle("Coincidence Milling Viewer")
    dlg.setWindowFlags(dlg.windowFlags() | Qt.Window)  # type: ignore[attr-defined]
    dlg.resize(1400, 800)

    widget = FluorescenceCoincidenceViewerWidget(
        microscope=microscope, experiment=experiment, viewer=viewer, parent=dlg
    )
    layout = QVBoxLayout(dlg)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(widget)

    dlg.show()
    return dlg


# ---------------------------------------------------------------------------
# Standalone smoke-test
# ---------------------------------------------------------------------------


def main():
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    # app.setStyle("Fusion")

    from fibsem import utils
    from fibsem.config import load_user_preferences
    from fibsem.applications.autolamella.structures import Experiment

    microscope, settings = utils.setup_session()
    import os

    path = load_user_preferences().experiment.last_experiment_path
    exp = Experiment.load(os.path.join(path, "experiment.yaml"))

    widget = FluorescenceCoincidenceViewerWidget(microscope=microscope, experiment=exp)
    widget.setWindowTitle("FluorescenceCoincidenceViewerWidget — preview")
    widget.resize(1400, 800)
    widget.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
