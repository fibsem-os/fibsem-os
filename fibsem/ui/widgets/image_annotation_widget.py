"""Simple widget to preview an image with an optional scalebar."""

from __future__ import annotations

import logging
import os
from typing import Optional
from datetime import datetime

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtGui import QColor

from fibsem.structures import FibsemImage, BeamType
from fibsem.utils import format_value
from scipy.ndimage import median_filter as scipy_median_filter

try:
    from matplotlib_scalebar.scalebar import ScaleBar
except Exception:  # pragma: no cover - optional dependency
    ScaleBar = None  # type: ignore


def _pixel_size_from_metadata(image: FibsemImage) -> Optional[float]:
    if image.metadata is None or getattr(image.metadata, "pixel_size", None) is None:
        return None
    value = getattr(image.metadata.pixel_size, "x", None)
    if value is None or value <= 0:
        return None
    return value


def _get_metadata_text(image: FibsemImage) -> str:

    metadata_text = ""
    beam_name = "N/A"
    beam_type: Optional[BeamType] = None

    if image.metadata and getattr(image.metadata, "image_settings", None):
        beam_type = image.metadata.image_settings.beam_type
        if beam_type is BeamType.ION:
            beam_name = "FIB"
        elif beam_type is BeamType.ELECTRON:
            beam_name = "SEM"

    DATE_FORMAT = "%Y-%m-%d %H:%M:%S %p"
    acq_date = datetime.now().strftime(DATE_FORMAT)
    if image.metadata and getattr(image.metadata, "microscope_state", None):
        ts = image.metadata.microscope_state.timestamp
        if isinstance(ts, (float, int)):
            acq_date = datetime.fromtimestamp(ts).strftime(DATE_FORMAT)
        elif isinstance(ts, str):
            for fmt in ("%m/%d/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                try:
                    dt = datetime.strptime(ts, fmt)
                    acq_date = dt.strftime(DATE_FORMAT)
                    break
                except Exception:
                    continue

    beam_current = beam_voltage = detector_name = None
    if image.metadata and getattr(image.metadata, "microscope_state", None):
        state = image.metadata.microscope_state
        if beam_type is BeamType.ION and getattr(state, "ion_beam", None):
            beam_current = getattr(state.ion_beam, "beam_current", None)
            beam_voltage = getattr(state.ion_beam, "voltage", None)
            detector_name = getattr(getattr(state, "ion_detector", None), "type", None)
        elif beam_type is BeamType.ELECTRON and getattr(state, "electron_beam", None):
            beam_current = getattr(state.electron_beam, "beam_current", None)
            beam_voltage = getattr(state.electron_beam, "voltage", None)
            detector_name = getattr(getattr(state, "electron_detector", None), "type", None)

    beam_current_str = "N/A"
    beam_voltage_str = "N/A"
    if beam_current is not None:
        beam_current_str = format_value(beam_current, "A", precision=0)
    if beam_voltage is not None:
        beam_voltage_str = format_value(beam_voltage, "V", precision=0)
    detector_name = detector_name or "N/A"

    device = None
    if image.metadata and getattr(image.metadata, "system", None):
        device = getattr(getattr(image.metadata.system, "info", None), "model", None)
    device = device or "N/A"

    metadata_text = f"{beam_name} | {beam_voltage_str}, {beam_current_str}, {detector_name} | {acq_date} | {device}"

    return metadata_text


def plot_image_with_scalebar(
    image: FibsemImage,
    show_scalebar: bool,
    apply_median_filter: bool = False,
    show_center_marker: bool = True,
    show_metadata: bool = True,
    metadata_font_size: int = 10,
    marker_color: str = "yellow",
    marker_size: int = 20,
) -> Figure:
    """Return a matplotlib Figure for the image with optional scalebar and filtering."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    data = image.data
    if apply_median_filter:
        try:
            data = scipy_median_filter(data, size=3)
        except Exception as exc:  # pragma: no cover - defensive
            logging.debug("Median filter failed: %s", exc)

    ax.imshow(data, cmap="gray" if data.ndim == 2 else None)

    pixel_size_m = _pixel_size_from_metadata(image)
    if show_scalebar and ScaleBar and pixel_size_m:
        try:
            ax.add_artist(
                ScaleBar(
                    dx=pixel_size_m,
                    units="m",
                    color="white",
                    box_color="black",
                    box_alpha=0.7,
                    location="lower right",
                )
            )
        except Exception as exc:  # pragma: no cover - defensive
            logging.debug("Could not draw scalebar: %s", exc)

    if show_center_marker:
        height, width = data.shape[0], data.shape[1]
        center_x = width / 2
        center_y = height / 2
        ax.scatter(
            [center_x],
            [center_y],
            marker="+",
            c=marker_color,
            s=max(marker_size, 1) ** 2,
            linewidths=2,
        )

    ax.axis("off")

    if show_metadata:
        metadata_text = _get_metadata_text(image)
        ax.text(
            x=0.01,
            y=0.03,
            s=metadata_text,
            transform=ax.transAxes,
            fontsize=max(metadata_font_size, 6),
            color="white",
            bbox=dict(facecolor="black", alpha=0.7),
            ha="left",
        )

    plt.tight_layout()

    return fig


class ImageAnnotationWidget(QWidget):
    """Minimal widget that loads an image, previews it, and toggles a scalebar."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.image: Optional[FibsemImage] = None
        self.image_path: Optional[str] = None

        self.figure: Optional[Figure] = None
        self.canvas: Optional[FigureCanvas] = None
        self._scroll_cid: Optional[int] = None
        self._pan_press_cid: Optional[int] = None
        self._pan_release_cid: Optional[int] = None
        self._pan_motion_cid: Optional[int] = None
        self._pan_active = False
        self._pan_axes = None
        self._pan_start = None
        self._pan_axes_limits = None
        self._initial_axes_limits = []

        self.canvas_holder = QWidget()
        self.canvas_layout = QVBoxLayout(self.canvas_holder)
        self.canvas_layout.setContentsMargins(0, 0, 0, 0)

        self.center_marker_color = QColor("#ffff0e")

        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self._on_load_image_clicked)

        self.save_button = QPushButton("Export Image")
        self.save_button.clicked.connect(self._on_save_clicked)
        self.save_button.setEnabled(False)

        self.reset_view_button = QPushButton("Reset View")
        self.reset_view_button.clicked.connect(self._reset_view)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.load_button)
        controls_layout.addWidget(self.reset_view_button)
        controls_layout.addWidget(self.save_button)
        controls_layout.addStretch()

        self.show_scalebar_checkbox = QCheckBox("")
        self.show_scalebar_checkbox.setChecked(True)
        self.show_scalebar_checkbox.setEnabled(False)
        self.show_scalebar_checkbox.stateChanged.connect(self._update_preview)

        self.show_center_marker_checkbox = QCheckBox("")
        self.show_center_marker_checkbox.setChecked(True)
        self.show_center_marker_checkbox.stateChanged.connect(self._update_preview)

        self.marker_size_spinbox = QSpinBox()
        self.marker_size_spinbox.setRange(5, 200)
        self.marker_size_spinbox.setValue(30)
        self.marker_size_spinbox.valueChanged.connect(self._update_preview)

        self.marker_color_label = QLabel("●")
        self._update_marker_color_label()
        self.marker_color_button = QPushButton("Marker Color")
        self.marker_color_button.clicked.connect(self._on_marker_color_clicked)

        self.median_filter_checkbox = QCheckBox("")
        self.median_filter_checkbox.setChecked(False)
        self.median_filter_checkbox.setEnabled(False)
        self.median_filter_checkbox.stateChanged.connect(self._update_preview)

        self.show_metadata_checkbox = QCheckBox("")
        self.show_metadata_checkbox.setChecked(True)
        self.show_metadata_checkbox.stateChanged.connect(self._update_preview)

        self.metadata_font_spinbox = QSpinBox()
        self.metadata_font_spinbox.setRange(6, 32)
        self.metadata_font_spinbox.setValue(10)
        self.metadata_font_spinbox.valueChanged.connect(self._update_preview)

        marker_color_widget = QWidget()
        marker_color_layout = QHBoxLayout(marker_color_widget)
        marker_color_layout.setContentsMargins(0, 0, 0, 0)
        marker_color_layout.addWidget(self.marker_color_label)
        marker_color_layout.addWidget(self.marker_color_button)
        marker_color_layout.addStretch()

        form_layout = QFormLayout()
        form_layout.addRow("Show Marker", self.show_center_marker_checkbox)
        form_layout.addRow("Marker Size", self.marker_size_spinbox)
        form_layout.addRow("Marker Color", marker_color_widget)
        form_layout.addRow("Median Filter", self.median_filter_checkbox)
        form_layout.addRow("Show Scalebar", self.show_scalebar_checkbox)
        form_layout.addRow("Show Metadata", self.show_metadata_checkbox)
        form_layout.addRow("Fontsize", self.metadata_font_spinbox)

        side_container_layout = QVBoxLayout()
        side_container_layout.addLayout(form_layout)
        side_container_layout.addStretch()

        preview_layout = QHBoxLayout()
        preview_layout.addWidget(self.canvas_holder, stretch=3)
        preview_layout.addLayout(side_container_layout, stretch=1)

        self.info_label = QLabel("Load an image to begin")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.main_layout = QVBoxLayout()
        self.main_layout.addLayout(preview_layout)
        self.main_layout.addLayout(controls_layout)
        self.main_layout.addWidget(self.info_label)

        self.setLayout(self.main_layout)

        self._draw_placeholder()

    # ------------------------------------------------------------------ #
    # UI helpers
    # ------------------------------------------------------------------ #
    def _set_canvas_figure(self, figure: Figure):
        """Replace the current canvas with the provided figure."""
        if self.canvas is not None:
            self._disconnect_canvas_events()
            self.canvas_layout.removeWidget(self.canvas)
            self.canvas.deleteLater()

        self.figure = figure
        self.canvas = FigureCanvas(figure)
        self.canvas_layout.addWidget(self.canvas)
        self._enable_canvas_events()
        self._capture_initial_view_limits()

    def _draw_placeholder(self):
        """Show placeholder text before an image is loaded."""
        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_facecolor("black")
        ax.text(
            0.5,
            0.5,
            "No preview available\nLoad an image to begin",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="white",
        )
        ax.axis("off")
        self._set_canvas_figure(fig)

    def _update_marker_color_label(self):
        self.marker_color_label.setStyleSheet(
            f"color: {self.center_marker_color.name()}; font-size: 18px;"
        )

    def _on_marker_color_clicked(self):
        color = QColorDialog.getColor(self.center_marker_color, self, "Select Marker Color")
        if not color.isValid():
            return
        self.center_marker_color = color
        self._update_marker_color_label()
        self._update_preview()

    def _scalebar_available(self) -> bool:
        if not ScaleBar or self.image is None:
            return False
        return _pixel_size_from_metadata(self.image) is not None

    def _disconnect_canvas_events(self):
        if self.canvas is None:
            return
        for attr in ("_scroll_cid", "_pan_press_cid", "_pan_release_cid", "_pan_motion_cid"):
            cid = getattr(self, attr)
            if cid is None:
                continue
            try:
                self.canvas.mpl_disconnect(cid)
            except Exception:
                pass
            setattr(self, attr, None)
        self._pan_active = False
        self._pan_axes = None
        self._pan_start = None
        self._pan_axes_limits = None

    def _enable_canvas_events(self):
        if self.canvas is None:
            return
        self._scroll_cid = self.canvas.mpl_connect("scroll_event", self._on_canvas_scroll)
        self._pan_press_cid = self.canvas.mpl_connect("button_press_event", self._on_canvas_press)
        self._pan_release_cid = self.canvas.mpl_connect("button_release_event", self._on_canvas_release)
        self._pan_motion_cid = self.canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)

    def _capture_initial_view_limits(self):
        self._initial_axes_limits = []
        if self.figure is None:
            return
        for ax in self.figure.axes:
            self._initial_axes_limits.append((ax, ax.get_xlim(), ax.get_ylim()))

    def _reset_view(self):
        if not self.figure or not self.canvas or not self._initial_axes_limits:
            return
        for ax, xlim, ylim in self._initial_axes_limits:
            if ax in self.figure.axes:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
        self.canvas.draw_idle()

    def _on_canvas_scroll(self, event):
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        ax = event.inaxes
        scale_factor = 0.9 if event.button == "up" else 1.1
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        x_range = cur_xlim[1] - cur_xlim[0]
        y_range = cur_ylim[1] - cur_ylim[0]
        if x_range == 0 or y_range == 0:
            return
        new_width = x_range * scale_factor
        new_height = y_range * scale_factor
        relx = (event.xdata - cur_xlim[0]) / x_range
        rely = (event.ydata - cur_ylim[0]) / y_range
        new_xlim = (event.xdata - relx * new_width, event.xdata + (1 - relx) * new_width)
        new_ylim = (event.ydata - rely * new_height, event.ydata + (1 - rely) * new_height)
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        self.canvas.draw_idle()

    def _on_canvas_press(self, event):
        if event.button != 1 or event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        self._pan_active = True
        self._pan_axes = event.inaxes
        self._pan_start = (event.xdata, event.ydata)
        self._pan_axes_limits = (event.inaxes.get_xlim(), event.inaxes.get_ylim())

    def _on_canvas_motion(self, event):
        if not self._pan_active or self._pan_axes is None:
            return
        if event.inaxes != self._pan_axes or event.xdata is None or event.ydata is None:
            return
        if self._pan_axes_limits is None or self._pan_start is None:
            return
        start_xlim, start_ylim = self._pan_axes_limits
        dx = event.xdata - self._pan_start[0]
        dy = event.ydata - self._pan_start[1]
        self._pan_axes.set_xlim(start_xlim[0] - dx, start_xlim[1] - dx)
        self._pan_axes.set_ylim(start_ylim[0] - dy, start_ylim[1] - dy)
        self.canvas.draw_idle()

    def _on_canvas_release(self, event):
        if event.button != 1:
            return
        self._pan_active = False
        self._pan_axes = None
        self._pan_start = None
        self._pan_axes_limits = None

    # ------------------------------------------------------------------ #
    # Image workflow
    # ------------------------------------------------------------------ #
    def _on_load_image_clicked(self):
        dialog = QFileDialog(self)
        dialog.setNameFilters(
            [
                "Image Files (*.tif *.tiff *.png *.jpg *.jpeg *.bmp)",
                "All Files (*)",
            ]
        )
        dialog.setAcceptMode(QFileDialog.AcceptOpen)

        if dialog.exec_() != QDialog.Accepted:
            return

        file_path = dialog.selectedFiles()[0]
        try:
            self._load_image(file_path)
            self.info_label.setText(f"Loaded: {os.path.basename(file_path)}")
        except Exception as exc:
            logging.error("Failed to load image %s: %s", file_path, exc)
            QMessageBox.critical(self, "Load Error", f"Could not load image:\n{exc}")
            self.image = None
            self.image_path = None
            self._draw_placeholder()

    def _load_image(self, file_path: str):
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        fibsem_image: Optional[FibsemImage] = None

        if ext in {".tif", ".tiff"}:
            try:
                fibsem_image = FibsemImage.load(file_path)
            except Exception as exc:
                logging.debug("Falling back to generic loader for %s: %s", file_path, exc)

        if fibsem_image is None:
            data = mpimg.imread(file_path)
            if data.ndim == 3 and data.shape[2] == 4:
                data = data[:, :, :3]
            fibsem_image = FibsemImage(data=data, metadata=None)
        self.set_image(fibsem_image, file_path=file_path)

    def set_image(self, fibsem_image: FibsemImage, file_path: Optional[str] = None):
        self.image = fibsem_image
        self.image_path = file_path

        can_show_scalebar = self._scalebar_available()
        self.show_scalebar_checkbox.setEnabled(can_show_scalebar)
        if not can_show_scalebar:
            self.show_scalebar_checkbox.setChecked(False)
            if ScaleBar is None:
                self.info_label.setText("Install matplotlib-scalebar to enable scalebar.")
            else:
                self.info_label.setText("Image metadata missing pixel size; scalebar disabled.")

        self.median_filter_checkbox.setEnabled(True)
        self.median_filter_checkbox.setChecked(False)

        self.save_button.setEnabled(True)
        self._update_preview()

    def _update_preview(self):
        if self.image is None:
            self._draw_placeholder()
            return

        show_scalebar = self.show_scalebar_checkbox.isChecked() and self._scalebar_available()
        fig = plot_image_with_scalebar(
            image=self.image,
            show_scalebar=show_scalebar,
            apply_median_filter=self.median_filter_checkbox.isChecked(),
            show_center_marker=self.show_center_marker_checkbox.isChecked(),
            show_metadata=self.show_metadata_checkbox.isChecked(),
            metadata_font_size=self.metadata_font_spinbox.value(),
            marker_color=self.center_marker_color.name(),
            marker_size=self.marker_size_spinbox.value(),
        )
        self._set_canvas_figure(fig)

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #
    def _on_save_clicked(self):
        if self.image is None or self.figure is None:
            QMessageBox.warning(self, "Export", "Nothing to export. Load an image first.")
            return

        default_name = "preview.png"
        if self.image_path:
            default_dir = os.path.dirname(self.image_path)
            basename = os.path.splitext(os.path.basename(self.image_path))[0]
            default_name = os.path.join(default_dir, f"{basename}_preview.png")
        else:
            default_name = os.path.join(os.getcwd(), default_name)

        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Image",
            default_name,
            "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;PDF Files (*.pdf)",
        )
        if not output_path:
            return

        try:
            self.figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
            self.info_label.setText(f"Saved to: {output_path}")
        except Exception as exc:
            logging.error("Failed to save image %s: %s", output_path, exc)
            QMessageBox.critical(self, "Export Error", f"Could not save image:\n{exc}")


def create_image_annotation_dialog(parent: Optional[QWidget] = None) -> QDialog:
    """Convenience helper to display the widget in a dialog."""
    dialog = QDialog(parent)
    dialog.setWindowTitle("Image Preview")
    # dialog.setMinimumSize(900, 600)

    layout = QVBoxLayout(dialog)
    widget = ImageAnnotationWidget(parent=dialog)
    layout.addWidget(widget)
    dialog.setLayout(layout)

    return dialog


if __name__ == "__main__":  # Simple test harness
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    dialog = create_image_annotation_dialog()

    import os, glob, random
    filenames = glob.glob("/home/patrick/github/fibsem/projects/20251021_Screen/ScreenTest/tile_0_7/*.tif")
    # PATH = "/home/patrick/github/fibsem/projects/20251021_Screen/ScreenTest/tile_0_7/tile_1_7_ib.tif"

    filename = random.choice(filenames)

    dialog.findChild(ImageAnnotationWidget)._load_image(filename)
    dialog.show()
    sys.exit(app.exec_())
