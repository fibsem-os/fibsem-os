"""Export an image with its scalebar, legend and a metadata bar.

The preview is drawn by the same function that writes the file,
:func:`fibsem.imaging.export.render_export`, so what the dialog shows is what is
saved -- at reduced size, and at 1x when the output is 2x.
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace
from typing import Dict, List, Optional, Sequence

import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from superqt import QDoubleSlider

from fibsem.imaging.export import (
    DEFAULT_PROVENANCE,
    MAX_FIELDS,
    ExportImage,
    auto_contrast_limits,
    default_export_name,
    default_options,
    export_shape,
    load_export_image,
    render_export,
    save_export,
)
from fibsem.ui import notification_service
from fibsem.ui.stylesheets import (
    PRIMARY_BUTTON_STYLESHEET,
    SECONDARY_BUTTON_STYLESHEET,
)
from fibsem.ui.tokens import (
    BORDER_COLOR,
    CAPTION_STYLE,
    CAPTION_VALUE_STYLE,
    CONTROL_STYLE,
    PANEL_COLOR,
    PANEL_TITLE_STYLE,
    SURFACE_COLOR,
    TEXT_MUTED_COLOR,
    TEXT_STRONG_COLOR,
)

_FORMATS = (("PNG", ".png"), ("TIFF", ".tif"))
_SCALES = (("1×", 1), ("2×", 2))
_LOCATIONS = (("Left", "lower left"), ("Right", "lower right"))

_SEGMENT_STYLE = f"""
    QPushButton {{
        background: transparent;
        color: {TEXT_MUTED_COLOR};
        border: 1px solid {BORDER_COLOR};
        padding: 2px 10px;
        font-size: 11px;
    }}
    QPushButton:checked {{
        background: {BORDER_COLOR};
        color: {TEXT_STRONG_COLOR};
    }}
"""


def _rgb_to_qimage(rgb: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    # copy(): the QImage otherwise borrows the array's buffer, which Python may free.
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class _Segmented(QWidget):
    """A row of mutually exclusive buttons: PNG | TIFF. Emits the chosen index."""

    changed = pyqtSignal(int)

    def __init__(self, labels: Sequence[str], index: int = 0, parent=None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        for i, label in enumerate(labels):
            button = QPushButton(label)
            button.setCheckable(True)
            button.setChecked(i == index)
            button.setStyleSheet(_SEGMENT_STYLE)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            self.group.addButton(button, i)
            layout.addWidget(button)
        # buttonClicked rather than idClicked, which is Qt 5.15 only.
        self.group.buttonClicked.connect(
            lambda button: self.changed.emit(self.group.id(button))
        )

    def index(self) -> int:
        return self.group.checkedId()


class _SliderRow(QWidget):
    """`Min ───●──── 0.12`: a caption, a slider and its value, as the canvas's
    contrast popover lays them out."""

    changed = pyqtSignal(float)

    def __init__(
        self, label: str, lo: float, hi: float, value: float, step: float
    ) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        caption = _caption(label)
        caption.setFixedWidth(42)
        self.slider = QDoubleSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(lo, hi)
        self.slider.setSingleStep(step)
        self.slider.setValue(value)
        self.label_value = QLabel(f"{value:.2f}")
        self.label_value.setStyleSheet(CAPTION_VALUE_STYLE)
        self.label_value.setFixedWidth(30)
        self.label_value.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        layout.addWidget(caption)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.label_value)
        self.slider.valueChanged.connect(self._on_value)

    def _on_value(self, value: float) -> None:
        self.label_value.setText(f"{value:.2f}")
        self.changed.emit(value)

    def value(self) -> float:
        return float(self.slider.value())

    def set_value(self, value: float) -> None:
        """Set without emitting: Auto and Reset move all three, then refresh once."""
        self.slider.blockSignals(True)
        self.slider.setValue(value)
        self.slider.blockSignals(False)
        self.label_value.setText(f"{value:.2f}")


def _caption(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet(CAPTION_STYLE)
    return label


def _row(label: str, widget: QWidget) -> QWidget:
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(_caption(label))
    layout.addStretch()
    layout.addWidget(widget)
    return row


def _tab(*widgets: QWidget) -> QWidget:
    tab = QWidget()
    # Controls at 12px among 11px captions, not the app's larger default.
    tab.setStyleSheet(f"QCheckBox {{ {CONTROL_STYLE} }}")
    layout = QVBoxLayout(tab)
    layout.setContentsMargins(12, 12, 12, 12)
    layout.setSpacing(6)
    for widget in widgets:
        layout.addWidget(widget)
    layout.addStretch()
    return tab


class ImageExportDialog(QDialog):
    """Preview on the left, settings in tabs on the right, Copy and Save below."""

    def __init__(self, image: ExportImage, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.image = image
        self.options = default_options(image)
        # What the provenance row shows when switched on. Kept while it is off so
        # turning it off and on again does not forget the choice.
        keys = image.provenance_keys()
        self._provenance = [k for k in DEFAULT_PROVENANCE if k in keys]
        self._extension = _FORMATS[0][1]
        self._preview: Optional[QPixmap] = None
        # A slider emits on every step of a drag; re-rendering a 6k image on each
        # one lags the handle, so the preview catches up when the drag pauses.
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(60)
        self._refresh_timer.timeout.connect(self._on_changed)

        self.setWindowTitle("Export image")
        self.setStyleSheet(f"QDialog {{ background-color: {SURFACE_COLOR}; }}")
        self.resize(900, 600)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_preview(), 1)
        body.addWidget(self._build_settings())
        root.addLayout(body, 1)
        root.addWidget(self._build_footer())

        self._refresh()

    # -- construction --------------------------------------------------------

    def _build_header(self) -> QWidget:
        header = QWidget()
        layout = QHBoxLayout(header)
        layout.setContentsMargins(16, 12, 16, 10)
        title = QLabel("Export image")
        title.setStyleSheet(PANEL_TITLE_STYLE)
        values = {f.key: f.value for f in self.image.provenance}
        parts = [self.image.kind, values.get("item"), values.get("task")]
        if self.image.path:
            parts.append(os.path.basename(self.image.path))
        self.label_meta = _caption(" · ".join(p for p in parts if p))
        layout.addWidget(title)
        layout.addSpacing(8)
        layout.addWidget(self.label_meta, 1)
        return header

    def _build_preview(self) -> QWidget:
        frame = QFrame()
        frame.setStyleSheet(f"QFrame {{ background-color: {PANEL_COLOR}; }}")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 16, 16, 16)
        self.label_preview = QLabel()
        self.label_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_preview.setMinimumSize(480, 320)
        self.label_preview.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored
        )
        layout.addWidget(self.label_preview)
        return frame

    def _build_settings(self) -> QWidget:
        self.tabs = QTabWidget()
        self.tabs.setFixedWidth(240)
        # Three tabs across the column, rather than the app's wide tab padding
        # pushing the third behind a scroll arrow.
        self.tabs.tabBar().setExpanding(True)
        self.tabs.setUsesScrollButtons(False)
        self.tabs.setStyleSheet(
            f"QTabBar::tab {{ padding: 6px 10px; {CONTROL_STYLE} }}"
        )
        self.tabs.addTab(self._build_image_tab(), "Image")
        self.tabs.addTab(self._build_metadata_tab(), "Metadata")
        self.tabs.addTab(self._build_output_tab(), "Output")
        return self.tabs

    def _build_image_tab(self) -> QWidget:
        self.checkbox_scalebar = QCheckBox("Scalebar")
        self.checkbox_scalebar.setChecked(self.options.scalebar)
        self.checkbox_scalebar.setEnabled(self.image.pixel_size is not None)
        if self.image.pixel_size is None:
            self.checkbox_scalebar.setToolTip("This file does not record a pixel size")
        self.checkbox_scalebar.toggled.connect(self._on_changed)

        locations = [loc for _, loc in _LOCATIONS]
        self.segment_location = _Segmented(
            [name for name, _ in _LOCATIONS],
            locations.index(self.options.scalebar_location),
        )
        self.segment_location.changed.connect(self._on_changed)
        # Left | Right beside the checkbox it positions, rather than a row of its own.
        scalebar_row = QWidget()
        layout = QHBoxLayout(scalebar_row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.checkbox_scalebar)
        layout.addStretch()
        layout.addWidget(self.segment_location)

        self.checkbox_crosshair = QCheckBox("Crosshair")
        self.checkbox_crosshair.setChecked(self.options.crosshair)
        self.checkbox_crosshair.toggled.connect(self._on_changed)

        widgets: List[QWidget] = [
            scalebar_row,
            self.checkbox_crosshair,
        ]
        self.checkbox_legend: Optional[QCheckBox] = None
        self.channel_checkboxes: List[QCheckBox] = []
        if self.image.channels:
            self.checkbox_legend = QCheckBox("Channel legend")
            self.checkbox_legend.setChecked(self.options.legend)
            self.checkbox_legend.toggled.connect(self._on_changed)
            widgets.append(self.checkbox_legend)
            widgets += self._build_channel_controls()
        if self.image.adjustable:
            widgets += self._build_display_controls()
        return _tab(*widgets)

    def _build_channel_controls(self) -> List[QWidget]:
        """One checkbox per channel, swatched in its colour: a hidden channel leaves
        the blend and the legend. A bright channel -- reflection, usually -- can
        otherwise wash out the ones the figure is about."""
        spacer = QWidget()
        spacer.setFixedHeight(6)
        widgets: List[QWidget] = [spacer, _caption("Channels")]
        hidden = set(self.options.hidden_channels)
        for i, channel in enumerate(self.image.channels):
            swatch = QPixmap(10, 10)
            swatch.fill(QColor(*channel.color))
            checkbox = QCheckBox(channel.name)
            checkbox.setIcon(QIcon(swatch))
            checkbox.setToolTip(channel.name)
            checkbox.setChecked(i not in hidden)
            checkbox.toggled.connect(self._on_changed)
            self.channel_checkboxes.append(checkbox)
            widgets.append(checkbox)
        return widgets

    def _build_display_controls(self) -> List[QWidget]:
        """Contrast and gamma, for a greyscale image. A fluorescence composite is
        contrasted per channel, which these three controls cannot express."""
        o = self.options
        self.slider_min = _SliderRow("Min", 0.0, 1.0, o.contrast_min, 0.01)
        self.slider_max = _SliderRow("Max", 0.0, 1.0, o.contrast_max, 0.01)
        self.slider_gamma = _SliderRow("Gamma", 0.1, 3.0, o.gamma, 0.05)
        for row in (self.slider_min, self.slider_max, self.slider_gamma):
            row.changed.connect(lambda _: self._refresh_timer.start())

        self.button_auto = QPushButton("Auto")
        self.button_auto.setToolTip("Limits at the 1st and 99th percentiles")
        self.button_auto.clicked.connect(self.auto_contrast)
        self.button_reset = QPushButton("Reset")
        self.button_reset.setToolTip("As acquired")
        self.button_reset.clicked.connect(self.reset_contrast)
        buttons = QWidget()
        layout = QHBoxLayout(buttons)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch()
        for button in (self.button_auto, self.button_reset):
            # Small, like the segmented buttons: these sit among 11px captions.
            button.setStyleSheet(
                SECONDARY_BUTTON_STYLESHEET
                + "QPushButton { font-size: 11px; padding: 2px 10px; }"
            )
            layout.addWidget(button)

        spacer = QWidget()
        spacer.setFixedHeight(6)
        return [
            spacer,
            _caption("Display"),
            self.slider_min,
            self.slider_max,
            self.slider_gamma,
            buttons,
        ]

    def auto_contrast(self) -> None:
        lo, hi = auto_contrast_limits(self.image)
        self.slider_min.set_value(lo)
        self.slider_max.set_value(hi)
        self._on_changed()

    def reset_contrast(self) -> None:
        self.slider_min.set_value(0.0)
        self.slider_max.set_value(1.0)
        self.slider_gamma.set_value(1.0)
        self._on_changed()

    def _build_metadata_tab(self) -> QWidget:
        widgets: List[QWidget] = []
        self.field_checkboxes: Dict[str, QCheckBox] = {}
        if self.image.fields:
            widgets.append(_caption(f"Bar · up to {MAX_FIELDS} values"))
        else:
            widgets.append(_caption("This file records no acquisition values"))
        for f in self.image.fields:
            checkbox = QCheckBox(f.name)
            checkbox.setChecked(f.key in self.options.fields)
            checkbox.toggled.connect(self._on_changed)
            self.field_checkboxes[f.key] = checkbox
            widgets.append(checkbox)

        self.provenance_checkboxes: Dict[str, QCheckBox] = {}
        self.checkbox_provenance = QCheckBox("Provenance")
        self.checkbox_provenance.setChecked(False)
        self.checkbox_provenance.setEnabled(bool(self.image.provenance))
        self.checkbox_provenance.toggled.connect(self._on_changed)
        spacer = QWidget()
        spacer.setFixedHeight(6)
        widgets += [spacer, self.checkbox_provenance]
        for f in self.image.provenance:
            checkbox = QCheckBox(f.name)
            checkbox.setChecked(f.key in self._provenance)
            checkbox.setStyleSheet("QCheckBox { margin-left: 18px; }")
            checkbox.toggled.connect(self._on_changed)
            self.provenance_checkboxes[f.key] = checkbox
            widgets.append(checkbox)
        return _tab(*widgets)

    def _build_output_tab(self) -> QWidget:
        self.segment_format = _Segmented([name for name, _ in _FORMATS])
        self.segment_format.changed.connect(self._on_changed)
        self.segment_scale = _Segmented([name for name, _ in _SCALES])
        self.segment_scale.changed.connect(self._on_changed)
        return _tab(
            _row("Format", self.segment_format),
            _row("Scale", self.segment_scale),
        )

    def _build_footer(self) -> QWidget:
        footer = QFrame()
        footer.setStyleSheet(
            f"QFrame {{ border-top: 1px solid {BORDER_COLOR}; }}"
            " QLabel, QPushButton { border-top: none; }"
        )
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(16, 10, 16, 10)
        self.label_size = _caption("")
        self.button_copy = QPushButton("Copy")
        self.button_copy.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        self.button_copy.setToolTip("Copy the exported image to the clipboard")
        self.button_copy.clicked.connect(self.copy_to_clipboard)
        self.button_save = QPushButton("Save…")
        self.button_save.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        self.button_save.clicked.connect(self._on_save)
        layout.addWidget(self.label_size, 1)
        layout.addWidget(self.button_copy)
        layout.addWidget(self.button_save)
        return footer

    # -- state ---------------------------------------------------------------

    def _on_changed(self, *_) -> None:
        self._read_options()
        self._refresh()

    def _read_options(self) -> None:
        o = self.options
        o.scalebar = self.checkbox_scalebar.isChecked()
        o.scalebar_location = _LOCATIONS[self.segment_location.index()][1]
        o.crosshair = self.checkbox_crosshair.isChecked()
        if self.checkbox_legend is not None:
            o.legend = self.checkbox_legend.isChecked()
        o.hidden_channels = [
            i for i, cb in enumerate(self.channel_checkboxes) if not cb.isChecked()
        ]
        o.fields = [k for k, cb in self.field_checkboxes.items() if cb.isChecked()]
        self._provenance = [
            k for k, cb in self.provenance_checkboxes.items() if cb.isChecked()
        ]
        o.provenance = (
            list(self._provenance) if self.checkbox_provenance.isChecked() else []
        )
        o.scale = _SCALES[self.segment_scale.index()][1]
        self._extension = _FORMATS[self.segment_format.index()][1]
        if self.image.adjustable:
            lo, hi = self.slider_min.value(), self.slider_max.value()
            # Crossed handles would invert nothing useful; keep a sliver of range.
            if hi <= lo:
                hi = min(1.0, lo + 0.01)
                lo = hi - 0.01
            o.contrast_min, o.contrast_max = lo, hi
            o.gamma = self.slider_gamma.value()

    def _refresh(self) -> None:
        self.segment_location.setEnabled(self.options.scalebar)
        provenance_on = self.checkbox_provenance.isChecked()
        for checkbox in self.provenance_checkboxes.values():
            checkbox.setEnabled(provenance_on)
        self._apply_field_cap()

        # The preview is always rendered at 1x: a 2x export is the same picture with
        # every pixel doubled, and rendering it only to shrink it costs four times as much.
        preview_options = self._preview_options()
        rgb = render_export(self.image, preview_options)
        self._preview = QPixmap.fromImage(_rgb_to_qimage(rgb))
        h, w = export_shape(self.image, self.options)
        self.label_size.setText(f"{w} × {h} px · {self._format_name()}")
        self._show_preview()

    def _preview_options(self):
        return replace(self.options, scale=1)

    def _apply_field_cap(self) -> None:
        """Stop at MAX_FIELDS labelled values; the detector or objective is free."""
        labelled = {f.key for f in self.image.fields if f.label}
        chosen = [k for k in self.options.fields if k in labelled]
        full = len(chosen) >= MAX_FIELDS
        for key, checkbox in self.field_checkboxes.items():
            blocked = full and key in labelled and not checkbox.isChecked()
            checkbox.setEnabled(not blocked)
            checkbox.setToolTip(
                f"Up to {MAX_FIELDS} values fit on one row" if blocked else ""
            )

    def _format_name(self) -> str:
        return _FORMATS[self.segment_format.index()][0]

    def _show_preview(self) -> None:
        if self._preview is None:
            return
        size = self.label_preview.size()
        self.label_preview.setPixmap(
            self._preview.scaled(
                size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._show_preview()

    # -- output --------------------------------------------------------------

    def rendered(self) -> np.ndarray:
        """The export at full size, as it would be saved."""
        return render_export(self.image, self.options)

    def copy_to_clipboard(self) -> None:
        clipboard = QApplication.clipboard()
        if clipboard is None:
            return
        clipboard.setImage(_rgb_to_qimage(self.rendered()))
        notification_service.show_toast("Image copied to the clipboard", "success")

    def save(self, path: str) -> str:
        """Write the export to ``path``, adding the chosen format's extension if it has none."""
        if not os.path.splitext(path)[1]:
            path += self._extension
        return save_export(self.rendered(), path)

    def _on_save(self) -> None:
        default = default_export_name(self.image, self._extension)
        filters = ";;".join(f"{name} (*{ext})" for name, ext in _FORMATS)
        selected = f"{self._format_name()} (*{self._extension})"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save exported image", default, filters, selected
        )
        if not path:
            return
        try:
            written = self.save(path)
        except (OSError, ValueError) as e:
            logging.exception("Failed to export image to %s", path)
            notification_service.show_toast(f"Couldn't save the image: {e}", "error")
            return
        notification_service.show_toast(
            f"Exported {os.path.basename(written)}", "success"
        )
        self.accept()


def open_image_export(parent: Optional[QWidget] = None, start_dir: str = "") -> None:
    """Pick an image file, then open the export dialog for it.

    ``start_dir`` is where the file picker opens -- the experiment, from the File menu.
    """
    path, _ = QFileDialog.getOpenFileName(
        parent,
        "Export image",
        start_dir,
        "Images (*.tif *.tiff);;All files (*)",
    )
    if not path:
        return
    # A fluorescence stack can be hundreds of megabytes; say that something is
    # happening rather than appear to hang.
    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    try:
        image = load_export_image(path)
    except Exception as e:
        logging.exception("Failed to read %s for export", path)
        notification_service.show_toast(
            f"Couldn't read {os.path.basename(path)}: {e}", "error"
        )
        return
    finally:
        QApplication.restoreOverrideCursor()
    ImageExportDialog(image, parent).exec_()
