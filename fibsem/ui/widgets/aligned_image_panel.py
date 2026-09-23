"""The controls for images aligned by hand over the Overview canvas (FIB-1030).

Which image, load and remove, Align and Reset, opacity, and a readout of where it has
been put. Emits; the host owns the images (`AlignedImages`) and the canvas's overlay
mode, so this panel says nothing about geometry and holds no state a record does not.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QPushButton,
    QSlider,
    QWidget,
)

from fibsem.ui.widgets.canvas.overlay_controls import panel_hint


class AlignedImagePanel(QWidget):
    load_requested = pyqtSignal()
    remove_requested = pyqtSignal(str)  # key
    selected = pyqtSignal(str)  # key, "" for none
    align_toggled = pyqtSignal(bool)
    reset_requested = pyqtSignal(str)  # key
    fit_requested = pyqtSignal(str)  # key: pick point pairs and fit
    opacity_changed = pyqtSignal(str, float)  # key, 0..1

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.combo = QComboBox()
        self.combo.currentIndexChanged.connect(self._on_index_changed)
        self.btn_load = QPushButton("Load…")
        self.btn_load.setToolTip("Lay a fluorescence image over the overview")
        self.btn_load.clicked.connect(self.load_requested)
        self.btn_remove = QPushButton("Remove")
        self.btn_remove.clicked.connect(
            lambda: self.remove_requested.emit(self.current_key or "")
        )
        layout.addRow("Image", self.combo)
        files = QHBoxLayout()
        files.setContentsMargins(0, 0, 0, 0)
        files.addWidget(self.btn_load)
        files.addWidget(self.btn_remove)
        layout.addRow(files)

        self.btn_align = QPushButton("Align image")
        self.btn_align.setCheckable(True)
        self.btn_align.setToolTip(
            "Drag the image onto the overview; take the handle to turn it"
        )
        self.btn_align.toggled.connect(self.align_toggled)
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setToolTip("Put the image back where its metadata says")
        self.btn_reset.clicked.connect(
            lambda: self.reset_requested.emit(self.current_key or "")
        )
        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.addWidget(self.btn_align)
        buttons.addWidget(self.btn_reset)
        layout.addRow(buttons)
        # The way users prefer: three matching points rather than a drag.
        self.btn_fit = QPushButton("Fit from points…")
        self.btn_fit.setToolTip(
            "Click matching features on the overview and on the image; "
            "the fit places the image"
        )
        self.btn_fit.clicked.connect(
            lambda: self.fit_requested.emit(self.current_key or "")
        )
        layout.addRow(self.btn_fit)

        self.slider_opacity = QSlider(Qt.Horizontal)
        self.slider_opacity.setRange(0, 100)
        self.slider_opacity.setValue(60)
        self.slider_opacity.valueChanged.connect(
            lambda v: self.opacity_changed.emit(self.current_key or "", v / 100.0)
        )
        layout.addRow("Opacity", self.slider_opacity)

        self.label_placement = panel_hint()
        layout.addRow(self.label_placement)
        self._refresh_enabled()

    # ── the list ──────────────────────────────────────────────────────────

    @property
    def current_key(self) -> Optional[str]:
        key = self.combo.currentData()
        return str(key) if key else None

    def add_image(self, key: str, label: str) -> None:
        self.combo.addItem(label, key)
        self.combo.setCurrentIndex(self.combo.count() - 1)
        self._refresh_enabled()

    def remove_image(self, key: str) -> None:
        index = self.combo.findData(key)
        if index >= 0:
            self.combo.removeItem(index)
        self._refresh_enabled()

    def set_placement_text(self, text: str) -> None:
        self.label_placement.setText(text)

    def _on_index_changed(self, _index: int) -> None:
        self.selected.emit(self.current_key or "")
        self._refresh_enabled()

    def _refresh_enabled(self) -> None:
        has = self.combo.count() > 0
        for widget in (
            self.btn_remove,
            self.btn_align,
            self.btn_reset,
            self.btn_fit,
            self.slider_opacity,
        ):
            widget.setEnabled(has)
        if not has:
            self.btn_align.setChecked(False)
