from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.structures import LamellaTemplateConfig
from fibsem.structures import FibsemRectangle, Point
from fibsem.ui.stylesheets import NAPARI_STYLE

_SECTION_STYLE = (
    "font-size: 10px; font-weight: bold; color: #707070;"
    " padding: 4px 0px 2px 0px; letter-spacing: 0.5px;"
)
_PREVIEW_STYLE = "color: #50a6ff; font-size: 10px; padding: 0px 0px 2px 0px;"
_LABEL_STYLE = "color: #a0a0a0; min-width: 80px; font-size: 11px;"

_EXAMPLE_PETNAME = "brave-tiger"


def _spinbox(minimum: float, maximum: float, decimals: int, step: float,
             suffix: str = "") -> QDoubleSpinBox:
    sb = QDoubleSpinBox()
    sb.setRange(minimum, maximum)
    sb.setDecimals(decimals)
    sb.setSingleStep(step)
    sb.setSuffix(suffix)
    sb.setFixedWidth(100)
    return sb


def _row(label_text: str, *widgets) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setSpacing(6)
    lbl = QLabel(label_text)
    lbl.setStyleSheet(_LABEL_STYLE)
    row.addWidget(lbl)
    for w in widgets:
        row.addWidget(w)
    row.addStretch()
    return row


class LamellaTemplateConfigWidget(QWidget):
    """Compact editor for LamellaTemplateConfig protocol defaults."""

    template_changed = pyqtSignal(object)  # LamellaTemplateConfig

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(NAPARI_STYLE)
        self._setup_ui()
        self._connect_signals()
        self._update_enabled()
        self._update_preview()

    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 4, 6, 6)
        root.setSpacing(4)

        # ── naming ────────────────────────────────────────────────────
        naming_lbl = QLabel("NAMING")
        naming_lbl.setStyleSheet(_SECTION_STYLE)
        root.addWidget(naming_lbl)

        self.use_petname_cb = QCheckBox("Use petname")
        root.addWidget(self.use_petname_cb)

        self.name_prefix_edit = QLineEdit()
        self.name_prefix_edit.setPlaceholderText("e.g. GridA")
        root.addLayout(_row("Prefix", self.name_prefix_edit))

        self.preview_lbl = QLabel()
        self.preview_lbl.setStyleSheet(_PREVIEW_STYLE)
        root.addWidget(self.preview_lbl)

        # ── alignment area ────────────────────────────────────────────
        self.alignment_area_cb = QCheckBox("Initial Alignment Area")
        root.addWidget(self.alignment_area_cb)

        self.aa_left   = _spinbox(0.0, 1.0, 3, 0.01)
        self.aa_top    = _spinbox(0.0, 1.0, 3, 0.01)
        self.aa_width  = _spinbox(0.0, 1.0, 3, 0.01)
        self.aa_height = _spinbox(0.0, 1.0, 3, 0.01)
        self._aa_widgets = [self.aa_left, self.aa_top, self.aa_width, self.aa_height]

        root.addLayout(_row("Left / Top",     self.aa_left,  self.aa_top))
        root.addLayout(_row("Width / Height", self.aa_width, self.aa_height))

        # ── point of interest ─────────────────────────────────────────
        self.poi_cb = QCheckBox("Initial Point of Interest")
        root.addWidget(self.poi_cb)

        self.poi_x = _spinbox(-5000.0, 5000.0, 3, 0.1, " µm")
        self.poi_y = _spinbox(-5000.0, 5000.0, 3, 0.1, " µm")
        self._poi_widgets = [self.poi_x, self.poi_y]

        root.addLayout(_row("X / Y", self.poi_x, self.poi_y))

    # ------------------------------------------------------------------
    def _connect_signals(self) -> None:
        self.use_petname_cb.toggled.connect(self._on_naming_changed)
        self.name_prefix_edit.editingFinished.connect(self._on_naming_changed)
        self.alignment_area_cb.toggled.connect(self._on_alignment_cb_toggled)
        self.poi_cb.toggled.connect(self._on_poi_cb_toggled)
        for sb in self._aa_widgets + self._poi_widgets:
            sb.valueChanged.connect(self._on_changed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_template(self, template: LamellaTemplateConfig) -> None:
        for w in [self.use_petname_cb, self.alignment_area_cb, self.poi_cb,
                  *self._aa_widgets, *self._poi_widgets]:
            w.blockSignals(True)
        self.name_prefix_edit.blockSignals(True)

        self.use_petname_cb.setChecked(template.use_petname)
        self.name_prefix_edit.setText(template.name_prefix or "")

        self.alignment_area_cb.setChecked(template.alignment_area is not None)
        if template.alignment_area is not None:
            self.aa_left.setValue(template.alignment_area.left)
            self.aa_top.setValue(template.alignment_area.top)
            self.aa_width.setValue(template.alignment_area.width)
            self.aa_height.setValue(template.alignment_area.height)

        self.poi_cb.setChecked(template.poi is not None)
        if template.poi is not None:
            self.poi_x.setValue(template.poi.x * 1e6)
            self.poi_y.setValue(template.poi.y * 1e6)

        for w in [self.use_petname_cb, self.alignment_area_cb, self.poi_cb,
                  *self._aa_widgets, *self._poi_widgets]:
            w.blockSignals(False)
        self.name_prefix_edit.blockSignals(False)

        self._update_enabled()
        self._update_preview()

    def get_template(self) -> LamellaTemplateConfig:
        aa = None
        if self.alignment_area_cb.isChecked():
            aa = FibsemRectangle(
                left=self.aa_left.value(),
                top=self.aa_top.value(),
                width=self.aa_width.value(),
                height=self.aa_height.value(),
            )
        poi = None
        if self.poi_cb.isChecked():
            poi = Point(x=self.poi_x.value() * 1e-6, y=self.poi_y.value() * 1e-6)

        return LamellaTemplateConfig(
            use_petname=self.use_petname_cb.isChecked(),
            name_prefix=self.name_prefix_edit.text(),
            alignment_area=aa,
            poi=poi,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_naming_changed(self) -> None:
        self._update_preview()
        self._on_changed()

    def _on_alignment_cb_toggled(self) -> None:
        self._update_enabled()
        self._on_changed()

    def _on_poi_cb_toggled(self) -> None:
        self._update_enabled()
        self._on_changed()

    def _update_enabled(self) -> None:
        for sb in self._aa_widgets:
            sb.setEnabled(self.alignment_area_cb.isChecked())
        for sb in self._poi_widgets:
            sb.setEnabled(self.poi_cb.isChecked())

    def _update_preview(self) -> None:
        prefix = self.name_prefix_edit.text()
        sep = "-" if prefix else ""
        if self.use_petname_cb.isChecked():
            name = f"{prefix}{sep}01-{_EXAMPLE_PETNAME}"
        else:
            name = f"{prefix}{sep}Lamella-01"
        self.preview_lbl.setText(f'Preview: "{name}"')

    def _on_changed(self) -> None:
        self.template_changed.emit(self.get_template())
