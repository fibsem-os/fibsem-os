from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from PIL import Image, ImageDraw

from fibsem.applications.autolamella.structures import LamellaDefaultConfig
from fibsem.ui.widgets.custom_widgets import IconToolButton, TitledPanel, ValueSpinBox
from fibsem.imaging.drawing import _get_font, draw_crosshair_at, draw_rectangle_reduced, draw_scalebar
from fibsem.structures import DEFAULT_ALIGNMENT_AREA, FibsemRectangle, Point
from fibsem.ui.stylesheets import NAPARI_STYLE

_DEFAULT_AA = FibsemRectangle.from_dict(DEFAULT_ALIGNMENT_AREA)
_DEFAULT_POI = Point(0, 0)

_SECTION_STYLE = (
    "font-size: 10px; font-weight: bold; color: #707070;"
    " padding: 4px 0px 2px 0px; letter-spacing: 0.5px;"
)
_LABEL_STYLE = "color: #a0a0a0; min-width: 80px; font-size: 11px; background: transparent;"
_INVALID_STYLE = "color: #E3B617; font-size: 10px; background: transparent;"

_EXAMPLE_PETNAME = "brave-tiger"

# ── Preview image settings ─────────────────────────────────────────────────────
_PREVIEW_HFW = 100e-6               # 100 µm horizontal field width
_PREVIEW_W, _PREVIEW_H = 384, 288
_PIXEL_SIZE = _PREVIEW_HFW / _PREVIEW_W

_COLOR_CENTRE = (255, 255, 0)       # yellow  — image centre crosshair
_COLOR_AA     = (0, 255, 0)         # lime    — alignment area rectangle
_COLOR_POI    = (255, 0, 255)       # magenta — point of interest crosshair

_PREVIEW_BG: np.ndarray = np.zeros((_PREVIEW_H, _PREVIEW_W), dtype=np.uint8)


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


def _render_template_preview(template: LamellaDefaultConfig) -> QPixmap:
    """Render a 100 µm FIB preview image with template overlays as a QPixmap."""
    arr = _PREVIEW_BG.copy()

    # Scale bar
    arr = draw_scalebar(arr, _PIXEL_SIZE)

    # Name text at top of image
    prefix = template.name_prefix or ""
    sep = "-" if prefix else ""
    name = (f"{prefix}{sep}01-{_EXAMPLE_PETNAME}" if template.use_petname
            else f"{prefix}{sep}Lamella-01")
    pil_img = Image.fromarray(arr).convert("RGBA")
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).text((4, 4), name, font=_get_font(16, bold=True), fill=(255, 255, 255, 220))
    arr = np.array(Image.alpha_composite(pil_img, overlay).convert("RGB"))

    # Yellow crosshair at image centre (always visible)
    arr = draw_crosshair_at(arr, 0.5, 0.5, color=_COLOR_CENTRE, alpha=0.9, size_ratio=0.06)

    # Lime green rectangle for alignment area
    aa = template.alignment_area if template.alignment_area is not None else _DEFAULT_AA
    arr = draw_rectangle_reduced(
        arr, aa.left, aa.top, aa.width, aa.height,
        color=_COLOR_AA, alpha=0.9, thickness=2,
    )

    # Magenta crosshair for POI (offset from image centre in microscope coords)
    poi = template.poi if template.poi is not None else _DEFAULT_POI
    vfw = _PREVIEW_HFW * _PREVIEW_H / _PREVIEW_W
    cx_frac = max(0.02, min(0.98, 0.5 + poi.x / _PREVIEW_HFW))
    cy_frac = max(0.02, min(0.98, 0.5 - poi.y / vfw))
    arr = draw_crosshair_at(arr, cx_frac, cy_frac, color=_COLOR_POI, alpha=0.9, size_ratio=0.05)

    arr_c = np.ascontiguousarray(arr)
    h, w = arr_c.shape[:2]
    qimg = QImage(arr_c.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


class LamellaDefaultConfigWidget(QWidget):
    """Compact editor for LamellaDefaultConfig protocol defaults."""

    template_changed = pyqtSignal(object)  # LamellaDefaultConfig

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._preview_pixmap: Optional[QPixmap] = None
        self.setStyleSheet(NAPARI_STYLE + "QCheckBox { background: transparent; }")
        self._setup_ui()
        self._connect_signals()
        self._validate_alignment_area()
        self._update_image_preview(LamellaDefaultConfig())

    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(6, 4, 6, 6)
        root.setSpacing(8)

        # ── left column: controls inside TitledPanel ──────────────────
        self._btn_reset = IconToolButton(icon="mdi:refresh", tooltip="Reset to defaults", size=24)

        controls_widget = QWidget()
        left = QVBoxLayout(controls_widget)
        left.setContentsMargins(4, 4, 4, 4)
        left.setSpacing(4)

        self.use_petname_cb = QCheckBox()
        self.use_petname_cb.setToolTip(
            "When checked, a random two-word petname (e.g. 'brave-tiger') is appended\n"
            "to the lamella name. When unchecked, names use 'Lamella-01' format."
        )
        left.addLayout(_row("Use petname", self.use_petname_cb))

        self.name_prefix_edit = QLineEdit()
        self.name_prefix_edit.setPlaceholderText("e.g. GridA")
        self.name_prefix_edit.setToolTip(
            "Optional text prefix prepended to every new lamella name.\n"
            "A dash separator is inserted automatically (e.g. 'GridA' → 'GridA-Lamella-01')."
        )
        left.addLayout(_row("Prefix", self.name_prefix_edit))

        aa_lbl = QLabel("ALIGNMENT AREA")
        aa_lbl.setStyleSheet(_SECTION_STYLE)
        left.addWidget(aa_lbl)

        self.aa_left   = ValueSpinBox(minimum=0.0, maximum=1.0, decimals=2, step=0.01)
        self.aa_left.setFixedWidth(120)
        self.aa_top    = ValueSpinBox(minimum=0.0, maximum=1.0, decimals=2, step=0.01)
        self.aa_top.setFixedWidth(120)
        self.aa_width  = ValueSpinBox(minimum=0.0, maximum=1.0, decimals=2, step=0.01)
        self.aa_width.setFixedWidth(120)
        self.aa_height = ValueSpinBox(minimum=0.0, maximum=1.0, decimals=2, step=0.01)
        self.aa_height.setFixedWidth(120)
        self._aa_widgets = [self.aa_left, self.aa_top, self.aa_width, self.aa_height]

        _aa_tip = (
            "The alignment area is a sub-region of the image used for cross-correlation\n"
            "based alignment during the lamella workflow. Values are normalised image\n"
            "coordinates in the range [0, 1] where (0, 0) is the top-left corner."
        )
        self.aa_left.setToolTip(f"Left edge of the alignment area.\n{_aa_tip}")
        self.aa_width.setToolTip(f"Width of the alignment area.\n{_aa_tip}")
        self.aa_top.setToolTip(f"Top edge of the alignment area.\n{_aa_tip}")
        self.aa_height.setToolTip(f"Height of the alignment area.\n{_aa_tip}")

        left.addLayout(_row("Left / Width",  self.aa_left,  self.aa_width))
        left.addLayout(_row("Top / Height", self.aa_top,   self.aa_height))

        self.aa_validation_lbl = QLabel()
        self.aa_validation_lbl.setVisible(False)
        left.addWidget(self.aa_validation_lbl)

        poi_lbl = QLabel("POINT OF INTEREST")
        poi_lbl.setStyleSheet(_SECTION_STYLE)
        left.addWidget(poi_lbl)

        self.poi_x = ValueSpinBox(suffix="µm", minimum=-5000.0, maximum=5000.0, decimals=2, step=0.1)
        self.poi_x.setFixedWidth(120)
        self.poi_y = ValueSpinBox(suffix="µm", minimum=-5000.0, maximum=5000.0, decimals=2, step=0.1)
        self.poi_y.setFixedWidth(120)
        self._poi_widgets = [self.poi_x, self.poi_y]

        _poi_tip = (
            "The point of interest is the initial target position within the image for\n"
            "the lamella workflow. It is defined as an offset from the image centre.\n"
            "Units: micrometres (µm). Positive X is right, positive Y is up."
        )
        self.poi_x.setToolTip(f"Horizontal offset from image centre.\n{_poi_tip}")
        self.poi_y.setToolTip(f"Vertical offset from image centre.\n{_poi_tip}")

        left.addLayout(_row("X (µm) / Y (µm)", self.poi_x, self.poi_y))
        left.addStretch()

        self._params_panel = TitledPanel("Default Parameters", content=controls_widget, collapsible=False)
        self._params_panel.add_header_widget(self._btn_reset)
        root.addWidget(self._params_panel)

        # ── right column: preview image ───────────────────────────────
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(4, 4, 4, 4)
        preview_layout.setSpacing(4)

        self.image_preview_lbl = QLabel()
        self.image_preview_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview_lbl.setMinimumSize(_PREVIEW_W // 2, _PREVIEW_H // 2)
        self.image_preview_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_preview_lbl.setStyleSheet("background: black;")
        preview_layout.addWidget(self.image_preview_lbl)

        preview_panel = TitledPanel("Preview  (100 µm FIB)", content=preview_widget, collapsible=False)
        root.addWidget(preview_panel)

    # ------------------------------------------------------------------
    def _connect_signals(self) -> None:
        self._btn_reset.clicked.connect(lambda: self.set_template(LamellaDefaultConfig()))
        self.use_petname_cb.toggled.connect(self._on_changed)
        self.name_prefix_edit.editingFinished.connect(self._on_changed)
        for sb in self._aa_widgets:
            sb.valueChanged.connect(self._on_alignment_changed)
        for sb in self._poi_widgets:
            sb.valueChanged.connect(self._on_changed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_template(self, template: LamellaDefaultConfig) -> None:
        for w in [self.use_petname_cb, *self._aa_widgets, *self._poi_widgets]:
            w.blockSignals(True)
        self.name_prefix_edit.blockSignals(True)

        self.use_petname_cb.setChecked(template.use_petname)
        self.name_prefix_edit.setText(template.name_prefix or "")

        aa = template.alignment_area if template.alignment_area is not None else _DEFAULT_AA
        self.aa_left.setValue(aa.left)
        self.aa_top.setValue(aa.top)
        self.aa_width.setValue(aa.width)
        self.aa_height.setValue(aa.height)

        poi = template.poi if template.poi is not None else _DEFAULT_POI
        self.poi_x.setValue(poi.x * 1e6)
        self.poi_y.setValue(poi.y * 1e6)

        for w in [self.use_petname_cb, *self._aa_widgets, *self._poi_widgets]:
            w.blockSignals(False)
        self.name_prefix_edit.blockSignals(False)

        self._validate_alignment_area()
        self._update_image_preview(template)

    def get_template(self) -> LamellaDefaultConfig:
        return LamellaDefaultConfig(
            use_petname=self.use_petname_cb.isChecked(),
            name_prefix=self.name_prefix_edit.text(),
            alignment_area=FibsemRectangle(
                left=self.aa_left.value(),
                top=self.aa_top.value(),
                width=self.aa_width.value(),
                height=self.aa_height.value(),
            ),
            poi=Point(x=self.poi_x.value() * 1e-6, y=self.poi_y.value() * 1e-6),
        )

    def is_valid(self) -> bool:
        """Return False if the alignment area values are invalid."""
        aa = FibsemRectangle(
            left=self.aa_left.value(),
            top=self.aa_top.value(),
            width=self.aa_width.value(),
            height=self.aa_height.value(),
        )
        return aa.is_valid_reduced_area

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_alignment_changed(self) -> None:
        self._validate_alignment_area()
        self._on_changed()

    def _validate_alignment_area(self) -> None:
        aa = FibsemRectangle(
            left=self.aa_left.value(),
            top=self.aa_top.value(),
            width=self.aa_width.value(),
            height=self.aa_height.value(),
        )
        if aa.is_valid_reduced_area:
            self.aa_validation_lbl.setVisible(False)
        else:
            issues = []
            if aa.width <= 0 or aa.height <= 0:
                issues.append("width/height must be > 0")
            if aa.left + aa.width > 1:
                issues.append(f"left + width = {aa.left + aa.width:.3f} > 1")
            if aa.top + aa.height > 1:
                issues.append(f"top + height = {aa.top + aa.height:.3f} > 1")
            self.aa_validation_lbl.setStyleSheet(_INVALID_STYLE)
            self.aa_validation_lbl.setText("⚠ " + ";  ".join(issues))
            self.aa_validation_lbl.setVisible(True)

    def _update_image_preview(self, template: LamellaDefaultConfig) -> None:
        self._preview_pixmap = _render_template_preview(template)
        self._scale_preview()

    def _scale_preview(self) -> None:
        if self._preview_pixmap is None:
            return
        scaled = self._preview_pixmap.scaled(
            self.image_preview_lbl.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_preview_lbl.setPixmap(scaled)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._scale_preview()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._scale_preview()

    def _on_changed(self) -> None:
        template = self.get_template()
        self._update_image_preview(template)
        self.template_changed.emit(template)
