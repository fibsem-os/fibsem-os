from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.structures import LamellaDefaultConfig
from fibsem.ui.widgets.custom_widgets import IconToolButton, TitledPanel, ValueSpinBox
from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas
from fibsem.ui.widgets.canvas.overlays.alignment_overlay import AlignmentAreaOverlay
from fibsem.ui.widgets.canvas.overlays.point_overlay import PointOverlay
from fibsem.structures import DEFAULT_ALIGNMENT_AREA, FibsemRectangle, Point
from fibsem.ui.stylesheets import NAPARI_STYLE
from fibsem.ui.tokens import (
    SEMANTIC_WARNING_COLOR,
)

_DEFAULT_AA = FibsemRectangle.from_dict(DEFAULT_ALIGNMENT_AREA)
_DEFAULT_POI = Point(0, 0)

_SECTION_STYLE = (
    "font-size: 10px; font-weight: bold; color: #707070;"
    " padding: 4px 0px 2px 0px; letter-spacing: 0.5px;"
)
_LABEL_STYLE = "color: #a0a0a0; min-width: 80px; font-size: 11px; background: transparent;"
_INVALID_STYLE = f"color: {SEMANTIC_WARNING_COLOR}; font-size: 10px; background: transparent;"
_HINT_STYLE = (
    "color: #808080; font-size: 10px; background: transparent;"
    " border-top: 1px solid #4a4a4a; padding-top: 6px;"
)

# Tells the user the preview is editable — without it the overlays look decorative
# and the spinboxes read as the only way in. Says what to drag, not what colour it
# is; the canvas legend already names the colours.
_CANVAS_HINT = (
    "Drag the box, or its corners, to set the alignment area.\n"
    "Drag the marker to set the point of interest.\n"
    "The values above stay in sync, both ways."
)

_EXAMPLE_PETNAME = "brave-tiger"

_AA_SPIN_W = 120   # holds a bare "0.20"
# Sized for the widest legal POI value, "-50.00 µm" — sign, digits and the suffix.
# Sizing for a typical "-3.00 µm" instead clips silently at the extremes, which is
# how the suffix used to render as "µn".
_POI_SPIN_W = 150

# ── Preview image settings ─────────────────────────────────────────────────────
_PREVIEW_HFW = 100e-6               # 100 µm horizontal field width
_PREVIEW_W, _PREVIEW_H = 384, 288
_PIXEL_SIZE = _PREVIEW_HFW / _PREVIEW_W
_PREVIEW_VFW = _PREVIEW_HFW * _PREVIEW_H / _PREVIEW_W  # vertical field width (metres)

# POI spinbox limits = half the preview frame, so every value the spinboxes accept
# lands inside the image. Previously ±5000 µm, 100x what a 100 µm frame can show:
# _poi_to_pixel clamps anything past the edge, so the marker parked on the border
# and displayed a position it did not have (500 µm and 5000 µm looked identical).
_POI_LIMIT_X_UM = _PREVIEW_HFW / 2 * 1e6   # 50.00
_POI_LIMIT_Y_UM = _PREVIEW_VFW / 2 * 1e6   # 37.50

_COLOR_AA = "limegreen"            # alignment area rectangle
_COLOR_POI = "magenta"             # point of interest marker

# Black background the overlays are drawn over (the canvas adds its own scalebar +
# centre crosshair from the pixel size, so only the AA rect + POI need overlays).
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


def _template_name(template: LamellaDefaultConfig) -> str:
    """The example lamella name a template would produce (for the preview hint)."""
    prefix = template.name_prefix or ""
    sep = "-" if prefix else ""
    if template.use_petname:
        return f"{prefix}{sep}01-{_EXAMPLE_PETNAME}"
    return f"{prefix}{sep}Lamella-01"


def _poi_to_pixel(poi: Point) -> Tuple[float, float]:
    """Microscope-coord POI offset (metres, +x right / +y up) → image pixel (top-left, y-down)."""
    px = (0.5 + poi.x / _PREVIEW_HFW) * _PREVIEW_W
    py = (0.5 - poi.y / _PREVIEW_VFW) * _PREVIEW_H
    px = max(0.0, min(px, _PREVIEW_W - 1))
    py = max(0.0, min(py, _PREVIEW_H - 1))
    return px, py


def _pixel_to_poi(px: float, py: float) -> Point:
    """Inverse of :func:`_poi_to_pixel` — image pixel → POI offset in metres."""
    x = (px / _PREVIEW_W - 0.5) * _PREVIEW_HFW
    y = (0.5 - py / _PREVIEW_H) * _PREVIEW_VFW
    return Point(x=x, y=y)


class LamellaDefaultConfigWidget(QWidget):
    """Compact editor for LamellaDefaultConfig protocol defaults.

    The right-hand preview is a live :class:`FibsemImageCanvas`: the alignment area
    (lime rectangle) and point of interest (magenta marker) are editable overlays
    that stay in two-way sync with the spinboxes — drag them on the canvas or type
    values, either updates the other.
    """

    template_changed = pyqtSignal(object)  # LamellaDefaultConfig

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(NAPARI_STYLE + "QCheckBox { background: transparent; }")
        self._setup_ui()
        self._connect_signals()
        # Position spinboxes + overlays + name hint from the defaults (silent — the
        # spinbox writes are blocked and the overlays don't echo programmatic sets).
        self.set_template(LamellaDefaultConfig())

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
        self.aa_left.setFixedWidth(_AA_SPIN_W)
        self.aa_top    = ValueSpinBox(minimum=0.0, maximum=1.0, decimals=2, step=0.01)
        self.aa_top.setFixedWidth(_AA_SPIN_W)
        self.aa_width  = ValueSpinBox(minimum=0.0, maximum=1.0, decimals=2, step=0.01)
        self.aa_width.setFixedWidth(_AA_SPIN_W)
        self.aa_height = ValueSpinBox(minimum=0.0, maximum=1.0, decimals=2, step=0.01)
        self.aa_height.setFixedWidth(_AA_SPIN_W)
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

        # Wider than the alignment-area boxes: those hold a bare "0.20", these hold a
        # signed value plus the µm suffix ("-50.00 µm"), which does not fit in 120px.
        self.poi_x = ValueSpinBox(suffix="µm", minimum=-_POI_LIMIT_X_UM, maximum=_POI_LIMIT_X_UM,
                                  decimals=2, step=0.1)
        self.poi_x.setFixedWidth(_POI_SPIN_W)
        self.poi_y = ValueSpinBox(suffix="µm", minimum=-_POI_LIMIT_Y_UM, maximum=_POI_LIMIT_Y_UM,
                                  decimals=2, step=0.1)
        self.poi_y.setFixedWidth(_POI_SPIN_W)
        self._poi_widgets = [self.poi_x, self.poi_y]

        _poi_tip = (
            "The point of interest is the initial target position within the image for\n"
            "the lamella workflow. It is defined as an offset from the image centre.\n"
            "Units: micrometres (µm). Positive X is right, positive Y is up.\n"
            f"Limited to the {_PREVIEW_HFW * 1e6:.0f} µm preview frame "
            f"(X ±{_POI_LIMIT_X_UM:.2f}, Y ±{_POI_LIMIT_Y_UM:.2f})."
        )
        self.poi_x.setToolTip(f"Horizontal offset from image centre.\n{_poi_tip}")
        self.poi_y.setToolTip(f"Vertical offset from image centre.\n{_poi_tip}")

        # One per row, unlike the paired alignment-area rows. Side by side, two boxes
        # wide enough for "-4888.88 µm" would push the whole panel out and shrink the
        # preview; stacked they fit inside the width the alignment rows already need,
        # using vertical space the panel has spare. Labels carry no unit — each
        # spinbox has the µm suffix.
        left.addLayout(_row("X", self.poi_x))
        left.addLayout(_row("Y", self.poi_y))

        # A protocol saved before the limits existed can carry a POI outside the frame.
        # setValue() clamps silently, which would rewrite the stored value the moment
        # the dialog opened — say so instead of quietly changing it.
        self.poi_clamped_lbl = QLabel()
        self.poi_clamped_lbl.setStyleSheet(_INVALID_STYLE)
        self.poi_clamped_lbl.setWordWrap(True)
        self.poi_clamped_lbl.setVisible(False)
        left.addWidget(self.poi_clamped_lbl)
        left.addStretch()

        # pinned below the stretch so it sits at the foot of the panel, clear of the controls
        self.canvas_hint_lbl = QLabel(_CANVAS_HINT)
        self.canvas_hint_lbl.setStyleSheet(_HINT_STYLE)
        self.canvas_hint_lbl.setWordWrap(True)
        left.addWidget(self.canvas_hint_lbl)

        self._params_panel = TitledPanel("Default Parameters", content=controls_widget, collapsible=False)
        self._params_panel.add_header_widget(self._btn_reset)
        root.addWidget(self._params_panel)

        # ── right column: live preview canvas + editable overlays ─────
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(4, 4, 4, 4)
        preview_layout.setSpacing(4)

        self.canvas = FibsemImageCanvas()
        self.canvas.setMinimumSize(_PREVIEW_W // 2, _PREVIEW_H // 2)
        # Config preview: keep only fit-to-view + scalebar + crosshair toggles;
        # drop contrast / ruler (mode is already hidden until an overlay arms it).
        self.canvas.btn_contrast.hide()
        self.canvas.btn_toggle_ruler.hide()
        # Seed the black background (gives the overlays their pixel dimensions and the
        # canvas its scalebar); then attach overlays. POI is added *before* the
        # alignment rect so its press handler wins when the marker sits inside the rect.
        self.canvas.set_array(_PREVIEW_BG, pixel_size=_PIXEL_SIZE)
        self._poi_overlay = PointOverlay(
            color=_COLOR_POI,
            selected_color=_COLOR_POI,
            marker="+",       # thin cross, reads like the (yellow) centre crosshair
            size=11.0,
            edge_width=1.2,   # thin lines (centre crosshair is linewidth=1)
            add_on_right_click=False,
            removable=False,
        )
        self._aa_overlay = AlignmentAreaOverlay(color=_COLOR_AA, editable=True)
        self.canvas.add_overlay(self._poi_overlay)
        self.canvas.add_overlay(self._aa_overlay)
        # Patch legend keying the three preview elements ("yellow" = the canvas's
        # built-in centre crosshair; see FibsemImageCanvas._refresh_crosshair).
        # Lower-left: name hint is upper-left, toolbar upper-right, scalebar lower-right.
        self.canvas.set_legend([
            ("yellow", "Image centre", "+"),      # matches the built-in crosshair glyph
            (_COLOR_AA, "Alignment area"),        # a region -> filled swatch
            (_COLOR_POI, "Point of interest", "+"),
        ], loc="lower left")
        preview_layout.addWidget(self.canvas)

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
        # Overlay → spinbox (only fires on user drag/resize, never on programmatic set)
        self._aa_overlay.alignment_area_changed.connect(self._on_overlay_alignment_changed)
        self._poi_overlay.point_dragging.connect(self._on_overlay_poi_dragging)
        self._poi_overlay.point_moved.connect(self._on_overlay_poi_moved)

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
        self._report_poi_clamped(poi)

        for w in [self.use_petname_cb, *self._aa_widgets, *self._poi_widgets]:
            w.blockSignals(False)
        self.name_prefix_edit.blockSignals(False)

        self._validate_alignment_area()
        self._sync_overlays(template)

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

    def _report_poi_clamped(self, poi: Point) -> None:
        """Warn if *poi* fell outside the preview frame and was clamped on load.

        The spinboxes bound the POI to the frame, so anything further in is silently
        pulled to the edge by setValue. Left unsaid, opening the dialog and pressing
        OK would rewrite a stored value the user never touched.
        """
        out = []
        if abs(poi.x * 1e6) > _POI_LIMIT_X_UM:
            out.append(f"X {poi.x * 1e6:.2f} → {self.poi_x.value():.2f} µm")
        if abs(poi.y * 1e6) > _POI_LIMIT_Y_UM:
            out.append(f"Y {poi.y * 1e6:.2f} → {self.poi_y.value():.2f} µm")
        if out:
            self.poi_clamped_lbl.setText(
                f"Outside the {_PREVIEW_HFW * 1e6:.0f} µm preview frame, clamped: "
                + ", ".join(out)
            )
        self.poi_clamped_lbl.setVisible(bool(out))

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

    def _sync_overlays(self, template: LamellaDefaultConfig) -> None:
        """Push a template onto the canvas overlays + name hint (no signals emitted)."""
        aa = template.alignment_area if template.alignment_area is not None else _DEFAULT_AA
        self._aa_overlay.set_area(aa)
        self._aa_overlay.set_visible(True)

        poi = template.poi if template.poi is not None else _DEFAULT_POI
        self._poi_overlay.set_points([_poi_to_pixel(poi)])

        self.canvas.set_hint(_template_name(template))

    # ── overlay → spinbox sync ────────────────────────────────────────

    def _on_overlay_alignment_changed(self, area: FibsemRectangle) -> None:
        """User dragged/resized the alignment rectangle on the canvas."""
        for sb in self._aa_widgets:
            sb.blockSignals(True)
        self.aa_left.setValue(area.left)
        self.aa_top.setValue(area.top)
        self.aa_width.setValue(area.width)
        self.aa_height.setValue(area.height)
        for sb in self._aa_widgets:
            sb.blockSignals(False)
        self._validate_alignment_area()
        self._emit_changed()

    def _on_overlay_poi_dragging(self, _idx: int, x: float, y: float) -> None:
        """Live spinbox feedback while dragging the POI marker (no emit)."""
        poi = _pixel_to_poi(x, y)
        for sb in self._poi_widgets:
            sb.blockSignals(True)
        self.poi_x.setValue(poi.x * 1e6)
        self.poi_y.setValue(poi.y * 1e6)
        for sb in self._poi_widgets:
            sb.blockSignals(False)

    def _on_overlay_poi_moved(self, _idx: int, x: float, y: float) -> None:
        """POI marker released — commit the value and notify listeners."""
        self._on_overlay_poi_dragging(_idx, x, y)  # ensure spinboxes match the release point
        self._emit_changed()

    # ── outward change notification ───────────────────────────────────

    def _emit_changed(self) -> None:
        """Emit template_changed without re-touching the overlays (avoids echo)."""
        self.template_changed.emit(self.get_template())

    def _on_changed(self) -> None:
        template = self.get_template()
        self._sync_overlays(template)
        self.template_changed.emit(template)
