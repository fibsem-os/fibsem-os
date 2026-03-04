from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Union

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from fibsem import utils
from fibsem.microscope import FibsemMicroscope
from fibsem.milling.base import FibsemMillingStage
from fibsem.milling.patterning import get_pattern, get_pattern_names
from fibsem.milling.patterning.patterns2 import BasePattern
from fibsem.structures import BeamType, Point
from fibsem.ui.widgets.custom_widgets import (
    FormRow,
    QFilePathLineEdit,
    ValueComboBox,
    ValueSpinBox,
)

_HIDDEN_FIELDS = {"shapes"}

_POINT_SCALE = 1e6      # µm display for x/y
_POINT_SUFFIX = "µm"


@dataclass
class _PointRow:
    """Special form row for the Point field (two spinboxes: x and y)."""
    label: QLabel
    x_control: ValueSpinBox
    y_control: ValueSpinBox
    field: str = "point"
    advanced: bool = False


class FibsemPatternSettingsWidget(QWidget):
    """Metadata-driven form widget for any BasePattern subclass.

    Selecting a different pattern type from the combobox rebuilds the form.
    """

    pattern_changed = pyqtSignal(object)  # BasePattern

    def __init__(
        self,
        microscope: FibsemMicroscope,
        pattern: BasePattern,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.microscope = microscope
        self._pattern = pattern
        self._advanced_visible = False
        self._rows: List[Union[FormRow, _PointRow]] = []

        self._setup_ui()
        self._connect_signals()
        self._build_controls(pattern)

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        # Pattern type selector — fixed row, not rebuilt
        type_form = QFormLayout()
        type_form.setContentsMargins(0, 0, 0, 0)
        self._type_combo = ValueComboBox(get_pattern_names(), value=self._pattern.name)
        type_form.addRow("Pattern:", self._type_combo)
        outer.addLayout(type_form)

        # Field form — rebuilt on type change
        self._fields_form = QFormLayout()
        self._fields_form.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(self._fields_form)

    def _connect_signals(self) -> None:
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)

    # ------------------------------------------------------------------
    # Control building
    # ------------------------------------------------------------------

    def _build_controls(self, pattern: BasePattern) -> None:
        """Clear and rebuild the fields form for the given pattern."""
        # Clear existing rows
        while self._fields_form.rowCount():
            self._fields_form.removeRow(0)
        self._rows.clear()

        for field_name, m in pattern.field_metadata.items():
            if field_name in _HIDDEN_FIELDS or m.get("hidden", False):
                continue

            advanced = m.get("advanced", False)
            label_text = m.get("label") or field_name.replace("_", " ").title()
            tooltip = m.get("tooltip", "")

            # --- Point: two spinboxes side-by-side ---
            if field_name == "point":
                point: Point = getattr(pattern, "point")
                x_ctrl = ValueSpinBox(
                    suffix=_POINT_SUFFIX,
                    minimum=m.get("minimum", -1000.0),
                    maximum=m.get("maximum", 1000.0),
                    step=m.get("step", 0.01),
                    decimals=m.get("decimals", 2),
                )
                x_ctrl.setValue(point.x * _POINT_SCALE)
                y_ctrl = ValueSpinBox(
                    suffix=_POINT_SUFFIX,
                    minimum=m.get("minimum", -1000.0),
                    maximum=m.get("maximum", 1000.0),
                    step=m.get("step", 0.01),
                    decimals=m.get("decimals", 2),
                )
                y_ctrl.setValue(point.y * _POINT_SCALE)

                point_widget = QWidget()
                point_layout = QHBoxLayout(point_widget)
                point_layout.setContentsMargins(0, 0, 0, 0)
                point_layout.setSpacing(4)
                point_layout.addWidget(x_ctrl)
                point_layout.addWidget(y_ctrl)

                label = QLabel(label_text)
                if tooltip:
                    label.setToolTip(tooltip)
                    point_widget.setToolTip(tooltip)
                self._fields_form.addRow(label, point_widget)

                row = _PointRow(label=label, x_control=x_ctrl, y_control=y_ctrl, advanced=advanced)
                x_ctrl.valueChanged.connect(self._on_changed)
                y_ctrl.valueChanged.connect(self._on_changed)
                self._rows.append(row)
                continue

            # --- All other fields ---
            value = getattr(pattern, field_name)
            items = m.get("items")
            type_ = m.get("type")
            base_scale = m.get("scale")
            dims = m.get("dimensions")
            effective_scale = (base_scale ** dims) if (base_scale and dims) else base_scale

            if items == "dynamic":
                cached = self.microscope.get_available_values_cached(
                    m["microscope_parameter"], BeamType.ION
                )
                items_list = cached if cached else [value]
                control = ValueComboBox(items_list, value, m.get("unit"))
            elif items:
                control = ValueComboBox(items, value, m.get("unit"))
            elif m.get("filepath"):
                control = QFilePathLineEdit()
                control.setText(str(value) if value else "")
            elif type_ is bool or isinstance(value, bool):
                control = QCheckBox()
                control.setChecked(bool(value))
            elif isinstance(value, (float, int)):
                suffix = (
                    utils._get_display_unit(base_scale, m.get("unit"))
                    if base_scale
                    else (m.get("unit") or "")
                )
                control = ValueSpinBox(
                    suffix, m.get("minimum"), m.get("maximum"), m.get("step"), m.get("decimals")
                )
                display_val = value * effective_scale if effective_scale else value
                control.setValue(display_val)
            else:
                continue  # unsupported type

            label = QLabel(label_text)
            if tooltip:
                label.setToolTip(tooltip)
                control.setToolTip(tooltip)
            self._fields_form.addRow(label, control)

            # Connect signal
            if isinstance(control, ValueComboBox):
                control.currentIndexChanged.connect(self._on_changed)
            elif isinstance(control, ValueSpinBox):
                control.valueChanged.connect(self._on_changed)
            elif isinstance(control, QCheckBox):
                control.toggled.connect(self._on_changed)
            elif isinstance(control, QFilePathLineEdit):
                control.editingFinished.connect(self._on_changed)

            self._rows.append(FormRow(
                label=label,
                control=control,
                field=field_name,
                advanced=advanced,
                scale=effective_scale,
            ))

        self._update_visibility()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_type_changed(self) -> None:
        name = self._type_combo.value()
        if name is None or name == self._pattern.name:
            return
        new_pattern = get_pattern(name)
        self._pattern = new_pattern
        self._build_controls(new_pattern)
        self.pattern_changed.emit(new_pattern)

    def _on_changed(self) -> None:
        self.pattern_changed.emit(self.get_pattern())

    # ------------------------------------------------------------------
    # Visibility
    # ------------------------------------------------------------------

    def _update_visibility(self) -> None:
        for row in self._rows:
            adv_ok = (not row.advanced) or self._advanced_visible
            if isinstance(row, _PointRow):
                row.label.setVisible(adv_ok)
                row.x_control.setVisible(adv_ok)
                row.y_control.setVisible(adv_ok)
            else:
                row.label.setVisible(adv_ok)
                row.control.setVisible(adv_ok)

    def set_advanced_visible(self, show: bool) -> None:
        self._advanced_visible = show
        self._update_visibility()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_pattern(self) -> BasePattern:
        pattern = deepcopy(self._pattern)
        for row in self._rows:
            if isinstance(row, _PointRow):
                x = row.x_control.value() / _POINT_SCALE
                y = row.y_control.value() / _POINT_SCALE
                pattern.point = Point(x=x, y=y)
            elif isinstance(row.control, ValueComboBox):
                data = row.control.value()
                if data is not None:
                    setattr(pattern, row.field, data)
            elif isinstance(row.control, ValueSpinBox):
                val = row.control.value()
                setattr(pattern, row.field, val / row.scale if row.scale else val)
            elif isinstance(row.control, QCheckBox):
                setattr(pattern, row.field, row.control.isChecked())
            elif isinstance(row.control, QFilePathLineEdit):
                setattr(pattern, row.field, row.control.text())
        return pattern

    def set_pattern(self, pattern: BasePattern) -> None:
        type_changed = pattern.name != self._pattern.name
        self._pattern = pattern

        if type_changed:
            self._type_combo.blockSignals(True)
            self._type_combo.set_value(pattern.name)
            self._type_combo.blockSignals(False)
            self._build_controls(pattern)
            return  # _build_controls already reads values from pattern

        # Same type — just update control values
        for row in self._rows:
            if isinstance(row, _PointRow):
                row.x_control.blockSignals(True)
                row.y_control.blockSignals(True)
                row.x_control.setValue(pattern.point.x * _POINT_SCALE)
                row.y_control.setValue(pattern.point.y * _POINT_SCALE)
                row.x_control.blockSignals(False)
                row.y_control.blockSignals(False)
            else:
                row.control.blockSignals(True)
        for row in self._rows:
            if isinstance(row, _PointRow):
                continue
            value = getattr(pattern, row.field)
            if isinstance(row.control, ValueComboBox):
                row.control.set_value(value)
            elif isinstance(row.control, ValueSpinBox):
                row.control.setValue(value * row.scale if row.scale else value)
            elif isinstance(row.control, QCheckBox):
                row.control.setChecked(bool(value))
            elif isinstance(row.control, QFilePathLineEdit):
                row.control.setText(str(value) if value else "")
        for row in self._rows:
            if not isinstance(row, _PointRow):
                row.control.blockSignals(False)
