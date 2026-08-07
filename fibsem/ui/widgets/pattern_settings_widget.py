from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from fibsem.microscope import FibsemMicroscope
from fibsem.milling.patterning import get_pattern, get_pattern_names
from fibsem.milling.patterning.patterns2 import BasePattern
from fibsem.structures import BeamType
from fibsem.ui.widgets.custom_widgets import ValueComboBox
from fibsem.ui.widgets.form_builder import Control, build_control


@dataclass
class _Row:
    """One built form row: the label, the control, and whether it is advanced."""
    label: QLabel
    control: Control
    field: str
    advanced: bool


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
        self._rows: List[_Row] = []

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
        # Clear rows BEFORE removing form widgets: if Qt moves focus during removeRow(),
        # a re-entrant set_pattern() call would iterate self._rows and call
        # blockSignals() on already-deleted C++ wrappers → seg fault.
        self._rows.clear()
        while self._fields_form.rowCount():
            self._fields_form.removeRow(0)

        for field_name, m in pattern.field_metadata.items():
            if m.get("hidden", False):
                continue

            control = build_control(
                m,
                getattr(pattern, field_name),
                dynamic_items=self._dynamic_items,
            )
            if control is None:
                logging.warning("Control for '%s' is unsupported", field_name)
                continue

            label = QLabel(m.get("label") or field_name.replace("_", " ").title())
            tooltip = m.get("tooltip", "")
            if tooltip:
                label.setToolTip(tooltip)
                control.widget.setToolTip(tooltip)

            self._fields_form.addRow(label, control.widget)
            control.connect(self._on_changed)
            self._rows.append(
                _Row(
                    label=label,
                    control=control,
                    field=field_name,
                    advanced=m.get("advanced", False),
                )
            )

        self._update_visibility()

    def _dynamic_items(self, parameter: str):
        """Resolve an `items: "dynamic"` field against the microscope."""
        return self.microscope.get_available_values_cached(parameter, BeamType.ION)

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
            row.label.setVisible(adv_ok)
            row.control.widget.setVisible(adv_ok)

    def set_advanced_visible(self, show: bool) -> None:
        self._advanced_visible = show
        self._update_visibility()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_pattern(self) -> BasePattern:
        pattern = deepcopy(self._pattern)
        for row in self._rows:
            value = row.control.read()
            # A combobox with nothing selected reads None; writing that would
            # replace a real setting with nothing.
            if value is not None:
                setattr(pattern, row.field, value)
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

        # Same type — just update control values. Every control is blocked
        # before any is written, so a mid-write signal cannot read a form that
        # is half updated.
        for row in self._rows:
            row.control.set_blocked(True)
        try:
            for row in self._rows:
                row.control.write(getattr(pattern, row.field))
        finally:
            for row in self._rows:
                row.control.set_blocked(False)
