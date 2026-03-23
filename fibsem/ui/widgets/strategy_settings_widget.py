from __future__ import annotations

from copy import deepcopy
from typing import Any, List, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from fibsem import utils
from fibsem.milling.base import MillingStrategy, get_strategy
from fibsem.milling.strategy import get_strategy_names
from fibsem.ui.widgets.custom_widgets import FormRow, ValueComboBox, ValueSpinBox


class FibsemStrategySettingsWidget(QWidget):
    """Metadata-driven form widget for any MillingStrategy subclass.

    Selecting a different strategy type from the combobox rebuilds the config form.
    Strategies with no config fields (e.g. Standard) show an empty-state label.
    """

    strategy_changed = pyqtSignal(object)  # MillingStrategy[Any]

    def __init__(
        self,
        strategy: MillingStrategy[Any],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._strategy = strategy
        self._advanced_visible = False
        self._rows: List[FormRow] = []

        self._setup_ui()
        self._connect_signals()
        self._build_controls(strategy)

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        # Strategy type selector — fixed, not rebuilt
        type_form = QFormLayout()
        type_form.setContentsMargins(0, 0, 0, 0)
        self._type_combo = ValueComboBox(get_strategy_names(), value=self._strategy.name)
        type_form.addRow("Strategy:", self._type_combo)
        outer.addLayout(type_form)

        # Config field form — rebuilt on type change
        self._config_form = QFormLayout()
        self._config_form.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(self._config_form)

        # Empty-state label (shown when strategy has no config fields)
        self._empty_label = QLabel("No configuration options.")
        self._empty_label.setStyleSheet("color: #606060; font-style: italic;")
        self._empty_label.setVisible(False)
        outer.addWidget(self._empty_label)

    def _connect_signals(self) -> None:
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)

    # ------------------------------------------------------------------
    # Control building
    # ------------------------------------------------------------------

    def _build_controls(self, strategy: MillingStrategy[Any]) -> None:
        """Clear and rebuild the config form for the given strategy."""
        # Clear rows BEFORE removing form widgets (same reason as pattern_settings_widget:
        # prevents re-entrant access to zombie C++ wrappers during focus-change events).
        self._rows.clear()
        while self._config_form.rowCount():
            self._config_form.removeRow(0)

        meta = strategy.config.field_metadata
        hidden_fields = {name for name, m in meta.items() if m.get("hidden", False)}

        for field_name, m in meta.items():
            if field_name in hidden_fields or m.get("hidden", False):
                continue

            value = getattr(strategy.config, field_name)
            advanced = m.get("advanced", False)
            items = m.get("items")
            type_ = m.get("type")
            base_scale = m.get("scale")
            dims = m.get("dimensions")
            effective_scale = (base_scale ** dims) if (base_scale and dims) else base_scale

            if items:
                control = ValueComboBox(
                    items, value, m.get("unit"), format_fn=m.get("format_fn")
                )
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
                control.setValue(value * effective_scale if effective_scale else value)
            else:
                continue  # unsupported type

            label_text = m.get("label") or field_name.replace("_", " ").title()
            tooltip = m.get("tooltip", "")
            label = QLabel(label_text)
            if tooltip:
                label.setToolTip(tooltip)
                control.setToolTip(tooltip)

            self._config_form.addRow(label, control)

            if isinstance(control, ValueComboBox):
                control.currentIndexChanged.connect(self._on_changed)
            elif isinstance(control, ValueSpinBox):
                control.valueChanged.connect(self._on_changed)
            elif isinstance(control, QCheckBox):
                control.toggled.connect(self._on_changed)

            self._rows.append(FormRow(
                label=label,
                control=control,
                field=field_name,
                advanced=advanced,
                scale=effective_scale,
            ))

        self._empty_label.setVisible(len(self._rows) == 0)
        self._update_visibility()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_type_changed(self) -> None:
        name = self._type_combo.value()
        if name is None or name == self._strategy.name:
            return
        new_strategy = get_strategy(name)
        self._strategy = new_strategy
        self._build_controls(new_strategy)
        self.strategy_changed.emit(new_strategy)

    def _on_changed(self) -> None:
        self.strategy_changed.emit(self.get_strategy())

    # ------------------------------------------------------------------
    # Visibility
    # ------------------------------------------------------------------

    def _update_visibility(self) -> None:
        for row in self._rows:
            adv_ok = (not row.advanced) or self._advanced_visible
            row.label.setVisible(adv_ok)
            row.control.setVisible(adv_ok)

    def set_advanced_visible(self, show: bool) -> None:
        self._advanced_visible = show
        self._update_visibility()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_strategy(self) -> MillingStrategy[Any]:
        strategy = deepcopy(self._strategy)
        for row in self._rows:
            if isinstance(row.control, ValueComboBox):
                data = row.control.value()
                if data is not None:
                    setattr(strategy.config, row.field, data)
            elif isinstance(row.control, ValueSpinBox):
                val = row.control.value()
                setattr(strategy.config, row.field, val / row.scale if row.scale else val)
            elif isinstance(row.control, QCheckBox):
                setattr(strategy.config, row.field, row.control.isChecked())
        return strategy

    def set_strategy(self, strategy: MillingStrategy[Any]) -> None:
        type_changed = strategy.name != self._strategy.name
        self._strategy = strategy

        if type_changed:
            self._type_combo.blockSignals(True)
            self._type_combo.set_value(strategy.name)
            self._type_combo.blockSignals(False)
            self._build_controls(strategy)
            return  # _build_controls reads values from strategy.config directly

        for row in self._rows:
            row.control.blockSignals(True)
        for row in self._rows:
            value = getattr(strategy.config, row.field)
            if isinstance(row.control, ValueComboBox):
                row.control.set_value(value)
            elif isinstance(row.control, ValueSpinBox):
                row.control.setValue(value * row.scale if row.scale else value)
            elif isinstance(row.control, QCheckBox):
                row.control.setChecked(bool(value))
        for row in self._rows:
            row.control.blockSignals(False)
