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
from fibsem.fm.strategy import DEFAULT_STRATEGY, get_strategy_names
from fibsem.fm.strategy.base import AutoFocusStrategy, AutoFocusStrategyConfig, get_autofocus_strategy
from fibsem.ui.widgets.custom_widgets import FormRow, ValueComboBox, ValueSpinBox


class AutoFocusStrategyWidget(QWidget):
    """Metadata-driven form widget for any AutoFocusStrategy.

    A combobox lets the user select from available strategies; the config form
    below is rebuilt automatically to show that strategy's parameters.

    Signals:
        config_changed(AutoFocusStrategyConfig): emitted whenever the selected
            strategy or any config value changes.
    """

    config_changed = pyqtSignal(object)  # AutoFocusStrategyConfig

    def __init__(
        self,
        config: Optional[AutoFocusStrategyConfig] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._config: AutoFocusStrategyConfig = config or DEFAULT_STRATEGY().config
        self._rows: List[FormRow] = []

        self._setup_ui()
        self._connect_signals()
        self._build_controls(self._config)

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        type_form = QFormLayout()
        type_form.setContentsMargins(0, 0, 0, 0)
        self._type_combo = ValueComboBox(get_strategy_names(), value=self._config.name)
        type_form.addRow("Strategy:", self._type_combo)
        outer.addLayout(type_form)

        self._config_form = QFormLayout()
        self._config_form.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(self._config_form)

        self._empty_label = QLabel("No configuration options.")
        self._empty_label.setStyleSheet("color: #606060; font-style: italic;")
        self._empty_label.setVisible(False)
        outer.addWidget(self._empty_label)

    def _connect_signals(self) -> None:
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)

    # ------------------------------------------------------------------
    # Control building
    # ------------------------------------------------------------------

    def _build_controls(self, config: AutoFocusStrategyConfig) -> None:
        """Clear and rebuild the config form for the given config."""
        self._rows.clear()
        while self._config_form.rowCount():
            self._config_form.removeRow(0)

        meta = config.field_metadata

        for field_name, m in meta.items():
            if m.get("hidden", False):
                continue

            value = getattr(config, field_name)
            advanced = m.get("advanced", False)
            items = m.get("items")
            type_ = m.get("type")
            base_scale = m.get("scale")
            dims = m.get("dimensions")
            effective_scale = (base_scale ** dims) if (base_scale and dims) else base_scale

            if items:
                control = ValueComboBox(items, value, format_fn=m.get("format_fn"))
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
                    suffix,
                    m.get("minimum"),
                    m.get("maximum"),
                    m.get("step"),
                    m.get("decimals"),
                )
                control.setValue(value * effective_scale if effective_scale else value)
            else:
                continue  # unsupported type — skip silently

            label_text = m.get("label") or field_name.replace("_", " ").title()
            tooltip = m.get("tooltip") or ""
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

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_type_changed(self) -> None:
        name = self._type_combo.value()
        if name is None or name == self._config.name:
            return
        strategy = get_autofocus_strategy(name)
        self._config = strategy.config
        self._build_controls(self._config)
        self.config_changed.emit(self._config)

    def _on_changed(self) -> None:
        self.config_changed.emit(self.get_config())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_config(self) -> AutoFocusStrategyConfig:
        """Read the current form values and return an updated config."""
        config = deepcopy(self._config)
        meta = config.field_metadata
        for row in self._rows:
            type_ = meta.get(row.field, {}).get("type")
            if isinstance(row.control, ValueComboBox):
                data = row.control.value()
                if data is not None:
                    setattr(config, row.field, data)
            elif isinstance(row.control, ValueSpinBox):
                val = row.control.value()
                val = val / row.scale if row.scale else val
                if type_ is int:
                    val = int(round(val))
                setattr(config, row.field, val)
            elif isinstance(row.control, QCheckBox):
                setattr(config, row.field, row.control.isChecked())
        return config

    def set_config(self, config: AutoFocusStrategyConfig) -> None:
        """Populate the widget from a config instance."""
        type_changed = config.name != self._config.name
        self._config = config

        if type_changed:
            self._type_combo.blockSignals(True)
            self._type_combo.set_value(config.name)
            self._type_combo.blockSignals(False)
            self._build_controls(config)
            return  # _build_controls reads values from config directly

        for row in self._rows:
            row.control.blockSignals(True)
        for row in self._rows:
            value = getattr(config, row.field)
            if isinstance(row.control, ValueComboBox):
                row.control.set_value(value)
            elif isinstance(row.control, ValueSpinBox):
                row.control.setValue(value * row.scale if row.scale else value)
            elif isinstance(row.control, QCheckBox):
                row.control.setChecked(bool(value))
        for row in self._rows:
            row.control.blockSignals(False)
