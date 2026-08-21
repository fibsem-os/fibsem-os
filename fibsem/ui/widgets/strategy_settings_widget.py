from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, List, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from fibsem.milling.base import MillingStrategy, get_strategy
from fibsem.milling.strategy import get_strategy_names
from fibsem.ui.tokens import (
    NEUTRAL_700,
)
from fibsem.ui.widgets.custom_widgets import ValueComboBox
from fibsem.ui.widgets.form_builder import Control, build_control


@dataclass
class _Row:
    """One built form row: the label, the control, and whether it is advanced."""

    label: QLabel
    control: Control
    field: str
    advanced: bool


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
        self._rows: List[_Row] = []

        self._setup_ui()
        self._connect_signals()
        self._build_controls(strategy)
        self._apply_strategy_type_selector(strategy)

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
        self._type_combo = ValueComboBox(
            get_strategy_names(), value=self._strategy.name
        )
        type_form.addRow("Strategy:", self._type_combo)
        outer.addLayout(type_form)

        # Config field form — rebuilt on type change
        self._config_form = QFormLayout()
        self._config_form.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(self._config_form)

        # Empty-state label (shown when strategy has no config fields)
        self._empty_label = QLabel("No configuration options.")
        self._empty_label.setStyleSheet(f"color: {NEUTRAL_700}; font-style: italic;")
        self._empty_label.setVisible(False)
        outer.addWidget(self._empty_label)

    def _connect_signals(self) -> None:
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)

    def _apply_strategy_type_selector(self, strategy: MillingStrategy[Any]) -> None:
        """Populate the strategy-type selector for *strategy*.

        Strategies absent from the selectable registry (e.g. Coincidence, which is
        special and FM-bound) are locked: show only that strategy and disable
        switching. Normal builtins show the full list and stay switchable.
        """
        names = get_strategy_names()
        locked = strategy.name not in names
        self._type_combo.blockSignals(True)
        self._type_combo.clear()
        for name in [strategy.name] if locked else names:
            self._type_combo.addItem(name, name)
        self._type_combo.set_value(strategy.name)
        self._type_combo.blockSignals(False)
        self._type_combo.setEnabled(not locked)

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

        for field_name, m in strategy.config.field_metadata.items():
            if m.get("hidden", False):
                continue

            control = build_control(m, getattr(strategy.config, field_name))
            if control is None:
                logging.warning("Control for '%s' is unsupported", field_name)
                continue

            label = QLabel(m.get("label") or field_name.replace("_", " ").title())
            tooltip = m.get("tooltip", "")
            if tooltip:
                label.setToolTip(tooltip)
                control.widget.setToolTip(tooltip)

            self._config_form.addRow(label, control.widget)
            control.connect(self._on_changed)
            self._rows.append(
                _Row(
                    label=label,
                    control=control,
                    field=field_name,
                    advanced=m.get("advanced", False),
                )
            )

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
            row.control.widget.setVisible(adv_ok)

    def set_advanced_visible(self, show: bool) -> None:
        self._advanced_visible = show
        self._update_visibility()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_strategy(self) -> MillingStrategy[Any]:
        strategy = deepcopy(self._strategy)
        for row in self._rows:
            value = row.control.read()
            # A combobox with nothing selected reads None; writing that would
            # replace a real setting with nothing.
            if value is not None:
                setattr(strategy.config, row.field, value)
        return strategy

    def set_strategy(self, strategy: MillingStrategy[Any]) -> None:
        type_changed = strategy.name != self._strategy.name
        self._strategy = strategy
        self._apply_strategy_type_selector(strategy)

        if type_changed:
            self._build_controls(strategy)
            return  # _build_controls reads values from strategy.config directly

        # Every control is blocked before any is written, so a mid-write signal
        # cannot read a form that is half updated.
        for row in self._rows:
            row.control.set_blocked(True)
        try:
            for row in self._rows:
                row.control.write(getattr(strategy.config, row.field))
        finally:
            for row in self._rows:
                row.control.set_blocked(False)
