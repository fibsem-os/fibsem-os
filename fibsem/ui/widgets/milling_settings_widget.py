from __future__ import annotations

from typing import Dict, List, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QFormLayout,
    QLabel,
    QWidget,
)

from fibsem import utils
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import BeamType, FibsemMillingSettings
from fibsem.ui.widgets.custom_widgets import FormRow, ValueComboBox, ValueSpinBox

_META = FibsemMillingSettings().field_metadata

# Fields hidden from UI — derived from metadata (hidden=True), not hardcoded
_HIDDEN_FIELDS = {name for name, m in _META.items() if m.get("hidden", False)}



class FibsemMillingSettingsWidget(QWidget):
    settings_changed = pyqtSignal(object)  # FibsemMillingSettings

    def __init__(
        self,
        microscope: FibsemMicroscope,
        settings: FibsemMillingSettings,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.microscope = microscope
        self._manufacturer: str = microscope.manufacturer
        self._settings = settings
        self._advanced_visible = False
        self._rows: List[FormRow] = []
        self._setup_ui()
        self._connect_signals()
        self._update_visibility()
        self.set_settings(settings)

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        for field_name, m in _META.items():
            if field_name in _HIDDEN_FIELDS or m.get("hidden", False):
                continue

            value = getattr(self._settings, field_name)
            mfr = m.get("manufacturer")
            advanced = m.get("advanced", False)
            items = m.get("items")

            # Compute effective scale (applies dimensions for cubic/area units)
            base_scale = m.get("scale")
            dims = m.get("dimensions")
            effective_scale = (base_scale ** dims) if (base_scale and dims) else base_scale

            # Create control
            if items == "dynamic":
                cached = self.microscope.get_available_values_cached(m["microscope_parameter"], BeamType.ION)
                items_list = cached if cached else [value]
                control = ValueComboBox(items_list, value, m.get("unit"))
                attr_name = f"{field_name}_combo"
            elif items:
                control = ValueComboBox(items, value, m.get("unit"))
                attr_name = f"{field_name}_combo"
            elif isinstance(value, (float, int)):
                # Display suffix uses base scale (not effective) for correct SI prefix
                suffix = utils._get_display_unit(base_scale, m.get("unit")) if base_scale else (m.get("unit") or "")
                control = ValueSpinBox(suffix, m.get("minimum"), m.get("maximum"), m.get("step"), m.get("decimals"))
                attr_name = f"{field_name}_spinbox"
            else:
                continue  # unsupported type

            label_text = m.get("label") or field_name.replace("_", " ").title()
            label = QLabel(label_text)
            if m.get("tooltip"):
                label.setToolTip(m["tooltip"])
                control.setToolTip(m["tooltip"])

            layout.addRow(label, control)
            setattr(self, attr_name, control)
            setattr(self, f"{field_name}_label", label)

            self._rows.append(FormRow(
                label=label,
                control=control,
                field=field_name,
                mfr=mfr,
                advanced=advanced,
                scale=effective_scale,
            ))

    def _connect_signals(self) -> None:
        for row in self._rows:
            if isinstance(row.control, ValueComboBox):
                row.control.currentIndexChanged.connect(self._on_changed)
            elif isinstance(row.control, ValueSpinBox):
                row.control.valueChanged.connect(self._on_changed)

    def _on_changed(self) -> None:
        self.settings_changed.emit(self.get_settings())

    # ------------------------------------------------------------------
    # Visibility
    # ------------------------------------------------------------------

    def _update_visibility(self) -> None:
        for row in self._rows:
            mfr_ok = (row.mfr is None) or (row.mfr == self._manufacturer)
            adv_ok = (not row.advanced) or self._advanced_visible
            row.label.setVisible(mfr_ok and adv_ok)
            row.control.setVisible(mfr_ok and adv_ok)

    def set_manufacturer(self, manufacturer: Optional[str]) -> None:
        self._manufacturer = manufacturer or ""
        self._update_visibility()

    def set_advanced_visible(self, visible: bool) -> None:
        self._advanced_visible = visible
        self._update_visibility()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_settings(self) -> FibsemMillingSettings:
        kwargs: Dict[str, object] = {}

        # Pass through hidden fields unchanged
        for field_name in _HIDDEN_FIELDS:
            kwargs[field_name] = getattr(self._settings, field_name)

        for row in self._rows:
            if isinstance(row.control, ValueComboBox):
                data = row.control.value()
                kwargs[row.field] = data if data is not None else getattr(self._settings, row.field)
            elif isinstance(row.control, ValueSpinBox):
                val = row.control.value()
                kwargs[row.field] = val / row.scale if row.scale else val

        return FibsemMillingSettings(**kwargs)

    def set_settings(self, settings: FibsemMillingSettings) -> None:
        self._settings = settings

        for row in self._rows:
            row.control.blockSignals(True)

        for row in self._rows:
            value = getattr(settings, row.field)
            if isinstance(row.control, ValueComboBox):
                row.control.set_value(value)
            elif isinstance(row.control, ValueSpinBox):
                row.control.setValue(value * row.scale if row.scale else value)

        for row in self._rows:
            row.control.blockSignals(False)
