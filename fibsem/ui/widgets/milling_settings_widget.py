from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QFormLayout,
    QLabel,
    QWidget,
)

from fibsem.microscope import FibsemMicroscope
from fibsem.structures import BeamType, FibsemMillingSettings
from fibsem.ui.widgets.form_builder import Control, build_control


@dataclass
class _Row:
    """One built form row. `mfr` is this form's own twist: a field declared for
    one manufacturer is hidden on the others."""

    label: QLabel
    control: Control
    field: str
    advanced: bool
    mfr: Optional[str]


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
        self._rows: List[_Row] = []
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
            if m.get("hidden", False):
                continue

            control = build_control(
                m,
                getattr(self._settings, field_name),
                dynamic_items=self._dynamic_items,
            )
            if control is None:
                continue

            label = QLabel(m.get("label") or field_name.replace("_", " ").title())
            if m.get("tooltip"):
                label.setToolTip(m["tooltip"])
                control.widget.setToolTip(m["tooltip"])

            layout.addRow(label, control.widget)
            self._rows.append(
                _Row(
                    label=label,
                    control=control,
                    field=field_name,
                    advanced=m.get("advanced", False),
                    mfr=m.get("manufacturer"),
                )
            )

    def _dynamic_items(self, parameter: str):
        """Resolve an `items: "dynamic"` field against the microscope."""
        return self.microscope.get_available_values_cached(parameter, BeamType.ION)

    def _connect_signals(self) -> None:
        for row in self._rows:
            row.control.connect(self._on_changed)

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
            row.control.widget.setVisible(mfr_ok and adv_ok)

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

        # Hidden fields have no control, so they pass through untouched rather
        # than reverting to the dataclass default.
        for field_name in _HIDDEN_FIELDS:
            kwargs[field_name] = getattr(self._settings, field_name)

        for row in self._rows:
            value = row.control.read()
            # A combobox with nothing selected reads None; keep what we had
            # rather than handing the constructor a hole.
            kwargs[row.field] = (
                value if value is not None else getattr(self._settings, row.field)
            )

        return FibsemMillingSettings(**kwargs)

    def set_settings(self, settings: FibsemMillingSettings) -> None:
        self._settings = settings

        for row in self._rows:
            row.control.set_blocked(True)
        try:
            for row in self._rows:
                row.control.write(getattr(settings, row.field))
        finally:
            for row in self._rows:
                row.control.set_blocked(False)
