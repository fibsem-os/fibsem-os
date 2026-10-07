from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QLabel,
    QWidget,
)

from fibsem.devices.core import ParameterMetadata
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import BeamType, FibsemMillingSettings, RangeLimit
from fibsem.ui.utils import beam_choices
from fibsem.ui.widgets.custom_widgets import FormGrid, align_form
from fibsem.ui.widgets.form_builder import Control, build_control, effective_scale


@dataclass
class _Row:
    """One built form row."""

    label: QLabel
    control: Control
    field: str
    advanced: bool


_META = FibsemMillingSettings().field_metadata

# Fields hidden from UI — derived from metadata (hidden=True), not hardcoded
_HIDDEN_FIELDS = {name for name, m in _META.items() if m.get("hidden", False)}


def _supported_settings(
    microscope: FibsemMicroscope,
) -> Optional[Dict[str, ParameterMetadata]]:
    """The recipe fields the microscope's milling service mills with on the ion beam,
    or None when it has no milling service (no ion beam, so nothing mills)."""
    milling = microscope.milling
    if milling is None:
        return None
    return milling.supported_settings(BeamType.ION)


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
        self._settings = settings
        self._advanced_visible = False
        self._rows: List[_Row] = []
        # The fields the instrument mills with and their choices, from its milling
        # service; None without one, which shows every field.
        self._supported = _supported_settings(microscope)
        self._setup_ui()
        self._connect_signals()
        self._update_visibility()
        self.set_settings(settings)

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = FormGrid(self)
        layout.setContentsMargins(0, 0, 0, 0)

        for field_name, m in _META.items():
            if m.get("hidden", False):
                continue

            control = build_control(
                self._instrument_metadata(field_name, m),
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
                )
            )

        align_form(layout)

    def _instrument_metadata(self, field_name: str, m: dict) -> dict:
        """The field's metadata with the milling service's choices and limits in
        place of its fixed ones, where the service reports them."""
        reported = (self._supported or {}).get(field_name)
        if reported is None:
            return m
        m = dict(m)
        if reported.choices is not None and m.get("items") not in (None, "dynamic"):
            m["items"] = list(reported.choices)
        if isinstance(reported.limits, RangeLimit):
            scale = effective_scale(m) or 1
            m["minimum"] = reported.limits.min * scale
            m["maximum"] = reported.limits.max * scale
        return m

    def _dynamic_items(self, parameter: str):
        """Resolve an `items: "dynamic"` field: the milling service's choices for the
        field, else the milling beam's choices for its parameter."""
        if self._supported is not None:
            for name, m in _META.items():
                if (
                    m.get("microscope_parameter") == parameter
                    and name in self._supported
                ):
                    choices = self._supported[name].choices
                    if choices is not None:
                        return list(choices)
        return beam_choices(self.microscope, parameter, BeamType.ION)

    def _connect_signals(self) -> None:
        for row in self._rows:
            row.control.connect(self._on_changed)

    def _on_changed(self) -> None:
        self.settings_changed.emit(self.get_settings())

    # ------------------------------------------------------------------
    # Visibility
    # ------------------------------------------------------------------

    def _update_visibility(self) -> None:
        # The milling service says which fields this instrument mills with; with no
        # service (no ion beam) nothing mills, so no field is shown.
        for row in self._rows:
            used = self._supported is not None and row.field in self._supported
            adv_ok = (not row.advanced) or self._advanced_visible
            row.label.setVisible(used and adv_ok)
            row.control.widget.setVisible(used and adv_ok)

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
