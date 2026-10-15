from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QLabel,
    QWidget,
)

from fibsem.devices.core import ParameterMetadata
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import BeamType, FibsemMillingSettings, FieldSpec
from fibsem.ui.utils import beam_choices
from fibsem.ui.widgets.custom_widgets import FormGrid, align_form
from fibsem.ui.widgets.form_builder import Control, build_control


@dataclass
class _Row:
    """One built form row."""

    label: QLabel
    control: Control
    field: str
    advanced: bool


_SPECS = FibsemMillingSettings().field_specs

# Fields hidden from UI — derived from their display (hidden=True), not hardcoded
_HIDDEN_FIELDS = {name for name, spec in _SPECS.items() if spec.hidden}


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

        for field_name, spec in _SPECS.items():
            if spec.hidden:
                continue

            control = build_control(
                self._instrument_spec(spec),
                getattr(self._settings, field_name),
                resolve_choices=self._instrument_choices,
                read_only_fallback=False,
            )
            if control is None:
                continue

            label = QLabel(spec.label or field_name.replace("_", " ").title())
            if spec.doc:
                label.setToolTip(spec.doc)
                control.widget.setToolTip(spec.doc)

            layout.addRow(label, control.widget)
            self._rows.append(
                _Row(
                    label=label,
                    control=control,
                    field=field_name,
                    advanced=spec.advanced,
                )
            )

        align_form(layout)

    def _instrument_spec(self, spec: FieldSpec) -> FieldSpec:
        """The field with the milling service's choices and limits in place of its
        fixed ones, where the service reports them."""
        reported = (self._supported or {}).get(spec.name)
        if reported is None:
            return spec
        if spec.choices is None:
            # a number box stays one; the instrument's choices come through
            # _instrument_choices
            reported = replace(reported, choices=None)
        return spec.with_reported(reported)

    def _instrument_choices(self, parameter: str):
        """The choices of a field the instrument decides: the milling service's for
        the field, else the milling beam's for its parameter."""
        if self._supported is not None:
            for name, spec in _SPECS.items():
                if spec.microscope_parameter == parameter and name in self._supported:
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
