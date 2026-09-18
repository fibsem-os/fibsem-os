from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QLabel,
    QWidget,
)

from fibsem.manufacturers import normalize_manufacturer
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import BeamType, FibsemMillingSettings
from fibsem.ui.widgets.custom_widgets import FormGrid, align_form
from fibsem.ui.widgets.form_builder import Control, build_control


@dataclass
class _Row:
    """One built form row. `mfr` is this form's own twist: a field declared for
    one manufacturer is hidden on the others it knows about."""

    label: QLabel
    control: Control
    field: str
    advanced: bool
    mfr: Optional[str]


_META = FibsemMillingSettings().field_metadata

# Fields hidden from UI — derived from metadata (hidden=True), not hardcoded
_HIDDEN_FIELDS = {name for name, m in _META.items() if m.get("hidden", False)}

# The manufacturers the field tags know about, derived from the tags themselves so a
# newly tagged field cannot be left out of it. A tagged field hides on these and on
# nothing else: an instrument this vocabulary says nothing about gets the whole form.
#
# The alternative is what shipped, and it emptied the form. `row.mfr == manufacturer`
# with no else hid every tagged field at once on a JEOL system, because neither of the
# two strings matched -- eight of the nine rows the form builds, including every way it
# offers of setting a milling current. A form with a few inapplicable rows is a smaller
# failure than a form with no rows, and the tag is a proxy for a question only the
# driver can answer (FIB-975, FIB-1011).
_TAGGED_MANUFACTURERS = frozenset(
    m["manufacturer"] for m in _META.values() if m.get("manufacturer")
)


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
        # Normalised on the way in, here and in `set_manufacturer`, because the tags
        # are canonical spellings and a caller may hold any of the others (FIB-300).
        self._manufacturer: str = normalize_manufacturer(microscope.manufacturer) or ""
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
        layout = FormGrid(self)
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

        align_form(layout)

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
        # An instrument the tags know about hides the other manufacturer's fields; one
        # they do not know about hides nothing, rather than everything.
        tagged_instrument = self._manufacturer in _TAGGED_MANUFACTURERS
        for row in self._rows:
            mfr_ok = (
                (row.mfr is None)
                or (not tagged_instrument)
                or (row.mfr == self._manufacturer)
            )
            adv_ok = (not row.advanced) or self._advanced_visible
            row.label.setVisible(mfr_ok and adv_ok)
            row.control.widget.setVisible(mfr_ok and adv_ok)

    def set_manufacturer(self, manufacturer: Optional[str]) -> None:
        self._manufacturer = normalize_manufacturer(manufacturer) or ""
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
