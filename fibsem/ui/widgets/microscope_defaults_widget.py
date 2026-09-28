"""The beam defaults a session starts from, read from the instrument and edited here.

`defaults:` in the microscope configuration holds what the columns start at -- voltage,
current, field of view, resolution, dwell time, detector -- and what the acquire tab
opens with (`defaults.imaging`). Nobody wants to type those
into a file, and somebody who does will type what they believe the instrument is doing.
So the panel reads them off the instrument ("it is set up how I like it, remember
this"), lets them be adjusted, and writes them into the configuration file the session
was started from -- that section and nothing else.

Nothing here touches the column. Apply, beside this panel, is what pushes the
defaults to the instrument; this panel only decides what they are. It is the Defaults
tab of the Microscope Configuration window, whose Save writes it.
"""

import logging
import math
from typing import Callable, List, Optional, Tuple

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)

from fibsem import utils
from fibsem.constants import METRE_TO_MICRON, MICRON_TO_METRE
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import BeamSystemSettings, BeamType, ImageSettings
from fibsem.ui import notification_service
from fibsem.ui.icon import ICON_READ_FROM_ACQUIRE_TAB, ICON_READ_FROM_MICROSCOPE
from fibsem.ui.widgets.custom_widgets import (
    IconToolButton,
    TitledPanel,
    ValueComboBox,
    ValueSpinBox,
)

# Offered when the instrument cannot list its own; the current value is always
# added beside them so a file's setting is never silently snapped to a neighbour.
STANDARD_RESOLUTIONS: List[Tuple[int, int]] = [
    (768, 512),
    (1536, 1024),
    (3072, 2048),
    (6144, 4096),
]


def _format_voltage(v) -> str:
    return f"{float(v) / 1e3:.2f} kV"


def _format_current(a) -> str:
    a = float(a)
    if a >= 1e-9:
        return f"{a * 1e9:.2f} nA"
    return f"{a * 1e12:.1f} pA"


# Resolutions travel through the combo box as "WxH" strings: a tuple does not survive
# the QVariant round trip `findData` uses to select an item, so `set_value` fell
# back to its closest-numeric rule, which cannot compare tuples and picked index 0.
def _resolution_key(r) -> str:
    return f"{int(r[0])}x{int(r[1])}"


def _resolution_from_key(key: str) -> Tuple[int, int]:
    w, h = key.split("x")
    return int(w), int(h)


def _format_resolution(key: str) -> str:
    return key.replace("x", " x ")


class BeamDefaultsForm(QWidget):
    """One column's defaults: what `defaults.electron` / `defaults.ion` carries."""

    def __init__(self, beam_type: BeamType, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.beam_type = beam_type

        self.voltage = ValueComboBox(format_fn=_format_voltage)
        self.current = ValueComboBox(format_fn=_format_current)
        self.hfw = ValueSpinBox(
            suffix="µm", minimum=1.0, maximum=5000.0, decimals=1, step=10.0
        )
        self.resolution = ValueComboBox(format_fn=_format_resolution)
        self.dwell_time = ValueSpinBox(
            suffix="µs", minimum=0.01, maximum=1000.0, decimals=2, step=0.1
        )
        self.detector_type = ValueComboBox()
        self.detector_mode = ValueComboBox()
        # A standing choice some sites make -- a FIB run at 180 degrees -- and Apply
        # sets it like the rest, so it is a default rather than alignment state.
        self.scan_rotation = ValueSpinBox(
            suffix="°", minimum=0.0, maximum=360.0, decimals=1, step=90.0
        )

        form = QFormLayout()
        form.addRow("Voltage", self.voltage)
        form.addRow("Current", self.current)
        form.addRow("Field of view", self.hfw)
        form.addRow("Resolution", self.resolution)
        form.addRow("Dwell time", self.dwell_time)
        form.addRow("Detector", self.detector_type)
        form.addRow("Mode", self.detector_mode)
        form.addRow("Scan rotation", self.scan_rotation)
        form.setContentsMargins(6, 6, 6, 6)
        self.setLayout(form)

    def populate_choices(self, microscope: FibsemMicroscope) -> None:
        """The values this instrument can be set to, with the current one kept."""
        beam = self.beam_type

        def available(key: str) -> list:
            try:
                return list(microscope.get_available_values_cached(key, beam) or [])
            except Exception as e:  # a backend that cannot list this key
                logging.debug(f"No available values for {key} ({beam.name}): {e}")
                return []

        record = getattr(microscope.system, beam.name.lower())
        self._set_choices(self.voltage, available("voltage"), record.beam.voltage)
        self._set_choices(self.current, available("current"), record.beam.beam_current)
        self._set_choices(
            self.resolution,
            [_resolution_key(r) for r in STANDARD_RESOLUTIONS],
            _resolution_key(record.beam.resolution) if record.beam.resolution else None,
        )
        self._set_choices(
            self.detector_type, available("detector_type"), record.detector.type
        )
        self._set_choices(
            self.detector_mode, available("detector_mode"), record.detector.mode
        )

    @staticmethod
    def _set_choices(combo: ValueComboBox, choices: list, current) -> None:
        items = list(choices)
        if current is not None and current not in items:
            items.append(current)
        combo.set_values(items, value=current)

    def show_settings(self, record: BeamSystemSettings) -> None:
        beam, detector = record.beam, record.detector
        if beam.voltage is not None:
            self.voltage.set_value(beam.voltage)
        if beam.beam_current is not None:
            self.current.set_value(beam.beam_current)
        if beam.hfw is not None:
            self.hfw.setValue(beam.hfw * METRE_TO_MICRON)
        if beam.resolution is not None:
            self.resolution.set_value(_resolution_key(beam.resolution))
        if beam.dwell_time is not None:
            self.dwell_time.setValue(beam.dwell_time * 1e6)
        if detector.type is not None:
            self.detector_type.set_value(detector.type)
        if detector.mode is not None:
            self.detector_mode.set_value(detector.mode)
        if beam.scan_rotation is not None:
            self.scan_rotation.setValue(math.degrees(beam.scan_rotation) % 360)

    def defaults_to_dict(self) -> dict:
        """The form's eight values, in the file's spelling."""
        key = self.resolution.value()
        return {
            "voltage": self.voltage.value(),
            "current": self.current.value(),
            "hfw": self.hfw.value() * MICRON_TO_METRE,
            "resolution": list(_resolution_from_key(key)) if key else None,
            "dwell_time": self.dwell_time.value() * 1e-6,
            "detector_type": self.detector_type.value(),
            "detector_mode": self.detector_mode.value(),
            "scan_rotation": math.radians(self.scan_rotation.value()),
        }

    def write_into(self, record: BeamSystemSettings) -> None:
        """Copy the form into the record. Only the defaults; nothing about the column."""
        record.beam.voltage = self.voltage.value()
        record.beam.beam_current = self.current.value()
        record.beam.hfw = self.hfw.value() * MICRON_TO_METRE
        key = self.resolution.value()
        record.beam.resolution = _resolution_from_key(key) if key else None
        record.beam.dwell_time = self.dwell_time.value() * 1e-6
        record.detector.type = self.detector_type.value()
        record.detector.mode = self.detector_mode.value()
        record.beam.scan_rotation = math.radians(self.scan_rotation.value())


class ImagingDefaultsForm(QWidget):
    """What the acquire tab opens with: `defaults.imaging`."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.beam_type = ValueComboBox(format_fn=lambda name: str(name).title())
        self.beam_type.set_values([BeamType.ELECTRON.name, BeamType.ION.name])
        self.hfw = ValueSpinBox(
            suffix="µm", minimum=1.0, maximum=5000.0, decimals=1, step=10.0
        )
        self.resolution = ValueComboBox(format_fn=_format_resolution)
        self.resolution.set_values([_resolution_key(r) for r in STANDARD_RESOLUTIONS])
        self.dwell_time = ValueSpinBox(
            suffix="µs", minimum=0.01, maximum=1000.0, decimals=2, step=0.1
        )
        self.autocontrast = QCheckBox()

        form = QFormLayout()
        form.addRow("Beam", self.beam_type)
        form.addRow("Field of view", self.hfw)
        form.addRow("Resolution", self.resolution)
        form.addRow("Dwell time", self.dwell_time)
        form.addRow("Autocontrast", self.autocontrast)
        form.setContentsMargins(6, 6, 6, 6)
        self.setLayout(form)

    def show_settings(self, settings: ImageSettings) -> None:
        self.beam_type.set_value(settings.beam_type.name)
        if settings.hfw is not None:
            self.hfw.setValue(settings.hfw * METRE_TO_MICRON)
        if settings.resolution is not None:
            key = _resolution_key(settings.resolution)
            if self.resolution.findData(key) == -1:
                self.resolution.add_value(key)  # a file's value is never snapped
            self.resolution.set_value(key)
        if settings.dwell_time is not None:
            self.dwell_time.setValue(settings.dwell_time * 1e6)
        self.autocontrast.setChecked(bool(settings.autocontrast))

    def defaults_to_dict(self) -> dict:
        """The form's five values, in the file's spelling."""
        key = self.resolution.value()
        return {
            "beam_type": self.beam_type.value(),
            "hfw": self.hfw.value() * MICRON_TO_METRE,
            "resolution": list(_resolution_from_key(key)) if key else None,
            "dwell_time": self.dwell_time.value() * 1e-6,
            "autocontrast": self.autocontrast.isChecked(),
        }

    def write_into(self, settings: ImageSettings) -> None:
        """Copy the form into the record; the rest of it (save, path) is left."""
        settings.beam_type = BeamType[self.beam_type.value()]
        settings.hfw = self.hfw.value() * MICRON_TO_METRE
        key = self.resolution.value()
        if key:
            settings.resolution = _resolution_from_key(key)
        settings.dwell_time = self.dwell_time.value() * 1e-6
        settings.autocontrast = self.autocontrast.isChecked()


class MicroscopeDefaultsWidget(QWidget):
    """Read the defaults from the instrument, edit them, save them to the configuration."""

    # Any value in the form changed; `is_modified` says whether it now differs
    # from what was last loaded or saved.
    changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.microscope: Optional[FibsemMicroscope] = None
        # What the configuration was loaded with; the acquire tab starts from it.
        self.image_settings: Optional[ImageSettings] = None
        # The acquire tab's current settings, when the application has one to offer.
        self._current_imaging: Optional[Callable[[], ImageSettings]] = None

        self.electron = BeamDefaultsForm(BeamType.ELECTRON)
        self.ion = BeamDefaultsForm(BeamType.ION)
        self.imaging = ImagingDefaultsForm()

        # Each section reads its own values from where they live: the beams from
        # the instrument, the imaging from the acquire tab.
        self.button_read_electron = IconToolButton(
            icon=ICON_READ_FROM_MICROSCOPE,
            tooltip="Take what the electron beam is doing now as its defaults.",
        )
        self.button_read_ion = IconToolButton(
            icon=ICON_READ_FROM_MICROSCOPE,
            tooltip="Take what the ion beam is doing now as its defaults.",
        )
        self.button_read_imaging = IconToolButton(
            icon=ICON_READ_FROM_ACQUIRE_TAB,
            tooltip="Take what the acquire tab is set to now as the imaging defaults.",
        )
        self.button_read_imaging.setEnabled(False)

        self.electron_panel = TitledPanel(
            "Electron Beam", content=self.electron, collapsible=False
        )
        self.electron_panel.add_header_widget(self.button_read_electron)
        self.ion_panel = TitledPanel("Ion Beam", content=self.ion, collapsible=False)
        self.ion_panel.add_header_widget(self.button_read_ion)
        self.imaging_panel = TitledPanel(
            "Imaging", content=self.imaging, collapsible=False
        )
        self.imaging_panel.add_header_widget(self.button_read_imaging)

        # Stored in the file (`defaults.apply_on_connect`) but not acted on yet, so
        # it is shown and cannot be changed.
        self.apply_on_connect = QCheckBox("Apply these defaults when connecting")
        self.apply_on_connect.setEnabled(False)
        self.apply_on_connect.setToolTip(
            "Not available yet: connecting does not set the columns. Use Apply to "
            "Microscope."
        )

        sections = QHBoxLayout()
        for panel in (self.electron_panel, self.ion_panel, self.imaging_panel):
            sections.addWidget(panel, alignment=Qt.AlignTop)
        layout = QVBoxLayout()
        layout.addLayout(sections)
        layout.addWidget(self.apply_on_connect)
        layout.addStretch()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # What the form held when it was last loaded or saved.
        self._saved: Optional[dict] = None
        for field in self.findChildren(QComboBox):
            field.currentIndexChanged.connect(self.changed)
        for field in self.findChildren(QDoubleSpinBox):
            field.valueChanged.connect(self.changed)
        for field in self.imaging.findChildren(QCheckBox):
            field.toggled.connect(self.changed)

        self.button_read_electron.clicked.connect(
            lambda: self.read_from_microscope(BeamType.ELECTRON)
        )
        self.button_read_ion.clicked.connect(
            lambda: self.read_from_microscope(BeamType.ION)
        )
        self.button_read_imaging.clicked.connect(self.read_from_acquire_tab)
        self.setEnabled(False)

    def set_microscope(
        self,
        microscope: Optional[FibsemMicroscope],
        image_settings: Optional[ImageSettings] = None,
    ) -> None:
        self.microscope = microscope
        self.image_settings = image_settings
        self.setEnabled(microscope is not None)
        # Without the settings the configuration was loaded with there is nothing
        # to show, and saving the form's own starting values would overwrite the
        # file's `defaults.imaging` -- so the section sits out.
        self.imaging_panel.setEnabled(image_settings is not None)
        if microscope is None:
            self.set_current_imaging(None)
            self._saved = None
            return
        for form in (self.electron, self.ion):
            form.populate_choices(microscope)
        self.show_system()
        self.apply_on_connect.setChecked(
            bool(microscope.system.apply_defaults_on_connect)
        )
        self._saved = self._values()
        self.changed.emit()

    def set_current_imaging(
        self, provider: Optional[Callable[[], ImageSettings]]
    ) -> None:
        """Where "Read from Acquire Tab" reads from: the application's acquire tab."""
        self._current_imaging = provider
        self.button_read_imaging.setEnabled(
            provider is not None and self.image_settings is not None
        )

    def show_system(self) -> None:
        """The form shows what `system.electron` / `system.ion` currently hold."""
        if self.microscope is None:
            return
        self.electron.show_settings(self.microscope.system.electron)
        self.ion.show_settings(self.microscope.system.ion)
        if self.image_settings is not None:
            self.imaging.show_settings(self.image_settings)

    def _values(self) -> dict:
        values = {
            "electron": self.electron.defaults_to_dict(),
            "ion": self.ion.defaults_to_dict(),
        }
        if self.image_settings is not None:
            values["imaging"] = self.imaging.defaults_to_dict()
        return values

    def is_modified(self) -> bool:
        """Whether the form differs from what was last loaded or saved."""
        return self._saved is not None and self._values() != self._saved

    def read_from_microscope(self, beam_type: Optional[BeamType] = None) -> None:
        """Take the live values of one beam, or of both, into the form."""
        if self.microscope is None:
            return
        self.microscope.capture_defaults(beam_type)
        # Only the forms that were read: the others may hold unsaved edits.
        for form in (self.electron, self.ion):
            if beam_type is None or form.beam_type is beam_type:
                record = getattr(self.microscope.system, form.beam_type.name.lower())
                form.show_settings(record)
        which = "the microscope" if beam_type is None else beam_type.name.lower()
        notification_service.show_toast(
            f"Defaults read from {which}. Save to keep them.", "info"
        )

    def read_from_acquire_tab(self) -> None:
        if self._current_imaging is None:
            return
        try:
            current = self._current_imaging()
        except Exception as e:  # noqa: BLE001 - the tab may be mid-teardown
            logging.warning(f"Could not read the acquire tab's settings: {e}")
            return
        self.imaging.show_settings(current)
        notification_service.show_toast(
            "Imaging defaults read from the acquire tab. Save to keep them.", "info"
        )

    def write_form_into_system(self) -> None:
        if self.microscope is None:
            return
        self.electron.write_into(self.microscope.system.electron)
        self.ion.write_into(self.microscope.system.ion)
        if self.image_settings is not None:
            self.imaging.write_into(self.image_settings)

    def save_to_configuration(self) -> bool:
        """Write the form into the configuration file. Returns whether it was saved."""
        if self.microscope is None:
            return False
        path = getattr(self.microscope, "configuration_path", None)
        if not path:
            notification_service.show_toast(
                "This session was not started from a configuration file, so there is "
                "nowhere to save the defaults.",
                "warning",
            )
            return False
        self.write_form_into_system()
        # Exactly the keys the forms show. Not the whole beam record: that also
        # carries the beam shift, stigmation and working distance, which are
        # alignment state -- written here they would be pushed back by Apply. Not the
        # whole image record either: its save path is the session's.
        # `apply_on_connect` is not this panel's to change.
        updates = {"defaults": self._values()}
        try:
            utils.write_configuration(path, updates)
        except Exception as e:
            logging.error(f"Could not save the defaults: {e}")
            notification_service.show_toast(
                f"Could not save the defaults: {e}", "error"
            )
            return False
        logging.info(f"Beam defaults saved to {path}")
        notification_service.show_toast("Defaults saved to the configuration.", "info")
        self._saved = self._values()
        self.changed.emit()
        return True
