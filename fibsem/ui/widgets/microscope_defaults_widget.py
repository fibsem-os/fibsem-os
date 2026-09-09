"""The beam defaults a session starts from, read from the instrument and edited here.

`defaults:` in the microscope configuration holds what the columns start at -- voltage,
current, field of view, resolution, dwell time, detector. Nobody wants to type those
into a file, and somebody who does will type what they believe the instrument is doing.
So the panel reads them off the instrument ("it is set up how I like it, remember
this"), lets them be adjusted, and writes them into the configuration file the session
was started from -- that section and nothing else.

Nothing here touches the column. Apply, beside this panel, is what pushes the
defaults to the instrument; this panel only decides what they are.
"""

import logging
from typing import List, Optional, Tuple

from PyQt5.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem import utils
from fibsem.constants import METRE_TO_MICRON, MICRON_TO_METRE
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import BeamSystemSettings, BeamType
from fibsem.ui import notification_service, stylesheets
from fibsem.ui.widgets.custom_widgets import ValueComboBox, ValueSpinBox

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


def _format_resolution(r) -> str:
    return f"{int(r[0])} x {int(r[1])}"


class BeamDefaultsForm(QGroupBox):
    """One column's defaults: what `defaults.electron` / `defaults.ion` carries."""

    def __init__(self, beam_type: BeamType, parent: Optional[QWidget] = None):
        super().__init__(beam_type.name.title() + " Beam", parent)
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

        form = QFormLayout()
        form.addRow("Voltage", self.voltage)
        form.addRow("Current", self.current)
        form.addRow("Field of view", self.hfw)
        form.addRow("Resolution", self.resolution)
        form.addRow("Dwell time", self.dwell_time)
        form.addRow("Detector", self.detector_type)
        form.addRow("Mode", self.detector_mode)
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
        resolutions = [tuple(r) for r in STANDARD_RESOLUTIONS]
        self._set_choices(
            self.resolution,
            resolutions,
            tuple(record.beam.resolution) if record.beam.resolution else None,
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
            self.resolution.set_value(tuple(beam.resolution))
        if beam.dwell_time is not None:
            self.dwell_time.setValue(beam.dwell_time * 1e6)
        if detector.type is not None:
            self.detector_type.set_value(detector.type)
        if detector.mode is not None:
            self.detector_mode.set_value(detector.mode)

    def write_into(self, record: BeamSystemSettings) -> None:
        """Copy the form into the record. Only the defaults; nothing about the column."""
        record.beam.voltage = self.voltage.value()
        record.beam.beam_current = self.current.value()
        record.beam.hfw = self.hfw.value() * MICRON_TO_METRE
        resolution = self.resolution.value()
        record.beam.resolution = tuple(resolution) if resolution else None
        record.beam.dwell_time = self.dwell_time.value() * 1e-6
        record.detector.type = self.detector_type.value()
        record.detector.mode = self.detector_mode.value()


class MicroscopeDefaultsWidget(QWidget):
    """Read the defaults from the instrument, edit them, save them to the configuration."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.microscope: Optional[FibsemMicroscope] = None

        self.electron = BeamDefaultsForm(BeamType.ELECTRON)
        self.ion = BeamDefaultsForm(BeamType.ION)
        self.pushButton_read = QPushButton("Read from Microscope")
        self.pushButton_read.setToolTip(
            "Take what the instrument is doing now as the defaults."
        )
        self.pushButton_read.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.pushButton_save = QPushButton("Save to Configuration")
        self.pushButton_save.setToolTip(
            "Write these defaults into the configuration file this session was "
            "started from. Nothing else in the file is changed."
        )
        self.pushButton_save.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)

        forms = QHBoxLayout()
        forms.addWidget(self.electron)
        forms.addWidget(self.ion)
        buttons = QHBoxLayout()
        buttons.addWidget(self.pushButton_read)
        buttons.addWidget(self.pushButton_save)
        layout = QVBoxLayout()
        layout.addLayout(forms)
        layout.addLayout(buttons)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.pushButton_read.clicked.connect(self.read_from_microscope)
        self.pushButton_save.clicked.connect(self.save_to_configuration)
        self.setEnabled(False)

    def set_microscope(self, microscope: Optional[FibsemMicroscope]) -> None:
        self.microscope = microscope
        self.setEnabled(microscope is not None)
        if microscope is None:
            return
        for form in (self.electron, self.ion):
            form.populate_choices(microscope)
        self.show_system()

    def show_system(self) -> None:
        """The form shows what `system.electron` / `system.ion` currently hold."""
        if self.microscope is None:
            return
        self.electron.show_settings(self.microscope.system.electron)
        self.ion.show_settings(self.microscope.system.ion)

    def read_from_microscope(self) -> None:
        if self.microscope is None:
            return
        self.microscope.capture_defaults()
        self.show_system()
        notification_service.show_toast(
            "Defaults read from the microscope. Save to keep them.", "info"
        )

    def write_form_into_system(self) -> None:
        if self.microscope is None:
            return
        self.electron.write_into(self.microscope.system.electron)
        self.ion.write_into(self.microscope.system.ion)

    def save_to_configuration(self) -> None:
        if self.microscope is None:
            return
        path = getattr(self.microscope, "configuration_path", None)
        if not path:
            notification_service.show_toast(
                "This session was not started from a configuration file, so there is "
                "nowhere to save the defaults.",
                "warning",
            )
            return
        self.write_form_into_system()
        # The beam halves of what `SystemSettings.to_dict` writes under `defaults:`.
        # The imaging defaults are the acquire tab's, and `apply_on_connect` is not
        # this panel's to change, so neither is written from here.
        defaults = self.microscope.system.to_dict()["defaults"]
        updates = {
            "defaults": {"electron": defaults["electron"], "ion": defaults["ion"]}
        }
        try:
            utils.write_configuration(path, updates)
        except Exception as e:
            logging.error(f"Could not save the defaults: {e}")
            notification_service.show_toast(
                f"Could not save the defaults: {e}", "error"
            )
            return
        logging.info(f"Beam defaults saved to {path}")
        notification_service.show_toast("Defaults saved to the configuration.", "info")
