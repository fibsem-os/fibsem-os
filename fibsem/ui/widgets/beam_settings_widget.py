import logging

import numpy as np
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QWidget,
)

from fibsem import constants, utils
from fibsem import config as cfg
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import BeamSettings, BeamType, Point
from fibsem.ui.widgets.custom_widgets import WheelBlocker, _create_combobox_control

# GUI Configuration Constants
WIDGET_CONFIG = {
    "hfw": {
        "label": "Field of View",
        "range": (1, 3000),
        "decimals": 0,
        "step": 50.0,
        "suffix": f" {constants.MICRON_SYMBOL}",
    },
    "dwell_time": {
        "label": "Dwell Time",
        "range": (0.001, 1000.0),
        "decimals": 3,
        "step": 0.01,
        "suffix": f" {constants.MICROSECOND_SYMBOL}",
    },
    "scan_rotation": {
        "label": "Scan Rotation",
        "range": (0, 180),
        "decimals": 0,
        "step": 180,
        "suffix": f" {constants.DEGREE_SYMBOL}",
    },
    "shift": {
        "label": "Shift X / Y",
        "range": (-50.0, 50.0),
        "decimals": 3,
        "step": 0.01,
        "suffix": f" {constants.MICRON_SYMBOL}",
    },
    "stigmation": {
        "label": "Stigmation X / Y",
        "range": (-1.0, 1.0),
        "decimals": 4,
        "step": 0.001,
        "suffix": None,
    },
    "working_distance": {
        "label": "Working Distance",
        "range": (1.0, 30.0),
        "decimals": 2,
        "step": 0.01,
        "suffix": f" {constants.MILLIMETRE_SYMBOL}",
    },
    "resolution": {"label": "Resolution"},
    "beam_current": {"label": "Beam Current"},
    "beam_voltage": {"label": "Beam Voltage"},
    "preset": {"label": "Preset"},
}


class FibsemBeamSettingsWidget(QWidget):
    settings_changed = pyqtSignal(BeamSettings)

    def __init__(
        self,
        microscope: FibsemMicroscope,
        beam_type: BeamType,
        parent=None,
    ):
        super().__init__(parent)
        self.microscope = microscope
        self.beam_type = beam_type
        self._advanced_visible = False
        self._setup_ui()
        self._connect_signals()
        self._update_visibility()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QFormLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        def _make_spinbox(key: str) -> QDoubleSpinBox:
            c = WIDGET_CONFIG[key]
            sb = QDoubleSpinBox()
            sb.setRange(*c["range"])
            sb.setDecimals(c["decimals"])
            sb.setSingleStep(c["step"])
            if c["suffix"] is not None:
                sb.setSuffix(c["suffix"])
            sb.setKeyboardTracking(False)
            sb.installEventFilter(WheelBlocker(parent=sb))
            return sb

        # --- Field of View ---
        self.hfw_spinbox = _make_spinbox("hfw")
        self.hfw_label = QLabel(WIDGET_CONFIG["hfw"]["label"])
        layout.addRow(self.hfw_label, self.hfw_spinbox)

        # --- Dwell Time ---
        self.dwell_time_spinbox = _make_spinbox("dwell_time")
        self.dwell_time_label = QLabel(WIDGET_CONFIG["dwell_time"]["label"])
        layout.addRow(self.dwell_time_label, self.dwell_time_spinbox)

        # --- Resolution ---
        self.resolution_combo = QComboBox()
        for res_str, res_data in cfg.STANDARD_RESOLUTIONS_ZIP:
            self.resolution_combo.addItem(res_str, tuple(res_data))
        self.resolution_combo.setCurrentText(cfg.DEFAULT_STANDARD_RESOLUTION)
        self.resolution_combo.installEventFilter(WheelBlocker(parent=self.resolution_combo))
        self.resolution_label = QLabel(WIDGET_CONFIG["resolution"]["label"])
        layout.addRow(self.resolution_label, self.resolution_combo)

        # --- Beam Current ---
        self.beam_current_combo = QComboBox()
        self.beam_current_combo.installEventFilter(WheelBlocker(parent=self.beam_current_combo))
        self.beam_current_label = QLabel(WIDGET_CONFIG["beam_current"]["label"])
        layout.addRow(self.beam_current_label, self.beam_current_combo)

        # --- Beam Voltage ---
        self.beam_voltage_combo = QComboBox()
        self.beam_voltage_combo.installEventFilter(WheelBlocker(parent=self.beam_voltage_combo))
        self.beam_voltage_label = QLabel(WIDGET_CONFIG["beam_voltage"]["label"])
        layout.addRow(self.beam_voltage_label, self.beam_voltage_combo)

        # --- Preset ---
        self.preset_combo = QComboBox()
        self.preset_combo.installEventFilter(WheelBlocker(parent=self.preset_combo))
        self.preset_label = QLabel(WIDGET_CONFIG["preset"]["label"])
        layout.addRow(self.preset_label, self.preset_combo)

        # --- Working Distance ---
        self.working_distance_spinbox = _make_spinbox("working_distance")
        self.working_distance_label = QLabel(WIDGET_CONFIG["working_distance"]["label"])
        layout.addRow(self.working_distance_label, self.working_distance_spinbox)

        # --- Scan Rotation ---
        self.scan_rotation_spinbox = _make_spinbox("scan_rotation")
        self.scan_rotation_label = QLabel(WIDGET_CONFIG["scan_rotation"]["label"])
        layout.addRow(self.scan_rotation_label, self.scan_rotation_spinbox)

        # --- Shift X / Y ---
        self.shift_x_spinbox = _make_spinbox("shift")
        self.shift_y_spinbox = _make_spinbox("shift")
        self.shift_row = QWidget()
        shift_row_layout = QHBoxLayout(self.shift_row)
        shift_row_layout.setContentsMargins(0, 0, 0, 0)
        shift_row_layout.addWidget(self.shift_x_spinbox)
        shift_row_layout.addWidget(self.shift_y_spinbox)
        self.shift_label = QLabel(WIDGET_CONFIG["shift"]["label"])
        layout.addRow(self.shift_label, self.shift_row)

        # --- Stigmation X / Y ---
        self.stigmation_x_spinbox = _make_spinbox("stigmation")
        self.stigmation_y_spinbox = _make_spinbox("stigmation")
        self.stigmation_row = QWidget()
        stigmation_row_layout = QHBoxLayout(self.stigmation_row)
        stigmation_row_layout.setContentsMargins(0, 0, 0, 0)
        stigmation_row_layout.addWidget(self.stigmation_x_spinbox)
        stigmation_row_layout.addWidget(self.stigmation_y_spinbox)
        self.stigmation_label = QLabel(WIDGET_CONFIG["stigmation"]["label"])
        layout.addRow(self.stigmation_label, self.stigmation_row)

        # All widgets that are shown only when advanced mode is active
        self._adv_widgets = [
            self.scan_rotation_label, self.scan_rotation_spinbox,
            self.shift_label, self.shift_row,
            self.stigmation_label, self.stigmation_row,
            self.beam_voltage_label, self.beam_voltage_combo,
        ]

    # ------------------------------------------------------------------
    # Signal connections
    # ------------------------------------------------------------------

    def _connect_signals(self):
        self.hfw_spinbox.valueChanged.connect(self._on_hfw_changed)
        self.dwell_time_spinbox.valueChanged.connect(self._on_dwell_time_changed)
        self.resolution_combo.currentIndexChanged.connect(self._on_resolution_changed)
        self.beam_current_combo.currentIndexChanged.connect(self._on_beam_current_changed)
        self.beam_voltage_combo.currentIndexChanged.connect(self._on_beam_voltage_changed)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        self.working_distance_spinbox.valueChanged.connect(self._on_working_distance_changed)
        self.scan_rotation_spinbox.valueChanged.connect(self._on_scan_rotation_changed)
        self.shift_x_spinbox.valueChanged.connect(self._on_shift_changed)
        self.shift_y_spinbox.valueChanged.connect(self._on_shift_changed)
        self.stigmation_x_spinbox.valueChanged.connect(self._on_stigmation_changed)
        self.stigmation_y_spinbox.valueChanged.connect(self._on_stigmation_changed)

    # ------------------------------------------------------------------
    # Live-update handlers
    # ------------------------------------------------------------------

    def _on_hfw_changed(self, value: float):
        self.microscope.set_field_of_view(value * constants.MICRO_TO_SI, self.beam_type)
        logging.info({"msg": "_on_hfw_changed", "beam_type": self.beam_type.name, "hfw": value})
        self.settings_changed.emit(self.get_settings())

    def _on_dwell_time_changed(self, value: float):
        self.microscope.set_dwell_time(value * constants.MICRO_TO_SI, self.beam_type)
        logging.info({"msg": "_on_dwell_time_changed", "beam_type": self.beam_type.name, "dwell_time": value})
        self.settings_changed.emit(self.get_settings())

    def _on_resolution_changed(self, index: int):
        resolution = self.resolution_combo.itemData(index)
        if resolution is not None:
            self.microscope.set_resolution(resolution, self.beam_type)
            logging.info({"msg": "_on_resolution_changed", "beam_type": self.beam_type.name, "resolution": resolution})
            self.settings_changed.emit(self.get_settings())

    def _on_beam_current_changed(self, index: int):
        current = self.beam_current_combo.itemData(index)
        if current is not None:
            self.microscope.set_beam_current(current, self.beam_type)
            logging.info({"msg": "_on_beam_current_changed", "beam_type": self.beam_type.name, "current": current})
            self.settings_changed.emit(self.get_settings())

    def _on_beam_voltage_changed(self, index: int):
        voltage = self.beam_voltage_combo.itemData(index)
        if voltage is not None:
            self.microscope.set_beam_voltage(voltage, self.beam_type)
            logging.info({"msg": "_on_beam_voltage_changed", "beam_type": self.beam_type.name, "voltage": voltage})
            self.settings_changed.emit(self.get_settings())

    def _on_preset_changed(self, index: int):
        preset = self.preset_combo.itemData(index)
        if preset is not None:
            self.microscope.set_preset(preset, self.beam_type)
            logging.info({"msg": "_on_preset_changed", "beam_type": self.beam_type.name, "preset": preset})
            self.settings_changed.emit(self.get_settings())

    def _on_working_distance_changed(self, value: float):
        wd = value * constants.MILLI_TO_SI
        self.microscope.set_working_distance(wd, self.beam_type)
        logging.info({"msg": "_on_working_distance_changed", "beam_type": self.beam_type.name, "working_distance": wd})
        self.settings_changed.emit(self.get_settings())

    def _on_scan_rotation_changed(self, value: float):
        self.microscope.set_scan_rotation(np.deg2rad(value), self.beam_type)
        logging.info({"msg": "_on_scan_rotation_changed", "beam_type": self.beam_type.name, "rotation": value})
        self.settings_changed.emit(self.get_settings())

    def _on_shift_changed(self):
        shift = Point(
            x=self.shift_x_spinbox.value() * constants.MICRO_TO_SI,
            y=self.shift_y_spinbox.value() * constants.MICRO_TO_SI,
        )
        self.microscope.set_beam_shift(shift, self.beam_type)
        logging.info({"msg": "_on_shift_changed", "beam_type": self.beam_type.name, "shift": shift})
        self.settings_changed.emit(self.get_settings())

    def _on_stigmation_changed(self):
        stigmation = Point(
            x=self.stigmation_x_spinbox.value(),
            y=self.stigmation_y_spinbox.value(),
        )
        self.microscope.set_stigmation(stigmation, self.beam_type)
        logging.info({"msg": "_on_stigmation_changed", "beam_type": self.beam_type.name, "stigmation": stigmation})
        self.settings_changed.emit(self.get_settings())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def populate_beam_combos(self):
        """Populate beam current and voltage comboboxes from the microscope.

        Call this once after construction (or whenever the beam type changes)
        so the comboboxes contain the correct available values.
        """
        self.beam_current_combo.blockSignals(True)
        self.beam_current_combo.clear()
        _create_combobox_control(
            value=self.microscope.get_beam_current(self.beam_type),
            items=self.microscope.get_available_values_cached("current", self.beam_type),
            units="A",
            format_fn=utils.format_value,
            control=self.beam_current_combo,
        )
        self.beam_current_combo.blockSignals(False)

        self.beam_voltage_combo.blockSignals(True)
        self.beam_voltage_combo.clear()
        _create_combobox_control(
            value=self.microscope.get_beam_voltage(self.beam_type),
            items=self.microscope.get_available_values_cached("voltage", self.beam_type),
            units="V",
            format_fn=utils.format_value,
            control=self.beam_voltage_combo,
        )
        self.beam_voltage_combo.blockSignals(False)

        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        presets = self.microscope.get_available_values_cached("preset", self.beam_type)
        if presets:
            for preset in presets:
                self.preset_combo.addItem(str(preset), str(preset))
            current = self.microscope.get("preset", self.beam_type)
            if current is not None:
                idx = self.preset_combo.findData(current)
                if idx != -1:
                    self.preset_combo.setCurrentIndex(idx)
        self.preset_combo.blockSignals(False)

    def get_settings(self) -> BeamSettings:
        """Return a BeamSettings built from the current widget values."""
        return BeamSettings(
            beam_type=self.beam_type,
            working_distance=self.working_distance_spinbox.value() * constants.MILLI_TO_SI,
            hfw=self.hfw_spinbox.value() * constants.MICRO_TO_SI,
            dwell_time=self.dwell_time_spinbox.value() * constants.MICRO_TO_SI,
            resolution=self.resolution_combo.currentData(),
            beam_current=self.beam_current_combo.currentData(),
            voltage=self.beam_voltage_combo.currentData(),
            preset=self.preset_combo.currentData(),
            scan_rotation=np.deg2rad(self.scan_rotation_spinbox.value()),
            shift=Point(
                x=self.shift_x_spinbox.value() * constants.MICRO_TO_SI,
                y=self.shift_y_spinbox.value() * constants.MICRO_TO_SI,
            ),
            stigmation=Point(
                x=self.stigmation_x_spinbox.value(),
                y=self.stigmation_y_spinbox.value(),
            ),
        )

    def update_from_settings(self, settings: BeamSettings):
        """Populate all controls from a BeamSettings object without triggering live updates."""
        widgets = [
            self.working_distance_spinbox,
            self.hfw_spinbox,
            self.dwell_time_spinbox,
            self.resolution_combo,
            self.beam_current_combo,
            self.beam_voltage_combo,
            self.preset_combo,
            self.scan_rotation_spinbox,
            self.shift_x_spinbox,
            self.shift_y_spinbox,
            self.stigmation_x_spinbox,
            self.stigmation_y_spinbox,
        ]
        for w in widgets:
            w.blockSignals(True)

        if settings.working_distance is not None:
            self.working_distance_spinbox.setValue(settings.working_distance * constants.METRE_TO_MILLIMETRE)
        if settings.hfw is not None:
            self.hfw_spinbox.setValue(settings.hfw * constants.SI_TO_MICRO)
        if settings.dwell_time is not None:
            self.dwell_time_spinbox.setValue(settings.dwell_time * constants.SI_TO_MICRO)
        if settings.resolution is not None:
            idx = self.resolution_combo.findData(tuple(settings.resolution))
            if idx != -1:
                self.resolution_combo.setCurrentIndex(idx)
        if settings.beam_current is not None:
            self._set_combo_closest(self.beam_current_combo, settings.beam_current)
        if settings.voltage is not None:
            self._set_combo_closest(self.beam_voltage_combo, settings.voltage)
        if settings.preset is not None:
            idx = self.preset_combo.findData(settings.preset)
            if idx != -1:
                self.preset_combo.setCurrentIndex(idx)
        if settings.scan_rotation is not None:
            self.scan_rotation_spinbox.setValue(np.degrees(settings.scan_rotation))
        if settings.shift is not None:
            self.shift_x_spinbox.setValue(settings.shift.x * constants.SI_TO_MICRO)
            self.shift_y_spinbox.setValue(settings.shift.y * constants.SI_TO_MICRO)
        if settings.stigmation is not None:
            self.stigmation_x_spinbox.setValue(settings.stigmation.x)
            self.stigmation_y_spinbox.setValue(settings.stigmation.y)

        for w in widgets:
            w.blockSignals(False)

    def set_advanced_visible(self, show: bool):
        """Show or hide advanced controls."""
        self._advanced_visible = show
        self._update_visibility()

    def _update_visibility(self):
        """Apply visibility based on manufacturer and advanced-mode state."""
        is_tescan = self.microscope.manufacturer == "TESCAN"
        adv = self._advanced_visible

        for w in self._adv_widgets:
            w.setVisible(adv)

        # Stigmation: also hidden for TESCAN
        for w in [self.stigmation_label, self.stigmation_row]:
            w.setVisible(adv and not is_tescan)
            w.setEnabled(not is_tescan)

        # Beam voltage: also hidden for TESCAN
        for w in [self.beam_voltage_label, self.beam_voltage_combo]:
            w.setVisible(adv and not is_tescan)

        # Beam current: always visible, hidden for TESCAN
        for w in [self.beam_current_label, self.beam_current_combo]:
            w.setVisible(not is_tescan)

        # Preset: TESCAN only
        for w in [self.preset_label, self.preset_combo]:
            w.setVisible(is_tescan)

    @staticmethod
    def _set_combo_closest(combo, value: float):
        """Set the combobox to the item whose stored data is closest to value."""
        idx = combo.findData(value)
        if idx == -1:
            items = [combo.itemData(i) for i in range(combo.count())]
            if items:
                idx = combo.findData(min(items, key=lambda x: abs(x - value)))
        if idx != -1:
            combo.setCurrentIndex(idx)


if __name__ == "__main__":
    import sys

    import napari
    from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout

    from fibsem import utils

    app = QApplication(sys.argv)

    microscope, settings = utils.setup_session(manufacturer="Demo", ip_address="localhost")

    main_widget = QWidget()
    layout = QVBoxLayout()
    main_widget.setLayout(layout)

    header1 = QLabel("Electron Beam Settings")
    header1.setStyleSheet("font-weight: bold; font-size: 16px;")
    layout.addWidget(header1)
    widget = FibsemBeamSettingsWidget(microscope=microscope, beam_type=BeamType.ELECTRON)
    widget.populate_beam_combos()
    widget.update_from_settings(microscope.get_beam_settings(BeamType.ELECTRON))
    layout.addWidget(widget)

    header = QLabel("Ion Beam Settings")
    header.setStyleSheet("font-weight: bold; font-size: 16px;")
    layout.addWidget(header)
    widget2 = FibsemBeamSettingsWidget(microscope=microscope, beam_type=BeamType.ION)
    widget2.populate_beam_combos()
    widget2.update_from_settings(microscope.get_beam_settings(BeamType.ION))
    layout.addWidget(widget2)

    from pprint import pprint
    def print_settings():
        s = widget.get_settings()
        pprint(s.to_dict())

    btn = QPushButton("Print Settings")
    btn.clicked.connect(print_settings)
    layout.addWidget(btn)

    widget.settings_changed.connect(lambda s: print(f"settings_changed: {s}"))

    viewer = napari.Viewer()
    viewer.window.add_dock_widget(main_widget, area="right")
    napari.run()
