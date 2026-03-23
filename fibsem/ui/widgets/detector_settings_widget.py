import logging

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QWidget,
)
from superqt import QDoubleSlider

from fibsem.microscope import FibsemMicroscope
from fibsem.structures import BeamType, FibsemDetectorSettings
from fibsem.ui.widgets.custom_widgets import WheelBlocker

WIDGET_CONFIG = {
    "brightness": {
        "label": "Brightness",
        "range": (0.0, 1.0),
        "decimals": 3,
        "step": 0.01,
        "suffix": None,
    },
    "contrast": {
        "label": "Contrast",
        "range": (0.0, 1.0),
        "decimals": 3,
        "step": 0.01,
        "suffix": None,
    },
    "type": {"label": "Detector Type"},
    "mode": {"label": "Detector Mode"},
}


class _LabeledSlider(QWidget):
    """A horizontal QDoubleSlider with a value label and a % toggle button."""

    valueChanged = pyqtSignal(float)

    def __init__(self, min_val: float = 0.0, max_val: float = 1.0, decimals: int = 3, parent=None):
        super().__init__(parent)
        self._decimals = decimals
        self._pct_mode = True

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._slider = QDoubleSlider(Qt.Orientation.Horizontal, parent=self)  # type: ignore[attr-defined]
        self._slider.setRange(min_val, max_val)
        self._slider.installEventFilter(WheelBlocker(parent=self._slider))

        self._value_label = QLabel(self._format(min_val), self)
        self._value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)  # type: ignore[attr-defined]
        self._value_label.setFixedWidth(52)

        layout.addWidget(self._slider)
        layout.addWidget(self._value_label)

        self._slider.valueChanged.connect(self._on_value_changed)

    def _format(self, value: float) -> str:
        if self._pct_mode:
            return f"{value * 100:.1f}%"
        return f"{value:.{self._decimals}f}"

    def set_pct_mode(self, enabled: bool):
        self._pct_mode = enabled
        self._value_label.setText(self._format(self._slider.value()))

    def _on_value_changed(self, value: float):
        self._value_label.setText(self._format(value))
        self.valueChanged.emit(value)

    def value(self) -> float:
        return self._slider.value()

    def setValue(self, value: float):
        self._slider.setValue(value)
        self._value_label.setText(self._format(value))

    def blockSignals(self, block: bool) -> bool:
        self._slider.blockSignals(block)
        return super().blockSignals(block)


class FibsemDetectorSettingsWidget(QWidget):
    settings_changed = pyqtSignal(FibsemDetectorSettings)

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

        # --- Detector Type ---
        self.type_combo = QComboBox()
        self.type_combo.installEventFilter(WheelBlocker(parent=self.type_combo))
        self.type_label = QLabel(WIDGET_CONFIG["type"]["label"])
        layout.addRow(self.type_label, self.type_combo)

        # --- Detector Mode ---
        self.mode_combo = QComboBox()
        self.mode_combo.installEventFilter(WheelBlocker(parent=self.mode_combo))
        self.mode_label = QLabel(WIDGET_CONFIG["mode"]["label"])
        layout.addRow(self.mode_label, self.mode_combo)

        # --- Brightness ---
        self.brightness_slider = _LabeledSlider(decimals=WIDGET_CONFIG["brightness"]["decimals"])
        self.brightness_label = QLabel(WIDGET_CONFIG["brightness"]["label"])
        layout.addRow(self.brightness_label, self.brightness_slider)

        # --- Contrast ---
        self.contrast_slider = _LabeledSlider(decimals=WIDGET_CONFIG["contrast"]["decimals"])
        self.contrast_label = QLabel(WIDGET_CONFIG["contrast"]["label"])
        layout.addRow(self.contrast_label, self.contrast_slider)

        # All widgets shown only when advanced mode is active
        self._adv_widgets = [
            self.type_label, self.type_combo,
            self.mode_label, self.mode_combo,
        ]

    # ------------------------------------------------------------------
    # Signal connections
    # ------------------------------------------------------------------

    def _connect_signals(self):
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self.brightness_slider.valueChanged.connect(self._on_brightness_changed)
        self.contrast_slider.valueChanged.connect(self._on_contrast_changed)

    # ------------------------------------------------------------------
    # Live-update handlers
    # ------------------------------------------------------------------

    def _on_type_changed(self, detector_type: str):
        if not detector_type:
            return
        self.microscope.set_detector_type(detector_type, self.beam_type)
        logging.info({"msg": "_on_type_changed", "beam_type": self.beam_type.name, "detector_type": detector_type})
        self.settings_changed.emit(self.get_settings())

    def _on_mode_changed(self, mode: str):
        if not mode:
            return
        self.microscope.set_detector_mode(mode, self.beam_type)
        logging.info({"msg": "_on_mode_changed", "beam_type": self.beam_type.name, "mode": mode})
        self.settings_changed.emit(self.get_settings())

    def _on_brightness_changed(self, value: float):
        self.microscope.set_detector_brightness(value, self.beam_type)
        logging.info({"msg": "_on_brightness_changed", "beam_type": self.beam_type.name, "brightness": value})
        self.settings_changed.emit(self.get_settings())

    def _on_contrast_changed(self, value: float):
        self.microscope.set_detector_contrast(value, self.beam_type)
        logging.info({"msg": "_on_contrast_changed", "beam_type": self.beam_type.name, "contrast": value})
        self.settings_changed.emit(self.get_settings())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_advanced_visible(self, show: bool):
        """Show or hide advanced controls (type, mode)."""
        self._advanced_visible = show
        self._update_visibility()

    def _update_visibility(self):
        adv = self._advanced_visible
        for w in self._adv_widgets:
            w.setVisible(adv)

    def populate_detector_combos(self):
        """Populate type and mode comboboxes from the microscope.

        Call this once after construction (or whenever the beam type changes)
        so the comboboxes contain the correct available values.
        """
        self.type_combo.blockSignals(True)
        self.type_combo.clear()
        available_types = self.microscope.get_available_values("detector_type", beam_type=self.beam_type)
        if available_types:
            self.type_combo.addItems(available_types)
            current_type = self.microscope.get_detector_type(self.beam_type)
            if current_type is not None:
                self.type_combo.setCurrentText(current_type)
        self.type_combo.blockSignals(False)

        self.mode_combo.blockSignals(True)
        self.mode_combo.clear()
        available_modes = self.microscope.get_available_values("detector_mode", beam_type=self.beam_type)
        if available_modes is None:
            available_modes = []
        self.mode_combo.addItems(available_modes)
        current_mode = self.microscope.get_detector_mode(self.beam_type)
        if current_mode is not None:
            self.mode_combo.setCurrentText(current_mode)
        self.mode_combo.blockSignals(False)

    def get_settings(self) -> FibsemDetectorSettings:
        """Return a FibsemDetectorSettings built from the current widget values."""
        return FibsemDetectorSettings(
            type=self.type_combo.currentText(),
            mode=self.mode_combo.currentText(),
            brightness=self.brightness_slider.value(),
            contrast=self.contrast_slider.value(),
        )

    def update_from_settings(self, settings: FibsemDetectorSettings):
        """Populate all controls from a FibsemDetectorSettings object without triggering live updates."""
        widgets = [
            self.type_combo,
            self.mode_combo,
            self.brightness_slider,
            self.contrast_slider,
        ]
        for w in widgets:
            w.blockSignals(True)

        if settings.type is not None:
            self.type_combo.setCurrentText(settings.type)
        if settings.mode is not None:
            self.mode_combo.setCurrentText(settings.mode)
        if settings.brightness is not None:
            self.brightness_slider.setValue(settings.brightness)
        if settings.contrast is not None:
            self.contrast_slider.setValue(settings.contrast)

        for w in widgets:
            w.blockSignals(False)


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

    header1 = QLabel("Electron Detector Settings")
    header1.setStyleSheet("font-weight: bold; font-size: 16px;")
    layout.addWidget(header1)
    widget = FibsemDetectorSettingsWidget(microscope=microscope, beam_type=BeamType.ELECTRON)
    widget.populate_detector_combos()
    widget.update_from_settings(microscope.get_detector_settings(BeamType.ELECTRON))
    layout.addWidget(widget)

    header2 = QLabel("Ion Detector Settings")
    header2.setStyleSheet("font-weight: bold; font-size: 16px;")
    layout.addWidget(header2)
    widget2 = FibsemDetectorSettingsWidget(microscope=microscope, beam_type=BeamType.ION)
    widget2.populate_detector_combos()
    widget2.update_from_settings(microscope.get_detector_settings(BeamType.ION))
    layout.addWidget(widget2)

    def print_settings():
        from pprint import pprint
        pprint(widget.get_settings().to_dict())

    btn = QPushButton("Print Settings")
    btn.clicked.connect(print_settings)
    layout.addWidget(btn)

    widget.settings_changed.connect(lambda s: print(f"settings_changed: {s}"))

    viewer = napari.Viewer()
    viewer.window.add_dock_widget(main_widget, area="right")
    napari.run()
