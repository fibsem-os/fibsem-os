import logging

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QWidget,
)

from fibsem.microscope import FibsemMicroscope
from fibsem.structures import BeamSettings, BeamType, FibsemDetectorSettings
from fibsem.ui import stylesheets
from fibsem.ui.widgets.beam_settings_widget import FibsemBeamSettingsWidget
from fibsem.ui.widgets.custom_widgets import IconToolButton, TitledPanel
from fibsem.ui.widgets.detector_settings_widget import FibsemDetectorSettingsWidget


class FibsemBeamWidget(QWidget):
    """Combined beam + detector settings widget for a single beam type.

    Composes FibsemBeamSettingsWidget and FibsemDetectorSettingsWidget inside
    labelled group boxes and exposes a unified get/set API.

    Signals:
        beam_settings_changed(BeamSettings): emitted whenever a beam setting changes.
        detector_settings_changed(FibsemDetectorSettings): emitted whenever a detector setting changes.
    """

    beam_settings_changed = pyqtSignal(BeamSettings)
    detector_settings_changed = pyqtSignal(FibsemDetectorSettings)

    def __init__(
        self,
        microscope: FibsemMicroscope,
        beam_type: BeamType,
        parent=None,
    ):
        super().__init__(parent)
        self.microscope = microscope
        self.beam_type = beam_type
        self._setup_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.setLayout(layout)

        # --- Beam Settings group ---
        self.beam_settings_widget = FibsemBeamSettingsWidget(
            microscope=self.microscope,
            beam_type=self.beam_type,
        )
        self._btn_advanced_beam = IconToolButton(
            icon="mdi:tune",
            checked_icon="mdi:tune-variant",
            checked_color=stylesheets.GRAY_WHITE_COLOR,
            tooltip="Show advanced settings",
            checked_tooltip="Hide advanced settings",
        )
        self.beam_group = TitledPanel(
            "Beam", content=self.beam_settings_widget, collapsible=False
        )
        self.beam_group.add_header_widget(self._btn_advanced_beam)
        layout.addWidget(self.beam_group)

        # --- Detector Settings group ---
        self.detector_settings_widget = FibsemDetectorSettingsWidget(
            microscope=self.microscope,
            beam_type=self.beam_type,
        )
        self._btn_advanced_detector = IconToolButton(
            icon="mdi:tune",
            checked_icon="mdi:tune-variant",
            checked_color=stylesheets.GRAY_WHITE_COLOR,
            tooltip="Show advanced settings",
            checked_tooltip="Hide advanced settings",
        )
        self.detector_group = TitledPanel(
            "Detector", content=self.detector_settings_widget, collapsible=False
        )
        self.detector_group.add_header_widget(self._btn_advanced_detector)
        layout.addWidget(self.detector_group)

        layout.addStretch()

    # ------------------------------------------------------------------
    # Signal connections
    # ------------------------------------------------------------------

    def _connect_signals(self):
        self.beam_settings_widget.settings_changed.connect(self.beam_settings_changed)
        self.detector_settings_widget.settings_changed.connect(
            self.detector_settings_changed
        )
        self._btn_advanced_beam.toggled.connect(
            self.beam_settings_widget.set_advanced_visible
        )
        self._btn_advanced_detector.toggled.connect(
            self.detector_settings_widget.set_advanced_visible
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def populate_combos(self):
        """Populate all comboboxes from the microscope.

        Call this once after construction so all comboboxes contain the correct
        available values for the current beam type.
        """
        self.beam_settings_widget.populate_beam_combos()
        self.detector_settings_widget.populate_detector_combos()

    def get_beam_settings(self) -> BeamSettings:
        """Return BeamSettings built from the current widget values."""
        return self.beam_settings_widget.get_settings()

    def get_detector_settings(self) -> FibsemDetectorSettings:
        """Return FibsemDetectorSettings built from the current widget values."""
        return self.detector_settings_widget.get_settings()

    def update_from_settings(
        self,
        beam_settings: BeamSettings,
        detector_settings: FibsemDetectorSettings,
    ):
        """Populate all controls without triggering live updates."""
        self.beam_settings_widget.update_from_settings(beam_settings)
        self.detector_settings_widget.update_from_settings(detector_settings)

    def sync_from_microscope(self):
        """Read current settings from the microscope and update the UI."""
        beam_settings = self.microscope.get_beam_settings(self.beam_type)
        detector_settings = self.microscope.get_detector_settings(self.beam_type)
        self.update_from_settings(beam_settings, detector_settings)
