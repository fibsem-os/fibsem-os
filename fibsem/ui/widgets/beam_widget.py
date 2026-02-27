import logging

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
    QWidget,
)

from fibsem.microscope import FibsemMicroscope
from fibsem.structures import BeamSettings, BeamType, FibsemDetectorSettings
from fibsem.ui.widgets.beam_settings_widget import FibsemBeamSettingsWidget
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
        self.beam_group = QGroupBox("Beam")
        self.beam_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        beam_group_layout = QVBoxLayout()
        beam_group_layout.setContentsMargins(6, 6, 6, 6)
        self.beam_group.setLayout(beam_group_layout)

        self.beam_settings_widget = FibsemBeamSettingsWidget(
            microscope=self.microscope,
            beam_type=self.beam_type,
        )
        beam_group_layout.addWidget(self.beam_settings_widget)
        layout.addWidget(self.beam_group)

        # --- Detector Settings group ---
        self.detector_group = QGroupBox("Detector")
        self.detector_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        detector_group_layout = QVBoxLayout()
        detector_group_layout.setContentsMargins(6, 6, 6, 6)
        self.detector_group.setLayout(detector_group_layout)

        self.detector_settings_widget = FibsemDetectorSettingsWidget(
            microscope=self.microscope,
            beam_type=self.beam_type,
        )
        detector_group_layout.addWidget(self.detector_settings_widget)
        layout.addWidget(self.detector_group)

        layout.addStretch()

    # ------------------------------------------------------------------
    # Signal connections
    # ------------------------------------------------------------------

    def _connect_signals(self):
        self.beam_settings_widget.settings_changed.connect(self.beam_settings_changed)
        self.detector_settings_widget.settings_changed.connect(self.detector_settings_changed)

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


if __name__ == "__main__":
    import sys

    from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QPushButton

    from fibsem import utils

    app = QApplication(sys.argv)

    microscope, settings = utils.setup_session(manufacturer="Demo", ip_address="localhost")

    main_widget = QWidget()
    main_layout = QHBoxLayout()
    main_widget.setLayout(main_layout)

    for beam_type in [BeamType.ELECTRON, BeamType.ION]:
        col = QWidget()
        col_layout = QVBoxLayout()
        col.setLayout(col_layout)

        header = QLabel(f"{beam_type.name.title()} Beam")
        header.setStyleSheet("font-weight: bold; font-size: 16px;")
        col_layout.addWidget(header)

        widget = FibsemBeamWidget(microscope=microscope, beam_type=beam_type)
        widget.populate_combos()
        widget.sync_from_microscope()
        col_layout.addWidget(widget)

        widget.beam_settings_changed.connect(
            lambda s, bt=beam_type: print(f"[{bt.name}] beam_settings_changed: {s}")
        )
        widget.detector_settings_changed.connect(
            lambda s, bt=beam_type: print(f"[{bt.name}] detector_settings_changed: {s}")
        )

        def _print(checked=False, w=widget, bt=beam_type):
            from pprint import pprint
            print(f"--- {bt.name} beam ---")
            pprint(w.get_beam_settings().to_dict())
            pprint(w.get_detector_settings().to_dict())

        btn = QPushButton("Print Settings")
        btn.clicked.connect(_print)
        col_layout.addWidget(btn)

        main_layout.addWidget(col)

    main_widget.show()
    sys.exit(app.exec_())