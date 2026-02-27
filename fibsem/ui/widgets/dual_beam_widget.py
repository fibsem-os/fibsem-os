import logging

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QRadioButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from superqt import QIconifyIcon

from fibsem.microscope import FibsemMicroscope
from fibsem.structures import BeamSettings, BeamType, FibsemDetectorSettings
from fibsem.ui.widgets.beam_widget import FibsemBeamWidget


class FibsemDualBeamWidget(QWidget):
    """Dual-beam widget with SEM / FIB radio buttons to switch between beam views.

    Stacks two FibsemBeamWidget instances vertically (one per beam type) and
    shows only the active one. Switching the radio button hides one and reveals
    the other.

    Sub-widgets are accessible directly as:
        .sem_widget  (FibsemBeamWidget for BeamType.ELECTRON)
        .fib_widget  (FibsemBeamWidget for BeamType.ION)

    Signals:
        beam_settings_changed(BeamSettings)
        detector_settings_changed(FibsemDetectorSettings)
    """

    beam_settings_changed = pyqtSignal(BeamSettings)
    detector_settings_changed = pyqtSignal(FibsemDetectorSettings)

    def __init__(
        self,
        microscope: FibsemMicroscope,
        initial_beam_type: BeamType = BeamType.ELECTRON,
        parent=None,
    ):
        super().__init__(parent)
        self.microscope = microscope
        self._initial_beam_type = initial_beam_type
        self._advanced_visible = False
        self._setup_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def beam_type(self) -> BeamType:
        """Currently selected beam type."""
        return BeamType.ELECTRON if self.sem_radio.isChecked() else BeamType.ION

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.setLayout(layout)

        # --- Radio buttons ---
        self.sem_radio = QRadioButton("SEM")
        self.fib_radio = QRadioButton("FIB")
        self._button_group = QButtonGroup(self)
        self._button_group.addButton(self.sem_radio)
        self._button_group.addButton(self.fib_radio)

        # --- Beam on / blanked buttons ---
        self.btn_beam_on = QToolButton()
        self.btn_beam_on.setIcon(QIconifyIcon("mdi:power", color="#c0c0c0"))
        self.btn_beam_on.setToolTip("Toggle beam on/off")

        self.btn_beam_blanked = QToolButton()
        self.btn_beam_blanked.setIcon(QIconifyIcon("mdi:eye", color="#c0c0c0"))
        self.btn_beam_blanked.setToolTip("Toggle beam blank/unblank")

        # --- Refresh button ---
        self.btn_refresh = QToolButton()
        self.btn_refresh.setIcon(QIconifyIcon("mdi:refresh", color="#c0c0c0"))
        self.btn_refresh.setToolTip("Refresh from microscope")

        # --- Advanced toggle button ---
        self.btn_advanced = QToolButton()
        self.btn_advanced.setIcon(QIconifyIcon("mdi:tune", color="#c0c0c0"))
        self.btn_advanced.setToolTip("Show advanced settings")
        self.btn_advanced.setCheckable(True)

        radio_row = QWidget()
        radio_layout = QHBoxLayout(radio_row)
        radio_layout.setContentsMargins(0, 0, 0, 0)
        radio_layout.addWidget(self.sem_radio)
        radio_layout.addWidget(self.fib_radio)
        radio_layout.addStretch()
        radio_layout.addWidget(self.btn_beam_on)
        radio_layout.addWidget(self.btn_beam_blanked)
        radio_layout.addWidget(self.btn_refresh)
        radio_layout.addWidget(self.btn_advanced)
        layout.addWidget(radio_row)

        # --- SEM widget ---
        self.sem_widget = FibsemBeamWidget(
            microscope=self.microscope,
            beam_type=BeamType.ELECTRON,
        )
        layout.addWidget(self.sem_widget)

        # --- FIB widget ---
        self.fib_widget = FibsemBeamWidget(
            microscope=self.microscope,
            beam_type=BeamType.ION,
        )
        layout.addWidget(self.fib_widget)

        layout.addStretch()

        # Set initial selection
        if self._initial_beam_type is BeamType.ELECTRON:
            self.sem_radio.setChecked(True)
            self.sem_widget.setVisible(True)
            self.fib_widget.setVisible(False)
        else:
            self.fib_radio.setChecked(True)
            self.sem_widget.setVisible(False)
            self.fib_widget.setVisible(True)

    # ------------------------------------------------------------------
    # Signal connections
    # ------------------------------------------------------------------

    def _connect_signals(self):
        self.sem_radio.toggled.connect(self._on_beam_selected)
        self.btn_beam_on.clicked.connect(self._on_beam_on_clicked)
        self.btn_beam_blanked.clicked.connect(self._on_beam_blanked_clicked)
        self.btn_refresh.clicked.connect(self._on_refresh_clicked)
        self.btn_advanced.toggled.connect(self._on_advanced_toggled)
        self.sem_widget.beam_settings_changed.connect(self.beam_settings_changed)
        self.sem_widget.detector_settings_changed.connect(self.detector_settings_changed)
        self.fib_widget.beam_settings_changed.connect(self.beam_settings_changed)
        self.fib_widget.detector_settings_changed.connect(self.detector_settings_changed)

    def _on_beam_selected(self, sem_checked: bool):
        self.sem_widget.setVisible(sem_checked)
        self.fib_widget.setVisible(not sem_checked)
        self.sync_beam_status()
        logging.info({"msg": "_on_beam_selected", "beam_type": self.beam_type.name})

    def _on_beam_on_clicked(self):
        bt = self.beam_type
        if self.microscope.is_on(bt):
            self.microscope.turn_off(bt)
        else:
            self.microscope.turn_on(bt)
        self.sync_beam_status()
        logging.info({"msg": "_on_beam_on_clicked", "beam_type": bt.name})

    def _on_beam_blanked_clicked(self):
        bt = self.beam_type
        if self.microscope.is_blanked(bt):
            self.microscope.unblank(bt)
        else:
            self.microscope.blank(bt)
        self.sync_beam_status()
        logging.info({"msg": "_on_beam_blanked_clicked", "beam_type": bt.name})

    def _on_refresh_clicked(self):
        self.sync_from_microscope()
        logging.info({"msg": "_on_refresh_clicked", "beam_type": self.beam_type.name})

    def _on_advanced_toggled(self, checked: bool):
        self._advanced_visible = checked
        tooltip = "Hide advanced settings" if checked else "Show advanced settings"
        self.btn_advanced.setToolTip(tooltip)
        for beam_widget in [self.sem_widget, self.fib_widget]:
            beam_widget.beam_settings_widget.set_advanced_visible(checked)
            beam_widget.detector_settings_widget.set_advanced_visible(checked)
        logging.info({"msg": "_on_advanced_toggled", "advanced": checked})

    # ------------------------------------------------------------------
    # Public API — delegates to the active sub-widget
    # ------------------------------------------------------------------

    @property
    def active_widget(self) -> FibsemBeamWidget:
        return self.sem_widget if self.beam_type is BeamType.ELECTRON else self.fib_widget

    def populate_combos(self):
        """Populate all comboboxes from the microscope for both beam types."""
        self.sem_widget.populate_combos()
        self.fib_widget.populate_combos()

    def get_beam_settings(self, beam_type: BeamType = None) -> BeamSettings:
        """Return BeamSettings for the given beam type (defaults to active beam)."""
        bt = beam_type if beam_type is not None else self.beam_type
        widget = self.sem_widget if bt is BeamType.ELECTRON else self.fib_widget
        return widget.get_beam_settings()

    def get_detector_settings(self, beam_type: BeamType = None) -> FibsemDetectorSettings:
        """Return FibsemDetectorSettings for the given beam type (defaults to active beam)."""
        bt = beam_type if beam_type is not None else self.beam_type
        widget = self.sem_widget if bt is BeamType.ELECTRON else self.fib_widget
        return widget.get_detector_settings()

    def update_from_settings(
        self,
        beam_settings: BeamSettings,
        detector_settings: FibsemDetectorSettings,
    ):
        """Route settings to the correct sub-widget based on beam_settings.beam_type."""
        widget = self.sem_widget if beam_settings.beam_type is BeamType.ELECTRON else self.fib_widget
        widget.update_from_settings(beam_settings, detector_settings)

    def sync_beam_status(self):
        """Refresh beam-on and blanked button icons for the active beam."""
        bt = self.beam_type
        is_on = self.microscope.is_on(bt)
        is_blanked = self.microscope.is_blanked(bt)

        power_color = "#4caf50" if is_on else "#ff9800"
        self.btn_beam_on.setIcon(QIconifyIcon("mdi:power", color=power_color))
        self.btn_beam_on.setToolTip("Beam ON — click to turn off" if is_on else "Beam OFF — click to turn on")

        eye_icon = "mdi:eye-off" if is_blanked else "mdi:eye"
        eye_color = "#ff9800" if is_blanked else "#c0c0c0"
        self.btn_beam_blanked.setIcon(QIconifyIcon(eye_icon, color=eye_color))
        self.btn_beam_blanked.setToolTip("Blanked — click to unblank" if is_blanked else "Unblanked — click to blank")

    def sync_from_microscope(self):
        """Read current settings from the microscope and update both sub-widgets."""
        self.sem_widget.sync_from_microscope()
        self.fib_widget.sync_from_microscope()
        self.sync_beam_status()


if __name__ == "__main__":
    import sys

    from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout

    from fibsem import utils

    app = QApplication(sys.argv)

    microscope, settings = utils.setup_session(manufacturer="Demo", ip_address="localhost", debug=True)

    main_widget = QWidget()
    main_layout = QVBoxLayout()
    main_widget.setLayout(main_layout)

    widget = FibsemDualBeamWidget(microscope=microscope)
    widget.populate_combos()
    widget.sync_from_microscope()
    main_layout.addWidget(widget)

    widget.beam_settings_changed.connect(lambda s: print(f"beam_settings_changed: {s}"))
    widget.detector_settings_changed.connect(lambda s: print(f"detector_settings_changed: {s}"))

    def _print(checked=False):
        from pprint import pprint
        bt = widget.beam_type
        print(f"--- {bt.name} beam ---")
        pprint(widget.get_beam_settings().to_dict())
        pprint(widget.get_detector_settings().to_dict())

    btn = QPushButton("Print Active Settings")
    btn.clicked.connect(_print)
    main_layout.addWidget(btn)

    import napari
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(main_widget, area="right")
    napari.run()
    # main_widget.show()
    # sys.exit(app.exec_())
