import logging

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QButtonGroup,
    QRadioButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.microscope import FibsemMicroscope
from fibsem.structures.settings import BeamSettings, BeamType, FibsemDetectorSettings
from fibsem.theme import tokens
from fibsem.ui import stylesheets
from fibsem.ui.control import control_of, operator_acts
from fibsem.ui.icon import fibsem_icon
from fibsem.ui.widgets.beam_widget import FibsemBeamWidget
from fibsem.ui.widgets.custom_widgets import IconToolButton, TitledPanel
from fibsem.workflows.actions import record_operator_action


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
        layout.setSpacing(0)
        self.setLayout(layout)

        # --- Radio buttons ---
        _RADIO_STYLE = "QRadioButton { font-weight: bold; background: transparent; }"
        self.sem_radio = QRadioButton("SEM")
        self.sem_radio.setStyleSheet(_RADIO_STYLE)
        self.fib_radio = QRadioButton("FIB")
        self.fib_radio.setStyleSheet(_RADIO_STYLE)
        self._button_group = QButtonGroup(self)
        self._button_group.addButton(self.sem_radio)
        self._button_group.addButton(self.fib_radio)

        # --- Beam on / blanked buttons ---
        self.btn_beam_on = QToolButton()
        self.btn_beam_on.setIcon(fibsem_icon("mdi:power", color=tokens.GRAY_ICON_COLOR))
        self.btn_beam_on.setToolTip("Toggle beam on/off")

        self.btn_beam_blanked = QToolButton()
        self.btn_beam_blanked.setIcon(
            fibsem_icon("mdi:eye", color=tokens.GRAY_ICON_COLOR)
        )
        self.btn_beam_blanked.setToolTip("Toggle beam blank/unblank")

        # --- Refresh button ---
        self.btn_refresh = IconToolButton(
            icon="mdi:refresh", tooltip="Refresh from microscope"
        )

        self.btn_beam_on.setStyleSheet(stylesheets.TOOLBUTTON_ICON_STYLESHEET)
        self.btn_beam_blanked.setStyleSheet(stylesheets.TOOLBUTTON_ICON_STYLESHEET)

        # --- Body: stacked beam widgets ---
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(4)

        self.sem_widget = FibsemBeamWidget(
            microscope=self.microscope,
            beam_type=BeamType.ELECTRON,
        )
        self.fib_widget = FibsemBeamWidget(
            microscope=self.microscope,
            beam_type=BeamType.ION,
        )
        body_layout.addWidget(self.sem_widget)
        body_layout.addWidget(self.fib_widget)
        body_layout.addStretch()

        # --- TitledPanel with radios + tool buttons in header ---
        self._panel = TitledPanel("", content=body, collapsible=False)

        # Replace the empty title label with radio buttons
        header_layout = self._panel._header_layout
        title_item = header_layout.takeAt(0)
        if title_item and title_item.widget():
            title_item.widget().deleteLater()
        header_layout.insertWidget(0, self.sem_radio)
        header_layout.insertWidget(1, self.fib_radio)

        # Add tool buttons to header (before hidden collapse btn)
        self._panel.add_header_widget(self.btn_beam_on)
        self._panel.add_header_widget(self.btn_beam_blanked)
        self._panel.add_header_widget(self.btn_refresh)

        layout.addWidget(self._panel)

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
        self.sem_widget.beam_settings_changed.connect(self.beam_settings_changed)
        self.sem_widget.detector_settings_changed.connect(
            self.detector_settings_changed
        )
        self.fib_widget.beam_settings_changed.connect(self.beam_settings_changed)
        self.fib_widget.detector_settings_changed.connect(
            self.detector_settings_changed
        )

    def _on_beam_selected(self, sem_checked: bool):
        self.sem_widget.setVisible(sem_checked)
        self.fib_widget.setVisible(not sem_checked)
        self.sync_beam_status()
        logging.debug({"msg": "_on_beam_selected", "beam_type": self.beam_type.name})

    def _on_beam_on_clicked(self):
        bt = self.beam_type
        if self.microscope.is_on(bt):
            fn, what = self.microscope.turn_off, f"Turn the {bt.name} beam off"
        else:
            fn, what = self.microscope.turn_on, f"Turn the {bt.name} beam on"
        done = operator_acts(
            control_of(self.microscope), what[0].lower() + what[1:], fn, bt
        )
        self.sync_beam_status()
        if done:
            record_operator_action(
                self.microscope, what, via="Beam on/off", beam=bt.name
            )

    def _on_beam_blanked_clicked(self):
        bt = self.beam_type
        if self.microscope.is_blanked(bt):
            fn, what = self.microscope.unblank, f"Unblank the {bt.name} beam"
        else:
            fn, what = self.microscope.blank, f"Blank the {bt.name} beam"
        done = operator_acts(
            control_of(self.microscope), what[0].lower() + what[1:], fn, bt
        )
        self.sync_beam_status()
        if done:
            record_operator_action(
                self.microscope, what, via="Blank/unblank", beam=bt.name
            )

    def _on_refresh_clicked(self):
        self.sync_from_microscope()
        logging.debug({"msg": "_on_refresh_clicked", "beam_type": self.beam_type.name})

    # ------------------------------------------------------------------
    # Public API — delegates to the active sub-widget
    # ------------------------------------------------------------------

    @property
    def active_widget(self) -> FibsemBeamWidget:
        return (
            self.sem_widget if self.beam_type is BeamType.ELECTRON else self.fib_widget
        )

    def populate_combos(self):
        """Populate all comboboxes from the microscope for both beam types."""
        self.sem_widget.populate_combos()
        self.fib_widget.populate_combos()

    def get_beam_settings(self, beam_type: BeamType = None) -> BeamSettings:
        """Return BeamSettings for the given beam type (defaults to active beam)."""
        bt = beam_type if beam_type is not None else self.beam_type
        widget = self.sem_widget if bt is BeamType.ELECTRON else self.fib_widget
        return widget.get_beam_settings()

    def get_detector_settings(
        self, beam_type: BeamType = None
    ) -> FibsemDetectorSettings:
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
        widget = (
            self.sem_widget
            if beam_settings.beam_type is BeamType.ELECTRON
            else self.fib_widget
        )
        widget.update_from_settings(beam_settings, detector_settings)

    def sync_beam_status(self):
        """Refresh beam-on and blanked button icons for the active beam."""
        bt = self.beam_type
        is_on = self.microscope.is_on(bt)
        is_blanked = self.microscope.is_blanked(bt)

        power_color = tokens.GREEN_COLOR if is_on else tokens.ORANGE_COLOR
        self.btn_beam_on.setIcon(fibsem_icon("mdi:power", color=power_color))
        self.btn_beam_on.setToolTip(
            "Beam ON — click to turn off" if is_on else "Beam OFF — click to turn on"
        )

        eye_icon = "mdi:eye-off" if is_blanked else "mdi:eye"
        eye_color = tokens.ORANGE_COLOR if is_blanked else tokens.GRAY_ICON_COLOR
        self.btn_beam_blanked.setIcon(fibsem_icon(eye_icon, color=eye_color))
        self.btn_beam_blanked.setToolTip(
            "Blanked — click to unblank" if is_blanked else "Unblanked — click to blank"
        )

    def sync_from_microscope(self):
        """Read current settings from the microscope and update both sub-widgets."""
        self.sem_widget.sync_from_microscope()
        self.fib_widget.sync_from_microscope()
        self.sync_beam_status()
