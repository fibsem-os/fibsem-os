import logging
from pprint import pprint
from typing import Optional

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import pyqtSignal

from fibsem import config as cfg
from fibsem import guided_setup, utils
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import MicroscopeSettings, SystemSettings
from fibsem.ui import notification_service, stylesheets
from fibsem.ui.icon import fibsem_icon
from fibsem.ui.tokens import (
    BORDER_COLOR,
    NEUTRAL_500,
    PANEL_COLOR,
    PRIMARY_COLOR,
    WHITE_ICON_COLOR,
)
from fibsem.ui.utils import message_box_ui, open_existing_file_dialog
from fibsem.ui.widgets.custom_widgets import (
    ValueComboBox,
)


class FibsemSystemSetupWidget(QtWidgets.QWidget):
    connected_signal = pyqtSignal()
    disconnected_signal = pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent=parent)

        self.microscope: Optional[FibsemMicroscope] = None
        self.settings: Optional[MicroscopeSettings] = None
        # Why the last attempt failed, or None if the last thing that happened was not
        # a failure. Held rather than only toasted: a toast is gone in five seconds,
        # and the reason a connection failed is exactly what the status card should
        # keep showing until the next attempt.
        self._last_connection_error: Optional[str] = None

        # grid layout
        self.gridLayout = QtWidgets.QGridLayout(self)
        self.pushButton_connect_to_microscope = QtWidgets.QPushButton(
            "Connect To Microscope"
        )
        self.pushButton_apply_configuration = QtWidgets.QPushButton(
            "Apply Microscope Configuration"
        )
        self.comboBox_configuration = ValueComboBox()
        self.toolButton_import_configuration = QtWidgets.QToolButton()
        self.label_connection_status = QtWidgets.QLabel("No Connected")
        self.label_connection_information = QtWidgets.QLabel("No Connected")
        self.label_connection = QtWidgets.QLabel("Configuration")

        # Row 0 is the first-run offer, above the configuration it is offering to
        # create. Everything else sits a row lower than it reads in the file.
        self._frame_first_run = self._create_first_run_callout()
        self.gridLayout.addWidget(self._frame_first_run, 0, 0, 1, 3)
        self.gridLayout.addWidget(self.label_connection, 1, 0, 1, 1)
        self.gridLayout.addWidget(self.comboBox_configuration, 1, 1, 1, 1)
        self.gridLayout.addWidget(self.toolButton_import_configuration, 1, 2, 1, 1)
        self.gridLayout.addWidget(self.pushButton_connect_to_microscope, 2, 0, 1, 3)
        self.gridLayout.addWidget(self.pushButton_apply_configuration, 3, 0, 1, 3)
        self.gridLayout.addWidget(self.label_connection_status, 4, 0, 1, 3)
        self.gridLayout.addWidget(self.label_connection_information, 5, 0, 1, 3)
        self.gridLayout.addItem(
            QtWidgets.QSpacerItem(
                20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
            ),
            6,
            0,
            1,
            3,
        )

        # hide the old status labels and replace with a card
        self.label_connection_status.setVisible(False)
        self.label_connection_information.setVisible(False)
        self._frame_status = self._create_connection_status_card()
        self.gridLayout.addWidget(self._frame_status, 5, 0, 1, 3)

        self.setup_connections()
        self.update_ui()
        self.refresh_first_run_offer()

    def _create_first_run_callout(self) -> QtWidgets.QFrame:
        """The offer to run the guided setup, shown only on a fresh install.

        Tinted rather than coloured, with an outline button: this is an offer, not a
        warning, and the tab it appears on is one people open every session. It is
        dismissible for the same reason -- someone who intends to configure by hand
        should be able to say so once.
        """
        frame = QtWidgets.QFrame()
        frame.setObjectName("frame_first_run")
        frame.setStyleSheet(f"""
            QFrame#frame_first_run {{
                background-color: rgba(0, 122, 204, 0.12);
                border: 1px solid {PRIMARY_COLOR};
                border-radius: 6px;
            }}
        """)

        layout = QtWidgets.QHBoxLayout(frame)
        layout.setContentsMargins(12, 10, 8, 10)
        layout.setSpacing(10)

        icon = QtWidgets.QLabel()
        icon.setStyleSheet("background: transparent; border: none;")
        icon.setFixedSize(20, 20)
        icon.setPixmap(fibsem_icon("mdi:auto-fix", color=PRIMARY_COLOR).pixmap(20, 20))
        layout.addWidget(icon, 0, QtCore.Qt.AlignTop)

        text = QtWidgets.QVBoxLayout()
        text.setSpacing(2)
        title = QtWidgets.QLabel("First time here?")
        title.setStyleSheet(
            f"background: transparent; border: none; color: {PRIMARY_COLOR};"
            " font-weight: bold; font-size: 11px;"
        )
        subtitle = QtWidgets.QLabel(
            "A guided walkthrough that configures fibsemOS to work with your microscope."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(
            f"background: transparent; border: none; color: {NEUTRAL_500}; font-size: 10px;"
        )
        text.addWidget(title)
        text.addWidget(subtitle)
        layout.addLayout(text, 1)

        self._button_run_wizard = QtWidgets.QPushButton("Start Guided Setup")
        self._button_run_wizard.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {PRIMARY_COLOR};
                border: 1px solid {PRIMARY_COLOR};
                border-radius: 4px;
                padding: 5px 12px;
                font-size: 11px;
                font-weight: bold;
            }}
            QPushButton:hover {{ background-color: rgba(0, 122, 204, 0.20); }}
        """)
        self._button_run_wizard.clicked.connect(self.run_guided_setup)
        layout.addWidget(self._button_run_wizard, 0, QtCore.Qt.AlignVCenter)

        self._button_dismiss_first_run = QtWidgets.QToolButton()
        self._button_dismiss_first_run.setIcon(
            fibsem_icon("mdi:close", color=NEUTRAL_500)
        )
        self._button_dismiss_first_run.setToolTip("Do not offer this again")
        self._button_dismiss_first_run.setAutoRaise(True)
        self._button_dismiss_first_run.setStyleSheet(
            "border: none; background: transparent;"
        )
        self._button_dismiss_first_run.clicked.connect(self._dismiss_first_run)
        layout.addWidget(self._button_dismiss_first_run, 0, QtCore.Qt.AlignTop)

        # Hidden until refresh_first_run_offer decides otherwise, which cannot happen
        # here -- it reads self._frame_first_run, and that is what this returns.
        frame.setVisible(False)
        return frame

    def refresh_first_run_offer(self, preferences=None) -> None:
        """Show the offer when nothing is configured yet and it has not been dismissed.

        Two separate conditions on purpose. Folding them together is what broke this
        the first time: dismissal used to be inferred from the same file whose
        absence meant "fresh install", which made recording a dismissal look like an
        undo of it.

        Takes the whole preferences object rather than a bool, so AutoLamella's
        `_apply_preferences` can pass the one it already holds. None reads them, for
        the standalone widget that has no host.
        """
        if preferences is None:
            preferences = cfg.load_user_preferences()
        self._frame_first_run.setVisible(
            not preferences.display.guided_setup_dismissed
            and guided_setup.is_first_run()
        )

    def _dismiss_first_run(self) -> None:
        """Hide the offer, and record that it was declined."""
        self._frame_first_run.setVisible(False)
        guided_setup.dismiss_first_run()

    def run_guided_setup(self) -> Optional[str]:
        """Open the wizard, and select whatever it saved.

        The live microscope is handed over so the wizard can read the stage without
        opening a second client against the same instrument.
        """
        from fibsem.ui.widgets.guided_setup_dialog import open_guided_setup

        name = open_guided_setup(parent=self, microscope=self.microscope)
        if name is None:
            # Backing out is not declining. The offer stays where it was, so someone
            # who cancelled to go and read the instrument's address can pick it up
            # again without hunting through the menus.
            return None
        # Finishing writes the preferences file that is_first_run reads, so this only
        # brings the change forward to now rather than to the next start.
        self._frame_first_run.setVisible(False)

        combo = self.comboBox_configuration
        combo.blockSignals(True)
        index = combo.findText(name)
        if index == -1:
            combo.addItem(name)
            index = combo.count() - 1
        combo.setCurrentIndex(index)
        combo.blockSignals(False)
        self.load_configuration(name)
        notification_service.show_toast(f"Configuration {name} is ready.", "info")
        return name

    def _create_connection_status_card(self) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setObjectName("frame_connection_status")
        frame.setStyleSheet(f"""
            QFrame#frame_connection_status {{
                background-color: {PANEL_COLOR};
                border-radius: 6px;
                border: 1px solid {BORDER_COLOR};
            }}
        """)

        layout = QtWidgets.QHBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self._label_status_icon = QtWidgets.QLabel()
        self._label_status_icon.setStyleSheet("border: none;")
        self._label_status_icon.setFixedSize(20, 20)
        layout.addWidget(self._label_status_icon)

        text_layout = QtWidgets.QVBoxLayout()
        text_layout.setSpacing(2)

        self._label_status_title = QtWidgets.QLabel("Microscope Connected")
        self._label_status_title.setStyleSheet(
            f"background-color: transparent; color: {WHITE_ICON_COLOR}; font-weight: bold; font-size: 11px; border: none;"
        )

        self._label_status_subtitle = QtWidgets.QLabel("")
        # Wrapped, because this now carries the reason a connection failed -- a full
        # sentence from the backend rather than a two-word status.
        self._label_status_subtitle.setWordWrap(True)
        self._label_status_subtitle.setStyleSheet(
            f"background-color: transparent; color: {NEUTRAL_500}; font-size: 10px; border: none;"
        )

        text_layout.addWidget(self._label_status_title)
        text_layout.addWidget(self._label_status_subtitle)
        layout.addLayout(text_layout)
        layout.addStretch()

        self._button_disconnect = QtWidgets.QPushButton("Disconnect")
        self._button_disconnect.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {NEUTRAL_500};
                border: 1px solid {BORDER_COLOR};
                border-radius: 3px;
                padding: 3px 8px;
                font-size: 10px;
            }}
            QPushButton:hover {{
                color: #f44336;
                border-color: #f44336;
            }}
        """)
        self._button_disconnect.clicked.connect(self.connect_to_microscope)
        layout.addWidget(self._button_disconnect)

        return frame

    def setup_connections(self):

        # connection
        self.pushButton_connect_to_microscope.clicked.connect(
            self.connect_to_microscope
        )

        # configuration
        self.comboBox_configuration.addItems(cfg.USER_CONFIGURATIONS.keys())
        self.comboBox_configuration.setCurrentText(cfg.DEFAULT_CONFIGURATION_NAME)
        self.comboBox_configuration.currentTextChanged.connect(
            lambda: self.load_configuration(None)
        )
        self.toolButton_import_configuration.clicked.connect(
            self.import_configuration_from_file
        )

        self.pushButton_apply_configuration.clicked.connect(
            lambda: self.apply_microscope_configuration(None)
        )
        self.pushButton_apply_configuration.setToolTip(
            "Apply configuration can take some time. Please make sure the microscope beams are both on."
        )
        self.toolButton_import_configuration.setIcon(
            fibsem_icon("mdi:add", color=NEUTRAL_500)
        )

    def load_configuration(
        self, configuration_name: Optional[str] = None
    ) -> Optional[str]:
        if configuration_name is None:
            configuration_name = self.comboBox_configuration.currentText()

        # this runs as a currentTextChanged slot, where an exception is fatal rather
        # than merely logged, so an unregistered name or an unreadable file has to
        # report itself instead of raising.
        configuration = cfg.USER_CONFIGURATIONS.get(configuration_name)
        configuration_path = configuration.get("path") if configuration else None

        if configuration_path is None:
            notification_service.show_toast(
                f"Configuration {configuration_name} not found.", "error"
            )
            return None

        # load the configuration
        try:
            self.settings = utils.load_microscope_configuration(configuration_path)
        except Exception as e:
            logging.warning(
                f"Unable to load configuration {configuration_name} from {configuration_path}: {e}"
            )
            notification_service.show_toast(
                f"Unable to load configuration {configuration_name}: {e}", "error"
            )
            return None

        pprint(self.settings.to_dict()["info"])

        return configuration_path

    def import_configuration_from_file(self):

        path = open_existing_file_dialog(
            msg="Select microscope configuration file",
            path=cfg.CONFIG_PATH,
            _filter="YAML (*.yaml *.yml)",
            parent=self,
        )

        if path == "":
            notification_service.show_toast(
                "No file selected. Configuration not loaded.", "error"
            )
            return

        # TODO: validate configuration

        # register the configuration. this is unconditional: registering is what makes a
        # configuration selectable, so skipping it left the combo box holding a name that
        # resolved to nothing, and selecting it took the app down.
        known_names = set(cfg.USER_CONFIGURATIONS)
        configuration_name = cfg.register_configuration(path=path)

        # set default configuration. only worth asking for a configuration we hadn't
        # already seen; re-importing a known one shouldn't re-prompt.
        if configuration_name not in known_names:
            msg = "Would you like to make this the default configuration?"
            ret = message_box_ui(
                text=msg, title="Set default configuration?", parent=self
            )

            if ret:
                cfg.set_default_configuration(configuration_name=configuration_name)

        # add configuration to combobox, reusing the entry if it is already listed.
        # select it quietly and load once: setCurrentIndex stays silent when the entry is
        # already current, so leaving the load to the signal would skip it on re-import.
        combo = self.comboBox_configuration
        combo.blockSignals(True)
        index = combo.findText(configuration_name)
        if index == -1:
            combo.addItem(configuration_name)
            index = combo.count() - 1
        combo.setCurrentIndex(index)
        combo.blockSignals(False)

        self.load_configuration(configuration_name)

    def connect_to_microscope(self):

        is_microscope_connected = bool(self.microscope)

        if is_microscope_connected:
            try:
                self.microscope.disconnect()
            except Exception as e:
                # Same reasoning as the connect path below: an exception escaping this
                # slot is a process abort rather than an error message (FIB-329). A
                # disconnect can fail for ordinary reasons -- the instrument went away,
                # the client is already dead -- and none of them are worth losing the
                # application over.
                #
                # Logged rather than only toasted, because a client that would not
                # close is a leak, and the log is where anyone would look for it.
                logging.error(f"Could not cleanly disconnect the microscope: {e}")
                # Phrased as one outcome, not two. The tab *does* drop to disconnected
                # a line below, so "Disconnect failed" beside a disconnected tab reads
                # as a contradiction; what actually happened is that we let go of a
                # client that would not close.
                notification_service.show_toast(
                    f"Disconnected, but the client did not close cleanly: {e}", "error"
                )
            # Cleared either way, and outside the try for that reason. The request was
            # to disconnect; holding a client that could not be closed would leave the
            # tab offering to disconnect something it can no longer reach, with no way
            # back to a working connection.
            self.microscope, self.settings = None, None
        else:
            # Cleared before the attempt, so a retry never shows the previous reason
            # beside a connection that is still being made.
            self._last_connection_error = None
            notification_service.show_toast("Connecting to microscope...", "info")

            configuration_path = self.load_configuration(None)

            if configuration_path is None:
                notification_service.show_toast("Configuration not selected.", "error")
                return

            # connect
            try:
                self.microscope, self.settings = utils.setup_session(
                    config_path=configuration_path,
                )
            except Exception as e:
                # Reported, not raised. This runs as a Qt slot, and PyQt5 turns an
                # unhandled exception in a slot into qFatal -- the entire application
                # aborts, leaving the traceback and nothing else (FIB-329). Failing to
                # connect is an ordinary outcome here rather than a defect: the vendor
                # API may not be installed, the instrument may be off, the address may
                # belong to a different bay.
                #
                # Broad on purpose. The backends raise whatever their own SDK raises,
                # and the point is that *nothing* from this call reaches Qt -- a
                # narrower except would leave the abort in place for the exception
                # nobody predicted, which is the one that will happen.
                self.microscope, self.settings = None, None
                self._last_connection_error = str(e)
                logging.error(f"Could not connect to the microscope: {e}")
                notification_service.show_toast(f"Could not connect: {e}", "error")
            else:
                # user notification
                msg = f"Connected to microscope at {self.microscope.system.info.ip_address}"
                logging.info(msg)
                notification_service.show_toast(msg, "info")

        self.update_ui()

    def apply_microscope_configuration(
        self, system_settings: Optional[SystemSettings] = None
    ):
        """Apply the microscope configuration to the microscope."""

        if self.microscope is None:
            notification_service.show_toast("Microscope not connected.", "error")
            return

        # apply the configuration
        self.microscope.apply_configuration(system_settings=system_settings)

    def update_ui(self):

        is_microscope_connected = bool(self.microscope)
        self.pushButton_apply_configuration.setVisible(is_microscope_connected)
        self.pushButton_apply_configuration.setEnabled(
            is_microscope_connected and cfg.APPLY_CONFIGURATION_ENABLED
        )

        if is_microscope_connected:
            self.pushButton_connect_to_microscope.setVisible(False)
            self.pushButton_apply_configuration.setStyleSheet(
                stylesheets.SECONDARY_BUTTON_STYLESHEET
            )
            self.connected_signal.emit()

            info = self.microscope.system.info
            self._label_status_icon.setPixmap(
                fibsem_icon("mdi:check-circle", color=stylesheets.GREEN_COLOR).pixmap(
                    20, 20
                )
            )
            self._label_status_title.setText("Microscope Connected")
            self._label_status_subtitle.setText(
                f"Connected to {info.manufacturer}-{info.model} at {info.ip_address}"
            )
            self._button_disconnect.setVisible(True)

        else:
            self.pushButton_connect_to_microscope.setVisible(True)
            self.pushButton_connect_to_microscope.setText("Connect To Microscope")
            self.pushButton_connect_to_microscope.setStyleSheet(
                stylesheets.PRIMARY_BUTTON_STYLESHEET
            )
            self.pushButton_apply_configuration.setStyleSheet(
                stylesheets.SECONDARY_BUTTON_STYLESHEET
            )
            self.disconnected_signal.emit()

            # "Not connected" and "tried and failed" are different states, and the
            # difference is exactly what someone needs. The card says which, and it is
            # on screen regardless of whether toasts are enabled.
            failed = self._last_connection_error is not None
            self._label_status_icon.setPixmap(
                fibsem_icon(
                    "mdi:alert-circle" if failed else "mdi:close-circle",
                    color="#f44336",
                ).pixmap(20, 20)
            )
            self._label_status_title.setText(
                "Connection Failed" if failed else "Not Connected"
            )
            self._label_status_subtitle.setText(
                self._last_connection_error or "No microscope connected"
            )
            self._button_disconnect.setVisible(False)


def main():
    """Standalone harness: show the widget in a plain Qt window.

    napari was only ever hosting the widget here (FIB-407). It also supplied the
    QApplication and the dark theme, so both are set up explicitly — this widget
    paints itself in the dark palette (PANEL_COLOR, BORDER_COLOR, the status
    icons), which needs NAPARI_STYLE underneath it to read correctly.
    """
    import sys

    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(stylesheets.NAPARI_STYLE)

    system_widget = FibsemSystemSetupWidget()
    system_widget.setWindowTitle("Microscope Setup")
    system_widget.resize(400, 300)
    system_widget.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
