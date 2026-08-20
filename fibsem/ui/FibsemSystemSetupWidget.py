import logging
from pprint import pprint
from typing import Optional

from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal

from fibsem import config as cfg
from fibsem import utils
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import MicroscopeSettings, SystemSettings
from fibsem.ui import notification_service, stylesheets
from fibsem.ui.icon import fibsem_icon
from fibsem.ui.tokens import (
    BORDER_COLOR,
    NEUTRAL_500,
    PANEL_COLOR,
    WHITE_ICON_COLOR,
)
from fibsem.ui.utils import message_box_ui, open_existing_file_dialog
from fibsem.ui.widgets.custom_widgets import (
    ValueComboBox,
)


class FibsemSystemSetupWidget(QtWidgets.QWidget):
    connected_signal = pyqtSignal()
    disconnected_signal = pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget]=None):
        super().__init__(parent=parent)

        self.microscope: Optional[FibsemMicroscope] = None
        self.settings: Optional[MicroscopeSettings] = None

        # grid layout
        self.gridLayout = QtWidgets.QGridLayout(self)
        self.pushButton_connect_to_microscope = QtWidgets.QPushButton("Connect To Microscope")
        self.pushButton_apply_configuration = QtWidgets.QPushButton("Apply Microscope Configuration")
        self.comboBox_configuration = ValueComboBox()
        self.toolButton_import_configuration = QtWidgets.QToolButton()
        self.label_connection_status = QtWidgets.QLabel("No Connected")
        self.label_connection_information = QtWidgets.QLabel("No Connected")
        self.label_connection = QtWidgets.QLabel("Configuration")

        self.gridLayout.addWidget(self.label_connection, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.comboBox_configuration, 0, 1, 1, 1)
        self.gridLayout.addWidget(self.toolButton_import_configuration, 0, 2, 1, 1)
        self.gridLayout.addWidget(self.pushButton_connect_to_microscope, 1, 0, 1, 3)
        self.gridLayout.addWidget(self.pushButton_apply_configuration, 2, 0, 1, 3)
        self.gridLayout.addWidget(self.label_connection_status, 3, 0, 1, 3)
        self.gridLayout.addWidget(self.label_connection_information, 4, 0, 1, 3)
        self.gridLayout.addItem(
            QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding),
            5, 0, 1, 3
        )

        # hide the old status labels and replace with a card
        self.label_connection_status.setVisible(False)
        self.label_connection_information.setVisible(False)
        self._frame_status = self._create_connection_status_card()
        self.gridLayout.addWidget(self._frame_status, 4, 0, 1, 3)

        self.setup_connections()
        self.update_ui()

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
        self.pushButton_connect_to_microscope.clicked.connect(self.connect_to_microscope)

        # configuration
        self.comboBox_configuration.addItems(cfg.USER_CONFIGURATIONS.keys())
        self.comboBox_configuration.setCurrentText(cfg.DEFAULT_CONFIGURATION_NAME)
        self.comboBox_configuration.currentTextChanged.connect(lambda: self.load_configuration(None))
        self.toolButton_import_configuration.clicked.connect(self.import_configuration_from_file)

        self.pushButton_apply_configuration.clicked.connect(lambda: self.apply_microscope_configuration(None))
        self.pushButton_apply_configuration.setToolTip("Apply configuration can take some time. Please make sure the microscope beams are both on.")
        self.toolButton_import_configuration.setIcon(fibsem_icon("mdi:add", color=NEUTRAL_500))

    def load_configuration(self, configuration_name: Optional[str] = None) -> Optional[str]:
        if configuration_name is None:
            configuration_name = self.comboBox_configuration.currentText()

        # this runs as a currentTextChanged slot, where an exception is fatal rather
        # than merely logged, so an unregistered name or an unreadable file has to
        # report itself instead of raising.
        configuration = cfg.USER_CONFIGURATIONS.get(configuration_name)
        configuration_path = configuration.get("path") if configuration else None

        if configuration_path is None:
            notification_service.show_toast(f"Configuration {configuration_name} not found.", "error")
            return None

        # load the configuration
        try:
            self.settings = utils.load_microscope_configuration(configuration_path)
        except Exception as e:
            logging.warning(f"Unable to load configuration {configuration_name} from {configuration_path}: {e}")
            notification_service.show_toast(f"Unable to load configuration {configuration_name}: {e}", "error")
            return None

        pprint(self.settings.to_dict()["info"])

        return configuration_path

    def import_configuration_from_file(self):

        path = open_existing_file_dialog(msg="Select microscope configuration file",
            path=cfg.CONFIG_PATH,
            _filter="YAML (*.yaml *.yml)",
            parent=self
        )

        if path == "":
            notification_service.show_toast("No file selected. Configuration not loaded.", "error")
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
            ret = message_box_ui(text=msg, title="Set default configuration?", parent=self)

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
            self.microscope.disconnect()
            self.microscope, self.settings = None, None
        else:

            notification_service.show_toast("Connecting to microscope...", "info")

            configuration_path = self.load_configuration(None)

            if configuration_path is None:
                notification_service.show_toast("Configuration not selected.", "error")
                return

            # connect
            self.microscope, self.settings = utils.setup_session(
                config_path=configuration_path,
            )

            # user notification
            msg = f"Connected to microscope at {self.microscope.system.info.ip_address}"
            logging.info(msg)
            notification_service.show_toast(msg, "info")

        self.update_ui()


    def apply_microscope_configuration(self, system_settings: Optional[SystemSettings] = None):
        """Apply the microscope configuration to the microscope."""

        if self.microscope is None:
            notification_service.show_toast("Microscope not connected.", "error")
            return

        # apply the configuration
        self.microscope.apply_configuration(system_settings=system_settings)

    def update_ui(self):

        is_microscope_connected = bool(self.microscope)
        self.pushButton_apply_configuration.setVisible(is_microscope_connected)
        self.pushButton_apply_configuration.setEnabled(is_microscope_connected and cfg.APPLY_CONFIGURATION_ENABLED)

        if is_microscope_connected:
            self.pushButton_connect_to_microscope.setVisible(False)
            self.pushButton_apply_configuration.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
            self.connected_signal.emit()

            info = self.microscope.system.info
            self._label_status_icon.setPixmap(
                fibsem_icon("mdi:check-circle", color=stylesheets.GREEN_COLOR).pixmap(20, 20)
            )
            self._label_status_title.setText("Microscope Connected")
            self._label_status_subtitle.setText(
                f"Connected to {info.manufacturer}-{info.model} at {info.ip_address}"
            )
            self._button_disconnect.setVisible(True)

        else:
            self.pushButton_connect_to_microscope.setVisible(True)
            self.pushButton_connect_to_microscope.setText("Connect To Microscope")
            self.pushButton_connect_to_microscope.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
            self.pushButton_apply_configuration.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
            self.disconnected_signal.emit()

            self._label_status_icon.setPixmap(
                fibsem_icon("mdi:close-circle", color="#f44336").pixmap(20, 20)
            )
            self._label_status_title.setText("Not Connected")
            self._label_status_subtitle.setText("No microscope connected")
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
