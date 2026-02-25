import logging
import os
from pprint import pprint
from typing import Optional

import napari
import napari.utils.notifications
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal
from superqt import QIconifyIcon

from fibsem import config as cfg
from fibsem import utils
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import MicroscopeSettings, SystemSettings
from fibsem.ui import stylesheets
from fibsem.ui.utils import message_box_ui, open_existing_file_dialog


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
        self.comboBox_configuration = QtWidgets.QComboBox()
        self.toolButton_import_configuration = QtWidgets.QToolButton()
        self.label_connection_status = QtWidgets.QLabel("No Connected")
        self.label_connection_information = QtWidgets.QLabel("No Connected")
        self.label_connection = QtWidgets.QLabel("Connection")

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
        frame.setStyleSheet("""
            QFrame#frame_connection_status {
                background-color: #1e2027;
                border-radius: 6px;
                border: 1px solid #3d4251;
            }
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
            "color: #ffffff; font-weight: bold; font-size: 11px; border: none;"
        )

        self._label_status_subtitle = QtWidgets.QLabel("")
        self._label_status_subtitle.setStyleSheet(
            "color: #a0a0a0; font-size: 10px; border: none;"
        )

        text_layout.addWidget(self._label_status_title)
        text_layout.addWidget(self._label_status_subtitle)
        layout.addLayout(text_layout)
        layout.addStretch()

        self._button_disconnect = QtWidgets.QPushButton("Disconnect")
        self._button_disconnect.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #a0a0a0;
                border: 1px solid #3d4251;
                border-radius: 3px;
                padding: 3px 8px;
                font-size: 10px;
            }
            QPushButton:hover {
                color: #f44336;
                border-color: #f44336;
            }
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
        self.toolButton_import_configuration.setIcon(QIconifyIcon("mdi:add", color="#a0a0a0"))

    def load_configuration(self, configuration_name: Optional[str] = None) -> Optional[str]:
        if configuration_name is None:
            configuration_name = self.comboBox_configuration.currentText()

        configuration_path = cfg.USER_CONFIGURATIONS[configuration_name]["path"]

        if configuration_path is None:
            napari.utils.notifications.show_error(f"Configuration {configuration_name} not found.")
            return

        # load the configuration
        self.settings = utils.load_microscope_configuration(configuration_path)

        pprint(self.settings.to_dict()["info"])

        return configuration_path

    def import_configuration_from_file(self):

        path = open_existing_file_dialog(msg="Select microscope configuration file",
            path=cfg.CONFIG_PATH,
            _filter="YAML (*.yaml *.yml)",
            parent=self
        )

        if path == "":
            napari.utils.notifications.show_error("No file selected. Configuration not loaded.")
            return

        # TODO: validate configuration

        # ask user to add to user configurations
        if hasattr(str, "removesuffix"):
            configuration_name = os.path.basename(path).removesuffix(".yaml")
        else:
            configuration_name = os.path.basename(path).replace(".yaml", "")

        if configuration_name not in cfg.USER_CONFIGURATIONS:
            msg = "Would you like to add this configuration to the user configurations?"
            ret = message_box_ui(text=msg, title="Add to user configurations?", parent=self)

            # add to user configurations
            if ret:
                cfg.add_configuration(configuration_name=configuration_name, path=path)

                # set default configuration
                msg = "Would you like to make this the default configuration?"
                ret = message_box_ui(text=msg, title="Set default configuration?", parent=self)

                if ret:
                    cfg.set_default_configuration(configuration_name=configuration_name)

        # add configuration to combobox
        self.comboBox_configuration.addItem(configuration_name)
        self.comboBox_configuration.setCurrentText(configuration_name)

    def connect_to_microscope(self):

        is_microscope_connected = bool(self.microscope)

        if is_microscope_connected:
            self.microscope.disconnect()
            self.microscope, self.settings = None, None
        else:

            napari.utils.notifications.show_info("Connecting to microscope...")

            configuration_path = self.load_configuration(None)

            if configuration_path is None:
                napari.utils.notifications.show_error("Configuration not selected.")
                return

            # connect
            self.microscope, self.settings = utils.setup_session(
                config_path=configuration_path,
            )

            # user notification
            msg = f"Connected to microscope at {self.microscope.system.info.ip_address}"
            logging.info(msg)
            napari.utils.notifications.show_info(msg)

        self.update_ui()


    def apply_microscope_configuration(self, system_settings: Optional[SystemSettings] = None):
        """Apply the microscope configuration to the microscope."""

        if self.microscope is None:
            napari.utils.notifications.show_error("Microscope not connected.")
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
                QIconifyIcon("mdi:check-circle", color="#4caf50").pixmap(20, 20)
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
                QIconifyIcon("mdi:close-circle", color="#f44336").pixmap(20, 20)
            )
            self._label_status_title.setText("Not Connected")
            self._label_status_subtitle.setText("No microscope connected")
            self._button_disconnect.setVisible(False)


def main():

    viewer = napari.Viewer(ndisplay=2)
    system_widget = FibsemSystemSetupWidget()
    viewer.window.add_dock_widget(
        system_widget, area="right", add_vertical_stretch=False
    )
    napari.run()


if __name__ == "__main__":
    main()
