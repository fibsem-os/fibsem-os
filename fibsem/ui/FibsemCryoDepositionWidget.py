import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import napari
from PyQt5 import QtWidgets
from PyQt5.QtGui import QFont

from fibsem import config as cfg
from fibsem import gis, utils
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import FibsemGasInjectionSettings
from fibsem.ui.utils import WheelBlocker


class FibsemCryoDepositionWidget(QtWidgets.QDialog):
    def __init__(
        self,
        microscope: FibsemMicroscope,
        parent=None,
    ):
        super(FibsemCryoDepositionWidget, self).__init__(parent=parent)
        self._setup_ui()
        self.setWindowTitle("Cryo Deposition")

        self.microscope = microscope

        self.setup_connections()

    def _setup_ui(self):
        """Hand-built replacement for the former Qt Designer form."""
        self.wheel_blocker = WheelBlocker()
        layout = QtWidgets.QGridLayout(self)

        self.label_title = QtWidgets.QLabel("Cryo Deposition")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title_font.setWeight(75)
        self.label_title.setFont(title_font)
        layout.addWidget(self.label_title, 0, 0, 1, 2)

        self.label_stage_position = QtWidgets.QLabel("Stage Position")
        self.comboBox_stage_position = QtWidgets.QComboBox()
        layout.addWidget(self.label_stage_position, 1, 0)
        layout.addWidget(self.comboBox_stage_position, 1, 1)

        self.label_port = QtWidgets.QLabel("Port")
        self.comboBox_port = QtWidgets.QComboBox()
        layout.addWidget(self.label_port, 2, 0)
        layout.addWidget(self.comboBox_port, 2, 1)

        self.label_gas = QtWidgets.QLabel("Gas")
        self.lineEdit_gas = QtWidgets.QLineEdit("Pt cryo")
        layout.addWidget(self.label_gas, 3, 0)
        layout.addWidget(self.lineEdit_gas, 3, 1)

        self.label_insert_position = QtWidgets.QLabel("Insert Position")
        self.lineEdit_insert_position = QtWidgets.QLineEdit("cryo")
        layout.addWidget(self.label_insert_position, 4, 0)
        layout.addWidget(self.lineEdit_insert_position, 4, 1)

        self.label_duration = QtWidgets.QLabel("Duration (s)")
        self.doubleSpinBox_duration = QtWidgets.QDoubleSpinBox()
        self.doubleSpinBox_duration.setMaximum(1000.0)
        self.doubleSpinBox_duration.setValue(30.0)
        layout.addWidget(self.label_duration, 5, 0)
        layout.addWidget(self.doubleSpinBox_duration, 5, 1)

        self.pushButton_run_sputter = QtWidgets.QPushButton("Run Cryo Deposition")
        layout.addWidget(self.pushButton_run_sputter, 6, 0, 1, 2)

        # block accidental scroll-to-change on the input widgets
        for w in (self.comboBox_stage_position, self.comboBox_port, self.doubleSpinBox_duration):
            w.installEventFilter(self.wheel_blocker)

    def setup_connections(self):

        self.pushButton_run_sputter.clicked.connect(self.run_sputter)

        positions = utils.load_yaml(Path(cfg.POSITION_PATH))
        self.comboBox_stage_position.addItems(["Current Position"] + [p["name"] for p in positions])
        available_ports = self.microscope.get_available_values("gis_ports")
        self.comboBox_port.addItems([str(p) for p in available_ports])


        # TODO: show / hide based on gis / multichem available
        multichem_available = self.microscope.is_available("gis_multichem")
        self.lineEdit_gas.setVisible(multichem_available)
        self.label_gas.setVisible(multichem_available)
        self.lineEdit_insert_position.setVisible(multichem_available)
        self.label_insert_position.setVisible(multichem_available)
        self.comboBox_port.setVisible(not multichem_available)  # gis only
        self.label_port.setVisible(not multichem_available)     # gis only

    def _get_protocol_from_ui(self) -> Dict[str, Union[str, float]]:

        protocol = {
            "port": self.comboBox_port.currentText(),
            "gas": self.lineEdit_gas.text(),
            "insert_position": self.lineEdit_insert_position.text(),
            "duration": self.doubleSpinBox_duration.value(),
            "name": self.comboBox_stage_position.currentText(),

        }

        return protocol

    # TODO: thread this, add progress bar, feedback
    def run_sputter(self):
        
        gdict = self._get_protocol_from_ui()
        gis_settings = FibsemGasInjectionSettings.from_dict(gdict)
        name: Optional[str] = gdict["name"]

        if name == "Current Position":
            name = None

        gis.cryo_deposition_v2(self.microscope, gis_settings, name=name)


def main():

    viewer = napari.Viewer(ndisplay=2)
    microscope, settings = utils.setup_session()
    cryo_sputter_widget = FibsemCryoDepositionWidget(microscope)
    viewer.window.add_dock_widget(
        cryo_sputter_widget, area="right", add_vertical_stretch=False
    )
    napari.run()


if __name__ == "__main__":
    main()  