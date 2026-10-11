import logging
from typing import TYPE_CHECKING, Optional

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox

from fibsem import constants
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import BeamType, FibsemManipulatorPosition, MicroscopeSettings
from fibsem.ui import notification_service, stylesheets
from fibsem.ui.utils import install_wheel_blocker_recursive, message_box_ui

if TYPE_CHECKING:
    # Annotation-only, to avoid a runtime dependency on a sibling widget module.
    from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget


class FibsemManipulatorWidget(QtWidgets.QWidget):
    def __init__(
        self,
        microscope: FibsemMicroscope = None,
        settings: MicroscopeSettings = None,
        # Unused, retained for call-site compatibility. Untyped: every host now passes
        # None, so naming napari here would be the module's only reason to know it exists.
        viewer=None,
        image_widget: Optional["FibsemImageSettingsWidget"] = None,
        parent=None,
    ):
        super().__init__(parent=parent)
        self._build_ui()
        # plain spinboxes/comboboxes: guard them against scroll-to-change
        install_wheel_blocker_recursive(self)

        self.microscope = microscope
        self.settings = settings
        self.viewer = viewer
        self.image_widget = image_widget
        self.saved_positions = {}

        # What the controls offer is the driver's answer, not the class's.
        self.named_positions = list(self.microscope.manipulator_named_positions())
        self.move_types = tuple(self.microscope.manipulator_move_types)
        self.has_rotation = bool(self.microscope.is_available("manipulator_rotation"))
        self._controls_shown = True

        self.setup_connections()

        self.update_ui()

        self.calibrated_status_label.setVisible(False)
        self.savedPosition_combobox.addItems(self.named_positions)
        self.move_type_comboBox.setCurrentIndex(0)
        self.move_type_comboBox.currentIndexChanged.connect(self.change_move_type)

        manipulator_inserted = self.microscope.get_manipulator_state()
        self._hide_show_buttons(manipulator_inserted)
        self.insertManipulator_button.setText(
            "Retract" if manipulator_inserted else "Insert"
        )
        self.manipulatorStatus_label.setText(
            "Manipulator Status: Inserted"
            if manipulator_inserted
            else "Manipulator Status: Retracted"
        )

    def _build_ui(self) -> None:
        layout = QtWidgets.QGridLayout(self)

        def spinbox(limit: float) -> QtWidgets.QDoubleSpinBox:
            box = QtWidgets.QDoubleSpinBox()
            box.setRange(-limit, limit)
            return box

        # status
        self.calibrated_status_label = QtWidgets.QLabel("")
        layout.addWidget(self.calibrated_status_label, 0, 0)
        self.manipulatorStatus_label = QtWidgets.QLabel("")
        layout.addWidget(self.manipulatorStatus_label, 1, 0, 1, 2)
        self.insertManipulator_button = QtWidgets.QPushButton("Insert")
        layout.addWidget(self.insertManipulator_button, 3, 0, 1, 2)
        self.pushButton_refresh_data = QtWidgets.QPushButton("Refresh Data")
        layout.addWidget(self.pushButton_refresh_data, 4, 0, 1, 2)

        # move
        self.move_type_comboBox = QtWidgets.QComboBox()
        self.move_type_comboBox.addItems(["Relative Move", "Corrected Move"])
        layout.addWidget(self.move_type_comboBox, 5, 0, 1, 2)
        self.dx_label = QtWidgets.QLabel("dX (um)")
        self.dX_spinbox = spinbox(1000.0)
        layout.addWidget(self.dx_label, 6, 0)
        layout.addWidget(self.dX_spinbox, 6, 1)
        self.dy_label = QtWidgets.QLabel("dY (um)")
        self.dY_spinbox = spinbox(1000.0)
        layout.addWidget(self.dy_label, 7, 0)
        layout.addWidget(self.dY_spinbox, 7, 1)
        self.dz_label = QtWidgets.QLabel("dZ (um)")
        self.dZ_spinbox = spinbox(1000.0)
        layout.addWidget(self.dz_label, 8, 0)
        layout.addWidget(self.dZ_spinbox, 8, 1)
        self.dr_label = QtWidgets.QLabel("dR (deg)")
        self.dR_spinbox = spinbox(365.0)
        layout.addWidget(self.dr_label, 9, 0)
        layout.addWidget(self.dR_spinbox, 9, 1)
        self.beam_type_label = QtWidgets.QLabel("Beam Type")
        self.beam_type_combobox = QtWidgets.QComboBox()
        self.beam_type_combobox.addItems(["ION", "ELECTRON"])
        layout.addWidget(self.beam_type_label, 10, 0)
        layout.addWidget(self.beam_type_combobox, 10, 1)
        self.moveRelative_button = QtWidgets.QPushButton("Move ")
        layout.addWidget(self.moveRelative_button, 11, 0, 1, 2)

        # saved positions
        self.addSavedPosition_button = QtWidgets.QPushButton("Save Position")
        self.savedPositionName_lineEdit = QtWidgets.QLineEdit()
        layout.addWidget(self.addSavedPosition_button, 12, 0)
        layout.addWidget(self.savedPositionName_lineEdit, 12, 1)
        self.goToPosition_button = QtWidgets.QPushButton("Go To Position")
        self.savedPosition_combobox = QtWidgets.QComboBox()
        layout.addWidget(self.goToPosition_button, 13, 0)
        layout.addWidget(self.savedPosition_combobox, 13, 1)

        layout.addItem(
            QtWidgets.QSpacerItem(
                20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
            ),
            14,
            0,
            1,
            2,
        )

        order = [
            self.insertManipulator_button,
            self.move_type_comboBox,
            self.dX_spinbox,
            self.dY_spinbox,
            self.dZ_spinbox,
            self.dR_spinbox,
            self.beam_type_combobox,
            self.moveRelative_button,
            self.addSavedPosition_button,
            self.savedPositionName_lineEdit,
            self.goToPosition_button,
            self.savedPosition_combobox,
        ]
        for first, second in zip(order, order[1:]):
            self.setTabOrder(first, second)

    def _is_corrected_move(self) -> bool:
        return (
            "corrected" in self.move_types
            and self.move_type_comboBox.currentText() == "Corrected Move"
        )

    def change_move_type(self):
        self.dZ_spinbox.setEnabled(not self._is_corrected_move())
        self._hide_show_buttons(self._controls_shown)

    def update_ui_state(self):

        is_inserted = self.microscope.get_manipulator_state()
        self._hide_show_buttons(show=is_inserted)
        self.manipulatorStatus_label.setText(
            "Manipulator Status: Inserted"
            if is_inserted
            else "Manipulator Status: Retracted"
        )
        self.insertManipulator_button.setText(
            "Insert" if not is_inserted else "Retract"
        )

    def update_ui(self):

        self.dX_spinbox.setValue(0)
        self.dY_spinbox.setValue(0)
        self.dZ_spinbox.setValue(0)
        self.dR_spinbox.setValue(0)

    def setup_connections(self):

        self.insertManipulator_button.clicked.connect(self.insert_retract_manipulator)
        self.insertManipulator_button.setStyleSheet(
            stylesheets.CONFIRM_BUTTON_STYLESHEET
        )
        self.addSavedPosition_button.clicked.connect(self.add_saved_position)
        self.addSavedPosition_button.setStyleSheet(
            stylesheets.CONFIRM_BUTTON_STYLESHEET
        )
        self.goToPosition_button.clicked.connect(self.move_to_saved_position)
        self.goToPosition_button.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.moveRelative_button.clicked.connect(self.move_relative)
        self.moveRelative_button.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)

        self.pushButton_refresh_data.clicked.connect(self.refresh_data)
        self.pushButton_refresh_data.setStyleSheet(
            stylesheets.SECONDARY_BUTTON_STYLESHEET
        )

    def refresh_data(self):
        self.manipulator_inserted = self.microscope.get_manipulator_state()

        self._hide_show_buttons(self.manipulator_inserted)
        self.insertManipulator_button.setText(
            "Retract" if self.manipulator_inserted else "Insert"
        )
        self.insertManipulator_button.setStyleSheet(
            stylesheets.DANGER_BUTTON_STYLESHEET
            if self.manipulator_inserted
            else stylesheets.CONFIRM_BUTTON_STYLESHEET
        )
        self.manipulatorStatus_label.setText(
            "Manipulator Status: Inserted"
            if self.manipulator_inserted
            else "Manipulator Status: Retracted"
        )

    def move_relative(self):
        """Move the manipulator relative to its current position."""

        dx = self.dX_spinbox.value() * constants.MICRO_TO_SI
        dy = self.dY_spinbox.value() * constants.MICRO_TO_SI
        dz = self.dZ_spinbox.value() * constants.MICRO_TO_SI
        dr = self.dR_spinbox.value() * constants.DEGREES_TO_RADIANS
        beam_type = getattr(BeamType, self.beam_type_combobox.currentText())

        if not self._is_corrected_move():
            try:
                position = FibsemManipulatorPosition(
                    x=dx, y=dy, z=dz, r=dr, coordinate_system="STAGE"
                )  # TODO migrate to raw manipulator movements
                self.microscope.move_manipulator_relative(position=position)
            except Exception as e:
                error_message = f"Error moving manipulator (Relative): {str(e)}"
                logging.error(error_message)
                notification_service.show_toast(error_message, "error")

        else:
            try:
                self.microscope.move_manipulator_corrected(
                    dx=dx, dy=dy, beam_type=beam_type
                )
            except Exception as e:
                error_message = f"Error moving manipulator (Corrected): {str(e)}"
                logging.error(error_message)
                notification_service.show_toast(error_message, "error")

        self.update_ui()

    def _hide_show_buttons(self, show: bool = True):
        self._controls_shown = show
        corrected = self._is_corrected_move()

        # the move-type box only when there is a choice, the beam only for a
        # corrected move, dZ and dR only for a relative one
        self.move_type_comboBox.setVisible(show and len(self.move_types) > 1)
        self.dX_spinbox.setVisible(show)
        self.dY_spinbox.setVisible(show)
        self.dZ_spinbox.setVisible(show and not corrected)
        self.dR_spinbox.setVisible(show and self.has_rotation and not corrected)
        self.dz_label.setVisible(show and not corrected)
        self.dx_label.setVisible(show)
        self.dy_label.setVisible(show)
        self.dr_label.setVisible(show and self.has_rotation and not corrected)
        self.beam_type_combobox.setVisible(show and corrected)
        self.beam_type_label.setVisible(show and corrected)
        self.moveRelative_button.setVisible(show)
        self.addSavedPosition_button.setVisible(show)
        self.goToPosition_button.setVisible(show)
        self.savedPositionName_lineEdit.setVisible(show)
        self.savedPosition_combobox.setVisible(show)

    def insert_retract_manipulator(self):

        if self.microscope.get_manipulator_state():
            self.microscope.retract_manipulator()
            self.insertManipulator_button.setText("Insert")
            self.manipulatorStatus_label.setText("Manipulator Status: Retracted")
            self.insertManipulator_button.setStyleSheet(
                stylesheets.CONFIRM_BUTTON_STYLESHEET
            )
            self.update_ui()
            self._hide_show_buttons(show=False)

        else:
            self.microscope.insert_manipulator()
            self.insertManipulator_button.setText("Retract")
            self.manipulatorStatus_label.setText("Manipulator Status: Inserted")
            self.insertManipulator_button.setStyleSheet(
                stylesheets.DANGER_BUTTON_STYLESHEET
            )
            self.update_ui()
            self._hide_show_buttons(show=True)

    def add_saved_position(self):

        if self.savedPositionName_lineEdit.text() == "":
            _ = message_box_ui(
                title="No name.",
                text="Please enter a position name.",
                buttons=QMessageBox.Ok,
            )
            return
        name = self.savedPositionName_lineEdit.text()
        position = self.microscope.get_manipulator_position()
        self.saved_positions[name] = position
        logging.info(f"Saved position {name} at {position}")
        self.savedPosition_combobox.addItem(name)
        self.savedPositionName_lineEdit.clear()

    def move_to_saved_position(self):
        name = self.savedPosition_combobox.currentText()

        if name in self.named_positions:
            # the instrument's own position: the driver knows how to get there
            try:
                position = self.microscope.move_manipulator_to_named_position(name)
            except Exception as e:
                error_message = f"Error moving manipulator to {name}: {e}"
                logging.error(error_message)
                notification_service.show_toast(error_message, "error")
                return
            logging.info(f"Moved to named position {name} at {position}")
            self.update_ui()
            return

        position = self.saved_positions[name]
        logging.info(f"Moving to saved position {name} at {position}")
        self.microscope.move_manipulator_absolute(position=position)
        self.update_ui()
