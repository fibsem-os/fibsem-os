
import logging
import os
from copy import deepcopy
from typing import List, Optional

import napari
import napari.utils.notifications
import numpy as np
import yaml
from napari.qt.threading import thread_worker
from PyQt5 import QtCore, QtWidgets
from superqt import ensure_main_thread

from fibsem import config as cfg
from fibsem import constants, conversions, utils
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import (
    BeamType,
    FibsemStagePosition,
    Point,
)
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.napari.utilities import update_text_overlay
from fibsem.ui.stylesheets import (
    BLUE_PUSHBUTTON_STYLE,
    DISABLED_PUSHBUTTON_STYLE,
    GRAY_PUSHBUTTON_STYLE,
    GREEN_PUSHBUTTON_STYLE,
    LABEL_INSTRUCTIONS_STYLE,
    ORANGE_PUSHBUTTON_STYLE,
    PRIMARY_BUTTON_STYLESHEET,
    RED_PUSHBUTTON_STYLE,
    SECONDARY_BUTTON_STYLESHEET,
)
from fibsem.ui.utils import (
    WheelBlocker,
    message_box_ui,
    open_existing_file_dialog,
    open_save_file_dialog,
)
from fibsem.ui.widgets.custom_widgets import IconToolButton, TitledPanel

INSTRUCTIONS_TEXT = """Instructions: Double Click to Move. Alt + Double Click to Move Vertically"""

SHOW_SAMPLE_HOLDER_WIDGET = True

class FibsemMovementWidget(QtWidgets.QWidget):
    saved_positions_updated_signal = QtCore.pyqtSignal(object)
    movement_progress_signal = QtCore.pyqtSignal(dict)

    def __init__(
        self,
        microscope: FibsemMicroscope,
        parent: QtWidgets.QWidget,
    ):
        super().__init__(parent=parent)
        self._setup_ui()
        self.parent = parent

        if not hasattr(parent, 'image_widget') or not isinstance(parent.image_widget, FibsemImageSettingsWidget):
            raise ValueError("Parent must have an 'image_widget' attribute of type FibsemImageSettingsWidget")
        if not hasattr(parent, "viewer") or not isinstance(parent.viewer, napari.Viewer):
            raise ValueError("Parent must have a 'viewer' attribute of type napari.Viewer")

        self.microscope = microscope
        self.viewer = parent.viewer
        self.image_widget: FibsemImageSettingsWidget = parent.image_widget
        self.positions: List[FibsemStagePosition] = []

        self.import_positions(cfg.POSITION_PATH)
        self.setup_connections()

    def _setup_ui(self):
        # Outer layout
        self.gridLayout = QtWidgets.QGridLayout(self)

        # Scroll area
        self.scrollArea = QtWidgets.QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.gridLayout_2 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.gridLayout.addWidget(self.scrollArea, 0, 0, 1, 2)

        # --- Panel: Stage Movement ---
        stage_content = QtWidgets.QWidget()
        self.gridLayout_3 = QtWidgets.QGridLayout(stage_content)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)

        self.label_movement_stage_x = QtWidgets.QLabel("X Coordinate")
        self.doubleSpinBox_movement_stage_x = QtWidgets.QDoubleSpinBox()
        self.doubleSpinBox_movement_stage_x.setDecimals(5)
        self.doubleSpinBox_movement_stage_x.setMinimum(-1e10)
        self.doubleSpinBox_movement_stage_x.setMaximum(1e17)
        self.doubleSpinBox_movement_stage_x.setSingleStep(0.001)
        self.doubleSpinBox_movement_stage_x.setSuffix(" mm")
        self.gridLayout_3.addWidget(self.label_movement_stage_x, 0, 0)
        self.gridLayout_3.addWidget(self.doubleSpinBox_movement_stage_x, 0, 1)

        self.label_movement_stage_y = QtWidgets.QLabel("Y Coordinate")
        self.doubleSpinBox_movement_stage_y = QtWidgets.QDoubleSpinBox()
        self.doubleSpinBox_movement_stage_y.setDecimals(5)
        self.doubleSpinBox_movement_stage_y.setMinimum(-1e20)
        self.doubleSpinBox_movement_stage_y.setMaximum(1e25)
        self.doubleSpinBox_movement_stage_y.setSingleStep(0.001)
        self.doubleSpinBox_movement_stage_y.setSuffix(" mm")
        self.gridLayout_3.addWidget(self.label_movement_stage_y, 1, 0)
        self.gridLayout_3.addWidget(self.doubleSpinBox_movement_stage_y, 1, 1)

        self.label_movement_stage_z = QtWidgets.QLabel("Z Coordinate")
        self.doubleSpinBox_movement_stage_z = QtWidgets.QDoubleSpinBox()
        self.doubleSpinBox_movement_stage_z.setDecimals(5)
        self.doubleSpinBox_movement_stage_z.setMinimum(-1e17)
        self.doubleSpinBox_movement_stage_z.setMaximum(1e23)
        self.doubleSpinBox_movement_stage_z.setSingleStep(0.001)
        self.doubleSpinBox_movement_stage_z.setSuffix(" mm")
        self.gridLayout_3.addWidget(self.label_movement_stage_z, 2, 0)
        self.gridLayout_3.addWidget(self.doubleSpinBox_movement_stage_z, 2, 1)

        self.label_movement_stage_rotation = QtWidgets.QLabel("Rotation")
        self.doubleSpinBox_movement_stage_rotation = QtWidgets.QDoubleSpinBox()
        self.doubleSpinBox_movement_stage_rotation.setMinimum(-360.0)
        self.doubleSpinBox_movement_stage_rotation.setMaximum(360.0)
        self.doubleSpinBox_movement_stage_rotation.setSuffix(f" {constants.DEGREE_SYMBOL}")
        self.gridLayout_3.addWidget(self.label_movement_stage_rotation, 3, 0)
        self.gridLayout_3.addWidget(self.doubleSpinBox_movement_stage_rotation, 3, 1)

        self.label_movement_stage_tilt = QtWidgets.QLabel("Tilt")
        self.doubleSpinBox_movement_stage_tilt = QtWidgets.QDoubleSpinBox()
        self.doubleSpinBox_movement_stage_tilt.setSuffix(f" {constants.DEGREE_SYMBOL}")
        self.gridLayout_3.addWidget(self.label_movement_stage_tilt, 4, 0)
        self.gridLayout_3.addWidget(self.doubleSpinBox_movement_stage_tilt, 4, 1)

        self.pushButton_move = QtWidgets.QPushButton("Move to Position")
        self.gridLayout_3.addWidget(self.pushButton_move, 5, 0, 1, 2)

        self.pushButton_move_flat_electron = QtWidgets.QPushButton("Move Flat to ELECTRON Beam")
        self.pushButton_move_flat_ion = QtWidgets.QPushButton("Move Flat to ION Beam")
        self.gridLayout_3.addWidget(self.pushButton_move_flat_electron, 6, 0)
        self.gridLayout_3.addWidget(self.pushButton_move_flat_ion, 6, 1)

        self.doubleSpinBox_milling_angle = QtWidgets.QDoubleSpinBox()
        self.pushButton_move_to_milling_angle = QtWidgets.QPushButton("Move to Milling Angle")
        self.gridLayout_3.addWidget(self.doubleSpinBox_milling_angle, 7, 0)
        self.gridLayout_3.addWidget(self.pushButton_move_to_milling_angle, 7, 1)

        self.label_movement_instructions = QtWidgets.QLabel()
        self.label_movement_instructions.setWordWrap(True)
        self.gridLayout_3.addWidget(self.label_movement_instructions, 8, 0, 1, 2)

        self.btn_refresh_stage = IconToolButton(icon="mdi:refresh", tooltip="Refresh stage position")
        self.stage_panel = TitledPanel("Stage Movement", content=stage_content, collapsible=False)
        self.stage_panel.add_header_widget(self.btn_refresh_stage)
        self.gridLayout_2.addWidget(self.stage_panel, 0, 0)

        # Options panel removed — movement acquisition prefs are now in Edit > Preferences

        # --- Panel: Saved Positions ---
        saved_content = QtWidgets.QWidget()
        self.gridLayout_5 = QtWidgets.QGridLayout(saved_content)
        self.gridLayout_5.setContentsMargins(0, 0, 0, 0)

        self.label_positions_header_info = QtWidgets.QLabel("All positions in mm and degrees")
        self.gridLayout_5.addWidget(self.label_positions_header_info, 0, 0, 1, 2)

        self.pushButton_save_position = QtWidgets.QPushButton("Add Position")
        self.pushButton_remove_position = QtWidgets.QPushButton("Remove Position")
        self.gridLayout_5.addWidget(self.pushButton_save_position, 1, 0)
        self.gridLayout_5.addWidget(self.pushButton_remove_position, 1, 1)

        self.label_saved_positions = QtWidgets.QLabel("Saved Positions")
        self.comboBox_positions = QtWidgets.QComboBox()
        self.gridLayout_5.addWidget(self.label_saved_positions, 2, 0)
        self.gridLayout_5.addWidget(self.comboBox_positions, 2, 1)

        self.pushButton_go_to = QtWidgets.QPushButton("Go To Position")
        self.label_current_position = QtWidgets.QLabel("")
        self.gridLayout_5.addWidget(self.pushButton_go_to, 3, 0)
        self.gridLayout_5.addWidget(self.label_current_position, 3, 1)

        self.pushButton_update_position = QtWidgets.QPushButton("Update Position")
        self.lineEdit_position_name = QtWidgets.QLineEdit()
        self.lineEdit_position_name.setSizePolicy(
            QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        )
        self.gridLayout_5.addWidget(self.pushButton_update_position, 4, 0)
        self.gridLayout_5.addWidget(self.lineEdit_position_name, 4, 1)

        self.pushButton_import = QtWidgets.QPushButton("Import Positions")
        self.pushButton_export = QtWidgets.QPushButton("Export Positions")
        self.gridLayout_5.addWidget(self.pushButton_import, 5, 0)
        self.gridLayout_5.addWidget(self.pushButton_export, 5, 1)

        self.saved_positions_panel = TitledPanel("Saved Positions", content=saved_content, collapsible=False)
        self.gridLayout_2.addWidget(self.saved_positions_panel, 2, 0)
        self.saved_positions_panel.setVisible(False)

        self._move_buttons = [
            self.pushButton_move,
            self.pushButton_move_flat_ion,
            self.pushButton_move_flat_electron,
            self.pushButton_move_to_milling_angle,
            self.pushButton_go_to,
        ]

        # Bottom spacer (row 4 — row 3 reserved for optional sample holder widget)
        self.gridLayout_2.addItem(
            QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding),
            4, 0,
        )

    def setup_connections(self):


        # buttons
        self.pushButton_move.clicked.connect(lambda: self.move_to_position(None))
        self.pushButton_move_flat_ion.clicked.connect(self.move_flat_to_beam)
        self.pushButton_move_flat_electron.clicked.connect(self.move_flat_to_beam)
        self.btn_refresh_stage.clicked.connect(lambda: self.update_ui(None))

        # register mouse callbacks
        if cfg.FEATURE_VIEWER_MOVEMENT_EVENTS:
            self.viewer.mouse_double_click_callbacks.append(self._viewer_double_click)
        else:
            self.image_widget.eb_layer.mouse_double_click_callbacks.append(self._double_click)
            self.image_widget.ib_layer.mouse_double_click_callbacks.append(self._double_click)

        # disable ui elements
        self.label_movement_instructions.setText(INSTRUCTIONS_TEXT)
        self.label_movement_instructions.setStyleSheet(LABEL_INSTRUCTIONS_STYLE)

        # saved positions
        self.comboBox_positions.currentIndexChanged.connect(self.current_selected_position_changed)
        self.pushButton_save_position.clicked.connect(self.add_position)
        self.pushButton_remove_position.clicked.connect(self.delete_position)
        self.pushButton_go_to.clicked.connect(self.move_to_selected_saved_position)
        self.pushButton_update_position.clicked.connect(self.update_saved_position)
        self.pushButton_export.clicked.connect(self.export_positions)
        self.pushButton_import.clicked.connect(self.import_positions)

        # signals
        self.movement_progress_signal.connect(self.handle_movement_progress_update)
        self.image_widget.acquisition_progress_signal.connect(self.handle_acquisition_update)
        self.saved_positions_updated_signal.connect(self.update_saved_positions_ui)

        stage_limits = self.microscope._stage.limits
        xlimits = stage_limits['x']
        ylimits = stage_limits['y']
        zlimits = stage_limits['z']
        tlimits = stage_limits['t']

        self.doubleSpinBox_movement_stage_tilt.setMinimum(tlimits.min)
        self.doubleSpinBox_movement_stage_tilt.setMaximum(tlimits.max)
        self.doubleSpinBox_movement_stage_x.setMinimum(xlimits.min * constants.SI_TO_MILLI)
        self.doubleSpinBox_movement_stage_x.setMaximum(xlimits.max * constants.SI_TO_MILLI)
        self.doubleSpinBox_movement_stage_y.setMinimum(ylimits.min * constants.SI_TO_MILLI)
        self.doubleSpinBox_movement_stage_y.setMaximum(ylimits.max * constants.SI_TO_MILLI)
        self.doubleSpinBox_movement_stage_z.setMinimum(zlimits.min * constants.SI_TO_MILLI)
        self.doubleSpinBox_movement_stage_z.setMaximum(zlimits.max * constants.SI_TO_MILLI)

        # set custom tilt limits for the compustage
        if self.microscope.stage_is_compustage:

            # NOTE: these values are expressed in mm in the UI, hence the conversion
            # set x, y, z step sizes to be 1 um
            self.doubleSpinBox_movement_stage_x.setSingleStep(1e-6 * constants.SI_TO_MILLI)
            self.doubleSpinBox_movement_stage_y.setSingleStep(1e-6 * constants.SI_TO_MILLI)
            self.doubleSpinBox_movement_stage_z.setSingleStep(1e-6 * constants.SI_TO_MILLI)

            # hide rotation control for compustage
            self.label_movement_stage_rotation.setVisible(False)
            self.doubleSpinBox_movement_stage_rotation.setVisible(False)

        # stylesheets
        self.pushButton_move.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        self.pushButton_move_flat_ion.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        self.pushButton_move_flat_electron.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        self.pushButton_move_to_milling_angle.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        self.pushButton_save_position.setStyleSheet(GREEN_PUSHBUTTON_STYLE)
        self.pushButton_remove_position.setStyleSheet(RED_PUSHBUTTON_STYLE)
        self.pushButton_go_to.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        self.pushButton_export.setStyleSheet(GRAY_PUSHBUTTON_STYLE)
        self.pushButton_import.setStyleSheet(GRAY_PUSHBUTTON_STYLE)
        self.pushButton_update_position.setStyleSheet(ORANGE_PUSHBUTTON_STYLE)

        # display orientation values on tooltips
        self.pushButton_move_flat_electron.setText("Move to SEM Orientation")
        self.pushButton_move_flat_ion.setText("Move to FIB Orientation")
        sem = self.microscope.get_orientation("SEM")
        fib = self.microscope.get_orientation("FIB")
        milling = self.microscope.get_orientation("MILLING")
        self.pushButton_move_flat_electron.setToolTip(sem.pretty_orientation)
        self.pushButton_move_flat_ion.setToolTip(fib.pretty_orientation)
        self.pushButton_move_to_milling_angle.setToolTip(milling.pretty_orientation)

        # milling angle controls
        self.doubleSpinBox_milling_angle.setValue(self.microscope.system.stage.milling_angle) # deg
        self.doubleSpinBox_milling_angle.setSuffix(constants.DEGREE_SYMBOL)
        self.doubleSpinBox_milling_angle.setSingleStep(1.0)
        self.doubleSpinBox_milling_angle.setDecimals(1)
        self.doubleSpinBox_milling_angle.setRange(0, 45)
        self.doubleSpinBox_milling_angle.setToolTip("The milling angle is the difference between the stage and the fib viewing angle.")
        self.doubleSpinBox_milling_angle.setKeyboardTracking(False)
        self.doubleSpinBox_milling_angle.valueChanged.connect(self._update_milling_angle)
        self.pushButton_move_to_milling_angle.clicked.connect(lambda: self.move_to_orientation("MILLING"))

        # set degree symbols for rotation and tilt
        self.doubleSpinBox_movement_stage_rotation.setSuffix(constants.DEGREE_SYMBOL)
        self.doubleSpinBox_movement_stage_tilt.setSuffix(constants.DEGREE_SYMBOL)

        # Install wheel blocker on all double spin boxes
        self.wheel_blocker = WheelBlocker()
        self.doubleSpinBox_movement_stage_x.installEventFilter(self.wheel_blocker)
        self.doubleSpinBox_movement_stage_y.installEventFilter(self.wheel_blocker)
        self.doubleSpinBox_movement_stage_z.installEventFilter(self.wheel_blocker)
        self.doubleSpinBox_movement_stage_rotation.installEventFilter(self.wheel_blocker)
        self.doubleSpinBox_movement_stage_tilt.installEventFilter(self.wheel_blocker)
        self.doubleSpinBox_milling_angle.installEventFilter(self.wheel_blocker)

        if SHOW_SAMPLE_HOLDER_WIDGET:
            from fibsem.ui.widgets.sample_holder_widget import SampleHolderWidget
            self.sample_holder_widget = SampleHolderWidget(microscope=self.microscope)
            self.sample_holder_widget.set_holder(self.microscope._stage.holder)
            self.gridLayout_2.addWidget(self.sample_holder_widget, 3, 0)

        self.update_ui()

    def _toggle_interactions(self, enable: bool, caller: Optional[str] = None):
        """Toggle the interactions in the widget depending on microscope state"""
        for btn in self._move_buttons:
            btn.setEnabled(enable)
        self.doubleSpinBox_milling_angle.setEnabled(enable)
        if caller is None:
            # self.parent.milling_widget._toggle_interactions(enable, caller="movement")
            self.parent.image_widget._toggle_interactions(enable, caller="movement")
        for btn in self._move_buttons:
            btn.setStyleSheet(DISABLED_PUSHBUTTON_STYLE if not enable else
                              PRIMARY_BUTTON_STYLESHEET if btn is self.pushButton_move else
                              SECONDARY_BUTTON_STYLESHEET)

    def handle_movement_progress_update(self, ddict: dict) -> None:
        """Handle movement progress updates from the microscope"""

        msg = ddict.get("msg", None)
        if msg is not None:
            logging.debug(msg)
            napari.utils.notifications.notification_manager.records.clear()
            napari.utils.notifications.show_info(msg)

        is_finished = ddict.get("finished", False)
        if is_finished:
            update_text_overlay(self.viewer, self.microscope)

    def handle_acquisition_update(self, ddict: dict):
        """Handle acquisition updates from the image widget"""
        is_finished = ddict.get("finished", False)
        if is_finished:
            self.update_ui()

    @ensure_main_thread
    def update_ui(self, stage_position: Optional[FibsemStagePosition] = None):
        """Update the UI with the current stage position and saved positions"""
        if stage_position is None:
            stage_position = self.microscope.get_stage_position()

        self.doubleSpinBox_movement_stage_x.setValue(stage_position.x * constants.SI_TO_MILLI)
        self.doubleSpinBox_movement_stage_y.setValue(stage_position.y * constants.SI_TO_MILLI)
        self.doubleSpinBox_movement_stage_z.setValue(stage_position.z * constants.SI_TO_MILLI)
        self.doubleSpinBox_movement_stage_rotation.setValue(np.degrees(stage_position.r))
        self.doubleSpinBox_movement_stage_tilt.setValue(np.degrees(stage_position.t))

        has_saved_positions = bool(len(self.positions))
        self.pushButton_remove_position.setEnabled(has_saved_positions)
        self.pushButton_go_to.setVisible(has_saved_positions)
        self.pushButton_update_position.setVisible(has_saved_positions)
        self.pushButton_export.setVisible(has_saved_positions)
        self.comboBox_positions.setVisible(has_saved_positions)
        self.pushButton_import.setVisible(has_saved_positions)
        self.label_saved_positions.setVisible(has_saved_positions)
        self.label_current_position.setVisible(has_saved_positions)
        self.lineEdit_position_name.setVisible(has_saved_positions)

        # update the current position label
        update_text_overlay(self.viewer, self.microscope, stage_position=stage_position)

    @ensure_main_thread
    def update_ui_after_movement(self, retake: bool = True):
        if (retake is False or self.microscope.is_acquiring):
            self.update_ui()
            return
        prefs = cfg.load_user_preferences()
        acquire_sem = prefs.movement.acquire_sem_after_stage_movement
        acquire_fib = prefs.movement.acquire_fib_after_stage_movement
        if acquire_sem and acquire_fib:
            self.image_widget.acquire_reference_images()
            return
        if acquire_sem:
            self.image_widget.acquire_sem_image()
        elif acquire_fib:
            self.image_widget.acquire_fib_image()
        else:
            self.update_ui()

    def _update_milling_angle(self):
        """Update the milling angle in the microscope and the UI"""
        milling_angle = self.doubleSpinBox_milling_angle.value() # deg
        self.microscope.set_milling_angle(milling_angle)

        # refresh tooltip and overlay
        milling = self.microscope.get_orientation("MILLING")
        self.pushButton_move_to_milling_angle.setToolTip(milling.pretty_orientation)
        update_text_overlay(self.viewer, self.microscope)

#### MOVEMENT

    def move_to_position(self, stage_position: Optional[FibsemStagePosition] = None):
        """Move the stage to the position specified in the UI"""
        if stage_position is None:
            stage_position = self.get_position_from_ui()
        self._move_to_absolute_position(stage_position)

    def _move_to_absolute_position(self, stage_position: FibsemStagePosition):
        """Move the stage to the specified position"""
        self._toggle_interactions(enable=False)
        worker = self.absolute_movement_worker(stage_position=stage_position)
        worker.finished.connect(self.move_stage_finished)
        worker.start()
    
    @thread_worker
    def absolute_movement_worker(self, stage_position: FibsemStagePosition) -> None:
        """Worker function to move the stage to the specified position"""
        self.movement_progress_signal.emit({"msg": f"Moving to {stage_position}"})
        self.microscope.safe_absolute_stage_movement(stage_position)
        self.movement_progress_signal.emit({"msg": "Move finished, taking new images"})
        self.update_ui_after_movement()

    def move_stage_finished(self):
        """Handle the completion of a stage movement"""
        self.movement_progress_signal.emit({"finished": True})
        if self.image_widget.is_acquiring:
            return
        self._toggle_interactions(enable=True)

    def get_position_from_ui(self):
        """Get the stage position from the UI"""

        stage_position = FibsemStagePosition(
            x=self.doubleSpinBox_movement_stage_x.value() * constants.MILLI_TO_SI,
            y=self.doubleSpinBox_movement_stage_y.value() * constants.MILLI_TO_SI,
            z=self.doubleSpinBox_movement_stage_z.value() * constants.MILLI_TO_SI,
            r=np.radians(self.doubleSpinBox_movement_stage_rotation.value()),
            t=np.radians(self.doubleSpinBox_movement_stage_tilt.value()),
            coordinate_system="RAW",
        )

        return stage_position

    def _viewer_double_click(self, viewer, event):
        """Viewer-level double-click callback (FEATURE_VIEWER_MOVEMENT_EVENTS).
        Determines which image layer was clicked and delegates to _double_click."""
        for layer in [self.image_widget.eb_layer, self.image_widget.ib_layer]:
            coords = layer.world_to_data(event.position)
            _, beam_type, _ = self.image_widget.get_data_from_coord(coords)
            if beam_type is not None:
                self._double_click(layer, event)
                return

    def _double_click(self, layer, event):
        """Callback for double-click mouse events on the image widget"""
        self._toggle_interactions(enable= False)

        worker = self._double_click_worker(layer, event)
        worker.finished.connect(self.move_stage_finished)
        worker.start()

    @thread_worker
    def _double_click_worker(self, layer, event):
        """Thread worker for double-click mouse events on the image widget"""
        if event.button != 1 or "Shift" in event.modifiers:
            return

        if hasattr(self.parent, "milling_widget") and self.parent.milling_widget.is_milling:
            napari.utils.notifications.show_info("Cannot move stage while milling is in progress.")
            return

        # get coords
        coords = layer.world_to_data(event.position)

        # TODO: dimensions are mixed which makes this confusing to interpret... resolve
        coords, beam_type, image = self.image_widget.get_data_from_coord(coords)
        self.movement_progress_signal.emit({"msg": "Click to move in progress..."})

        if beam_type is None:
            napari.utils.notifications.show_info(
                "Clicked outside image dimensions. Please click inside the image to move."
            )
            return
        if image.metadata is None:
            napari.utils.notifications.show_info(
                "Image metadata is not set. Please set the image metadata before moving."
            )
            return

        point = conversions.image_to_microscope_image_coordinates(
            coord=Point(x=coords[1], y=coords[0]), 
            image=image.data, 
            pixelsize=image.metadata.pixel_size.x,
        )

        # move
        vertical_move = True if "Alt" in event.modifiers else False
        movement_mode = "Vertical" if vertical_move else "Stable"

        logging.debug({
            "msg": "stage_movement",                    # message type
            "movement_mode": movement_mode,             # movement mode
            "beam_type": beam_type.name,                # beam type
            "dm": point.to_dict(),                      # shift in microscope coordinates
            "coords": {"x": coords[1], "y": coords[0]}, # coords in image coordinates
        })

        self.movement_progress_signal.emit({"msg": "Moving stage..."})
        # eucentric is only supported for ION beam
        if beam_type is BeamType.ION and vertical_move:
            self.microscope.vertical_move(dx=point.x, dy=point.y)
        elif beam_type is BeamType.ELECTRON and vertical_move and hasattr(self.microscope, "move_coincident_from_sem"):
            # move coincident from SEM
            self.microscope.move_coincident_from_sem(dx=0, dy=point.y) # TMP: disable dx for now
        else:
            # corrected stage movement
            self.microscope.stable_move(
                dx=point.x,
                dy=point.y,
                beam_type=beam_type,
            )
        self.movement_progress_signal.emit({"msg": "Move finished, updating UI"})
        self.update_ui_after_movement()

    def move_flat_to_beam(self)-> None:
        """Move to the specifed beam orientation"""
        beam_type = BeamType.ION if self.sender() == self.pushButton_move_flat_ion else BeamType.ELECTRON
        self._toggle_interactions(False)
        worker = self.move_flat_to_beam_worker(beam_type)
        worker.finished.connect(self.move_stage_finished)
        worker.start()

    @thread_worker
    def move_flat_to_beam_worker(self, beam_type: BeamType) -> None:
        """Threaded worker function to move the stage to the specified beam"""
        self.movement_progress_signal.emit({"msg": f"Moving flat to {beam_type.name} beam"})
        self.microscope.move_flat_to_beam(beam_type=beam_type)
        self.update_ui_after_movement()

    # TODO: migrate to this from move_flat_to_beam
    def move_to_orientation(self, orientation: str)-> None:
        """Move to the specifed orientation"""
        if orientation not in ["SEM", "FIB", "MILLING"]:
            raise ValueError(f"Invalid orientation: {orientation}")
        self._toggle_interactions(False)
        worker = self.move_to_orientation_worker(orientation)
        worker.finished.connect(self.move_stage_finished)
        worker.start()

    @thread_worker
    def move_to_orientation_worker(self, orientation: str) -> None:
        """Threaded worker function to move the stage to the specified orientation"""
        self.movement_progress_signal.emit({"msg": f"Moving flat to {orientation}"})
        stage_orientation = self.microscope.get_orientation(orientation)
        self.microscope.safe_absolute_stage_movement(stage_orientation)
        self.update_ui_after_movement()

#### SAVED POSITIONS

    def current_selected_position_changed(self):
        """Callback for when the selected position is changed"""
        current_index = self.comboBox_positions.currentIndex()
        if current_index == -1:
            return
        position = self.positions[current_index]
        self.label_current_position.setText(position.pretty_string)
        self.lineEdit_position_name.setText(position.name)

    def add_position(self) -> None:
        """Add the current stage position to the saved positions"""

        position = self.microscope.get_stage_position()
        position.name = f"Position {len(self.positions):02d}"
        self.positions.append(deepcopy(position))
        self.saved_positions_updated_signal.emit(self.positions)
        logging.info(f"Added position {position.name}")

    def delete_position(self):
        """Delete the currently selected position"""
        current_index = self.comboBox_positions.currentIndex()
        position = self.positions.pop(current_index)
        self.saved_positions_updated_signal.emit(self.positions)
        logging.info(f"Removed position {position.name}")

    def update_saved_positions_ui(self, positions: List[FibsemStagePosition]) -> None:
        """Update the positions in the widget"""

        self.comboBox_positions.clear()
        for position in positions:
            self.comboBox_positions.addItem(position.name)

        # set index to the last position
        self.comboBox_positions.setCurrentIndex(len(positions) - 1)
        self.lineEdit_position_name.setText(positions[-1].name)
        self.update_ui()

    def update_saved_position(self):
        """Update the currently selected saved position to the current stage position"""
        current_index = self.comboBox_positions.currentIndex()
        if current_index == -1:
            napari.utils.notifications.show_warning("Please select a position to update")
            return

        position: FibsemStagePosition = self.microscope.get_stage_position()
        position.name = self.lineEdit_position_name.text()
        if position.name == "":
            napari.utils.notifications.show_warning("Please enter a name for the position")
            return

        # TODO: add dialog confirmation
        self.positions[current_index] = deepcopy(position)

        logging.info(f"Updated position {position}")
        self.saved_positions_updated_signal.emit(self.positions) # redraw combobox, TODO: need to re-select the current index

    def move_to_selected_saved_position(self) -> None: 
        """Move the stage to the selected saved position"""
        current_index = self.comboBox_positions.currentIndex()
        stage_position = self.positions[current_index]
        self.move_to_position(stage_position)

    def export_positions(self):

        path = open_save_file_dialog(msg="Select or create file", 
            path=cfg.POSITION_PATH, 
            _filter="YAML Files (*.yaml)")

        if path == '':
            napari.utils.notifications.show_info("No file selected, positions not saved")
            return

        response = message_box_ui(text="Do you want to overwrite the file ? Click no to append the new positions to the existing file.", 
            title="Overwrite ?", buttons=QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)

        # save positions
        utils.save_positions(self.positions, path, overwrite=response)

        logging.info(f"Positions saved to {path}")

    def import_positions(self, path: Optional[str] = None):
        """Import saved positions from a file"""
        
        if path is None:
            path = open_existing_file_dialog(msg="Select a positions file", path=cfg.POSITION_PATH, _filter="YAML Files (*.yaml)")
        
        if path == "":
            napari.utils.notifications.show_info("No file selected, positions not loaded")
            return

        def load_saved_positions_from_yaml(path: Optional[str] = None) -> List[FibsemStagePosition]:
            if path is None or not os.path.exists(path):
                return []
            with open(path, "r") as f:
                ddict = yaml.safe_load(f)
            return [FibsemStagePosition.from_dict(pdict) for pdict in ddict]

        self.positions = load_saved_positions_from_yaml(path)
        self.saved_positions_updated_signal.emit(self.positions)
