
import logging
from typing import Optional

import napari
import numpy as np
from PyQt5 import QtCore, QtWidgets
from superqt import ensure_main_thread

from fibsem import config as cfg
from fibsem.ui import notification_service
from fibsem import constants, conversions
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import (
    BeamType,
    FibsemStagePosition,
    Point,
)
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.napari.utilities import update_text_overlay
from fibsem.ui.qt.threading import thread_worker
from fibsem.ui.stylesheets import (
    DISABLED_PUSHBUTTON_STYLE,
    LABEL_INSTRUCTIONS_STYLE,
    PRIMARY_BUTTON_STYLESHEET,
    SECONDARY_BUTTON_STYLESHEET,
)
from fibsem.ui.utils import WheelBlocker
from fibsem.ui.widgets.custom_widgets import IconToolButton, TitledPanel

INSTRUCTIONS_TEXT = """Instructions: Double Click to Move. Alt + Double Click to Move Vertically"""


class FibsemMovementWidget(QtWidgets.QWidget):
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

        self.pushButton_move_to_sem_orientation = QtWidgets.QPushButton("Move Flat to ELECTRON Beam")
        self.pushButton_move_to_fib_orientation = QtWidgets.QPushButton("Move Flat to ION Beam")
        self.gridLayout_3.addWidget(self.pushButton_move_to_sem_orientation, 6, 0)
        self.gridLayout_3.addWidget(self.pushButton_move_to_fib_orientation, 6, 1)

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
        from fibsem.ui.widgets.saved_position_widget import SavedPositionListWidget
        self.saved_positions_widget = SavedPositionListWidget(microscope=None)
        self.saved_positions_panel = TitledPanel("Saved Positions", content=self.saved_positions_widget, collapsible=True)
        self.gridLayout_2.addWidget(self.saved_positions_panel, 2, 0)

        self._move_buttons = [
            self.pushButton_move,
            self.pushButton_move_to_fib_orientation,
            self.pushButton_move_to_sem_orientation,
            self.pushButton_move_to_milling_angle,
        ]

        # Bottom spacer (row 4 — row 3 reserved for optional sample holder widget)
        self.gridLayout_2.addItem(
            QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding),
            4, 0,
        )

    def setup_connections(self):


        # buttons
        self.pushButton_move.clicked.connect(lambda: self.move_to_position(None))
        self.pushButton_move_to_fib_orientation.clicked.connect(lambda: self.move_to_orientation("FIB"))
        self.pushButton_move_to_sem_orientation.clicked.connect(lambda: self.move_to_orientation("SEM"))
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
        self.saved_positions_widget.microscope = self.microscope
        self.saved_positions_widget._header.btn_add.setEnabled(True)
        self.saved_positions_widget._load_default_positions()
        self.saved_positions_widget.move_to_requested.connect(self.move_to_position)

        # signals
        self.movement_progress_signal.connect(self.handle_movement_progress_update)
        self.image_widget.acquisition_progress_signal.connect(self.handle_acquisition_update)

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
        self.pushButton_move_to_fib_orientation.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        self.pushButton_move_to_sem_orientation.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        self.pushButton_move_to_milling_angle.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)

        # display orientation values on tooltips
        self.pushButton_move_to_sem_orientation.setText("Move to SEM Orientation")
        self.pushButton_move_to_fib_orientation.setText("Move to FIB Orientation")
        sem = self.microscope.get_orientation("SEM")
        fib = self.microscope.get_orientation("FIB")
        milling = self.microscope.get_orientation("MILLING")
        self.pushButton_move_to_sem_orientation.setToolTip(sem.pretty_orientation)
        self.pushButton_move_to_fib_orientation.setToolTip(fib.pretty_orientation)
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

        if cfg.FEATURE_SAMPLE_HOLDER_WIDGET_ENABLED:
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
            notification_service.show_toast(msg)

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

        # update the current position label
        update_text_overlay(self.viewer, self.microscope, stage_position=stage_position)

    @ensure_main_thread
    def update_ui_after_movement(self, retake: bool = True):
        # disable taking images after movement here
        if (retake is False or self.microscope.is_acquiring or
            self.microscope.fm is not None and
            self.microscope.fm.objective.state == "Inserted"):
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
            notification_service.show_toast("Cannot move stage while milling is in progress.")
            return

        # get coords
        coords = layer.world_to_data(event.position)

        # TODO: dimensions are mixed which makes this confusing to interpret... resolve
        coords, beam_type, image = self.image_widget.get_data_from_coord(coords)
        self.movement_progress_signal.emit({"msg": "Click to move in progress..."})

        if beam_type is None:
            notification_service.show_toast(
                "Clicked outside image dimensions. Please click inside the image to move."
            )
            return
        if image.metadata is None:
            notification_service.show_toast(
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
        self.movement_progress_signal.emit({"msg": f"Moving to {orientation} orientation..."})
        self.microscope.move_to_orientation(orientation)
        self.update_ui_after_movement()

