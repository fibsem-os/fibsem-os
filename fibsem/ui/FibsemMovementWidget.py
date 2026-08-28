import logging
from functools import partial
from typing import Optional

import numpy as np
from PyQt5 import QtCore, QtWidgets
from superqt import ensure_main_thread

from fibsem import config as cfg
from fibsem import constants, conversions
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import (
    BeamType,
    FibsemStagePosition,
    Point,
)
from fibsem.ui import notification_service
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.qt.threading import thread_worker
from fibsem.ui.stylesheets import (
    LABEL_INSTRUCTIONS_STYLE,
    PRIMARY_BUTTON_STYLESHEET,
    SECONDARY_BUTTON_STYLESHEET,
)
from fibsem.ui.utils import install_wheel_blocker
from fibsem.ui.widgets.custom_widgets import IconToolButton, TitledPanel

INSTRUCTIONS_TEXT = (
    """Instructions: Double Click to Move. Alt + Double Click to Move Vertically"""
)

# What every path says once the stage has arrived and the images are being retaken.
# A constant rather than three string literals: the three movement paths had drifted
# to "updating images", "taking new images" and, on the orientation path, nothing at
# all -- so the same phase looked different depending on how the move was started.
ACQUIRING_IMAGES = "Acquiring images…"


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

        if not hasattr(parent, "image_widget") or not isinstance(
            parent.image_widget, FibsemImageSettingsWidget
        ):
            raise ValueError(
                "Parent must have an 'image_widget' attribute of type FibsemImageSettingsWidget"
            )

        self.microscope = microscope
        self.image_widget: FibsemImageSettingsWidget = parent.image_widget
        self.setup_connections()

    def _view_controller(self):
        """Return the quad-view MicroscopeViewController, or None if unavailable.

        Resolved like the image widget: the direct parent (standalone ``FibsemUI``) or
        parent -> ``parent_widget`` (AutoLamella) holds ``view_controller``.
        """
        controller = getattr(self.parent, "view_controller", None)
        if controller is not None:
            return controller
        parent_ui = getattr(self.parent, "parent_widget", None)
        return getattr(parent_ui, "view_controller", None)

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
        self.doubleSpinBox_movement_stage_rotation.setSuffix(
            f" {constants.DEGREE_SYMBOL}"
        )
        self.gridLayout_3.addWidget(self.label_movement_stage_rotation, 3, 0)
        self.gridLayout_3.addWidget(self.doubleSpinBox_movement_stage_rotation, 3, 1)

        self.label_movement_stage_tilt = QtWidgets.QLabel("Tilt")
        self.doubleSpinBox_movement_stage_tilt = QtWidgets.QDoubleSpinBox()
        self.doubleSpinBox_movement_stage_tilt.setSuffix(f" {constants.DEGREE_SYMBOL}")
        self.gridLayout_3.addWidget(self.label_movement_stage_tilt, 4, 0)
        self.gridLayout_3.addWidget(self.doubleSpinBox_movement_stage_tilt, 4, 1)

        self.pushButton_move = QtWidgets.QPushButton("Move to Position")
        self.gridLayout_3.addWidget(self.pushButton_move, 5, 0, 1, 2)

        self.pushButton_move_to_sem_orientation = QtWidgets.QPushButton(
            "Move Flat to ELECTRON Beam"
        )
        self.pushButton_move_to_fib_orientation = QtWidgets.QPushButton(
            "Move Flat to ION Beam"
        )
        self.gridLayout_3.addWidget(self.pushButton_move_to_sem_orientation, 6, 0)
        self.gridLayout_3.addWidget(self.pushButton_move_to_fib_orientation, 6, 1)

        self.doubleSpinBox_milling_angle = QtWidgets.QDoubleSpinBox()
        self.pushButton_move_to_milling_angle = QtWidgets.QPushButton(
            "Move to Milling Angle"
        )
        self.gridLayout_3.addWidget(self.doubleSpinBox_milling_angle, 7, 0)
        self.gridLayout_3.addWidget(self.pushButton_move_to_milling_angle, 7, 1)

        self.label_movement_instructions = QtWidgets.QLabel()
        self.label_movement_instructions.setWordWrap(True)
        self.gridLayout_3.addWidget(self.label_movement_instructions, 8, 0, 1, 2)

        self.btn_refresh_stage = IconToolButton(
            icon="mdi:refresh", tooltip="Refresh stage position"
        )
        self.stage_panel = TitledPanel(
            "Stage Movement", content=stage_content, collapsible=False
        )
        self.stage_panel.add_header_widget(self.btn_refresh_stage)
        self.gridLayout_2.addWidget(self.stage_panel, 0, 0)

        # Options panel removed — movement acquisition prefs are now in Edit > Preferences

        # --- Panel: Saved Positions ---
        from fibsem.ui.widgets.saved_position_widget import SavedPositionListWidget

        self.saved_positions_widget = SavedPositionListWidget(microscope=None)
        self.saved_positions_panel = TitledPanel(
            "Saved Positions", content=self.saved_positions_widget, collapsible=True
        )
        self.gridLayout_2.addWidget(self.saved_positions_panel, 2, 0)

        self._move_buttons = [
            self.pushButton_move,
            self.pushButton_move_to_fib_orientation,
            self.pushButton_move_to_sem_orientation,
            self.pushButton_move_to_milling_angle,
        ]

        # Bottom spacer (row 4 — row 3 reserved for optional sample holder widget)
        self.gridLayout_2.addItem(
            QtWidgets.QSpacerItem(
                20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
            ),
            4,
            0,
        )

    def setup_connections(self):

        # buttons
        self.pushButton_move.clicked.connect(lambda: self.move_to_position(None))
        self.pushButton_move_to_fib_orientation.clicked.connect(
            lambda: self.move_to_orientation("FIB")
        )
        self.pushButton_move_to_sem_orientation.clicked.connect(
            lambda: self.move_to_orientation("SEM")
        )
        self.btn_refresh_stage.clicked.connect(lambda: self.update_ui(None))

        # register mouse callbacks — one canvas per beam. The canvases are app-lifetime
        # (owned by the controller), so store each (canvas, slot) pair and disconnect it in
        # _teardown_connections: this widget is torn down via removeTab + deleteLater
        # (which fires neither closeEvent nor close), and a stale double-click firing on the
        # deleted widget makes PyQt call qFatal -> the process aborts (FIB-329).
        # partial (not lambda) so the exact slot object can be disconnected.
        self._canvas_dbl_click_conns = []
        controller = self._view_controller()
        if controller is not None:
            for canvas, beam in (
                (controller.sem_canvas, BeamType.ELECTRON),
                (controller.fib_canvas, BeamType.ION),
            ):
                slot = partial(self._on_canvas_double_click, beam)
                canvas.canvas_double_clicked.connect(slot)
                self._canvas_dbl_click_conns.append((canvas, slot))

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
        self.image_widget.acquisition_progress_signal.connect(
            self.handle_acquisition_update
        )

        stage_limits = self.microscope._stage.limits
        xlimits = stage_limits["x"]
        ylimits = stage_limits["y"]
        zlimits = stage_limits["z"]
        tlimits = stage_limits["t"]

        self.doubleSpinBox_movement_stage_tilt.setMinimum(tlimits.min)
        self.doubleSpinBox_movement_stage_tilt.setMaximum(tlimits.max)
        self.doubleSpinBox_movement_stage_x.setMinimum(
            xlimits.min * constants.SI_TO_MILLI
        )
        self.doubleSpinBox_movement_stage_x.setMaximum(
            xlimits.max * constants.SI_TO_MILLI
        )
        self.doubleSpinBox_movement_stage_y.setMinimum(
            ylimits.min * constants.SI_TO_MILLI
        )
        self.doubleSpinBox_movement_stage_y.setMaximum(
            ylimits.max * constants.SI_TO_MILLI
        )
        self.doubleSpinBox_movement_stage_z.setMinimum(
            zlimits.min * constants.SI_TO_MILLI
        )
        self.doubleSpinBox_movement_stage_z.setMaximum(
            zlimits.max * constants.SI_TO_MILLI
        )

        # set custom tilt limits for the compustage
        if self.microscope.stage_is_compustage:
            # NOTE: these values are expressed in mm in the UI, hence the conversion
            # set x, y, z step sizes to be 1 um
            self.doubleSpinBox_movement_stage_x.setSingleStep(
                1e-6 * constants.SI_TO_MILLI
            )
            self.doubleSpinBox_movement_stage_y.setSingleStep(
                1e-6 * constants.SI_TO_MILLI
            )
            self.doubleSpinBox_movement_stage_z.setSingleStep(
                1e-6 * constants.SI_TO_MILLI
            )

            # hide rotation control for compustage
            self.label_movement_stage_rotation.setVisible(False)
            self.doubleSpinBox_movement_stage_rotation.setVisible(False)

        # stylesheets
        self.pushButton_move.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        self.pushButton_move_to_fib_orientation.setStyleSheet(
            SECONDARY_BUTTON_STYLESHEET
        )
        self.pushButton_move_to_sem_orientation.setStyleSheet(
            SECONDARY_BUTTON_STYLESHEET
        )
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
        self.doubleSpinBox_milling_angle.setValue(
            self.microscope.system.stage.milling_angle
        )  # deg
        self.doubleSpinBox_milling_angle.setSuffix(constants.DEGREE_SYMBOL)
        self.doubleSpinBox_milling_angle.setSingleStep(1.0)
        self.doubleSpinBox_milling_angle.setDecimals(1)
        self.doubleSpinBox_milling_angle.setRange(0, 45)
        self.doubleSpinBox_milling_angle.setToolTip(
            "The milling angle is the difference between the stage and the fib viewing angle."
        )
        self.doubleSpinBox_milling_angle.setKeyboardTracking(False)
        self.doubleSpinBox_milling_angle.valueChanged.connect(
            self._update_milling_angle
        )
        self.pushButton_move_to_milling_angle.clicked.connect(
            lambda: self.move_to_orientation("MILLING")
        )

        # set degree symbols for rotation and tilt
        self.doubleSpinBox_movement_stage_rotation.setSuffix(constants.DEGREE_SYMBOL)
        self.doubleSpinBox_movement_stage_tilt.setSuffix(constants.DEGREE_SYMBOL)

        # Install wheel blocker on all double spin boxes
        install_wheel_blocker(self.doubleSpinBox_movement_stage_x)
        install_wheel_blocker(self.doubleSpinBox_movement_stage_y)
        install_wheel_blocker(self.doubleSpinBox_movement_stage_z)
        install_wheel_blocker(self.doubleSpinBox_movement_stage_rotation)
        install_wheel_blocker(self.doubleSpinBox_movement_stage_tilt)
        install_wheel_blocker(self.doubleSpinBox_milling_angle)

        if cfg.FEATURE_SAMPLE_HOLDER_WIDGET_ENABLED:
            from fibsem.ui.widgets.sample_holder_widget import SampleHolderWidget

            self.sample_holder_widget = SampleHolderWidget(microscope=self.microscope)
            self.sample_holder_widget.set_holder(self.microscope._stage.holder)
            self.gridLayout_2.addWidget(self.sample_holder_widget, 3, 0)

        self.update_ui()

    def _teardown_connections(self) -> None:
        """Disconnect from the app-lifetime quad-view canvases before this widget is
        destroyed. The canvases outlive the per-connection movement widget; without this a
        stale double-click after teardown fires on a deleted widget and PyQt aborts the
        process. Idempotent — safe to call more than once."""
        for canvas, slot in getattr(self, "_canvas_dbl_click_conns", []):
            try:
                canvas.canvas_double_clicked.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        self._canvas_dbl_click_conns = []

    def _toggle_interactions(self, enable: bool, caller: Optional[str] = None):
        """Toggle the interactions in the widget depending on microscope state"""
        for btn in self._move_buttons:
            btn.setEnabled(enable)
        self.doubleSpinBox_milling_angle.setEnabled(enable)
        if caller is None:
            # self.parent.milling_widget._toggle_interactions(enable, caller="movement")
            self.parent.image_widget._toggle_interactions(enable, caller="movement")
        # No disabled branch: both sheets carry a :disabled rule, so setEnabled
        # above is what greys these out.
        for btn in self._move_buttons:
            btn.setStyleSheet(
                PRIMARY_BUTTON_STYLESHEET
                if btn is self.pushButton_move
                else SECONDARY_BUTTON_STYLESHEET
            )

    def handle_movement_progress_update(self, ddict: dict) -> None:
        """Handle movement progress updates from the microscope.

        Written to the canvas info bar rather than shown as toasts. These messages
        bracket a blocking move -- one click-to-move emits four of them inside ~45 ms
        -- so as popups they stack into a wall that says nothing the moving stage and
        the refreshing images do not already show. On the info bar the same words stay
        put for the duration of the move, which is when they are worth reading.

        Toasts show unconditionally (FIB-781), so that wall of popups is what these
        messages would actually produce rather than a thing to worry about later.
        """
        msg = ddict.get("msg", None)
        if msg is not None:
            logging.debug(msg)
            self._set_move_status(msg)

        is_finished = ddict.get("finished", False)
        if is_finished:
            # Cleared, or the last line sits there afterwards as though the stage were
            # still moving. Every path reaches here: each worker connects `finished` to
            # `move_stage_finished`, which emits this.
            #
            # Except while the images are still being retaken -- which is the usual
            # case, because `update_ui_after_movement` only *queues* the acquisition
            # before the worker returns. `move_stage_finished` already declines to
            # re-enable the buttons in that window; the status has to keep the same
            # counsel or it says "done" over a second of acquisition, and offers a
            # double-click that is still disabled. `handle_acquisition_update` clears
            # it when the images actually land.
            if not self.image_widget.is_acquiring:
                self._set_move_status(None)
            self._update_position_readout()

    def _set_move_status(self, msg: Optional[str]) -> None:
        """Put *msg* on the info bar of every canvas, or clear it when None.

        Not the instructions label on this tab. Five of the six paths that start a
        stage move start it from somewhere else -- the canvas beside the tabs (which
        takes a double-click whatever tab is showing), either minimap, or the lamella
        list -- so a message on the Movement tab is one the operator is usually not
        looking at, where a toast could be read from anywhere. The info bar is beside
        the canvas that was clicked, is visible from every tab, and is already where
        the stage position this move is changing gets written.
        """
        controller = self._view_controller()
        if controller is None:
            return
        controller.set_info(BeamType.ELECTRON, "move", msg)
        controller.set_info(BeamType.ION, "move", msg)
        controller.set_fm_info("move", msg)

    def _update_position_readout(
        self, stage_position: Optional[FibsemStagePosition] = None
    ) -> None:
        """Refresh the stage/beam readout on the quad-view info bar (debounced render)."""
        controller = self._view_controller()
        if controller is not None:
            controller.update_info(self.microscope, stage_position=stage_position)

    def handle_acquisition_update(self, ddict: dict):
        """Handle acquisition updates from the image widget"""
        is_finished = ddict.get("finished", False)
        if is_finished:
            # The other half of the hand-off above: a move's status outlives its
            # `finished` so it covers the acquisition that follows, and this is what
            # takes it down. A no-op when no move put anything there.
            self._set_move_status(None)
            self.update_ui()

    @ensure_main_thread
    def update_ui(self, stage_position: Optional[FibsemStagePosition] = None):
        """Update the UI with the current stage position and saved positions"""
        if stage_position is None:
            stage_position = self.microscope.get_stage_position()

        self.doubleSpinBox_movement_stage_x.setValue(
            stage_position.x * constants.SI_TO_MILLI
        )
        self.doubleSpinBox_movement_stage_y.setValue(
            stage_position.y * constants.SI_TO_MILLI
        )
        self.doubleSpinBox_movement_stage_z.setValue(
            stage_position.z * constants.SI_TO_MILLI
        )
        self.doubleSpinBox_movement_stage_rotation.setValue(
            np.degrees(stage_position.r)
        )
        self.doubleSpinBox_movement_stage_tilt.setValue(np.degrees(stage_position.t))

        # update the current position label
        self._update_position_readout(stage_position=stage_position)

    @ensure_main_thread
    def update_ui_after_movement(self, retake: bool = True):
        # disable taking images after movement here
        if (
            retake is False
            or self.microscope.is_acquiring
            or self.microscope.fm is not None
            and self.microscope.fm.objective.state == "Inserted"
        ):
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
        milling_angle = self.doubleSpinBox_milling_angle.value()  # deg
        self.microscope.set_milling_angle(milling_angle)

        # refresh tooltip and overlay
        milling = self.microscope.get_orientation("MILLING")
        self.pushButton_move_to_milling_angle.setToolTip(milling.pretty_orientation)
        self._update_position_readout()

    #### MOVEMENT

    def move_to_position(self, stage_position: Optional[FibsemStagePosition] = None):
        """Move the stage to the position specified in the UI"""
        if stage_position is None:
            stage_position = self.get_position_from_ui()
        self._move_to_absolute_position(stage_position)

    def _move_to_absolute_position(self, stage_position: FibsemStagePosition):
        """Move the stage to the specified position"""
        self._toggle_interactions(enable=False)
        # Said here rather than from inside the worker: this is the GUI thread, so the
        # status is on the info bar before the stage starts moving instead of racing the
        # move through the event loop.
        self.movement_progress_signal.emit(
            {"msg": f"Moving to {stage_position.pretty}…"}
        )
        worker = self.absolute_movement_worker(stage_position=stage_position)
        worker.returned.connect(self._on_stage_move_returned)
        worker.finished.connect(self.move_stage_finished)
        worker.start()

    @thread_worker
    def absolute_movement_worker(self, stage_position: FibsemStagePosition) -> None:
        """Move the stage. Runs off the GUI thread — only signals may cross back."""
        self.microscope.safe_absolute_stage_movement(stage_position)

    def _on_stage_move_returned(self, _result: object = None) -> None:
        """The move landed: say the images are coming, and retake them. GUI thread.

        Hangs off ``returned``, not ``finished``, so a move that raised skips it — the
        widget never claims to be retaking images for a move that did not happen.

        Ordering matters here and is not incidental. ``FunctionWorker`` emits
        ``returned`` before ``finished``, so the acquisition this queues is already
        under way by the time ``move_stage_finished`` asks whether it is, and the
        buttons stay disabled until the images actually land.
        """
        self.movement_progress_signal.emit({"msg": ACQUIRING_IMAGES})
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

    def _click_to_move_available(self) -> bool:
        """Whether a click may start a stage move right now.

        Click-to-move is the same action as the Move button, so it honours the same
        enabled state — `_toggle_interactions` disables the buttons for the whole move
        *and* the acquisition that follows it (`move_stage_finished` deliberately returns
        early while `image_widget.is_acquiring`). Only the buttons were ever gated,
        though: a click landing in that window started a second, overlapping stage move
        and a second acquisition on the same microscope, which is how a SEM frame ended
        up on the FIB canvas.
        """
        return self.pushButton_move.isEnabled()

    def _execute_stage_move(
        self,
        beam_type: BeamType,
        point: Point,
        vertical_move: bool,
        coords: Optional[dict] = None,
    ) -> None:
        """Dispatch a stage move from a microscope-space delta (worker thread).

        ``coords`` is the
        originating image-space click, carried through purely so the debug log still
        records where the operator actually clicked when a move goes wrong on hardware.
        """
        movement_mode = "Vertical" if vertical_move else "Stable"

        logging.debug(
            {
                "msg": "stage_movement",  # message type
                "movement_mode": movement_mode,  # movement mode
                "beam_type": beam_type.name,  # beam type
                "dm": point.to_dict(),  # shift in microscope coordinates
                "coords": coords,  # coords in image coordinates
            }
        )

        # refuse rather than silently fall through to stable_move below: on backends
        # without move_coincident_from_sem (e.g. the simulator) the operator would
        # ask to restore coincidence and get a sample-plane move instead
        if (
            beam_type is BeamType.ELECTRON
            and vertical_move
            and not hasattr(self.microscope, "move_coincident_from_sem")
        ):
            logging.warning(
                "Vertical move from the SEM view is not supported on this system."
            )
            notification_service.show_toast(
                "Vertical move from the SEM view is not supported on this system - use the FIB view.",
                "warning",
            )
            return

        # Which kind of move, because they are different operations and the user chose
        # between them: a plain double-click moves laterally, Alt + double-click moves
        # along the beam axis to hold eucentricity. "Vertically" rather than
        # "eucentric" so the message matches the words already on screen in
        # INSTRUCTIONS_TEXT.
        self.movement_progress_signal.emit(
            {
                "msg": "Moving the stage vertically…"
                if vertical_move
                else "Moving the stage…"
            }
        )
        # eucentric is only supported for ION beam
        if beam_type is BeamType.ION and vertical_move:
            self.microscope.vertical_move(dx=point.x, dy=point.y)
        elif beam_type is BeamType.ELECTRON and vertical_move:
            # move coincident from SEM
            self.microscope.move_coincident_from_sem(
                dx=0, dy=point.y
            )  # TMP: disable dx for now
        else:
            # corrected stage movement
            self.microscope.stable_move(
                dx=point.x,
                dy=point.y,
                beam_type=beam_type,
            )
        self.movement_progress_signal.emit({"msg": ACQUIRING_IMAGES})
        self.update_ui_after_movement()

    def _on_canvas_double_click(
        self, beam_type: BeamType, x: float, y: float, modifiers
    ) -> None:
        """Canvas double-click -> move stage."""
        if not self._click_to_move_available():
            return
        self._toggle_interactions(enable=False)
        worker = self._canvas_double_click_worker(beam_type, x, y, modifiers)
        worker.finished.connect(self.move_stage_finished)
        worker.start()

    @thread_worker
    def _canvas_double_click_worker(
        self, beam_type: BeamType, x: float, y: float, modifiers
    ):
        """Thread worker for quad-view double-clicks (one image per canvas).

        ``x, y`` are already beam-local, full-resolution image pixels — the canvas emits
        data coords, one image per canvas, so there is no side-by-side offset to undo.
        """
        if "Shift" in modifiers:
            return
        if (
            hasattr(self.parent, "milling_widget")
            and self.parent.milling_widget.is_milling
        ):
            notification_service.show_toast(
                "Cannot move stage while milling is in progress."
            )
            return
        image = (
            self.image_widget.eb_image
            if beam_type is BeamType.ELECTRON
            else self.image_widget.ib_image
        )
        if image is None or image.metadata is None:
            notification_service.show_toast("No image available to move from.")
            return
        h, w = image.data.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return  # click landed outside the image area
        point = conversions.image_to_microscope_image_coordinates(
            coord=Point(x=x, y=y),
            image=image.data,
            pixelsize=image.metadata.pixel_size.x,
        )
        self._execute_stage_move(
            beam_type, point, "Alt" in modifiers, coords={"x": x, "y": y}
        )

    def move_to_orientation(self, orientation: str) -> None:
        """Move to the specifed orientation"""
        if orientation not in ["SEM", "FIB", "MILLING"]:
            raise ValueError(f"Invalid orientation: {orientation}")
        self._toggle_interactions(False)
        self.movement_progress_signal.emit(
            {"msg": f"Moving to the {orientation} orientation…"}
        )
        worker = self.move_to_orientation_worker(orientation)
        # The orientation path never reported the acquisition, so the label sat on
        # "Moving to the SEM orientation…" while the move had finished and the images
        # were being retaken. Both paths now share the one slot that says it.
        worker.returned.connect(self._on_stage_move_returned)
        worker.finished.connect(self.move_stage_finished)
        worker.start()

    @thread_worker
    def move_to_orientation_worker(self, orientation: str) -> None:
        """Move the stage. Runs off the GUI thread — only signals may cross back."""
        self.microscope.move_to_orientation(orientation)
