import logging
from typing import Optional

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
import numpy as np
from fibsem.ui.qt.threading import FunctionWorker, thread_worker
from fibsem.constants import METRE_TO_MICRON, MICRON_TO_METRE
from fibsem.fm.microscope import FluorescenceMicroscope
from fibsem.ui import notification_service
from fibsem.ui.stylesheets import (
    PRIMARY_BUTTON_STYLESHEET,
    SECONDARY_BUTTON_STYLESHEET,
)
from superqt import ensure_main_thread
from superqt.utils import qdebounced
from fibsem.ui.utils import message_box_ui
from fibsem.ui.napari.utilities import update_text_overlay
from fibsem.ui.widgets.custom_widgets import (
    ValueSpinBox,
)

OBJECTIVE_CONFIG = {
    "position": {
        "step_size": 1.0,  # µm
        "decimals": 1,  # number of decimal places
        "suffix": " µm",  # unit suffix
        "tooltip": "Objective position in microns relative to current position",
    },
    "step_size_control": {
        "range": (0.1, 250.0),  # µm
        "step": 0.1,  # µm
        "default": 1.0,  # µm
        "decimals": 1,  # number of decimal places
        "suffix": " µm",  # unit suffix
        "tooltip": "Step size for objective movement in microns",
    },
}
MAX_OBJECTIVE_STEP_SIZE_UM = 100.0  # µm
_SAFE_TILT_TOLERANCE_DEG = 0.5


class _InsertObjectiveDialog(QDialog):
    """Combined stage-tilt selection + insert confirmation dialog."""

    def __init__(self, tilt_deg: float, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Insert Objective")
        self.setMinimumWidth(480)
        self.selected_tilt: Optional[float] = None

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        info = QLabel(
            f"Current stage tilt: <b>{tilt_deg:.1f}°</b><br><br>"
            "Select the target stage tilt. The stage will move to the selected "
            "tilt if needed before inserting the objective."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self._btn_0 = QPushButton("Insert at 0°")
        self._btn_180 = QPushButton("Insert at -180°")
        btn_cancel = QPushButton("Cancel")
        self._btn_0.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        self._btn_180.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        btn_row.addWidget(self._btn_0)
        btn_row.addWidget(self._btn_180)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)

        self._btn_0.clicked.connect(lambda: self._accept(0.0))
        self._btn_180.clicked.connect(lambda: self._accept(-180.0))
        btn_cancel.clicked.connect(self.reject)

    def _accept(self, tilt: float) -> None:
        self.selected_tilt = tilt
        self.accept()


class ObjectiveControlWidget(QWidget):
    # parent is duck-typed: hosts optionally supply .viewer, .microscope,
    # ._view_controller() and .is_acquisition_active, all read behind guards.
    def __init__(self, fm: FluorescenceMicroscope, parent: Optional[QWidget] = None, microscope=None):
        super().__init__(parent)
        self.fm = fm
        self.parent_widget = parent
        # microscope is required for insert (stage tilt read/move); fall back to
        # the parent widget's microscope when embedded in a full acquisition UI.
        self.microscope = microscope
        if self.microscope is None and parent is not None:
            self.microscope = getattr(parent, "microscope", None)
        self.initUI()

    def initUI(self):
        self.setContentsMargins(0, 0, 0, 0)
        self.pushButton_insert_objective = QPushButton("Insert Objective", self)
        self.pushButton_retract_objective = QPushButton("Retract Objective", self)
        self.pushButton_refresh_position = QPushButton("Refresh Position Data", self)
        self.pushButton_refresh_position.setToolTip("Refresh the objective position data from the microscope")
        self.pushButton_refresh_position.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        self.pushButton_refresh_position.setVisible(False)

        # add focus position controls
        self.label_focus_position = QLabel("Focus Position", self)
        self.doubleSpinBox_focus_position = ValueSpinBox(parent=self)
        self.doubleSpinBox_focus_position.setRange(self.fm.objective.limits[0] * METRE_TO_MICRON,
                                                   self.fm.objective.limits[1] * METRE_TO_MICRON)
        # Set initial value from objective's focus position if available
        if self.fm.objective.focus_position is not None:
            self.doubleSpinBox_focus_position.setValue(self.fm.objective.focus_position * METRE_TO_MICRON)
        else:
            self.doubleSpinBox_focus_position.setValue(0.0)
        self.doubleSpinBox_focus_position.setSingleStep(OBJECTIVE_CONFIG["position"]["step_size"])
        self.doubleSpinBox_focus_position.setDecimals(OBJECTIVE_CONFIG["position"]["decimals"])
        self.doubleSpinBox_focus_position.setSuffix(OBJECTIVE_CONFIG["position"]["suffix"])
        self.doubleSpinBox_focus_position.setToolTip("Focus position in microns - set and move to autofocus position")
        self.doubleSpinBox_focus_position.setKeyboardTracking(False)
        self.pushButton_move_to_focus = QPushButton("Move to Focus Position", self)
        self.pushButton_move_to_focus.setToolTip("Move objective to stored focus position")
        self.pushButton_set_focus_position = QPushButton("Set Focus Position", self)
        self.pushButton_set_focus_position.setToolTip("Set current objective position as focus position")

        # add double spin box for objective position
        self.label_objective_control = QLabel("Current Position", self)
        self.label_objective_step_size = QLabel("Step Size", self)
        self.doubleSpinBox_objective_position = ValueSpinBox(parent=self)
        self.doubleSpinBox_objective_position.setRange(self.fm.objective.limits[0] * METRE_TO_MICRON,
                                                        self.fm.objective.limits[1] * METRE_TO_MICRON)
        self.doubleSpinBox_objective_position.setValue(self.fm.objective.position * METRE_TO_MICRON)  # Convert m to µm
        self.doubleSpinBox_objective_position.setSingleStep(OBJECTIVE_CONFIG["position"]["step_size"])
        self.doubleSpinBox_objective_position.setDecimals(OBJECTIVE_CONFIG["position"]["decimals"])
        self.doubleSpinBox_objective_position.setSuffix(OBJECTIVE_CONFIG["position"]["suffix"])
        self.doubleSpinBox_objective_position.setToolTip(OBJECTIVE_CONFIG["position"]["tooltip"])
        self.doubleSpinBox_objective_position.setKeyboardTracking(False)  # Disable keyboard tracking for immediate updates
        self.doubleSpinBox_objective_step_size = ValueSpinBox(parent=self)
        self.doubleSpinBox_objective_step_size.setRange(*OBJECTIVE_CONFIG["step_size_control"]["range"])
        self.doubleSpinBox_objective_step_size.setSingleStep(OBJECTIVE_CONFIG["step_size_control"]["step"])
        self.doubleSpinBox_objective_step_size.setValue(OBJECTIVE_CONFIG["step_size_control"]["default"])
        self.doubleSpinBox_objective_step_size.setDecimals(OBJECTIVE_CONFIG["step_size_control"]["decimals"])
        self.doubleSpinBox_objective_step_size.setSuffix(OBJECTIVE_CONFIG["step_size_control"]["suffix"])
        self.doubleSpinBox_objective_step_size.setToolTip(OBJECTIVE_CONFIG["step_size_control"]["tooltip"])
        self.doubleSpinBox_objective_step_size.setKeyboardTracking(False)  # Disable keyboard tracking for immediate updates

        # add user-defined limit controls
        self.label_objective_limit = QLabel("Limit Position", self)
        self.doubleSpinBox_objective_limit = ValueSpinBox(parent=self)
        self.doubleSpinBox_objective_limit.setRange(0.0, self.fm.objective.limits[1] * METRE_TO_MICRON)
        self.doubleSpinBox_objective_limit.setValue(self.fm.objective.limit_position * METRE_TO_MICRON)
        self.doubleSpinBox_objective_limit.setSingleStep(OBJECTIVE_CONFIG["position"]["step_size"])
        self.doubleSpinBox_objective_limit.setDecimals(OBJECTIVE_CONFIG["position"]["decimals"])
        self.doubleSpinBox_objective_limit.setSuffix(OBJECTIVE_CONFIG["position"]["suffix"])
        self.doubleSpinBox_objective_limit.setToolTip("User-defined upper limit for objective position in microns")
        self.doubleSpinBox_objective_limit.setKeyboardTracking(False)

        # instructions label
        self.label_instructions = QLabel("Instructions: Use Shift + Mouse Wheel to move objective by the step size", self)
        self.label_instructions.setStyleSheet("font-style: italic; color: gray;")

        # Create the layout
        layout = QGridLayout()
        layout.addWidget(self.pushButton_insert_objective, 0, 0)
        layout.addWidget(self.pushButton_retract_objective, 0, 1)
        layout.addWidget(self.label_objective_limit, 1, 0)
        layout.addWidget(self.doubleSpinBox_objective_limit, 1, 1)
        layout.addWidget(self.label_focus_position, 2, 0)
        layout.addWidget(self.doubleSpinBox_focus_position, 2, 1)
        layout.addWidget(self.pushButton_set_focus_position, 3, 0, 1, 1)
        layout.addWidget(self.pushButton_move_to_focus, 3, 1, 1, 1)
        layout.addWidget(self.label_objective_control, 4, 0)
        layout.addWidget(self.doubleSpinBox_objective_position, 4, 1)
        layout.addWidget(self.label_objective_step_size, 5, 0)
        layout.addWidget(self.doubleSpinBox_objective_step_size, 5, 1)
        layout.addWidget(self.pushButton_refresh_position, 6, 0, 1, 2)
        layout.addWidget(self.label_instructions, 7, 0, 1, 2)
        layout.setContentsMargins(0, 0, 0, 0)  # Remove margins around the grid layout
        self.setLayout(layout)

        # connect signals
        self.pushButton_insert_objective.clicked.connect(self.insert_objective)
        self.pushButton_retract_objective.clicked.connect(self.retract_objective)
        self.doubleSpinBox_focus_position.valueChanged.connect(self.on_focus_position_changed)
        self.pushButton_move_to_focus.clicked.connect(self.move_to_focus_position)
        self.pushButton_set_focus_position.clicked.connect(self.set_focus_position)
        self.doubleSpinBox_objective_position.valueChanged.connect(self.on_objective_position_changed)
        self.doubleSpinBox_objective_step_size.valueChanged.connect(lambda value: self.doubleSpinBox_objective_position.setSingleStep(value))
        self.doubleSpinBox_objective_limit.valueChanged.connect(self.on_objective_limit_changed)
        self.pushButton_refresh_position.clicked.connect(lambda: self.update_objective_position_labels(None))

        # Moves this widget did not make -- an autofocus sweep, a z-stack, the overview
        # runner -- left the spinbox showing the position from before them. It only ever
        # refreshed after its own actions (FIB-576).
        #
        # A plain bound method, never a Qt signal's `emit`: psygnal holds bound methods
        # weakly and matches them at disconnect, and PyQt rebuilds `emit` on every access
        # so psygnal can neither weakref it nor remove it later. Marshalled by
        # `@ensure_main_thread` on the slot, which is the contract `FibsemMicroscope`
        # states for its psygnals -- the sweep that moves the objective runs on a worker.
        #
        # This is the widget's only psygnal, and it is why `closeEvent` now exists: the
        # signal outlives the widget, and an emit into one Qt has torn down on the C++
        # side is a segfault rather than an exception (FIB-550).
        self.fm.objective.position_changed.connect(self._on_objective_moved)

        # set stylesheets
        self.pushButton_insert_objective.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        self.pushButton_retract_objective.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        self.pushButton_move_to_focus.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        self.pushButton_set_focus_position.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)

        # set initial ranges based on user-defined limit
        self.on_objective_limit_changed(self.doubleSpinBox_objective_limit.value())

        # Debounce state for mouse wheel objective movement.
        #
        # `_wheel_target_um` is where the user has scrolled to; `_wheel_worker` is the
        # move currently on its way there, or None. One field, not a separate "busy"
        # flag: it is set when a move starts and cleared only in the completion slots,
        # which run on the GUI thread -- so `is not None` means "in flight, or landed
        # and not yet accounted for", with no window between the two. `is_alive()`
        # would open exactly that window.
        self._wheel_target_um: float = 0.0
        self._wheel_worker: Optional[FunctionWorker] = None
        # The threaded insert / retract, tracked for the same reason: it drives the
        # objective, so everything else must be able to see that it is doing so
        # (FIB-628). Cleared in the completion slots, which run on the GUI thread.
        self._action_worker: Optional[FunctionWorker] = None
        self._wheel_moving_to_um: Optional[float] = None
        self._execute_wheel_move = qdebounced(self._execute_wheel_move_impl, timeout=150)

        # `hasattr` is not enough once hosts go viewer-less — the attribute is still
        # there, holding None.
        if (self.parent_widget is not None
                and getattr(self.parent_widget, "viewer", None) is not None):
            self.parent_widget.viewer.mouse_wheel_callbacks.append(self._on_mouse_wheel)

    def _set_objective_actions_enabled(self, enabled: bool) -> None:
        """Enable/disable insert & retract buttons while a threaded move runs."""
        self.pushButton_insert_objective.setEnabled(enabled)
        self.pushButton_retract_objective.setEnabled(enabled)

    def _objective_busy_reason(self) -> Optional[str]:
        """What is driving the objective right now, phrased for a warning, or None.

        One question — *is anything driving the objective* — with three sources, because
        three different parties can be:

        * **Another acquisition.** A tileset, z-stack or autofocus sweep steps the
          objective itself, and a second hand on it corrupts the run. The microscope
          answers this and only the microscope can: asking whether *this widget* was
          busy is what let another tab's tileset through (FIB-513). A live stream is
          deliberately not busy — focusing while watching is what this panel is for.
        * **A focus adjustment.** The wheel move runs on a worker now (FIB-625), so
          everything here stays clickable while it is on its way.
        * **An insert or retract.** Also threaded, and always was — so a scroll during a
          retraction has always gone through (FIB-628). This is the collision-relevant
          one: `insert_objective` moves the *stage* before inserting.

        A guard on the action, not on the button. The buttons are not this widget's to
        drive — `FMControlWidget._update_acquisition_button_states` owns them and
        re-derives them from `may_start` — so disabling them here and re-enabling on
        completion overwrote its answer, and scrolling during a live stream left Insert
        and Retract clickable when the stream forbids them. That is the shape FIB-513
        closed one layer up (FIB-515).

        What it cannot see is another `ObjectiveControlWidget` — the coincidence viewer
        holds a second one, with its own workers. Closing that means the objective
        marking itself busy rather than the widget doing it, which is the FIB-441 shape
        and a driver-level change; recorded on FIB-628 rather than assumed here.
        """
        if not self.fm.is_interactive:
            return self.fm.acquiring_reason or "another acquisition"
        if self._wheel_worker is not None:
            return "a focus adjustment"
        if self._action_worker is not None:
            return "an objective insert or retract"
        return None

    def _refuse_if_objective_busy(self, what: str) -> bool:
        """Whether *what* must stand down, saying so in the log and on screen.

        Visible rather than silent: the control that triggered it stays enabled — that
        is the point of guarding the action instead of the button — so a refusal with no
        feedback reads as a dead button.
        """
        reason = self._objective_busy_reason()
        if reason is None:
            return False
        logging.info(f"{what} refused: {reason} is driving the objective")
        notification_service.show_toast(
            f"The objective is busy with {reason}.", "warning"
        )
        return True

    def _on_objective_action_finished(self, *_) -> None:
        """Re-enable controls and refresh labels after a threaded objective move."""
        self._action_worker = None
        self._set_objective_actions_enabled(True)
        self.update_objective_position_labels()

    def _on_objective_action_error(self, exc: Exception) -> None:
        """Report a failed objective operation without crashing the app."""
        self._action_worker = None
        logging.error(f"Objective operation failed: {exc}", exc_info=exc)
        self._set_objective_actions_enabled(True)
        self.update_objective_position_labels()
        message_box_ui(
            title="Objective Error",
            text=f"The objective operation failed:\n\n{exc}",
            buttons=QMessageBox.Ok,
            parent=self,
        )

    def insert_objective(self):
        """Insert the objective. The hardware move runs on a worker thread."""
        if self._refuse_if_objective_busy("Objective insertion"):
            return
        if self.fm.objective.state == "Inserted":
            message_box_ui(
                title="Objective Already Inserted",
                text="The objective is already inserted.",
                buttons=QMessageBox.Ok,
                parent=self,
            )
            return

        if self.microscope is None:
            message_box_ui(
                title="Objective Error",
                text="Cannot insert objective: no microscope connection available.",
                buttons=QMessageBox.Ok,
                parent=self,
            )
            return

        stage_pos = self.microscope.get_stage_position()
        tilt_deg = np.degrees(stage_pos.t)

        dlg = _InsertObjectiveDialog(tilt_deg=tilt_deg, parent=self)
        if dlg.exec_() != QDialog.Accepted or dlg.selected_tilt is None:
            logging.info("Objective insertion cancelled by user.")
            return

        target_tilt_deg = dlg.selected_tilt
        target_stage_pos = None
        if not np.isclose(tilt_deg, target_tilt_deg, atol=_SAFE_TILT_TOLERANCE_DEG):
            stage_pos.t = np.radians(target_tilt_deg)
            target_stage_pos = stage_pos

        self._set_objective_actions_enabled(False)
        self._action_worker = worker = self._insert_objective_worker(target_stage_pos)
        worker.returned.connect(self._on_objective_action_finished)
        worker.errored.connect(self._on_objective_action_error)
        worker.start()

    @thread_worker
    def _insert_objective_worker(self, target_stage_pos) -> None:
        """Worker: move stage to the insert tilt (if needed), then insert."""
        if target_stage_pos is not None:
            logging.info(
                f"Moving stage to {np.degrees(target_stage_pos.t):.0f}° tilt "
                "before inserting objective."
            )
            self.microscope.move_stage_absolute(target_stage_pos)
        logging.info("Inserting objective...")
        self.fm.objective.insert()
        logging.info("Objective inserted.")

    def retract_objective(self):
        """Retract the objective. The hardware move runs on a worker thread."""
        if self._refuse_if_objective_busy("Objective retraction"):
            return
        if self.fm.objective.state == "Retracted":
            message_box_ui(
                title="Objective Already Retracted",
                text="The objective is already retracted.",
                buttons=QMessageBox.Ok,
                parent=self,
            )
            return

        # confirmation dialog
        ret = message_box_ui(
            title="Retract Objective",
            text="Are you sure you want to retract the objective?",
            parent=self
        )
        if ret is False:
            logging.info("Objective retraction cancelled by user.")
            return

        self._set_objective_actions_enabled(False)
        self._action_worker = worker = self._retract_objective_worker()
        worker.returned.connect(self._on_objective_action_finished)
        worker.errored.connect(self._on_objective_action_error)
        worker.start()

    @thread_worker
    def _retract_objective_worker(self) -> None:
        """Worker: retract the objective."""
        logging.info("Retracting objective...")
        self.fm.objective.retract()
        logging.info("Objective retracted.")

    def update_objective_position_labels(self, objective_position: Optional[float] = None):
        """Update the objective position input and label."""
        if objective_position is None:
            objective_position = self.fm.objective.position
        self.doubleSpinBox_objective_position.blockSignals(True)  # Block signals to prevent recursion
        self.doubleSpinBox_objective_position.setValue(objective_position * METRE_TO_MICRON)  # Convert m to µm
        self.doubleSpinBox_objective_position.blockSignals(False)  # Unblock signals

        if self.parent_widget is not None:
            if getattr(self.parent_widget, "viewer", None) is not None:
                update_text_overlay(self.parent_widget.viewer,
                                    self.parent_widget.microscope,
                                    objective_position=objective_position)  # TODO: migrate to objective_position_changed signal
            # quad-view info bar (OBJECTIVE on the FM canvas), via the controller
            controller = None
            try:
                controller = self.parent_widget._view_controller()
            except Exception:
                controller = None
            if controller is not None:
                controller.update_info(self.parent_widget.microscope,
                                       objective_position=objective_position)

    @ensure_main_thread
    def _on_objective_moved(self, position: float, state: str) -> None:
        """Something moved the objective, and said where it ended up.

        The position comes from the signal rather than a fresh read, so this costs
        nothing and cannot disagree with the move that prompted it.

        Routed through `update_objective_position_labels`, which blocks the spinbox's
        signals around `setValue`. Without that block this closes a loop -- setValue
        raises `valueChanged`, which is wired to `on_objective_position_changed`, which
        moves the objective, which emits `position_changed` again. Qt does not warn
        about that; it simply never returns.

        `state` is accepted and unused: the button states it would drive are decided by
        guards that read the device deliberately, and a stale "Retracted" there is what
        moves the stage with the objective still in the chamber (FIB-534).

        Stands aside while the wheel is driving: every emit then comes from the wheel's
        own move and reports a position the user has already scrolled past. Not decided
        by which slot Qt happens to deliver first -- both consult the same predicate.
        """
        if self._wheel_is_driving():
            return
        self.update_objective_position_labels(position)

    def closeEvent(self, event) -> None:
        """Drop the psygnal before Qt takes the widgets down.

        It belongs to the microscope and outlives this widget, and it holds a bound
        method of an object whose C++ half `close` has already destroyed -- so the next
        emit writes into freed memory. A segfault, not an exception (FIB-550).
        """
        try:
            self.fm.objective.position_changed.disconnect(self._on_objective_moved)
        except (TypeError, RuntimeError, ValueError, KeyError):
            pass
        super().closeEvent(event)

    @pyqtSlot(float)
    def on_objective_position_changed(self, position: float):
        """Handle changes to the objective position."""
        # The last unguarded way in, and the awkward one: refusing a `valueChanged` means
        # putting the spinbox back, and that would raise `valueChanged` again and re-enter
        # here. `update_objective_position_labels` blocks the spinbox's signals around
        # `setValue`, which is what makes the put-back safe -- the same reason the
        # large-move cancel below already routes through it (FIB-515).
        if self._refuse_if_objective_busy("Objective move"):
            self.update_objective_position_labels()
            return

        is_large_change = abs(self.fm.objective.position - (position * MICRON_TO_METRE)) > 1e-3  # 1 mm threshold

        if is_large_change:
            logging.info(f"Large change in objective position requested: {position:.1f} µm")

            ret = message_box_ui(
                title="Large Objective Movement",
                text=f"Are you sure you want to move the objective position to {position:.1f} µm?",
                parent=self
            )

            if ret is False:
                logging.info("Objective position move cancelled by user.")
                # Reset the spin box to the current position
                self.update_objective_position_labels()
                return

        logging.info(f"Changing objective position to: {position:.1f} µm")
        try:
            self.fm.objective.move_absolute(position * MICRON_TO_METRE)
            logging.info(f"Objective moved to position: {position:.1f} µm")
        except Exception as e:
            logging.error(f"Failed to move objective: {e}", exc_info=e)
            message_box_ui(
                title="Objective Error",
                text=f"Failed to move the objective:\n\n{e}",
                buttons=QMessageBox.Ok,
                parent=self,
            )

        # Update the objective position label (re-syncs to the actual position)
        self.update_objective_position_labels()

    @pyqtSlot(float)
    def on_objective_limit_changed(self, limit_um: float):
        """Handle changes to the user-defined objective limit."""
        limit_m = limit_um * MICRON_TO_METRE
        logging.info(f"Updating objective user limit to: {limit_um:.1f} µm")
        self.fm.objective.limit_position = limit_m
        max_limit_um = min(limit_um, self.fm.objective.limits[1] * METRE_TO_MICRON)
        min_limit_um = self.fm.objective.limits[0] * METRE_TO_MICRON
        self.doubleSpinBox_objective_position.setRange(min_limit_um, max_limit_um)
        self.doubleSpinBox_focus_position.setRange(min_limit_um, max_limit_um)

    @pyqtSlot(float)
    def on_focus_position_changed(self, position: float):
        """Handle changes to the focus position setting."""
        logging.info(f"Focus position updated to: {position:.1f} µm")
        self.fm.objective.focus_position = position * MICRON_TO_METRE  # Convert µm to m

    def move_to_focus_position(self):
        """Move the objective to the stored focus position."""
        if self._refuse_if_objective_busy("Move to focus position"):
            return
        if self.fm.objective.focus_position is None:
            logging.warning("No focus position has been set")
            return

        focus_position_um = self.fm.objective.focus_position * METRE_TO_MICRON

        # confirmation dialog for large movements
        current_position_um = self.fm.objective.position * METRE_TO_MICRON
        is_large_change = abs(current_position_um - focus_position_um) > 1000.0  # 1 mm threshold

        if is_large_change:
            ret = message_box_ui(
                title="Move to Focus Position",
                text=f"Move objective to focus position {focus_position_um:.1f} µm?",
                parent=self
            )
            if ret is False:
                logging.info("Move to focus position cancelled by user.")
                return

        logging.info(f"Moving objective to focus position: {focus_position_um:.1f} µm")
        try:
            self.fm.objective.move_absolute(self.fm.objective.focus_position)
            logging.info(f"Objective moved to focus position: {focus_position_um:.1f} µm")
        except Exception as e:
            logging.error(f"Failed to move objective to focus position: {e}", exc_info=e)
            message_box_ui(
                title="Objective Error",
                text=f"Failed to move the objective:\n\n{e}",
                buttons=QMessageBox.Ok,
                parent=self,
            )

        # Update the objective position label
        self.update_objective_position_labels()

    def set_focus_position(self):
        """Set the focus position to the current objective position."""
        # Records rather than moves, so it looked exempt -- but the number it records is
        # a fresh device read, and mid-move that is wherever the objective happens to be
        # passing. Saving a focus of "halfway there" is the failure. FIB-515 left this
        # one to be decided rather than assumed; this is the reason to guard it.
        if self._refuse_if_objective_busy("Set focus position"):
            return
        current_position_um = self.fm.objective.position * METRE_TO_MICRON

        ret = message_box_ui(
            title="Set Focus Position",
            text=f"Set focus position to {current_position_um:.1f} µm?",
            parent=self,
        )

        if ret is False:
            logging.info("Set focus position cancelled by user.")
            return

        # set microscope objective focus position
        logging.info(f"Setting focus position to {current_position_um:.1f} µm")
        self.doubleSpinBox_focus_position.setValue(current_position_um)
        self.fm.objective.focus_position = self.fm.objective.position

    def _set_focus_position(self, position: float):
        """Set the focus position programmatically.
        Args:
            position (float): Focus position in meters.
        """
        logging.info(f"Programmatically setting focus position to: {position * METRE_TO_MICRON:.1f} µm")
        self.doubleSpinBox_focus_position.blockSignals(True)
        self.doubleSpinBox_focus_position.setValue(position * METRE_TO_MICRON)
        self.doubleSpinBox_focus_position.blockSignals(False)
        self.fm.objective.focus_position = position

    def _set_limit_position(self, position: float):
        """Set the user-defined limit position programmatically.
        Args:
            position (float): Limit position in meters.
        """
        logging.info(f"Programmatically setting limit position to: {position * METRE_TO_MICRON:.1f} µm")
        self.doubleSpinBox_objective_limit.blockSignals(True)
        self.doubleSpinBox_objective_limit.setValue(position * METRE_TO_MICRON)
        self.doubleSpinBox_objective_limit.blockSignals(False)
        self.fm.objective.limit_position = position

    def _on_mouse_wheel(self, viewer, event):
        """Handle mouse wheel events in the napari viewer.

        This method avoids all blocking microscope calls. Position is read from
        the spinbox (already synced), and the actual hardware move is debounced
        via _execute_wheel_move so rapid scroll events coalesce into a single move.
        """

        if self.parent_widget is None:
            event.handled = False
            return

        if 'Shift' not in event.modifiers:
            event.handled = False  # Let napari handle zooming if Shift is not pressed
            return

        # Prevent objective movement while something is driving the objective itself:
        # another acquisition, or this widget's own threaded insert / retract (FIB-628).
        # Asked of the microscope, not the parent widget: `is_acquisition_active` is only
        # the parent's own work, so another tab's tileset let this through (FIB-513).
        # Still allowed during a live stream -- shift-scrolling to focus while watching
        # is what this handler is for.
        #
        # Logged, not toasted: this fires per notch, and a burst would be a burst of
        # toasts. The deliberate single actions each say so on screen instead.
        reason = self._objective_busy_reason()
        if reason is not None:
            logging.info(f"Objective movement disabled: {reason}")
            event.handled = True
            return

        # Calculate step size based on wheel delta
        objective_step_size = self.doubleSpinBox_objective_step_size.value()  # µm
        step_um = np.clip(objective_step_size * event.delta[1], -MAX_OBJECTIVE_STEP_SIZE_UM, MAX_OBJECTIVE_STEP_SIZE_UM)

        # Read current position from the spinbox (no hardware call)
        current_pos_um = self.doubleSpinBox_objective_position.value()
        new_pos_um = current_pos_um + step_um

        logging.info(f"Mouse wheel: step={step_um:.1f} µm, target={new_pos_um:.1f} µm")

        # Update spinbox immediately for visual feedback (no hardware call)
        self.doubleSpinBox_objective_position.blockSignals(True)
        self.doubleSpinBox_objective_position.setValue(new_pos_um)
        self.doubleSpinBox_objective_position.blockSignals(False)

        # Store target and trigger debounced hardware move
        self._wheel_target_um = new_pos_um
        self._execute_wheel_move()

        event.handled = True

    def _on_canvas_scroll(self, x, y, direction, modifiers):
        """Quad-view equivalent of :meth:`_on_mouse_wheel`: Shift + scroll on the FM
        canvas steps the objective by the step size. Plain scroll is left to the canvas
        (zoom). ``direction`` is +1/-1 per notch; ``modifiers`` is a set/tuple like
        ``{"Shift"}``. The hardware move is debounced via ``_execute_wheel_move`` so
        rapid scroll events coalesce into a single move.
        """
        if "Shift" not in modifiers:
            return  # plain scroll -> canvas zoom

        if self.parent_widget is None:
            return
        # The microscope's answer, not the parent's: `is_acquisition_active` is only that
        # widget's own work, so another tab's tileset went straight through here
        # (FIB-513), and neither of them notices this widget's threaded insert or retract
        # (FIB-628). Logged rather than toasted -- see `_on_mouse_wheel`.
        reason = self._objective_busy_reason()
        if reason is not None:
            logging.info(f"Objective movement disabled: {reason}")
            return

        step_um = float(
            np.clip(
                self.doubleSpinBox_objective_step_size.value() * direction,
                -MAX_OBJECTIVE_STEP_SIZE_UM,
                MAX_OBJECTIVE_STEP_SIZE_UM,
            )
        )
        new_pos_um = self.doubleSpinBox_objective_position.value() + step_um

        # update spinbox immediately for visual feedback (no hardware call)
        self.doubleSpinBox_objective_position.blockSignals(True)
        self.doubleSpinBox_objective_position.setValue(new_pos_um)
        self.doubleSpinBox_objective_position.blockSignals(False)

        self._wheel_target_um = new_pos_um
        self._execute_wheel_move()

    def _execute_wheel_move_impl(self):
        """Scrolling has settled: send the objective after it, on a worker thread.

        `qdebounced` fires on a QTimer in the GUI thread, so running the move here --
        as this did -- blocked every repaint for its duration, and the live FM canvas
        stopped updating while you focused. Frames kept arriving throughout: the stream
        has its own thread, they simply could not be drawn (FIB-625).

        A move already on its way is not joined by a second one. It re-reads the target
        when it lands, so a nudge arriving mid-move *replaces* where the objective is
        going instead of queueing another trip -- which is the same coalescing the
        150 ms debounce does for the burst, extended over the move itself.
        """
        if self._wheel_worker is not None:
            return  # in flight; `_on_wheel_move_finished` picks up the newer target
        self._start_wheel_move()

    def _start_wheel_move(self) -> None:
        """Send one move to the current target, unless something else has the objective.

        The last moment before the command goes out, and the only one that is safe to
        check at: the scroll handlers guard when the notch arrives, but the debounce
        puts 150 ms between that and the move, and the chase below can dispatch later
        still. An insert, a retract or a tileset starting inside either window would
        otherwise find a move already on its way to it.

        Both callers come through here, so this is the single place that dispatches.
        """
        reason = self._objective_busy_reason()
        if reason is not None:
            logging.info(f"Wheel move abandoned: {reason} is driving the objective")
            self._wheel_moving_to_um = None  # give the readout back to the device
            self.update_objective_position_labels()
            return
        position_um = self._wheel_target_um
        logging.info(f"Executing debounced wheel move to: {position_um:.1f} µm")
        self._wheel_moving_to_um = position_um
        worker = self._wheel_move_worker(position_um * MICRON_TO_METRE)
        worker.returned.connect(self._on_wheel_move_finished)
        worker.errored.connect(self._on_wheel_move_error)
        self._wheel_worker = worker
        worker.start()

    @thread_worker
    def _wheel_move_worker(self, position_m: float) -> float:
        """Worker: move the objective, and report back where it was sent."""
        self.fm.objective.move_absolute(position_m)
        return position_m

    def _wheel_is_driving(self) -> bool:
        """Whether the readout belongs to the wheel rather than to the device.

        True from the moment a wheel move starts until the objective has caught up with
        the last notch. In between, where the objective *is* runs behind where the user
        has scrolled to, and writing it to the spinbox snaps the number backwards past
        their own scrolling for the length of the next move -- several hundred ms of
        showing the wrong position, on the readout they are focusing by. The scroll
        handler already writes each notch, so there is nothing to restore: the wheel's
        target is the truthful answer to "where is this going".

        Invisible before the move came off the GUI thread, because nothing repainted
        between the notch and the move landing.
        """
        if self._wheel_worker is not None:
            return True
        return (
            self._wheel_moving_to_um is not None
            and self._wheel_target_um != self._wheel_moving_to_um
        )

    def _on_wheel_move_finished(self, position_m: float) -> None:
        """The move landed. Chase the target if it has moved on, else settle."""
        self._wheel_worker = None
        if self._wheel_target_um != self._wheel_moving_to_um:
            self._start_wheel_move()  # stale before it could be shown; do not write it
            return
        self._wheel_moving_to_um = None  # the burst is over; the device owns the readout
        # Kept even though `position_changed` now drives the same labels (FIB-576):
        # that signal is skipped when the read behind it fails, by design, and this is
        # the path that always reports. Two label updates cost ~0.4 ms together.
        self.update_objective_position_labels(objective_position=position_m)

    def _detach_wheel_worker(self) -> None:
        """Cut a move in flight loose from this widget.

        Not joined. The objective finishes moving either way, and waiting for it is the
        blocking this change exists to remove — teardown would be the one place still
        doing it. What must not survive is the callback: these slots touch widgets, and
        an emit into one Qt has already destroyed on the C++ side is a segfault rather
        than an exception (FIB-550). Cutting the connections leaves the worker to finish
        into nothing, which is what we want.
        """
        worker = self._wheel_worker
        self._wheel_worker = None
        # Cleared with it, or `_wheel_is_driving` stays true with nothing left to drive
        # and the readout never follows the device again.
        self._wheel_moving_to_um = None
        if worker is None:
            return
        for signal, slot in (
            (worker.returned, self._on_wheel_move_finished),
            (worker.errored, self._on_wheel_move_error),
        ):
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):
                pass

    def _on_wheel_move_error(self, exc: Exception) -> None:
        """The move failed. Report it and re-sync to wherever the objective actually is."""
        self._wheel_worker = None
        # Abandon the chase too: the target is unreachable, and the honest thing to show
        # is where the objective actually stopped, not where the wheel was pointing.
        self._wheel_moving_to_um = None
        logging.error(f"Objective wheel move failed: {exc}", exc_info=exc)
        notification_service.show_toast(f"Objective move failed: {exc}", "warning")
        self.update_objective_position_labels()  # re-sync to actual position

    def cleanup(self):
        """Drop the napari mouse-wheel callback, any pending debounced move, and any
        move already on its way.

        Call on widget teardown: the callback binds this widget onto a viewer that
        outlives it, so without this a scroll after a microscope reconnect drives a dead
        widget — and PyQt turns an exception in a callback into qFatal (FIB-329)."""
        try:
            self._execute_wheel_move.cancel()
        except Exception:
            pass
        self._detach_wheel_worker()
        if (self.parent_widget is not None
                and getattr(self.parent_widget, "viewer", None) is not None):
            try:
                self.parent_widget.viewer.mouse_wheel_callbacks.remove(self._on_mouse_wheel)
            except (ValueError, RuntimeError):
                pass