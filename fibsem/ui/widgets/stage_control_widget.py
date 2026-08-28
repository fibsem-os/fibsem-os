"""The stage movement actions: the buttons, their workers, and click-to-move.

Everything on the Movement tab that *moves* the stage. The readout it moves is
``StagePositionWidget``, which this widget writes to but does not otherwise own.

Extracted from ``FibsemMovementWidget`` (FIB-783), which is now the container: it lays
out the panels and forwards the four calls hosts make.

**Why this half keeps the parent contract.** It reaches three neighbours, and none of
them is optional decoration:

* ``host.image_widget`` -- the interaction handshake (the buttons stay down through the
  acquisition that follows a move, and the image widget brings them back), the live
  image a canvas click is resolved against, and the acquisition-finished signal.
* ``host.milling_widget`` -- refuses a click-to-move while milling. Optional, so it is
  reached through ``hasattr``.
* the quad-view controller -- where move status and the position readout are written.

``StagePositionWidget`` needs none of that, which is why it came out first and needs no
host at all.

**The teardown is load-bearing.** The canvases are app-lifetime and outlive this widget,
which is destroyed by ``removeTab`` + ``deleteLater`` -- neither of which fires
``closeEvent``. A stale double-click arriving after that makes PyQt call ``qFatal`` and
the process aborts (FIB-329). Whatever else changes here, registration and
``_teardown_connections`` stay together.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Optional

from PyQt5.QtWidgets import QDoubleSpinBox, QGridLayout, QLabel, QPushButton, QWidget
from superqt import ensure_main_thread

from fibsem import config as cfg
from fibsem import constants, conversions
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import BeamType, FibsemStagePosition, Point
from fibsem.ui import notification_service
from fibsem.ui.qt.threading import thread_worker
from fibsem.ui.stylesheets import (
    LABEL_INSTRUCTIONS_STYLE,
    PRIMARY_BUTTON_STYLESHEET,
    SECONDARY_BUTTON_STYLESHEET,
)
from fibsem.ui.utils import install_wheel_blocker
from fibsem.ui.widgets.stage_position_widget import StagePositionWidget

INSTRUCTIONS_TEXT = (
    """Instructions: Double Click to Move. Alt + Double Click to Move Vertically"""
)

# What every path says once the stage has arrived and the images are being retaken.
# A constant rather than three string literals: the three movement paths had drifted
# to "updating images", "taking new images" and, on the orientation path, nothing at
# all -- so the same phase looked different depending on how the move was started.
ACQUIRING_IMAGES = "Acquiring images…"

ORIENTATIONS = ("SEM", "FIB", "MILLING")


class StageControlWidget(QWidget):
    """The move actions for one stage, and the state that says whether they may run.

    Parameters
    ----------
    microscope:
        The stage being moved.
    position_widget:
        The readout this widget writes after a move, and reads when the Move button is
        pressed with nothing passed in.
    host:
        The window that owns the Movement tab -- ``FibsemUI`` or ``AutoLamellaUI``. Must
        carry ``image_widget``; ``view_controller`` and ``milling_widget`` are resolved
        from it when present.
    """

    def __init__(
        self,
        microscope: FibsemMicroscope,
        position_widget: StagePositionWidget,
        host: QWidget,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.microscope = microscope
        self.position_widget = position_widget
        self.host = host
        self.image_widget = host.image_widget
        self._setup_ui()
        self.setup_connections()

    # --- construction --------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.pushButton_move = QPushButton("Move to Position")
        layout.addWidget(self.pushButton_move, 0, 0, 1, 2)

        self.pushButton_move_to_sem_orientation = QPushButton(
            "Move Flat to ELECTRON Beam"
        )
        self.pushButton_move_to_fib_orientation = QPushButton("Move Flat to ION Beam")
        layout.addWidget(self.pushButton_move_to_sem_orientation, 1, 0)
        layout.addWidget(self.pushButton_move_to_fib_orientation, 1, 1)

        self.doubleSpinBox_milling_angle = QDoubleSpinBox()
        self.pushButton_move_to_milling_angle = QPushButton("Move to Milling Angle")
        layout.addWidget(self.doubleSpinBox_milling_angle, 2, 0)
        layout.addWidget(self.pushButton_move_to_milling_angle, 2, 1)

        self.label_movement_instructions = QLabel()
        self.label_movement_instructions.setWordWrap(True)
        layout.addWidget(self.label_movement_instructions, 3, 0, 1, 2)

        self._move_buttons = [
            self.pushButton_move,
            self.pushButton_move_to_fib_orientation,
            self.pushButton_move_to_sem_orientation,
            self.pushButton_move_to_milling_angle,
        ]

    def setup_connections(self) -> None:
        # buttons
        self.pushButton_move.clicked.connect(lambda: self.move_to_position(None))
        self.pushButton_move_to_fib_orientation.clicked.connect(
            lambda: self.move_to_orientation("FIB")
        )
        self.pushButton_move_to_sem_orientation.clicked.connect(
            lambda: self.move_to_orientation("SEM")
        )

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

        # signals
        self.image_widget.acquisition_progress_signal.connect(
            self.handle_acquisition_update
        )

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

        # The form's own boxes are guarded where they are built. This one is the milling
        # angle, which belongs to this half.
        install_wheel_blocker(self.doubleSpinBox_milling_angle)

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

    # --- the neighbours ------------------------------------------------------

    def _view_controller(self):
        """Return the quad-view MicroscopeViewController, or None if unavailable.

        Resolved like the image widget: the host (standalone ``FibsemUI``) or
        host -> ``parent_widget`` (AutoLamella) holds ``view_controller``.
        """
        controller = getattr(self.host, "view_controller", None)
        if controller is not None:
            return controller
        parent_ui = getattr(self.host, "parent_widget", None)
        return getattr(parent_ui, "view_controller", None)

    def _toggle_interactions(self, enable: bool, caller: Optional[str] = None):
        """Toggle the interactions in the widget depending on microscope state"""
        for btn in self._move_buttons:
            btn.setEnabled(enable)
        self.doubleSpinBox_milling_angle.setEnabled(enable)
        if caller is None:
            self.host.image_widget._toggle_interactions(enable, caller="movement")
        # No disabled branch: both sheets carry a :disabled rule, so setEnabled
        # above is what greys these out.
        for btn in self._move_buttons:
            btn.setStyleSheet(
                PRIMARY_BUTTON_STYLESHEET
                if btn is self.pushButton_move
                else SECONDARY_BUTTON_STYLESHEET
            )

    # --- saying what the stage is doing --------------------------------------

    def _report_move(self, msg: str) -> None:
        """Say what the stage is doing, on the canvas info bar.

        Written there rather than shown as toasts. These messages bracket a blocking
        move -- one click-to-move produces four of them inside ~45 ms -- so as popups
        they stack into a wall that says nothing the moving stage and the refreshing
        images do not already show. On the info bar the same words stay put for the
        duration of the move, which is when they are worth reading.

        Toasts show unconditionally (FIB-781), so that wall of popups is what these
        messages would actually produce rather than a thing to worry about later.
        """
        logging.debug(msg)
        self._set_move_status(msg)

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

        self.position_widget.set_position(stage_position)

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

    # --- moving to a position ------------------------------------------------

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
        self._report_move(f"Moving to {stage_position.pretty}…")
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
        self._report_move(ACQUIRING_IMAGES)
        self.update_ui_after_movement()

    def move_stage_finished(self):
        """Handle the completion of a stage movement.

        Every path reaches here: each worker connects ``finished`` to this.
        """
        is_acquiring = self.image_widget.is_acquiring
        if not is_acquiring:
            # Cleared, or the last line sits there afterwards as though the stage were
            # still moving.
            self._set_move_status(None)
        # The usual case is that it *is* still acquiring, because
        # `update_ui_after_movement` only queues the acquisition before the worker
        # returns. Neither the status nor the buttons come back in that window: the
        # status would say "done" over a second of acquisition, and offer a
        # double-click that is still disabled. `handle_acquisition_update` clears it
        # when the images actually land.
        self._update_position_readout()
        if is_acquiring:
            return
        self._toggle_interactions(enable=True)

    def get_position_from_ui(self) -> FibsemStagePosition:
        """Get the stage position from the UI"""

        return self.position_widget.get_position()

    # --- moving from a canvas click ------------------------------------------

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
        """Start a stage move from a microscope-space delta (GUI thread).

        ``coords`` is the
        originating image-space click, carried through purely so the debug log still
        records where the operator actually clicked when a move goes wrong on hardware.

        The backend-capability refusal below is the last thing that can decline the
        move, so nothing is committed until it has had its say: the buttons are not
        disabled, and no worker is started, for a move that is about to be refused.
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

        # refuse rather than silently fall through to stable_move below: on a backend
        # that cannot correct coincidence from this view, the operator would ask to
        # restore coincidence and get a sample-plane move instead
        if vertical_move and not self.microscope.supports_vertical_move(beam_type):
            view = "SEM" if beam_type is BeamType.ELECTRON else "FIB"
            logging.warning(
                f"Vertical move from the {view} view is not supported on this system."
            )
            notification_service.show_toast(
                f"Vertical move from the {view} view is not supported on this system - use the other view.",
                "warning",
            )
            return

        self._toggle_interactions(enable=False)
        # Which kind of move, because they are different operations and the user chose
        # between them: a plain double-click moves laterally, Alt + double-click moves
        # along the beam axis to hold eucentricity. "Vertically" rather than
        # "eucentric" so the message matches the words already on screen in
        # INSTRUCTIONS_TEXT.
        self._report_move(
            "Moving the stage vertically…" if vertical_move else "Moving the stage…"
        )
        worker = self._stage_move_worker(beam_type, point, vertical_move)
        worker.returned.connect(self._on_stage_move_returned)
        worker.finished.connect(self.move_stage_finished)
        worker.start()

    @thread_worker
    def _stage_move_worker(
        self, beam_type: BeamType, point: Point, vertical_move: bool
    ) -> None:
        """Move the stage. Runs off the GUI thread — only signals may cross back.

        Which of the two calls applies is decided from arguments, not from widget
        state: the beam and the modifier were resolved by the caller while it was still
        on the GUI thread.
        """
        if vertical_move:
            # TMP: an x offset measured in the SEM view is still discarded
            dx = point.x if beam_type is BeamType.ION else 0
            self.microscope.vertical_move(dx=dx, dy=point.y, beam_type=beam_type)
        else:
            # corrected stage movement
            self.microscope.stable_move(
                dx=point.x,
                dy=point.y,
                beam_type=beam_type,
            )

    def _on_canvas_double_click(
        self, beam_type: BeamType, x: float, y: float, modifiers
    ) -> None:
        """Canvas double-click -> move stage.

        Every guard is answered here, on the GUI thread, before anything is committed.
        They used to run inside the worker, which meant a click that was never going to
        move the stage still disabled the buttons, spawned a thread to decide that, and
        re-enabled them — and still reported ``finished``, refreshing the stage readout
        for a move that never happened.

        Answering them here also makes them atomic with respect to the event loop: the
        whole decision is one handler, so nothing can change ``is_milling`` or swap the
        image out between the check and the dispatch.

        ``x, y`` are already beam-local, full-resolution image pixels — the canvas emits
        data coords, one image per canvas, so there is no side-by-side offset to undo.
        """
        if not self._click_to_move_available():
            return
        if "Shift" in modifiers:
            return
        if hasattr(self.host, "milling_widget") and self.host.milling_widget.is_milling:
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

    # --- moving to an orientation --------------------------------------------

    def move_to_orientation(self, orientation: str) -> None:
        """Move to the specifed orientation"""
        if orientation not in ORIENTATIONS:
            raise ValueError(f"Invalid orientation: {orientation}")
        self._toggle_interactions(False)
        self._report_move(f"Moving to the {orientation} orientation…")
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
