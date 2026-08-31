"""The Qt implementation of the workflow's ``Responder`` protocol.

``QtResponder`` is *given* the window rather than being the window: the workflow
layer holds an object with one method, and the ``QMainWindow`` — with every widget
and attribute a call site could reach into — stays on this side of the seam.

``submit`` is called on the workflow thread. It marshals the request to the GUI
thread through a queued signal (this object is constructed on the GUI thread, so
Qt picks the queued path from the receiver's affinity), dispatches on the request
type, and completes the future: ``set_result`` with the answer, ``set_exception``
if acting on it failed. That second path is the point — today an exception in this
widget code escapes a queued slot and PyQt5 aborts the whole process (FIB-329);
here it re-raises on the workflow thread inside ``wait_for``, where the task's
error handling already exists.

Handlers are added one request type at a time, as each ``workflow_update_signal``
site converts; an unhandled type completes the future with a ``TypeError`` rather
than leaving the caller to its timeout.

Questions are the deferred half: their handlers show the prompt and return
*without* completing the future — the answer arrives from a click, later, through
:meth:`answer_confirm`. Only one question can be pending at a time, because the
workflow thread blocks on each; a pending future found when the next question
arrives belonged to a waiter that aborted and unwound, and is cancelled.
"""

from concurrent.futures import InvalidStateError
from copy import deepcopy
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple, Type

from PyQt5.QtCore import QObject, pyqtSignal

from fibsem.applications.autolamella.workflows.interaction import (
    ClearMillingConfig,
    Confirm,
    ConfirmDetection,
    EditAlignmentArea,
    PickPOI,
    Request,
    RunMillingTask,
    RunSpotBurn,
    SetFluorescenceChannels,
    SetImages,
    SetMillingConfig,
)
from fibsem.structures import BeamType

if TYPE_CHECKING:
    from concurrent.futures import Future

    from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI

__all__ = ["QtResponder"]


class QtResponder(QObject):
    """Answers workflow requests by driving the window's widgets, on the GUI thread."""

    _submitted = pyqtSignal(object, object)  # (Request, Future)

    def __init__(self, ui: "AutoLamellaUI"):
        super().__init__()
        self._ui = ui
        self._handlers: Dict[Type[Request], Callable] = {
            SetImages: self._set_images,
            SetMillingConfig: self._set_milling_config,
            ClearMillingConfig: self._clear_milling_config,
            SetFluorescenceChannels: self._set_fluorescence_channels,
        }
        # Deferred: the handler shows the prompt and someone else completes the
        # future later. Kept out of _handlers so _dispatch cannot complete these
        # with the handler's return value.
        self._deferred_handlers: Dict[Type[Request], Callable] = {
            Confirm: self._confirm,
            ConfirmDetection: self._confirm_detection,
            EditAlignmentArea: self._edit_alignment_area,
            PickPOI: self._pick_poi,
            RunMillingTask: self._run_milling_task,
            RunSpotBurn: self._run_spot_burn,
        }
        self._pending_question: Optional[Tuple[Request, "Future"]] = None
        # A RunMillingTask whose mill is currently running: the prompt is down,
        # the future is pending, and finished_milling_signal decides what next.
        self._active_milling: Optional[Tuple[RunMillingTask, "Future"]] = None
        # Which milling widget's finished signal is wired to _on_milling_finished
        # (by identity — the widget is rebuilt on reconnect, taking the wire with it).
        self._milling_finished_wired: Optional[object] = None
        # Same pair for the spot-burn question.
        self._active_spot_burn: Optional[Tuple[RunSpotBurn, "Future"]] = None
        self._spot_burn_finished_wired: Optional[object] = None
        self._submitted.connect(self._dispatch)

    def submit(self, request: "Request", future: "Future") -> None:
        """Hand ``request`` to the GUI thread; never blocks. Any thread."""
        self._submitted.emit(request, future)

    def _dispatch(self, request: "Request", future: "Future") -> None:
        """GUI thread. Complete the future, whatever happens in the handler."""
        deferred = self._deferred_handlers.get(type(request))
        if deferred is not None:
            try:
                deferred(request, future)
            except Exception as exc:  # noqa: BLE001 - the caller owns the failure
                self._fail(future, exc)
            return
        handler = self._handlers.get(type(request))
        try:
            if handler is None:
                raise TypeError(
                    f"QtResponder has no handler for {type(request).__name__}"
                )
            self._deliver(future, handler(request))
        except Exception as exc:  # noqa: BLE001 - the caller owns the failure
            self._fail(future, exc)

    # An abandoned ask cancels its future (wait_for, on abort or timeout), and
    # the cancel runs on the workflow thread — so any GUI-side completion can
    # find the future already cancelled, including between a cancelled() check
    # and the set. set_result on a cancelled future is InvalidStateError inside
    # a Qt slot, i.e. a process abort; these two tolerate it instead. Dropping
    # the value is correct: cancellation means nobody will read it.

    @staticmethod
    def _deliver(future: "Future", value) -> None:
        try:
            future.set_result(value)
        except InvalidStateError:
            pass

    @staticmethod
    def _fail(future: "Future", exc: Exception) -> None:
        try:
            future.set_exception(exc)
        except InvalidStateError:
            pass

    # --- handlers, one per request type ------------------------------------------

    def _set_images(self, request: SetImages) -> None:
        """Show the acquisition images. Moved from handle_workflow_update."""
        image_widget = self._ui.image_widget
        if image_widget is None:
            # Under the old signal this raise was a process abort; now it surfaces
            # in the workflow thread as this instruction's failure.
            raise RuntimeError("No image widget available to display images.")

        if request.sem_image is not None:
            image_widget.eb_image = request.sem_image
            image_widget._on_acquire(request.sem_image)
            image_widget.set_ui_from_settings(
                image_settings=request.sem_image.metadata.image_settings,
                beam_type=BeamType.ELECTRON,
            )
        if request.fib_image is not None:
            image_widget.ib_image = request.fib_image
            image_widget._on_acquire(request.fib_image)
            image_widget.set_ui_from_settings(
                image_settings=request.fib_image.metadata.image_settings,
                beam_type=BeamType.ION,
            )

    def _milling_widget(self):
        widget = self._ui.milling_task_config_widget
        if widget is None:
            # Under the old signal this raise was a process abort; now it surfaces
            # in the workflow thread as the instruction's failure.
            raise RuntimeError("No milling task config widget available.")
        return widget

    def _set_milling_config(self, request: SetMillingConfig) -> None:
        """Load the config into the milling editor and bring its tab forward."""
        widget = self._milling_widget()
        widget.update_from_settings(request.config)
        widget.setEnabled(True)
        self._ui.tabWidget.setCurrentWidget(widget)

    def _clear_milling_config(self, request: ClearMillingConfig) -> None:
        """Clear the milling editor. Moved from handle_workflow_update."""
        self._milling_widget().clear()

    # Unlike the image and milling widgets, whose absence is a defect, the FM and
    # spot-burn widgets legitimately do not exist on systems without the hardware —
    # and a workflow must not fail over that. The old handler skipped silently;
    # these acknowledge the instruction and do the same.

    def _set_fluorescence_channels(self, request: SetFluorescenceChannels) -> None:
        """Load the channel settings into the fluorescence widget, if there is one."""
        widget = self._ui.fm_control_widget
        if widget is None:
            return
        widget.channelSettingsWidget.channel_settings = request.channels

    # --- questions: the answer arrives from a click, later -------------------------

    def _park_question(
        self, request: Request, future: "Future", msg: str, pos: str, neg: Optional[str]
    ) -> None:
        """Hold ``future`` for :meth:`answer_confirm` and put the prompt up."""
        if self._pending_question is not None:
            # The workflow thread blocks on each question, so a live second
            # question is impossible: a pending future here belonged to a waiter
            # that aborted and unwound without an answer. Cancel it so nobody
            # trips over the corpse.
            self._pending_question[1].cancel()
        self._pending_question = (request, future)
        # Display state, not a handshake: the workflow no longer polls this flag
        # for converted questions, but the attention button, border and timeline
        # pause still read it.
        self._ui.WAITING_FOR_USER_INTERACTION = True
        # Reuse the whole existing display path — prompt label, yes/no buttons,
        # and the main window's waiting indicators — by emitting the payload the
        # old mechanism showed. We are on the GUI thread, so both windows' slots
        # run directly, before this returns.
        self._ui.workflow_update_signal.emit({"msg": msg, "pos": pos, "neg": neg})

    def _confirm(self, request: Confirm, future: "Future") -> None:
        """Show a yes/no prompt; :meth:`answer_confirm` completes the future."""
        self._park_question(
            request, future, request.message, request.positive, request.negative
        )

    def _confirm_detection(self, request: ConfirmDetection, future: "Future") -> None:
        """Show detected features for correction; the click answers with the set."""
        det_widget = self._ui.det_widget
        if det_widget is None:
            # Detection reached the UI without a detection widget: a defect, not
            # optional hardware — the model already ran to produce this request.
            raise RuntimeError("No detection widget available to confirm features.")
        det_widget.set_detected_features(request.detection)
        tab_widget = self._ui.tabWidget
        det_idx = tab_widget.indexOf(det_widget)
        if det_idx != -1:
            tab_widget.setTabVisible(det_idx, True)
            tab_widget.setCurrentIndex(det_idx)
        self._park_question(
            request,
            future,
            "Confirm Feature Detection. Press Continue to proceed.",
            "Continue",
            None,
        )

    def _edit_alignment_area(
        self, request: EditAlignmentArea, future: "Future"
    ) -> None:
        """Show the editable alignment overlay; the click answers with the area."""
        image_widget = self._ui.image_widget
        if image_widget is None:
            raise RuntimeError("No image widget available to edit the alignment area.")
        image_widget.toggle_alignment_area(request.initial)
        self._park_question(request, future, request.message, "Continue", None)

    def _view_controller(self):
        """The main window's quad-view controller, or None standalone."""
        return getattr(self._ui.parent_widget, "view_controller", None)

    def _pick_poi(self, request: PickPOI, future: "Future") -> None:
        """Show a draggable POI marker on the FIB canvas; the click answers with it.

        The overlay lives on the main window's quad view. Without one (the
        standalone embedded window) there is nothing to pick on: the old flow
        no-oped and delivered None, so answer None immediately rather than
        putting up a prompt about a marker that is not there.
        """
        controller = self._view_controller()
        if controller is None:
            self._deliver(future, None)
            return

        from fibsem import conversions
        from fibsem.ui.widgets.canvas.canvas_state import PointsSpec

        image = request.image
        if request.initial is not None:
            px = conversions.microscope_image_to_image_coordinates(
                request.initial, image.data.shape, image.metadata.pixel_size.x
            )
            col, row = px.x, px.y
        else:
            row = image.data.shape[0] / 2
            col = image.data.shape[1] / 2
        controller.set_overlay(
            BeamType.ION,
            PointsSpec(
                id="poi",
                points=[(col, row)],
                # Matches the protocol editor's POI, down to the legend entry: same
                # concept, same marker. Left to the defaults this drew at size 18 with
                # PointOverlay's 2.0 edge, a cross visibly fatter than the thin one the
                # editor and the config preview draw, and absent from the legend
                # (FIB-582). 1.2 keeps it reading like the centre crosshair.
                color="magenta",
                selected_color="magenta",
                marker="+",
                size=14,
                edge_width=1.2,
                legend_label="Point of Interest",
                add_on_right_click=False,
                removable=False,
            ),
        )
        # POI owns FIB-canvas input: stage-move + milling menu stand down. The toolbar
        # toggle lets the user drop to Move and back. (See active-overlay model.)
        controller.arm_overlay(BeamType.ION, "poi", label="POI", icon="mdi:map-marker")
        controller.fib_canvas.set_hint("drag to move")
        self._park_question(request, future, request.message, "Continue", None)

    def _run_milling_task(self, request: RunMillingTask, future: "Future") -> None:
        """Hand the config to the editor; run and re-ask until Continue.

        The whole mill loop lives here now: show the config, ask (when
        ``confirm``), run on the Run Milling click, wait for the widget's own
        ``finished_milling_signal``, re-ask, and only complete the future — with
        the config as actually used — on Continue. The workflow thread just
        blocks on its future, instead of emitting ``start_milling_signal`` with
        a BlockingQueuedConnection and sleep-polling ``is_milling`` across the
        seam.
        """
        widget = self._milling_widget()
        widget.update_from_settings(request.config)
        widget.setEnabled(True)
        self._ui.tabWidget.setCurrentWidget(widget)
        # The workflow runs the mill; the editor's own Run button stands down
        # (moved from handle_workflow_update's milling_enabled branch).
        widget.milling_widget.pushButton_run_milling.setVisible(False)
        self._wire_milling_finished(widget)

        if not request.confirm:
            if request.enabled:
                self._start_milling_run(request, future)
            else:
                self._deliver(future, self._finish_milling_question())
            return
        pos, neg = (
            ("Run Milling", "Continue") if request.enabled else ("Continue", None)
        )
        self._park_question(request, future, request.message, pos, neg)

    def _wire_milling_finished(self, widget) -> None:
        """Connect the (possibly rebuilt) milling widget's finished signal, once."""
        if self._milling_finished_wired is widget:
            return
        widget.milling_widget.finished_milling_signal.connect(self._on_milling_finished)
        self._milling_finished_wired = widget

    def _start_milling_run(self, request: RunMillingTask, future: "Future") -> None:
        """Run the editor's current config; the finished signal decides what next."""
        self._active_milling = (request, future)
        self._ui.workflow_update_signal.emit(
            {"msg": f"Milling {request.config.name}..."}
        )
        # None: the widget builds the config from the editor, so the operator's
        # edits are what actually runs — as the old start_milling_signal path did.
        self._milling_widget().milling_widget.run_milling(None)

    def _on_milling_finished(self) -> None:
        """GUI thread, from finished_milling_signal — success and failure alike."""
        active = self._active_milling
        if active is None:
            return  # a mill the operator ran outside a question
        self._active_milling = None
        request, future = active
        if future.cancelled():
            return
        self._ui.workflow_update_signal.emit(
            {
                "msg": f"Milling {request.config.name} Complete: "
                f"{len(request.config.stages)} stages completed."
            }
        )
        if request.confirm:
            # Same prompt again: Run Milling reruns (after edits), Continue ends.
            self._park_question(
                request, future, request.message, "Run Milling", "Continue"
            )
            return
        try:
            self._deliver(future, self._finish_milling_question())
        except Exception as exc:  # noqa: BLE001 - the caller owns the failure
            self._fail(future, exc)

    def _finish_milling_question(self):
        """The answer: the config as the editor holds it, with the editor cleared."""
        widget = self._milling_widget()
        config = deepcopy(widget.get_config())
        widget.clear()
        return config

    def abandon(self) -> None:
        """Drop whatever a finished run left behind. GUI thread, workflow end.

        An abort races the GUI: a cancelled mill or burn that finishes quickly
        re-parks the prompt in the gap before the aborting waiter cancels its
        future. The run is over when this is called — the workflow thread has
        exited — so a question still parked, or a run still tracked, belongs to
        nobody: cancel it and take the prompt down.
        """
        pending, self._pending_question = self._pending_question, None
        milling, self._active_milling = self._active_milling, None
        burning, self._active_spot_burn = self._active_spot_burn, None
        for pair in (pending, milling, burning):
            if pair is not None:
                pair[1].cancel()
        if pending is not None:
            self._ui.WAITING_FOR_USER_INTERACTION = False
            self._ui.workflow_update_signal.emit({"msg": ""})

    def _run_spot_burn(self, request: RunSpotBurn, future: "Future") -> None:
        """Place-points-and-burn, mirroring the milling question.

        Without a spot-burn widget the answer is None immediately: optional
        hardware must not fail a workflow, same as the spot-burn instructions.
        """
        widget = self._ui.spot_burn_widget
        if widget is None:
            self._deliver(future, None)
            return
        widget.set_settings(request.settings)
        # Front the tab and enter workflow mode (moved from
        # handle_workflow_update's spot_burn branch).
        self._ui.set_spot_burn_widget_active(True)
        widget.set_workflow_mode(True)
        self._wire_spot_burn_finished(widget)
        self._park_question(
            request, future, request.message, "Run Spot Burn", "Continue"
        )

    def _wire_spot_burn_finished(self, widget) -> None:
        """Connect the (possibly rebuilt) spot-burn widget's finished signal, once."""
        if self._spot_burn_finished_wired is widget:
            return
        widget.finished_spot_burn_signal.connect(self._on_spot_burn_finished)
        self._spot_burn_finished_wired = widget

    def _start_spot_burn_run(self, request: RunSpotBurn, future: "Future") -> None:
        """Run the widget's current points; the finished signal re-asks."""
        self._active_spot_burn = (request, future)
        self._ui.workflow_update_signal.emit({"msg": "Running Spot Burn..."})
        widget = self._ui.spot_burn_widget
        widget.run_spot_burn_worker()
        if not widget.is_burning:
            # Refused — no in-bounds points — so no finished signal will come.
            # The old is_milling-style poll fell straight through and re-asked;
            # do the same now.
            self._active_spot_burn = None
            self._park_question(
                request, future, request.message, "Run Spot Burn", "Continue"
            )

    def _on_spot_burn_finished(self) -> None:
        """GUI thread, from finished_spot_burn_signal — success and failure alike."""
        active = self._active_spot_burn
        if active is None:
            return  # a burn the operator ran outside a question
        self._active_spot_burn = None
        request, future = active
        if future.cancelled():
            return
        self._park_question(
            request, future, request.message, "Run Spot Burn", "Continue"
        )

    def _finish_spot_burn_question(self):
        """The answer: the settings as the widget holds them, widget cleared."""
        widget = self._ui.spot_burn_widget
        settings = widget.get_settings()
        widget.clear_points_layer()
        widget.set_workflow_mode(False)
        return settings

    def answer_confirm(self, clicked_yes: bool) -> bool:
        """Complete the pending question from the yes/no click.

        Returns False when no question is pending, so the caller can fall through
        to the legacy flag path. Clears the prompt and the waiting state first,
        then delivers the answer — the waiter wakes to a consistent UI. Computing
        the answer can itself fail (it may read widgets); a failure completes the
        future with the exception, which re-raises on the workflow thread instead
        of escaping this slot as a process abort.
        """
        pending = self._pending_question
        if pending is None:
            return False
        request, future = pending
        self._pending_question = None
        self._ui.WAITING_FOR_USER_INTERACTION = False
        if future.cancelled():
            # The asker aborted while the prompt stood. The click means nothing
            # beyond taking the stale prompt down — in particular it must not
            # start a mill for a question nobody is waiting on.
            self._ui.workflow_update_signal.emit({"msg": ""})
            return True
        if isinstance(request, RunMillingTask) and clicked_yes and request.enabled:
            # Run Milling: the prompt comes down but the question stays open —
            # the finished signal re-asks or completes. (No {"msg": ""} clear;
            # _start_milling_run puts the running status up in its place.)
            self._start_milling_run(request, future)
            return True
        if isinstance(request, RunSpotBurn) and clicked_yes:
            # Run Spot Burn: same shape — the finished signal re-asks.
            self._start_spot_burn_run(request, future)
            return True
        self._ui.workflow_update_signal.emit({"msg": ""})
        try:
            self._deliver(future, self._answer(request, clicked_yes))
        except Exception as exc:  # noqa: BLE001 - the caller owns the failure
            self._fail(future, exc)
        return True

    def _answer(self, request: Request, clicked_yes: bool):
        """What the click means for this question's asker."""
        if isinstance(request, RunMillingTask):
            # Only the not-run clicks reach here (answer_confirm intercepts Run
            # Milling): Continue, or either button when running is disabled.
            return self._finish_milling_question()
        if isinstance(request, RunSpotBurn):
            # Only Continue reaches here; Run Spot Burn is intercepted above.
            return self._finish_spot_burn_question()
        if isinstance(request, ConfirmDetection):
            # Both the read-back and the save used to straddle threads: the
            # workflow thread called _get_detected_features across the seam, then
            # signalled back for confirm_button_clicked. Both now run here, on the
            # GUI thread that owns the widget, before the waiter wakes.
            det_widget = self._ui.det_widget
            detection = det_widget._get_detected_features()
            det_widget.confirm_button_clicked()
            return detection
        if isinstance(request, EditAlignmentArea):
            # Read, then hide. This absorbs the old flow's second handshake: the
            # workflow used to emit "clear" and poll WAITING_FOR_UI_UPDATE before
            # reading the area across the seam — here both run in the click's
            # slot, and clear_alignment_area hides the overlay but keeps the rect.
            image_widget = self._ui.image_widget
            area = deepcopy(image_widget.get_alignment_area())
            image_widget.clear_alignment_area()
            return area
        if isinstance(request, PickPOI):
            # Read the marker, convert against the image the request carried (the
            # one the marker was placed on), and take the overlay down — the old
            # flow's second handshake plus its SELECTED_POI read-back, all in the
            # click's slot.
            from fibsem import conversions
            from fibsem.structures import Point

            controller = self._view_controller()
            poi = None
            if controller is not None:
                pts = controller.overlay_points(BeamType.ION, "poi")
                if pts:
                    col, row = pts[0]
                    image = request.image
                    poi = conversions.image_to_microscope_image_coordinates(
                        Point(x=col, y=row), image.data, image.metadata.pixel_size.x
                    )
                controller.arm_overlay(BeamType.ION, None)  # restore Move
                controller.remove_overlay(BeamType.ION, "poi")
                controller.fib_canvas.set_hint(None)
            return poi
        return clicked_yes
