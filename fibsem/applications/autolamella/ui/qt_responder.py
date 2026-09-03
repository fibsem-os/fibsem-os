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

Every interaction has a handler; an unhandled type — a new request nobody wired
up — completes the future with a ``TypeError`` rather than leaving the caller to
its timeout.

Questions are the deferred half: their handlers show the prompt and return
*without* completing the future — the answer arrives from a click, later, through
:meth:`answer_confirm`. Only one question can be pending at a time, because the
workflow thread blocks on each; a pending future found when the next question
arrives belonged to a waiter that aborted and unwound, and is cancelled.
"""

import logging
from concurrent.futures import InvalidStateError
from copy import deepcopy
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Type

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
    SetupCoincidenceMilling,
    StalePromptError,
)
from fibsem.applications.autolamella.workflows.tasks.status import (
    WorkflowStatusEvent,
)
from fibsem.structures import BeamType

if TYPE_CHECKING:
    from concurrent.futures import Future

    from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI

__all__ = ["QtResponder"]


def _has_coincidence_strategy(config) -> bool:
    from fibsem.milling.strategy.coincidence import CoincidenceMillingStrategy

    return any(
        isinstance(stage.strategy, CoincidenceMillingStrategy)
        for stage in config.enabled_stages
    )


class QtResponder(QObject):
    """Answers workflow requests by driving the window's widgets, on the GUI thread."""

    _submitted = pyqtSignal(object, object)  # (Request, Future)
    _agent_answered = pyqtSignal(
        bool, object, object, object
    )  # (clicked_yes, nonce, value, Future[bool])
    _agent_peeked = pyqtSignal(object)  # (Future[(nonce, live value)])

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
            SetupCoincidenceMilling: self._setup_coincidence_milling,
        }
        # (request, future, nonce): one attribute so a cross-thread reader sees
        # a question and its nonce as a single consistent pair. The nonce names
        # this posting of a question — a re-park (post-mill re-ask, replaced
        # corpse) is a new question and gets a new one.
        self._pending_question: Optional[Tuple[Request, "Future", int]] = None
        self._question_seq = 0
        # A RunMillingTask whose mill is currently running: the prompt is down,
        # the future is pending, and finished_milling_signal decides what next.
        self._active_milling: Optional[Tuple[RunMillingTask, "Future"]] = None
        # Which milling widget's finished signal is wired to _on_milling_finished
        # (by identity — the widget is rebuilt on reconnect, taking the wire with it).
        self._milling_finished_wired: Optional[object] = None
        # Same pair for the spot-burn question.
        self._active_spot_burn: Optional[Tuple[RunSpotBurn, "Future"]] = None
        self._spot_burn_finished_wired: Optional[object] = None
        # Question-lifecycle observers (prompt_raised / prompt_answered /
        # prompt_cancelled): the agent server's hosting feeds the event buffer,
        # the GUI timeline records who answered. A plain list rather than a Qt
        # signal so each observer is exception-isolated — a failing observer
        # must never break the click (or the other observers) it is watching.
        self._question_observers: List[Callable[[str, Dict], None]] = []
        self._submitted.connect(self._dispatch)
        self._agent_answered.connect(self._apply_agent_answer)
        self._agent_peeked.connect(self._apply_agent_peek)

    def add_question_observer(
        self, observer: Callable[[str, Dict], None]
    ) -> Callable[[], None]:
        """Subscribe to question-lifecycle events; returns the unsubscribe."""
        self._question_observers.append(observer)

        def dispose() -> None:
            try:
                self._question_observers.remove(observer)
            except ValueError:
                pass

        return dispose

    def _emit_question_event(self, kind: str, payload: Dict) -> None:
        """Tell every observer, each on its own; failures are logged, never raised."""
        for observer in list(self._question_observers):
            try:
                observer(kind, payload)
            except Exception:  # noqa: BLE001 - observers are not allowed to matter
                logging.exception("question-event observer failed; continuing")

    def submit(self, request: "Request", future: "Future") -> None:
        """Hand ``request`` to the GUI thread; never blocks. Any thread."""
        self._submitted.emit(request, future)

    # --- the agent's side of the seam (FIB-851) --------------------------------

    def pending_question(self) -> Optional["Request"]:
        """The request currently awaiting an answer, or None. Any thread.

        Returns the frozen request itself — safe to read cross-thread, and it
        carries its own context by contract ("a responder must be able to
        answer from the request alone"), which is exactly what lets a remote
        agent see what it is being asked. A question whose asker already
        aborted reads as None: nobody is waiting on it.
        """
        request, _ = self.pending_question_and_nonce()
        return request

    def pending_question_and_nonce(self) -> Tuple[Optional["Request"], Optional[int]]:
        """The pending request and the nonce naming it, or (None, None). Any thread.

        The nonce is how a remote answer names *which* question it answers:
        echo it back through :meth:`submit_answer` and the answer applies only
        if that exact posting is still standing.
        """
        pending = self._pending_question
        if pending is None or pending[1].cancelled():
            return None, None
        return pending[0], pending[2]

    def submit_answer(
        self,
        clicked_yes: bool,
        nonce: Optional[int] = None,
        value: Optional[object] = None,
    ) -> "Future":
        """Answer the pending question as if the matching button were clicked.

        Any thread; returns a Future[bool] that resolves True when the answer
        was applied and False when there was nothing pending (the human beat
        this answer to it, or the prompt was already gone). Routes through
        :meth:`answer_confirm` on the GUI thread — the *same* path as the
        buttons, so widget read-backs, the Run Milling interception, and the
        first-writer-wins guards all behave identically for agent and human.

        With a ``nonce`` (from :meth:`pending_question_and_nonce`), the answer
        applies only to that posting of the question: if it has meanwhile been
        answered, withdrawn, or replaced, the future fails with
        :class:`StalePromptError` and nothing is clicked. Without one, the
        answer takes whatever is pending — the trusting form, for callers that
        just looked (tests, a local console).

        ``value`` optionally carries an adjusted answer for the questions read
        from live widget state — a :class:`FibsemRectangle` for
        ``EditAlignmentArea``, a :class:`Point` (microscope image coordinates)
        for ``PickPOI``. It is applied to the widget first, exactly as if the
        operator had dragged it there, and then the ordinary click path reads
        it back — so the proposed geometry lands on screen, and attribution,
        the nonce, and first-writer-wins all hold unchanged. A value on any
        other question type fails the future without clicking anything.
        """
        from concurrent.futures import Future as _Future

        outcome: "_Future" = _Future()
        self._agent_answered.emit(clicked_yes, nonce, value, outcome)
        return outcome

    def _apply_agent_answer(
        self,
        clicked_yes: bool,
        nonce: Optional[int],
        value: Optional[object],
        outcome: "Future",
    ) -> None:
        """GUI thread. Complete ``outcome`` with whether the answer applied.

        The nonce check happens here, on the thread that posts and clears
        questions — the one place it cannot race a swap.
        """
        pending = self._pending_question
        if nonce is not None:
            if pending is None or pending[1].cancelled() or pending[2] != nonce:
                self._fail(
                    outcome,
                    StalePromptError(
                        f"answer named question {nonce}, which is no longer pending"
                    ),
                )
                return
        try:
            if value is not None:
                if pending is None or pending[1].cancelled():
                    raise StalePromptError(
                        "a value-carrying answer needs a pending question"
                    )
                self._apply_value(pending[0], value)
            applied = self.answer_confirm(
                clicked_yes, source="agent", adjusted=value is not None
            )
        except Exception as exc:  # noqa: BLE001 - the asker owns the failure
            self._fail(outcome, exc)
            return
        self._deliver(outcome, applied)

    def _apply_value(self, request: "Request", value: object) -> None:
        """Put ``value`` into the widget the pending question reads from.

        GUI thread. This is the write-side mirror of :meth:`_live_answer`: the
        same widget the operator would drag, so the read-back in
        :meth:`_answer` — and the operator's own eyes — see the adjusted
        geometry before anything is accepted.
        """
        from fibsem.structures import FibsemRectangle, Point

        if isinstance(request, EditAlignmentArea):
            if not isinstance(value, FibsemRectangle):
                raise ValueError("an EditAlignmentArea value must be a FibsemRectangle")
            image_widget = self._ui.image_widget
            if image_widget is None:
                raise RuntimeError("No image widget to place the alignment area on.")
            image_widget.toggle_alignment_area(value)
            return
        if isinstance(request, PickPOI):
            if not isinstance(value, Point):
                raise ValueError("a PickPOI value must be a Point")
            controller = self._view_controller()
            if controller is None:
                raise RuntimeError("No canvas to place the point of interest on.")
            from fibsem import conversions

            image = request.image
            px = conversions.microscope_image_to_image_coordinates(
                value, image.data.shape, image.metadata.pixel_size.x
            )
            controller.set_points(BeamType.ION, "poi", [(px.x, px.y)])
            return
        raise ValueError(
            f"{type(request).__name__} answers cannot carry a value; "
            "only EditAlignmentArea and PickPOI can."
        )

    def peek_live_answer(self) -> "Future":
        """What would the answer be if Yes were clicked right now? Any thread.

        Some questions are answered from live widget state — the POI marker the
        operator is dragging, the alignment area being resized — which the
        frozen request cannot show a remote agent. This reads that state on the
        GUI thread, without disturbing it: no overlay comes down, nothing is
        answered. Resolves to ``(nonce, value)`` — the nonce pairs the value
        with a posting of the question, and value is None for question types
        with no live half (or no question at all, in which case nonce is None
        too).
        """
        from concurrent.futures import Future as _Future

        outcome: "_Future" = _Future()
        self._agent_peeked.emit(outcome)
        return outcome

    def _apply_agent_peek(self, outcome: "Future") -> None:
        """GUI thread. Complete ``outcome`` with (nonce, live value)."""
        try:
            result = self._live_answer()
        except Exception as exc:  # noqa: BLE001 - the peeker owns the failure
            self._fail(outcome, exc)
            return
        self._deliver(outcome, result)

    def _live_answer(self):
        """The live half of the pending question, read without taking anything
        down — the same widget reads as :meth:`answer_confirm`, minus the
        teardown."""
        pending = self._pending_question
        if pending is None or pending[1].cancelled():
            return None, None
        request, _, nonce = pending
        if isinstance(request, PickPOI):
            controller = self._view_controller()
            if controller is None:
                return nonce, None
            pts = controller.overlay_points(BeamType.ION, "poi")
            if not pts:
                return nonce, None
            from fibsem import conversions
            from fibsem.structures import Point

            col, row = pts[0]
            image = request.image
            return nonce, conversions.image_to_microscope_image_coordinates(
                Point(x=col, y=row), image.data, image.metadata.pixel_size.x
            )
        if isinstance(request, EditAlignmentArea):
            image_widget = self._ui.image_widget
            if image_widget is None:
                return nonce, None
            return nonce, deepcopy(image_widget.get_alignment_area())
        return nonce, None

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

        # _on_acquire owns the eb_image/ib_image references, routed by
        # metadata.beam_type — assigning them positionally here too would
        # let a mislabeled image split the two authorities silently.
        if request.sem_image is not None:
            image_widget._on_acquire(request.sem_image)
            image_widget.set_ui_from_settings(
                image_settings=request.sem_image.metadata.image_settings,
                beam_type=BeamType.ELECTRON,
            )
        if request.fib_image is not None:
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
        self._question_seq += 1
        self._pending_question = (request, future, self._question_seq)
        self._emit_question_event(
            "prompt_raised",
            {
                "type": type(request).__name__,
                "message": msg,
                "positive": pos,
                "negative": neg,
                "nonce": self._question_seq,
            },
        )
        # Display state, not a handshake: the workflow no longer polls this flag
        # for converted questions, but the attention button, border and timeline
        # pause still read it.
        self._ui.WAITING_FOR_USER_INTERACTION = True
        # We are on the GUI thread that owns the widgets: show the prompt
        # directly, and ping the status channel (message=None: says nothing
        # about the prompt) so the main window's waiting chrome refreshes.
        self._ui.set_instructions_msg(msg, pos, neg)
        self._ui.workflow_status_signal.emit(WorkflowStatusEvent())

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

        if not request.confirm():
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
        self._ui.workflow_status_signal.emit(
            WorkflowStatusEvent(message=f"Milling {request.config.name}...")
        )
        # None: the widget builds the config from the editor, so the operator's
        # edits are what actually runs — as the old start_milling_signal path did.
        milling_widget = self._milling_widget().milling_widget
        milling_widget.run_milling(None)
        self._monitor_coincidence_mill(request, milling_widget)

    def _monitor_coincidence_mill(
        self, request: RunMillingTask, milling_widget
    ) -> None:
        """A supervised coincidence mill opens the viewer attached to the run.

        Only when the question is supervised (``confirm``): an unsupervised batch
        mills with the drop as its stop and never opens a window. The viewer
        attaches to the strategies of the config the widget is actually running,
        so the plot, the chip and the Supervised toggle act on the live run; its
        Stop is the widget's Stop. Nothing here can fail the mill: a viewer that
        will not open is logged and the run carries on without it.
        """
        if not request.confirm():
            return
        running = milling_widget.running_config
        if running is None or not _has_coincidence_strategy(running):
            return
        try:
            viewer = self._coincidence_viewer()
            if viewer is None:
                return
            viewer.enter_monitor_mode(
                milling_config=running,
                on_stop=milling_widget.stop_milling,
                title=request.config.name,
            )
        except Exception:
            logging.exception(
                "Could not open the coincidence viewer to monitor the mill"
            )

    def _release_coincidence_monitor(self) -> None:
        viewer = getattr(self._ui, "_coincidence_viewer_window", None)
        if viewer is not None and viewer.in_monitor_mode:
            viewer.exit_monitor_mode()

    def _on_milling_finished(self) -> None:
        """GUI thread, from finished_milling_signal — success and failure alike."""
        active = self._active_milling
        if active is None:
            return  # a mill the operator ran outside a question
        self._active_milling = None
        self._release_coincidence_monitor()
        request, future = active
        if future.cancelled():
            return
        self._ui.workflow_status_signal.emit(
            WorkflowStatusEvent(
                message=f"Milling {request.config.name} Complete: "
                f"{len(request.config.stages)} stages completed."
            )
        )
        if request.confirm():
            # Same prompt again: Run Milling reruns (after edits), Continue ends —
            # and confirm is read live, so a supervision flip made during the mill
            # decides here: auto→supervised drops the operator into the loop,
            # supervised→auto continues without re-asking.
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
        self._release_coincidence_monitor()
        for pair in (pending, milling, burning):
            if pair is not None:
                pair[1].cancel()
        if pending is not None:
            self._ui.WAITING_FOR_USER_INTERACTION = False
            self._ui.workflow_status_signal.emit(WorkflowStatusEvent(message=""))
            self._emit_question_event(
                "prompt_cancelled",
                {"type": type(pending[0]).__name__, "nonce": pending[2]},
            )

    def _setup_coincidence_milling(
        self, request: SetupCoincidenceMilling, future: "Future"
    ) -> None:
        """Open the coincidence viewer in setup mode; the click answers with the setup.

        The viewer is the existing non-modal window: reused if the operator already
        has it open, opened otherwise. Its own Save and Continue / Skip Site buttons
        route to :meth:`answer_confirm`, the same click the main window's prompt
        makes, so either surface answers the one question.
        """
        viewer = self._coincidence_viewer()
        if viewer is None:
            raise RuntimeError(
                "No coincidence viewer available (a fluorescence microscope and a "
                "loaded experiment are required)."
            )
        viewer.enter_setup_mode(
            lamella=request.lamella,
            config=request.config,
            milling_config=request.milling_config,
            fib_image=request.fib_image,
            fm_image=request.fm_image,
            on_continue=lambda: self.answer_confirm(True),
            on_skip=lambda: self.answer_confirm(False),
        )
        self._park_question(
            request, future, request.message, "Save and Continue", "Skip Site"
        )

    def _coincidence_viewer(self):
        """The main window's coincidence viewer, opened if it is not up."""
        viewer = getattr(self._ui, "_coincidence_viewer_window", None)
        if viewer is None or not viewer.isVisible():
            self._ui._open_coincidence_milling_viewer()
            viewer = getattr(self._ui, "_coincidence_viewer_window", None)
        return viewer

    def _finish_coincidence_setup(self, clicked_yes: bool):
        """The answer: what the viewer holds on Save, None on Skip; viewer released."""
        viewer = getattr(self._ui, "_coincidence_viewer_window", None)
        if viewer is None:
            return None
        try:
            return viewer.read_setup_result() if clicked_yes else None
        finally:
            viewer.exit_setup_mode()

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
        self._ui.workflow_status_signal.emit(
            WorkflowStatusEvent(message="Running Spot Burn...")
        )
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

    def answer_confirm(
        self, clicked_yes: bool, source: str = "operator", adjusted: bool = False
    ) -> bool:
        """Complete the pending question from the yes/no click.

        Returns False when no question is pending, so the caller can fall through
        to the legacy flag path. Clears the prompt and the waiting state first,
        then delivers the answer — the waiter wakes to a consistent UI. Computing
        the answer can itself fail (it may read widgets); a failure completes the
        future with the exception, which re-raises on the workflow thread instead
        of escaping this slot as a process abort.

        ``source`` says who is answering — ``"operator"`` for the buttons,
        ``"agent"`` when routed from :meth:`submit_answer`. It rides the one
        call that applies the answer, so the attribution on the emitted
        ``prompt_answered`` event can never disagree with what happened.
        ``adjusted`` marks an answer that carried its own geometry (the value
        path), so the record shows the agent changed the widget, not just
        accepted what stood.
        """
        pending = self._pending_question
        if pending is None:
            return False
        request, future = pending[0], pending[1]
        self._pending_question = None
        self._ui.WAITING_FOR_USER_INTERACTION = False
        if future.cancelled():
            # The asker aborted while the prompt stood. The click means nothing
            # beyond taking the stale prompt down — in particular it must not
            # start a mill for a question nobody is waiting on.
            self._ui.workflow_status_signal.emit(WorkflowStatusEvent(message=""))
            self._emit_question_event(
                "prompt_cancelled",
                {"type": type(request).__name__, "nonce": pending[2]},
            )
            return True
        logging.info(
            f"prompt answered: {type(request).__name__} "
            f"response={clicked_yes} by={source} adjusted={adjusted}"
        )
        answered_payload = {
            "type": type(request).__name__,
            "response": bool(clicked_yes),
            "answered_by": source,
            "nonce": pending[2],
        }
        if adjusted:
            answered_payload["adjusted"] = True
        self._emit_question_event("prompt_answered", answered_payload)
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
        self._ui.workflow_status_signal.emit(WorkflowStatusEvent(message=""))
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
        if isinstance(request, SetupCoincidenceMilling):
            return self._finish_coincidence_setup(clicked_yes)
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
