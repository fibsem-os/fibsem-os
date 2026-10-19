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

from fibsem.acting import AGENT, OPERATOR, TASK
from fibsem.applications.autolamella.ui.poi_marker import PoiMarker
from fibsem.structures import BeamType
from fibsem.workflows.interaction import (
    ClearMillingConfig,
    ClearSpotBurn,
    Confirm,
    ConfirmDetection,
    EditAlignmentArea,
    PickPOI,
    Request,
    SetFluorescenceChannels,
    SetImages,
    SetMillingConfig,
    ShowDetection,
    StalePromptError,
)
from fibsem.workflows.status import (
    Hold,
    HoldKind,
    WorkflowStatusEvent,
)

if TYPE_CHECKING:
    from concurrent.futures import Future

    from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI
    from fibsem.structures import FibsemImage, Point

__all__ = ["QtResponder"]


class QtResponder(QObject):
    """Answers workflow requests by driving the window's widgets, on the GUI thread."""

    _submitted = pyqtSignal(object, object)  # (Request, Future)
    _agent_answered = pyqtSignal(
        bool, object, object, object
    )  # (clicked_yes, question_id, value, Future[bool])
    _agent_peeked = pyqtSignal(object)  # (Future[(question_id, live value)])
    _agent_suggested = pyqtSignal(
        bool, object, object, object, object, object
    )  # (response, question_id, value, reason, image, Future[bool])

    def __init__(self, ui: "AutoLamellaUI"):
        super().__init__()
        self._ui = ui
        # The pending prompt's buttons, positive and negative, as it shows them.
        self._buttons: Tuple[Optional[str], Optional[str]] = (None, None)
        self._handlers: Dict[Type[Request], Callable] = {
            SetImages: self._set_images,
            SetMillingConfig: self._set_milling_config,
            ClearMillingConfig: self._clear_milling_config,
            ClearSpotBurn: self._clear_spot_burn,
            ShowDetection: self._show_detection,
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
        }
        # (request, future, question_id): one attribute so a cross-thread reader sees
        # a question and its id as a single consistent pair. The question id names
        # this posting of a question — a re-park (post-mill re-ask, replaced
        # corpse) is a new question and gets a new one.
        self._pending_question: Optional[Tuple[Request, "Future", int]] = None
        self._question_seq = 0
        # A pending PickPOI's marker, the window's one (FIB-1410), and the
        # image widget whose frame changes it follows.
        self._marker = getattr(ui, "poi_marker", None) or PoiMarker(ui)
        self._view_frames_followed: Optional[object] = None
        # Question-lifecycle observers (prompt_raised / prompt_answered /
        # prompt_cancelled): the App Server's hosting feeds the event buffer,
        # the GUI timeline records who answered. A plain list rather than a Qt
        # signal so each observer is exception-isolated — a failing observer
        # must never break the click (or the other observers) it is watching.
        self._question_observers: List[Callable[[str, Dict], None]] = []
        # The agent's suggested answer to the pending question (FIB-1283):
        # (question_id, response, value, reason), shown as a button the operator
        # may take. Dropped when that question goes.
        self._suggestion: Optional[
            Tuple[int, bool, Optional[object], str, Optional[str]]
        ] = None
        # The operator's last Dismiss of a suggestion: (question_id, reason),
        # for the agent reading the question again (sim session 2026-10-16).
        self._dismissed: Optional[Tuple[int, str]] = None
        self._submitted.connect(self._dispatch)
        self._agent_suggested.connect(self._apply_agent_suggestion)
        self._agent_answered.connect(self._apply_agent_answer)
        self._agent_peeked.connect(self._apply_agent_peek)

    def add_question_observer(
        self, observer: Callable[[str, Dict, str], None]
    ) -> Callable[[], None]:
        """Subscribe to question-lifecycle events, ``observer(kind, payload,
        actor)``; returns the unsubscribe."""
        self._question_observers.append(observer)

        def dispose() -> None:
            try:
                self._question_observers.remove(observer)
            except ValueError:
                pass

        return dispose

    def _emit_question_event(self, kind: str, payload: Dict, actor: str = TASK) -> None:
        """Tell every observer, each on its own; failures are logged, never raised.

        On the GUI thread, but every question is a task's: it is raised and
        withdrawn on the task's behalf, so *actor* is the task for the
        experiment's record (FIB-1062). An answer's is whoever answered. A
        suggestion is the agent's own.
        """
        for observer in list(self._question_observers):
            try:
                observer(kind, payload, actor)
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
        request, _ = self.pending_question_and_id()
        return request

    def pending_question_and_id(self) -> Tuple[Optional["Request"], Optional[int]]:
        """The pending request and the question id naming it, or (None, None). Any thread.

        The question id is how a remote answer names *which* question it answers:
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
        question_id: Optional[int] = None,
        value: Optional[object] = None,
    ) -> "Future":
        """Answer the pending question as if the matching button were clicked.

        Any thread; returns a Future[bool] that resolves True when the answer
        was applied and False when there was nothing pending (the human beat
        this answer to it, or the prompt was already gone). Routes through
        :meth:`answer_confirm` on the GUI thread — the *same* path as the
        buttons, so widget read-backs, the Run Milling interception, and the
        first-writer-wins guards all behave identically for agent and human.

        With a ``question_id`` (from :meth:`pending_question_and_id`), the answer
        applies only to that posting of the question: if it has meanwhile been
        answered, withdrawn, or replaced, the future fails with
        :class:`StalePromptError` and nothing is clicked. Without one, the
        answer takes whatever is pending — the trusting form, for callers that
        just looked (tests, a local console).

        ``value`` optionally carries an adjusted answer for the questions read
        from live widget state — a :class:`FibsemRectangle` for
        ``EditAlignmentArea``, a :class:`Point` (microscope image coordinates)
        for ``PickPOI``, ``{name: Point}`` in image pixels for
        ``ConfirmDetection``. It is applied to the widget first, exactly as if the
        operator had dragged it there, and then the ordinary click path reads
        it back — so the proposed geometry lands on screen, and attribution,
        the question id, and first-writer-wins all hold unchanged. A value on any
        other question type fails the future without clicking anything. A
        value equal to what the widget already shows moves nothing, and the
        future's ``adjusted`` attribute (set before it resolves) says which
        it was, and ``moved_from`` holds what the widget showed before.
        """
        from concurrent.futures import Future as _Future

        outcome: "_Future" = _Future()
        self._agent_answered.emit(clicked_yes, question_id, value, outcome)
        return outcome

    def _apply_agent_answer(
        self,
        clicked_yes: bool,
        question_id: Optional[int],
        value: Optional[object],
        outcome: "Future",
    ) -> None:
        """GUI thread. Complete ``outcome`` with whether the answer applied.

        The question id check happens here, on the thread that posts and clears
        questions — the one place it cannot race a swap.
        """
        pending = self._pending_question
        if question_id is not None:
            if pending is None or pending[1].cancelled() or pending[2] != question_id:
                self._fail(
                    outcome,
                    StalePromptError(
                        f"That question is no longer pending (question {question_id})"
                    ),
                )
                return
        adjusted = False
        try:
            if value is not None:
                if pending is None or pending[1].cancelled():
                    raise StalePromptError(
                        "a value-carrying answer needs a pending question"
                    )
                # Echoing what stands is accepting it, not adjusting it.
                _, live = self._live_answer()
                adjusted = not pending[0].same_value(live, value, self._marker.frame)
                if adjusted:
                    self._apply_value(pending[0], value)
            applied = self.answer_confirm(
                clicked_yes, source="agent", adjusted=adjusted, value=value
            )
        except Exception as exc:  # noqa: BLE001 - the asker owns the failure
            self._fail(outcome, exc)
            return
        outcome.adjusted = adjusted
        # What the value replaced, so the answer can say what moved.
        outcome.moved_from = live if adjusted else None
        self._deliver(outcome, applied)

    def submit_suggestion(
        self,
        response: bool,
        question_id: int,
        value: Optional[object] = None,
        reason: str = "",
        image: Optional[str] = None,
    ) -> "Future":
        """Suggest an answer to the pending question, for the operator to take
        or not (FIB-1283). Any thread; a Future[bool], True once the suggestion
        is shown. *image* is the path of the image the agent's evidence is in,
        which the card shows (FIB-1424). Nothing is answered: the operator's click on the suggestion
        answers, as theirs, with the agent named as the one who suggested it.

        Fails with :class:`StalePromptError` when that question is no longer
        pending, and with ValueError for a value on a question that takes none.
        """
        from concurrent.futures import Future as _Future

        outcome: "_Future" = _Future()
        self._agent_suggested.emit(response, question_id, value, reason, image, outcome)
        return outcome

    def dismissed_reason(self) -> Optional[str]:
        """Why the operator dismissed the agent's suggestion to the pending
        question ("" for no reason given); None when they haven't."""
        dismissed = self._dismissed
        _, question_id = self.pending_question_and_id()
        if dismissed is None or dismissed[0] != question_id:
            return None
        return dismissed[1]

    @property
    def suggestion(
        self,
    ) -> Optional[Tuple[int, bool, Optional[object], str, Optional[str]]]:
        """``(question_id, response, value, reason, image)`` while one stands,
        else None."""
        suggestion = self._suggestion
        if suggestion is None:
            return None
        _, question_id = self.pending_question_and_id()
        return suggestion if suggestion[0] == question_id else None

    def _apply_agent_suggestion(
        self,
        response: bool,
        question_id: int,
        value: Optional[object],
        reason: str,
        image: Optional[str],
        outcome: "Future",
    ) -> None:
        """GUI thread: put the suggestion up beside the prompt's buttons."""
        pending = self._pending_question
        if pending is None or pending[1].cancelled() or pending[2] != question_id:
            self._fail(
                outcome,
                StalePromptError(
                    f"That question is no longer pending (question {question_id})"
                ),
            )
            return
        if value is not None and type(pending[0]).value_shape is None:
            self._fail(outcome, ValueError("this question takes no value"))
            return
        self._suggestion = (
            question_id,
            bool(response),
            value,
            str(reason),
            str(image) if image else None,
        )
        payload = {
            "type": type(pending[0]).__name__,
            "question_id": question_id,
            "response": bool(response),
            "reason": str(reason),
        }
        if value is not None:
            payload["has_value"] = True
        if image:
            payload["image"] = str(image)
        self._emit_question_event("prompt_suggested", payload, actor=AGENT)
        self._show_suggestion()
        self._deliver(outcome, True)

    def take_suggestion(self) -> bool:
        """GUI thread, from the suggestion's button: answer with it, as the
        operator. A suggested value is placed in the widget first, so what is
        answered is what the operator saw. False when the suggestion is gone."""
        suggestion = self.suggestion
        if suggestion is None:
            self._clear_suggestion()
            return False
        _question_id, response, value = suggestion[:3]
        pending = self._pending_question
        adjusted = False
        if value is not None and pending is not None:
            _, live = self._live_answer()
            adjusted = not pending[0].same_value(live, value, self._marker.frame)
            if adjusted:
                self._apply_value(pending[0], value)
        return self.answer_confirm(
            response,
            source="operator",
            adjusted=adjusted,
            suggested_by="agent",
            value=value,
        )

    def dismiss_suggestion(self, reason: str = "") -> bool:
        """GUI thread, from Dismiss on the agent's card: the operator declines
        the suggestion, saying why if they like. Recorded as
        ``prompt_suggestion_dismissed`` with the reason, for the agent to read
        (FIB-1424); the question stays up. False when it had gone."""
        suggestion = self.suggestion
        pending = self._pending_question
        if suggestion is None or pending is None:
            self._clear_suggestion()
            return False
        payload = {
            "type": type(pending[0]).__name__,
            "question_id": suggestion[0],
            "response": suggestion[1],
        }
        if str(reason or "").strip():
            payload["reason"] = str(reason).strip()
        self._dismissed = (suggestion[0], str(reason or "").strip())
        self._clear_suggestion()
        self._emit_question_event("prompt_suggestion_dismissed", payload, OPERATOR)
        return True

    def _clear_suggestion(self) -> None:
        if self._suggestion is not None:
            self._suggestion = None
            self._show_suggestion()

    def _show_suggestion(self) -> None:
        show = getattr(self._ui, "show_suggestion", None)
        if show is None:
            return
        suggestion = self.suggestion
        if suggestion is None:
            show(None, "")
            return
        _question_id, response, value, reason = suggestion[:4]
        pending = self._pending_question
        label = self._button_label(pending[0], response, value) if pending else None
        shown = self._buttons[0 if response else 1]
        label = shown or label
        point = None
        if pending is not None and isinstance(pending[0], PickPOI):
            point = self._suggested_pixel(pending[0], value)
        show(label or ("Yes" if response else "No"), reason, point)

    def _suggested_pixel(self, request: "PickPOI", value: object):
        """The suggested point of interest in the FIB view's pixels, for the
        view to draw; None for none. The same frame Accept places it on and
        the answer is read back from (FIB-1410)."""
        from fibsem.structures import Point

        if not isinstance(value, Point):
            return None
        return self._marker.pixel(value, request.image)

    def _sync_poi_frame(self) -> None:
        """Keep a pending PickPOI on the frame the FIB view shows.

        GUI thread. The marker is held in the view's pixels, so when the view
        takes another frame (a new image, another field of view) the marker is
        moved to the same point on it, and the suggestion's cross redrawn.
        Placing, previewing and reading back then always use the one frame the
        operator sees (FIB-1410). Called on every view update and before any
        read of the marker.
        """
        pending = self._pending_question
        if pending is None or not isinstance(pending[0], PickPOI):
            return
        if not self._marker.sync():
            return
        if self._suggestion is not None:
            self._show_suggestion()

    @staticmethod
    def _button_label(
        request: "Request", response: bool, value: object = None
    ) -> Optional[str]:
        """The prompt button the suggested answer stands for."""
        return getattr(request, "positive" if response else "negative", None)

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
            self._sync_poi_frame()
            self._marker.place(value)
            return
        if isinstance(request, ConfirmDetection):
            if not isinstance(value, dict) or not all(
                isinstance(px, Point) for px in value.values()
            ):
                raise ValueError("a ConfirmDetection value must be {name: Point}")
            det_widget = self._ui.det_widget
            if det_widget is None:
                raise RuntimeError("No detection widget to move the features on.")
            det_widget.move_features(value)
            return
        raise ValueError(f"{type(request).__name__} answers carry no value.")

    def peek_live_answer(self) -> "Future":
        """What would the answer be if Yes were clicked right now? Any thread.

        Some questions are answered from live widget state — the POI marker the
        operator is dragging, the alignment area being resized — which the
        frozen request cannot show a remote agent. This reads that state on the
        GUI thread, without disturbing it: no overlay comes down, nothing is
        answered. Resolves to ``(question_id, value)`` — the question id pairs the value
        with a posting of the question, and value is None for question types
        with no live half (or no question at all, in which case question_id is None
        too).
        """
        from concurrent.futures import Future as _Future

        outcome: "_Future" = _Future()
        self._agent_peeked.emit(outcome)
        return outcome

    def _apply_agent_peek(self, outcome: "Future") -> None:
        """GUI thread. Complete ``outcome`` with (question_id, live value)."""
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
        request, _, question_id = pending
        if isinstance(request, PickPOI):
            self._sync_poi_frame()
            return question_id, self._marker.read()
        if isinstance(request, EditAlignmentArea):
            image_widget = self._ui.image_widget
            if image_widget is None:
                return question_id, None
            return question_id, deepcopy(image_widget.get_alignment_area())
        if isinstance(request, ConfirmDetection):
            det = getattr(self._ui.det_widget, "det", None)
            if det is None:
                return question_id, None
            return question_id, {f.name: deepcopy(f.px) for f in det.features}
        return question_id, None

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
        self._ui.front_tab(widget)

    def _clear_milling_config(self, request: ClearMillingConfig) -> None:
        """Clear the milling editor. Moved from handle_workflow_update."""
        self._milling_widget().clear()

    def _show_detection(self, request: ShowDetection) -> None:
        """Load the detection into the detection widget. The window fronts its
        tab when the question is asked."""
        det_widget = self._ui.det_widget
        if det_widget is None:
            # As for ConfirmDetection: the model already ran, so a missing
            # widget is a defect, not optional hardware.
            raise RuntimeError("No detection widget available to show the features.")
        det_widget.set_detected_features(request.detection)

    def _clear_spot_burn(self, request: ClearSpotBurn) -> None:
        """Clear the spot-burn widget's points and leave workflow mode, if
        there is one."""
        widget = self._ui.spot_burn_widget
        if widget is None:
            return
        widget.clear_points_layer()
        widget.set_workflow_mode(False)

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
        self,
        request: Request,
        future: "Future",
        msg: str,
        pos: str,
        neg: Optional[str],
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
        # The buttons the prompt shows: what a suggestion is said as.
        self._buttons = (pos, neg)
        self._clear_suggestion()
        self._emit_question_event(
            "prompt_raised",
            {
                "type": type(request).__name__,
                "message": msg,
                "positive": pos,
                "negative": neg,
                "question_id": self._question_seq,
            },
        )
        # Display state, not a handshake: the workflow does not poll this, but
        # the attention button, border, status bar and timeline pause read it.
        # The main window re-addresses it to a connected agent when the task
        # is designated so.
        self._ui.hold = Hold(
            kind=HoldKind.proposal, releases="answer the question on the Microscope tab"
        )
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

    def question_host(self) -> Optional[object]:
        """The widget the question now up is asked on, when it has a tab of its
        own rather than the shared prompt bar.

        The attention button uses this to go back to a question the operator
        navigated away from: a detection is corrected on its own tab, not on
        the Microscope tab the button otherwise goes to. Derived from the pending request rather than
        remembered, so it cannot drift out of step with what was actually
        raised; ``None`` means the prompt is where the button already goes.
        """
        if self._pending_question is None:
            return None
        request = self._pending_question[0]
        if isinstance(request, ConfirmDetection):
            return self._ui.det_widget
        return None

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

    def _follow_view_frames(self) -> None:
        """Hear the image widget's frame changes, once per widget."""
        image_widget = getattr(self._ui, "image_widget", None)
        signal = getattr(image_widget, "viewer_update_signal", None)
        if signal is None or self._view_frames_followed is image_widget:
            return
        signal.connect(self._sync_poi_frame)
        self._view_frames_followed = image_widget

    def _pick_poi(self, request: PickPOI, future: "Future") -> None:
        """Show a draggable POI marker on the FIB canvas; the click answers with it.

        The overlay lives on the main window's quad view. Without one (the
        standalone embedded window) there is nothing to pick on: the old flow
        no-oped and delivered None, so answer None immediately rather than
        putting up a prompt about a marker that is not there.
        """
        if not self._marker.show(request.initial, request.image):
            self._deliver(future, None)
            return
        self._follow_view_frames()
        self._park_question(request, future, request.message, "Continue", None)

    def abandon(self) -> None:
        """Drop whatever a finished run left behind. GUI thread, workflow end.

        The run is over when this is called — the workflow thread has exited —
        so a question still parked belongs to nobody: cancel it and take the
        prompt down.
        """
        pending, self._pending_question = self._pending_question, None
        if pending is not None:
            pending[1].cancel()
            self._ui.hold = None
            self._ui.workflow_status_signal.emit(WorkflowStatusEvent(message=""))
            self._emit_question_event(
                "prompt_cancelled",
                {"type": type(pending[0]).__name__, "question_id": pending[2]},
            )

    def answer_confirm(
        self,
        clicked_yes: bool,
        source: str = "operator",
        adjusted: bool = False,
        suggested_by: Optional[str] = None,
        value: Optional[object] = None,
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
        accepted what stood. ``suggested_by`` names who suggested the answer the
        operator took (FIB-1283): the answer is still the operator's.
        ``value`` is the answer's value, already placed in the widget.
        """
        pending = self._pending_question
        if pending is None:
            return False
        request, future = pending[0], pending[1]
        self._pending_question = None
        self._clear_suggestion()
        self._ui.hold = None
        if future.cancelled():
            # The asker aborted while the prompt stood. The click means nothing
            # beyond taking the stale prompt down — in particular it must not
            # start a mill for a question nobody is waiting on.
            self._ui.workflow_status_signal.emit(WorkflowStatusEvent(message=""))
            self._emit_question_event(
                "prompt_cancelled",
                {"type": type(request).__name__, "question_id": pending[2]},
            )
            return True
        logging.info(
            f"prompt answered: {type(request).__name__} "
            f"response={clicked_yes} by={source} adjusted={adjusted}"
        )
        answered_payload = {
            "type": type(request).__name__,
            "response": bool(clicked_yes),
            "question_id": pending[2],
        }
        if adjusted:
            answered_payload["adjusted"] = True
        if suggested_by:
            answered_payload["suggested_by"] = suggested_by
        try:
            answer = self._answer(request, clicked_yes, value)
        except Exception as exc:  # noqa: BLE001 - the caller owns the failure
            answer, failure = None, exc
        else:
            failure = None
            # What it decided, for a line about it (the agent card's Recent).
            decided = request.decided(answer)
            if decided:
                answered_payload["decided"] = decided
        self._emit_question_event("prompt_answered", answered_payload, actor=source)
        self._ui.workflow_status_signal.emit(WorkflowStatusEvent(message=""))
        if failure is not None:
            self._fail(future, failure)
        else:
            self._deliver(future, answer)
        return True

    def _answer(self, request: Request, clicked_yes: bool, value: object = None):
        """What the click means for this question's asker."""
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
            # Read the marker, convert against the frame it sits on (the one
            # the FIB view shows), and take the overlay down — the old flow's
            # second handshake plus its SELECTED_POI read-back, all in the
            # click's slot.
            self._sync_poi_frame()
            poi = self._marker.read()
            self._marker.remove()
            return poi
        return clicked_yes
