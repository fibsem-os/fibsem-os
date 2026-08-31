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

from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple, Type

from PyQt5.QtCore import QObject, pyqtSignal

from fibsem.applications.autolamella.workflows.interaction import (
    ClearMillingConfig,
    ClearSpotBurn,
    Confirm,
    ConfirmDetection,
    Request,
    SetFluorescenceChannels,
    SetImages,
    SetMillingConfig,
    SetSpotBurnSettings,
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
            SetSpotBurnSettings: self._set_spot_burn_settings,
            ClearSpotBurn: self._clear_spot_burn,
        }
        # Deferred: the handler shows the prompt and someone else completes the
        # future later. Kept out of _handlers so _dispatch cannot complete these
        # with the handler's return value.
        self._deferred_handlers: Dict[Type[Request], Callable] = {
            Confirm: self._confirm,
            ConfirmDetection: self._confirm_detection,
        }
        self._pending_question: Optional[Tuple[Request, "Future"]] = None
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
                future.set_exception(exc)
            return
        handler = self._handlers.get(type(request))
        try:
            if handler is None:
                raise TypeError(
                    f"QtResponder has no handler for {type(request).__name__}"
                )
            future.set_result(handler(request))
        except Exception as exc:  # noqa: BLE001 - the caller owns the failure
            future.set_exception(exc)

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

    def _set_spot_burn_settings(self, request: SetSpotBurnSettings) -> None:
        """Load the settings into the spot-burn widget, if there is one."""
        widget = self._ui.spot_burn_widget
        if widget is None:
            return
        widget.set_settings(request.settings)

    def _clear_spot_burn(self, request: ClearSpotBurn) -> None:
        """Clear the spot-burn overlay and leave workflow mode, if there is one."""
        widget = self._ui.spot_burn_widget
        if widget is None:
            return
        widget.clear_points_layer()
        widget.set_workflow_mode(False)

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
        self._ui.workflow_update_signal.emit({"msg": ""})
        try:
            future.set_result(self._answer(request, clicked_yes))
        except Exception as exc:  # noqa: BLE001 - the caller owns the failure
            future.set_exception(exc)
        return True

    def _answer(self, request: Request, clicked_yes: bool):
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
        return clicked_yes
