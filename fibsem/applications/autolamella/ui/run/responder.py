"""The Qt implementation of the workflow's ``Responder`` protocol.

``QtResponder`` is *given* the window's panel rather than being the window: the
workflow layer holds an object with one method, and the widgets stay on this side
of the seam.

``submit`` is called on the workflow thread. It marshals the instruction to the GUI
thread through a queued signal, dispatches on its type, and completes the future:
``set_result`` with the acknowledgement, ``set_exception`` if acting on it failed.
The failure re-raises on the workflow thread inside ``wait_for`` rather than
escaping a queued slot and aborting the process (FIB-329). An unhandled type
completes the future with a ``TypeError`` rather than leaving the caller to its
timeout.
"""

from concurrent.futures import InvalidStateError
from typing import TYPE_CHECKING, Callable, Dict, Type

from PyQt5.QtCore import QObject, pyqtSignal

from fibsem.structures import BeamType
from fibsem.workflows.interaction import (
    ClearMillingConfig,
    ClearSpotBurn,
    Request,
    SetFluorescenceChannels,
    SetImages,
    SetMillingConfig,
    ShowDetection,
)

if TYPE_CHECKING:
    from concurrent.futures import Future

    from fibsem.applications.autolamella.ui.panel.panel import MicroscopePanel

__all__ = ["QtResponder"]


class QtResponder(QObject):
    """Acts on workflow instructions by driving the panel's widgets, on the GUI thread."""

    _submitted = pyqtSignal(object, object)  # (Request, Future)

    def __init__(self, panel: "MicroscopePanel"):
        super().__init__()
        self._panel = panel
        self._handlers: Dict[Type[Request], Callable] = {
            SetImages: self._set_images,
            SetMillingConfig: self._set_milling_config,
            ClearMillingConfig: self._clear_milling_config,
            ClearSpotBurn: self._clear_spot_burn,
            ShowDetection: self._show_detection,
            SetFluorescenceChannels: self._set_fluorescence_channels,
        }
        self._submitted.connect(self._dispatch)

    def submit(self, request: "Request", future: "Future") -> None:
        """Hand ``request`` to the GUI thread; never blocks. Any thread."""
        self._submitted.emit(request, future)

    def _dispatch(self, request: "Request", future: "Future") -> None:
        """GUI thread. Complete the future, whatever happens in the handler."""
        handler = self._handlers.get(type(request))
        try:
            if handler is None:
                raise TypeError(
                    f"QtResponder has no handler for {type(request).__name__}"
                )
            self._deliver(future, handler(request))
        except Exception as exc:  # noqa: BLE001 - the caller owns the failure
            self._fail(future, exc)

    # An abandoned ask cancels its future on the workflow thread, so the GUI
    # side can find it already cancelled. set_result on a cancelled future is
    # InvalidStateError inside a Qt slot, i.e. a process abort; these two
    # tolerate it instead.

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
        """Show the acquisition images."""
        image_widget = self._panel.image_widget
        if image_widget is None:
            # Surfaces in the workflow thread as this instruction's failure.
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
        widget = self._panel.milling_task_config_widget
        if widget is None:
            # Surfaces in the workflow thread as the instruction's failure.
            raise RuntimeError("No milling task config widget available.")
        return widget

    def _set_milling_config(self, request: SetMillingConfig) -> None:
        """Load the config into the milling editor and bring its tab forward."""
        widget = self._milling_widget()
        widget.update_from_settings(request.config)
        widget.setEnabled(True)
        self._panel.front_tab(widget)

    def _clear_milling_config(self, request: ClearMillingConfig) -> None:
        """Clear the milling editor."""
        self._milling_widget().clear()

    def _show_detection(self, request: ShowDetection) -> None:
        """Load the detection into the detection widget. The window fronts its
        tab when the proposal is asked."""
        det_widget = self._panel.det_widget
        if det_widget is None:
            # The model already ran, so a missing widget is a defect, not
            # optional hardware.
            raise RuntimeError("No detection widget available to show the features.")
        det_widget.set_detected_features(request.detection)

    def _clear_spot_burn(self, request: ClearSpotBurn) -> None:
        """Clear the spot-burn widget's points and leave workflow mode, if
        there is one."""
        widget = self._panel.spot_burn_widget
        if widget is None:
            return
        widget.clear_points_layer()
        widget.set_workflow_mode(False)

    # Unlike the image and milling widgets, whose absence is a defect, the FM and
    # spot-burn widgets legitimately do not exist on systems without the hardware —
    # and a workflow must not fail over that: these acknowledge the instruction
    # and skip.

    def _set_fluorescence_channels(self, request: SetFluorescenceChannels) -> None:
        """Load the channel settings into the fluorescence widget, if there is one."""
        widget = self._panel.fm_control_widget
        if widget is None:
            return
        widget.channelSettingsWidget.channel_settings = request.channels
