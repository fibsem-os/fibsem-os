"""The run, as the Qt app hears it: the runner's signals carried over to the GUI
thread, and who holds the run.

The runner reports from the run's thread; Qt signals are thread-safe, so each
report is re-emitted here and delivered on the GUI thread. Built on the GUI
thread for that reason.
"""

from typing import TYPE_CHECKING, Optional

from PyQt5.QtCore import QObject, pyqtSignal

if TYPE_CHECKING:
    from fibsem.workflows.runner import WorkflowRunner
    from fibsem.workflows.status import Hold, WorkflowStatusEvent

__all__ = ["RunRelay"]


class RunRelay(QObject):
    # Everything the workflow says without needing an answer, as a
    # WorkflowStatusEvent. Instructions are typed requests to the QtResponder
    # (fibsem/workflows/interaction.py); proposals are on the record.
    status = pyqtSignal(object)
    # Its own signal: a queue edit is not a step in the task lifecycle and must
    # not disturb the interaction UI.
    queue_changed = pyqtSignal(dict)
    step = pyqtSignal(str)  # a human-readable step label
    finished = pyqtSignal(bool)  # cancelled
    # A stop was asked of the running workflow, by anyone: the operator's Stop,
    # the agent's workflow_stop (from the App Server's thread), closing the
    # window. The window shows "stopping" until the run has unwound.
    stopping = pyqtSignal()
    notice = pyqtSignal(str, str)  # (message, notification_type), from a hook

    def __init__(self, runner: "WorkflowRunner", parent=None) -> None:
        super().__init__(parent)
        # Display state, not a handshake: the run is held -- a question is up
        # for a click, or the run is parked on decisions -- and by whom. The
        # attention button, border, status bar and timeline pause read it.
        # Every workflow interaction is a typed request on its own future
        # (fibsem/workflows/interaction.py).
        self.hold: Optional["Hold"] = None
        # A run is active but nothing is executing -- today only during a
        # scheduled-start wait. Set from the worker thread, read by the border.
        self.pending: bool = False

        runner.finished.connect(self._on_finished)
        runner.notice.connect(self._on_notice)
        runner.queue_changed.connect(self._on_queue_changed)
        runner.status.connect(self._on_status)
        runner.held.connect(self._on_held)
        runner.pending.connect(self._on_pending)
        runner.step.connect(self._on_step)

    def _on_finished(self, cancelled: bool) -> None:
        self.finished.emit(cancelled)

    def _on_notice(self, message: str, notification_type: str) -> None:
        self.notice.emit(message, notification_type)

    def _on_queue_changed(self, info: dict) -> None:
        self.queue_changed.emit(info)

    def _on_status(self, event: "WorkflowStatusEvent") -> None:
        self.status.emit(event)

    def _on_held(self, hold: Optional["Hold"]) -> None:
        """Kept here for the chrome, which the status line that follows redraws."""
        self.hold = hold

    def _on_pending(self, pending: bool) -> None:
        self.pending = pending

    def _on_step(self, step: str) -> None:
        self.step.emit(step)
