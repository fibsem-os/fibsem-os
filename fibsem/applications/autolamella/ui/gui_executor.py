"""The session's executor in the Qt app: a call from any thread, run on the GUI
thread as whoever made it.

Edits to the experiment run on the GUI thread because its widgets listen to it.
Built on the GUI thread: the queued signal delivers to the thread this lives on.
"""

import functools
from concurrent.futures import Future

from PyQt5.QtCore import QObject, pyqtSignal

from fibsem.acting import carried

__all__ = ["GuiExecutor"]


def _complete(call, outcome) -> None:
    """Complete ``outcome`` with ``call()``'s result, or its exception."""
    try:
        result = call()
    except Exception as exc:  # noqa: BLE001 - the requester owns the failure
        outcome.set_exception(exc)
        return
    outcome.set_result(result)


class GuiExecutor(QObject):
    # The work, bound to its arguments and its Future, and marked as whoever
    # asked (FIB-1276) -- the mark stays on the requesting thread.
    _call = pyqtSignal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        # A bound method, not a lambda: the slot must run on this object's thread.
        self._call.connect(self._run)

    def __call__(self, call) -> Future:
        """``call()`` on the GUI thread as whoever called this; any thread.
        Resolves to its result, or its exception."""
        outcome: Future = Future()
        self._call.emit(carried(functools.partial(_complete, call, outcome)))
        return outcome

    def _run(self, call) -> None:
        """GUI thread."""
        call()
