"""The editors' edits, recorded and saved once they settle: ``SETTLE_MS`` after
the last touch or save asked for, or on :meth:`PendingEdits.flush`. The record
itself is ``fibsem.applications.autolamella.edit_recording``'s; the save is
``config_edits.save_edits``."""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

from PyQt5.QtCore import QObject, QTimer

from fibsem.applications.autolamella import edit_recording
from fibsem.applications.autolamella.config_edits import save_edits

# Long enough to swallow a run of field edits, short enough that the window a
# crash could take an edit from stays small. Every place where waiting longer
# would risk one flushes instead.
SETTLE_MS = 400


class PendingEdits(edit_recording.PendingEdits):
    """:class:`edit_recording.PendingEdits`, flushed by a timer, with the
    experiment's write the edits wait for (FIB-683: a write is the whole
    experiment, so a burst of edits makes one)."""

    def __init__(
        self,
        microscope: Callable[[], Any],
        via: str,
        parent: Optional[QObject] = None,
    ):
        super().__init__(microscope, via)
        self._save: Optional[Tuple[Any, str]] = None
        self._timer = QTimer(parent)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.flush)

    def _touched(self) -> None:
        self._timer.start(SETTLE_MS)

    def save_later(self, experiment: Any, level: str) -> None:
        """Write *experiment* after an edit at *level* once the edits settle.
        Held, not looked up at flush, so a write asked for one experiment never
        lands on the next one loaded."""
        self._save = (experiment, level)
        self._timer.start(SETTLE_MS)

    def flush(self) -> None:
        self._timer.stop()
        super().flush()
        pending, self._save = self._save, None
        if pending is not None:
            save_edits(*pending)
