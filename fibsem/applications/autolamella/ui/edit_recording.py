"""The editors' edits, recorded once they settle: ``SETTLE_MS`` after the last
touch, or on :meth:`PendingEdits.flush`. The record itself is
``fibsem.applications.autolamella.edit_recording``'s."""

from __future__ import annotations

from typing import Any, Callable, Optional

from PyQt5.QtCore import QObject, QTimer

from fibsem.applications.autolamella import edit_recording

# As long as the lamella editor waits to save: the two settle together.
SETTLE_MS = 400


class PendingEdits(edit_recording.PendingEdits):
    """:class:`edit_recording.PendingEdits`, flushed by a timer."""

    def __init__(
        self,
        microscope: Callable[[], Any],
        via: str,
        parent: Optional[QObject] = None,
    ):
        super().__init__(microscope, via)
        self._timer = QTimer(parent)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.flush)

    def _touched(self) -> None:
        self._timer.start(SETTLE_MS)

    def flush(self) -> None:
        self._timer.stop()
        super().flush()
