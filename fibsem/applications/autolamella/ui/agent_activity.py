"""What the agent is doing on the microscope, said in the window (FIB-1375).

The agent acts through the App Server, and every action it takes holds control of
the microscope (FIB-1270) under a name an operator reads: "focus sweep", "move
stage relative". Who is in control is announced on ``microscope.control.changed``,
so that is all this listens to:

* the status bar's line says "**Agent** focus sweep · 12s" while it holds control;
* an action still going after `ANNOUNCE_MS` gets a toast, and one more when it
  ends. Quick ones (an image, a stage move) get none: a toast for each of those
  would bury the window during a session.

A run's or the operator's control is not the agent's and is left to their own
lines. A mill the agent starts and leaves running lets go of control at once; its
progress is the milling line's.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

from PyQt5.QtCore import QObject, QTimer
from superqt import ensure_main_thread

from fibsem.acting import AGENT
from fibsem.ui import notification_service
from fibsem.util.durations import format_duration_rounded

if TYPE_CHECKING:
    from fibsem.control import InControl
    from fibsem.microscope import FibsemMicroscope
    from fibsem.ui.widgets.status_bar import FibsemStatusBar

# How long an agent's action runs before the window says so with a toast.
ANNOUNCE_MS = 3000


class AgentActivity(QObject):
    """Follows who is in control of the microscope and says what the agent does."""

    def __init__(self, status_bar: "FibsemStatusBar", parent=None) -> None:
        super().__init__(parent)
        self._status_bar = status_bar
        self._control = None
        self._action: Optional[str] = None
        self._since = 0.0
        self._announced = False
        self._announce_timer = QTimer(self)
        self._announce_timer.setSingleShot(True)
        self._announce_timer.timeout.connect(self._announce)

    @property
    def action(self) -> Optional[str]:
        """The agent's action while it is in control, else None."""
        return self._action

    def set_microscope(self, microscope: Optional["FibsemMicroscope"]) -> None:
        """Follow *microscope*'s control; None lets the last one go."""
        if self._control is not None:
            try:
                self._control.changed.disconnect(self._on_control_changed)
            except Exception:
                pass  # already gone with its microscope
        self._control = getattr(microscope, "control", None)
        self._set_action(None)
        if self._control is not None:
            self._control.changed.connect(self._on_control_changed)
            self._on_control_changed(self._control.current)

    @ensure_main_thread
    def _on_control_changed(self, in_control: Optional["InControl"]) -> None:
        # Emitted on whichever thread took or let go: a server worker, the run's.
        if in_control is not None and in_control.by == AGENT:
            self._set_action(in_control.action)
        else:
            self._set_action(None)

    def _set_action(self, action: Optional[str]) -> None:
        if action == self._action:
            return
        if self._action is not None and self._announced:
            took = format_duration_rounded(time.monotonic() - self._since)
            notification_service.show_toast(
                f"Agent: {self._action} ended after {took}", "info"
            )
        self._action = action
        self._since = time.monotonic()
        self._announced = False
        self._announce_timer.stop()
        if action is not None:
            self._announce_timer.start(ANNOUNCE_MS)
        self._status_bar.set_agent_action(action)

    def _announce(self) -> None:
        if self._action is None:
            return
        self._announced = True
        notification_service.show_toast(f"Agent: {self._action}", "info")
