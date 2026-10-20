"""What the agent is doing on the microscope, said in the window (FIB-1375).

The agent acts through the App Server, and every action it takes holds control of
the microscope (FIB-1270) under a name an operator reads: "focus sweep", "move
stage relative". Who is in control is announced on ``microscope.control.changed``,
and this follows it:

* the status bar's line says "**Agent** focus sweep · 12s" while it holds control;
* an action still going after `ANNOUNCE_MS` gets a toast, and one more when it
  ends. Quick ones (an image, a stage move) get none: a toast for each of those
  would bury the window during a session. A long command the agent started (a
  mill, an FM overview) gets its end toast when its run ends, from the record,
  whether or not it held control to the end;
* the patterns the agent draws for a mill are drawn on the FIB view, labelled
  "Agent" in the agent's purple, until they are removed (the run ends, or the next setup). The milling
  service announces what is drawn (``patterns_changed``); they are the agent's
  when the agent is in control as they are drawn.

The agent card's "Now" line is `now` (`fibsem.ai.agent.now`); `now_changed`
says when it changes.

The card's Recent list is `recent`: what the agent did lately, from the session's
record (`follow_events`), as `recent.RecentActions` reads it (FIB-1414);
`recent_changed` says when it changes.

A run's or the operator's control is not the agent's and is left to their own
lines. A mill the agent starts and leaves running lets go of control at once; its
progress is the milling line's, and the milling panel's bar.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal
from superqt import ensure_main_thread

from fibsem.acting import AGENT
from fibsem.ai.agent.now import AgentNow
from fibsem.ai.agent.recent import RecentActions, RecentRow
from fibsem.structures import BeamType
from fibsem.ui import notification_service
from fibsem.ui.tokens import PURPLE_COLOR
from fibsem.util.durations import format_duration_rounded

if TYPE_CHECKING:
    from fibsem.control import InControl
    from fibsem.microscope import FibsemMicroscope
    from fibsem.ui.widgets.status_bar import FibsemStatusBar

# How long an agent's action runs before the window says so with a toast.
ANNOUNCE_MS = 3000

# How close together two ends of one action are the same end: its control let
# go and its run's ``command_ended``.
SAME_END_S = 5.0

# How a long command's run ended, in the toast.
_ENDED = {"done": "finished", "failed": "failed", "cancelled": "was stopped"}

# The FIB view's overlay for the agent's patterns, beside the milling panel's own.
OVERLAY_ID = "agent_milling"

# Where the agent's pattern legends go: clear of the view's toolbar and of the
# milling panel's own legend, both top-right.
AGENT_LEGEND = "lower left"


class DrawnStage:
    """The agent's patterns as the milling overlay draws a stage: a name for the
    legend and the shapes. No point it was placed at, so no crosshair."""

    pattern = None

    def __init__(self, patterns: List, name: str = "Agent") -> None:
        self._patterns = patterns
        self.name = name

    def define_patterns(self) -> List:
        return self._patterns


class AgentActivity(QObject):
    """Follows who is in control of the microscope and says what the agent does."""

    now_changed = pyqtSignal(object)  # the `now` text, or None
    recent_changed = pyqtSignal(object)  # the `recent` rows
    # A record of the session's, from whichever thread recorded it, to this one.
    _record_received = pyqtSignal(object)

    def __init__(
        self,
        status_bar: "FibsemStatusBar",
        view_controller: Optional[Callable[[], object]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._status_bar = status_bar
        # Asked when there is something to draw: the views are built after the bar.
        self._view_controller = view_controller or (lambda: None)
        self._control = None
        self._milling = None
        self._showing_patterns = False
        self._action: Optional[str] = None
        self._since = 0.0
        self._announced = False
        self._announce_timer = QTimer(self)
        self._announce_timer.setSingleShot(True)
        self._announce_timer.timeout.connect(self._announce)
        # The Now line's action and mill, as the chat page follows them.
        self._following = AgentNow()
        self._now: Optional[str] = None
        self._events = None
        self._unsubscribe: Optional[Callable[[], None]] = None
        self._recent = RecentActions()
        self._seen = 0
        # The last action an end toast was shown for, and when.
        self._ended: Tuple[Optional[str], float] = (None, 0.0)
        self._record_received.connect(self._on_record, Qt.QueuedConnection)  # type: ignore

    @property
    def action(self) -> Optional[str]:
        """The agent's action while it is in control, else None."""
        return self._action

    @property
    def now(self) -> Optional[str]:
        """What the agent is doing, for the card's Now line: its mill and how far
        it is while it mills, else its action; None when it does nothing."""
        return self._now

    @property
    def recent(self) -> List[RecentRow]:
        """What the agent did lately, oldest first."""
        return self._recent.rows

    def follow_events(self, buffer) -> None:
        """Read the agent's actions from *buffer*, the session's event stream
        (``EventBuffer``), from what it already holds on; None stops."""
        if buffer is self._events:
            return
        if self._unsubscribe is not None:
            try:
                self._unsubscribe()
            except Exception:
                pass
            self._unsubscribe = None
        self._events = buffer
        self._recent.clear()
        self._seen = 0
        if buffer is not None:
            # Subscribed first, then caught up: a record in between is read
            # once, by its sequence number.
            self._unsubscribe = buffer.subscribe(self._record_received.emit)
            for record in buffer.events_since(0).get("events", []):
                self._take(record)
        self.recent_changed.emit(self._recent.rows)

    def _on_record(self, record: object) -> None:
        if not isinstance(record, dict):
            return
        fresh = (record.get("seq") or 0) > self._seen
        if self._take(record):
            self.recent_changed.emit(self._recent.rows)
        if fresh and record.get("kind") == "command_ended":
            self._command_ended(record)

    def _command_ended(self, record: dict) -> None:
        """The agent's long command ended: say so, once."""
        if record.get("actor") != AGENT:
            return
        p = record.get("payload") or {}
        action = p.get("action") or f"{p.get('service')} {p.get('command')}"
        ended = _ENDED.get(str(p.get("state")), "ended")
        text = f"Agent: {action} {ended}"
        if p.get("error"):
            text += f": {p['error']}"
        self._toast_end(action, text, "error" if ended == "failed" else "info")

    def _toast_end(self, action: str, text: str, kind: str = "info") -> None:
        last, at = self._ended
        now = time.monotonic()
        if last == action and now - at < SAME_END_S:
            return
        self._ended = (action, now)
        notification_service.show_toast(text, kind)

    def _take(self, record: dict) -> bool:
        seq = record.get("seq") or 0
        if seq and seq <= self._seen:
            return False
        self._seen = max(self._seen, seq)
        return self._recent.add(record)

    def set_microscope(self, microscope: Optional["FibsemMicroscope"]) -> None:
        """Follow *microscope*'s control and milling; None lets the last one go."""
        for signal, slot in (
            (getattr(self._control, "changed", None), self._on_control_changed),
            (
                getattr(self._milling, "patterns_changed", None),
                self._on_patterns_changed,
            ),
            (
                getattr(getattr(self._milling, "progress", None), "changed", None),
                self._on_milling_progress,
            ),
        ):
            if signal is None:
                continue
            try:
                signal.disconnect(slot)
            except Exception:
                pass  # already gone with its microscope
        self._control = getattr(microscope, "control", None)
        self._milling = getattr(microscope, "milling", None)
        if not hasattr(self._milling, "patterns_changed"):
            self._milling = None
        self._set_action(None)
        self._show_patterns([])
        self._following.clear()
        self._say_now()
        if self._control is not None:
            self._control.changed.connect(self._on_control_changed)
            self._on_control_changed(self._control.current)
        if self._milling is not None:
            self._milling.patterns_changed.connect(self._on_patterns_changed)
            progress = getattr(self._milling, "progress", None)
            if hasattr(progress, "changed"):
                progress.changed.connect(self._on_milling_progress)

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
            self._toast_end(self._action, f"Agent: {self._action} ended after {took}")
        self._action = action
        self._following.action = action
        self._since = time.monotonic()
        self._announced = False
        self._announce_timer.stop()
        if action is not None:
            self._announce_timer.start(ANNOUNCE_MS)
        self._status_bar.set_agent_action(action)
        self._say_now()

    @ensure_main_thread
    def _on_milling_progress(self, payload: object) -> None:
        self._following.take_progress(payload)
        self._say_now()

    def _say_now(self) -> None:
        now = self._following.text
        if now != self._now:
            self._now = now
            self.now_changed.emit(now)

    def _on_patterns_changed(self, patterns: List) -> None:
        # On the drawing thread, inside the take: whose they are is read now, not
        # after the hop, when the agent may have let go.
        current = self._control.current if self._control is not None else None
        if patterns and (current is None or current.by != AGENT):
            patterns = None  # a run's or the operator's: the milling panel has them
        self._show_patterns(patterns)

    @ensure_main_thread
    def _show_patterns(self, patterns: Optional[List]) -> None:
        """Draw the agent's *patterns* on the FIB view; empty removes them, None
        (someone else's) leaves the view as it is."""
        if patterns is None:
            return
        controller = self._view_controller()
        if controller is None:
            return
        if not patterns:
            if self._showing_patterns:
                controller.remove_overlay(BeamType.ION, OVERLAY_ID)
                self._showing_patterns = False
            return
        from fibsem.ui.widgets.canvas.canvas_state import MillingSpec

        controller.set_overlay(
            BeamType.ION,
            MillingSpec(
                id=OVERLAY_ID,
                stages=[DrawnStage(patterns)],
                colour=PURPLE_COLOR,
                legend_loc=AGENT_LEGEND,
            ),
        )
        self._showing_patterns = True

    @property
    def showing_patterns(self) -> bool:
        """Whether the agent's patterns are drawn on the FIB view."""
        return self._showing_patterns

    def _announce(self) -> None:
        if self._action is None:
            return
        self._announced = True
        notification_service.show_toast(f"Agent: {self._action}", "info")
