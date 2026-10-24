"""What the agent is doing on the microscope, said in the window.

The agent acts through the App Server, and every action it takes holds control of
the microscope under a name an operator reads: "focus sweep", "move
stage relative". Who is in control is announced on ``microscope.control.changed``,
and this follows it:

* the status bar's line says "**Agent** focus sweep · 12s" while it holds control;
* the patterns the agent draws for a mill are drawn on the FIB view, labelled
  "Agent" in the agent's purple, until they are removed (the run ends, or the next setup). The milling
  service announces what is drawn (``patterns_changed``); they are the agent's
  when the agent is in control as they are drawn.

The agent card's "Now" line is `now` (`fibsem.ai.agent.now`); `now_changed`
says when it changes.

The window tells the operator nothing else about the agent: what it did, and
what it asks, are on the web Agent page.

A run's or the operator's control is not the agent's and is left to their own
lines. A mill the agent starts and leaves running lets go of control at once; its
progress is the milling line's, and the milling panel's bar.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Optional

from PyQt5.QtCore import QObject, pyqtSignal
from superqt import ensure_main_thread

from fibsem.acting import AGENT
from fibsem.ai.agent.now import AgentNow
from fibsem.structures.settings import BeamType
from fibsem.theme.tokens import PURPLE_COLOR

if TYPE_CHECKING:
    from fibsem.control import InControl
    from fibsem.microscope import FibsemMicroscope
    from fibsem.ui.widgets.status_bar import FibsemStatusBar

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
        # The Now line's action and mill, as the web app follows them.
        self._following = AgentNow()
        self._now: Optional[str] = None

    @property
    def action(self) -> Optional[str]:
        """The agent's action while it is in control, else None."""
        return self._action

    @property
    def now(self) -> Optional[str]:
        """What the agent is doing, for the card's Now line: its mill and how far
        it is while it mills, else its action; None when it does nothing."""
        return self._now

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
        self._action = action
        self._following.action = action
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
