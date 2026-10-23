"""The agent chip in the status bar: the agent's level, and the way to its card.


The agent is a second operator at the app, and its level is the one knob: Off,
Watch, Assist or Drive. The chip shows the level; clicking it opens the agent
card (`card.AgentCard`) above the chip: the level, Stop agent, what waits and
the way to the web Agent page, where the operator answers the agent.

At Assist the agent acts by asking. While its request waits, the chip
turns amber, "Agent asks". Only one request waits at a time, so the chip carries
no count. A suggestion on the pending proposal says "Agent suggests" in its
level's look, not amber (the run's Attention Required already is). Either one
asks the system for the operator's attention (a dock bounce or taskbar flash)
and opens nothing: the window never comes forward on its own.

While the agent acts on the microscope (the card's Now line), the chip pulses.

Like `WorkflowControls`, this only shows what it is told and says what was picked.
What a pick does -- setting the App Server's level -- is the window's.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional

from PyQt5.QtCore import QPoint, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import QApplication, QFrame, QPushButton, QVBoxLayout, QWidget

from fibsem.ai.agent.levels import AgentLevel
from fibsem.ui.agent.card import (
    AgentCard,
    last_heard_text,
)
from fibsem.ui.agent.style import (
    LEVEL_STYLES,
    LEVEL_TIPS,
    LEVEL_WORDS,
    REQUEST_CHIP,
    pulse_chip,
)
from fibsem.ui.tokens import BORDER_COLOR

__all__ = [
    "AgentLevelChip",
    "LEVEL_WORDS",
    "OFFERED_LEVELS",
    "chip_request_text",
    "last_heard_text",
]

# While the agent acts on the microscope the chip pulses: its fill fades and
# comes back once per cycle, and a cycle always runs to its end, so a quick
# action (an image) is one whole pulse rather than a flicker.
PULSE_MS = 1600
PULSE_STEP_MS = 50

# The card's width on the chip.
CARD_WIDTH = 500

_LEVELS = {
    level: (LEVEL_WORDS[level], LEVEL_STYLES[level], LEVEL_TIPS[level])
    for level in LEVEL_WORDS
}

# The levels the card offers, in order.
OFFERED_LEVELS = (AgentLevel.OFF, AgentLevel.WATCH, AgentLevel.ASSIST, AgentLevel.DRIVE)

_HELP = (
    ".\n\nClick for the agent's card. “Agent” is any client holding the App "
    "Server's agent token: the fibsem-mcp sidecar, FibsemClient or a script."
)


def _doing(tool: Any) -> str:
    """What a request's step is doing, in a word: "applying" read as writing a
    config whatever the step was."""
    tool = str(tool or "")
    if tool.startswith(("milling_", "spot_burn")):
        return "milling"
    if tool.startswith(("stage_", "item_move_", "grid_load", "grid_unload")):
        return "moving"
    if "acquire" in tool or tool.startswith(("fm_auto_focus", "beam_auto_")):
        return "imaging"
    return "running"


def chip_request_text(request: Dict[str, Any]) -> str:
    """The chip's word while a request waits or is being applied."""
    if request.get("status") == "applying":
        actions = request.get("actions", [])
        total = len(actions)
        step = min(len(request.get("results", [])) + 1, total)
        doing = _doing(actions[step - 1].get("tool")) if actions else "running"
        return f"Agent: {doing} {step}/{total}"
    return "Agent asks"


class _CardPopup(QFrame):
    """The card on its own, above the chip; a click elsewhere closes it."""

    def __init__(self, card: AgentCard, parent: QWidget) -> None:
        super().__init__(parent, Qt.Popup)  # type: ignore
        self.setObjectName("agentCardPopup")
        self.setStyleSheet(
            f"#agentCardPopup {{ border: 1px solid {BORDER_COLOR}; border-radius: 4px; }}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.addWidget(card)
        self.setFixedWidth(CARD_WIDTH)


class AgentLevelChip(QPushButton):
    """The agent's level, and the card it opens."""

    level_chosen = pyqtSignal(object)  # AgentLevel
    stop_clicked = pyqtSignal()
    event_log_clicked = pyqtSignal()
    chat_clicked = pyqtSignal()

    def __init__(
        self,
        last_heard: Callable[[], Optional[float]] = lambda: None,
        parent: Optional[QWidget] = None,
        agent_client: Callable[[], Optional[str]] = lambda: None,
        agent_session: Callable[[], Optional[str]] = lambda: None,
        agent_budget_low: Callable[[], bool] = lambda: False,
    ) -> None:
        super().__init__(parent)
        self._last_heard = last_heard
        self._agent_client = agent_client
        self._agent_session = agent_session
        self._agent_budget_low = agent_budget_low
        self._level: Optional[AgentLevel] = None
        self._request: Optional[Dict[str, Any]] = None
        self._suggestion: Optional[Dict[str, str]] = None
        self._asking: Optional[Dict[str, Any]] = None
        self.setCursor(Qt.PointingHandCursor)  # type: ignore
        self._acting = False
        self._pulse_ms = 0
        self._pulse = QTimer(self)
        self._pulse.setInterval(PULSE_STEP_MS)
        self._pulse.timeout.connect(self._on_pulse)

        self.card = AgentCard()
        self.card.level_chosen.connect(self._on_chosen)
        self.card.stop_clicked.connect(self._closing(self.stop_clicked))
        self.card.event_log_clicked.connect(self._closing(self.event_log_clicked))
        self.card.chat_clicked.connect(self._closing(self.chat_clicked))
        self.popup = _CardPopup(self.card, self)
        self.clicked.connect(self.toggle_card)

        # One width for every level and request state, so the status bar doesn't
        # shift when it changes.
        widths = []
        for level in _LEVELS:
            self.set_level(level)
            self.ensurePolished()
            widths.append(self.sizeHint().width())
        for text in ("Agent asks", "Agent suggests", "Agent: imaging 20/20"):
            self.setText(text)
            widths.append(self.sizeHint().width())
        self.setFixedWidth(max(widths))
        self.set_level(AgentLevel.WATCH)
        self.set_request(None)

    @property
    def level(self) -> Optional[AgentLevel]:
        return self._level

    def set_level(self, level: AgentLevel) -> None:
        """Show *level*. Does not emit: this is the window telling the chip."""
        level = AgentLevel(level)
        self._level = level
        self.card.set_level(level)
        self._paint()

    @property
    def request(self) -> Optional[Dict[str, Any]]:
        return self._request

    def set_request(self, request: Optional[Dict[str, Any]]) -> None:
        """Show the agent's request waiting or being applied, or None for none.
        Does not emit: this is the window telling the chip."""
        shown = (self._request or {}).get("request_id")
        arrived = (request or {}).get("status") == "pending" and (request or {}).get(
            "request_id"
        ) != shown
        self._request = dict(request) if request is not None else None
        self.card.set_request(self._request)
        self._paint()
        self._fit_popup()
        if arrived:
            self._ask_for_attention()

    @property
    def asking(self) -> Optional[Dict[str, Any]]:
        return self._asking

    def set_asking(self, asking: Optional[Dict[str, Any]]) -> None:
        """Show an agent on this PC asking to connect, or None for none.
        Does not emit: this is the window telling the chip."""
        arrived = asking is not None and (asking or {}).get("id") != (
            self._asking or {}
        ).get("id")
        self._asking = dict(asking) if asking is not None else None
        self.card.set_asking(self._asking)
        self._paint()
        self._fit_popup()
        if arrived:
            self._ask_for_attention()

    @property
    def suggestion(self) -> Optional[Dict[str, str]]:
        return self._suggestion

    def set_suggestion(self, suggestion: Optional[Dict[str, Any]]) -> None:
        """Show the agent's suggestion (see `AgentCard.set_suggestion`),
        or None for none. Does not emit: this is the window telling the chip."""
        arrived = suggestion is not None and dict(suggestion) != self._suggestion
        self._suggestion = dict(suggestion) if suggestion is not None else None
        self.card.set_suggestion(self._suggestion)
        self._paint()
        self._fit_popup()
        if arrived:
            self._ask_for_attention()

    def set_now(self, action: Optional[str]) -> None:
        """What the agent is doing on the microscope, for the card; the chip
        pulses while it does something."""
        self.card.set_now(action)
        self._fit_popup()
        self._acting = action is not None
        if self._acting and not self._pulse.isActive():
            self._pulse_ms = 0
            self._pulse.start()

    @property
    def pulsing(self) -> bool:
        return self._pulse.isActive()

    def _on_pulse(self) -> None:
        self._pulse_ms += PULSE_STEP_MS
        if self._pulse_ms >= PULSE_MS:
            self._pulse_ms = 0
            if not self._acting:
                self._pulse.stop()
        self._paint()

    def _paint(self) -> None:
        if self._level is None:
            return
        text, style, tooltip = _LEVELS[self._level]
        if self._asking is not None:
            text, style = "Agent asks", REQUEST_CHIP
            tooltip = "An agent asks to connect: decide on the Agent page"
        elif self._request is not None:
            text, style = chip_request_text(self._request), REQUEST_CHIP
            tooltip = "The agent is asking: decide on the Agent page"
        elif self._suggestion is not None:
            # Not amber: the run's Attention Required already is, for the same
            # proposal.
            text = "Agent suggests"
            tooltip = "The agent suggests a decision on the proposal: see it on the Agent page"
        elif self._pulse.isActive():
            # Full at the start and end of a cycle, faintest halfway.
            phase = 2 * math.pi * self._pulse_ms / PULSE_MS
            style = pulse_chip(0.5 + 0.5 * math.cos(phase))
        self.setText(text)
        self.setStyleSheet(style)
        self.setToolTip(tooltip + _HELP)

    # --- the card -------------------------------------------------------------

    def toggle_card(self) -> None:
        if self.popup.isVisible():
            self.popup.close()
            return
        self.refresh_last_heard()
        self._fit_popup()
        self.popup.show()
        self._fit_popup()

    def _ask_for_attention(self) -> None:
        """Something new waits for the operator: the system's attention
        request, which neither raises nor activates the window. Opening the
        card here took focus from the operator's browser."""
        if self.isVisible():
            QApplication.alert(self.window())

    def _fit_popup(self) -> None:
        """Above the chip, its right edge on the chip's, inside the screen."""
        popup = self.popup
        popup.adjustSize()
        # adjustSize guesses wrapped text's height at another width.
        layout = popup.layout()
        if layout is not None and layout.hasHeightForWidth():
            popup.resize(
                popup.width(),
                max(popup.height(), layout.totalHeightForWidth(popup.width())),
            )
        if not popup.isVisible():
            return
        popup.move(self._above_chip(popup))

    def _above_chip(self, window: QWidget) -> QPoint:
        """Where *window* stands above the chip, its right edge on the chip's,
        inside the screen."""
        corner = self.mapToGlobal(QPoint(self.width(), 0))
        x = corner.x() - window.width()
        y = corner.y() - window.height() - 4
        screen = self.screen().availableGeometry() if self.screen() else None
        if screen is not None:
            x = max(screen.left(), min(x, screen.right() - window.width()))
            y = max(screen.top(), y)
        return QPoint(x, y)

    def _closing(self, signal):
        def emit() -> None:
            self.popup.close()
            signal.emit()

        return emit

    def refresh_last_heard(self) -> None:
        try:
            seconds = self._last_heard()
        except Exception:
            seconds = None
        try:
            client = self._agent_client()
        except Exception:
            client = None
        self.card.set_last_heard(last_heard_text(seconds, client))
        try:
            session = self._agent_session()
            low = self._agent_budget_low()
        except Exception:
            session, low = None, False
        self.card.set_session(session, low)

    def _on_chosen(self, level: AgentLevel) -> None:
        if level is not self._level:
            self.level_chosen.emit(level)
