"""The agent chip in the status bar: the agent's level, and the way to its card
(FIB-1035, FIB-1404).

The agent is a second operator at the app, and its level is the one knob: Off,
Watch, Assist or Drive. The chip shows the level; clicking it opens the agent
card (`card.AgentCard`) on its own, above the chip: what is waiting for the
operator, and the controls (the level, Stop agent, who the agent is, the event
log). No menu and no dialog.

At Assist the agent acts by asking (FIB-1283). While its request waits, the chip
turns amber, "Agent asks", and the card shows the request. Only one request waits
at a time, so the chip carries no count. A suggestion on the pending proposal
says "Agent suggests" in its level's look, not amber (the run's Attention
Required already is), and the card shows it with Approve. Either one asks the
system for the operator's attention (a dock bounce or taskbar flash) and opens
nothing: the window never comes forward on its own (FIB-1501).

While the agent acts on the microscope (the card's Now line), the chip pulses.

The card can float (FIB-1404): its button moves it into a window of its own, a
tool window over the app (never over other apps) that the operator moves and
sizes, narrow enough and it stacks. Closing that window, or the button again,
puts the card back on the chip. The window keeps its place for the session, and
its pin, off at first, keeps it over every window for the session (FIB-1418): a
waiting request stays in sight while the operator is in another app.

Like `WorkflowControls`, this only shows what it is told and says what was picked.
What a pick does -- setting the App Server's level -- is the window's.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional

from PyQt5.QtCore import QPoint, QRect, Qt, QTimer, pyqtSignal
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
from fibsem.ui.tokens import (
    BORDER_COLOR,
    PURPLE_COLOR,
    TEXT_MUTED_COLOR,
    TEXT_STRONG_COLOR,
    WAITING_COLOR,
)

__all__ = [
    "AgentLevelChip",
    "LEVEL_WORDS",
    "OFFERED_LEVELS",
    "chip_request_text",
    "last_heard_text",
    "request_outcome",
]

# While the agent acts on the microscope the chip pulses: its fill fades and
# comes back once per cycle, and a cycle always runs to its end, so a quick
# action (an image) is one whole pulse rather than a flicker.
PULSE_MS = 1600
PULSE_STEP_MS = 50

# The card's width on the chip: wide while something waits for the operator (the
# actions beside the reason and the decision), narrow otherwise.
WIDE_WIDTH = 780
NARROW_WIDTH = 560

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
    config whatever the step was (FIB-1492)."""
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


def request_outcome(request: Dict[str, Any]) -> Optional[tuple]:
    """``(message, kind)`` for a toast saying how a request ended, or None for
    one still open or rejected (the operator rejected it: nothing to tell them).
    Kind is a toast's: success, error or info."""
    status = request.get("status")
    actions = request.get("actions", [])
    results = request.get("results", [])
    total = len(actions)
    first = (actions[0].get("change") or actions[0].get("tool", "")) if actions else ""
    first = first.split("\n", 1)[0]
    more = f", and {total - 1} more" if total > 1 else ""
    if status == "applied":
        return f"Agent's request applied: {first}{more}", "success"
    if status == "failed":
        refused = next((r for r in results if not r.get("ok")), None)
        if refused is None:
            return f"Agent's request failed: {first}{more}", "error"
        error = refused.get("error")
        message = error.get("message") if isinstance(error, dict) else error
        message = str(message or refused.get("status_code") or "refused")
        at = refused.get("index", 0) + 1
        rest = " Nothing else ran." if at < total else ""
        return (
            f"Agent's request failed at {at} of {total}: "
            f"{message[:1].lower() + message[1:]}{'' if message.endswith('.') else '.'}"
            f"{rest}",
            "error",
        )
    if status == "cancelled":
        return f"Agent's request cancelled: {first}{more}", "info"
    return None


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
        self.setFixedWidth(NARROW_WIDTH)


# The floating card over the app's window only: a tool window. Pinned, over
# every window: a plain window with the on-top hint. On macOS a tool window is an
# NSPanel, whose floating level holds only while its app is active, so pinned it
# went behind the next app (FIB-1427).
_OVER_APP = Qt.Tool  # type: ignore
_OVER_ALL = Qt.Window | Qt.WindowStaysOnTopHint  # type: ignore


class FloatingPanel(QWidget):
    """The card in a window of its own, over the app's window only (a tool
    window); closing it puts the card back on the chip."""

    closed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent, _OVER_APP)
        self.setWindowTitle("Agent")
        self.setObjectName("agentPanel")
        # Opened by a request arriving, not only by a click: it never takes the
        # keyboard from what the operator is typing in.
        self.setAttribute(Qt.WA_ShowWithoutActivating)  # type: ignore
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

    def set_keep_on_top(self, on: bool) -> None:
        """Over every window, or over the app's only."""
        if bool(self.windowFlags() & Qt.WindowStaysOnTopHint) == on:  # type: ignore
            return
        shown = self.isVisible()
        geometry = self.geometry()
        self.setWindowFlags(_OVER_ALL if on else _OVER_APP)
        if shown:
            # A window's flags change only as it is shown again.
            self.setGeometry(geometry)
            self.show()

    def hold(self, card: AgentCard) -> None:
        self._layout.addWidget(card)
        card.show()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt's name
        super().closeEvent(event)
        self.closed.emit()


class AgentLevelChip(QPushButton):
    """The agent's level, and the card it opens."""

    level_chosen = pyqtSignal(object)  # AgentLevel
    stop_clicked = pyqtSignal()
    connect_clicked = pyqtSignal()
    event_log_clicked = pyqtSignal()
    chat_clicked = pyqtSignal()
    approve_clicked = pyqtSignal(str)  # request id
    reject_clicked = pyqtSignal(str, str)  # request id, the operator's reason
    suggestion_approved = pyqtSignal()  # the suggestion shown
    suggestion_declined = pyqtSignal(str)  # the suggestion shown, the operator's why
    message_sent = pyqtSignal(str)  # the operator's message to the agent

    def __init__(
        self,
        last_heard: Callable[[], Optional[float]] = lambda: None,
        parent: Optional[QWidget] = None,
        agent_client: Callable[[], Optional[str]] = lambda: None,
    ) -> None:
        super().__init__(parent)
        self._last_heard = last_heard
        self._agent_client = agent_client
        self._level: Optional[AgentLevel] = None
        self._request: Optional[Dict[str, Any]] = None
        self._suggestion: Optional[Dict[str, str]] = None
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
        self.card.connect_clicked.connect(self._closing(self.connect_clicked))
        self.card.approve_clicked.connect(self._on_approve)
        self.card.reject_clicked.connect(self._on_reject)
        self.card.suggestion_approved.connect(self._closing(self.suggestion_approved))
        self.card.suggestion_declined.connect(self._on_decline)
        # The card stays open: the operator may write again.
        self.card.message_sent.connect(self.message_sent)
        self.card.float_clicked.connect(self._on_float_clicked)
        self.card.keep_on_top_toggled.connect(self.set_keep_on_top)
        self._keep_on_top = False
        self.popup = _CardPopup(self.card, self)
        # The card's own window, made the first time it floats (the chip has no
        # window to sit over until it is in the status bar).
        self.panel: Optional[FloatingPanel] = None
        self._panel_geometry: Optional[QRect] = None
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

    def set_recent(self, rows) -> None:
        """What the agent did lately, for the card's Recent."""
        self.card.set_recent(rows)
        self._fit_popup()

    @property
    def keep_on_top(self) -> bool:
        return self._keep_on_top

    def set_keep_on_top(self, on: bool) -> None:
        """Keep the floating card over every window, for the session."""
        self._keep_on_top = bool(on)
        self.card.set_keep_on_top(self._keep_on_top)
        if self.panel is not None:
            self.panel.set_keep_on_top(self._keep_on_top)

    def _paint(self) -> None:
        if self._level is None:
            return
        text, style, tooltip = _LEVELS[self._level]
        if self._request is not None:
            text, style = chip_request_text(self._request), REQUEST_CHIP
            tooltip = "The agent is asking: click to see what, and decide"
        elif self._suggestion is not None:
            # Not amber: the run's Attention Required already is, for the same
            # proposal; the card shows the suggestion when opened.
            text = "Agent suggests"
            tooltip = "The agent suggests a decision on the proposal: click to see it"
        elif self._pulse.isActive():
            # Full at the start and end of a cycle, faintest halfway.
            phase = 2 * math.pi * self._pulse_ms / PULSE_MS
            style = pulse_chip(0.5 + 0.5 * math.cos(phase))
        self.setText(text)
        self.setStyleSheet(style)
        self.setToolTip(tooltip + _HELP)

    # --- the card -------------------------------------------------------------

    @property
    def floating(self) -> bool:
        return self.panel is not None and self.panel.isVisible()

    def toggle_card(self) -> None:
        if self.floating:
            self.panel.raise_()
            self.panel.activateWindow()
            return
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
        card here took focus from the operator's browser (FIB-1501)."""
        if self.isVisible():
            QApplication.alert(self.window())

    def _on_float_clicked(self) -> None:
        if self.floating:
            self.dock_card()
        else:
            self.float_card()

    def float_card(self) -> None:
        """The card into its own window, where it was last, else where the
        popup stood."""
        if self.floating:
            return
        start = self.popup.geometry() if self.popup.isVisible() else None
        self.popup.close()
        if self.panel is None:
            self.panel = FloatingPanel(self.window())
            self.panel.closed.connect(self.dock_card)
            self.panel.set_keep_on_top(self._keep_on_top)
        self.refresh_last_heard()
        self.panel.hold(self.card)
        self.card.set_floating(True)
        if self._panel_geometry is not None:
            self.panel.setGeometry(self._panel_geometry)
        else:
            self.panel.resize(self._card_width(), self.card.sizeHint().height())
            self.panel.move(
                start.topLeft() if start is not None else self._above_chip(self.panel)
            )
        self.panel.show()
        self.panel.raise_()

    def dock_card(self) -> None:
        """The card back on the chip; the window keeps its place for next time."""
        panel = self.panel
        if panel is None or self.card.parent() is self.popup:
            return
        self._panel_geometry = panel.geometry()
        self.card.set_floating(False)
        self.popup.layout().addWidget(self.card)
        if panel.isVisible():
            panel.hide()

    def _card_width(self) -> int:
        """Wide while a request or a suggestion waits, else narrow (no narrower
        than the stacked card needs)."""
        if self._request is not None or self._suggestion is not None:
            return WIDE_WIDTH
        return max(NARROW_WIDTH, self.card.minimumSizeHint().width() + 2)

    def _fit_popup(self) -> None:
        """Above the chip, its right edge on the chip's, inside the screen."""
        if self.floating:
            return
        popup = self.popup
        popup.setFixedWidth(self._card_width())
        popup.adjustSize()
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

    def _on_approve(self, request_id: str) -> None:
        self.popup.close()
        self.approve_clicked.emit(request_id)

    def _on_decline(self, reason: str) -> None:
        self.popup.close()
        self.suggestion_declined.emit(reason)

    def _on_reject(self, request_id: str, reason: str) -> None:
        self.popup.close()
        self.reject_clicked.emit(request_id, reason)

    def refresh_last_heard(self) -> None:
        try:
            seconds = self._last_heard()
        except Exception:
            seconds = None
        try:
            client = self._agent_client()
        except Exception:
            client = None
        self.card.set_last_heard(
            last_heard_text(seconds, client), heard=seconds is not None
        )

    def _on_chosen(self, level: AgentLevel) -> None:
        if level is not self._level:
            self.level_chosen.emit(level)
