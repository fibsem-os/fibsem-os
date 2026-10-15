"""The agent chip in the status bar: the agent's level, and the way to its card
(FIB-1035, FIB-1404).

The agent is a second operator at the app, and its level is the one knob: Off,
Watch, Assist or Drive. The chip shows the level; clicking it opens the agent
card (`agent_card.AgentCard`) on its own, above the chip: what is waiting for the
operator, and the controls (the level, Stop agent, who the agent is, the event
log). No menu and no dialog.

At Assist the agent acts by asking (FIB-1283). While its request waits, the chip
turns amber, "Agent asks", and the card shows the request. Only one request waits
at a time, so the chip carries no count. A suggested answer to the run's question
turns it amber too, "Agent suggests", and the card shows it with Accept
(FIB-1407).

The card can float (FIB-1404): its button moves it into a window of its own, a
tool window over the app (never over other apps) that the operator moves and
sizes, narrow enough and it stacks. Closing that window, or the button again,
puts the card back on the chip. The window keeps its place for the session.

Like `WorkflowControls`, this only shows what it is told and says what was picked.
What a pick does -- setting the App Server's level -- is the window's.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from PyQt5.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt5.QtWidgets import QFrame, QPushButton, QVBoxLayout, QWidget

from fibsem.applications.autolamella.ui.agent_card import (
    AgentCard,
    last_heard_text,
)

# The card's width on the chip: wide while something waits for the operator (the
# actions beside the reason and the decision), narrow otherwise.
WIDE_WIDTH = 820
NARROW_WIDTH = 440
from fibsem.server.levels import AgentLevel
from fibsem.ui.tokens import (
    BORDER_COLOR,
    PURPLE_COLOR,
    REVIEW_COLOR,
    TEXT_MUTED_COLOR,
    TEXT_STRONG_COLOR,
)

__all__ = [
    "AgentLevelChip",
    "LEVEL_WORDS",
    "OFFERED_LEVELS",
    "chip_request_text",
    "last_heard_text",
    "request_outcome",
]

_CHIP = """
    QPushButton {{
        color: {color};
        background-color: {background};
        border: {border};
        padding: 4px 10px;
        border-radius: 3px;
        font-weight: bold;
    }}
"""

# Assisting's fill: the purple at a third strength, between Watching's outline and
# Driving's solid fill.
_PURPLE_TINT = "rgba({}, {}, {}, 0.35)".format(
    *(int(PURPLE_COLOR[i : i + 2], 16) for i in (1, 3, 5))
)

# Something waiting for the operator, or being applied: amber, as the Review
# tab's waiting decisions.
_REQUEST_CHIP = _CHIP.format(
    color=TEXT_STRONG_COLOR,
    background="rgba({}, {}, {}, 0.30)".format(
        *(int(REVIEW_COLOR[i : i + 2], 16) for i in (1, 3, 5))
    ),
    border=f"1px solid {REVIEW_COLOR}",
)

# Each level: the chip's word, its look, and its tooltip.
_LEVELS = {
    AgentLevel.OFF: (
        "Agent: Off",
        _CHIP.format(color=TEXT_MUTED_COLOR, background="transparent", border="none"),
        "No access",
    ),
    AgentLevel.WATCH: (
        "Agent: Watching",
        _CHIP.format(
            color=TEXT_STRONG_COLOR,
            background="transparent",
            border=f"1px solid {PURPLE_COLOR}",
        ),
        "Read only",
    ),
    AgentLevel.ASSIST: (
        "Agent: Assisting",
        _CHIP.format(
            color=TEXT_STRONG_COLOR,
            background=_PURPLE_TINT,
            border=f"1px solid {PURPLE_COLOR}",
        ),
        "Asks first",
    ),
    AgentLevel.DRIVE: (
        "Agent: Driving",
        _CHIP.format(
            color=TEXT_STRONG_COLOR,
            background=PURPLE_COLOR,
            border=f"1px solid {PURPLE_COLOR}",
        ),
        "Full control",
    ),
}

# The chip's word for each level, for anywhere else that names it.
LEVEL_WORDS = {level: text for level, (text, _, _) in _LEVELS.items()}

# The levels the card offers, in order.
OFFERED_LEVELS = (AgentLevel.OFF, AgentLevel.WATCH, AgentLevel.ASSIST, AgentLevel.DRIVE)

_HELP = (
    ".\n\nClick for the agent's card. “Agent” is any client holding the App "
    "Server's agent token: the fibsem-mcp sidecar, FibsemClient or a script."
)


def chip_request_text(request: Dict[str, Any]) -> str:
    """The chip's word while a request waits or is being applied."""
    if request.get("status") == "applying":
        total = len(request.get("actions", []))
        done = len(request.get("results", []))
        return f"Agent: applying {min(done + 1, total)}/{total}"
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


class FloatingPanel(QWidget):
    """The card in a window of its own, over the app's window only (a tool
    window); closing it puts the card back on the chip."""

    closed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent, Qt.Tool)  # type: ignore
        self.setWindowTitle("Agent")
        self.setObjectName("agentPanel")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

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
    approve_clicked = pyqtSignal(str)  # request id
    reject_clicked = pyqtSignal(str, str)  # request id, the operator's reason
    accept_clicked = pyqtSignal()  # the suggestion shown

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

        self.card = AgentCard()
        self.card.level_chosen.connect(self._on_chosen)
        self.card.stop_clicked.connect(self._closing(self.stop_clicked))
        self.card.event_log_clicked.connect(self._closing(self.event_log_clicked))
        self.card.connect_clicked.connect(self._closing(self.connect_clicked))
        self.card.approve_clicked.connect(self._on_approve)
        self.card.reject_clicked.connect(self._on_reject)
        self.card.accept_clicked.connect(self._closing(self.accept_clicked))
        self.card.float_clicked.connect(self._on_float_clicked)
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
        for text in ("Agent asks", "Agent suggests", "Agent: applying 20/20"):
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
        self._request = dict(request) if request is not None else None
        self.card.set_request(self._request)
        self._paint()
        self._fit_popup()

    @property
    def suggestion(self) -> Optional[Dict[str, str]]:
        return self._suggestion

    def set_suggestion(self, suggestion: Optional[Dict[str, str]]) -> None:
        """Show the agent's suggested answer (see `AgentCard.set_suggestion`),
        or None for none. Does not emit: this is the window telling the chip."""
        self._suggestion = dict(suggestion) if suggestion is not None else None
        self.card.set_suggestion(self._suggestion)
        self._paint()
        self._fit_popup()

    def set_now(self, action: Optional[str]) -> None:
        """What the agent is doing on the microscope, for the card."""
        self.card.set_now(action)

    def _paint(self) -> None:
        if self._level is None:
            return
        text, style, tooltip = _LEVELS[self._level]
        if self._request is not None:
            text, style = chip_request_text(self._request), _REQUEST_CHIP
            tooltip = "The agent is asking: click to see what, and decide"
        elif self._suggestion is not None:
            text, style = "Agent suggests", _REQUEST_CHIP
            tooltip = (
                "The agent suggests an answer to the run's question: click to see it"
            )
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
        self.refresh_last_heard()
        self.panel.hold(self.card)
        self.card.set_floating(True)
        if self._panel_geometry is not None:
            self.panel.setGeometry(self._panel_geometry)
        else:
            self.panel.resize(self._card_width(), self.card.sizeHint().height())
            if start is not None:
                self.panel.move(start.topLeft())
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
        corner = self.mapToGlobal(QPoint(self.width(), 0))
        x = corner.x() - popup.width()
        y = corner.y() - popup.height() - 4
        screen = self.screen().availableGeometry() if self.screen() else None
        if screen is not None:
            x = max(screen.left(), min(x, screen.right() - popup.width()))
            y = max(screen.top(), y)
        popup.move(x, y)

    def _closing(self, signal):
        def emit() -> None:
            self.popup.close()
            signal.emit()

        return emit

    def _on_approve(self, request_id: str) -> None:
        self.popup.close()
        self.approve_clicked.emit(request_id)

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
