"""The agent chip in the status bar: the agent's level, set in two clicks (FIB-1035).

The agent is a second operator at the app, and its level is the one knob: Off, Watch,
Assist or Drive. The chip shows the level; clicking it opens a menu that sets it, with
when the agent was last heard from at the bottom. No dialog.

Like `WorkflowControls`, this only shows what it is told and says what was picked.
What a pick does -- setting the App Server's level -- is the window's.

At Assist the agent acts by asking (FIB-1283). While its request waits, the chip
turns amber ("Agent asks · 3") and the menu opens on the request: each action as the
server describes it, the agent's reason, and Reject… and Approve. After Approve the
same block shows each action's progress.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import (
    QAction,
    QActionGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from fibsem.server.levels import AgentLevel
from fibsem.ui.icon import fibsem_icon
from fibsem.ui.stylesheets import (
    PRIMARY_BUTTON_STYLESHEET,
    SECONDARY_BUTTON_STYLESHEET,
)
from fibsem.ui.tokens import (
    BORDER_COLOR,
    GRAY_ICON_COLOR,
    OK_COLOR,
    PURPLE_COLOR,
    RED_COLOR,
    REVIEW_COLOR,
    TEXT_COLOR,
    TEXT_MUTED_COLOR,
    TEXT_STRONG_COLOR,
)

_CHIP = """
    QPushButton {{
        color: {color};
        background-color: {background};
        border: {border};
        padding: 4px 10px;
        border-radius: 3px;
        font-weight: bold;
    }}
    QPushButton::menu-indicator {{ width: 0px; }}
"""

# Assisting's fill: the purple at a third strength, between Watching's outline and
# Driving's solid fill.
_PURPLE_TINT = "rgba({}, {}, {}, 0.35)".format(
    *(int(PURPLE_COLOR[i : i + 2], 16) for i in (1, 3, 5))
)

# A request waiting, or being applied: amber, as the Review tab's waiting decisions.
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

# The menu's choices, in order.
OFFERED_LEVELS = (AgentLevel.OFF, AgentLevel.WATCH, AgentLevel.ASSIST, AgentLevel.DRIVE)

_MENU_NAMES = {
    AgentLevel.OFF: "Off",
    AgentLevel.WATCH: "Watch",
    AgentLevel.ASSIST: "Assist",
    AgentLevel.DRIVE: "Drive",
}

_HELP = (
    ".\n\nClick to change. “Agent” is any client holding the App Server's "
    "agent token: the fibsem-mcp sidecar, FibsemClient or a script."
)


# The menu has more room than the app's default, and the level rows paint their own
# hover: a QWidgetAction's row gets none from QMenu::item:selected.
_MENU = f"""
    QMenu {{ padding: 4px 0px; }}
    QMenu::item {{ padding: 6px 20px 6px 12px; }}
    QMenu::icon {{ padding-left: 12px; }}
    QMenu::separator {{ height: 1px; background: {BORDER_COLOR}; margin: 4px 0px; }}
    #levelRow {{ background: transparent; }}
    #levelRow:hover {{ background: {BORDER_COLOR}; }}
    #levelRow QLabel {{ background: transparent; }}
    #requestBlock {{ background: transparent; }}
    #requestBlock QLabel {{ background: transparent; }}
    #requestActions {{ border: 1px solid {BORDER_COLOR}; border-radius: 3px; }}
    #requestBlock QLineEdit {{
        border: 1px solid {BORDER_COLOR}; border-radius: 3px; padding: 4px 6px;
        color: {TEXT_STRONG_COLOR};
    }}
"""


class _LevelRow(QWidget):
    """One level in the menu: a dot, the level's name, and its line under it.

    Shows its action's checked state, and triggers it on click.
    """

    def __init__(self, action: QAction, menu: QMenu, line: str) -> None:
        super().__init__()
        self._action = action
        self._menu = menu
        self.setObjectName("levelRow")
        self.setAttribute(Qt.WA_Hover)  # type: ignore
        self.setAttribute(Qt.WA_StyledBackground)  # type: ignore
        self.setCursor(Qt.PointingHandCursor)  # type: ignore

        layout = QHBoxLayout(self)
        layout.setContentsMargins(9, 6, 20, 6)
        layout.setSpacing(10)
        self._dot = QLabel()
        self._dot.setAlignment(Qt.AlignTop)  # type: ignore
        layout.addWidget(self._dot)
        text = QVBoxLayout()
        text.setSpacing(1)
        name = QLabel(action.text())
        name.setStyleSheet(f"color: {TEXT_STRONG_COLOR};")
        sub = QLabel(line)
        font = sub.font()
        font.setPointSizeF(font.pointSizeF() - 1)
        sub.setFont(font)
        sub.setStyleSheet(f"color: {TEXT_MUTED_COLOR};")
        text.addWidget(name)
        text.addWidget(sub)
        layout.addLayout(text, 1)

        action.toggled.connect(self._paint)
        self._paint(action.isChecked())

    def _paint(self, checked: bool) -> None:
        self._dot.setText("●" if checked else "○")
        colour = PURPLE_COLOR if checked else TEXT_MUTED_COLOR
        self._dot.setStyleSheet(f"color: {colour};")

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:  # type: ignore
            self._menu.close()
            self._action.trigger()


def chip_request_text(request: Dict[str, Any]) -> str:
    """The chip's word while a request waits or is being applied."""
    total = len(request.get("actions", []))
    if request.get("status") == "applying":
        done = len(request.get("results", []))
        return f"Agent: applying {min(done + 1, total)}/{total}"
    return f"Agent asks · {total}"


def request_failure_text(request: Dict[str, Any]) -> str:
    """The status bar's failure line for a request that was refused part way."""
    results = request.get("results", [])
    total = len(request.get("actions", []))
    applied = sum(1 for r in results if r.get("ok"))
    refused = next((r for r in results if not r.get("ok")), None)
    if refused is None:
        return f"{applied} of {total} applied"
    error = refused.get("error")
    message = error.get("message") if isinstance(error, dict) else error
    return (
        f"{applied} of {total} applied; refused at {refused['index'] + 1}: "
        f"{message or refused.get('status_code')}"
    )


def _label(text: str, color: str, small: bool = False, bold: bool = False) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet(f"color: {color};" + (" font-weight: bold;" if bold else ""))
    if small:
        font = label.font()
        font.setPointSizeF(font.pointSizeF() - 1)
        label.setFont(font)
    return label


class RequestBlock(QWidget):
    """The agent's request, at the top of the chip's menu: its actions as the server
    describes them, the agent's reason, and the operator's Reject… and Approve.

    Says what was clicked; what that does is the window's.
    """

    approve_clicked = pyqtSignal(str)  # request id
    reject_clicked = pyqtSignal(str, str)  # request id, the operator's reason

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("requestBlock")
        self.setAttribute(Qt.WA_StyledBackground)  # type: ignore
        self.setFixedWidth(380)
        self._request: Dict[str, Any] = {}
        self._rejecting = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(6)
        head = QHBoxLayout()
        head.addWidget(_label("●", REVIEW_COLOR))
        head.addWidget(_label("Agent wants to", TEXT_STRONG_COLOR, bold=True), 1)
        self._count = _label("", TEXT_MUTED_COLOR, small=True)
        head.addWidget(self._count)
        layout.addLayout(head)

        self._actions = QFrame()
        self._actions.setObjectName("requestActions")
        frame = QVBoxLayout(self._actions)
        frame.setContentsMargins(10, 8, 10, 8)
        self._rows: Optional[QWidget] = None
        layout.addWidget(self._actions)

        self._why = _label("", TEXT_COLOR)
        self._why.setWordWrap(True)
        layout.addWidget(self._why)
        self._progress = _label("", TEXT_MUTED_COLOR, small=True)
        layout.addWidget(self._progress)

        self._reason = QLineEdit()
        self._reason.setPlaceholderText("Tell the agent why (optional)")
        layout.addWidget(self._reason)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self._back = QPushButton("Back")
        self._back.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        self._back.clicked.connect(lambda: self._set_rejecting(False))
        self._reject = QPushButton("Reject…")
        self._reject.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        self._reject.clicked.connect(lambda: self._set_rejecting(True))
        self._confirm_reject = QPushButton("Reject")
        self._confirm_reject.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        self._confirm_reject.clicked.connect(self._on_reject)
        self._approve = QPushButton("Approve")
        self._approve.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        self._approve.clicked.connect(
            lambda: self.approve_clicked.emit(self._request.get("request_id", ""))
        )
        for button in (self._back, self._reject, self._confirm_reject, self._approve):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self._buttons = buttons
        self._render()

    @property
    def request_id(self) -> Optional[str]:
        return self._request.get("request_id")

    def show_request(self, request: Dict[str, Any]) -> None:
        """Show *request* (the server's ``to_dict``): waiting, or being applied."""
        if request.get("request_id") != self.request_id:
            self._rejecting = False
            self._reason.clear()
        self._request = dict(request)
        self._render()

    def _set_rejecting(self, rejecting: bool) -> None:
        self._rejecting = rejecting
        self._render()
        if rejecting:
            self._reason.setFocus()

    def _on_reject(self) -> None:
        self.reject_clicked.emit(
            self._request.get("request_id", ""), self._reason.text()
        )

    def _render(self) -> None:
        request = self._request
        actions: List[Dict[str, Any]] = request.get("actions", [])
        results = request.get("results", [])
        applying = request.get("status") == "applying"
        n = len(actions)
        self._count.setText(f"{n} action{'s' if n != 1 else ''}")

        # The rows are rebuilt whole: a request is a handful of actions.
        if self._rows is not None:
            self._rows.hide()
            self._actions.layout().removeWidget(self._rows)
            self._rows.deleteLater()
        self._rows = QWidget()
        rows = QVBoxLayout(self._rows)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setSpacing(8)
        self._actions.layout().addWidget(self._rows)
        for index, action in enumerate(actions):
            row = QHBoxLayout()
            row.setSpacing(8)
            if applying:
                if index < len(results):
                    ok = results[index].get("ok")
                    mark = _label("✓" if ok else "✕", OK_COLOR if ok else RED_COLOR)
                elif index == len(results):
                    mark = _label("…", TEXT_STRONG_COLOR)
                else:
                    mark = _label("○", TEXT_MUTED_COLOR)
            else:
                mark = _label(str(index + 1), TEXT_MUTED_COLOR)
            mark.setFixedWidth(12)
            mark.setAlignment(Qt.AlignTop)  # type: ignore
            row.addWidget(mark)
            text = QVBoxLayout()
            text.setSpacing(1)
            text.addWidget(
                _label(
                    action.get("what") or action.get("tool", ""),
                    TEXT_MUTED_COLOR,
                    small=True,
                )
            )
            change = _label(action.get("change") or "", TEXT_STRONG_COLOR)
            change.setWordWrap(True)
            text.addWidget(change)
            row.addLayout(text, 1)
            rows.addLayout(row)

        self._why.setText(f"Why: {request.get('reason', '')}")
        self._progress.setVisible(applying)
        if applying:
            self._progress.setText(
                f"Approved by you. Applying {min(len(results) + 1, n)} of {n}…"
            )
        deciding = not applying
        self._reason.setVisible(deciding and self._rejecting)
        self._back.setVisible(deciding and self._rejecting)
        self._confirm_reject.setVisible(deciding and self._rejecting)
        self._reject.setVisible(deciding and not self._rejecting)
        self._approve.setVisible(deciding and not self._rejecting)
        self.adjustSize()


def last_heard_text(seconds: Optional[float]) -> str:
    """The menu's bottom line, from seconds since the agent's token was last used."""
    if seconds is None:
        return "Agent · not heard from yet"
    if seconds < 60:
        return f"Agent · last heard {int(seconds)} s ago"
    return f"Agent · last heard {int(seconds // 60)} min ago"


class AgentLevelChip(QPushButton):
    """The agent's level, and the menu that sets it."""

    level_chosen = pyqtSignal(object)  # AgentLevel
    stop_clicked = pyqtSignal()
    connect_clicked = pyqtSignal()
    approve_clicked = pyqtSignal(str)  # request id
    reject_clicked = pyqtSignal(str, str)  # request id, the operator's reason

    def __init__(
        self,
        last_heard: Callable[[], Optional[float]] = lambda: None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._last_heard = last_heard
        self._level: Optional[AgentLevel] = None
        self.setCursor(Qt.PointingHandCursor)  # type: ignore

        self.menu_ = QMenu(self)
        self.menu_.setStyleSheet(_MENU)
        self._request: Optional[Dict[str, Any]] = None
        # The agent's request, first: it is what the chip is amber about.
        self.request_block = RequestBlock()
        self.request_block.approve_clicked.connect(self._on_approve)
        self.request_block.reject_clicked.connect(self._on_reject)
        self.request_action = QWidgetAction(self)
        self.request_action.setDefaultWidget(self.request_block)
        self.menu_.addAction(self.request_action)
        self._request_separator = self.menu_.addSeparator()
        # The brake, first whenever the agent can act: no confirmation. Red, as
        # the queue's Stop Task: it interrupts something.
        self.stop_action = QAction(
            fibsem_icon("mdi:stop-circle-outline", color=RED_COLOR), "Stop agent", self
        )
        self.stop_action.setToolTip(
            "Stop the agent acting: down to Watch at once. A running workflow "
            "keeps running; its own Stop is beside this."
        )
        self.stop_action.triggered.connect(self.stop_clicked)
        self.menu_.addAction(self.stop_action)
        self._stop_separator = self.menu_.addSeparator()
        self._group = QActionGroup(self)
        self._group.setExclusive(True)
        self._actions = {}
        for level in OFFERED_LEVELS:
            action = QWidgetAction(self)
            action.setText(_MENU_NAMES[level])
            action.setCheckable(True)
            action.setDefaultWidget(_LevelRow(action, self.menu_, _LEVELS[level][2]))
            action.triggered.connect(
                lambda _checked=False, chosen=level: self._on_chosen(chosen)
            )
            self._group.addAction(action)
            self.menu_.addAction(action)
            self._actions[level] = action
        self.menu_.addSeparator()
        self.last_heard_action = QAction(last_heard_text(None), self)
        self.last_heard_action.setEnabled(False)
        self.menu_.addAction(self.last_heard_action)
        # An icon here keeps the menu's icon column when Stop is hidden, so every
        # line's text starts where the level names do.
        self.connect_action = QAction(
            fibsem_icon("mdi:link-variant", color=GRAY_ICON_COLOR),
            "Connect an agent…",
            self,
        )
        self.connect_action.triggered.connect(self.connect_clicked)
        self.menu_.addAction(self.connect_action)
        # Read when the menu opens, so it is never a stale number.
        self.menu_.aboutToShow.connect(self.refresh_last_heard)
        self.setMenu(self.menu_)

        # One width for every level and request state, so the status bar doesn't
        # shift when it changes.
        widths = []
        for level in _LEVELS:
            self.set_level(level)
            self.ensurePolished()
            widths.append(self.sizeHint().width())
        for text in ("Agent asks · 20", "Agent: applying 20/20"):
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
        self._paint()
        action = self._actions.get(level)
        if action is not None:
            action.setChecked(True)
        can_act = level in (AgentLevel.ASSIST, AgentLevel.DRIVE)
        self.stop_action.setVisible(can_act)
        self._stop_separator.setVisible(can_act)

    @property
    def request(self) -> Optional[Dict[str, Any]]:
        return self._request

    def set_request(self, request: Optional[Dict[str, Any]]) -> None:
        """Show the agent's request waiting or being applied, or None for none.
        Does not emit: this is the window telling the chip."""
        self._request = dict(request) if request is not None else None
        showing = self._request is not None
        if showing:
            self.request_block.show_request(self._request)
        self.request_action.setVisible(showing)
        self._request_separator.setVisible(showing)
        self._paint()
        if self.menu_.isVisible():
            self.menu_.adjustSize()

    def _paint(self) -> None:
        if self._level is None:
            return
        text, style, tooltip = _LEVELS[self._level]
        if self._request is not None:
            text, style = chip_request_text(self._request), _REQUEST_CHIP
            tooltip = "The agent is asking: click to see what, and decide"
        self.setText(text)
        self.setStyleSheet(style)
        self.setToolTip(tooltip + _HELP)

    def _on_approve(self, request_id: str) -> None:
        self.menu_.close()
        self.approve_clicked.emit(request_id)

    def _on_reject(self, request_id: str, reason: str) -> None:
        self.menu_.close()
        self.reject_clicked.emit(request_id, reason)

    def refresh_last_heard(self) -> None:
        try:
            seconds = self._last_heard()
        except Exception:
            seconds = None
        self.last_heard_action.setText(last_heard_text(seconds))

    def _on_chosen(self, level: AgentLevel) -> None:
        if level is not self._level:
            self.level_chosen.emit(level)
