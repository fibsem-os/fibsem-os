"""The agent card: what the agent asks of the operator, and its controls (FIB-1404).

The chip in the status bar opens it. It holds, top to bottom:

* what is waiting for the operator: the agent's request, its actions one plain
  line each as the server wrote them (FIB-1403), "Show details" for every value,
  the agent's reason, and Reject… and Approve; while an approved request runs,
  each action's progress. Or the agent's suggested answer to the run's
  question, with Accept (FIB-1407). With nothing waiting, what the agent is
  doing now;
* the controls: the level (Off, Watch, Assist, Drive), Stop agent, who the agent
  is and when it was last heard from, and the event log.

It is wide: the actions on the left, the reason and the decision on the right.
Narrower than `STACK_BELOW` px it stacks them, so the same card floats as a
window the operator sizes (the button at its top right floats it, and puts it
back on the chip).

Like the chip, it only shows what it is told and says what was clicked. What a
click does is the window's.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QBoxLayout,
    QButtonGroup,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from fibsem.server.levels import AgentLevel
from fibsem.ui.icon import fibsem_icon
from fibsem.ui.stylesheets import (
    PRIMARY_BUTTON_STYLESHEET,
    SECONDARY_BUTTON_STYLESHEET,
)
from fibsem.ui.tokens import (
    ACCENT_COLOR,
    BORDER_COLOR,
    OK_COLOR,
    PANEL_COLOR,
    PURPLE_COLOR,
    RED_COLOR,
    REVIEW_COLOR,
    SURFACE_COLOR,
    TEXT_COLOR,
    TEXT_MUTED_COLOR,
    TEXT_STRONG_COLOR,
)

# Narrower than this, the reason and the decision go under the actions.
STACK_BELOW = 600
# The narrowest the stacked card goes.
NARROWEST = 360

# The level switch, in order, as the operator reads it.
LEVEL_NAMES = {
    AgentLevel.OFF: "Off",
    AgentLevel.WATCH: "Watch",
    AgentLevel.ASSIST: "Assist",
    AgentLevel.DRIVE: "Drive",
}
LEVEL_TIPS = {
    AgentLevel.OFF: "No access",
    AgentLevel.WATCH: "Read only",
    AgentLevel.ASSIST: "Asks first",
    AgentLevel.DRIVE: "Full control",
}

_CARD = f"""
    #agentCard {{ background: {SURFACE_COLOR}; }}
    #agentCard QLabel {{ background: transparent; }}
    #agentActions {{ border: 1px solid {BORDER_COLOR}; border-radius: 3px; }}
    #agentDetails {{ background: {PANEL_COLOR}; border-radius: 3px; }}
    #agentRule {{ background: {BORDER_COLOR}; }}
    #agentCard QLineEdit {{
        border: 1px solid {BORDER_COLOR}; border-radius: 3px; padding: 4px 6px;
        color: {TEXT_STRONG_COLOR};
    }}
    #agentLink {{
        background: transparent; color: {ACCENT_COLOR}; border: none;
        padding: 0px; text-align: left;
    }}
    #agentLink:hover {{ color: #7bc0ff; }}
"""

# One switch: the segments share borders, the level in force is filled purple.
_SEGMENT = """
    QPushButton {{
        color: {text}; background: transparent;
        border: 1px solid {border}; padding: 3px 10px;
        border-top-left-radius: {left}; border-bottom-left-radius: {left};
        border-top-right-radius: {right}; border-bottom-right-radius: {right};
    }}
    QPushButton:hover {{ background: {hover}; }}
    QPushButton:checked {{
        color: {strong}; background: {purple}; border-color: {purple};
    }}
"""


def _label(
    text: str = "",
    color: str = TEXT_COLOR,
    small: bool = False,
    bold: bool = False,
    wrap: bool = False,
) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet(f"color: {color};" + (" font-weight: bold;" if bold else ""))
    if small:
        font = label.font()
        font.setPointSizeF(font.pointSizeF() - 1)
        label.setFont(font)
    label.setWordWrap(wrap)
    return label


def _section(text: str) -> QLabel:
    return _label(text.upper(), TEXT_MUTED_COLOR, small=True, bold=True)


def _link(text: str) -> QPushButton:
    button = QPushButton(text)
    button.setObjectName("agentLink")
    button.setCursor(Qt.PointingHandCursor)  # type: ignore
    font = button.font()
    font.setPointSizeF(font.pointSizeF() - 1)
    button.setFont(font)
    return button


def _button(text: str, primary: bool = False) -> QPushButton:
    button = QPushButton(text)
    button.setStyleSheet(
        PRIMARY_BUTTON_STYLESHEET if primary else SECONDARY_BUTTON_STYLESHEET
    )
    return button


def _clear(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.hide()
            widget.deleteLater()
        elif item.layout() is not None:
            _clear(item.layout())


def last_heard_text(seconds: Optional[float], client: Optional[str] = None) -> str:
    """Who the agent is and when its token was last used: "claude-code/2.1.3"
    reads "claude-code"."""
    who = client.split("/", 1)[0] if client else "Agent"
    if seconds is None:
        return f"{who} · not heard from yet"
    if seconds < 60:
        return f"{who} · last heard {int(seconds)} s ago"
    return f"{who} · last heard {int(seconds // 60)} min ago"


class LevelSwitch(QWidget):
    """Off | Watch | Assist | Drive. Says what was clicked; shows what it is told."""

    level_chosen = pyqtSignal(object)  # AgentLevel

    def __init__(self) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self.buttons: Dict[AgentLevel, QPushButton] = {}
        levels = list(LEVEL_NAMES)
        for index, level in enumerate(levels):
            button = QPushButton(LEVEL_NAMES[level])
            button.setCheckable(True)
            button.setToolTip(LEVEL_TIPS[level])
            button.setCursor(Qt.PointingHandCursor)  # type: ignore
            button.setStyleSheet(
                _SEGMENT.format(
                    text=TEXT_COLOR,
                    strong=TEXT_STRONG_COLOR,
                    border=BORDER_COLOR,
                    hover=BORDER_COLOR,
                    purple=PURPLE_COLOR,
                    left="3px" if index == 0 else "0px",
                    right="3px" if index == len(levels) - 1 else "0px",
                )
            )
            button.clicked.connect(
                lambda _checked=False, chosen=level: self._on_clicked(chosen)
            )
            button.ensurePolished()
            button.setMinimumWidth(button.sizeHint().width())
            self._group.addButton(button)
            self.buttons[level] = button
            layout.addWidget(button)
        self._level: Optional[AgentLevel] = None

    def set_level(self, level: AgentLevel) -> None:
        self._level = AgentLevel(level)
        self.buttons[self._level].setChecked(True)

    def _on_clicked(self, level: AgentLevel) -> None:
        # The switch shows what the window says, not the click: put it back
        # until the window answers.
        if self._level is not None:
            self.buttons[self._level].setChecked(True)
        if level is not self._level:
            self.level_chosen.emit(level)


class _Details(QFrame):
    """An action's every value: the table the server wrote."""

    def __init__(self, details: Dict[str, Any]) -> None:
        super().__init__()
        self.setObjectName("agentDetails")
        grid = QGridLayout(self)
        grid.setContentsMargins(8, 6, 8, 6)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(3)
        columns = list(details.get("columns") or [])
        for column, name in enumerate(columns):
            grid.addWidget(_label(name, TEXT_MUTED_COLOR, small=True), 0, column)
        for row, values in enumerate(details.get("rows") or [], start=1):
            for column, value in enumerate(values):
                cell = _label(str(value), TEXT_COLOR, small=True, wrap=True)
                cell.setTextInteractionFlags(Qt.TextSelectableByMouse)  # type: ignore
                grid.addWidget(cell, row, column)
        if columns:
            grid.setColumnStretch(len(columns) - 1, 1)


class AgentCard(QFrame):
    """What the agent asks, and its controls."""

    level_chosen = pyqtSignal(object)  # AgentLevel
    stop_clicked = pyqtSignal()
    event_log_clicked = pyqtSignal()
    connect_clicked = pyqtSignal()
    approve_clicked = pyqtSignal(str)  # request id
    reject_clicked = pyqtSignal(str, str)  # request id, the operator's reason
    accept_clicked = pyqtSignal()  # the suggestion shown
    float_clicked = pyqtSignal()  # float the card, or put it back on the chip

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("agentCard")
        self.setAttribute(Qt.WA_StyledBackground)  # type: ignore
        self.setStyleSheet(_CARD)
        self._request: Optional[Dict[str, Any]] = None
        self._suggestion: Optional[Dict[str, str]] = None
        self.accept_button: Optional[QPushButton] = None
        self._now: Optional[str] = None
        self._details_open = False
        self._rejecting = False
        self._wide = True

        layout = QVBoxLayout(self)
        # The wide layout's minimum must not stop the card narrowing to stack:
        # minimumSizeHint says how small it may go.
        layout.setSizeConstraint(QLayout.SetNoConstraint)
        layout.setContentsMargins(14, 10, 14, 12)
        layout.setSpacing(8)

        head = QHBoxLayout()
        head.setSpacing(8)
        self._dot = _label("●", PURPLE_COLOR)
        self._title = _label("Agent", TEXT_STRONG_COLOR, bold=True)
        head.addWidget(self._dot)
        head.addWidget(self._title, 1)
        self.float_button = QPushButton()
        self.float_button.setObjectName("agentLink")
        self.float_button.setCursor(Qt.PointingHandCursor)  # type: ignore
        self.float_button.clicked.connect(self.float_clicked)
        head.addWidget(self.float_button)
        self.set_floating(False)
        self.head = head
        layout.addLayout(head)

        # What is waiting: rebuilt whole on each change, a handful of rows.
        self._body = QBoxLayout(QBoxLayout.LeftToRight)
        self._body.setSpacing(16)
        layout.addLayout(self._body)

        rule = QFrame()
        rule.setObjectName("agentRule")
        rule.setFixedHeight(1)
        layout.addWidget(rule)

        # The controls, always.
        controls = QHBoxLayout()
        controls.setSpacing(10)
        controls.addWidget(_label("Level", TEXT_MUTED_COLOR, small=True))
        self.level_switch = LevelSwitch()
        self.level_switch.level_chosen.connect(self.level_chosen)
        controls.addWidget(self.level_switch)
        self.stop_button = QPushButton(" Stop agent")
        self.stop_button.setIcon(
            fibsem_icon("mdi:stop-circle-outline", color=RED_COLOR)
        )
        self.stop_button.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        self.stop_button.setToolTip(
            "Stop the agent acting: down to Watch at once. A running workflow "
            "keeps running; its own Stop is beside the chip."
        )
        self.stop_button.clicked.connect(self.stop_clicked)
        controls.addWidget(self.stop_button)
        controls.addStretch(1)
        self._controls = controls
        layout.addLayout(controls)

        # Who the agent is, and the way to the record: under the controls when
        # narrow, at their end when wide.
        self.heard_label = _label(last_heard_text(None), TEXT_MUTED_COLOR, small=True)
        self.connect_link = _link("Connect an agent…")
        self.connect_link.clicked.connect(self.connect_clicked)
        self.event_log_link = _link("Event log")
        self.event_log_link.setToolTip(
            "What the agent, the operator and the run did, in your browser."
        )
        self.event_log_link.clicked.connect(self.event_log_clicked)
        self._about = QHBoxLayout()
        self._about.setSpacing(10)
        layout.addLayout(self._about)

        self.set_level(AgentLevel.WATCH)
        self.set_last_heard(None)
        self._place_about()
        self._render()

    # --- told by the window ---------------------------------------------------

    @property
    def request(self) -> Optional[Dict[str, Any]]:
        return self._request

    def set_level(self, level: AgentLevel) -> None:
        level = AgentLevel(level)
        self.level_switch.set_level(level)
        self.stop_button.setVisible(level in (AgentLevel.ASSIST, AgentLevel.DRIVE))

    def set_request(self, request: Optional[Dict[str, Any]]) -> None:
        """The agent's request waiting or being applied (the server's
        ``to_dict``), or None for none."""
        request = dict(request) if request is not None else None
        if (request or {}).get("request_id") != (self._request or {}).get("request_id"):
            self._rejecting = False
            self._details_open = False
        if request == self._request:
            return
        self._request = request
        self._render()

    @property
    def suggestion(self) -> Optional[Dict[str, str]]:
        return self._suggestion

    def set_suggestion(self, suggestion: Optional[Dict[str, str]]) -> None:
        """The agent's suggested answer to the run's question -- ``question``
        (the prompt's text), ``answer`` (the prompt's button it stands for) and
        ``reason`` -- or None for none. A request, when one waits, shows first."""
        suggestion = dict(suggestion) if suggestion is not None else None
        if suggestion == self._suggestion:
            return
        self._suggestion = suggestion
        self._render()

    def set_now(self, action: Optional[str]) -> None:
        """What the agent is doing on the microscope, or None."""
        if action == self._now:
            return
        self._now = action
        if self._request is None and self._suggestion is None:
            self._render()

    def set_floating(self, floating: bool) -> None:
        """Whether the card is in its own window: its button then puts it back."""
        self.float_button.setIcon(
            fibsem_icon(
                "mdi:dock-bottom" if floating else "mdi:open-in-new",
                color=TEXT_MUTED_COLOR,
            )
        )
        self.float_button.setToolTip(
            "Put the card back on the chip"
            if floating
            else "Keep the card open in its own window, above the app"
        )

    def set_last_heard(self, text: Optional[str], heard: bool = False) -> None:
        """Who the agent is and when it was last heard; not heard from yet, the
        card offers to connect one instead."""
        self.heard_label.setText(text or last_heard_text(None))
        self.connect_link.setVisible(not heard)

    # --- layout ---------------------------------------------------------------

    @property
    def wide(self) -> bool:
        return self._wide

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt's name
        hint = self.layout().minimumSize()
        margins = self.layout().contentsMargins()
        controls = self._controls
        # Stacked, the controls row holds only the level and Stop agent.
        narrow = (
            margins.left()
            + margins.right()
            + sum(
                controls.itemAt(i).widget().sizeHint().width()
                for i in range(3)  # "Level", the switch, Stop agent
            )
            + 2 * controls.spacing()
        )
        return QSize(min(hint.width(), max(narrow, NARROWEST)), hint.height())

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt's name
        super().resizeEvent(event)
        wide = event.size().width() >= STACK_BELOW
        if wide != self._wide:
            self._wide = wide
            self._place_about()
            self._render()

    def _place_about(self) -> None:
        for widget in (self.heard_label, self.connect_link, self.event_log_link):
            self._controls.removeWidget(widget)
            self._about.removeWidget(widget)
        while self._about.count():
            self._about.takeAt(0)
        if self._wide:
            for widget in (self.heard_label, self.connect_link, self.event_log_link):
                self._controls.addWidget(widget)
        else:
            self._about.addWidget(self.heard_label)
            self._about.addWidget(self.connect_link)
            self._about.addStretch(1)
            self._about.addWidget(self.event_log_link)

    # --- what is shown --------------------------------------------------------

    def _render(self) -> None:
        _clear(self._body)
        self._body.setDirection(
            QBoxLayout.LeftToRight if self._wide else QBoxLayout.TopToBottom
        )
        request = self._request
        self.accept_button = self.details_link = None
        self.approve_button = self.reject_button = None
        self.confirm_reject_button = self.back_button = None
        self.reason_edit = None
        if request is None and self._suggestion is None:
            self._dot.setStyleSheet(f"color: {PURPLE_COLOR};")
            self._title.setText("Agent")
            self._body.addLayout(self._nothing(), 1)
            self.updateGeometry()
            return
        self._dot.setStyleSheet(f"color: {REVIEW_COLOR};")
        if request is None:
            self._title.setText("Agent suggests an answer")
            left, right = self._suggestion_columns(self._suggestion)
        else:
            applying = request.get("status") == "applying"
            self._title.setText(
                "Agent is doing what you approved" if applying else "Agent wants to"
            )
            left, right = self._request_columns(request, applying)
        self._body.addLayout(left, 1)
        holder = QWidget()
        holder.setLayout(right)
        if self._wide:
            holder.setFixedWidth(260)
        self._body.addWidget(holder)
        self.updateGeometry()

    def _nothing(self) -> QVBoxLayout:
        column = QVBoxLayout()
        column.setSpacing(6)
        column.addWidget(_section("Now"))
        column.addWidget(
            _label(
                self._now or "Nothing.", TEXT_COLOR if self._now else TEXT_MUTED_COLOR
            )
        )
        column.addWidget(_section("Waiting for you"))
        column.addWidget(_label("Nothing.", TEXT_MUTED_COLOR))
        return column

    def _request_columns(self, request: Dict[str, Any], applying: bool):
        actions: List[Dict[str, Any]] = request.get("actions", [])
        results = request.get("results", [])

        left = QVBoxLayout()
        left.setSpacing(6)
        box = QFrame()
        box.setObjectName("agentActions")
        rows = QVBoxLayout(box)
        rows.setContentsMargins(10, 8, 10, 8)
        rows.setSpacing(8)
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
            mark.setFixedWidth(16)
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
            change = _label(action.get("change") or "", TEXT_STRONG_COLOR, wrap=True)
            change.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            text.addWidget(change)
            if self._details_open and action.get("details"):
                text.addSpacing(4)
                text.addWidget(_Details(action["details"]))
            row.addLayout(text, 1)
            rows.addLayout(row)
        left.addWidget(box)
        if any(action.get("details") for action in actions):
            toggle = _link("Hide details" if self._details_open else "Show details")
            toggle.clicked.connect(self._toggle_details)
            self.details_link = toggle
            line = QHBoxLayout()
            line.addWidget(toggle)
            line.addStretch(1)
            left.addLayout(line)
        else:
            self.details_link = None
        left.addStretch(1)

        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(8)
        right.addWidget(_label("Why", TEXT_MUTED_COLOR, small=True))
        why = _label(request.get("reason", ""), TEXT_COLOR, wrap=True)
        why.setTextInteractionFlags(Qt.TextSelectableByMouse)  # type: ignore
        right.addWidget(why)
        right.addStretch(1)
        self.approve_button = self.reject_button = None
        self.confirm_reject_button = self.back_button = None
        self.reason_edit = None
        n = len(actions)
        if applying:
            right.addWidget(
                _label(
                    f"Approved by you. Running {min(len(results) + 1, n)} of {n}…",
                    TEXT_MUTED_COLOR,
                    small=True,
                    wrap=True,
                )
            )
            return left, right
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        request_id = request.get("request_id", "")
        if self._rejecting:
            self.reason_edit = QLineEdit()
            self.reason_edit.setPlaceholderText("Tell the agent why (optional)")
            right.addWidget(self.reason_edit)
            self.back_button = _button("Back")
            self.back_button.clicked.connect(lambda: self._set_rejecting(False))
            self.confirm_reject_button = _button("Reject", primary=True)
            self.confirm_reject_button.clicked.connect(
                lambda: self.reject_clicked.emit(
                    request_id, self.reason_edit.text() if self.reason_edit else ""
                )
            )
            buttons.addWidget(self.back_button)
            buttons.addWidget(self.confirm_reject_button)
        else:
            self.reject_button = _button("Reject…")
            self.reject_button.clicked.connect(lambda: self._set_rejecting(True))
            self.approve_button = _button("Approve", primary=True)
            self.approve_button.clicked.connect(
                lambda: self.approve_clicked.emit(request_id)
            )
            buttons.addWidget(self.reject_button)
            buttons.addWidget(self.approve_button)
        right.addLayout(buttons)
        return left, right

    def _suggestion_columns(self, suggestion: Dict[str, str]):
        left = QVBoxLayout()
        left.setSpacing(6)
        box = QFrame()
        box.setObjectName("agentActions")
        rows = QVBoxLayout(box)
        rows.setContentsMargins(10, 8, 10, 8)
        rows.setSpacing(1)
        if suggestion.get("question"):
            rows.addWidget(
                _label(suggestion["question"], TEXT_MUTED_COLOR, small=True, wrap=True)
            )
        answer = _label(
            f"Answer: {suggestion.get('answer', '')}", TEXT_STRONG_COLOR, wrap=True
        )
        answer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        rows.addWidget(answer)
        left.addWidget(box)
        if suggestion.get("point") is not None:
            note = QHBoxLayout()
            note.addStretch(1)
            note.addWidget(
                _label(
                    "The point is shown on the FIB view", TEXT_MUTED_COLOR, small=True
                )
            )
            left.addLayout(note)
        left.addStretch(1)

        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(8)
        right.addWidget(_label("Why", TEXT_MUTED_COLOR, small=True))
        why = _label(suggestion.get("reason", ""), TEXT_COLOR, wrap=True)
        why.setTextInteractionFlags(Qt.TextSelectableByMouse)  # type: ignore
        right.addWidget(why)
        right.addStretch(1)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.accept_button = _button("Accept", primary=True)
        self.accept_button.setToolTip(
            "Answer the question with this, as your answer. The record names the "
            "agent as the one who suggested it."
        )
        self.accept_button.clicked.connect(self.accept_clicked)
        buttons.addWidget(self.accept_button)
        right.addLayout(buttons)
        right.addWidget(
            _label(
                "Answering on the prompt yourself drops it.",
                TEXT_MUTED_COLOR,
                small=True,
                wrap=True,
            )
        )
        return left, right

    def _toggle_details(self) -> None:
        self._details_open = not self._details_open
        self._render()

    def _set_rejecting(self, rejecting: bool) -> None:
        self._rejecting = rejecting
        self._render()
        if rejecting and self.reason_edit is not None:
            self.reason_edit.setFocus()
