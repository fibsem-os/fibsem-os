"""The agent card: what the agent asks of the operator, and its controls (FIB-1404).

The chip in the status bar opens it. It holds, top to bottom:

* what is waiting for the operator: the agent's request, its actions one plain
  line each as the server wrote them (FIB-1403), "Show details" for every value,
  the agent's reason, and Reject… and Approve; while an approved request runs,
  each action's progress. Or the agent's suggested answer to the run's
  question, with Accept (FIB-1407). With nothing waiting, only what the agent
  is doing now;
* what the agent did lately, Recent: on at first at Drive, where it acts
  without asking, off below it (FIB-1414);
* the controls: the level, as a menu that opens the four with what each allows
  (FIB-1418), Stop agent, who the agent is and when it was last heard from, and
  the event log.

It is wide: the actions on the left, the reason and the decision on the right.
Narrower than `STACK_BELOW` px it stacks them, so the same card floats as a
window the operator sizes (the button at its top right floats it, and puts it
back on the chip; its pin keeps it on top of every window).

Like the chip, it only shows what it is told and says what was clicked. What a
click does is the window's.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QBoxLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QPushButton,
    QScrollArea,
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

# The agent's reason, before "more".
WHY_LINES = 3

# How a Recent row ended (agent_recent's tones), as a colour.
_TONES = {
    "waiting": REVIEW_COLOR,
    "ok": OK_COLOR,
    "error": RED_COLOR,
    "muted": TEXT_MUTED_COLOR,
}

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
    #agentDetails, #agentDetailsTable {{
        background: {PANEL_COLOR}; border-radius: 3px;
    }}
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
    #agentRecent {{
        background: transparent; color: {TEXT_COLOR}; border: none;
        padding: 1px 0px; text-align: left;
    }}
    #agentRecent:hover {{ color: {TEXT_STRONG_COLOR}; }}
"""

# The level menu's rows: the level in force is marked with a filled dot.
_LEVEL_ROW = f"""
    QPushButton {{
        color: {TEXT_COLOR}; background: transparent; border: none;
        padding: 3px 6px; text-align: left;
    }}
    QPushButton:hover {{ background: {BORDER_COLOR}; }}
    QPushButton:checked {{ color: {TEXT_STRONG_COLOR}; font-weight: bold; }}
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


class LevelMenu(QWidget):
    """The level in force, "Level: Assist ▾"; opened, the four levels with what
    each allows, inline in the card (FIB-1418). The level changes rarely and
    each change matters, so it is one click further than a switch would be.
    Says what was picked; shows what it is told."""

    level_chosen = pyqtSignal(object)  # AgentLevel

    def __init__(self) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.toggle = QPushButton()
        self.toggle.setObjectName("agentLink")
        self.toggle.setCursor(Qt.PointingHandCursor)  # type: ignore
        self.toggle.setToolTip("Change what the agent may do")
        self.toggle.clicked.connect(lambda: self.set_open(not self.is_open))
        layout.addWidget(self.toggle)
        # The levels, which the card puts under its controls row.
        self.list = QWidget()
        rows = QGridLayout(self.list)
        rows.setContentsMargins(0, 2, 0, 2)
        rows.setHorizontalSpacing(8)
        rows.setVerticalSpacing(0)
        self.buttons: Dict[AgentLevel, QPushButton] = {}
        self._dots: Dict[AgentLevel, QLabel] = {}
        for row, level in enumerate(LEVEL_NAMES):
            dot = _label("○", TEXT_MUTED_COLOR)
            button = QPushButton(LEVEL_NAMES[level])
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)  # type: ignore
            button.setStyleSheet(_LEVEL_ROW)
            button.clicked.connect(
                lambda _checked=False, chosen=level: self._on_clicked(chosen)
            )
            rows.addWidget(dot, row, 0)
            rows.addWidget(button, row, 1)
            rows.addWidget(
                _label(LEVEL_TIPS[level], TEXT_MUTED_COLOR, small=True), row, 2
            )
            self.buttons[level] = button
            self._dots[level] = dot
        rows.setColumnStretch(2, 1)
        self._level: Optional[AgentLevel] = None
        self.set_open(False)

    @property
    def is_open(self) -> bool:
        return not self.list.isHidden()

    def set_open(self, open_: bool) -> None:
        self.list.setVisible(open_)
        self._paint()

    def set_level(self, level: AgentLevel) -> None:
        self._level = AgentLevel(level)
        self._paint()

    def _paint(self) -> None:
        level = self._level
        name = LEVEL_NAMES.get(level, "") if level is not None else ""
        self.toggle.setText(f"Level: {name} {'▴' if self.is_open else '▾'}")
        for each, button in self.buttons.items():
            button.setChecked(each is level)
            self._dots[each].setText("●" if each is level else "○")
            self._dots[each].setStyleSheet(
                f"color: {PURPLE_COLOR if each is level else TEXT_MUTED_COLOR};"
            )

    def _on_clicked(self, level: AgentLevel) -> None:
        # The menu shows what the window says, not the click: put it back
        # until the window answers.
        self._paint()
        self.set_open(False)
        if level is not self._level:
            self.level_chosen.emit(level)


class _Details(QScrollArea):
    """An action's every value: the table the server wrote.

    One line per row, never wrapped: a wrapped cell's height depends on its
    width, which the card's layout does not ask, so rows were squeezed into
    each other in the floating window (FIB-1413). A tall or wide table
    scrolls inside the card instead.
    """

    # Rows shown before the table scrolls.
    ROWS_SHOWN = 8

    def __init__(self, details: Dict[str, Any]) -> None:
        super().__init__()
        self.setObjectName("agentDetails")
        self.setFrameShape(QFrame.NoFrame)
        self.setWidgetResizable(True)
        table = QWidget()
        table.setObjectName("agentDetailsTable")
        grid = QGridLayout(table)
        grid.setContentsMargins(8, 6, 8, 6)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(3)
        columns = list(details.get("columns") or [])
        for column, name in enumerate(columns):
            grid.addWidget(_label(name, TEXT_MUTED_COLOR, small=True), 0, column)
        rows = list(details.get("rows") or [])
        for row, values in enumerate(rows, start=1):
            for column, value in enumerate(values):
                cell = _label(str(value), TEXT_COLOR, small=True)
                cell.setTextInteractionFlags(Qt.TextSelectableByMouse)  # type: ignore
                grid.addWidget(cell, row, column)
        if columns:
            grid.setColumnStretch(len(columns) - 1, 1)
        self.setWidget(table)
        # Every row at full height, up to ROWS_SHOWN; then it scrolls.
        full = table.sizeHint().height()
        shown = min(len(rows), self.ROWS_SHOWN) + 1  # and the header
        per_row = full / max(len(rows) + 1, 1)
        height = full if len(rows) <= self.ROWS_SHOWN else round(per_row * shown)
        self.setFixedHeight(height + self.horizontalScrollBar().sizeHint().height())


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
    keep_on_top_toggled = pyqtSignal(bool)  # the floating card over every window

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
        self._why_open = False
        self._wide = True
        self._level = AgentLevel.WATCH
        self._recent: List[Any] = []
        # The operator's choice to show Recent, per level, for the session.
        self._recent_choice: Dict[AgentLevel, bool] = {}

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
        self.pin_button = QPushButton()
        self.pin_button.setObjectName("agentLink")
        self.pin_button.setCheckable(True)
        self.pin_button.setCursor(Qt.PointingHandCursor)  # type: ignore
        self.pin_button.toggled.connect(self.keep_on_top_toggled)
        head.addWidget(self.pin_button)
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

        # What the agent did lately (FIB-1414): shown or not, per level.
        self._recent_rows = QVBoxLayout()
        self._recent_rows.setSpacing(2)
        layout.addLayout(self._recent_rows)

        rule = QFrame()
        rule.setObjectName("agentRule")
        rule.setFixedHeight(1)
        layout.addWidget(rule)

        # The controls, always.
        controls = QHBoxLayout()
        controls.setSpacing(10)
        self.level_menu = LevelMenu()
        self.level_menu.level_chosen.connect(self.level_chosen)
        controls.addWidget(self.level_menu)
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
        layout.addWidget(self.level_menu.list)

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
        self.recent_link = _link("")
        self.recent_link.setToolTip("The agent's last few actions and how each ended")
        self.recent_link.clicked.connect(self._toggle_recent)
        self._about = QHBoxLayout()
        self._about.setSpacing(10)
        layout.addLayout(self._about)

        self.set_level(AgentLevel.WATCH)
        self.set_last_heard(None)
        self.set_keep_on_top(False)
        self._place_about()
        self._render()

    # --- told by the window ---------------------------------------------------

    @property
    def request(self) -> Optional[Dict[str, Any]]:
        return self._request

    def set_level(self, level: AgentLevel) -> None:
        level = AgentLevel(level)
        self._level = level
        self.level_menu.set_level(level)
        self.stop_button.setVisible(level in (AgentLevel.ASSIST, AgentLevel.DRIVE))
        self._render_recent()

    @property
    def recent_shown(self) -> bool:
        """Whether Recent shows: the operator's choice at this level, else on at
        Drive, where the agent acts without asking, and off below it."""
        return self._recent_choice.get(self._level, self._level is AgentLevel.DRIVE)

    def set_recent(self, rows: List[Any]) -> None:
        """What the agent did lately, newest first (`agent_recent.RecentRow`)."""
        rows = list(rows)
        if rows == self._recent:
            return
        self._recent = rows
        self._render_recent()

    def set_keep_on_top(self, on: bool) -> None:
        """Whether the floating card stays over every window, not only the app's."""
        self.pin_button.blockSignals(True)
        self.pin_button.setChecked(on)
        self.pin_button.blockSignals(False)
        self.pin_button.setIcon(
            fibsem_icon(
                "mdi:pin" if on else "mdi:pin-outline",
                color=PURPLE_COLOR if on else TEXT_MUTED_COLOR,
            )
        )
        self.pin_button.setToolTip(
            "Kept on top of every window: click to keep it over the app only"
            if on
            else "Keep on top of every window, not only the app"
        )

    def set_request(self, request: Optional[Dict[str, Any]]) -> None:
        """The agent's request waiting or being applied (the server's
        ``to_dict``), or None for none."""
        request = dict(request) if request is not None else None
        if (request or {}).get("request_id") != (self._request or {}).get("request_id"):
            self._rejecting = False
            self._details_open = False
            self._why_open = False
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
        if (suggestion or {}).get("reason") != (self._suggestion or {}).get("reason"):
            self._why_open = False
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
        """Whether the card is in its own window: its button then puts it back,
        and the pin keeps it on top of every window."""
        self.pin_button.setVisible(floating)
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
                for i in range(2)  # the level, Stop agent
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
        about = (
            self.heard_label,
            self.connect_link,
            self.recent_link,
            self.event_log_link,
        )
        for widget in about:
            self._controls.removeWidget(widget)
            self._about.removeWidget(widget)
        while self._about.count():
            self._about.takeAt(0)
        if self._wide:
            for widget in about:
                self._controls.addWidget(widget)
        else:
            self._about.addWidget(self.heard_label)
            self._about.addWidget(self.connect_link)
            self._about.addStretch(1)
            self._about.addWidget(self.recent_link)
            self._about.addWidget(self.event_log_link)

    # --- what is shown --------------------------------------------------------

    def _render(self) -> None:
        _clear(self._body)
        self._body.setDirection(
            QBoxLayout.LeftToRight if self._wide else QBoxLayout.TopToBottom
        )
        request = self._request
        self.accept_button = self.details_link = self.why_link = None
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
        # "Waiting for you" shows only when something waits.
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
        self._add_why(right, request.get("reason", ""))
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
        self._add_why(right, suggestion.get("reason", ""))
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

    def _add_why(self, column: QVBoxLayout, reason: str) -> None:
        """The agent's reason: always shown, as it is what makes an approval
        mean something, but clamped to `WHY_LINES` lines with "more" (FIB-1412)."""
        column.addWidget(_label("Why", TEXT_MUTED_COLOR, small=True))
        why = _label(reason, TEXT_COLOR, wrap=True)
        why.setTextInteractionFlags(Qt.TextSelectableByMouse)  # type: ignore
        column.addWidget(why)
        self.why_label, self.why_link = why, None
        metrics = why.fontMetrics()
        width = 260 if self._wide else max(self.width() - 60, 200)
        lines = metrics.boundingRect(
            0,
            0,
            width,
            10_000,
            Qt.TextWordWrap,
            reason,  # type: ignore
        ).height() / max(metrics.lineSpacing(), 1)
        if lines <= WHY_LINES + 0.5:
            return
        if not self._why_open:
            why.setMaximumHeight(metrics.lineSpacing() * WHY_LINES)
        self.why_link = _link("less" if self._why_open else "more")
        self.why_link.clicked.connect(self._toggle_why)
        column.addWidget(self.why_link, 0, Qt.AlignLeft)  # type: ignore

    def _toggle_why(self) -> None:
        self._why_open = not self._why_open
        self._render()

    def _toggle_recent(self) -> None:
        self._recent_choice[self._level] = not self.recent_shown
        self._render_recent()

    def _render_recent(self) -> None:
        """Recent, under what waits: one line per action, how it ended at the
        right, a click away from the event log."""
        _clear(self._recent_rows)
        shown = self.recent_shown
        self.recent_link.setText("Hide recent" if shown else "Show recent")
        self.recent_buttons = []
        if not shown:
            self.updateGeometry()
            return
        self._recent_rows.addWidget(_section("Recent"))
        if not self._recent:
            self._recent_rows.addWidget(
                _label("Nothing yet.", TEXT_MUTED_COLOR, small=True)
            )
        for row in self._recent:
            line = QHBoxLayout()
            line.setSpacing(8)
            button = QPushButton(f"{row.when}   {row.text}")
            button.setObjectName("agentRecent")
            button.setCursor(Qt.PointingHandCursor)  # type: ignore
            button.setToolTip(f"{row.text}\n\nOpen the event log")
            # A long line is cut by the card, not allowed to widen it.
            button.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            button.clicked.connect(self.event_log_clicked)
            line.addWidget(button, 1)
            line.addWidget(
                _label(row.outcome, _TONES.get(row.tone, TEXT_MUTED_COLOR), small=True)
            )
            self._recent_rows.addLayout(line)
            self.recent_buttons.append(button)
        self.updateGeometry()

    def _toggle_details(self) -> None:
        self._details_open = not self._details_open
        self._render()

    def _set_rejecting(self, rejecting: bool) -> None:
        self._rejecting = rejecting
        self._render()
        if rejecting and self.reason_edit is not None:
            self.reason_edit.setFocus()
