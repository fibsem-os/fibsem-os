"""The agent card: the agent's controls in the window.

The chip in the status bar opens it. It holds what waits for the operator, in
one line with the way to the web Agent page, where it is answered; what the
agent is doing on the microscope now; the level, a menu of the four with what
each allows, and Stop agent; links to the Agent page and the event
log; and who the agent is, when it was last heard and its session.

The operator answers, messages and reads the agent on the web Agent page, not
here. Like the chip, the card only shows what it is told and says what was
clicked. What a click does is the window's.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import (
    QAction,
    QActionGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from fibsem.ai.agent.levels import AgentLevel
from fibsem.theme.tokens import (
    ACCENT_COLOR,
    BORDER_COLOR,
    PURPLE_COLOR,
    RED_COLOR,
    SURFACE_COLOR,
    TEXT_COLOR,
    TEXT_MUTED_COLOR,
    TEXT_STRONG_COLOR,
    WAITING_COLOR,
    WARN_COLOR,
)
from fibsem.ui.agent.style import (
    LEVEL_MENU,
    LEVEL_NAMES,
    LEVEL_STYLES,
    LEVEL_TIPS,
    LEVEL_WORDS,
    STOP_CHIP,
)
from fibsem.ui.icon import fibsem_icon

_CARD = f"""
    #agentCard {{ background: {SURFACE_COLOR}; }}
    #agentCard QLabel {{ background: transparent; }}
    #agentRule {{ background: {BORDER_COLOR}; }}
    #agentLink {{
        background: transparent; color: {ACCENT_COLOR}; border: none;
        padding: 0px; text-align: left;
    }}
    #agentLink:hover {{ color: #7bc0ff; }}
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


class _ElidedLabel(QLabel):
    """One line that ends "…" where the card cuts it, never mid-word with no
    sign it goes on."""

    def __init__(self, color: str) -> None:
        super().__init__()
        self._full = ""
        self.setStyleSheet(f"color: {color};")
        font = self.font()
        font.setPointSizeF(font.pointSizeF() - 1)
        self.setFont(font)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt's name
        self._full = text
        self.setToolTip(text)
        self._elide()

    def text(self) -> str:
        return self._full

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt's name
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        shown = self.fontMetrics().elidedText(
            self._full,
            Qt.ElideRight,  # type: ignore
            max(0, self.width()),
        )
        super().setText(shown)


def _link(text: str) -> QPushButton:
    button = QPushButton(text)
    button.setObjectName("agentLink")
    button.setCursor(Qt.PointingHandCursor)  # type: ignore
    font = button.font()
    font.setPointSizeF(font.pointSizeF() - 1)
    button.setFont(font)
    return button


def last_heard_text(seconds: Optional[float], client: Optional[str] = None) -> str:
    """Who the agent is and when its token was last used: "claude-code/2.1.3"
    reads "claude-code"."""
    who = client.split("/", 1)[0] if client else "Agent"
    if seconds is None:
        return f"{who} · not heard from yet"
    if seconds < 60:
        return f"{who} · last heard {int(seconds)} s ago"
    return f"{who} · last heard {int(seconds // 60)} min ago"


def waiting_text(
    request: Optional[Dict[str, Any]],
    suggestion: Optional[Dict[str, Any]],
    asking: Optional[Dict[str, Any]],
) -> Optional[tuple]:
    """``(word, what)`` for what waits for the operator, in the order the chip
    shows them, or None. A request being applied waits for no one."""
    if asking is not None:
        who = str(asking.get("client") or "An agent").split("/", 1)[0]
        return "Connect", f"{who} asks to connect"
    if request is not None and request.get("status") == "pending":
        actions = request.get("actions") or []
        first = (
            (actions[0].get("change") or actions[0].get("tool", "")) if actions else ""
        )
        first = str(first).split("\n", 1)[0]
        more = f", and {len(actions) - 1} more" if len(actions) > 1 else ""
        return "Asks", f"{first}{more}"
    if suggestion is not None:
        return "Suggests", str(suggestion.get("decision") or suggestion.get("proposal"))
    return None


class _LevelRow(QWidget):
    """One level in the menu: a dot, the level's name, and what it allows under
    it. Shows its action's checked state, and triggers it on click."""

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
        text.addWidget(_label(action.text(), TEXT_STRONG_COLOR))
        text.addWidget(_label(line, TEXT_MUTED_COLOR, small=True))
        layout.addLayout(text, 1)
        action.toggled.connect(self._paint)
        self._paint(action.isChecked())

    def _paint(self, checked: bool) -> None:
        self._dot.setText("●" if checked else "○")
        colour = PURPLE_COLOR if checked else TEXT_MUTED_COLOR
        self._dot.setStyleSheet(f"color: {colour};")

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:  # type: ignore
            self._menu.close()
            self._action.trigger()


class LevelMenu(QPushButton):
    """The level in force, drawn as the status bar chip draws it ("Assisting");
    clicked, a menu of the four levels with what each allows. The
    level changes rarely and each change matters, so it is one click further
    than a switch would be. Says what was picked; shows what it is told."""

    level_chosen = pyqtSignal(object)  # AgentLevel

    def __init__(self) -> None:
        super().__init__()
        self.setCursor(Qt.PointingHandCursor)  # type: ignore
        self.setToolTip("Change what the agent may do")
        self.menu_ = QMenu(self)
        self.menu_.setStyleSheet(LEVEL_MENU)
        group = QActionGroup(self)
        group.setExclusive(True)
        self.actions_: Dict[AgentLevel, QWidgetAction] = {}
        for level, name in LEVEL_NAMES.items():
            action = QWidgetAction(self)
            action.setText(name)
            action.setCheckable(True)
            action.setDefaultWidget(_LevelRow(action, self.menu_, LEVEL_TIPS[level]))
            action.triggered.connect(
                lambda _checked=False, chosen=level: self._on_chosen(chosen)
            )
            group.addAction(action)
            self.menu_.addAction(action)
            self.actions_[level] = action
        self.setMenu(self.menu_)
        self._level: Optional[AgentLevel] = None
        # One width for every level, so the controls beside it do not shift.
        widths = []
        for level in LEVEL_NAMES:
            self.set_level(level)
            self.ensurePolished()
            widths.append(self.sizeHint().width())
        self.setFixedWidth(max(widths))

    @property
    def level(self) -> Optional[AgentLevel]:
        return self._level

    def set_level(self, level: AgentLevel) -> None:
        self._level = AgentLevel(level)
        self.setText(f"{LEVEL_WORDS[self._level].split(': ', 1)[1]}  ▾")
        self.setStyleSheet(LEVEL_STYLES[self._level])
        self.actions_[self._level].setChecked(True)

    def choose(self, level: AgentLevel) -> None:
        """As a click on *level*'s row."""
        self.actions_[level].trigger()

    def _on_chosen(self, level: AgentLevel) -> None:
        # The menu shows what the window says, not the click: put it back
        # until the window answers.
        if self._level is not None:
            self.actions_[self._level].setChecked(True)
        if level is not self._level:
            self.level_chosen.emit(level)


class AgentCard(QFrame):
    """What waits, what the agent is doing, and its controls."""

    level_chosen = pyqtSignal(object)  # AgentLevel
    stop_clicked = pyqtSignal()
    event_log_clicked = pyqtSignal()
    chat_clicked = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("agentCard")
        self.setAttribute(Qt.WA_StyledBackground)  # type: ignore
        self.setStyleSheet(_CARD)
        self._request: Optional[Dict[str, Any]] = None
        self._suggestion: Optional[Dict[str, Any]] = None
        self._asking: Optional[Dict[str, Any]] = None
        self._now: Optional[str] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 12)
        layout.setSpacing(8)

        # What waits for the operator, answered on the Agent page.
        self.waiting_row = QWidget()
        waiting = QVBoxLayout(self.waiting_row)
        waiting.setContentsMargins(0, 0, 0, 0)
        waiting.setSpacing(2)
        line = QHBoxLayout()
        line.setSpacing(8)
        self.waiting_word = _label("", WAITING_COLOR, bold=True)
        self.waiting_word.setAlignment(Qt.AlignTop)  # type: ignore
        line.addWidget(self.waiting_word)
        self.waiting_label = _label("", TEXT_STRONG_COLOR, wrap=True)
        line.addWidget(self.waiting_label, 1)
        waiting.addLayout(line)
        self.answer_link = _link("Open in Agent page")
        self.answer_link.setToolTip("Answer the agent in your browser.")
        self.answer_link.clicked.connect(self.chat_clicked)
        waiting.addWidget(self.answer_link, 0, Qt.AlignLeft)  # type: ignore
        layout.addWidget(self.waiting_row)

        # What the agent is doing on the microscope, at every level.
        self.now_row = QWidget()
        now = QHBoxLayout(self.now_row)
        now.setContentsMargins(0, 0, 0, 0)
        now.setSpacing(8)
        now.addWidget(_label("Now", TEXT_MUTED_COLOR, small=True))
        self.now_label = _label("", TEXT_COLOR, wrap=True)
        now.addWidget(self.now_label, 1)
        layout.addWidget(self.now_row)

        # Under what waits and Now; with neither, the controls lead.
        self.rule = QFrame()
        self.rule.setObjectName("agentRule")
        self.rule.setFixedHeight(1)
        layout.addWidget(self.rule)

        controls = QHBoxLayout()
        controls.setSpacing(10)
        self.level_menu = LevelMenu()
        self.level_menu.level_chosen.connect(self.level_chosen)
        controls.addWidget(self.level_menu)
        self.stop_button = QPushButton(" Stop agent")
        self.stop_button.setIcon(
            fibsem_icon("mdi:stop-circle-outline", color=RED_COLOR)
        )
        self.stop_button.setStyleSheet(STOP_CHIP)
        self.stop_button.setToolTip(
            "Stop the agent acting: down to Watch at once. A running workflow "
            "keeps running; its own Stop is beside the chip."
        )
        self.stop_button.clicked.connect(self.stop_clicked)
        self.stop_button.ensurePolished()
        self.stop_button.setMinimumWidth(self.stop_button.sizeHint().width())
        controls.addWidget(self.stop_button)
        controls.addStretch(1)
        self.chat_link = _link("Agent page")
        self.chat_link.setToolTip(
            "Answer, message and follow the agent in your browser."
        )
        self.chat_link.clicked.connect(self.chat_clicked)
        controls.addWidget(self.chat_link)
        self.event_log_link = _link("Event log")
        self.event_log_link.setToolTip(
            "What the agent, the operator and the run did, in your browser."
        )
        self.event_log_link.clicked.connect(self.event_log_clicked)
        controls.addWidget(self.event_log_link)
        layout.addLayout(controls)

        # Who the agent is, and its session as it reported it.
        self.heard_label = _label(last_heard_text(None), TEXT_MUTED_COLOR, small=True)
        layout.addWidget(self.heard_label)
        self.session_label = _ElidedLabel(TEXT_MUTED_COLOR)
        self.session_label.hide()
        layout.addWidget(self.session_label)

        self.set_level(AgentLevel.WATCH)
        self._render()

    # --- told by the window ---------------------------------------------------

    @property
    def request(self) -> Optional[Dict[str, Any]]:
        return self._request

    def set_level(self, level: AgentLevel) -> None:
        level = AgentLevel(level)
        self.level_menu.set_level(level)
        self.stop_button.setVisible(level in (AgentLevel.ASSIST, AgentLevel.DRIVE))

    def set_request(self, request: Optional[Dict[str, Any]]) -> None:
        """The agent's request waiting or being applied (the server's
        ``to_dict``), or None for none."""
        self._request = dict(request) if request is not None else None
        self._render()

    def set_asking(self, asking: Optional[Dict[str, Any]]) -> None:
        """An agent on this PC asking to connect (``AgentSession.asking``),
        or None for none."""
        self._asking = dict(asking) if asking is not None else None
        self._render()

    def set_suggestion(self, suggestion: Optional[Dict[str, Any]]) -> None:
        """The agent's suggestion on the pending proposal (``proposal`` and the
        ``decision`` approving it makes), or None for none."""
        self._suggestion = dict(suggestion) if suggestion is not None else None
        self._render()

    def set_now(self, action: Optional[str]) -> None:
        """What the agent is doing on the microscope, or None."""
        self._now = action
        self.now_label.setText(action or "")
        self.now_row.setVisible(bool(action))
        self.rule.setVisible(bool(action) or not self.waiting_row.isHidden())

    def set_last_heard(self, text: Optional[str]) -> None:
        self.heard_label.setText(text or last_heard_text(None))

    def set_session(self, text: Optional[str], low: bool = False) -> None:
        """The agent's session in one line, or none shown; in the warning
        colour when its budget is low."""
        self.session_label.setText(text or "")
        self.session_label.setStyleSheet(
            f"color: {WARN_COLOR if low else TEXT_MUTED_COLOR};"
        )
        self.session_label.setVisible(bool(text))

    def _render(self) -> None:
        waiting = waiting_text(self._request, self._suggestion, self._asking)
        self.waiting_row.setVisible(waiting is not None)
        if waiting is not None:
            self.waiting_word.setText(waiting[0])
            self.waiting_label.setText(waiting[1])
        self.set_now(self._now)
        self.adjustSize()
