"""The agent chip in the status bar: the agent's level, set in two clicks (FIB-1035).

The agent is a second operator at the app, and its level is the one knob: Off, Watch,
Assist or Drive. The chip shows the level; clicking it opens a menu that sets it, with
when the agent was last heard from at the bottom. No dialog.

Like `WorkflowControls`, this only shows what it is told and says what was picked.
What a pick does -- setting the App Server's level -- is the window's.

Assist is not offered until its approvals exist (FIB-1283).
"""

from __future__ import annotations

from typing import Callable, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import (
    QAction,
    QActionGroup,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from fibsem.server.levels import AgentLevel
from fibsem.ui.icon import fibsem_icon
from fibsem.ui.tokens import (
    BORDER_COLOR,
    GRAY_ICON_COLOR,
    GRAY_SECONDARY_COLOR,
    PURPLE_COLOR,
    RED_COLOR,
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
            color=TEXT_COLOR,
            background="transparent",
            border=f"1px solid {GRAY_SECONDARY_COLOR}",
        ),
        "Read only",
    ),
    AgentLevel.ASSIST: (
        "Agent: Assisting",
        _CHIP.format(
            color=TEXT_COLOR,
            background="transparent",
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

# The menu's choices, in order. Assist joins with FIB-1283.
OFFERED_LEVELS = (AgentLevel.OFF, AgentLevel.WATCH, AgentLevel.DRIVE)

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

        self.set_level(AgentLevel.WATCH)

    @property
    def level(self) -> Optional[AgentLevel]:
        return self._level

    def set_level(self, level: AgentLevel) -> None:
        """Show *level*. Does not emit: this is the window telling the chip."""
        level = AgentLevel(level)
        self._level = level
        text, style, tooltip = _LEVELS[level]
        self.setText(text)
        self.setStyleSheet(style)
        self.setToolTip(tooltip + _HELP)
        action = self._actions.get(level)
        if action is not None:
            action.setChecked(True)
        can_act = level in (AgentLevel.ASSIST, AgentLevel.DRIVE)
        self.stop_action.setVisible(can_act)
        self._stop_separator.setVisible(can_act)

    def refresh_last_heard(self) -> None:
        try:
            seconds = self._last_heard()
        except Exception:
            seconds = None
        self.last_heard_action.setText(last_heard_text(seconds))

    def _on_chosen(self, level: AgentLevel) -> None:
        if level is not self._level:
            self.level_chosen.emit(level)
