"""The agent card: what the agent asks of the operator, and its controls (FIB-1404).

The chip in the status bar opens it. It holds, top to bottom:

* what the agent is doing on the microscope now, at every level (FIB-1414);
* what is waiting for the operator: the agent's request, its actions one plain
  line each as the server wrote them (FIB-1403), "Show details" for every value,
  the agent's reason, under it the image the reason's values were measured on
  with them drawn on it (FIB-1411), and Reject… and Approve; while an approved
  request runs, each action's progress. Or the agent's suggested answer to the
  run's question, with Accept (FIB-1407);
* what the agent did lately, Recent: open at first at Drive, where it acts
  without asking, folded to its last line below it (FIB-1414); the event log
  at its right;
* a line to message the agent, once one has been heard from, which it hears
  from its watch and answers with a note (FIB-1423);
* the controls: the level, drawn as the chip draws it, a menu of the four with
  what each allows (FIB-1418); Stop agent, drawn the same way; who the agent is
  and when it was last heard from.

It is wide: the actions on the left, the reason and the decision on the right.
Narrower than `STACK_BELOW` px it stacks them, so the same card floats as a
window the operator sizes (the button at its top right floats it, and puts it
back on the chip; its pin keeps it on top of every window).

Like the chip, it only shows what it is told and says what was clicked. What a
click does is the window's.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QMouseEvent, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QActionGroup,
    QBoxLayout,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from fibsem.ai.agent.levels import AgentLevel
from fibsem.ui.agent.style import (
    LEVEL_MENU,
    LEVEL_NAMES,
    LEVEL_STYLES,
    LEVEL_TIPS,
    LEVEL_WORDS,
    STOP_CHIP,
)
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
    SURFACE_COLOR,
    TEXT_COLOR,
    TEXT_MUTED_COLOR,
    TEXT_STRONG_COLOR,
    WAITING_COLOR,
)

# The agent's reason, before "more".
WHY_LINES = 3
# When wide, the steps and the reason sit left and the image the agent's values
# were measured on, drawn on it (FIB-1411), right: at least `PICTURE_MIN` wide,
# and any width past the steps' `STEPS_MIN`, up to the image's own width. With
# details open, or narrow, they stack and the image takes the card's width.
PICTURE_MIN = 420
STEPS_MIN = 300
# The legend drawn on the image: this size on screen, however wide it shows.
LEGEND_PX = 11
# The image, clicked: at most this wide.
ENLARGED_WIDTH = 1024

# How a Recent row ended (agent_recent's tones), as a colour.
_TONES = {
    "waiting": WAITING_COLOR,
    "ok": OK_COLOR,
    "error": RED_COLOR,
    "muted": TEXT_MUTED_COLOR,
}

# Narrower than this, the reason and the decision go under the actions.
STACK_BELOW = 600
# The narrowest the stacked card goes.
NARROWEST = 360
# The longest message to the agent, in characters (FIB-1423).
MESSAGE_LIMIT = 2000

_CARD = f"""
    #agentCard {{ background: {SURFACE_COLOR}; }}
    #agentCard QLabel {{ background: transparent; }}
    #agentActions {{ border: 1px solid {BORDER_COLOR}; border-radius: 3px; }}
    #agentDetails, #agentDetailsTable {{
        background: {PANEL_COLOR}; border-radius: 3px;
    }}
    #agentRule {{ background: {BORDER_COLOR}; }}
    #agentCard QLineEdit {{
        background: {PANEL_COLOR}; border: 1px solid {BORDER_COLOR}; border-radius: 3px;
        padding: 4px 6px;
        color: {TEXT_STRONG_COLOR};
    }}
    #agentLink {{
        background: transparent; color: {ACCENT_COLOR}; border: none;
        padding: 0px; text-align: left;
    }}
    #agentLink:hover {{ color: #7bc0ff; }}
    #agentRecent {{ color: {TEXT_COLOR}; padding: 1px 0px; }}
    #agentRecent:hover {{ color: {TEXT_STRONG_COLOR}; }}
    #agentRecentRows {{ background: {PANEL_COLOR}; border-radius: 3px; }}
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
    sign it goes on (sim session 2026-10-16: "Stage · Stable move in the")."""

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


class _RecentLine(QLabel):
    """One Recent action: wraps rather than being cut by the card, and opens
    the event log when clicked."""

    clicked = pyqtSignal()

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.setObjectName("agentRecent")
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)  # type: ignore
        self.setCursor(Qt.PointingHandCursor)  # type: ignore

    def resizeEvent(self, event) -> None:
        # The card is sized from its hints, which ignore wrapping: hold the
        # height the wrapped text needs at this width, and no more, or a row
        # keeps a blank line from a narrower pass (sim session 2026-10-16).
        super().resizeEvent(event)
        height = self.heightForWidth(self.width())
        if height > 0 and height != self.height():
            self.setFixedHeight(height)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:  # type: ignore
            self.click()
        super().mouseReleaseEvent(event)

    def click(self) -> None:
        self.clicked.emit()


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
    clicked, a menu of the four levels with what each allows (FIB-1418). The
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


class _Picture(QLabel):
    """An image scaled to its column, its legend drawn on it; a click shows it
    at full size."""

    def __init__(
        self, image: QImage, width: int, caption: str, legend: Sequence = ()
    ) -> None:
        super().__init__()
        self.setObjectName("agentImage")
        self._image = image
        self._caption = caption
        self._legend = tuple(legend)
        self.set_width(width)
        self.setCursor(Qt.PointingHandCursor)  # type: ignore
        # What it is ("FIB · 07:41"): here, and the enlarged window's title.
        self.setToolTip(
            f"{caption} · click to enlarge" if caption else "Click to enlarge"
        )
        self.window_: Optional[QDialog] = None

    def set_width(self, width: int) -> None:
        """Shown *width* wide, at most the image's own width."""
        width = max(1, min(int(width), self._image.width()))
        if self.pixmap() is not None and self.pixmap().width() == width:
            return
        pixmap = self._drawn(width)
        self.setPixmap(pixmap)
        # All of it: squeezed, the label would show its middle and cut the legend.
        self.setFixedHeight(pixmap.height())

    def _drawn(self, width: int) -> QPixmap:
        """The image *width* wide with its legend, `LEGEND_PX` on screen."""
        from fibsem.ui.agent.previews import draw_legend

        pixmap = QPixmap.fromImage(self._image)
        if pixmap.width() != width:
            pixmap = pixmap.scaledToWidth(width, Qt.SmoothTransformation)  # type: ignore
        if self._legend:
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            draw_legend(painter, self._legend, LEGEND_PX)
            painter.end()
        return pixmap

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:  # type: ignore
            self.enlarge()

    def enlarge(self) -> None:
        dialog = QDialog(self.window())
        dialog.setWindowTitle(self._caption or "Agent")
        dialog.setAttribute(Qt.WA_DeleteOnClose)  # type: ignore
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(0, 0, 0, 0)
        full = QLabel()
        full.setPixmap(self._drawn(min(self._image.width(), ENLARGED_WIDTH)))
        layout.addWidget(full)
        dialog.show()
        self.window_ = dialog


class AgentCard(QFrame):
    """What the agent asks, and its controls."""

    level_chosen = pyqtSignal(object)  # AgentLevel
    stop_clicked = pyqtSignal()
    event_log_clicked = pyqtSignal()
    connect_clicked = pyqtSignal()
    approve_clicked = pyqtSignal(str)  # request id
    reject_clicked = pyqtSignal(str, str)  # request id, the operator's reason
    accept_clicked = pyqtSignal()  # the suggestion shown
    dismiss_clicked = pyqtSignal(str)  # the suggestion shown, the operator's why
    float_clicked = pyqtSignal()  # float the card, or put it back on the chip
    keep_on_top_toggled = pyqtSignal(bool)  # the floating card over every window
    message_sent = pyqtSignal(str)  # the operator's message to the agent

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("agentCard")
        self.setAttribute(Qt.WA_StyledBackground)  # type: ignore
        self.setStyleSheet(_CARD)
        self._request: Optional[Dict[str, Any]] = None
        self._suggestion: Optional[Dict[str, str]] = None
        self.accept_button: Optional[QPushButton] = None
        self.dismiss_button: Optional[QPushButton] = None
        self._now: Optional[str] = None
        # The images under Why, with their captions: the request's, the
        # suggestion's.
        self._images: Dict[str, Tuple[QImage, str, Tuple]] = {}
        # Why there is no image, in its place: "No FIB image of 05 yet".
        self._notes: Dict[str, str] = {}
        self.image_label: Optional[_Picture] = None
        self._picture_holder: Optional[QWidget] = None
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

        # What the agent is doing on the microscope, at every level (FIB-1414):
        # under the title, beside whatever waits.
        self.now_row = QWidget()
        now = QHBoxLayout(self.now_row)
        now.setContentsMargins(0, 0, 0, 0)
        now.setSpacing(8)
        now.addWidget(_label("Now", TEXT_MUTED_COLOR, small=True))
        self.now_label = _label("", TEXT_COLOR, wrap=True)
        now.addWidget(self.now_label, 1)
        layout.addWidget(self.now_row)

        # What is waiting: rebuilt whole on each change, a handful of rows.
        self._body = QBoxLayout(QBoxLayout.LeftToRight)
        self._body.setSpacing(16)
        layout.addLayout(self._body)

        # What the agent did lately (FIB-1414), folded to its last line or open,
        # per level; the event log, the whole record, at its right.
        recent = QHBoxLayout()
        recent.setSpacing(8)
        self.recent_link = _link("")
        self.recent_link.setToolTip("The agent's last few actions and how each ended")
        self.recent_link.clicked.connect(self._toggle_recent)
        recent.addWidget(self.recent_link)
        self.recent_last = _ElidedLabel(TEXT_MUTED_COLOR)
        self.recent_last.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        recent.addWidget(self.recent_last, 1)
        self.event_log_link = _link("Event log")
        self.event_log_link.setToolTip(
            "What the agent, the operator and the run did, in your browser."
        )
        self.event_log_link.clicked.connect(self.event_log_clicked)
        recent.addWidget(self.event_log_link)
        layout.addLayout(recent)
        self._recent_rows = QVBoxLayout()
        self._recent_rows.setSpacing(2)
        layout.addLayout(self._recent_rows)

        # A message to the agent (FIB-1423): it hears it from its watch and
        # answers with a note. It is not an approval: the level still decides.
        # Shown once an agent has been heard from: until then no one listens.
        self.message_row = QWidget()
        message = QHBoxLayout(self.message_row)
        message.setContentsMargins(0, 0, 0, 0)
        message.setSpacing(8)
        self.message_edit = QLineEdit()
        self.message_edit.setPlaceholderText("Message the agent")
        self.message_edit.setMaxLength(MESSAGE_LIMIT)
        self.message_edit.setToolTip(
            "The agent reads it and answers with a note. It is not an approval: "
            "what the agent may do still depends on its level."
        )
        self.message_edit.returnPressed.connect(self._send_message)
        message.addWidget(self.message_edit, 1)
        self.send_button = _button("Send")
        self.send_button.clicked.connect(self._send_message)
        message.addWidget(self.send_button)
        layout.addWidget(self.message_row)

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
        self.stop_button.setStyleSheet(STOP_CHIP)
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
        self._about = QHBoxLayout()
        self._about.setSpacing(10)
        layout.addLayout(self._about)
        # A tall floating window keeps the rows packed at the top (FIB-1427).
        layout.addStretch(1)

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
        """What the agent did lately, newest first (`recent.RecentRow`)."""
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
        if suggestion is None or suggestion.get("question") != (
            self._suggestion or {}
        ).get("question"):
            self._rejecting = False
        if (suggestion or {}).get("reason") != (self._suggestion or {}).get("reason"):
            self._why_open = False
        self._suggestion = suggestion
        self._render()

    def set_image(
        self,
        image: Optional[QImage],
        caption: str = "",
        suggestion: bool = False,
        legend: Sequence = (),
    ) -> None:
        """The image the waiting request's (or, *suggestion*, the suggested
        answer's) values were measured on, drawn on it, with what it is
        ("FIB · 07:41") and its *legend* (`previews.draw_legend`'s
        entries); None for none, *caption* then why there is none."""
        key = "suggestion" if suggestion else "request"
        if image is None or image.isNull():
            had = self._images.pop(key, None) is not None
            note = self._notes.pop(key, "")
            if caption:
                self._notes[key] = caption
            if not had and note == caption:
                return
        else:
            self._notes.pop(key, None)
            self._images[key] = (image, caption, tuple(legend))
        self._render()

    def set_now(self, action: Optional[str]) -> None:
        """What the agent is doing on the microscope, or None."""
        if action == self._now:
            return
        self._now = action
        self._paint_now()

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
        self.message_row.setVisible(heard)

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
        elif self.image_label is not None:
            # The image follows the card's width.
            if self._picture_holder is not None:
                self._picture_holder.setFixedWidth(self._picture_width())
            self.image_label.set_width(self._picture_width())

    def _place_about(self) -> None:
        about = (self.heard_label, self.connect_link)
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

    # --- what is shown --------------------------------------------------------

    def _render(self) -> None:
        _clear(self._body)
        self._body.setDirection(
            QBoxLayout.TopToBottom if self._stacked() else QBoxLayout.LeftToRight
        )
        # The body's widgets are gone: so is the picture (else a resize touches it).
        self._picture_holder = self.image_label = None
        request = self._request
        self.accept_button = self.details_link = self.why_link = None
        self.dismiss_button = None
        self.approve_button = self.reject_button = None
        self.confirm_reject_button = self.back_button = None
        self.reason_edit = None
        if request is None and self._suggestion is None:
            self._dot.setStyleSheet(f"color: {PURPLE_COLOR};")
            self._title.setText("Agent")
            self._paint_now()
            self.updateGeometry()
            return
        self._dot.setStyleSheet(f"color: {WAITING_COLOR};")
        self._paint_now()
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
        if not self._stacked():
            holder.setFixedWidth(self._picture_width())
            self._picture_holder = holder
        self._body.addWidget(holder)
        self.updateGeometry()

    def _paint_now(self) -> None:
        """The Now line: what the agent is doing, shown only while it does
        something (FIB-1426)."""
        self.now_label.setText(self._now or "")
        self.now_row.setVisible(bool(self._now))

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
        self.details_link = None
        if any(action.get("details") for action in actions):
            self.details_link = _link(
                "Hide details" if self._details_open else "Show details"
            )
            self.details_link.clicked.connect(self._toggle_details)
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
            what = _label(
                action.get("what") or action.get("tool", ""),
                TEXT_MUTED_COLOR,
                small=True,
            )
            if index == 0 and self.details_link is not None:
                # Top right of the list.
                head = QHBoxLayout()
                head.addWidget(what, 1)
                head.addWidget(self.details_link, 0, Qt.AlignTop)  # type: ignore
                text.addLayout(head)
            else:
                text.addWidget(what)
            change = _label(action.get("change") or "", TEXT_STRONG_COLOR, wrap=True)
            change.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            text.addWidget(change)
            if self._details_open and action.get("details"):
                text.addSpacing(4)
                text.addWidget(_Details(action["details"]))
            row.addLayout(text, 1)
            rows.addLayout(row)
        left.addWidget(box)
        left.addSpacing(4)
        self._add_reason(left, request.get("reason", ""))
        left.addStretch(1)

        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(8)
        self._add_picture(right, "request")
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
        left.addSpacing(4)
        self._add_reason(left, suggestion.get("reason", ""))
        left.addStretch(1)

        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(8)
        self._add_picture(right, "suggestion")
        right.addStretch(1)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.back_button = self.confirm_reject_button = self.reason_edit = None
        if self._rejecting and self._request is None:
            # Dismiss, with why if the operator likes: recorded for the agent
            # (FIB-1424). The question stays up.
            self.reason_edit = QLineEdit()
            self.reason_edit.setPlaceholderText("Tell the agent why (optional)")
            right.addWidget(self.reason_edit)
            self.back_button = _button("Back")
            self.back_button.clicked.connect(lambda: self._set_rejecting(False))
            self.dismiss_button = _button("Dismiss", primary=True)
            self.dismiss_button.clicked.connect(
                lambda: self.dismiss_clicked.emit(
                    self.reason_edit.text() if self.reason_edit else ""
                )
            )
            buttons.addWidget(self.back_button)
            buttons.addWidget(self.dismiss_button)
            right.addLayout(buttons)
            return left, right
        self.dismiss_button = _button("Dismiss…")
        self.dismiss_button.setToolTip(
            "Decline the suggestion and tell the agent why. The question stays "
            "up for you to answer."
        )
        self.dismiss_button.clicked.connect(lambda: self._set_rejecting(True))
        self.accept_button = _button("Accept", primary=True)
        self.accept_button.setToolTip(
            "Answer the question with this, as your answer. The record names the "
            "agent as the one who suggested it."
        )
        self.accept_button.clicked.connect(self.accept_clicked)
        buttons.addWidget(self.dismiss_button)
        buttons.addWidget(self.accept_button)
        right.addLayout(buttons)
        return left, right

    def _stacked(self) -> bool:
        """The picture under the steps and the reason, not beside them: narrow,
        or with details open, so their table has the card's width. With no
        picture, the buttons go under them too: an empty column beside them
        would squeeze the steps for nothing."""
        return (
            not self._wide
            or (self._details_open and self._request is not None)
            or not self._showing_picture()
        )

    def _showing_picture(self) -> bool:
        """Whether what waits has a picture. A note in its place ("No FIB image
        of 05 yet") is a line under the steps, not a column of its own (sim
        session 2026-10-16: it took the picture's 650 px)."""
        key = "suggestion" if self._request is None else "request"
        return key in self._images

    def _inner_width(self) -> int:
        margins = self.layout().contentsMargins()
        return max(self.width() - margins.left() - margins.right(), 200)

    def _picture_width(self) -> int:
        """The picture's column: the card's width stacked, else `PICTURE_MIN`
        and whatever the steps' `STEPS_MIN` leaves."""
        inner = self._inner_width()
        if self._stacked():
            return inner
        return max(PICTURE_MIN, inner - STEPS_MIN - self._body.spacing())

    def _why_width(self) -> int:
        """The reason's width: the steps' column."""
        if self._stacked():
            return self._inner_width()
        return max(
            self._inner_width() - self._picture_width() - self._body.spacing(), 200
        )

    def _add_picture(self, column: QVBoxLayout, image: str) -> None:
        """The *image* the values were measured on (FIB-1411), its legend on it."""
        if image not in self._images:
            if self._notes.get(image):
                column.addWidget(
                    _label(self._notes[image], TEXT_MUTED_COLOR, small=True, wrap=True)
                )
            return
        picture, caption, legend = self._images[image]
        label = _Picture(picture, self._picture_width(), caption, legend)
        self.image_label = label
        column.addWidget(label)

    def _add_reason(self, column: QVBoxLayout, reason: str) -> None:
        """The agent's reason: always shown, as it is what makes an approval
        mean something, but clamped to `WHY_LINES` lines with "more" (FIB-1412)."""
        column.addWidget(_label("Why", TEXT_MUTED_COLOR, small=True))
        why = _label(reason, TEXT_COLOR, wrap=True)
        why.setTextInteractionFlags(Qt.TextSelectableByMouse)  # type: ignore
        column.addWidget(why)
        self.why_label, self.why_link = why, None
        metrics = why.fontMetrics()
        width = self._why_width()
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
        """Recent, under what waits: folded, its newest line beside the link;
        open, one row per action (a long one wraps), how it ended at the right,
        a click away from the event log."""
        _clear(self._recent_rows)
        shown = self.recent_shown
        self.recent_link.setText("Recent ▾" if shown else "Recent ▸")
        newest = self._recent[0] if self._recent else None
        last = ""
        if newest is not None and not shown:
            # A message or a note has no outcome to add.
            last = ", ".join(t for t in (newest.text, newest.outcome) if t)
            last = f"last: {last}"
        self.recent_last.setText(last)
        self.recent_buttons = []
        if not shown:
            self.updateGeometry()
            return
        box = QFrame()
        box.setObjectName("agentRecentRows")
        rows = QVBoxLayout(box)
        rows.setContentsMargins(10, 6, 10, 6)
        rows.setSpacing(2)
        if not self._recent:
            rows.addWidget(_label("Nothing yet.", TEXT_MUTED_COLOR, small=True))
        for row in self._recent:
            line = QHBoxLayout()
            line.setSpacing(12)
            top = Qt.AlignTop  # type: ignore
            line.addWidget(_label(row.when, TEXT_MUTED_COLOR, small=True), 0, top)
            # A long line wraps under itself, not allowed to widen the card.
            button = _RecentLine(row.text)
            button.setToolTip("Open the event log")
            button.clicked.connect(self.event_log_clicked)
            line.addWidget(button, 1)
            line.addWidget(
                _label(row.outcome, _TONES.get(row.tone, TEXT_MUTED_COLOR), small=True),
                0,
                top,
            )
            # At which level it was done (FIB-1426): Assist's were approved.
            level = str(getattr(row, "level", "") or "")
            tag = _label(level.title(), TEXT_MUTED_COLOR, small=True)
            tag.setObjectName("agentLevelTag")
            tag.setVisible(bool(level))
            line.addWidget(tag, 0, top)
            rows.addLayout(line)
            self.recent_buttons.append(button)
        self._recent_rows.addWidget(box)
        self.updateGeometry()

    def _send_message(self) -> None:
        text = self.message_edit.text().strip()
        if not text:
            return
        self.message_edit.clear()
        self.message_sent.emit(text)

    def _toggle_details(self) -> None:
        self._details_open = not self._details_open
        self._render()

    def _set_rejecting(self, rejecting: bool) -> None:
        self._rejecting = rejecting
        self._render()
        if rejecting and self.reason_edit is not None:
            self.reason_edit.setFocus()
