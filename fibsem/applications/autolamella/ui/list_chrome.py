"""What the workflow panel's lists share: one header, one row size, one type scale,
and the attention chip's tint.

The Workflow tab's lists -- lamellae and tasks on the Lamella page, grids and tasks
on the Grids page -- had a header each, each its own: a bold "Select All" over a
"Status" column, a larger bold "Select All" with a +, a 16 px "Grids" with a present
count. Their rows were 34, 40 and 32 px, with names at 11, 15 and 16 px. Here is the
one version they all use, so the four lists read as one widget.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QToolButton, QWidget

from fibsem.ui.icon import fibsem_icon
from fibsem.ui.tokens import (
    CANVAS_BG,
    GRAY_ICON_COLOR,
    TEXT_MUTED_COLOR,
    TEXT_STRONG_COLOR,
)

# Every row in the panel's lists, and the type in them: the name, then whatever
# qualifies it (a status, what a task waits for, a task's type).
ROW_HEIGHT = 32
NAME_PX = 13
DETAIL_PX = 12
# A chip that sits in a row: tall enough to read, short enough to leave the row its
# margin.
CHIP_HEIGHT = 22


class ListHeader(QWidget):
    """A list's header: a tick box for the whole list, the list's title and a muted
    count, then whatever the list puts on the right (`add_widget`).

    The tick box is `checkbox_all`, as it was on each list's own header: the lists
    keep their tri-state sync and only the widget they write to changed.
    """

    select_all_changed = pyqtSignal(bool)

    def __init__(
        self,
        title: str,
        title_width: Optional[int] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"background: {CANVAS_BG};")
        self.setFixedHeight(ROW_HEIGHT + 2)

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(6, 0, 6, 0)
        self._layout.setSpacing(8)

        self.checkbox_all = QCheckBox()
        self.checkbox_all.setChecked(True)
        self.checkbox_all.setStyleSheet("background: transparent;")
        self.checkbox_all.setToolTip("Select all")
        self._layout.addWidget(self.checkbox_all)

        # Title and count as one block, so a column the list adds after it (the
        # lamellae's Status) starts where the rows' column does.
        title_block = QWidget()
        title_block.setStyleSheet("background: transparent;")
        block = QHBoxLayout(title_block)
        block.setContentsMargins(0, 0, 0, 0)
        block.setSpacing(6)
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(
            f"background: transparent; color: {TEXT_STRONG_COLOR}; "
            f"font-size: {NAME_PX}px; font-weight: 600;"
        )
        self.count_label = QLabel()
        self.count_label.setStyleSheet(
            f"background: transparent; color: {TEXT_MUTED_COLOR}; "
            f"font-size: {DETAIL_PX}px;"
        )
        block.addWidget(self.title_label)
        block.addWidget(self.count_label)
        block.addStretch(1)
        if title_width is not None:
            title_block.setMinimumWidth(title_width)
        self._layout.addWidget(title_block)

        self.checkbox_all.stateChanged.connect(
            lambda state: self.select_all_changed.emit(bool(state))
        )

    def set_count(self, text: str) -> None:
        """The muted figure after the title: "6", "1 · 0 present"."""
        self.count_label.setText(text)

    def add_widget(self, widget: QWidget, stretch: int = 0) -> None:
        self._layout.addWidget(widget, stretch)

    def add_stretch(self) -> None:
        self._layout.addStretch(1)


def column_label(text: str) -> QLabel:
    """A column's name in a header, over the rows' column of that name."""
    label = QLabel(text.upper())
    label.setStyleSheet(
        f"background: transparent; color: {TEXT_MUTED_COLOR}; "
        f"font-size: 11px; font-weight: 600; letter-spacing: 0.5px;"
    )
    return label


def tinted_chip_style(colour: str, selector: str = "QToolButton") -> str:
    """A chip that sits in a row: the mode's colour as a faint fill and as the
    text, no outline. Quieter in a column of them than an outlined chip, and
    still apart from the status bar's solid one, which says what is running."""
    c = QColor(colour)
    fill = f"rgba({c.red()}, {c.green()}, {c.blue()}, 46)"
    hover = f"rgba({c.red()}, {c.green()}, {c.blue()}, 80)"
    return (
        f"{selector} {{ border: none; border-radius: 3px; padding: 0 6px 0 4px; "
        f"background: {fill}; color: {colour}; font-size: {DETAIL_PX}px; "
        "text-align: left; }"
        f"{selector}:hover {{ background: {hover}; }}"
    )


def row_chip(text: str, colour: str) -> QLabel:
    """A label chip in a row ("not present", "Loaded"): the attention chip's tint
    and height, so the chips in the panel's lists are one kind."""
    label = QLabel(text)
    label.setFixedHeight(CHIP_HEIGHT)
    label.setAlignment(Qt.AlignCenter)
    label.setStyleSheet(
        tinted_chip_style(colour, selector="QLabel").replace(
            "padding: 0 6px 0 4px", "padding: 0 7px"
        )
    )
    return label


def help_button(tooltip: str) -> QToolButton:
    """The ? on a list's header: how to work the list, as its tooltip, where a
    line of hints under the list said it on every view."""
    button = QToolButton()
    button.setFixedSize(24, 24)
    button.setAutoRaise(True)
    button.setIcon(fibsem_icon("mdi:help-circle-outline", color=GRAY_ICON_COLOR))
    button.setToolTip(tooltip)
    button.setStyleSheet("QToolButton { border: none; background: transparent; }")
    return button


def apply_name_style(label: QLabel) -> None:
    label.setStyleSheet(f"background: transparent; font-size: {NAME_PX}px;")
