"""The Add Task dialog: a table of every task type, and one name to type.

Replaces a two-field form whose central problem was that both fields said
almost the same words -- the type combo read ``Trench Milling (MILL_TRENCH)``
and the name box beneath it was prefilled ``Trench Milling``, with nothing to
say that the first is a fixed class from the registry and the second is an
arbitrary label that becomes the protocol's dictionary key.

Choosing the type is now selecting a row, so nothing on screen can be mistaken
for the name. Exactly one thing is typed, it sits alone in the footer, and it
arrives already free -- the collision the old dialog was guaranteed to hit on
the second copy of any task type never happens.

A table rather than a list with a detail pane: with a description, a purpose and
a source for every task there is enough to compare, and comparing is the whole
job. A detail pane shows one row at a time and leaves most of the dialog empty.
"""

from typing import Any, Dict, List, Optional, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.workflows.tasks.catalogue import (
    has_advanced_tasks,
    task_info,
    task_source,
    tasks_by_tag,
    unique_task_name,
)
from fibsem.plugins.report import Extension, ExtensionSource
from fibsem.ui.stylesheets import (
    ACCENT_COLOR,
    BORDER_COLOR,
    ERROR_COLOR,
    PANEL_COLOR,
    PRIMARY_BUTTON_STYLESHEET,
    ROW_ALT_COLOR,
    SECONDARY_BUTTON_STYLESHEET,
    SURFACE_COLOR,
    TEXT_COLOR,
    TEXT_MUTED_COLOR,
    TEXT_STRONG_COLOR,
    WARN_COLOR,
)
from fibsem.ui.widgets.custom_widgets import chip, style_with_tooltip

# "Also used for", not "Tags": the heading a row sits under already states its
# primary purpose, so this column carries only the others. Naming it for what
# it holds means an empty cell reads as "no second use" rather than as missing
# data -- which is what most rows are, and is worth knowing.
_COLUMNS = ("Task", "What it does", "Also used for")
_COL_TASK, _COL_WHAT, _COL_TAGS = range(3)

# Small. At the sizes a chip is drawn elsewhere it is the loudest thing in the
# row, and there are two or three of them per row here.
_CHIP_SIZE = 10

_ALL_PURPOSES = "All purposes"

# Same vocabulary and colours the Plugins dialog uses. Only the sources that can
# reach a picker are listed: a failed or shadowed extension registered nothing,
# so it never appears in the registry this dialog reads.
_SOURCE_LABEL = {
    ExtensionSource.PLUGIN: "Plugin",
    ExtensionSource.SCRIPT: "Script",
    ExtensionSource.REGISTERED: "Registered",
    ExtensionSource.BUILTIN: "Built-in",
}

# Description cell: margins here, and the font set programmatically below. A
# stylesheet font-size never reaches QWidget.font(), so metrics taken from a
# stylesheet-styled label measure the wrong face.
_DESC_MARGINS = (12, 7, 14, 7)

# The padding QTableWidget::item carries, as a constant rather than a literal
# inside the stylesheet below. A row is taller than the widget it holds by
# twice the padding plus the separator border, and a row sized without
# allowing for all of it clips the last line off the description -- the
# border alone is a one-pixel shortfall, which is enough. Interpolated into the QSS *and* used in the
# arithmetic, so the two cannot drift -- the earlier version measured it at
# runtime, which read zero before the first layout and compounded after it.
_ITEM_PADDING_V = 7
_ITEM_PADDING_H = 10
_ITEM_BORDER = 1
_ROW_CHROME = 2 * _ITEM_PADDING_V + _ITEM_BORDER

_TABLE_STYLE = f"""
QTableWidget {{
    background-color: {SURFACE_COLOR};
    alternate-background-color: {ROW_ALT_COLOR};
    border: 1px solid {BORDER_COLOR};
    border-radius: 6px;
    color: {TEXT_COLOR};
    gridline-color: transparent;
    outline: none;
}}
QHeaderView::section {{
    background-color: {PANEL_COLOR};
    color: {TEXT_MUTED_COLOR};
    border: none;
    padding: 6px 10px;
    font-size: 11px;
}}
QTableWidget::item {{
    padding: {_ITEM_PADDING_V}px {_ITEM_PADDING_H}px;
    border-bottom: {_ITEM_BORDER}px solid #31353f;
}}
QTableWidget::item:selected {{
    background-color: #2d3947;
    color: {TEXT_STRONG_COLOR};
}}
"""

_FIELD_STYLE = f"""
QLineEdit, QComboBox {{
    background-color: #16181d;
    border: 1px solid {BORDER_COLOR};
    border-radius: 3px;
    padding: 5px 8px;
    color: {TEXT_STRONG_COLOR};
}}
QLineEdit:focus {{ border: 1px solid {ACCENT_COLOR}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
"""


def _description_font() -> QFont:
    font = QFont()
    font.setPixelSize(12)
    return font


def _transparent(widget: QWidget) -> QWidget:
    style_with_tooltip(widget, "background: transparent;")
    return widget


class AddTaskDialog(QDialog):
    """Pick a task type from the registry and name the copy being added."""

    def __init__(
        self,
        existing_task_config: Dict[str, Any],
        lamella_count: int = 0,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.existing_task_config = existing_task_config
        self.lamella_count = lamella_count
        self.task_types: Dict[str, Any] = {}
        self._rows: Dict[int, str] = {}  # table row -> task_type
        self._sizing = False  # re-entrancy guard for _size_rows
        self._named_for: Optional[str] = None  # the selection the name belongs to

        self.setWindowTitle("Add task")
        self.setModal(True)
        self.setStyleSheet(f"QDialog {{ background-color: {SURFACE_COLOR}; }}")
        self.setMinimumSize(820, 460)
        self.resize(980, 580)

        self._build()
        self._populate()

    # --- construction ---

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(10)

        title = QLabel("Add a task to this protocol")
        style_with_tooltip(
            title, f"font-size: 15px; font-weight: 600; color: {TEXT_STRONG_COLOR};"
        )
        outer.addWidget(title)
        outer.addLayout(self._build_toolbar())
        outer.addWidget(self._build_table(), 1)
        outer.addWidget(self._build_footer())

    def _build_toolbar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setSpacing(9)

        self.search_field = QLineEdit()
        self.search_field.setPlaceholderText("Search tasks")
        self.search_field.setStyleSheet(_FIELD_STYLE)
        self.search_field.setClearButtonEnabled(True)
        self.search_field.setFixedWidth(240)
        self.search_field.textChanged.connect(self._refresh_visibility)
        bar.addWidget(self.search_field)

        self.tag_filter = QComboBox()
        self.tag_filter.setStyleSheet(_FIELD_STYLE)
        self.tag_filter.setFixedWidth(180)
        self.tag_filter.setToolTip(
            "Show only tasks used for one purpose. Matches every tag a task "
            "declares, not only the heading it sits under."
        )
        self.tag_filter.currentIndexChanged.connect(self._refresh_visibility)
        bar.addWidget(self.tag_filter)

        bar.addStretch(1)

        self.advanced_checkbox = QCheckBox("Show advanced")
        style_with_tooltip(
            self.advanced_checkbox, f"font-size: 12px; color: {TEXT_MUTED_COLOR};"
        )
        self.advanced_checkbox.setToolTip(
            "Primitives and escape hatches: tasks that run whatever they are "
            "handed, rather than performing one step of a protocol."
        )
        self.advanced_checkbox.setVisible(has_advanced_tasks())
        self.advanced_checkbox.toggled.connect(self._on_advanced_toggled)
        bar.addWidget(self.advanced_checkbox)
        return bar

    def _build_table(self) -> QTableWidget:
        table = QTableWidget()
        table.setStyleSheet(_TABLE_STYLE)
        table.setColumnCount(len(_COLUMNS))
        table.setHorizontalHeaderLabels(list(_COLUMNS))
        table.verticalHeader().setVisible(False)
        table.setShowGrid(False)
        table.setAlternatingRowColors(False)  # group headings do the banding
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setWordWrap(True)
        table.itemSelectionChanged.connect(self._validate)

        header = table.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        header.setSectionResizeMode(_COL_TASK, QHeaderView.Fixed)
        header.setSectionResizeMode(_COL_WHAT, QHeaderView.Stretch)
        header.setSectionResizeMode(_COL_TAGS, QHeaderView.Fixed)
        table.setColumnWidth(_COL_TASK, 218)
        table.setColumnWidth(_COL_TAGS, 152)
        self.table = table
        return table

    def _build_footer(self) -> QWidget:
        """Form content, a rule, then an action bar.

        The convention this restores: the bottom strip holds actions and nothing
        else. Sharing that row with a text input makes the input read as part of
        the button group, and it puts the primary action next to something you
        are still typing in. Validation belongs with the field it is about, not
        underneath the buttons where it is both easy to miss and easy to mistake
        for a caption on them.
        """
        footer = QWidget()
        layout = QVBoxLayout(footer)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(6)

        label = QLabel("Name this task")
        style_with_tooltip(
            label, f"font-size: 12px; font-weight: 600; color: {TEXT_STRONG_COLOR};"
        )
        layout.addWidget(label)

        self.name_field = QLineEdit()
        self.name_field.setStyleSheet(_FIELD_STYLE)
        # A task name is a handful of words. Stretching the field to the width
        # of a four-column table makes the footer read as a separate, emptier
        # dialog stuck to the bottom of this one. Fixed rather than capped:
        # beside a stretch, a capped field collapses to its size hint and
        # scrolls the very text it is meant to show.
        self.name_field.setFixedWidth(360)
        # The sentence the old dialog was missing: "Task Name:" never said the
        # name is the protocol's key, which is why it read as a duplicate of the
        # type rather than as a decision.
        self.name_field.setToolTip(
            "The label this task is known by in the workflow, and its key in "
            "protocol.yaml. Must be unique within the protocol."
        )
        self.name_field.textChanged.connect(self._validate)
        self.name_field.returnPressed.connect(self._on_return_pressed)
        name_row = QHBoxLayout()
        name_row.setContentsMargins(0, 0, 0, 0)
        name_row.addWidget(self.name_field)
        name_row.addStretch(1)
        layout.addLayout(name_row)

        self.error_label = QLabel()
        style_with_tooltip(self.error_label, f"font-size: 11px; color: {ERROR_COLOR};")
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(False)
        layout.addWidget(self.error_label)

        rule = QFrame()
        rule.setFrameShape(QFrame.HLine)
        rule.setStyleSheet(f"color: {BORDER_COLOR};")
        layout.addWidget(rule)

        actions = QHBoxLayout()
        actions.setSpacing(9)
        # What Add will do, beside the button that will do it.
        self.status_label = QLabel()
        style_with_tooltip(
            self.status_label, f"font-size: 11px; color: {TEXT_MUTED_COLOR};"
        )
        self.status_label.setWordWrap(True)
        actions.addWidget(self.status_label, 1)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        self.cancel_button.clicked.connect(self.reject)
        self.add_button = QPushButton("Add task")
        self.add_button.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        self.add_button.setDefault(True)
        self.add_button.clicked.connect(self.accept)
        actions.addWidget(self.cancel_button)
        actions.addWidget(self.add_button)
        layout.addLayout(actions)
        return footer

    # --- cells ---

    def _existing_count(self, task_type: str) -> int:
        """How many tasks of this type the protocol already holds.

        Duplicates are legitimate -- a protocol can want three reference-image
        tasks -- so this is context, not a warning. It is also what explains the
        name field: without it, a default of "Trench Milling 2" appears out of
        nowhere.
        """
        return sum(
            1
            for config in self.existing_task_config.values()
            if getattr(config, "task_type", None) == task_type
        )

    def _task_cell(self, task_type: str, display_name: str, info) -> QWidget:
        widget = _transparent(QWidget())
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 6, 10, 6)
        layout.setSpacing(1)

        name = QLabel(display_name)
        style_with_tooltip(
            name,
            f"font-size: 13px; font-weight: 600; color: {TEXT_STRONG_COLOR};"
            " background: transparent;",
        )
        name.setWordWrap(True)
        layout.addWidget(name)

        count = self._existing_count(task_type)
        if count:
            already = QLabel(f"{count} in this protocol")
            style_with_tooltip(
                already,
                f"font-size: 11px; color: {TEXT_MUTED_COLOR}; background: transparent;",
            )
            layout.addWidget(already)

        # No identifier line. In capitals it was the loudest thing in the row
        # and it is not what anyone is reading for -- it lives in the tooltip,
        # where someone holding a traceback can still find it.
        if info.experimental:
            marks = QHBoxLayout()
            marks.setSpacing(4)
            marks.addWidget(chip("Experimental", WARN_COLOR, font_size=_CHIP_SIZE))
            marks.addStretch(1)
            layout.addLayout(marks)
        return widget

    def _description_cell(self, text: str) -> QWidget:
        widget = _transparent(QWidget())
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(*_DESC_MARGINS)
        label = QLabel(text)
        label.setWordWrap(True)
        label.setFont(_description_font())
        label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        style_with_tooltip(label, f"color: {TEXT_COLOR}; background: transparent;")
        layout.addWidget(label)
        widget.setProperty("_label", label)
        return widget

    def _chips_cell(self, tags: Tuple[str, ...]) -> QWidget:
        widget = _transparent(QWidget())
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(10, 7, 10, 7)
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignTop)
        for tag in tags:
            layout.addWidget(chip(tag, TEXT_COLOR, font_size=_CHIP_SIZE))
        layout.addStretch(1)
        return widget

    def _tooltip(self, task_type: str, task_cls, info) -> str:
        """Everything the table does not have room for, on hover."""
        source = task_source(task_type)
        lines = [
            f"<b>{task_cls.config_cls.display_name}</b>",
            f"<code style='color:{TEXT_MUTED_COLOR}'>{task_type}</code>",
        ]
        if info.description:
            lines.append(f"<p>{info.description}</p>")
        rows = []
        if info.tags:
            rows.append(("Used for", ", ".join(info.tags)))
        related = [
            c.display_name for c in getattr(task_cls.config_cls, "related_tasks", [])
        ]
        if related:
            rows.append(("Used with", ", ".join(related)))
        if source is not None:
            label = _SOURCE_LABEL.get(source.source, "Built-in")
            rows.append(
                ("Source", f"{label} · {source.origin}" if source.origin else label)
            )
        rows.append(("Level", info.level))
        if info.experimental:
            rows.append(
                ("Note", "Experimental — behaviour and parameters may still change.")
            )
        lines.append(
            "<table cellspacing='0' cellpadding='0'>"
            + "".join(
                f"<tr><td style='color:{TEXT_MUTED_COLOR}'>{k}&nbsp;&nbsp;</td>"
                f"<td>{v}</td></tr>"
                for k, v in rows
            )
            + "</table>"
        )
        return "".join(lines)

    # --- population ---

    def _populate(self, keep: Optional[str] = None) -> None:
        include_advanced = self.advanced_checkbox.isChecked()
        grouped = tasks_by_tag(include_advanced)
        self.task_types = {
            task_type: cls
            for tasks in grouped.values()
            for task_type, cls in tasks.items()
        }
        self._rows = {}

        self.table.blockSignals(True)
        self.table.clearContents()
        self.table.setRowCount(0)
        row = 0
        for tag, tasks in grouped.items():
            self.table.insertRow(row)
            heading = QTableWidgetItem(tag.upper())
            font = heading.font()
            font.setBold(True)
            font.setPointSize(max(8, font.pointSize() - 2))
            heading.setFont(font)
            heading.setForeground(QColor(TEXT_MUTED_COLOR))
            heading.setFlags(Qt.NoItemFlags)
            self.table.setItem(row, 0, heading)
            self.table.setSpan(row, 0, 1, len(_COLUMNS))
            self.table.setRowHeight(row, 30)
            row += 1

            for task_type, task_cls in tasks.items():
                info = task_info(task_cls)
                self.table.insertRow(row)
                self.table.setCellWidget(
                    row,
                    _COL_TASK,
                    self._task_cell(task_type, task_cls.config_cls.display_name, info),
                )
                self.table.setCellWidget(
                    row, _COL_WHAT, self._description_cell(info.summary)
                )
                self.table.setCellWidget(
                    row, _COL_TAGS, self._chips_cell(info.tags[1:])
                )
                tooltip = self._tooltip(task_type, task_cls, info)
                for column in range(len(_COLUMNS)):
                    item = QTableWidgetItem()  # empty items keep row selection working
                    item.setToolTip(tooltip)
                    self.table.setItem(row, column, item)
                    widget = self.table.cellWidget(row, column)
                    if widget is not None:
                        widget.setToolTip(tooltip)
                self._rows[row] = task_type
                row += 1

        self._rebuild_tag_filter(grouped)
        self._fit_tag_column()
        self.table.blockSignals(False)
        self._refresh_visibility()
        if keep is not None and keep in self.task_types:
            self.select(keep)
        elif self.selected_task_type is None:
            self._select_first_visible()

    def _fit_tag_column(self) -> None:
        """Widen the tag column to the widest row of chips.

        ResizeToContents is no use here: it measures the item's size hint, and
        these cells hold widgets over empty items. Too narrow and the chips do
        not elide, they overlap.
        """
        widths = [
            self.table.cellWidget(row, _COL_TAGS).sizeHint().width()
            for row in self._rows
            if self.table.cellWidget(row, _COL_TAGS) is not None
        ]
        self.table.setColumnWidth(_COL_TAGS, max(widths, default=120) + 8)

    def _rebuild_tag_filter(self, grouped) -> None:
        """Every tag any offered task declares, not only the headings."""
        tags: List[str] = []
        for tasks in grouped.values():
            for task_cls in tasks.values():
                for tag in task_info(task_cls).tags:
                    if tag not in tags:
                        tags.append(tag)
        previous = self.tag_filter.currentText()
        self.tag_filter.blockSignals(True)
        self.tag_filter.clear()
        self.tag_filter.addItem(_ALL_PURPOSES)
        self.tag_filter.addItems(sorted(tags))
        index = self.tag_filter.findText(previous)
        self.tag_filter.setCurrentIndex(max(0, index))
        self.tag_filter.blockSignals(False)

    # --- filtering ---

    def _refresh_visibility(self) -> None:
        query = self.search_field.text().strip().lower()
        wanted_tag = self.tag_filter.currentText()
        if wanted_tag == _ALL_PURPOSES:
            wanted_tag = ""

        visible_rows = set()
        for row, task_type in self._rows.items():
            info = task_info(self.task_types[task_type])
            display_name = self.task_types[task_type].config_cls.display_name
            # Match the identifier and every tag as well as the visible text:
            # someone reading a protocol.yaml or a traceback has the TASK_TYPE.
            haystack = " ".join(
                [display_name, task_type, info.summary, *info.tags]
            ).lower()
            hidden = bool(query) and query not in haystack
            if wanted_tag and wanted_tag not in info.tags:
                hidden = True
            self.table.setRowHidden(row, hidden)
            if not hidden:
                visible_rows.add(row)

        # A heading with nothing under it is noise, and a run of them reads as
        # if the filter is broken.
        for row in range(self.table.rowCount()):
            if row in self._rows:
                continue
            following = [r for r in self._rows if r > row]
            next_heading = min(
                (
                    r
                    for r in range(row + 1, self.table.rowCount())
                    if r not in self._rows
                ),
                default=self.table.rowCount(),
            )
            members = [r for r in following if r < next_heading]
            self.table.setRowHidden(row, not any(r in visible_rows for r in members))

        current = self.table.currentRow()
        if current < 0 or current not in visible_rows:
            self._select_first_visible()
        self._size_rows()
        self._validate()

    def _select_first_visible(self) -> None:
        for row in sorted(self._rows):
            if not self.table.isRowHidden(row):
                self.table.selectRow(row)
                return
        self.table.clearSelection()
        self.table.setCurrentCell(-1, -1)

    # --- row sizing ---

    def _size_rows(self) -> None:
        """Row heights from the labels themselves, at the width they actually got.

        Every arithmetic approach here is wrong by a line. `resizeRowsToContents`
        measures the item rather than the cell widget, so a row of widgets over
        empty items collapses; and deriving the wrap width from the column means
        guessing at the layout margins and the QSS item padding together.

        Two things make this correct. The label is asked for its own height at
        its own width -- trustworthy only because the description font is set
        programmatically, not through a stylesheet. And the difference between
        the row and the widget it holds is *measured*, because
        `QTableWidget::item` carries `padding: 8px 10px`, which is 16px of every
        row that never reaches the cell. Assuming that number instead of
        measuring it silently clips the last line off every description.
        """
        if self._sizing:
            return
        self._sizing = True
        try:
            for row, _ in self._rows.items():
                if self.table.isRowHidden(row):
                    continue
                cell = self.table.cellWidget(row, _COL_WHAT)
                label = cell.property("_label")
                _, top, _, bottom = _DESC_MARGINS
                chrome = _ROW_CHROME
                needed = label.heightForWidth(label.width()) + top + bottom + chrome
                others = [
                    self.table.cellWidget(row, c).sizeHint().height() + chrome
                    for c in range(len(_COLUMNS))
                    if c != _COL_WHAT and self.table.cellWidget(row, c) is not None
                ]
                self.table.setRowHeight(row, max([needed] + others))
        finally:
            self._sizing = False

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # Widths only settle once the table has been laid out, and the wrapped
        # height depends on them, so the first pass measures and the second uses
        # what it measured.
        for _ in range(2):
            self._size_rows()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # The description column stretches, so a resize re-wraps every row.
        self._size_rows()

    # --- selection ---

    @property
    def selected_task_type(self) -> Optional[str]:
        row = self.table.currentRow()
        if row < 0 or row not in self._rows or self.table.isRowHidden(row):
            return None
        return self._rows[row]

    @property
    def task_name(self) -> str:
        return self.name_field.text().strip()

    def select(self, task_type: str) -> None:
        """Select the row for ``task_type``, if it is on offer and visible."""
        for row, candidate in self._rows.items():
            if candidate == task_type and not self.table.isRowHidden(row):
                self.table.selectRow(row)
                return

    def _on_advanced_toggled(self, _checked: bool) -> None:
        self._populate(keep=self.selected_task_type)

    # --- validation ---

    def _on_return_pressed(self) -> None:
        if self.add_button.isEnabled():
            self.accept()

    def _suggested_name(self) -> str:
        task_type = self.selected_task_type
        if task_type is None:
            return ""
        display_name = self.task_types[task_type].config_cls.display_name
        return unique_task_name(display_name, self.existing_task_config)

    def _validate(self) -> bool:
        """Enable Add only when there is something valid to add, and say why not.

        The old dialog left OK enabled and returned silently from its accept
        handler, so a duplicate name produced a button that looked live, did
        nothing when clicked, and never said what to do instead.
        """
        task_type = self.selected_task_type
        if task_type != self._named_for:
            # The selection moved, so whatever is in the box belonged to the
            # previous one. Replace it with a name that is already free.
            self._named_for = task_type
            self.name_field.blockSignals(True)
            self.name_field.setText(self._suggested_name())
            self.name_field.blockSignals(False)

        name = self.task_name
        # The destination stays put whether or not the name is valid: it
        # describes the button, not the input, and watching it vanish while you
        # fix a typo is how a static line becomes wallpaper.
        self.status_label.setText(
            self._destination_summary() if task_type is not None else ""
        )

        if task_type is None:
            return self._reject("")
        if not name:
            return self._reject("")
        if name in self.existing_task_config:
            suggestion = unique_task_name(name, self.existing_task_config)
            return self._reject(
                f"“{name}” is already in this protocol. Try “{suggestion}”."
            )

        self._set_error("")
        self.add_button.setEnabled(True)
        return True

    def _reject(self, message: str) -> bool:
        self._set_error(message)
        self.add_button.setEnabled(False)
        return False

    def _destination_summary(self) -> str:
        """Where adding this actually writes.

        Three places, only one of which a user would guess: the protocol's task
        config, the workflow order, and a copy on every lamella that already
        exists. Saying so is the difference between an edit and a surprise.
        """
        base = "Adds to the protocol and the end of the workflow"
        if self.lamella_count == 1:
            return f"{base}, and to the 1 existing lamella."
        if self.lamella_count > 1:
            return f"{base}, and to all {self.lamella_count} existing lamellae."
        return f"{base}."

    def _set_error(self, text: str) -> None:
        """Show a problem on the field itself, or clear it."""
        self.error_label.setText(text)
        self.error_label.setVisible(bool(text))
        border = ERROR_COLOR if text else BORDER_COLOR
        self.name_field.setStyleSheet(
            _FIELD_STYLE + f"QLineEdit {{ border: 1px solid {border}; }}"
        )

    # --- result ---

    def accept(self) -> None:
        if not self._validate():
            return
        super().accept()

    def get_task_info(self) -> Tuple[Optional[str], str]:
        """The selected task type and the name to give it."""
        return self.selected_task_type, self.task_name
