"""The Plugins dialog.

Answers the question a user has after installing a plugin -- "did it work?" --
without making them hunt for it in the Add Task combo and guess when it is
missing. Deliberately not a place to run anything: that is the Scripts dialog.
This one only ever reports.

Everything shown comes from ``fibsem.plugins.report``, the same function behind
``fibsem-cli plugins``, so a report a user pastes into a bug tracker says the
same thing as the dialog support is reading over their shoulder.
"""

from typing import Dict, List, Optional, Tuple

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QFontMetrics
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fibsem.plugins.report import (
    Extension,
    ExtensionGroup,
    ExtensionSource,
    collect_extensions,
    environment_line,
    group_counts_phrase,
    render_report,
    summarise,
)
from fibsem.ui.icon import fibsem_icon
from fibsem.ui.stylesheets import (
    ACCENT_COLOR,
    BORDER_COLOR,
    DISABLED_BG_COLOR,
    ERROR_COLOR,
    GRAY_ICON_COLOR,
    OK_COLOR,
    PANEL_COLOR,
    ROW_ALT_COLOR,
    SURFACE_COLOR,
    TEXT_COLOR,
    TEXT_MUTED_COLOR,
    TEXT_STRONG_COLOR,
    WARN_COLOR,
)
from fibsem.ui.utils import open_path_in_file_explorer
from fibsem.ui.widgets.custom_widgets import (
    ElidedLabel,
    IconToolButton,
    chip,
    style_with_tooltip,
)

# Short local names for the shared palette. These appear inside dozens of
# f-strings below, where the full names would wrap every one of them.
_BG = SURFACE_COLOR
_PANEL = PANEL_COLOR
_ROW_ALT = ROW_ALT_COLOR
_BORDER = BORDER_COLOR
_TEXT = TEXT_COLOR
_TEXT_STRONG = TEXT_STRONG_COLOR
_TEXT_MUTED = TEXT_MUTED_COLOR
_ACCENT = ACCENT_COLOR
_OK = OK_COLOR
_WARN = WARN_COLOR
_ERROR = ERROR_COLOR

_COLUMNS = ["Extension", "Source", "Entry point group"]

_MONOSPACE = "font-family: Menlo, Consolas, monospace;"

# Colour carries state, never provenance. A plugin and a script are both
# installed and both in use -- they differ only in where they came from, which
# the chip's own label already says, so separate colours would imply a
# distinction that does not exist.
#
#   blue   in use
#   amber  installed, but not the copy being used -- look at this
#   grey   unremarkable: a built-in, or something in use that nobody came for
#   red    broken
#
# Amber rather than grey for an unavailable plugin, even though nothing is
# technically wrong: a user installed it expecting it to work, and grey is what
# built-ins wear. Painting one of the two rows this dialog exists to surface in
# the colour of background noise buries it.
_SOURCE_STYLE: Dict[ExtensionSource, Tuple[str, str]] = {
    ExtensionSource.PLUGIN: ("Plugin", _ACCENT),
    ExtensionSource.SCRIPT: ("Script", _ACCENT),
    ExtensionSource.FAILED: ("Failed", _ERROR),
    ExtensionSource.REGISTERED: ("Registered", _TEXT_MUTED),
    ExtensionSource.BUILTIN: ("Built-in", _TEXT_MUTED),
}

# What a shadowed row is, in terms of the consequence rather than the mechanism:
# "shadowed" describes fibsem's merge order, which is not the user's problem.
_UNAVAILABLE_LABEL = "Unavailable"

_TABLE_STYLE = f"""
QTableWidget {{
    background-color: {_BG};
    alternate-background-color: {_ROW_ALT};
    border: 1px solid {_BORDER};
    border-radius: 6px;
    color: {_TEXT};
    gridline-color: transparent;
    outline: none;
}}
QHeaderView::section {{
    background-color: {_PANEL};
    color: {_TEXT_MUTED};
    border: none;
    padding: 6px 8px;
    font-size: 11px;
}}
QTableWidget::item {{ padding: 6px 8px; border-bottom: 1px solid {DISABLED_BG_COLOR}; }}
QTableWidget::item:selected {{ background-color: #2d3947; color: {_TEXT_STRONG}; }}
"""

_CHECKBOX_STYLE = f"font-size: 12px; color: {_TEXT};"

_COPY_ICON = "mdi:content-copy"


class PluginsDialog(QDialog):
    """Every registered pattern, strategy and task, and where each came from."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        groups: Optional[Tuple[ExtensionGroup, ...]] = None,
    ) -> None:
        super().__init__(parent)

        # Snapshotted once. The registries load their entry points on first use
        # and cache the result for the process, so this dialog can only ever
        # show the state as of startup -- a Refresh button would either lie or,
        # if it really re-loaded, leave the registries disagreeing with the
        # module-level snapshots taken at import time.
        #
        # `groups` is for tests, which need rows this environment does not have
        # installed (a failure, a shadowed plugin, a script).
        self.groups: Tuple[ExtensionGroup, ...] = (
            collect_extensions() if groups is None else tuple(groups)
        )
        self.rows: List[Extension] = []

        self.setWindowTitle("Plugins")
        self.setStyleSheet(f"QDialog {{ background-color: {_BG}; }}")
        self.resize(880, 620)
        self.setMinimumWidth(760)
        self._build()
        self.refresh()

    # --- construction ---

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        header = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(2)
        self.title_label = QLabel("Registered extensions")
        style_with_tooltip(
            self.title_label,
            f"font-size: 15px; font-weight: 500; color: {_TEXT_STRONG};",
        )
        # counts are meta, not the heading -- the heading says what this dialog
        # is, the line under it says what is currently in it.
        self.meta_label = QLabel()
        style_with_tooltip(self.meta_label, f"font-size: 12px; color: {_TEXT_MUTED};")
        titles.addWidget(self.title_label)
        titles.addWidget(self.meta_label)
        header.addLayout(titles, 1)

        self.builtins_checkbox = QCheckBox("Show built-ins")
        style_with_tooltip(self.builtins_checkbox, _CHECKBOX_STYLE)
        self.builtins_checkbox.setToolTip(
            "Built-in extensions ship with fibsem and are always present; they "
            "are counted but hidden so installed ones stand out."
        )
        self.builtins_checkbox.toggled.connect(self.refresh)
        header.addWidget(self.builtins_checkbox)

        # Icons, not labelled buttons: neither is the reason anyone opens this
        # dialog -- the table is -- and a row of word-buttons competes with the
        # heading for a glance that should land on the counts.
        self.open_folder_button = IconToolButton(
            icon="mdi:folder-open-outline",
            tooltip="Reveal the selected script in the file manager",
            size=28,
        )
        self.open_folder_button.clicked.connect(self.open_selected_folder)
        header.addWidget(self.open_folder_button)

        self.copy_button = IconToolButton(
            icon=_COPY_ICON,
            tooltip="Copy this listing as text, exactly as `fibsem-cli plugins` prints it",
            size=28,
        )
        self.copy_button.clicked.connect(self.copy_report)
        header.addWidget(self.copy_button)
        layout.addLayout(header)

        # No detail strip and no footer. Every row now explains itself in place
        # -- including why an inactive one is inactive -- so a panel repeating
        # the selection was spending a whole widget on it. There is nothing to
        # confirm or cancel either, which leaves Esc and the title bar to
        # dismiss it, the same as any other read-only panel.
        self.table = self._build_table()
        layout.addWidget(self.table)

    def _build_table(self) -> QTableWidget:
        table = QTableWidget()
        table.setStyleSheet(_TABLE_STYLE)
        table.setColumnCount(len(_COLUMNS))
        table.setHorizontalHeaderLabels(_COLUMNS)
        table.verticalHeader().setVisible(False)
        # rows carry a stacked name + class path, so they need room to breathe
        table.verticalHeader().setDefaultSectionSize(56)
        table.setShowGrid(False)
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.itemSelectionChanged.connect(self._on_selection_changed)

        header = table.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, header.Stretch)
        table.setColumnWidth(2, _group_column_width())
        return table

    def _name_cell(self, extension: Extension) -> QWidget:
        """Two lines: what it registered as, and what it resolved to."""
        widget = QWidget()
        style_with_tooltip(widget, "background: transparent;")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 8, 10, 8)
        layout.setSpacing(2)

        # A failed entry point registered nothing, so it has no name -- the
        # declared target takes the line instead. A listing that only shows what
        # resolved cannot answer "why isn't mine here?".
        inactive = extension.is_problem
        name = ElidedLabel(extension.name or extension.target)
        style_with_tooltip(
            name,
            f"{_MONOSPACE} font-size: 13px; background: transparent;"
            f"color: {_TEXT_MUTED if inactive else _TEXT_STRONG};",
        )
        detail = ElidedLabel(_detail_line(extension))
        style_with_tooltip(
            detail,
            f"font-size: 11px; background: transparent;"
            f"color: {_reason_colour(extension)};",
        )
        layout.addWidget(name)
        layout.addWidget(detail)
        widget.setToolTip(_tooltip(extension))
        # `ElidedLabel` gives itself one of its own full text, which is right where a
        # label is on its own and wrong here: a child's tooltip shadows its parent's, so
        # hovering the row would lose everything `_tooltip` says about the extension.
        name.setToolTip("")
        detail.setToolTip("")
        return widget

    def _source_cell(self, extension: Extension) -> QWidget:
        """The source chip, plus a shadowed chip where the name was taken."""
        widget = QWidget()
        style_with_tooltip(widget, "background: transparent;")
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        label, colour = _SOURCE_STYLE[extension.source]
        if extension.shadowed:
            # Where it came from is still worth saying -- the fix for a shadowed
            # plugin is not the fix for a shadowed script -- but it is not in
            # use, so it does not get the in-use colour. The state chip beside
            # it carries the emphasis instead.
            colour = _TEXT_MUTED
        layout.addWidget(chip(label, colour))
        if extension.shadowed:
            layout.addWidget(chip(_UNAVAILABLE_LABEL, _WARN))
        layout.addStretch()
        widget.setToolTip(extension.problem or f"Source: {label}")
        return widget

    # --- data ---

    def refresh(self) -> None:
        """Rebuild the table for the current Show built-ins state."""
        show_builtins = self.builtins_checkbox.isChecked()
        previous = self.selected_extension()

        self.rows = [e for g in self.groups for e in g.visible(show_builtins)]
        self._update_meta(show_builtins)

        # Only a script-loaded extension has a file to reveal, and nothing
        # produces those until FIB-339 lands -- so in every environment today
        # this button would sit permanently greyed out, which reads as broken
        # rather than as inapplicable. Hidden until there is something to open.
        self.open_folder_button.setVisible(any(e.path is not None for e in self.rows))

        self.table.clearContents()
        self.table.setRowCount(0)  # destroys the previous rows' cell widgets
        self.table.setRowCount(len(self.rows))
        for row, extension in enumerate(self.rows):
            # cell widgets, not items: a QTableWidgetItem cannot carry a pill
            # background, and the name cell needs two lines.
            self.table.setCellWidget(row, 0, self._name_cell(extension))
            self.table.setCellWidget(row, 1, self._source_cell(extension))

            group_item = QTableWidgetItem(extension.group)
            group_item.setForeground(QColor(_TEXT_MUTED))
            group_item.setFont(_group_font())
            # The literal group string, not the friendly label: a group mistyped
            # in a plugin's pyproject registers nothing and logs nothing, so
            # comparing the real string against their own file is the only way a
            # user can find that mistake.
            self.table.setItem(row, 2, group_item)
            # an empty item under each widget keeps row selection working
            for col in (0, 1):
                self.table.setItem(row, col, QTableWidgetItem())

        self._fit_source_column()

        if self.rows:
            keys = [_key(e) for e in self.rows]
            row = keys.index(_key(previous)) if previous and _key(previous) in keys else 0
            self.table.selectRow(row)
        self._on_selection_changed()

        # Toggling built-ins changes what a copy would produce, so the button
        # has to come back out of its confirmation state.
        self._reset_copy_button()

    def _fit_source_column(self) -> None:
        """Widen the source column to the widest row of chips.

        ResizeToContents is no use here: it measures the item's size hint, and
        these cells hold widgets over empty items.
        """
        widths = [
            self.table.cellWidget(row, 1).sizeHint().width()
            for row in range(self.table.rowCount())
            if self.table.cellWidget(row, 1) is not None
        ]
        self.table.setColumnWidth(1, max(widths, default=120) + 12)

    def _update_meta(self, show_builtins: bool) -> None:
        summary, hidden = summarise(self.groups)
        if hidden and not show_builtins:
            summary += f" · {hidden} built-in hidden"
        self.meta_label.setText(summary)
        self.meta_label.setToolTip(
            "\n".join(
                [f"{g.group}: {group_counts_phrase(g)}" for g in self.groups]
                # Which install this is, for a machine with several environments.
                # Went with the footer it used to hang off; it is the first thing
                # anyone asks about a pasted report, so it stays reachable.
                + ["", environment_line()]
            )
        )

    # --- selection ---

    def selected_extension(self) -> Optional[Extension]:
        row = self.table.currentRow() if hasattr(self, "table") else -1
        if 0 <= row < len(self.rows):
            return self.rows[row]
        return None

    def _on_selection_changed(self) -> None:
        extension = self.selected_extension()
        self.open_folder_button.setEnabled(
            extension is not None and extension.path is not None
        )

    # --- actions ---

    def open_selected_folder(self) -> None:
        import os

        extension = self.selected_extension()
        if extension is None or extension.path is None:
            return
        # The stored path is home-relative for the report; expand it back before
        # handing it to the file manager.
        open_path_in_file_explorer(os.path.dirname(os.path.expanduser(extension.path)))

    def copy_report(self) -> None:
        clipboard = QApplication.clipboard()
        if clipboard is None:  # pragma: no cover - no clipboard on some displays
            return
        clipboard.setText(self.report_text())
        # An icon button gives no other sign it did anything, and a clipboard is
        # invisible until pasted somewhere else.
        self.copy_button.setIcon(fibsem_icon("mdi:check", color=_OK))
        self.copy_button.setToolTip("Copied")
        # Confirmation, not a one-shot: the report is often copied twice, once
        # per Show built-ins state.
        QTimer.singleShot(1500, self._reset_copy_button)

    def _reset_copy_button(self) -> None:
        self.copy_button.setIcon(fibsem_icon(_COPY_ICON, color=GRAY_ICON_COLOR))
        self.copy_button.setToolTip(
            "Copy this listing as text, exactly as `fibsem-cli plugins` prints it"
        )

    def report_text(self) -> str:
        """The listing as text -- byte for byte what ``fibsem-cli plugins`` prints."""
        return render_report(
            self.groups, show_builtins=self.builtins_checkbox.isChecked()
        )


def _key(extension: Extension) -> Tuple[str, str, str]:
    """Identity for restoring the selection across a rebuild."""
    return (extension.group, extension.name or "", extension.target)


def _detail_line(extension: Extension) -> str:
    """The line under the name: what it resolved to, or why it is not in use.

    A row with a problem spends this line on the reason. For anything inactive
    -- a failure, or a plugin whose name a built-in took -- that is the only
    thing the reader wants, and a class path they cannot use is worth less than
    the sentence explaining why. The path, the package and the group all stay in
    the tooltip.
    """
    if extension.problem:
        return _sentence(extension.problem)
    parts = [extension.target]
    if extension.origin:
        parts.append(extension.origin)
    return "   ·   ".join(parts)


def _reason_colour(extension: Extension) -> str:
    """The reason wears the same colour as the chip that raised it.

    Red for broken, amber for installed-but-unused, muted for a row with nothing
    wrong (where this line is a class path, not a reason).
    """
    if extension.source is ExtensionSource.FAILED:
        return _ERROR
    if extension.problem:
        return _WARN
    return _TEXT_MUTED


def _group_column_width() -> int:
    """Wide enough for the longest group string, and no wider.

    There are exactly three of them and they never change at runtime, so this is
    a measurement rather than a guess -- a fixed width elides `fibsem.strategies`
    on some platforms and wastes a column on others.
    """
    metrics = QFontMetrics(_group_font())
    longest = max(
        metrics.horizontalAdvance(g)
        for g in ("fibsem.patterns", "fibsem.strategies", "fibsem.tasks")
    )
    return longest + 24


def _group_font():
    from PyQt5.QtGui import QFont

    font = QFont("Menlo")
    font.setStyleHint(QFont.Monospace)
    font.setPointSize(11)
    return font


def _tooltip(extension: Extension) -> str:
    lines = [extension.name or extension.target, extension.target]
    if extension.origin:
        lines.append(f"Declared by {extension.origin}")
    lines.append(extension.group)
    if extension.problem:
        lines.append(_sentence(extension.problem))
    return "\n".join(dict.fromkeys(lines))


def _sentence(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text
