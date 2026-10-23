"""Workflow tab · Grids: pick grids, pick and order tasks, run.

The run view for grids, beside the lamella one and sharing its chrome: the Run
and Stop buttons, the timeline on the right, the confirmation before a start.
Grid rows come from the experiment's records with the hardware's word on each
(slot, loaded) drawn as chips; only a present grid can be ticked. Task rows
come from the grid protocol in its order, and the order can be changed here.
Task settings live on the Protocol tab's Grid page; how a run ends (unload the
last grid, write the screening report) is behind the ⚙ on the task list.

"Screen all", on the grid list's header, is the Arctis user's one action:
inventory, every present grid, the ticked tasks.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.structures import Experiment
from fibsem.applications.autolamella.ui.grids.card import grid_headline
from fibsem.applications.autolamella.workflows.estimate import exchanges_for
from fibsem.structures import GridInventoryEntry, GridSlotState
from fibsem.theme import tokens
from fibsem.theme.tokens import (
    ACCENT_COLOR,
    BORDER_COLOR,
    NEUTRAL_200,
    NEUTRAL_550,
    NEUTRAL_700,
    OK_COLOR,
    TEXT_COLOR,
    TEXT_MUTED_COLOR,
)
from fibsem.ui import stylesheets
from fibsem.ui.icon import (
    DRAG_HANDLE_HEIGHT,
    DRAG_HANDLE_WIDTH,
    drag_handle_pixmap,
    fibsem_icon,
)
from fibsem.ui.widgets.custom_widgets import (
    ElidedLabel,
    IconToolButton,
    style_with_tooltip,
)
from fibsem.ui.widgets.lists import (
    CHIP_HEIGHT,
    DETAIL_PX,
    NAME_PX,
    ROW_HEIGHT,
    ListHeader,
    OptionsPopup,
    ReorderableListWidget,
    column_label,
    help_button,
    row_chip,
)
from fibsem.ui.widgets.preflight import TEXT_STRONG
from fibsem.ui.widgets.workflow.task_list import (
    ATTENTION_LABELS,
    BTN_SIZE,
    CHIP_ICONS,
    CHIP_WIDTH,
    chip_style,
    requires_phrase,
)
from fibsem.workflows.attention import Attention
from fibsem.workflows.grid import (
    FluorescenceOverviewGridTaskConfig,
    GridTaskConfig,
)
from fibsem.workflows.grid.protocol import GridTaskProtocol
from fibsem.workflows.grids import GridRecord

# The lamella list's metrics, so the two run views read alike.
_ROW_HEIGHT = ROW_HEIGHT
_NAME_MIN_WIDTH = 160
_SIDE_COLUMN_WIDTH = 170  # the muted right-hand column: the task's kind, or why not
# The slot column is narrower: "slot 02" is all it ever says, and at the task
# column's width a row carrying the Loaded pill overran the list and lost it.
_SLOT_COLUMN_WIDTH = 56
_SIDE_FONT_PX = DETAIL_PX
_HEADER_STYLE = "font-weight: bold; background: transparent;"


TASK_LIST_HINTS = (
    "The order here is the order they run on each grid: drag to reorder"
    "  \u2022  use \u270e to edit what a task requires"
    "  \u2022  settings are on the Protocol tab's Grid page"
)


def task_unavailable_reason(config: GridTaskConfig, microscope) -> Optional[str]:
    """Why this system cannot run the task, or None when it can (or no system
    is connected to say)."""
    if microscope is None:
        return None
    if (
        isinstance(config, FluorescenceOverviewGridTaskConfig)
        and getattr(microscope, "fm", None) is None
    ):
        return "This system has no fluorescence microscope"
    return None


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


class _GridRow(QWidget):
    selection_changed = pyqtSignal(object, bool)  # GridRecord, checked

    def __init__(self, grid: GridRecord, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.grid = grid
        self._entry: Optional[GridInventoryEntry] = None
        self._chip_widgets: List[QLabel] = []
        self.setAttribute(Qt.WA_TranslucentBackground)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 0, 6, 0)
        layout.setSpacing(8)
        self.checkbox = QCheckBox()
        self.checkbox.setStyleSheet("background: transparent;")
        self.checkbox.toggled.connect(
            lambda checked: self.selection_changed.emit(self.grid, checked)
        )
        layout.addWidget(self.checkbox)
        self.name_label = QLabel()
        self.name_label.setMinimumWidth(_NAME_MIN_WIDTH)
        self.name_label.setTextFormat(Qt.PlainText)
        layout.addWidget(self.name_label)
        self.status_label = ElidedLabel()
        layout.addWidget(self.status_label, 1)
        self._chips = QHBoxLayout()
        self._chips.setSpacing(4)
        layout.addLayout(self._chips)
        # The slot, in its own right-aligned column like the lamella list's
        # dependencies: read down the list, not hunted for after each name.
        self.slot_label = QLabel()
        self.slot_label.setFixedWidth(_SLOT_COLUMN_WIDTH)
        self.slot_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.slot_label.setStyleSheet(
            f"background: transparent; color: {NEUTRAL_700}; "
            f"font-size: {_SIDE_FONT_PX}px;"
        )
        layout.addWidget(self.slot_label)
        self.refresh()

    @property
    def is_present(self) -> bool:
        return self._entry is not None and self._entry.present

    @property
    def selectable(self) -> bool:
        return self.is_present

    @property
    def loaded(self) -> bool:
        return self._entry is not None and self._entry.loaded

    def set_inventory(self, entry: Optional[GridInventoryEntry]) -> None:
        self._entry = entry
        self.refresh()

    def refresh(self) -> None:
        self.name_label.setText(self.grid.name)
        present = self.is_present
        style_with_tooltip(
            self.name_label,
            f"background: transparent; font-size: {NAME_PX}px; "
            f"color: {NEUTRAL_200 if present else NEUTRAL_550};",
        )
        text, colour = grid_headline(self.grid)
        self.status_label.setText(text)
        self.status_label.setStyleSheet(
            f"font-size: {DETAIL_PX}px; color: {colour}; background: transparent;"
        )
        for old in self._chip_widgets:
            self._chips.removeWidget(old)
            old.deleteLater()
        self._chip_widgets = []
        entry = self._entry
        chips: List[Tuple[str, str]] = []
        if entry is not None and entry.state is GridSlotState.UNKNOWN:
            chips.append(("not scanned", TEXT_MUTED_COLOR))
        elif entry is None or not entry.present:
            chips.append(("not present", TEXT_MUTED_COLOR))
        elif entry.loaded:
            chips.append(("Loaded", OK_COLOR))
        for label, chip_colour in chips:
            widget = row_chip(label, chip_colour)
            self._chips.addWidget(widget)
            self._chip_widgets.append(widget)
        self.slot_label.setText(
            f"slot {entry.index + 1:02d}" if entry is not None and entry.present else ""
        )
        # Only a present grid can be run on; an absent one is unticked and stays so.
        if not present and self.checkbox.isChecked():
            self.checkbox.setChecked(False)
        self.checkbox.setEnabled(present)
        self.checkbox.setToolTip(
            "" if present else "Not in the magazine or holder; run an inventory"
        )


class GridTaskEditDialog(QDialog):
    """What a grid task requires, as the lamella workflow's Edit Task dialog
    sets it: one box per other grid task. Only that: a grid task has no
    schedule and is not optional, and it is removed on the Protocol tab, where
    its settings are."""

    def __init__(
        self,
        config: GridTaskConfig,
        requires: Sequence[str],
        task_names: Sequence[str],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("Edit Task")
        self.setModal(True)
        self.setMinimumWidth(420)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        title = QLabel(config.task_name)
        title.setTextFormat(Qt.PlainText)
        title.setStyleSheet(
            f"font-size: 13px; font-weight: bold; color: {TEXT_STRONG};"
        )
        layout.addWidget(title)
        heading = QLabel("Requires")
        heading.setStyleSheet(_HEADER_STYLE)
        layout.addWidget(heading)
        hint = QLabel(
            "Tasks whose result this one uses. It is skipped on a grid where "
            "one of them did not complete."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"font-size: 11px; color: {TEXT_MUTED_COLOR};")
        layout.addWidget(hint)
        self.checks: Dict[str, QCheckBox] = {}
        others = [n for n in task_names if n != config.task_name]
        for name in others:
            box = QCheckBox(name)
            box.setChecked(name in requires)
            layout.addWidget(box)
            self.checks[name] = box
        if not others:
            layout.addWidget(_empty_line("No other grid tasks to require."))
        layout.addStretch(1)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.btn_cancel.clicked.connect(self.reject)
        buttons.addWidget(self.btn_cancel)
        self.btn_apply = QPushButton("Apply")
        self.btn_apply.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.btn_apply.clicked.connect(self.accept)
        buttons.addWidget(self.btn_apply)
        layout.addLayout(buttons)

    @property
    def requires(self) -> List[str]:
        """The ticked tasks, in the order the boxes list them (the run order)."""
        return [name for name, box in self.checks.items() if box.isChecked()]


class _TaskRow(QWidget):
    selection_changed = pyqtSignal(str, bool)  # task name, checked
    attention_changed = pyqtSignal(str)  # task name; the protocol is already changed
    edit_clicked = pyqtSignal(str)  # task name

    def __init__(
        self,
        config: GridTaskConfig,
        protocol: GridTaskProtocol,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.config = config
        self.protocol = protocol
        self._reason: Optional[str] = None
        self.setAttribute(Qt.WA_TranslucentBackground)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 0, 6, 0)
        layout.setSpacing(8)
        self.checkbox = QCheckBox()
        self.checkbox.setStyleSheet("background: transparent;")
        self.checkbox.toggled.connect(
            lambda checked: self.selection_changed.emit(self.task_name, checked)
        )
        layout.addWidget(self.checkbox)
        self.name_label = QLabel()
        self.name_label.setMinimumWidth(_NAME_MIN_WIDTH)
        self.name_label.setTextFormat(Qt.PlainText)
        layout.addWidget(self.name_label)
        layout.addStretch(1)
        # What it waits for, or its kind, or why this system cannot run it: the
        # muted column left of the chip, where the lamella list says "after". A
        # reason can be longer than the column; elided from the right it keeps
        # its start, where a plain right-aligned label lost it off the left edge.
        self.detail_label = ElidedLabel()
        self.detail_label.setFixedWidth(_SIDE_COLUMN_WIDTH)
        # ElidedLabel is Ignored horizontally so a stretching column cannot be
        # driven wide by its text. This column is fixed, and left Ignored the
        # layout placed it as if it had no width and let the stretch push it
        # past the row's edge; Fixed makes it a proper 170 px column.
        self.detail_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.detail_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self.detail_label)
        # Who decides the record, as on the lamella task list, with the two
        # states a grid task has: Automated or Supervised. A click toggles.
        self.btn_attention = QToolButton()
        self.btn_attention.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_attention.setIconSize(QSize(14, 14))
        self.btn_attention.setFixedSize(CHIP_WIDTH, CHIP_HEIGHT)
        self.btn_attention.setCursor(Qt.PointingHandCursor)
        self.btn_attention.setFocusPolicy(Qt.NoFocus)
        self.btn_attention.clicked.connect(self._toggle_attention)
        layout.addWidget(self.btn_attention)
        # What it requires is edited here, as on the lamella task list; its
        # settings are on the Protocol tab.
        self.btn_edit = IconToolButton(
            icon="mdi:pencil",
            tooltip="Edit what this task requires",
            size=BTN_SIZE.width(),
        )
        self.btn_edit.clicked.connect(lambda: self.edit_clicked.emit(self.task_name))
        layout.addWidget(self.btn_edit)
        # Reordered by dragging the handle, as the lamella task list is.
        self.drag_handle = QLabel()
        self.drag_handle.setFixedSize(DRAG_HANDLE_WIDTH, DRAG_HANDLE_HEIGHT)
        self.drag_handle.setPixmap(drag_handle_pixmap())
        self.drag_handle.setStyleSheet("background: transparent;")
        self.drag_handle.setCursor(Qt.CursorShape.OpenHandCursor)
        layout.addWidget(self.drag_handle)
        self.refresh()

    @property
    def task_name(self) -> str:
        return self.config.task_name

    @property
    def selectable(self) -> bool:
        return self._reason is None

    def set_unavailable(self, reason: Optional[str]) -> None:
        self._reason = reason
        self.refresh()

    @property
    def supervised(self) -> bool:
        return self.protocol.get_attention(self.task_name) is Attention.supervised

    def _toggle_attention(self) -> None:
        why = self.protocol.set_attention(
            self.task_name,
            Attention.automated if self.supervised else Attention.supervised,
        )
        if why:
            logging.warning(why)
            return
        self.refresh()
        self.attention_changed.emit(self.task_name)

    def refresh(self) -> None:
        available = self._reason is None
        self.name_label.setText(self.task_name)
        style_with_tooltip(
            self.name_label,
            f"background: transparent; font-size: {NAME_PX}px; "
            f"color: {NEUTRAL_200 if available else NEUTRAL_550};",
        )
        # the right column, most pressing first: why it cannot run, what it
        # waits for (as the lamella list says it), and otherwise the task's kind
        requires = self.protocol.requirements(self.task_name)
        after = "after " + requires_phrase(requires) if requires else ""
        if not available:
            detail = self._reason
        elif after:
            detail = after
        else:
            detail = self.config.display_name
        self.detail_label.setText(detail)
        self.detail_label.setToolTip(
            self._reason or "\n".join(t for t in (self.config.display_name, after) if t)
        )
        style_with_tooltip(
            self.detail_label,
            f"background: transparent; "
            f"color: {NEUTRAL_700}; "
            f"font-size: {_SIDE_FONT_PX}px;",
        )
        self._refresh_attention()
        if not available and self.checkbox.isChecked():
            self.checkbox.setChecked(False)
        self.checkbox.setEnabled(available)
        self.checkbox.setToolTip(self._reason or "")

    def _refresh_attention(self) -> None:
        supervised = self.supervised
        state = "supervised" if supervised else "automated"
        if supervised:
            colour = tokens.PRIMARY_COLOR
            tooltip = (
                "Supervised — a question the task needs answered is asked while "
                "it runs, and the run waits for your answer. Click for Automated."
            )
        else:
            colour = tokens.AUTOMATED_COLOR
            tooltip = (
                "Automated — runs without anyone; what it acquired is recorded "
                "for you to check. Click for Supervised."
            )
        self.btn_attention.setText(ATTENTION_LABELS[state])
        self.btn_attention.setIcon(fibsem_icon(CHIP_ICONS[state], color=colour))
        self.btn_attention.setToolTip(tooltip)
        self.btn_attention.setStyleSheet(chip_style(colour))


# Rows sit flush with their header, as on the Lamella page.
_LIST_STYLESHEET = (
    stylesheets.LIST_WIDGET_STYLESHEET + "QListWidget::item { padding: 0; }"
)


def _list() -> QListWidget:
    widget = QListWidget()
    widget.setStyleSheet(_LIST_STYLESHEET)
    widget.setSelectionMode(QListWidget.NoSelection)
    widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    widget.setResizeMode(QListWidget.Adjust)
    widget.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    return widget


def _empty_line(text: str) -> QLabel:
    """What an empty list means and what to do about it; hidden once it fills."""
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet(
        f"font-size: 11px; color: {TEXT_MUTED_COLOR}; background: transparent; "
        "padding: 6px 8px;"
    )
    return label


def _screen_all_button() -> QToolButton:
    """Screen all, on the grid list's header beside what it acts on: secondary,
    since Run on the shared footer is the tab's primary action."""
    button = QToolButton()
    button.setText("Screen all")
    button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    button.setIcon(fibsem_icon("mdi:play-circle-outline", color=TEXT_COLOR))
    button.setIconSize(QSize(14, 14))
    button.setFixedHeight(CHIP_HEIGHT)
    button.setCursor(Qt.PointingHandCursor)
    button.setStyleSheet(
        f"QToolButton {{ border: 1px solid {BORDER_COLOR}; border-radius: 3px; "
        f"padding: 0 6px 0 4px; background: transparent; color: {TEXT_COLOR}; "
        f"font-size: {DETAIL_PX}px; }}"
        f"QToolButton:hover {{ background: {BORDER_COLOR}; }}"
        f"QToolButton:disabled {{ color: {NEUTRAL_550}; }}"
    )
    button.setToolTip(
        "Run inventory, then the ticked tasks on every grid it finds present"
    )
    return button


def _add_row(widget: QListWidget, row: QWidget, key: Optional[str] = None) -> None:
    item = QListWidgetItem(widget)
    item.setSizeHint(QSize(0, _ROW_HEIGHT))
    flags = Qt.ItemFlag.ItemIsEnabled
    if key is not None:
        item.setData(Qt.ItemDataRole.UserRole, key)
        flags |= Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsDragEnabled
    item.setFlags(flags)
    widget.addItem(item)
    widget.setItemWidget(item, row)


# ---------------------------------------------------------------------------
# The view
# ---------------------------------------------------------------------------


class GridRunOptionsPopup(OptionsPopup):
    """How a grid run ends, behind the ⚙ on the task list's header, as the Lamella
    page keeps its workflow settings: unload the last grid, write the screening
    report. Set once if at all, and kept for the session: every run started from
    this page ends the way they say."""

    changed = pyqtSignal()

    def __init__(self, parent: QWidget) -> None:
        super().__init__("Grid workflow options", parent)
        self.unload_check = QCheckBox("Unload the grid when the run finishes")
        self.unload_check.setToolTip(
            "Return the last grid to the magazine once every queued task has "
            "run. A Stop leaves it loaded."
        )
        self.report_check = QCheckBox("Write the grid screening report at the end")
        self.report_check.setToolTip(
            "Writes the grid screening report to the experiment folder, after "
            "the unload. No grid has a verdict yet then: write it again once "
            "they do."
        )
        # Unload is off until asked for, since it moves the hardware after the
        # operator has stopped watching; the report only writes a file, so it is on.
        self.unload_check.setChecked(False)
        self.report_check.setChecked(True)
        for box in (self.unload_check, self.report_check):
            box.setStyleSheet(
                f"QCheckBox {{ color: {TEXT_COLOR}; font-size: 12px; "
                "background: transparent; } "
                f"QCheckBox:disabled {{ color: {TEXT_MUTED_COLOR}; }}"
            )
            box.toggled.connect(self.changed)
            self.body.addWidget(box)
        self.set_availability(can_unload=False)

    def set_availability(self, can_unload: bool) -> None:
        """What this setup can do at the end of a run: an unload only with a
        loader to return the grid to. The choice is kept, so it comes back with
        the microscope."""
        self.unload_check.setVisible(can_unload)
        self.changed.emit()

    def unload_at_end(self) -> bool:
        """Whether a run unloads its last grid once the queue is done."""
        return not self.unload_check.isHidden() and self.unload_check.isChecked()

    def report_at_end(self) -> bool:
        """Whether a run writes the grid screening report as it ends."""
        return self.report_check.isChecked()


class GridWorkflowWidget(QWidget):
    selection_changed = pyqtSignal()
    # The one-click path. The host runs it: inventory, then the ticked tasks on
    # every present grid.
    screen_all_requested = pyqtSignal()
    # The task order was changed here and saved to the protocol.
    protocol_changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._experiment: Optional[Experiment] = None
        self._microscope = None
        self._grid_rows: Dict[str, _GridRow] = {}
        self._task_rows: Dict[str, _TaskRow] = {}
        self._controls_enabled = True
        self._setup_ui()
        self.refresh()

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # The Lamella page's headers: tick box, title, muted count, then what
        # acts on the list.
        self.grid_header = ListHeader("Grids", title_width=_NAME_MIN_WIDTH)
        self.grid_header.select_all_changed.connect(self.set_all_grids_selected)
        self.grid_header.add_widget(column_label("Status"), 1)
        self.btn_screen_all = _screen_all_button()
        self.btn_screen_all.clicked.connect(self.screen_all_requested)
        self.grid_header.add_widget(self.btn_screen_all)
        root.addWidget(self.grid_header)
        self.grid_list = _list()
        root.addWidget(self.grid_list, 1)
        self.grid_empty = _empty_line(
            "No grids in this experiment yet. Run an inventory on the Grids tab, "
            "or press Screen all to inventory and run in one go."
        )
        root.addWidget(self.grid_empty)

        self.task_header = ListHeader("Tasks")
        self.task_header.select_all_changed.connect(self.set_all_tasks_selected)
        self.task_header.add_stretch()
        # ⚙ where the Lamella page has its workflow settings: how a run ends.
        self.options = GridRunOptionsPopup(self)
        self.btn_settings = IconToolButton(
            icon="mdi:cog-outline", tooltip="Grid workflow options", size=24
        )
        self.btn_settings.clicked.connect(
            lambda: self.options.show_under(self.btn_settings)
        )
        self.options.changed.connect(self._refresh_settings_button)
        self._refresh_settings_button()
        self.task_header.add_widget(self.btn_settings)
        self.btn_help = help_button(TASK_LIST_HINTS)
        self.task_header.add_widget(self.btn_help)
        root.addWidget(self.task_header)
        self.task_list = ReorderableListWidget()
        self.task_list.setStyleSheet(_LIST_STYLESHEET)
        self.task_list.setSelectionMode(QListWidget.SingleSelection)
        self.task_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.task_list.setResizeMode(QListWidget.Adjust)
        self.task_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.task_list.reordered.connect(self._on_reordered)
        root.addWidget(self.task_list, 1)
        self.task_empty = _empty_line(
            "No grid tasks in the protocol. Add them on the Protocol tab's Grid page."
        )
        root.addWidget(self.task_empty)
        # No footer: what the selection allows is the status line's to say
        # (`selection_summary`).

    # -- model -----------------------------------------------------------------

    def set_experiment(self, experiment: Optional[Experiment]) -> None:
        self._experiment = experiment
        self._rebuild()

    def set_microscope(self, microscope) -> None:
        self._microscope = microscope
        stage = self.stage
        self.options.set_availability(
            can_unload=stage is not None and stage.magazine is not None,
        )
        self.refresh()

    def unload_at_end(self) -> bool:
        """The ⚙'s choice: a run unloads its last grid once the queue is done."""
        return self.options.unload_at_end()

    def report_at_end(self) -> bool:
        """The ⚙'s choice: a run writes the grid screening report as it ends."""
        return self.options.report_at_end()

    def _refresh_settings_button(self) -> None:
        """The ⚙ in the accent colour while an option is off its default, as the
        Lamella page's is while something in it is set."""
        changed = self.unload_at_end() or not self.report_at_end()
        self.btn_settings.setIcon(
            fibsem_icon(
                "mdi:cog-outline",
                color=ACCENT_COLOR if changed else tokens.GRAY_ICON_COLOR,
            )
        )
        self.btn_settings.setToolTip(
            "Grid workflow options (some set)" if changed else "Grid workflow options"
        )

    @property
    def stage(self):
        return getattr(self._microscope, "sample_handling", None)

    def _protocol(self):
        if self._experiment is None or self._experiment.task_protocol is None:
            return None
        return self._experiment.grid_protocol

    def _rebuild(self) -> None:
        """New experiment, or the protocol's task set changed: new rows."""
        checked_grids = {
            n for n, r in self._grid_rows.items() if r.checkbox.isChecked()
        }
        checked_tasks = {
            n for n, r in self._task_rows.items() if r.checkbox.isChecked()
        }
        self.grid_list.clear()
        self._grid_rows = {}
        self.task_list.clear()
        self._task_rows = {}
        if self._experiment is not None:
            for grid in self._experiment.grids:
                row = _GridRow(grid)
                row.selection_changed.connect(lambda *_: self._on_selection())
                _add_row(self.grid_list, row)
                self._grid_rows[grid.name] = row
                row.checkbox.setChecked(grid.name in checked_grids)
        protocol = self._protocol()
        if protocol is not None:
            for name in protocol.ordered_task_names:
                row = _TaskRow(protocol.task_config[name], protocol)
                row.selection_changed.connect(lambda *_: self._on_selection())
                row.attention_changed.connect(self._on_attention_changed)
                row.edit_clicked.connect(self._on_edit_requested)
                _add_row(self.task_list, row, key=name)
                self._task_rows[name] = row
                # Every task ticked by default: the usual run is the whole protocol.
                row.checkbox.setChecked(name in checked_tasks or not checked_tasks)
        self._sync_headers()
        self.refresh()

    def _sync_headers(self) -> None:
        """Each header's box reads its rows: ticked, unticked, or partly, as on
        the Lamella page, so its next click means the opposite. Only the rows
        that can be ticked count: an absent grid never is."""
        for header, rows in (
            (self.grid_header, self._grid_rows),
            (self.task_header, self._task_rows),
        ):
            # the row's own word, not isEnabled(): that is False for every row
            # while the Workflow tab is, before an experiment is loaded
            boxes = [r.checkbox for r in rows.values() if r.selectable]
            header.sync(sum(b.isChecked() for b in boxes), len(boxes))

    def refresh(self) -> None:
        """Redraw from the inventory and the records; no hardware call."""
        inventory: Dict[str, GridInventoryEntry] = {}
        stage = self.stage
        if stage is not None:
            try:
                inventory = {e.name: e for e in stage.grid_inventory() if e.name}
            except Exception as e:  # noqa: BLE001 - drawn as "not present"
                logging.warning(f"Could not read the grid inventory: {e}")
        # Records added since the last rebuild (an inventory just made them) or
        # tasks added on the Protocol view need rows.
        if self._experiment is not None:
            names = [g.name for g in self._experiment.grids]
            protocol = self._protocol()
            tasks = list(protocol.ordered_task_names) if protocol is not None else []
            if names != list(self._grid_rows) or tasks != list(self._task_rows):
                self._rebuild()
                return
        for row in self._grid_rows.values():
            row.set_inventory(inventory.get(row.grid.name))
        for row in self._task_rows.values():
            row.set_unavailable(task_unavailable_reason(row.config, self._microscope))
        self.grid_empty.setVisible(not self._grid_rows)
        self.task_empty.setVisible(not self._task_rows)
        present = sum(1 for r in self._grid_rows.values() if r.is_present)
        self.grid_header.set_count(
            f"{len(self._grid_rows)} \u00b7 {present} present"
            if self._grid_rows
            else ""
        )
        self.task_header.set_count(str(len(self._task_rows)) if self._task_rows else "")
        self._sync_headers()
        self._apply_controls()

    def refresh_grid(self, grid: GridRecord) -> None:
        row = self._grid_rows.get(grid.name)
        if row is not None:
            row.refresh()

    # -- selection -------------------------------------------------------------

    def get_selected_grids(self) -> List[GridRecord]:
        return [
            r.grid
            for r in self._grid_rows.values()
            if r.checkbox.isChecked() and r.is_present
        ]

    def get_selected_task_names(self) -> List[str]:
        """Ticked tasks, in the list's (the protocol's) order."""
        return [n for n, r in self._task_rows.items() if r.checkbox.isChecked()]

    def set_all_grids_selected(self, checked: bool) -> None:
        for row in self._grid_rows.values():
            if row.is_present:
                row.checkbox.setChecked(checked)
        # The window clears the ticks too (a run starting, an add); the header
        # follows, so its next click selects all rather than doing nothing.
        self._sync_headers()

    def set_all_tasks_selected(self, checked: bool) -> None:
        for row in self._task_rows.values():
            if row.checkbox.isEnabled():
                row.checkbox.setChecked(checked)
        self._sync_headers()

    def _on_selection(self) -> None:
        self._sync_headers()
        self._apply_controls()
        self.selection_changed.emit()

    def selection_summary(self) -> str:
        """What the ticks come to, for the status line: "3 grids, 2 tasks",
        and on a loader the exchanges that run would make."""
        grids = self.get_selected_grids()
        tasks = self.get_selected_task_names()
        text = (
            f"{len(grids)} grid{'s' if len(grids) != 1 else ''}, "
            f"{len(tasks)} task{'s' if len(tasks) != 1 else ''}"
        )
        stage = self.stage
        if grids and stage is not None and stage.magazine is not None:
            exchanges = exchanges_for(stage, [g.name for g in grids])
            text += f" \u00b7 {exchanges} exchange{'s' if exchanges != 1 else ''}"
        return text

    # -- controls --------------------------------------------------------------

    def set_controls_enabled(self, enabled: bool) -> None:
        """The host's lockout while a run is going."""
        self._controls_enabled = enabled
        self._apply_controls()

    def _apply_controls(self) -> None:
        protocol = self._protocol()
        can_screen = (
            self._controls_enabled
            and self._experiment is not None
            and protocol is not None
            and self.stage is not None
            and bool(self.get_selected_task_names())
        )
        self.btn_screen_all.setEnabled(can_screen)
        self.task_list.setDragEnabled(self._controls_enabled)

    # -- order -----------------------------------------------------------------

    def _on_attention_changed(self, task_name: str) -> None:
        """A chip click changed a task's attention: saved with the protocol,
        like its order."""
        self._save_protocol()
        self.protocol_changed.emit()

    def edit_dialog(self, task_name: str) -> Optional[GridTaskEditDialog]:
        protocol = self._protocol()
        if protocol is None or task_name not in protocol.task_config:
            return None
        return GridTaskEditDialog(
            protocol.task_config[task_name],
            protocol.requirements(task_name),
            protocol.ordered_task_names,
            parent=self,
        )

    def _on_edit_requested(self, task_name: str) -> None:
        dialog = self.edit_dialog(task_name)
        if dialog is not None and dialog.exec_() == QDialog.Accepted:
            self.set_requires(task_name, dialog.requires)

    def set_requires(self, task_name: str, names: List[str]) -> None:
        """``task_name`` now requires ``names``, in run order: saved with the
        protocol, and every row redrawn, since what waits on what changed."""
        protocol = self._protocol()
        if protocol is None or task_name not in protocol.task_config:
            return
        why = protocol.set_requires(task_name, [n for n in names if n != task_name])
        if why:
            logging.warning(why)
            return
        self._save_protocol()
        for row in self._task_rows.values():
            row.refresh()
        self.protocol_changed.emit()

    def _save_protocol(self) -> None:
        if self._experiment is None:
            return
        try:
            self._experiment.save(save_protocol=True)
        except Exception as e:  # noqa: BLE001 - the change is made; say so
            logging.warning(f"Could not save the grid protocol: {e}")

    def _on_reordered(self, names: List[str]) -> None:
        """A drag put the tasks in a new order: into the protocol, and saved."""
        protocol = self._protocol()
        if protocol is None or names == list(protocol.ordered_task_names):
            self._rebuild()
            return
        why = protocol.reorder(list(names))
        if why:
            logging.warning(why)
            self._rebuild()
            return
        self._save_protocol()
        self._rebuild()
        self.protocol_changed.emit()
