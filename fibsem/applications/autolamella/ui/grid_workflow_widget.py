"""Workflow tab · Grids: pick grids, pick and order tasks, run.

The run view for grids, beside the lamella one and sharing its chrome: the Run
and Stop buttons, the timeline on the right, the confirmation before a start.
Grid rows come from the experiment's records with the hardware's word on each
(slot, loaded) drawn as chips; only a present grid can be ticked. Task rows
come from the grid protocol in its order, and the order can be changed here.
Settings live on Grids → Protocol.

"Screen all grids" is the Arctis user's one action: inventory, every present
grid, the ticked tasks.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QFrame,
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

from fibsem.applications.autolamella.structures import Experiment, GridRecord
from fibsem.applications.autolamella.ui.grid_card_widget import grid_headline
from fibsem.applications.autolamella.workflows.tasks.grid import (
    FluorescenceOverviewGridTaskConfig,
    GridTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.grid.manager import (
    LOAD_ENTRY_NAME,
    plan_grid_run,
)
from fibsem.microscopes._stage import GridInventoryEntry, GridSlotState
from fibsem.ui import stylesheets
from fibsem.ui.icon import (
    DRAG_HANDLE_HEIGHT,
    DRAG_HANDLE_WIDTH,
    drag_handle_pixmap,
    fibsem_icon,
)
from fibsem.ui.stylesheets import CANVAS_BG
from fibsem.ui.tokens import (
    NEUTRAL_200,
    NEUTRAL_550,
    NEUTRAL_700,
    OK_COLOR,
    TEXT_MUTED_COLOR,
)
from fibsem.ui.widgets.custom_widgets import ElidedLabel, chip, style_with_tooltip
from fibsem.ui.widgets.preflight import (
    BACKGROUND,
    ON_PANEL,
    TEXT_STRONG,
    detail_block,
    meta_label,
    metric,
)

# The lamella list's metrics, so the two run views read alike.
_ROW_HEIGHT = 40
_NAME_MIN_WIDTH = 160
_SIDE_COLUMN_WIDTH = 170  # the muted right-hand column: the task's kind, or why not
# The slot column is narrower: "slot 02" is all it ever says, and at the task
# column's width a row carrying the Loaded pill overran the list and lost it.
_SLOT_COLUMN_WIDTH = 56
_SIDE_FONT_PX = 10
_HEADER_STYLE = "font-weight: bold; background: transparent;"


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
        layout.setContentsMargins(6, 3, 6, 3)
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
            f"background: transparent; color: {NEUTRAL_200 if present else NEUTRAL_550};",
        )
        text, colour = grid_headline(self.grid)
        self.status_label.setText(text)
        self.status_label.setStyleSheet(
            f"font-size: 11px; color: {colour}; background: transparent;"
        )
        for old in self._chip_widgets:
            self._chips.removeWidget(old)
            old.deleteLater()
        self._chip_widgets = []
        entry = self._entry
        chips: List[Tuple[str, str]] = []
        if entry is not None and entry.state is GridSlotState.UNKNOWN:
            chips.append(("not scanned", NEUTRAL_700))
        elif entry is None or not entry.present:
            chips.append(("not present", NEUTRAL_700))
        elif entry.loaded:
            chips.append(("Loaded", OK_COLOR))
        for label, chip_colour in chips:
            widget = chip(label, chip_colour)
            widget.setFixedHeight(20)  # a pill: the radius is half the height
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


class _TaskRow(QWidget):
    selection_changed = pyqtSignal(str, bool)  # task name, checked

    def __init__(
        self, config: GridTaskConfig, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self.config = config
        self._reason: Optional[str] = None
        self.setAttribute(Qt.WA_TranslucentBackground)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
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
        # The kind, or why this system cannot run it: the muted right column. A
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

    def set_unavailable(self, reason: Optional[str]) -> None:
        self._reason = reason
        self.refresh()

    def refresh(self) -> None:
        available = self._reason is None
        self.name_label.setText(self.task_name)
        style_with_tooltip(
            self.name_label,
            f"background: transparent; "
            f"color: {NEUTRAL_200 if available else NEUTRAL_550};",
        )
        self.detail_label.setText(
            self.config.display_name if available else self._reason
        )
        self.detail_label.setToolTip(self._reason or "")
        style_with_tooltip(
            self.detail_label,
            f"background: transparent; color: {NEUTRAL_700}; "
            f"font-size: {_SIDE_FONT_PX}px;",
        )
        if not available and self.checkbox.isChecked():
            self.checkbox.setChecked(False)
        self.checkbox.setEnabled(available)
        self.checkbox.setToolTip(self._reason or "")


class _ListHeader(QWidget):
    select_all_changed = pyqtSignal(bool)

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"background: {CANVAS_BG};")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(8)
        self.select_all = QCheckBox(title)
        self.select_all.setStyleSheet(_HEADER_STYLE)
        self.select_all.setMinimumWidth(24 + 8 + _NAME_MIN_WIDTH)
        self.select_all.setToolTip("Select all / none")
        self.select_all.toggled.connect(self.select_all_changed)
        layout.addWidget(self.select_all)
        layout.addStretch(1)
        self.trailing = QLabel()
        self.trailing.setStyleSheet(
            f"font-size: 11px; color: {TEXT_MUTED_COLOR}; background: transparent;"
        )
        layout.addWidget(self.trailing)


class _DraggableList(QListWidget):
    """The task list: InternalMove drag-and-drop that says the new order after a
    drop. Qt drops the item widgets when items move, so the owner rebuilds rows."""

    reordered = pyqtSignal(list)  # task names, in the new order

    def dropEvent(self, event) -> None:
        super().dropEvent(event)
        names = [
            self.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.count())
            if self.item(i).data(Qt.ItemDataRole.UserRole) is not None
        ]
        self.reordered.emit(names)


def _list() -> QListWidget:
    widget = QListWidget()
    widget.setStyleSheet(stylesheets.LIST_WIDGET_STYLESHEET)
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
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        self.grid_header = _ListHeader("Grids")
        self.grid_header.select_all_changed.connect(self.set_all_grids_selected)
        root.addWidget(self.grid_header)
        self.grid_list = _list()
        root.addWidget(self.grid_list, 1)
        self.grid_empty = _empty_line(
            "No grids in this experiment yet. Run an inventory on the Grids tab, "
            "or press Screen all grids to inventory and run in one go."
        )
        root.addWidget(self.grid_empty)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #3a3d42;")
        root.addWidget(sep)

        self.task_header = _ListHeader("Tasks")
        self.task_header.select_all_changed.connect(self.set_all_tasks_selected)
        root.addWidget(self.task_header)
        self.task_list = _DraggableList()
        self.task_list.setStyleSheet(stylesheets.LIST_WIDGET_STYLESHEET)
        self.task_list.setSelectionMode(QListWidget.SingleSelection)
        self.task_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.task_list.setResizeMode(QListWidget.Adjust)
        self.task_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.task_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.task_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.task_list.reordered.connect(self._on_reordered)
        root.addWidget(self.task_list, 1)
        self.task_empty = _empty_line(
            "No grid tasks in the protocol. Add them on the Protocol tab's Grid page."
        )
        root.addWidget(self.task_empty)
        self.task_hint = QLabel(
            "Settings are on the Protocol tab's Grid page. The order here is the "
            "order they run on each grid."
        )
        self.task_hint.setWordWrap(True)
        self.task_hint.setStyleSheet(
            f"font-size: 11px; color: {TEXT_MUTED_COLOR}; background: transparent;"
        )
        root.addWidget(self.task_hint)

        self.summary_label = QLabel()
        self.summary_label.setStyleSheet(
            f"font-size: 11px; color: {TEXT_MUTED_COLOR}; background: transparent;"
        )
        root.addWidget(self.summary_label)

        bottom = QHBoxLayout()
        self.btn_screen_all = QPushButton("Screen all grids")
        # Secondary: Run, on the shared footer, is the primary action of the tab.
        self.btn_screen_all.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.btn_screen_all.setIcon(
            fibsem_icon("mdi:play-circle", color=stylesheets.GRAY_ICON_COLOR)
        )
        self.btn_screen_all.setToolTip(
            "Run inventory, then the ticked tasks on every grid it finds present"
        )
        self.btn_screen_all.clicked.connect(self.screen_all_requested)
        bottom.addWidget(self.btn_screen_all)
        bottom.addStretch(1)
        root.addLayout(bottom)

    # -- model -----------------------------------------------------------------

    def set_experiment(self, experiment: Optional[Experiment]) -> None:
        self._experiment = experiment
        self._rebuild()

    def set_microscope(self, microscope) -> None:
        self._microscope = microscope
        self.refresh()

    @property
    def stage(self):
        return getattr(self._microscope, "_stage", None)

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
                row = _TaskRow(protocol.task_config[name])
                row.selection_changed.connect(lambda *_: self._on_selection())
                _add_row(self.task_list, row, key=name)
                self._task_rows[name] = row
                # Every task ticked by default: the usual run is the whole protocol.
                row.checkbox.setChecked(name in checked_tasks or not checked_tasks)
        # The header box reads the rows, so its next click means the opposite.
        for header, rows in (
            (self.grid_header, self._grid_rows),
            (self.task_header, self._task_rows),
        ):
            header.select_all.blockSignals(True)
            header.select_all.setChecked(
                bool(rows) and all(r.checkbox.isChecked() for r in rows.values())
            )
            header.select_all.blockSignals(False)
        self.refresh()

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
        self.grid_header.trailing.setText(
            f"{present} of {len(self._grid_rows)} present" if self._grid_rows else ""
        )
        self._apply_controls()
        self._update_summary()

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

    def set_all_tasks_selected(self, checked: bool) -> None:
        for row in self._task_rows.values():
            if row.checkbox.isEnabled():
                row.checkbox.setChecked(checked)

    def exchanges_for(self, grids: List[GridRecord]) -> int:
        """How many of these grids are not loaded now: the exchanges a run
        would make on a loader, zero on a fixed holder."""
        stage = self.stage
        if stage is None or stage.loader is None:
            return 0
        count = 0
        for g in grids:
            row = self._grid_rows.get(g.name)
            if row is not None and not row.loaded:
                count += 1
        return count

    def _on_selection(self) -> None:
        self._update_summary()
        self._apply_controls()
        self.selection_changed.emit()

    def _update_summary(self) -> None:
        grids = self.get_selected_grids()
        tasks = self.get_selected_task_names()
        exchanges = self.exchanges_for(grids)
        text = (
            f"{len(grids)} grid{'s' if len(grids) != 1 else ''}, "
            f"{len(tasks)} task{'s' if len(tasks) != 1 else ''} selected"
        )
        if grids:
            text += f" · {exchanges} exchange{'s' if exchanges != 1 else ''}"
        self.summary_label.setText(text)

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

    def _on_reordered(self, names: List[str]) -> None:
        """A drag put the tasks in a new order: into the protocol, and saved."""
        protocol = self._protocol()
        if protocol is None or names == list(protocol.ordered_task_names):
            self._rebuild()
            return
        protocol.order = list(names)
        if self._experiment is not None:
            try:
                self._experiment.save(save_protocol=True)
            except Exception as e:  # noqa: BLE001 - the order is changed; say so
                logging.warning(f"Could not save the grid protocol: {e}")
        self._rebuild()
        self.protocol_changed.emit()


# ---------------------------------------------------------------------------
# The confirmation
# ---------------------------------------------------------------------------


class GridRunPreflightDialog(QDialog):
    """What a grid run is about to do: how many grids and tasks, how many
    exchanges, where the files go, and the first steps of the plan.

    Its own dialog rather than the lamella preflight: that one is built on a
    lamella estimate with durations, and grid tasks have no duration estimate
    yet. When they do, this grows a time column.
    """

    _STEPS_SHOWN = 12

    def __init__(
        self,
        task_names: List[str],
        grid_names: List[str],
        exchanges: int,
        output_root: str,
        screen_all: bool = False,
        adding: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        """``adding``: the same plan, going onto the end of a running queue
        rather than starting one."""
        super().__init__(parent)
        self.setWindowTitle(
            "Add to the queue"
            if adding
            else "Screen all grids"
            if screen_all
            else "Run grid workflow"
        )
        self.setMinimumWidth(460)
        self.setStyleSheet(f"background: {BACKGROUND};")
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        title = QLabel(
            "Add the tasks on the selected grids after everything queued"
            if adding
            else "Run inventory, then the tasks on every present grid"
            if screen_all
            else "Run the tasks on the selected grids"
        )
        title.setStyleSheet(
            f"font-size: 14px; font-weight: bold; color: {TEXT_STRONG}; {ON_PANEL}"
        )
        layout.addWidget(title)

        metrics = QHBoxLayout()
        metrics.addWidget(
            metric(
                "Grids",
                "every present" if screen_all else str(len(grid_names)),
                f"{len(grid_names)} known now" if screen_all else "",
            )
        )
        metrics.addWidget(metric("Tasks per grid", str(len(task_names))))
        metrics.addWidget(
            metric(
                "Exchanges",
                "one per grid" if screen_all else str(exchanges),
                "" if exchanges or screen_all else "all loaded",
            )
        )
        layout.addLayout(metrics)

        plan = plan_grid_run(task_names, grid_names)
        lines = [
            f"{grid}  ·  {'load' if step == LOAD_ENTRY_NAME else step}"
            for grid, step in plan[: self._STEPS_SHOWN]
        ]
        if len(plan) > self._STEPS_SHOWN:
            lines.append(f"… and {len(plan) - self._STEPS_SHOWN} more steps")
        if screen_all:
            lines.insert(0, "inventory")
        rows = [("Plan", lines[0])] + [("", line) for line in lines[1:]]
        layout.addWidget(detail_block(rows if lines else [("Plan", "nothing yet")]))
        layout.addWidget(meta_label(f"Output: {output_root}/grids/<grid>/"))
        layout.addWidget(
            meta_label(
                "A grid that will not load is skipped and the run continues. "
                "Stop ends the run at the next step."
            )
        )

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_run = QPushButton("Add" if adding else "Run")
        self.btn_run.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.btn_run.clicked.connect(self.accept)
        self.btn_run.setDefault(True)
        buttons.addWidget(self.btn_cancel)
        buttons.addWidget(self.btn_run)
        layout.addLayout(buttons)
