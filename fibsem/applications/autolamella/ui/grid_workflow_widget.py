"""Workflow tab · Grids: pick grids, pick and order tasks, run.

The run view for grids, beside the lamella one and sharing its chrome: the Run
and Stop buttons, the timeline on the right, the confirmation before a start.
Grid rows come from the experiment's records with the hardware's word on each
(slot, in beam) drawn as chips; only a present grid can be ticked. Task rows
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
    QCheckBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
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
from fibsem.microscopes._stage import GridInventoryEntry
from fibsem.ui import stylesheets
from fibsem.ui.icon import fibsem_icon
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

_ROW_HEIGHT = 34
_ICON_BTN = 22
_HEADER_STYLE = f"font-size: 12px; font-weight: bold; color: {NEUTRAL_200}; background: transparent;"
_ICON_BTN_STYLE = """
QToolButton { background: transparent; border: none; border-radius: 4px; }
QToolButton:hover { background: rgba(255, 255, 255, 30); }
QToolButton:disabled { background: transparent; }
"""


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
        self.setFixedHeight(_ROW_HEIGHT)
        self.setAttribute(Qt.WA_TranslucentBackground)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 6, 0)
        layout.setSpacing(8)
        self.checkbox = QCheckBox()
        self.checkbox.setStyleSheet("background: transparent;")
        self.checkbox.toggled.connect(
            lambda checked: self.selection_changed.emit(self.grid, checked)
        )
        layout.addWidget(self.checkbox)
        self.name_label = QLabel()
        layout.addWidget(self.name_label)
        self.status_label = ElidedLabel()
        layout.addWidget(self.status_label, 1)
        self._chips = QHBoxLayout()
        self._chips.setSpacing(4)
        layout.addLayout(self._chips)
        self.refresh()

    @property
    def is_present(self) -> bool:
        return self._entry is not None and self._entry.present

    @property
    def in_beam(self) -> bool:
        return self._entry is not None and self._entry.in_beam

    def set_inventory(self, entry: Optional[GridInventoryEntry]) -> None:
        self._entry = entry
        self.refresh()

    def refresh(self) -> None:
        self.name_label.setText(self.grid.name)
        present = self.is_present
        style_with_tooltip(
            self.name_label,
            f"font-weight: bold; background: transparent; "
            f"color: {NEUTRAL_200 if present else NEUTRAL_550};",
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
        if entry is None or not entry.present:
            chips.append(("not present", NEUTRAL_700))
        else:
            chips.append((f"slot {entry.index + 1:02d}", NEUTRAL_550))
            if entry.in_beam:
                chips.append(("in beam", OK_COLOR))
        for label, colour in chips:
            widget = chip(label, colour)
            self._chips.addWidget(widget)
            self._chip_widgets.append(widget)
        # Only a present grid can be run on; an absent one is unticked and stays so.
        if not present and self.checkbox.isChecked():
            self.checkbox.setChecked(False)
        self.checkbox.setEnabled(present)
        self.checkbox.setToolTip(
            "" if present else "Not in the magazine or holder; run an inventory"
        )


class _TaskRow(QWidget):
    selection_changed = pyqtSignal(str, bool)  # task name, checked
    move_up = pyqtSignal(str)
    move_down = pyqtSignal(str)

    def __init__(
        self, config: GridTaskConfig, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self.config = config
        self._reason: Optional[str] = None
        self.setFixedHeight(_ROW_HEIGHT)
        self.setAttribute(Qt.WA_TranslucentBackground)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 6, 0)
        layout.setSpacing(8)
        self.checkbox = QCheckBox()
        self.checkbox.setStyleSheet("background: transparent;")
        self.checkbox.toggled.connect(
            lambda checked: self.selection_changed.emit(self.task_name, checked)
        )
        layout.addWidget(self.checkbox)
        self.name_label = QLabel()
        layout.addWidget(self.name_label)
        self.detail_label = ElidedLabel()
        layout.addWidget(self.detail_label, 1)
        self.btn_up = QToolButton()
        self.btn_down = QToolButton()
        for button, icon, signal in (
            (self.btn_up, "mdi:chevron-up", self.move_up),
            (self.btn_down, "mdi:chevron-down", self.move_down),
        ):
            button.setFixedSize(_ICON_BTN, _ICON_BTN)
            button.setIconSize(QSize(16, 16))
            button.setStyleSheet(_ICON_BTN_STYLE)
            button.setIcon(fibsem_icon(icon, color=stylesheets.GRAY_ICON_COLOR))
            button.clicked.connect(lambda _c, s=signal: s.emit(self.task_name))
            layout.addWidget(button)
        self.btn_up.setToolTip("Run earlier")
        self.btn_down.setToolTip("Run later")
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
            f"font-weight: bold; background: transparent; "
            f"color: {NEUTRAL_200 if available else NEUTRAL_550};",
        )
        self.detail_label.setText(
            self.config.display_name if available else self._reason
        )
        self.detail_label.setStyleSheet(
            f"font-size: 11px; color: {TEXT_MUTED_COLOR}; background: transparent;"
        )
        if not available and self.checkbox.isChecked():
            self.checkbox.setChecked(False)
        self.checkbox.setEnabled(available)
        self.checkbox.setToolTip(self._reason or "")


class _ListHeader(QWidget):
    select_all_changed = pyqtSignal(bool)

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 6, 0)
        self.select_all = QCheckBox()
        self.select_all.setStyleSheet("background: transparent;")
        self.select_all.setToolTip("Select all / none")
        self.select_all.toggled.connect(self.select_all_changed)
        layout.addWidget(self.select_all)
        self.title = QLabel(title)
        self.title.setStyleSheet(_HEADER_STYLE)
        layout.addWidget(self.title)
        layout.addStretch(1)
        self.trailing = QLabel()
        self.trailing.setStyleSheet(
            f"font-size: 11px; color: {TEXT_MUTED_COLOR}; background: transparent;"
        )
        layout.addWidget(self.trailing)


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


def _add_row(widget: QListWidget, row: QWidget) -> None:
    item = QListWidgetItem(widget)
    item.setSizeHint(QSize(0, _ROW_HEIGHT))
    item.setFlags(Qt.ItemFlag.ItemIsEnabled)
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
        self.task_list = _list()
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
        self.btn_screen_all.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
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
                row.move_up.connect(lambda n: self._move_task(n, -1))
                row.move_down.connect(lambda n: self._move_task(n, +1))
                _add_row(self.task_list, row)
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
        """How many of these grids are not in the beam now: the exchanges a run
        would make on a loader, zero on a fixed holder."""
        stage = self.stage
        if stage is None or stage.loader is None:
            return 0
        count = 0
        for g in grids:
            row = self._grid_rows.get(g.name)
            if row is not None and not row.in_beam:
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
        for row in self._task_rows.values():
            row.btn_up.setEnabled(self._controls_enabled)
            row.btn_down.setEnabled(self._controls_enabled)

    # -- order -----------------------------------------------------------------

    def _move_task(self, name: str, delta: int) -> None:
        protocol = self._protocol()
        if protocol is None:
            return
        order = list(protocol.ordered_task_names)
        i = order.index(name)
        j = i + delta
        if not 0 <= j < len(order):
            return
        order[i], order[j] = order[j], order[i]
        protocol.order = order
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
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Screen all grids" if screen_all else "Run grid workflow")
        self.setMinimumWidth(460)
        self.setStyleSheet(f"background: {BACKGROUND};")
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        title = QLabel(
            "Run inventory, then the tasks on every present grid"
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
                "" if exchanges or screen_all else "all in the beam",
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
        self.btn_run = QPushButton("Run")
        self.btn_run.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.btn_run.clicked.connect(self.accept)
        self.btn_run.setDefault(True)
        buttons.addWidget(self.btn_cancel)
        buttons.addWidget(self.btn_run)
        layout.addLayout(buttons)
