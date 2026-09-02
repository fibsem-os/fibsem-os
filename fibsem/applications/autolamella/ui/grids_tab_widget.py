"""The Grids tab: the experiment's grid records, one card each.

Mirrors the Lamella tab: cards on the left, Protocol and Results sub-tabs on the
right. The tab's object is what grids are *available to this experiment*, not
what the hardware holds; empty magazine slots never appear here (that is
Microscope → Sample). Slot, present and in-beam are read from the stage's
inventory on every refresh and drawn as chips.

Run inventory, Load and Unload are conveniences over the same ``Stage``
primitives the Sample view uses, run off the GUI thread. Running tasks is not on
this tab; that is Workflow → Grids.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional

from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.structures import Experiment, GridRecord
from fibsem.applications.autolamella.ui.grid_card_widget import GridCardContainer
from fibsem.microscopes._stage import GridInventoryEntry, SampleGrid
from fibsem.ui import stylesheets
from fibsem.ui.icon import fibsem_icon
from fibsem.ui.qt.threading import thread_worker
from fibsem.ui.tokens import ERROR_COLOR, NEUTRAL_200, TEXT_MUTED_COLOR
from fibsem.ui.widgets.sample_loader_widget import _ICON_BTN_STYLE, ICON_INVENTORY

_STRIP_WIDTH = 340


class GridsTabWidget(QWidget):
    """Cards for the experiment's grids, with Protocol and Results beside them."""

    grid_selected = pyqtSignal(object)  # GridRecord | None
    # The experiment's grid records changed here (inventory, a rename, a quality),
    # and were saved; hosts drawing them elsewhere should refresh.
    experiment_changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None, synchronous: bool = False):
        super().__init__(parent)
        self._experiment: Optional[Experiment] = None
        self._microscope = None
        self._synchronous = synchronous
        self._busy = False
        self._controls_enabled = True
        self._worker = None
        self._inventory: Dict[str, GridInventoryEntry] = {}
        self._setup_ui()
        self.refresh()

    # -- layout ----------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(True)

        # -- left: the strip -----------------------------------------------------
        strip = QWidget()
        strip_layout = QVBoxLayout(strip)
        strip_layout.setContentsMargins(8, 8, 4, 4)
        strip_layout.setSpacing(4)

        header = QHBoxLayout()
        header.setSpacing(6)
        title = QLabel("Grids")
        title.setStyleSheet(
            f"font-size: 14px; font-weight: bold; color: {NEUTRAL_200}; "
            "background: transparent;"
        )
        header.addWidget(title)
        header.addStretch(1)
        self.btn_inventory = QToolButton()
        self.btn_inventory.setFixedSize(26, 26)
        self.btn_inventory.setIconSize(QSize(18, 18))
        self.btn_inventory.setStyleSheet(_ICON_BTN_STYLE)
        self.btn_inventory.setIcon(
            fibsem_icon(ICON_INVENTORY, color=stylesheets.GRAY_ICON_COLOR)
        )
        self.btn_inventory.setToolTip(
            "Run inventory: refresh what the hardware holds and add a record for "
            "every grid it finds"
        )
        self.btn_inventory.clicked.connect(self._on_run_inventory)
        header.addWidget(self.btn_inventory)
        strip_layout.addLayout(header)

        self.summary_label = QLabel()
        self.summary_label.setStyleSheet(
            f"font-size: 11px; color: {TEXT_MUTED_COLOR}; background: transparent;"
        )
        strip_layout.addWidget(self.summary_label)

        self.cards = GridCardContainer()
        self.cards.grid_selected.connect(self.grid_selected)
        self.cards.quality_changed.connect(self._on_quality_changed)
        self.cards.load_requested.connect(self._on_load)
        self.cards.unload_requested.connect(self._on_unload)
        self.cards.rename_requested.connect(self._on_rename)
        self.cards.remove_requested.connect(self._on_remove)
        scroll = QScrollArea()
        scroll.setWidget(self.cards)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        strip_layout.addWidget(scroll, 1)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(
            f"font-size: 11px; color: {TEXT_MUTED_COLOR}; background: transparent;"
        )
        strip_layout.addWidget(self.status_label)

        strip.setMaximumWidth(_STRIP_WIDTH)
        splitter.addWidget(strip)
        splitter.setStretchFactor(0, 0)

        # -- right: sub-tabs -----------------------------------------------------
        self.sub_tabs = QTabWidget()
        self.protocol_tab = _placeholder(
            "Grid task settings: the tasks in the experiment's grid protocol and "
            "each one's settings. Coming next."
        )
        self.results_tab = _placeholder(
            "The selected grid's overviews and history. Coming next."
        )
        self.sub_tabs.addTab(self.protocol_tab, "Protocol")
        self.sub_tabs.addTab(self.results_tab, "Results")
        splitter.addWidget(self.sub_tabs)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([_STRIP_WIDTH, 99999])
        layout.addWidget(splitter)

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

    @property
    def selected_grid(self) -> Optional[GridRecord]:
        return self.cards.selected_grid

    @property
    def busy(self) -> bool:
        return self._busy

    def _rebuild(self) -> None:
        """A new experiment: new cards. Everything else is a refresh."""
        selected = self.cards.selected_grid
        self.cards.clear()
        if self._experiment is not None:
            for grid in self._experiment.grids:
                self.cards.add_grid(grid, self._experiment)
            if selected is not None:
                again = self._experiment.get_grid_by_id(selected.id)
                self.cards.select_grid(again)
        self.refresh()

    def refresh(self) -> None:
        """Re-read the inventory and redraw every card from it and its record.
        No hardware call: the stage answers from what it already knows."""
        stage = self.stage
        self._inventory = {}
        has_loader = False
        if stage is not None:
            try:
                self._inventory = {e.name: e for e in stage.grid_inventory() if e.name}
            except Exception as e:  # noqa: BLE001 - drawn as "not present", said once
                logging.warning(f"Could not read the grid inventory: {e}")
            has_loader = stage.loader is not None

        # A record added since the last rebuild (an inventory just created it)
        # needs a card; one removed needs its card gone.
        if self._experiment is not None:
            known = {card.grid.id for card in self.cards.cards}
            for grid in self._experiment.grids:
                if grid.id not in known:
                    self.cards.add_grid(grid, self._experiment)
        for card in self.cards.cards:
            card.set_inventory(self._inventory.get(card.grid.name), has_loader)
            card.set_controls_enabled(self._controls_enabled and not self._busy)

        n = len(self.cards.cards)
        present = sum(1 for c in self.cards.cards if c.is_present)
        if self._experiment is None:
            self.summary_label.setText("No experiment loaded.")
        elif n == 0:
            self.summary_label.setText(
                "No grids yet. Run inventory to add a record for every grid the "
                "hardware holds."
            )
        else:
            self.summary_label.setText(f"{n} in this experiment · {present} present")
        self.btn_inventory.setEnabled(
            self._experiment is not None
            and stage is not None
            and self._controls_enabled
            and not self._busy
        )

    def set_controls_enabled(self, enabled: bool) -> None:
        """The host's lockout while a workflow owns the loader."""
        self._controls_enabled = enabled
        self.refresh()

    # -- edits -----------------------------------------------------------------

    def _on_quality_changed(self, grid: GridRecord) -> None:
        self._save()
        self.experiment_changed.emit()

    def _on_rename(self, grid: GridRecord, name: str) -> None:
        experiment = self._experiment
        if experiment is None:
            return
        if experiment.get_grid_by_name(name) is not None:
            self._say(f"There is already a grid named {name}.", error=True)
            return
        old = grid.name
        entry = self._inventory.get(old)
        grid.name = name
        # The record links to the hardware by name, so the hardware follows: on the
        # autoloader that writes the slot description, on a fixed holder the
        # occupancy file.
        if entry is not None and entry.present and self.stage is not None:
            try:
                self.stage.assign_grid(entry.slot_name, SampleGrid(name=name))
            except Exception as e:  # noqa: BLE001 - the record is renamed; say so
                logging.warning(f"Renamed the record but not the hardware slot: {e}")
                self._say(
                    f"Renamed, but the slot could not be updated: {e}", error=True
                )
        self._save()
        self.refresh()
        self.experiment_changed.emit()

    def _on_remove(self, grid: GridRecord) -> None:
        """Stop tracking a grid. Its files stay; its lamellae are orphaned, not
        deleted (Experiment.remove_grid)."""
        if self._experiment is None:
            return
        self._experiment.remove_grid(grid.name)
        self.cards.remove_grid(grid)
        self._save()
        self.refresh()
        self.experiment_changed.emit()

    def _save(self) -> None:
        if self._experiment is None:
            return
        try:
            self._experiment.save()
        except Exception as e:  # noqa: BLE001 - a failed save is worth a line
            logging.warning(f"Could not save the experiment: {e}")

    # -- hardware, off the GUI thread --------------------------------------------

    def _on_run_inventory(self) -> None:
        stage = self.stage
        if stage is None or self._experiment is None:
            return
        if stage.loader is not None and not self._confirm(
            "Run inventory",
            "Scan the magazine? The autoloader checks every slot for a grid and "
            "reads the names on the slot descriptions; it takes a moment and the "
            "magazine must not be opened while it runs.",
        ):
            return
        experiment = self._experiment

        def job() -> None:
            stage.run_inventory()
            experiment.sync_grids_from_inventory(stage)

        self._start(job, "Running inventory…", "Inventory complete.")

    def _on_load(self, grid: GridRecord) -> None:
        stage = self.stage
        if stage is None:
            return

        def job() -> None:
            stage.ensure_loaded(grid.name)

        self._start(job, f"Loading {grid.name}…", f"{grid.name} is in the beam.")

    def _on_unload(self, grid: GridRecord) -> None:
        stage = self.stage
        if stage is None:
            return

        def job() -> None:
            stage.unload()

        self._start(
            job, f"Unloading {grid.name}…", f"{grid.name} returned to the magazine."
        )

    def _confirm(self, title: str, text: str) -> bool:
        if self._synchronous:
            return True
        return (
            QMessageBox.question(
                self, title, text, QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            == QMessageBox.Yes
        )

    def _start(self, job: Callable[[], None], doing: str, done: str) -> None:
        if self._busy:
            return
        self._set_busy(True)
        self._say(doing)
        self._done_text = done
        if self._synchronous:
            try:
                job()
                self._on_returned()
            except Exception as e:  # noqa: BLE001 - reported on the strip
                self._on_errored(e)
            finally:
                self._on_finished()
            return
        worker = self._job_worker(job)
        worker.returned.connect(self._on_returned)
        worker.errored.connect(self._on_errored)
        worker.finished.connect(self._on_finished)
        self._worker = worker
        worker.start()

    @thread_worker
    def _job_worker(self, job: Callable[[], None]) -> None:
        job()

    def _on_returned(self, _result: object = None) -> None:
        self._say(self._done_text)

    def _on_errored(self, error: Exception) -> None:
        logging.warning(f"Grid operation failed: {error}")
        self._say(str(error), error=True)

    def _on_finished(self) -> None:
        self._worker = None
        self._set_busy(False)
        self._save()
        self.refresh()
        self.experiment_changed.emit()

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.refresh()

    def _say(self, text: str, error: bool = False) -> None:
        colour = ERROR_COLOR if error else TEXT_MUTED_COLOR
        self.status_label.setStyleSheet(
            f"font-size: 11px; color: {colour}; background: transparent;"
        )
        self.status_label.setText(text)


def _placeholder(text: str) -> QWidget:
    widget = QWidget()
    layout = QVBoxLayout(widget)
    label = QLabel(text)
    label.setWordWrap(True)
    label.setAlignment(Qt.AlignTop)
    label.setStyleSheet(f"color: {TEXT_MUTED_COLOR}; background: transparent;")
    layout.addWidget(label)
    layout.addStretch(1)
    return widget
