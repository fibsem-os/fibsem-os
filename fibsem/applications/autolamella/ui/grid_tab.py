"""The Grids tab — experiment grid records (cards) + Protocol / Results.

Extracted from ``AutoLamellaMainUI`` into its own widget. It owns all grid-tab
UI and handlers; execution is driven from the shared Workflow tab. Shared state
(status bar, workflow timeline, the workflow selection widget) is reached
through ``main`` — the AutoLamella main window — which is assumed available for
the widget's lifetime.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSplitter, QTabWidget, QVBoxLayout, QWidget
from superqt.iconify import QIconifyIcon

from fibsem.ui import notification_service
from fibsem.ui.stylesheets import GRAY_ICON_COLOR
from fibsem.ui.widgets.grid_card_widget import GridCardContainer
from fibsem.ui.widgets.grid_protocol_editor_widget import GridProtocolEditorWidget
from fibsem.ui.widgets.grid_results_widget import GridResultsWidget

if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import Experiment, GridRecord


class GridTabWidget(QWidget):
    """Grids tab content: cards (left) + Protocol / Results sub-tabs (right)."""

    def __init__(self, main: QWidget) -> None:
        super().__init__(main)
        self.main = main
        self._setup_ui()

    @property
    def autolamella_ui(self):
        return self.main.autolamella_ui

    @staticmethod
    def tab_icon() -> QIconifyIcon:
        return QIconifyIcon("mdi:grid", color=GRAY_ICON_COLOR)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # left: experiment grid records (cards)
        self.grids_view = GridCardContainer()
        self.grids_view.add_from_loader_requested.connect(self._on_grids_add_from_loader)
        self.grids_view.remove_requested.connect(self._on_grids_remove)
        self.grids_view.grid_selected.connect(self._on_grid_selected)
        self.grids_view.load_requested.connect(self._on_grid_card_load)
        self.grids_view.unload_requested.connect(self._on_grid_card_unload)
        self.grids_view.quality_changed.connect(self._on_grid_quality_changed)
        left = QWidget()
        left.setMaximumWidth(340)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.grids_view)
        splitter.addWidget(left)

        # right: Protocol (experiment-wide grid task config) | Results (per grid)
        self.grids_right_tabs = QTabWidget()
        self.grid_protocol_editor = GridProtocolEditorWidget()
        self.grids_right_tabs.addTab(self.grid_protocol_editor, "Protocol")
        self.grids_results_widget = GridResultsWidget()
        self.grids_results_widget.lamella_selected.connect(self._on_lamella_selected)
        self.grids_results_widget.edit_positions_requested.connect(self._on_edit_positions)
        self.grids_right_tabs.addTab(self.grids_results_widget, "Results")
        self._position_dialog = None  # keep a ref so the non-modal dialog isn't GC'd

        splitter.addWidget(self.grids_right_tabs)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 99999])

        layout.addWidget(splitter)

    # --- public API (called by the main window) ---

    def set_experiment(self, experiment: "Experiment") -> None:
        """Refresh the grid list + point the protocol editor at the experiment."""
        self.refresh()
        self.grid_protocol_editor.set_experiment(experiment)

    def refresh(self) -> None:
        """Repopulate the grid cards and sync the Workflow-tab grid checklist."""
        ui = self.autolamella_ui
        if ui is None or ui.experiment is None:
            return
        stage = getattr(getattr(ui, "microscope", None), "_stage", None)
        loader_present = getattr(stage, "loader", None) is not None
        self.grids_view.set_grids(
            ui.experiment.grids,
            self._grid_slot_labels(),
            self._grid_beam_names(),
            loader_present,
            self._grid_thumbnails(),
            self._grid_lamella_counts(),
        )
        # keep the Workflow-tab grid checklist + task instances in sync
        gw = getattr(self.main, "grid_workflow_widget", None)
        if gw is not None:
            gw.set_grids(ui.experiment.grids)
            gw.set_protocol(ui.experiment.grid_protocol)

    def run_workflow(self) -> None:
        """Start a grid workflow from the Grids workflow sub-tab selections."""
        # lazy import to avoid a circular import with the main window module
        from fibsem.applications.autolamella.ui.AutoLamellaMainUI import (
            confirm_run_workflow_dialog,
        )

        ui = self.autolamella_ui
        gw = self.main.grid_workflow_widget
        grids = gw.get_selected_grids()
        tasks = gw.get_selected_tasks()
        if not grids or not tasks:
            return
        grid_names = [g.name for g in grids]
        if not confirm_run_workflow_dialog(grid_names, tasks, parent=self.main, unit="grid"):
            return
        self.main._workflow_timeline_initialized = False
        self.main._set_border_state("automated")
        self.main.set_workflow_running(
            f"Running grid workflow: {len(grid_names)} grid(s), {len(tasks)} task(s)"
        )
        ui._start_grid_workflow_thread(list(tasks), grid_names)

    def on_workflow_update(self, info: dict) -> None:
        """Reflect live grid workflow progress: drive the shared timeline + cards."""
        status_msg = info.get("status")
        if status_msg is not None:
            if not self.main._workflow_timeline_initialized:
                queue_items = status_msg.get("queue_items", [])
                if queue_items:
                    self.main.workflow_timeline.set_workflow(queue_items)
                    self.main._workflow_timeline_initialized = True
            self.main.workflow_timeline.update_from_status(status_msg)
            self.main.set_workflow_running(info.get("msg") or "Running grid workflow")
            # drive the shared supervised/automated chip for the running grid task
            task_name = status_msg.get("task_name")
            if status_msg.get("status") == "Completed":
                self.main._active_grid_workflow = False
            elif task_name:
                self.main._current_task_name = task_name
                self.main._active_grid_workflow = True
                self.main._update_supervised_status()
        self.refresh()
        # refresh the Results view for the running grid as artifacts appear
        running = status_msg.get("item_name") if status_msg else None
        if running and self.grids_view.selected_grid is not None \
                and self.grids_view.selected_grid.name == running:
            self._on_grid_selected(self.grids_view.selected_grid)

    # --- data helpers ---

    def _grid_slot_labels(self) -> dict:
        """Map grid name → loader magazine slot label (e.g. '01') for display."""
        ui = self.autolamella_ui
        stage = getattr(getattr(ui, "microscope", None), "_stage", None)
        loader = getattr(stage, "loader", None)
        labels = {}
        if loader is not None:
            for slot in loader.loaded_magazine_slots:
                if slot.loaded_grid is not None:
                    labels[slot.loaded_grid.name] = f"{slot.index + 1:02d}"
        return labels

    def _grid_beam_names(self) -> set:
        """Names of grids currently in a holder working slot (in the beam)."""
        stage = getattr(getattr(self.autolamella_ui, "microscope", None), "_stage", None)
        holder = getattr(stage, "holder", None)
        if holder is None:
            return set()
        return {
            s.loaded_grid.name
            for s in holder.slots.values()
            if s.loaded_grid is not None
        }

    def _grid_thumbnails(self) -> dict:
        """Map grid name → overview thumbnail PNG path (from results), if it exists."""
        out = {}
        for g in self.autolamella_ui.experiment.grids:
            for art in g.results.values():
                thumb = art.get("thumbnail") if isinstance(art, dict) else None
                if thumb and os.path.exists(thumb):
                    out[g.name] = thumb
                    break
        return out

    def _grid_lamella_counts(self) -> dict:
        """Map grid name → number of lamellae linked to it."""
        exp = self.autolamella_ui.experiment
        return {g.name: len(exp.get_lamellae_for_grid(g)) for g in exp.grids}

    # --- card handlers ---

    def _on_grids_add_from_loader(self) -> None:
        """Import grids loaded in the magazine / working slot into the experiment."""
        ui = self.autolamella_ui
        if ui is None or ui.experiment is None or ui.microscope is None:
            notification_service.show_toast(
                "Connect to a microscope and load an experiment first.", "warning"
            )
            return
        before = len(ui.experiment.grids)
        ui.experiment.sync_grids_from_holder(ui.microscope)  # add-only; saves
        self.refresh()
        added = len(ui.experiment.grids) - before
        notification_service.show_toast(
            f"Added {added} grid(s) from the loader." if added
            else "No new grids to add.", "info"
        )

    def _on_grids_remove(self, record: "GridRecord") -> None:
        ui = self.autolamella_ui
        if ui is None or ui.experiment is None:
            return
        ui.experiment.remove_grid(record.name)
        ui.experiment.save()
        self.refresh()

    def _on_grid_quality_changed(self, record: "GridRecord") -> None:
        """Persist a manual grid-quality change. The card already updated its own
        button; just save (no full refresh, which would rebuild every card)."""
        ui = self.autolamella_ui
        if ui is None or ui.experiment is None:
            return
        ui.experiment.save()

    def _on_grid_card_load(self, record: "GridRecord") -> None:
        """Load a grid into the working slot from the Grids tab (delegates to the
        Sample tab's threaded exchange; the spinner + refresh follow from there)."""
        if self.autolamella_ui is not None:
            notification_service.show_toast(
                f"Loading '{record.name}' into the microscope…", "info"
            )
            self.autolamella_ui.request_grid_load(record.name)

    def _on_grid_card_unload(self, record: "GridRecord") -> None:
        if self.autolamella_ui is not None:
            notification_service.show_toast(
                f"Unloading '{record.name}' from the microscope…", "info"
            )
            self.autolamella_ui.request_grid_unload()

    def _on_grid_selected(self, record: "GridRecord") -> None:
        """Show the selected grid's results in the Results sub-tab."""
        exp = self.autolamella_ui.experiment if self.autolamella_ui else None
        slot = self._grid_slot_labels().get(record.name, "") if record else ""
        in_beam = record.name in self._grid_beam_names() if record else False
        self.grids_results_widget.set_grid(record, exp, slot, in_beam)

    def _on_lamella_selected(self, lamella) -> None:
        """Focus a grid's lamella (from the Results table) in the Lamella tab."""
        if lamella is not None:
            self.main._on_lamella_edit(lamella)

    def _on_edit_positions(self, record) -> None:
        """Open the overview to place/move lamella positions for this grid."""
        from fibsem.structures import FibsemImage
        from fibsem.ui.widgets.lamella_selection_dialog import LamellaSelectionDialog
        from fibsem.ui import notification_service

        ui = self.autolamella_ui
        if record is None or ui is None or ui.experiment is None:
            return
        overview = self.grids_results_widget._overview_path()
        if not overview:
            return
        try:
            image = FibsemImage.load(overview)
        except Exception as e:  # noqa: BLE001
            notification_service.show_toast(f"Could not load overview: {e}", "error")
            return
        if image.metadata is None:
            notification_service.show_toast(
                "Overview image has no metadata; can't place positions.", "warning")
            return

        self._position_dialog = LamellaSelectionDialog(
            experiment=ui.experiment, grid_record=record, image=image,
            microscope=ui.microscope, host=ui, parent=self.main)
        self._position_dialog.accepted_positions.connect(self.refresh)
        self._position_dialog.show()
