from __future__ import annotations

import logging
import os
import sys
import threading
import warnings
from typing import Any, Optional

try:
    sys.modules.pop("PySide6.QtCore")
except Exception:
    pass

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QSplitter,
    QVBoxLayout,
    QWidget,
)

import fibsem.config as fibsem_cfg
from fibsem.applications.autolamella import config as cfg
from fibsem.applications.autolamella.experiment_setup import load_for_adoption
from fibsem.applications.autolamella.poses import remove_lamella
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    AutoLamellaWorkflowConfig,
    Lamella,
)
from fibsem.applications.autolamella.ui.common.run_refusal import refused_during_run
from fibsem.applications.autolamella.ui.dialogs.create_experiment import (
    create_experiment_dialog,
)
from fibsem.applications.autolamella.ui.dialogs.load_experiment import (
    load_experiment_dialog,
)
from fibsem.applications.autolamella.ui.dialogs.overview_plot import (
    create_overview_image_widget,
)
from fibsem.applications.autolamella.ui.grids.overview import QuadOverviewPage
from fibsem.applications.autolamella.ui.grids.tab import GridsTabWidget
from fibsem.applications.autolamella.ui.header import ExperimentHeader
from fibsem.applications.autolamella.ui.lamella.tab import LamellaTab
from fibsem.applications.autolamella.ui.menu_bar import MainMenuBar, MenuActions
from fibsem.applications.autolamella.ui.overview.container import (
    AutoLamellaOverviewContainerTab,
)
from fibsem.applications.autolamella.ui.panel.experiment import ExperimentTab
from fibsem.applications.autolamella.ui.protocol.load import load_task_protocol_dialog
from fibsem.applications.autolamella.ui.protocol.protocol_tab import (
    AutoLamellaProtocolTaskConfigEditor,
)
from fibsem.applications.autolamella.ui.workflow.tab import WorkflowTab
from fibsem.ui import notification_service
from fibsem.ui import utils as fui
from fibsem.ui.app_window import AppWindow
from fibsem.ui.fm.widgets import FMImageViewerWidget
from fibsem.ui.icon import fibsem_icon
from fibsem.ui.run.attention import AttentionRouter
from fibsem.ui.run.view import RunView
from fibsem.ui.stylesheets import GRAY_ICON_COLOR
from fibsem.ui.widgets.canvas.quad_view import MicroscopeViewController
from fibsem.ui.widgets.microscope_panel import MicroscopePanel
from fibsem.workflows.grid.naming import (
    grid_rename_refusal,
    record_inventoried_grids,
    rename_and_save_grid,
)
from fibsem.workflows.protocol_files import adopt_protocol
from fibsem.workflows.session import (
    CONFIG,
    EXPERIMENT,
    GRID_RECORDS,
    MICROSCOPE,
    PROTOCOL,
)
from fibsem.workflows.status import (
    TaskStatus,
    WorkflowStatusEvent,
    WorkflowStatusUpdate,
)
from fibsem.workflows.workflow_edits import save_workflow_change

# Suppress a specific upstream Napari/NumPy warning from shapes miter computation. This
# module no longer imports napari, but the minimap still loads it, and a filter is matched
# by module *name* — so this keeps working and is still worth having.
warnings.filterwarnings(
    "ignore",
    message=r"'where' used without 'out', expect unit?ialized memory in output\. If this is intentional, use out=None\.",
    category=UserWarning,
    module=r"napari\.layers\.shapes\._shapes_utils",
)

# The PDF report needs the `reporting` extra (reportlab). The overview plot does not --
# it is matplotlib and Qt -- so it is imported with the other dialogs above rather than
# here, where a missing reportlab used to take it down too.
REPORTING_AVAILABLE: bool = False
try:
    from fibsem.applications.autolamella.ui.dialogs.report import generate_report_dialog

    REPORTING_AVAILABLE = True
except ImportError as e:
    logging.debug(
        f"Could not import generate_report from fibsem.applications.autolamella.tools.reporting: {e}"
    )

# What to do next, shown in the window's status bar. Each names something visible
# on screen at the moment it is shown -- three of these used to name menus that do
# not exist ("Connection ->", "Experiment ->" are not menus; the File entries are
# "New Experiment" and "Load Experiment"), which is worse than saying nothing.
INSTRUCTIONS = {
    "NOT_CONNECTED": "Connect to the microscope to begin.",
    "NO_EXPERIMENT": "Create or load an experiment to begin.",
    "NO_PROTOCOL": "Load a protocol for this experiment (File \u2192 Load Protocol).",
    "NO_LAMELLA": (
        "Add a lamella position with + in the lamella list, "
        "or mark one on the Overview tab."
    ),
    "AUTOLAMELLA_READY": "Ready to run. Choose lamella and tasks in the Workflow tab.",
}


class AutoLamellaWindow(AppWindow):
    """Main window for AutoLamella UI.

    No napari viewer of its own any more: every tab here renders on the matplotlib
    canvas. napari survives in this application only in labelling and segmentation,
    which open windows of their own (FIB-407).
    """

    def __init__(self):
        super().__init__("AutoLamella")
        self._protocol_lock = threading.RLock()

        # create menus, status bar, and tabs
        self.menu_bar = MainMenuBar(
            self,
            session=self.session,
            actions=MenuActions(
                new_experiment=self.create_experiment,
                load_experiment=self.load_experiment,
                open_experiment_directory=self._open_experiment_directory,
                load_protocol=self.load_protocol,
                save_protocol=self.export_protocol_ui,
                fm_image_viewer=self._open_fm_image_viewer,
                coincidence_viewer=self._open_coincidence_milling_viewer,
                generate_report=self.action_generate_report,
                generate_overview_plot=self.action_generate_overview_plot,
                grid_report=self._on_generate_grid_report,
                app_server=self._on_app_server_dialog,
                export_ml_data=self.export_targeting_ml_data,
                about=self.open_information_dialog,
            ),
            panel=lambda: self.panel,
            system_widget=lambda: self.system_widget,
            view_controller=self._view_controller,
            run_border=self.run_border,
            preferences=self._preferences,
            show_toast=self.show_toast,
            parent=self,
        )
        self.menu_bar.preferences_saved.connect(self._on_preferences_saved)
        self._create_status_bar()
        self.create_tabs()
        self._apply_preferences()
        self._update_instructions()

    def _on_preferences_saved(self, preferences) -> None:
        self._preferences = preferences
        self._apply_preferences()

    def _apply_preferences(self):
        """Apply current preferences to UI state."""
        # The embedded App Server follows its saved switch (set in Tools → App
        # Server) against the connected microscope.
        self.sync_app_server_with_preference()
        d = self._preferences.display
        self.run_border.set_on(d.border_enabled)
        self.toast_manager.problems_only = d.toast_problems_only
        # The lamella strip's density. Guarded because _apply_preferences also runs
        # before the Lamella tab is built.
        if hasattr(self, "lamella_tab"):
            self.lamella_tab.cards.set_mode(d.lamella_card_mode)
        self.menu_bar.apply_preferences(self._preferences)
        self.system_widget.refresh_first_run_offer(self._preferences)

    def _on_generate_grid_report(self):
        """Tools → Reporting → Generate Grid Screening Report.

        The Grids tab does the writing and says how it went on its strip, so the
        window shows that tab first; a refusal that needs no tab is a toast.
        """
        tab = getattr(self, "grids_tab", None)
        experiment = self.experiment
        if tab is None or experiment is None:
            self.show_toast("Open an experiment to report on.", "warning")
            return
        if not experiment.grids:
            self.show_toast(
                "No grids in this experiment. Run inventory on the Grids tab first.",
                "warning",
            )
            return
        self.tab_widget.setCurrentWidget(tab)
        tab.generate_report()

    def _refresh_after_naming(self) -> None:
        self.grids_tab.refresh()
        self.workflow_tab.grid_workflow_widget.refresh()
        self._refresh_grid_context()
        sample = self.panel.sample_widget
        for panel in ("loader_widget", "holder_widget"):
            widget = getattr(sample, panel, None)
            if widget is not None:
                widget.refresh()

    def _say_selection(self, text: str) -> None:
        """What the Workflow tab's selection means, as the status line's last
        instruction: the panel's footer said it, under the lists, before the
        status line could."""
        self._selection_instruction = text
        self._update_instructions()

    def _update_instructions(self):
        """Update the status bar with the current instruction based on application state."""
        if self.status_bar is None:
            return
        is_connected = self.microscope is not None
        experiment = self.experiment
        is_experiment_loaded = experiment is not None
        # Carried here from the panel, which used to render the same ladder into a
        # label of its own: an experiment with no protocol cannot run anything, and
        # this was the one rung the status bar did not have.
        is_protocol_loaded = self.protocol is not None
        has_positions = is_experiment_loaded and len(experiment.positions) > 0

        if not is_connected:
            msg = INSTRUCTIONS["NOT_CONNECTED"]
        elif not is_experiment_loaded:
            msg = INSTRUCTIONS["NO_EXPERIMENT"]
        elif not is_protocol_loaded:
            msg = INSTRUCTIONS["NO_PROTOCOL"]
        elif not has_positions:
            msg = INSTRUCTIONS["NO_LAMELLA"]
        else:
            # What the selection allows, once the Workflow tab has one to say.
            msg = (
                getattr(self, "_selection_instruction", None)
                or INSTRUCTIONS["AUTOLAMELLA_READY"]
            )

        self.status_bar.set_instruction(msg)

    def _refresh_fm_placeholder(self) -> None:
        """Say so on the quad's FM panel when the connected system has no FM."""
        view_controller = getattr(self, "view_controller", None)
        if view_controller is None:
            return
        microscope = self.microscope
        no_fm = microscope is not None and getattr(microscope, "fm", None) is None
        view_controller.fm_canvas.set_placeholder(
            "No fluorescence microscope available" if no_fm else "No image"
        )

    def _on_session_changed(self, what: str, detail: Any) -> None:
        if what == MICROSCOPE:
            self._on_microscope_changed()
        elif what == EXPERIMENT:
            self._setup_experiment_connections()
            self._on_experiment_update()
        elif what == PROTOCOL:
            self._on_experiment_update()
        elif what == GRID_RECORDS:
            self._refresh_after_naming()
        elif what == CONFIG:
            try:
                self._show_config_edit(**detail)
            except Exception:
                # Chrome only: the edit has applied whatever happens here.
                logging.exception("showing an agent's config edit failed")

    def _on_microscope_changed(self):
        """Another microscope, or none: hand it to everything that holds one."""
        microscope = self.microscope
        if microscope is not None:
            # The protocol editors keep the last microscope through a disconnect.
            self.task_widget._on_microscope_connected()
            self.lamella_tab.editor._on_microscope_connected()
        # Both overview modalities hold the microscope for life, so each is handed
        # the new one. It also re-answers whether the tab can be used, which is
        # only knowable now -- whether this system has a fluorescence detector.
        self._refresh_overview_microscope()
        if getattr(self, "quad_overview_page", None) is not None:
            self.quad_overview_page.set_microscope(microscope)
        self._refresh_fm_placeholder()
        # Milling, tiles, spot burn: the bar listens for them itself (FIB-1188).
        self.status_bar.set_microscope(microscope)
        self.agent_host.activity.set_microscope(microscope)
        self._refresh_grids_tab_microscope()
        self._refresh_header()
        self._update_instructions()

    def _create_panel(self):
        """The right-hand panel, and the session's run, responder and executor."""
        self.panel = MicroscopePanel(
            self.session,
            experiment_tab=lambda panel: ExperimentTab(
                self.session, movement_widget=lambda: panel.movement_widget
            ),
            view_controller=self._view_controller,
            overview_acquiring=self._overview_is_acquiring,
        )
        self.panel.setAutoFillBackground(True)
        self._attach_run(self.panel)
        # The Connection tab is a gate in a bar that otherwise means "the instrument
        # needs you here now" -- it is the only tab that can never be a workflow
        # step, and it sits at position 0, the default landing spot, for something
        # done once a session. With the connection dialog on it has somewhere else
        # to be reached from, so it goes (FIB-775).
        #
        # The widget itself stays either way: it owns the connection, and everything
        # in the application follows its signals. Only the tab is conditional.
        if not self.menu_bar.connection_chip_enabled:
            self.panel.tabWidget.insertTab(0, self.system_widget, "Connection")
        # After the Connection tab is in: hiding the Experiment tab while it is
        # the only one leaves the panel showing nothing.
        self.panel.refresh()

    def _create_main_tab(self):
        """Create the main AutoLamella tab."""
        # Create the embedded viewer container
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        self._create_panel()

        # Everything the workflow says without needing an answer arrives here
        self.run_relay.step.connect(self._on_step_update)
        # A grid task records its overview as it finishes, and says so in a status.
        self.run_relay.status.connect(self._refresh_quad_overviews)
        self.run_relay.finished.connect(self._refresh_quad_overviews)
        self.run_relay.notice.connect(self.show_toast)
        notification_service._get_service().toast.connect(self._on_notification_service)
        notification_service._get_service().toast_action.connect(
            self._on_notification_service_action
        )
        self._session_changed.connect(self._on_session_changed)
        self.session.changed.connect(self._hear_session)
        self.panel.experiment_tab.lamella_list.defect_changed.connect(
            self._on_lamella_defect_changed
        )
        self.panel.experiment_tab.lamella_list.lamella_selected.connect(
            self._on_experiment_lamella_selected
        )

        # Layout: quad view (left) | autolamella controls (right) via splitter.
        # The controller's SEM/FIB/FM canvases are the display and drive the control
        # widgets, which are handed it.
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        self.view_controller = MicroscopeViewController(parent=self)
        self.status_bar.follow(self.view_controller)
        splitter.addWidget(self.view_controller.widget)
        splitter.addWidget(self.panel)
        # The fourth cell's overview page: the experiment's overviews of the grid the
        # stage is on. Fed from the same places as the Overview tabs, below.
        self.quad_overview_page = QuadOverviewPage()
        self.view_controller.widget.page_cell.add_page(
            "overview",
            "Overview",
            self.quad_overview_page,
            header=self.quad_overview_page.header,
        )
        self.view_controller.stage_updated.connect(self.quad_overview_page.set_stage)
        self.quad_overview_page.lamella_selected.connect(
            self._on_quad_overview_lamella_selected
        )

        splitter.setSizes([700, 550])
        # set minimum width of right panel to 500
        splitter.widget(1).setMinimumWidth(500)
        layout.addWidget(splitter)
        self.tab_widget.addTab(
            container,
            fibsem_icon("mdi:microscope", color=GRAY_ICON_COLOR),
            "Microscope",
        )
        # The F5/Esc (View) and F2/F6/F9/F11 (Imaging) shortcuts are defined on QActions —
        # one source of truth for the menu item and its keybinding. Adding those actions to
        # the Microscope-tab container makes their WidgetWithChildrenShortcut scope resolve
        # against this tab, so the keys only fire when focus is inside it (not the
        # overview, editor or workflow tabs).
        for action in (
            self.menu_bar.action_toggle_fullscreen,
            self.menu_bar.action_exit_fullscreen,
            *self.menu_bar.imaging_actions,
        ):
            container.addAction(action)

    def create_tabs(self):
        """Create the tabs for the AutoLamella UI."""
        self._create_main_tab()
        self.add_overview_tab()
        self.add_protocol_editor_tab()
        self.add_lamella_editor_tab()
        self.add_grids_tab()
        self.add_workflow_tab()
        self.attention = AttentionRouter(
            prompts=self.proposals,
            tab_widget=self.tab_widget,
            controls=self.workflow_controls,
            status_bar=self.status_bar,
            timeline=self.workflow_tab.timeline,
            sound_enabled=lambda: self._preferences.display.sound_enabled,
            parent=self,
        )
        self.run_view = RunView(
            self.session,
            self.run_relay,
            host=self,
            status_bar=self.status_bar,
            controls=self.workflow_controls,
            border=self.run_border,
            attention=self.attention,
            flush_editors=self.flush_editors,
            lock=self._lock_for_run,
            show_failures=self._show_run_failures,
            after_supervision=(
                self.workflow_tab.lamella_workflow_widget.workflow.refresh_all
            ),
            reset_panel=self._reset_panel_after_run,
        )
        # After the run view's, so a run has locked or released what it owns
        # before the tabs redraw.
        self.run_relay.status.connect(self._on_workflow_status)
        self.run_relay.finished.connect(self._on_workflow_finished)

        self.experiment_header = ExperimentHeader(
            chip_enabled=self.menu_bar.connection_chip_enabled,
            new_experiment=self.menu_bar.action_new_experiment,
            load_experiment=self.menu_bar.action_load_experiment,
            open_experiment_directory=self.menu_bar.action_open_experiment_directory,
            on_new_experiment=self.create_experiment,
            on_load_experiment=self.load_experiment,
            on_connect=self.menu_bar.connect_microscope,
            on_about=self.open_information_dialog,
            toast_manager=self.toast_manager,
        )
        self.tab_widget.setCornerWidget(self.experiment_header)

    def _refresh_header(self) -> None:
        self.experiment_header.refresh(self.experiment, self.microscope)

    def _on_experiment_update(self):
        """Handle experiment update signal and propagate to tabs."""

        if self.experiment is None:
            return

        self.overview_tab.refresh_experiment()
        self.task_widget.set_experiment(self.experiment)
        self.lamella_tab.editor.set_experiment()
        self.grids_tab.set_experiment(self.experiment)
        self.workflow_tab.grid_workflow_widget.set_experiment(self.experiment)
        experiment = self.experiment
        self.quad_overview_page.set_experiment(experiment)
        self.attention.listen(experiment)
        self.agent_host.listen(experiment)
        if experiment is not None and experiment.task_protocol is not None:
            self.workflow_tab.lamella_workflow_widget.set_experiment(experiment)
            self.workflow_tab.lamella_workflow_widget.set_workflow_config(
                experiment.task_protocol.workflow_config
            )
            self.workflow_tab.lamella_workflow_widget.set_options(
                experiment.task_protocol.options
            )

        # Set widget minimum widths (allows resize)
        self.task_widget.setMinimumWidth(500)
        self.lamella_tab.editor.setMinimumWidth(500)
        self.workflow_tab.lamella_workflow_widget.setMinimumWidth(600)

        self._refresh_header()

        # Show run workflow button when experiment is loaded
        self.workflow_controls.run_btn.show()

        # enable all the tabs (except lamella tab, which is managed by _update_lamella_tab_enabled)
        lamella_tab_index = (
            self.tab_widget.indexOf(self.lamella_tab)
            if hasattr(self, "lamella_tab")
            else -1
        )
        # The Overview tab is enabled by what the instrument has, not by an experiment,
        # so loading one must not switch it on for a system where neither modality can be
        # reached -- it would open onto an empty container.
        overview_tab_index = (
            self.tab_widget.indexOf(self.overview_tab)
            if getattr(self, "overview_tab", None) is not None
            and not self.overview_tab.is_available
            else -1
        )
        for i in range(self.tab_widget.count()):
            if i in (lamella_tab_index, overview_tab_index):
                continue
            self.tab_widget.setTabEnabled(i, True)

        self._update_instructions()

        # Rebuild lamella list and wire position events for the new experiment
        self._rebuild_lamella_list()
        self.workflow_tab.selection_changed()  # evaluate after lamella are populated
        self._wire_position_events()

    def _wire_position_events(self):
        """Follow the current experiment's position list, and stop following the last.

        Everything that draws the set of lamellae -- the list rows, the cards, the FM
        overview canvas -- is rebuilt from these, so a display that is not reachable from
        here is a display that goes stale the moment a lamella is added anywhere else.
        That was how a position marked on the FIB/SEM overview never appeared on the FM
        one.
        """
        experiment = self.experiment
        if experiment is self._lamella_list_experiment:
            return

        if self._lamella_list_experiment is not None:
            events = self._lamella_list_experiment.positions.events
            try:
                events.inserted.disconnect(self._rebuild_lamella_list)
                events.removed.disconnect(self._rebuild_lamella_list)
                events.changed.disconnect(self._refresh_overview_positions)
            except Exception as e:
                logging.debug(f"Could not disconnect position events: {e}")

        # Bound methods rather than lambdas, so the disconnects above can find them.
        # psygnal passes each callback only as many arguments as it accepts, so a
        # no-argument handler needs no wrapper -- and a lambda cannot be disconnected by
        # naming the method it calls, which made the disconnect a silent no-op (psygnal
        # does not complain about disconnecting something that was never connected).
        # Nothing broke, because the experiment being disconnected from is on its way out
        # and never emits again, but it did not do what it said.
        if experiment is not None:
            events = experiment.positions.events
            events.inserted.connect(self._rebuild_lamella_list)
            events.removed.connect(self._rebuild_lamella_list)
            # `changed` means a lamella was edited in place rather than added or removed
            # -- `update_lamella_position_ui` emits it after replacing a milling pose.
            # The list rows do not care, but the canvas draws the positions themselves,
            # so it has to be told.
            events.changed.connect(self._refresh_overview_positions)
        self._lamella_list_experiment = experiment

    def add_protocol_editor_tab(self):
        """Add the protocol editor as a separate tab with its own viewer."""
        container = QWidget(parent=self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create the protocol editor widget (viewer is created internally)
        self.task_widget = AutoLamellaProtocolTaskConfigEditor(session=self.session)
        layout.addWidget(self.task_widget)
        self.tab_widget.addTab(
            container,
            fibsem_icon("mdi:file-document-edit", color=GRAY_ICON_COLOR),
            "Protocol",
        )

        # disable the tab by default
        self.tab_widget.setTabEnabled(self.tab_widget.indexOf(container), False)

    def add_lamella_editor_tab(self):
        """The Lamella tab: the card strip, and the selected lamella's protocol
        and history."""
        self.lamella_tab = LamellaTab(
            self.session,
            card_mode=self._preferences.display.lamella_card_mode,
        )
        self.lamella_tab.lamella_selected.connect(self._on_lamella_card_selected)
        cards = self.lamella_tab.cards
        cards.defect_changed.connect(self._on_lamella_defect_changed)
        cards.move_to_requested.connect(self._on_lamella_move_to)
        cards.update_position_requested.connect(self._on_lamella_card_update_position)
        cards.remove_requested.connect(self._on_lamella_remove_requested)
        self.tab_widget.addTab(
            self.lamella_tab,
            fibsem_icon("mdi:layers", color=GRAY_ICON_COLOR),
            "Lamella",
        )
        index = self.tab_widget.indexOf(self.lamella_tab)
        self.tab_widget.setTabEnabled(index, False)
        self.tab_widget.setTabToolTip(index, "Add lamella positions to enable this tab")

    def add_grids_tab(self):
        """The Grids tab, between Lamella and Workflow: the experiment's grid records."""
        # The Positions view makes lamellae through the application widget, the
        # one path every lamella is made by.
        self.grids_tab = GridsTabWidget(
            session=self.session,
            add_lamella=self.panel.experiment_tab.add_new_lamella,
        )
        self.tab_widget.addTab(
            self.grids_tab,
            fibsem_icon("mdi:view-grid-outline", color=GRAY_ICON_COLOR),
            "Grids",
        )
        self.tab_widget.setTabEnabled(self.tab_widget.indexOf(self.grids_tab), False)

    def _refresh_grid_protocol_editor(self) -> None:
        """The task order changed on the Workflow tab: the Protocol tab's grid
        list follows. Guarded: the editor builds lazily on the first connect."""
        grid_protocol = getattr(self.task_widget, "grid_protocol", None)
        if grid_protocol is not None:
            grid_protocol.refresh()

    def _record_inventoried_grids(self) -> None:
        """A record for every grid the inventory lists and the experiment does not.

        The Sample view's inventory updated the loader and nothing else, so the
        Grids tab and the Workflow grid list stayed empty until the Grids tab's
        own refresh was pressed. Reads the stage's cached inventory: no hardware.
        """
        stage = getattr(self.microscope, "sample_handling", None)
        if self.experiment is None or stage is None:
            return
        try:
            added = record_inventoried_grids(
                self.experiment, stage, self.session.record_event
            )
        except Exception as e:  # noqa: BLE001 - the inventory still shows
            logging.warning(f"Could not record the inventoried grids: {e}")
            return
        if added:
            logging.info(f"Recorded {len(added)} grid(s) from the inventory.")

    def _refresh_grids_tab_microscope(self):
        if getattr(self, "grids_tab", None) is None:
            return
        self.grids_tab.set_microscope(self.microscope)
        self.workflow_tab.grid_workflow_widget.set_microscope(self.microscope)
        # An exchange from Microscope -> Sample changes what is loaded; the
        # Grids tab's chips and the run view's rows follow. The Sample view is
        # rebuilt on every connect, so this is wired here, on every connect.
        sample = self.panel.sample_widget
        loader = getattr(sample, "loader_widget", None)
        if loader is not None:
            # A rename there renames the experiment's record, before the
            # inventory sync below could add a second one under the new name.
            loader.set_rename_check(self._grid_rename_refusal)
            loader.grid_renamed.connect(self._on_slot_grid_renamed)
            # Records first: an inventory read there lists grids the experiment
            # has no record of, and the refreshes below draw from the records.
            loader.loader_changed.connect(self._record_inventoried_grids)
            loader.loader_changed.connect(self.grids_tab.refresh)
            loader.loader_changed.connect(
                self.workflow_tab.grid_workflow_widget.refresh
            )
            loader.loader_changed.connect(self._refresh_grid_context)
        self._refresh_grid_context()
        # Renaming a grid in the Sample view has to follow the experiment's records.
        # Calibrating a slot there needs nothing here: the Overview tabs hear of it
        # from `microscope.sample_handling.holder_changed`, whichever window ran the calibration.
        holder_panel = getattr(sample, "holder_widget", None)
        if holder_panel is not None:
            holder_panel.set_rename_check(self._grid_rename_refusal)
            holder_panel.grid_renamed.connect(self._on_slot_grid_renamed)

    def _grid_rename_refusal(self, old: str, new: str) -> str:
        """Why the Sample view may not rename grid *old* to *new*, or "".

        The experiment's record follows a slot's name, so the Grids tab's rules
        apply: a grid that has run keeps its name, and two grids cannot share
        one. A grid a run is on, or has queued, keeps its name until the run
        is done with it (run_edits). With no experiment open there is no record
        to protect.
        """
        from fibsem.applications.autolamella.run_edits import grid_edit_refusal

        experiment = self.experiment
        if experiment is None:
            return ""
        why = grid_edit_refusal(
            self.session.runner, experiment, old, "rename it", renaming=True
        )
        return why or grid_rename_refusal(experiment, old, new)

    def _on_slot_grid_renamed(self, old: str, new: str) -> None:
        """A grid renamed on the Sample view: its record takes the new name, with
        its history, verdict, note and lamellae, instead of the next inventory
        sync adding a second record under the new name."""
        experiment = self.experiment
        if experiment is None or experiment.get_grid_by_name(old) is None:
            return  # no record yet: the sync adds one under the new name
        # The Grids tab's and the App Server's rename. The slot already has the
        # new name, so only the record changes here.
        stage = getattr(self.microscope, "sample_handling", None)
        try:
            rename_and_save_grid(experiment, stage, old, new, self.session.record_event)
        except Exception as e:  # noqa: BLE001 - renamed in memory; saved next time
            logging.warning(f"Could not save the experiment after a rename: {e}")
        self.grids_tab.refresh()
        self.workflow_tab.grid_workflow_widget.refresh()
        self._refresh_grid_context()

    def _grid_context(self):
        """`GridRecord.id -> (name, on the stage)` for the lamella displays,
        from the experiment's records and what the stage already knows. No
        hardware call: the stage answers from its last read."""
        experiment = self.experiment
        if experiment is None or not experiment.grids:
            return None
        loaded = set()
        stage = getattr(self.microscope, "sample_handling", None)
        if stage is not None:
            try:
                loaded = {e.name for e in stage.grid_inventory() if e.loaded}
            except Exception as e:  # noqa: BLE001 - drawn as "not on the stage"
                logging.debug(f"Could not read the grid inventory: {e}")
        return {g.id: (g.name, g.name in loaded) for g in experiment.grids}

    def _refresh_grid_context(self, *_args) -> None:
        """Push the grid context to every lamella display. Called on every
        load, unload and exchange, and on every list rebuild."""
        context = self._grid_context()
        for widget in (
            getattr(getattr(self, "lamella_tab", None), "cards", None),
            self.workflow_tab.lamella_workflow_widget.lamella_list,
            self.panel.experiment_tab.lamella_list,
        ):
            if widget is not None:
                widget.set_grid_context(context)
        # The Overview canvases mark only the lamellae on the stage, so what
        # they draw changes with every load and unload too.
        self._refresh_overview_positions()

    def _refresh_sample_view(self) -> None:
        """Redraw Microscope → Sample from the stage. Looked up each time: the
        Sample view is rebuilt on every connect."""
        sample = self.panel.sample_widget
        if sample is not None:
            sample.refresh()

    def add_workflow_tab(self):
        """The Workflow tab: what to run, the queue and its timeline."""
        self.workflow_tab = WorkflowTab(
            self.session,
            controls=self.workflow_controls,
            flush_edits=self.flush_editors,
            on_config_changed=self._on_workflow_config_changed,
            stop_task=self.session.runner.stop_task,
            after_naming=self._refresh_after_naming,
            say=self._say_selection,
            show_toast=self.show_toast,
        )
        self.run_relay.queue_changed.connect(self.workflow_tab.on_queue_changed)
        lamella = self.workflow_tab.lamella_workflow_widget
        lamella.lamella_edit_requested.connect(self._on_lamella_edit)
        lamella.lamella_remove_requested.connect(self._on_lamella_remove_requested)
        lamella.lamella_defect_changed.connect(self._on_lamella_defect_changed)
        grids = self.workflow_tab.grid_workflow_widget
        grids.protocol_changed.connect(self._refresh_grid_protocol_editor)
        # An inventory, a rename, a manual load on the Grids tab: the run view's
        # rows and chips follow. Built after the Grids tab, so the signal exists.
        self.grids_tab.experiment_changed.connect(grids.refresh)
        # And the Sample view: a load or unload from a card changes what is on
        # the stage, and that view draws from the stage without polling it.
        self.grids_tab.experiment_changed.connect(self._refresh_sample_view)
        # And the lamella lists: their grid chips say whether each lamella's
        # grid is on the stage.
        self.grids_tab.experiment_changed.connect(self._refresh_grid_context)

        self.tab_widget.addTab(
            self.workflow_tab,
            fibsem_icon("mdi:play-circle-outline", color=GRAY_ICON_COLOR),
            "Workflow",
        )
        # disable the workflow tab by default
        self.tab_widget.setTabEnabled(self.tab_widget.indexOf(self.workflow_tab), False)

        # Track which experiment's position events we're connected to
        self._lamella_list_experiment = None

        # Guard against bidirectional selection sync loops
        self._syncing_selection = False

        # Connect protocol editor → workflow tab (deferred here since task_widget is created first)
        self.task_widget.workflow_config_changed.connect(lamella.set_workflow_config)
        # A task added or removed there is added to or removed from every lamella:
        # the lamella editor's task list is rebuilt, so it never edits a task its
        # lamella no longer has (FIB-1109).
        self.task_widget.workflow_config_changed.connect(
            lambda _: self.lamella_tab.editor._refresh_experiment_positions()
        )
        # And the Grid page → Workflow → Grids: a task added on the Protocol tab
        # gets its row in the run view without an inventory or a reload.
        self.task_widget.grid_protocol_changed.connect(grids.refresh)

    def _on_lamella_card_selected(self, lamella: Lamella | None):
        """The Lamella tab's card: the overviews and the Microscope tab's list follow."""
        self.overview_tab.set_selected(lamella)
        if lamella is not None:
            if not self._syncing_selection:
                self._syncing_selection = True
                try:
                    self.panel.experiment_tab.lamella_list.select(lamella.name)
                finally:
                    self._syncing_selection = False

    def _on_experiment_lamella_selected(self, lamella):
        """Sync card container and overviews when experiment-tab list selection changes."""
        self.overview_tab.set_selected(lamella)
        self.quad_overview_page.set_selected(lamella)
        if getattr(self, "_syncing_selection", False) or lamella is None:
            return
        if not hasattr(self, "lamella_tab"):
            return
        self._syncing_selection = True
        try:
            self.lamella_tab.select(lamella)
        finally:
            self._syncing_selection = False

    def _on_workflow_status(self, event: "WorkflowStatusEvent"):
        """A status update from the run relay: the tabs redraw for its report.
        The run line, chip and border are the run view's."""
        if event.report is not None:
            self._apply_status_report(event.report)

    def _apply_status_report(self, report: "WorkflowStatusUpdate") -> None:
        """Redraw the tabs for one task-lifecycle report: timeline, editor lock,
        list refreshes."""
        # No build-once guard: update_from_status reconciles rows against the
        # snapshot by WorkItem id, so first paint and every later change go
        # through the same path.
        self.workflow_tab.timeline.update_from_status(report)

        task_name = report.task_name
        lamella_name = report.item_name
        status = report.status

        # Lock editor when the active lamella/task is being processed
        if status is TaskStatus.InProgress:
            self.lamella_tab.editor.set_active_lamella_name(lamella_name, task_name)
        else:
            self.lamella_tab.editor.set_active_lamella_name(None)
        self.panel.experiment_tab.refresh_run_edits()

        # Refresh only the affected lamella if we can identify it
        lamella = None
        experiment = self.experiment
        if experiment is not None and lamella_name is not None:
            lamella = experiment.get_lamella_by_name(lamella_name)

        # The name list is refreshed from here rather than subscribing per row.
        # `LamellaRow` used to connect to `task_state.events.name`/`.status`, which
        # the workflow writes from its worker thread; the marshalled refresh then
        # arrived after the row had been replaced and its widgets destroyed
        # (`RuntimeError: wrapped C/C++ object of type QLabel has been deleted`).
        # It joins the other two on the same named-lamella path, so it costs one row
        # -- unlike them, though, nothing else was refreshing this list mid-workflow.
        grid = (
            experiment.get_grid_by_name(lamella_name)
            if experiment is not None and lamella_name is not None
            else None
        )
        if grid is not None:
            # A grid run's report: the grid's card and its row follow it. The
            # lamella lists have nothing to redraw.
            card = self.grids_tab.cards.card_for(grid)
            if card is not None:
                card.refresh()
            self.workflow_tab.grid_workflow_widget.refresh_grid(grid)
            self.grids_tab.results_widget.refresh()
            # The run's exchanges change what is loaded; the Sample view
            # draws from the stage and never polls it, so it is told here.
            sample = self.panel.sample_widget
            if sample is not None:
                sample.refresh()
        elif lamella is not None:
            self.workflow_tab.lamella_workflow_widget.lamella_list.refresh_lamella(
                lamella
            )
            self.panel.experiment_tab.lamella_list.refresh_lamella(lamella)
            # The Overview pages' lists carry the same status column; without
            # this they sat stale for the whole run (FIB-995).
            for lamella_list in self.overview_tab.lamella_lists():
                lamella_list.refresh_lamella(lamella)
        else:
            self.workflow_tab.lamella_workflow_widget.lamella_list.refresh_all()
            self.panel.experiment_tab.lamella_list.refresh_all()
            for lamella_list in self.overview_tab.lamella_lists():
                lamella_list.refresh_all()
        if grid is None:
            self.lamella_tab.task_reported(lamella, status)
        self._on_lamella_card_selected(self.lamella_tab.selected)

    def _rebuild_lamella_list(self):
        """Clear and repopulate the lamella list and card container from the current experiment."""
        if not hasattr(self, "workflow_tab"):
            return
        experiment = self.experiment
        # The ticks and the selected card are the operator's, and this runs on every
        # insert or removal: rebuild around them rather than through them (FIB-966).
        # The overview canvases are further displays of the same set, so they are
        # rebuilt here rather than from subscriptions of their own -- one handler for
        # "the lamellae changed" means the displays cannot end up disagreeing about what
        # the experiment holds. Before the `experiment is None` return, because an
        # experiment closing has to clear them as much as an experiment opening has to
        # fill them.
        self._refresh_overview_positions()
        if experiment is None:
            self.workflow_tab.lamella_workflow_widget.lamella_list.clear()
            self._on_lamella_card_selected(self.lamella_tab.set_lamellae([]))
            return
        self._refresh_grid_context()
        self.workflow_tab.lamella_workflow_widget.lamella_list.set_lamellae(
            list(experiment.positions)
        )
        self._on_lamella_card_selected(
            self.lamella_tab.set_lamellae(list(experiment.positions))
        )
        self.workflow_tab.selection_changed()
        self._update_lamella_tab_enabled()

    def _update_lamella_tab_enabled(self):
        """Enable or disable the Lamella tab based on whether positions exist."""
        if not hasattr(self, "lamella_tab"):
            return
        index = self.tab_widget.indexOf(self.lamella_tab)
        if index < 0:
            return
        experiment = self.experiment
        has_positions = experiment is not None and len(experiment.positions) > 0
        self.tab_widget.setTabEnabled(index, has_positions)
        self.tab_widget.setTabToolTip(
            index, "" if has_positions else "Add lamella positions to enable this tab"
        )

    def _on_lamella_move_to(self, lamella: "Lamella"):
        """Move the stage to the given lamella's milling position."""
        if self.experiment is None:
            return
        self.panel.experiment_tab.lamella_list.select(lamella.name)
        self.panel.experiment_tab.move_to_lamella_position()

    def _on_lamella_card_update_position(self, lamella: "Lamella"):
        """Update the stage position of the given lamella to the current stage position."""
        self.panel.experiment_tab.lamella_list.select(lamella.name)
        self.panel.experiment_tab.update_lamella_position_ui()

    def _on_lamella_edit(self, lamella: "Lamella"):
        """Switch to the Lamella tab and select the given lamella in the protocol editor."""
        if self.experiment is None:
            return
        self.panel.experiment_tab.lamella_list.select(lamella.name)

        # Select the lamella in the protocol editor
        self.lamella_tab.editor.select_lamella(lamella.name)

        # Switch to the Lamella tab
        for i in range(self.tab_widget.count()):
            if self.tab_widget.tabText(i) == "Lamella":
                self.tab_widget.setCurrentIndex(i)
                break

    def _on_lamella_defect_changed(self, lamella: "Lamella"):
        """Persist defect state change to disk and sync all widgets."""
        if self.experiment is None:
            return
        self.experiment.sign_verdict(lamella)
        self.experiment.save()
        # Sync defect icon across all widgets
        self.panel.experiment_tab.lamella_list.refresh_all()
        self.workflow_tab.lamella_workflow_widget.lamella_list.refresh_lamella(lamella)
        self.lamella_tab.cards.refresh_lamella(lamella)

    def _on_lamella_remove_requested(self, lamella: "Lamella"):
        """Remove the given lamella from the experiment after the list widget has already removed its row."""
        from fibsem.applications.autolamella.run_edits import lamella_edit_refusal

        if self.experiment is None:
            return
        why = lamella_edit_refusal(
            self.session.runner, self.experiment, lamella, "remove it"
        )
        if why:
            self._rebuild_lamella_list()  # its row has gone already
            notification_service.show_toast(why, "warning")
            return
        if not remove_lamella(self.experiment, lamella):
            return
        self.panel.experiment_tab.update_items(latest=True)
        self.panel.refresh()

    def _on_step_update(self, label: str) -> None:
        """Handle per-step update from the workflow worker thread."""
        self.workflow_tab.timeline.update_step(label)

    def _on_workflow_finished(self, cancelled: bool = False):
        """The tabs' half of a run ending; the run view has said how it ended."""
        # The next run's items carry new ids, so the timeline rebuilds itself on
        # its first status update — nothing to reset here.
        # Resolve any outer row left in ACTIVE state (e.g. if workflow was cancelled)
        self.workflow_tab.timeline.finish_current_step(failed=cancelled)
        self.workflow_tab.timeline.clear_steps()
        self.lamella_tab.editor.set_active_lamella_name(None)
        self.workflow_tab.grid_workflow_widget.refresh()
        sample = self.panel.sample_widget
        if sample is not None:
            sample.refresh()
        self.workflow_tab.lamella_workflow_widget.lamella_list.refresh_all()
        self.lamella_tab.cards.refresh_all()

    def _lock_for_run(self, running: bool) -> None:
        """Take from the tabs what a run owns while it goes, or give it back."""
        self.overview_tab.locks.set_run_allows(not running)
        # A run owns the loader: no manual exchange from the Grids tab meanwhile.
        self.grids_tab.set_controls_enabled(not running)
        self.workflow_tab.grid_workflow_widget.set_controls_enabled(not running)
        sample = self.panel.sample_widget
        if sample is not None:
            sample.set_controls_enabled(not running)
        self.workflow_tab.set_running(running)

    def _reset_panel_after_run(self) -> None:
        self.panel.reset_after_run(str(self.experiment.path))
        self.panel.refresh()
        self.panel.set_current_workflow_message(msg=None, show=False)

    def _show_run_failures(self) -> None:
        """The failure line's Show: the Workflow tab, with the run's
        first failed row selected and in view. The timeline is where the run is
        laid out, failures marked, and where they are queued again."""
        self.tab_widget.setCurrentWidget(self.workflow_tab)
        self.workflow_tab.timeline.show_first_failed()

    def add_overview_tab(self):
        """Reserve the Overview tab: both modalities, one tab.

        The FIB/SEM and fluorescence overviews were two tabs here, and are now two pages
        of one -- `AutoLamellaOverviewContainerTab` holds both host tabs and a chip strip
        that chooses between them (FIB-780). Both host tabs stay built and stay
        subscribed while the other is showing, so everything the window tells them is
        still told to both.

        The tab is created on every system and never hidden, so the tab bar keeps the
        same shape whatever the instrument turned out to be. When neither modality has
        anything to drive it is greyed out with a tooltip saying why -- see
        :meth:`_on_overview_availability`. The widgets inside are built and destroyed to
        match, so a dead tab costs nothing but the container.

        The tab is created empty. Both overview widgets require a microscope at
        construction -- every scale on their canvases comes from the instrument -- and at
        this point there may be no microscope at all. The container fills itself in on
        connection and says so through `availability_changed`.
        """
        self.overview_tab = AutoLamellaOverviewContainerTab(
            self.session,
            self.panel.experiment_tab.add_new_lamella,
        )
        self.overview_tab.availability_changed.connect(self._on_overview_availability)
        self.overview_tab.lamella_selected.connect(self._on_overview_lamella_selected)
        # A finished run has saved its overview by the time it says so.
        self.overview_tab.acquiring_changed.connect(self._refresh_quad_overviews)

        self.tab_widget.insertTab(
            2,
            self.overview_tab,
            fibsem_icon("mdi:map-search-outline", color=GRAY_ICON_COLOR),
            "Overview",
        )
        self.tab_widget.setTabEnabled(self.tab_widget.indexOf(self.overview_tab), False)
        # Emits availability either way, which is what puts the reason on the tab.
        self._refresh_overview_microscope()

    def _on_overview_availability(self, available: bool) -> None:
        """Enable the tab when either modality has something to drive, and say why not.

        The one thing about the tab that is not the tab's own business: it has no
        business reaching out to the tab bar it happens to sit in.

        The tab is never hidden. A greyed tab that explains itself is easier to live
        with than one that appears and vanishes depending on what the microscope turned
        out to be -- but that is only true *because* of the tooltip, so the two are set
        together here rather than in separate passes.

        Qt does show a tooltip on a disabled tab: the `QTabBar` stays enabled and only
        the tab within it is disabled, so the hover still lands. Worth stating because
        disabled *widgets* do swallow tooltips, which makes this look doubtful.
        """
        index = self.tab_widget.indexOf(self.overview_tab)
        self.tab_widget.setTabEnabled(index, available)
        _, reason = self.overview_tab.unavailable_summary()
        self.tab_widget.setTabToolTip(index, "" if available else reason)

    def _refresh_overview_microscope(self):
        """Build or drop both overview widgets to match the instrument.

        The widgets are built and destroyed rather than left behind hidden. Each
        subscribes to the microscope's stage signal for its lifetime, so one kept around
        would go on doing work on every poll for a page nobody can reach -- and would
        still be holding a psygnal reference to tear down later.

        Whether the tab can be *used* is a separate question, answered by
        `availability_changed` coming back through :meth:`_on_overview_availability`.
        This method does not touch the tab bar.
        """
        if getattr(self, "overview_tab", None) is None:
            return
        self.overview_tab.refresh_microscope()

    def _on_quad_overview_lamella_selected(self, lamella):
        """A lamella clicked on the Microscope tab's overview page: select it in the
        lists and the Overview tabs, as a click on either Overview tab does."""
        self.overview_tab.set_selected(lamella)
        self._on_overview_lamella_selected(lamella)

    def _on_overview_lamella_selected(self, lamella):
        """A lamella selected on either Overview page: select it in the lists."""
        if getattr(self, "_syncing_selection", False) or lamella is None:
            return
        self._syncing_selection = True
        try:
            self.panel.experiment_tab.lamella_list.select(lamella.name)
            if hasattr(self, "lamella_tab"):
                self.lamella_tab.select(lamella)
        finally:
            self._syncing_selection = False

    def _refresh_quad_overviews(self, *_args) -> None:
        """Have the quad view's overview page look for new overviews. It redraws
        only if it finds one, and not at all while hidden."""
        page = getattr(self, "quad_overview_page", None)
        if page is not None:
            page.refresh()

    def _refresh_overview_positions(self):
        """Re-mark every canvas that shows the lamellae (FIB-709). A method on the
        window rather than each tab connected to the experiment: a tab is rebuilt
        when the microscope changes, and a subscription holding its bound method
        would go stale."""
        self.overview_tab.refresh_positions()
        # The Microscope tab's stage map and overview page mark the same lamellae,
        # from the same place.
        experiment = self.experiment
        page = getattr(self, "quad_overview_page", None)
        if page is not None:
            if page.experiment is not experiment:
                page.set_experiment(experiment)
            else:
                page.refresh_positions()
        controller = getattr(self, "view_controller", None)
        if controller is not None:
            controller.set_map_positions(
                [lamella.stage_position for lamella in experiment.positions]
                if experiment is not None
                else []
            )

    @property
    def protocol(self) -> Optional[AutoLamellaTaskProtocol]:
        with self._protocol_lock:
            return (
                self.experiment.task_protocol if self.experiment is not None else None
            )

    def _overview_is_acquiring(self) -> bool:
        """Whether either overview modality is mid-tileset.

        Asked of the Overview tab rather than of a widget held here: the tab owns both
        modalities and rebuilds its widgets on every reconnection, so anything holding
        one of them directly would be answering for a widget that had been replaced.
        """
        tab = getattr(self, "overview_tab", None)
        return tab is not None and tab.is_acquiring

    def create_experiment(self) -> None:
        """Create a new experiment using the experiment creation dialog."""
        if refused_during_run(self.session.runner, "change experiments"):
            return
        if self.microscope is None:
            notification_service.show_toast(
                "Please connect to microscope first.", "warning"
            )
            return

        # Open the experiment creation dialog
        experiment = create_experiment_dialog(parent=self)  # type: ignore

        if experiment is None:
            notification_service.show_toast("Experiment creation cancelled.", "info")
            return

        self.session.adopt(experiment)

    def load_experiment(self) -> None:
        """Load an existing experiment using the experiment loading dialog."""
        if refused_during_run(self.session.runner, "change experiments"):
            return
        if self.microscope is None:
            notification_service.show_toast(
                "Please connect to microscope first.", "warning"
            )
            return

        # Open the experiment loading dialog
        experiment = load_experiment_dialog(parent=self)  # type: ignore

        if experiment is None:
            notification_service.show_toast("Experiment loading cancelled.", "info")
            return

        self.session.adopt(experiment)

    def quickstart(self, load_experiment: bool = False) -> None:
        """Connect to the microscope without waiting to be clicked (``--quickstart``).

        The developer shortcut past the two clicks that start every session: the same
        calls the Connection tab and the load dialog make, with the default microscope
        configuration and the most recent experiment assumed.

        Every failure here is reported and swallowed rather than raised. This runs
        unattended on the way up, and an unreachable microscope or an experiment that
        has moved should still leave the ordinary window -- the one where the user can
        pick a different configuration -- rather than a half-started application.
        """
        if self.system_widget.microscope is None:
            try:
                self.system_widget.connect_to_microscope()
            except Exception as e:
                logging.warning(f"Quickstart: unable to connect to the microscope: {e}")
                notification_service.show_toast(
                    f"Quickstart: could not connect to the microscope: {e}", "error"
                )
                return

        # connect_to_microscope reports its own failures and returns without a
        # microscope (an unselectable configuration, say). Nothing below works
        # without one, so stop here rather than reporting a second failure.
        if self.system_widget.microscope is None:
            return

        if load_experiment:
            self.quickload_experiment()

    def quickload_experiment(self) -> None:
        """Reopen the most recent experiment, skipping the load dialog (``--quickload``).

        Requires a connected microscope, as the dialog path does: the tabs that adopt
        an experiment are built at connection time.
        """
        experiment_path = fibsem_cfg.get_last_experiment_file()

        if experiment_path is None:
            msg = "Quickload: no recent experiment to load."
            logging.warning(msg)
            notification_service.show_toast(msg, "warning")
            return

        experiment, why = load_for_adoption(experiment_path)
        if experiment is None:
            msg = f"Quickload: {why}; not loaded."
            logging.warning(msg)
            notification_service.show_toast(msg, "warning")
            return

        self.session.adopt(experiment)
        logging.info(
            f"Quickload: loaded experiment {experiment.name} from {experiment_path}"
        )

    # TODO: create a dialog to get the user to connect to microscope and create load experiment before continuing
    # then remove the system widget entirely... you will always be connected once you start
    def app_context(self, event_buffer):
        from fibsem.applications.autolamella.server.context import (
            AutoLamellaAppContext,
        )

        return AutoLamellaAppContext(self.session, event_buffer=event_buffer)

    #### REPORT GENERATION
    def action_generate_report(self) -> None:
        """Generate a pdf report of the experiment."""
        if self.experiment is None:
            return

        if not REPORTING_AVAILABLE:
            notification_service.show_toast(
                "Reporting tools are not available. "
                'Install the reporting extra: pip install "fibsem[reporting]"',
                "warning",
            )
            return

        generate_report_dialog(self.experiment, parent=self)
        return

    def action_generate_overview_plot(self) -> None:
        """Generate an plot with the lamella position on an overview image."""
        if self.experiment is None:
            return

        dialog = create_overview_image_widget(experiment=self.experiment, parent=self)
        dialog.exec_()

        return

    def open_information_dialog(self) -> None:
        # No connection guard: checking which version you are running is exactly
        # what you want to do *before* connecting. The dialog drops the
        # microscope section when there is nothing to report.
        fui.open_information_dialog(self.microscope, self, application="AutoLamella")

    def _open_experiment_directory(self) -> None:
        """Open the experiment directory in the system file explorer."""
        if self.experiment is None or self.experiment.path is None:
            notification_service.show_toast(
                "Please load an experiment first... [No Experiment Loaded]", "warning"
            )
            return

        experiment_path = os.fspath(self.experiment.path)
        if not os.path.isdir(experiment_path):
            notification_service.show_toast(
                f"Experiment directory not found: {experiment_path}", "error"
            )
            return

        if not fui.open_path_in_file_explorer(experiment_path):
            notification_service.show_toast(
                "Failed to open experiment directory.", "error"
            )

    def export_targeting_ml_data(self) -> None:
        """Export the experiment's lamella targets as targeting ML training data.

        One sample per lamella: the final FIB reference image from its Select Milling
        Position task, labelled with the operator's point of interest. Writes to
        <experiment>/targeting-export. See
        fibsem.applications.autolamella.tools.ml_export for the layout.
        """
        from fibsem.applications.autolamella.tools import ml_export

        if self.experiment is None:
            notification_service.show_toast(
                "Please load an experiment first... [No Experiment Loaded]", "warning"
            )
            return

        # no directory prompt: the destination is a subfolder of the experiment that
        # does not exist yet, which an existing-directory dialog cannot select. The
        # folder is opened afterwards so the location is still discoverable.
        output_path = ml_export.default_output_path(self.experiment)

        try:
            summary = ml_export.export_experiment(self.experiment, output_path)
        except Exception as e:
            logging.error(f"Failed to export targeting ML data: {e}", exc_info=True)
            notification_service.show_toast(f"Export failed: {e}", "error")
            return

        if summary.n_samples == 0:
            reason = summary.skipped[0] if summary.skipped else "nothing to export"
            notification_service.show_toast(f"Nothing exported: {reason}", "warning")
            return

        message = f"Exported {summary.n_samples} lamella target(s)."
        if summary.skipped:
            message += f" Skipped {len(summary.skipped)}."
        notification_service.show_toast(message, "success")
        fui.open_path_in_file_explorer(output_path)

    #### FLUORESCENCE IMAGE VIEWER

    def _fm_image_viewer_start_directory(self) -> str:
        """Where the viewer's Load dialog opens: the open experiment, else the folder
        of the most recent one, else the log directory. The viewer reads files, so it
        needs no experiment (FIB-942); this only picks a sensible first folder."""
        if self.experiment is not None and self.experiment.path:
            return str(self.experiment.path)
        recent = fibsem_cfg.load_user_preferences().experiment.recent_experiments
        for path in recent:
            parent = os.path.dirname(path)
            if os.path.isdir(parent):
                return parent
        return fibsem_cfg.LOG_PATH

    def _open_fm_image_viewer(self):
        """Open the FM Image Viewer as a standalone window."""
        # Parented to None so it gets its own taskbar entry and native minimise, like the
        # coincidence viewer. That means nothing else owns it, so the reference here is
        # what keeps it alive — drop it and Python collects the window mid-session.
        self._fm_image_viewer_window = FMImageViewerWidget(
            start_directory=self._fm_image_viewer_start_directory()
        )
        self._fm_image_viewer_window.resize(1180, 700)
        self._fm_image_viewer_window.show()
        self._fm_image_viewer_window.activateWindow()

    def _open_coincidence_milling_viewer(self):
        """Open the Coincidence Milling Viewer as a standalone dialog."""
        if self.microscope is None or self.experiment is None:
            notification_service.show_toast(
                "Please connect a microscope and load an experiment first.", "warning"
            )
            return
        if self.microscope.fm is None:
            notification_service.show_toast(
                "Coincidence milling requires a fluorescence microscope.", "warning"
            )
            return
        from fibsem.applications.autolamella.ui.coincidence.viewer import (
            open_coincidence_viewer_window,
        )

        # seed the viewer's FM tab from the live main-UI FM configuration
        fm_config = None
        if self.panel.fm_control_widget is not None:
            try:
                fm_config = (
                    self.panel.fm_control_widget._build_fluorescence_configuration()
                )
            except Exception as e:
                logging.warning(f"Could not read current FM configuration: {e}")

        self._coincidence_viewer_window = open_coincidence_viewer_window(
            microscope=self.microscope,
            experiment=self.experiment,
            parent=self,
            fm_config=fm_config,
        )
        # This window's border switch and agent purple, not its own.
        self.run_border.mirror_to(self._coincidence_viewer_window.run_border)

    def flush_editors(self) -> None:
        """Record and write the edits the Protocol and Lamella editors still
        hold (FIB-683): before anything else writes the experiment."""
        self.task_widget.flush_pending_save()
        self.lamella_tab.editor.flush_pending_save()

    def _show_config_edit(
        self, level: str, item_name: str, task_name: str, result: dict
    ) -> None:
        """An agent's config edit applied: toast it and rebuild the editor
        showing it.

        The rebuild is what stops a stale open form writing old values back on
        the operator's next edit; the toast is what stops the form appearing to
        change by itself. show(), not show_toast(): agent changes are workflow
        events -- they persist in the notification bell, not just flash for 3 s
        while the operator is looking at the canvas.
        """
        if level == "apply_protocol":
            names = ", ".join(result.get("task_names") or [])
            notification_service.show(
                f"Agent applied protocol to {item_name} — {names}", "info"
            )
            self.lamella_tab.editor.refresh_if_showing(item_name)
            return
        changes = result.get("changes") or []
        if not changes:
            return
        first = changes[0]
        leaf = first["path"].rsplit(".", 1)[-1]
        summary = f"{leaf}: {first['old']!r} → {first['new']!r}"
        if len(changes) > 1:
            summary += f" (+{len(changes) - 1} more)"
        if level == "item":
            target = f"{task_name} for {item_name}"
        elif level == "item_fields":
            target = item_name
        else:
            target = task_name
        notification_service.show(f"Agent edited {target} — {summary}", "info")
        if level in ("item", "item_fields"):
            self.lamella_tab.editor.refresh_if_showing(item_name)
        else:
            self.task_widget.refresh_if_showing_task(task_name)

    def _on_workflow_config_changed(
        self, wcfg: AutoLamellaWorkflowConfig, *changes: dict
    ):
        """Save the Workflow tab's edit with the protocol and record each
        change, as the App Server's edits are."""
        if self.experiment is None or self.experiment.task_protocol is None:
            return
        self.experiment.task_protocol.workflow_config = wcfg
        for change in changes:
            save_workflow_change(self.experiment, change, self.session.record_event)

        self.panel.refresh()

    #### PROTOCOL
    def load_protocol(self):
        """Load a protocol into the current experiment using the protocol loading dialog."""
        if refused_during_run(self.session.runner, "change experiments"):
            return
        if self.microscope is None:
            notification_service.show_toast(
                "Please connect to microscope first.", "warning"
            )
            return

        if self.experiment is None:
            notification_service.show_toast(
                "Please load an experiment first.", "warning"
            )
            return

        # Open the protocol loading dialog
        protocol = load_task_protocol_dialog(experiment=self.experiment, parent=self)

        if protocol is None:
            notification_service.show_toast("Protocol loading cancelled.", "info")
            return

        adopt_protocol(self.session, protocol)
        notification_service.show_toast(
            f"Protocol '{protocol.name}' loaded successfully with {len(protocol.task_config)} tasks.",
            "info",
        )

    def export_protocol_ui(self):
        """Export the current protocol to file."""

        if self.experiment is None or self.experiment.task_protocol is None:
            notification_service.show_toast("No protocol loaded.", "warning")
            return

        protocol_path = fui.open_save_file_dialog(
            msg="Select a protocol file",
            path=str(cfg.TASK_PROTOCOL_PATH),
            _filter="*.yaml",
            parent=self,
        )

        if protocol_path == "":
            notification_service.show_toast("No path selected", "info")
            return

        self.experiment.task_protocol.save(protocol_path)
        notification_service.show_toast(
            f"Saved Protocol to {os.path.basename(protocol_path)}", "info"
        )

    def _on_notification_service(
        self, message: str, notification_type: str, temporary: bool, always: bool
    ) -> None:
        self.show_toast(message, notification_type, temporary=temporary, always=always)

    def _on_notification_service_action(
        self,
        message: str,
        notification_type: str,
        temporary: bool,
        always: bool,
        action_text: str,
        callback,
    ) -> None:
        self.show_toast(
            message,
            notification_type,
            temporary=temporary,
            action=(action_text, callback),
            always=always,
        )

    def _closing(self) -> None:
        # TypeError: a second close finds it already disconnected -- close() sends
        # the close event again even when the window is hidden.
        try:
            notification_service._get_service().toast.disconnect(
                self._on_notification_service
            )
        except (RuntimeError, TypeError):
            pass
