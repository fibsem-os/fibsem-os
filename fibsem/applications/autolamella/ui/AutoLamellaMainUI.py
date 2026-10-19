from __future__ import annotations

import logging
import sys
from dataclasses import replace
from typing import List, Optional, Tuple

try:
    sys.modules.pop("PySide6.QtCore")
except Exception:
    pass

from datetime import datetime

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import fibsem
import fibsem.config as fibsem_cfg
from fibsem.acting import OPERATOR
from fibsem.applications.autolamella.poses import remove_lamella
from fibsem.applications.autolamella.structures import Lamella
from fibsem.applications.autolamella.task_outputs import PreviewSources
from fibsem.applications.autolamella.ui.attention import AttentionRouter
from fibsem.applications.autolamella.ui.autolamella_task_config_editor import (
    AutoLamellaProtocolTaskConfigEditor,
)
from fibsem.applications.autolamella.ui.AutoLamellaUI import INSTRUCTIONS, AutoLamellaUI
from fibsem.applications.autolamella.ui.experiment_header import ExperimentHeader
from fibsem.applications.autolamella.ui.grids_tab_widget import GridsTabWidget
from fibsem.applications.autolamella.ui.lamella_tab import LamellaTab
from fibsem.applications.autolamella.ui.main_menu_bar import MainMenuBar
from fibsem.applications.autolamella.ui.overview_container_tab import (
    AutoLamellaOverviewContainerTab,
)
from fibsem.applications.autolamella.ui.quad_overview_page import QuadOverviewPage
from fibsem.applications.autolamella.ui.workflow_controls import (
    AUTOMATED,
    SUPERVISED,
    WorkflowControls,
)
from fibsem.applications.autolamella.ui.workflow_tab import WorkflowTab
from fibsem.applications.autolamella.workflows.tasks.tasks import get_task_supervision
from fibsem.ui import notification_service
from fibsem.ui.agent.chip_host import AgentChipHost
from fibsem.ui.icon import fibsem_icon
from fibsem.ui.stylesheets import (
    GRAY_ICON_COLOR,
    NAPARI_STYLE,
    STATUS_BAR_STYLESHEET,
    border_stylesheet,
)
from fibsem.ui.widgets.canvas.quad_view import MicroscopeViewController
from fibsem.ui.widgets.notifications import ToastManager
from fibsem.ui.widgets.run_border import RunBorder, run_border_state
from fibsem.ui.widgets.status_bar import FibsemStatusBar
from fibsem.versioning import get_version_string
from fibsem.workflows.attention import Attention
from fibsem.workflows.grid.naming import grid_rename_refusal, rename_grid
from fibsem.workflows.runner import GRIDS, RunRequest
from fibsem.workflows.session import EXPERIMENT, MICROSCOPE
from fibsem.workflows.status import (
    TaskStatus,
    WorkflowStatusEvent,
    WorkflowStatusUpdate,
)


class AutoLamellaSingleWindowUI(QMainWindow):
    """Main window for AutoLamella UI.

    No napari viewer of its own any more: every tab here renders on the matplotlib
    canvas. napari survives in this application only in labelling and segmentation,
    which open windows of their own (FIB-407).
    """

    # The session's `changed`, carried to the GUI thread.
    _session_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"AutoLamella v{get_version_string()} ")
        self.resize(1600, 1000)

        # Apply napari-style dark theme. Border state rules live here (on the parent)
        # so that setProperty + unpolish/polish on the RunBorder re-evaluates them.
        self.setStyleSheet(NAPARI_STYLE + border_stylesheet("workflow_border_frame"))

        self.tab_widget = QTabWidget()
        self.run_border = RunBorder("workflow_border_frame")
        self.run_border.set_content(self.tab_widget)
        self.setCentralWidget(self.run_border)

        self.autolamella_ui: AutoLamellaUI

        # Toast notification manager
        self.toast_manager = ToastManager(self)

        # Load user preferences
        self._preferences = fibsem_cfg.load_user_preferences()

        self.run_border.set_on(self._preferences.display.border_enabled)

        # create menus, status bar, and tabs
        self.menu_bar = MainMenuBar(
            self,
            ui=lambda: self.autolamella_ui,
            view_controller=lambda: getattr(self, "view_controller", None),
            run_border=self.run_border,
            preferences=self._preferences,
            show_toast=self.show_toast,
            on_app_server=self._on_app_server_dialog,
            on_grid_report=self._on_generate_grid_report,
            on_coincidence_viewer=self._open_coincidence_milling_viewer,
            parent=self,
        )
        self.menu_bar.preferences_saved.connect(self._on_preferences_saved)
        self._create_status_bar()
        self.create_tabs()
        self._apply_preferences()
        self._update_instructions()

        # Connect tab change to status bar update
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

    def _on_preferences_saved(self, preferences) -> None:
        self._preferences = preferences
        self._apply_preferences()

    def _apply_preferences(self):
        """Apply current preferences to UI state."""
        # The embedded App Server follows its saved switch (set in Tools → App
        # Server) against the connected microscope.
        if getattr(self, "autolamella_ui", None) is not None:
            self.autolamella_ui.sync_app_server_with_preference()
        d = self._preferences.display
        self.run_border.set_on(d.border_enabled)
        # The lamella strip's density. Guarded because _apply_preferences also runs
        # before the Lamella tab is built.
        if hasattr(self, "lamella_tab"):
            self.lamella_tab.cards.set_mode(d.lamella_card_mode)
        self.menu_bar.apply_preferences(self._preferences)
        # getattr because this also runs from the preferences dialog, and the tab it
        # reaches into is built by AutoLamellaUI rather than here.
        system_widget = getattr(self.autolamella_ui, "system_widget", None)
        if system_widget is not None:
            system_widget.refresh_first_run_offer(self._preferences)

    def show_toast(
        self,
        message: str,
        notification_type: str = "info",
        duration: int = 5000,
        temporary: bool = False,
        action=None,
    ):
        """Show a toast notification, with an optional (text, callback) link.

        Unconditional: there is no longer a preference for turning toasts off, so the
        branch that used to route a suppressed message to the notification bell has
        gone with it. Nothing is lost -- `ToastManager.show_toast` records every
        non-temporary message in the bell itself.
        """
        self.toast_manager.show_toast(
            message, notification_type, duration, temporary=temporary, action=action
        )

    def _on_generate_grid_report(self):
        """Tools → Reporting → Generate Grid Screening Report.

        The Grids tab does the writing and says how it went on its strip, so the
        window shows that tab first; a refusal that needs no tab is a toast.
        """
        tab = getattr(self, "grids_tab", None)
        experiment = getattr(self.autolamella_ui, "experiment", None)
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

    def _open_coincidence_milling_viewer(self):
        """Open the Coincidence Milling Viewer dialog."""
        if self.autolamella_ui is not None:
            self.autolamella_ui._open_coincidence_milling_viewer()
            viewer = getattr(self.autolamella_ui, "_coincidence_viewer_window", None)
            if viewer is not None:
                # This window's border switch and agent purple, not its own.
                self.run_border.mirror_to(viewer.run_border)

    def _create_status_bar(self):
        """Create the status bar: the shared bar's line on the left (FIB-1188), with
        the microscope's progress in it, and this window's workflow buttons on the
        right."""
        self.status_bar = FibsemStatusBar(self)
        self.setStatusBar(self.status_bar)
        # This run's failures, for the line it leaves when it ends (FIB-1188). Here
        # rather than on the Run click: an agent starts runs too, and the report
        # handlers read these unconditionally (FIB-329).
        self._run_active = False
        self._run_failures: List[Tuple[str, str]] = []
        self._run_total = 0
        self.status_bar.failure_details_requested.connect(self._show_run_failures)

        self._current_task_name = None  # Track current task for supervision toggle
        # The agent's chip, beside the workflow buttons (FIB-1035).
        self.agent_host = AgentChipHost(
            self.status_bar,
            session=lambda: self.autolamella_ui.session,
            view_controller=lambda: getattr(self, "view_controller", None),
            sources=PreviewSources(lambda: self.autolamella_ui.experiment),
            on_driving=self.run_border.set_agent_driving,
            parent=self,
        )

        # Attention Required, the supervision chip, Run and Stop: one component on the
        # right of the bar, which shows what it is told; what a click does is here.
        self.workflow_controls = WorkflowControls(self.status_bar)
        self.workflow_controls.supervision_clicked.connect(
            self._on_supervised_status_clicked
        )
        self.workflow_controls.stop_clicked.connect(self._on_stop_workflow_clicked)
        self.status_bar.add_action(self.workflow_controls)

    def _on_stop_workflow_clicked(self):
        """Handle stop workflow button click with confirmation."""
        reply = QMessageBox.question(
            self,
            "Stop Workflow",
            "Are you sure you want to stop the workflow?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes and self.autolamella_ui is not None:
            # The border goes red on the run relay's stopping, as for any stop.
            self.autolamella_ui.stop_task_workflow(actor=OPERATOR, via="Stop")

    def _on_workflow_stopping(self) -> None:
        """A stop was asked of the run, from any source. A bound method rather
        than a lambda, so a stop from the App Server's thread is delivered on
        this window's (GUI) thread."""
        self.run_border.set_state("stopping")

    def _refresh_after_naming(self) -> None:
        self.grids_tab.refresh()
        self.workflow_tab.grid_workflow_widget.refresh()
        self._refresh_grid_context()
        sample = self.autolamella_ui.panel.sample_widget
        for panel in ("loader_widget", "holder_widget"):
            widget = getattr(sample, panel, None)
            if widget is not None:
                widget.refresh()

    def _on_run_starting(self, request: RunRequest) -> None:
        """Any run, whoever started it (the Run click, the Grids view, the
        agent), before its thread starts: the border and the Stop button."""
        if request.kind == GRIDS:
            self.run_border.set_state("automated")
        else:
            self.run_border.set_state(self._running_border_state(request.task_names[0]))
        # The run writes the experiment from its own thread. Land any edit still
        # waiting in the editor first, so there is only ever one writer (FIB-683).
        self.lamella_tab.editor.flush_pending_save()
        # Show the Stop button immediately so the run is cancellable even while
        # waiting for a scheduled first task (before any task status arrives).
        self.set_workflow_running()

    def _say_selection(self, text: str) -> None:
        """What the Workflow tab's selection means, as the status line's last
        instruction: the panel's footer said it, under the lists, before the
        status line could."""
        self._selection_instruction = text
        self._update_instructions()

    def set_workflow_running(self):
        """Show the stop button, and lock what a run owns.

        Called on every report of a run, so the start is the first call after
        `hide_workflow_running`: that is when the last run's failure line goes."""
        if not self._run_active:
            self._run_active = True
            self._run_failures = []
            self._run_total = 0
            self.status_bar.dismiss_failure()
        self.workflow_controls.set_running(True)
        self.overview_tab.locks.set_run_allows(False)
        # A run owns the loader: no manual exchange from the Grids tab meanwhile.
        if getattr(self, "grids_tab", None) is not None:
            self.grids_tab.set_controls_enabled(False)
        self.workflow_tab.grid_workflow_widget.set_controls_enabled(False)
        sample = self.autolamella_ui.panel.sample_widget
        if sample is not None:
            sample.set_controls_enabled(False)
        self.workflow_tab.set_running(True)

    def hide_workflow_running(self):
        """Hide the stop button and show run button."""
        self._run_active = False
        self.status_bar.set_waiting(None)
        self.status_bar.set_run(None)
        self.workflow_controls.set_running(False)
        self.overview_tab.locks.set_run_allows(True)
        self.workflow_tab.set_running(False)

    def _running_border_state(self, task_name: Optional[str]) -> str:
        """The border for a running workflow: automated or supervised."""
        if task_name is None or not get_task_supervision(
            task_name, self.autolamella_ui.experiment
        ):
            return "automated"
        return "supervised"

    def _update_supervised_status(self) -> bool:
        """Update the supervised status chip for the current task."""
        task_name = self._current_task_name
        if task_name is None or self.autolamella_ui is None:
            return False
        supervised = get_task_supervision(task_name, self.autolamella_ui.experiment)
        if supervised:
            mode = SUPERVISED
        else:
            mode = AUTOMATED
        self.workflow_controls.set_supervision(mode, task_name)

        return supervised

    def _on_supervised_status_clicked(self):
        """Toggle supervision for the current task in the protocol."""
        if self._current_task_name is None or self.autolamella_ui is None:
            return
        protocol = self.autolamella_ui.protocol
        if protocol is None:
            return
        for task in protocol.workflow_config.tasks:
            if task.name == self._current_task_name:
                task.attention = (
                    Attention.automated
                    if task.attention is Attention.supervised
                    else Attention.supervised
                )
                break
        self._update_supervised_status()
        if self.autolamella_ui.session.runner.is_running:
            self.run_border.set_state(
                self._running_border_state(self._current_task_name)
            )
        # Refresh the workflow widget to reflect the toggled supervise state
        self.workflow_tab.lamella_workflow_widget.workflow.refresh_all()

    def _update_instructions(self):
        """Update the status bar with the current instruction based on application state."""
        if self.autolamella_ui is None or self.status_bar is None:
            return
        is_connected = self.autolamella_ui.microscope is not None
        experiment = self.autolamella_ui.experiment
        is_experiment_loaded = experiment is not None
        # Carried here from the panel, which used to render the same ladder into a
        # label of its own: an experiment with no protocol cannot run anything, and
        # this was the one rung the status bar did not have.
        is_protocol_loaded = self.autolamella_ui.protocol is not None
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
        microscope = self.autolamella_ui.microscope if self.autolamella_ui else None
        no_fm = microscope is not None and getattr(microscope, "fm", None) is None
        view_controller.fm_canvas.set_placeholder(
            "No fluorescence microscope available" if no_fm else "No image"
        )

    def _hear_session(self, what: str) -> None:
        self._session_changed.emit(what)

    def _on_session_changed(self, what: str) -> None:
        if what == MICROSCOPE:
            self._on_microscope_changed()
        elif what == EXPERIMENT:
            self._on_experiment_update()

    def _on_microscope_changed(self):
        """Another microscope, or none: hand it to everything that holds one."""
        microscope = self.autolamella_ui.microscope if self.autolamella_ui else None
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

    def _on_tab_changed(self, index: int):
        """Handle tab change and update status bar."""
        self.status_bar.setStyleSheet(STATUS_BAR_STYLESHEET)

    def _create_main_tab(self):
        """Create the main AutoLamella tab."""
        # Create the embedded viewer container
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        # Viewer-less: the quad-view controller is the display, so no napari viewer is
        # created here.
        self.main_viewer = None

        # Create the AutoLamellaUI widget
        self.autolamella_ui = AutoLamellaUI(parent_ui=self)

        # Everything the workflow says without needing an answer arrives here
        self.autolamella_ui.run_relay.status.connect(self._on_workflow_status)
        self.autolamella_ui.run_relay.step.connect(self._on_step_update)
        self.autolamella_ui.experiment_update_signal.connect(self._on_experiment_update)
        self.autolamella_ui.run_relay.finished.connect(self._on_workflow_finished)
        self.autolamella_ui.session.runner.starting.connect(self._on_run_starting)
        # Red while a stop unwinds, whoever asked for it (the agent's too).
        self.autolamella_ui.run_relay.stopping.connect(self._on_workflow_stopping)
        # A grid task records its overview as it finishes, and says so in a status.
        self.autolamella_ui.run_relay.status.connect(self._refresh_quad_overviews)
        self.autolamella_ui.run_relay.finished.connect(self._refresh_quad_overviews)
        self.autolamella_ui.run_relay.notice.connect(self.show_toast)
        self.autolamella_ui.grids_changed.connect(self._refresh_after_naming)
        notification_service._get_service().toast.connect(self._on_notification_service)
        notification_service._get_service().toast_action.connect(
            self._on_notification_service_action
        )
        self._session_changed.connect(self._on_session_changed)
        self.autolamella_ui.session.changed.connect(self._hear_session)
        self.autolamella_ui.panel.experiment_tab.lamella_list.defect_changed.connect(
            self._on_lamella_defect_changed
        )
        self.autolamella_ui.panel.experiment_tab.lamella_list.lamella_selected.connect(
            self._on_experiment_lamella_selected
        )

        # hide menu bar
        self.autolamella_ui.menuBar().setVisible(False)
        self.autolamella_ui.setMinimumWidth(550)

        # Layout: quad view (left) | autolamella controls (right) via splitter.
        # The controller's SEM/FIB/FM canvases are the display and drive the control
        # widgets, which resolve it through parent -> parent_widget -> view_controller.
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        self.view_controller = MicroscopeViewController(parent=self)
        self.status_bar.follow(self.view_controller)
        splitter.addWidget(self.view_controller.widget)
        splitter.addWidget(self.autolamella_ui)
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
            prompts=self.autolamella_ui.proposals,
            tab_widget=self.tab_widget,
            controls=self.workflow_controls,
            status_bar=self.status_bar,
            timeline=self.workflow_tab.timeline,
            sound_enabled=lambda: self._preferences.display.sound_enabled,
            parent=self,
        )

        self.experiment_header = ExperimentHeader(
            chip_enabled=self.menu_bar.connection_chip_enabled,
            new_experiment=self.menu_bar.action_new_experiment,
            load_experiment=self.menu_bar.action_load_experiment,
            open_experiment_directory=self.menu_bar.action_open_experiment_directory,
            on_new_experiment=self.autolamella_ui.create_experiment,
            on_load_experiment=self.autolamella_ui.load_experiment,
            on_connect=self.menu_bar.connect_microscope,
            on_about=self.autolamella_ui.open_information_dialog,
            toast_manager=self.toast_manager,
        )
        self.tab_widget.setCornerWidget(self.experiment_header)

    def _refresh_header(self) -> None:
        ui = self.autolamella_ui
        self.experiment_header.refresh(
            ui.experiment if ui else None, ui.microscope if ui else None
        )

    def _on_experiment_update(self):
        """Handle experiment update signal and propagate to tabs."""

        if self.autolamella_ui is None:
            return
        if self.autolamella_ui.experiment is None:
            return

        self.overview_tab.refresh_experiment()
        self.task_widget.set_experiment(self.autolamella_ui.experiment)
        self.lamella_tab.editor.set_experiment()
        self.grids_tab.set_experiment(self.autolamella_ui.experiment)
        self.workflow_tab.grid_workflow_widget.set_experiment(
            self.autolamella_ui.experiment
        )
        experiment = self.autolamella_ui.experiment
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
        self.autolamella_ui.setMinimumWidth(500)
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
        experiment = self.autolamella_ui.experiment if self.autolamella_ui else None
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
        self.task_widget = AutoLamellaProtocolTaskConfigEditor(
            session=self.autolamella_ui.session
        )
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
            self.autolamella_ui.session,
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
            session=self.autolamella_ui.session,
            add_lamella=self.autolamella_ui.panel.experiment_tab.add_new_lamella,
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
        ui = self.autolamella_ui
        stage = getattr(getattr(ui, "microscope", None), "sample_handling", None)
        if ui is None or ui.experiment is None or stage is None:
            return
        try:
            added = ui.experiment.sync_grids_from_inventory(stage)
        except Exception as e:  # noqa: BLE001 - the inventory still shows
            logging.warning(f"Could not record the inventoried grids: {e}")
            return
        if not added:
            return
        logging.info(f"Recorded {len(added)} grid(s) from the inventory.")
        try:
            ui.experiment.save()
        except Exception as e:  # noqa: BLE001 - recorded; saved next time
            logging.warning(f"Could not save the experiment: {e}")

    def _refresh_grids_tab_microscope(self):
        if getattr(self, "grids_tab", None) is None or self.autolamella_ui is None:
            return
        self.grids_tab.set_microscope(self.autolamella_ui.microscope)
        self.workflow_tab.grid_workflow_widget.set_microscope(
            self.autolamella_ui.microscope
        )
        # An exchange from Microscope -> Sample changes what is loaded; the
        # Grids tab's chips and the run view's rows follow. The Sample view is
        # rebuilt on every connect, so this is wired here, on every connect.
        sample = self.autolamella_ui.panel.sample_widget
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
        one. Nothing is renamed while a workflow runs, since its queue holds
        grids by name. With no experiment open there is no record to protect.
        """
        ui = self.autolamella_ui
        experiment = getattr(ui, "experiment", None)
        if experiment is None:
            return ""
        if ui.session.runner.is_running:
            return "Grids cannot be renamed while a workflow is running."
        return grid_rename_refusal(experiment, old, new)

    def _on_slot_grid_renamed(self, old: str, new: str) -> None:
        """A grid renamed on the Sample view: its record takes the new name, with
        its history, verdict, note and lamellae, instead of the next inventory
        sync adding a second record under the new name."""
        experiment = getattr(self.autolamella_ui, "experiment", None)
        if experiment is None or experiment.get_grid_by_name(old) is None:
            return  # no record yet: the sync adds one under the new name
        # The Grids tab's and the App Server's rename. The slot already has the
        # new name, so only the record changes here.
        stage = getattr(self.autolamella_ui.microscope, "sample_handling", None)
        rename_grid(experiment, stage, old, new)
        try:
            experiment.save()
        except Exception as e:  # noqa: BLE001 - renamed in memory; saved next time
            logging.warning(f"Could not save the experiment after a rename: {e}")
        self.grids_tab.refresh()
        self.workflow_tab.grid_workflow_widget.refresh()
        self._refresh_grid_context()

    def _grid_context(self):
        """`GridRecord.id -> (name, on the stage)` for the lamella displays,
        from the experiment's records and what the stage already knows. No
        hardware call: the stage answers from its last read."""
        experiment = self.autolamella_ui.experiment if self.autolamella_ui else None
        if experiment is None or not experiment.grids:
            return None
        loaded = set()
        stage = getattr(self.autolamella_ui.microscope, "sample_handling", None)
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
            self.autolamella_ui.panel.experiment_tab.lamella_list,
        ):
            if widget is not None:
                widget.set_grid_context(context)
        # The Overview canvases mark only the lamellae on the stage, so what
        # they draw changes with every load and unload too.
        self._refresh_overview_positions()

    def _refresh_sample_view(self) -> None:
        """Redraw Microscope → Sample from the stage. Looked up each time: the
        Sample view is rebuilt on every connect."""
        sample = self.autolamella_ui.panel.sample_widget
        if sample is not None:
            sample.refresh()

    def add_workflow_tab(self):
        """The Workflow tab: what to run, the queue and its timeline."""
        ui = self.autolamella_ui
        self.workflow_tab = WorkflowTab(
            ui.session,
            controls=self.workflow_controls,
            flush_edits=self.lamella_tab.editor.flush_pending_save,
            on_config_changed=ui._on_workflow_config_changed,
            stop_task=ui.stop_current_task,
            after_naming=self._refresh_after_naming,
            say=self._say_selection,
            show_toast=self.show_toast,
        )
        ui.run_relay.queue_changed.connect(self.workflow_tab.on_queue_changed)
        lamella = self.workflow_tab.lamella_workflow_widget
        lamella.lamella_move_to_requested.connect(self._on_lamella_move_to)
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
                    self.autolamella_ui.panel.experiment_tab.lamella_list.select(
                        lamella.name
                    )
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
        """Handle a fire-and-forget status update, from the run relay.

        The one channel for everything the workflow says without needing an
        answer — the dict workflow_update_signal is gone. The run indicators
        refresh on every event because any of them can change what they show: a
        lifecycle report starts or finishes the run, and the responder pings
        this signal when a question flips the waiting state.
        """
        # What the run is doing now (a scheduled-start countdown, waiting on
        # decisions, loading a grid), under the task it is for; "" clears it.
        if event.status_bar is not None:
            self.status_bar.set_run_step(event.status_bar)

        if event.report is not None:
            self._apply_status_report(event.report)

        if self.autolamella_ui is None:
            return
        self._refresh_workflow_indicators()

    def _apply_status_report(self, report: "WorkflowStatusUpdate") -> None:
        """Render one task-lifecycle report: timeline, run message, list refreshes."""
        # No build-once guard: update_from_status reconciles rows against the
        # snapshot by WorkItem id, so first paint and every later change go
        # through the same path.
        self.workflow_tab.timeline.update_from_status(report)

        task_name = report.task_name
        lamella_name = report.item_name
        queue_position = report.queue_position
        queue_total = report.queue_total
        status = report.status

        # Position in the live queue, not the launch matrix — stays correct
        # when the queue is added to or reordered mid-run.
        # `queue_total` is a count, so 0 means "nothing to be in the middle of"
        # rather than "unknown" -- truthiness, not `is not None`, or a malformed
        # payload renders "3 of 0".
        where = None
        if queue_position is not None and queue_total:
            where = f"{queue_position} of {queue_total}"

        self.status_bar.set_run(f"{lamella_name} › {task_name}", None, where)
        self.set_workflow_running()
        if queue_total:
            self._run_total = queue_total
        if status is TaskStatus.Failed:
            # Said for a moment under its own name -- the run moves on to the next
            # item -- and kept for the line the run leaves when it ends.
            what = f"{lamella_name} › {task_name}"
            reason = report.error_message or "failed"
            self._run_failures.append((what, reason))
            self.status_bar.show_outcome(
                what, f"failed: {reason}", failed=True, standalone=True
            )

        # update current task
        self._current_task_name = task_name

        # Lock editor when the active lamella/task is being processed
        if status is TaskStatus.InProgress:
            self.lamella_tab.editor.set_active_lamella_name(lamella_name, task_name)
        else:
            self.lamella_tab.editor.set_active_lamella_name(None)

        # Refresh only the affected lamella if we can identify it
        lamella = None
        experiment = self.autolamella_ui.experiment
        if experiment is not None and lamella_name is not None:
            lamella = experiment.get_lamella_by_name(lamella_name)

        # The name list is refreshed from here rather than subscribing per row.
        # `_LamellaRow` used to connect to `task_state.events.name`/`.status`, which
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
            sample = self.autolamella_ui.panel.sample_widget
            if sample is not None:
                sample.refresh()
        elif lamella is not None:
            self.workflow_tab.lamella_workflow_widget.lamella_list.refresh_lamella(
                lamella
            )
            self.autolamella_ui.panel.experiment_tab.lamella_list.refresh_lamella(
                lamella
            )
            # The Overview pages' lists carry the same status column; without
            # this they sat stale for the whole run (FIB-995).
            for lamella_list in self.overview_tab.lamella_lists():
                lamella_list.refresh_lamella(lamella)
        else:
            self.workflow_tab.lamella_workflow_widget.lamella_list.refresh_all()
            self.autolamella_ui.panel.experiment_tab.lamella_list.refresh_all()
            for lamella_list in self.overview_tab.lamella_lists():
                lamella_list.refresh_all()
        if grid is None:
            self.lamella_tab.task_reported(lamella, status)
        self._on_lamella_card_selected(self.lamella_tab.selected)

    def _on_app_server_dialog(self) -> None:
        """Open Tools → App Server: status, services, clients, the dashboard."""
        from fibsem.ui.agent.app_server_dialog import (
            AppServerDialog,
        )

        preference = fibsem_cfg.load_user_preferences().app_server
        dialog = AppServerDialog(
            host_provider=lambda: self.autolamella_ui.session.app_server,
            parent=self,
            set_enabled=self._set_app_server_enabled,
            set_microscope=self._set_app_server_microscope,
            enabled=preference.enabled,
            serve_microscope=preference.microscope,
        )
        dialog.exec_()

    def _set_app_server_enabled(self, enabled: bool) -> None:
        """Tools → App Server: save whether the server runs, and act on it now."""
        self._preferences = fibsem_cfg.update_user_preferences(
            lambda p: setattr(p.app_server, "enabled", bool(enabled))
        )
        self.autolamella_ui.sync_app_server_with_preference()

    def _set_app_server_microscope(self, serve: bool) -> None:
        """Tools → App Server: save whether the microscope is served, and act on it
        now (a running server restarts)."""
        self._preferences = fibsem_cfg.update_user_preferences(
            lambda p: setattr(p.app_server, "microscope", bool(serve))
        )
        self.autolamella_ui.set_app_server_microscope(serve)

    def _refresh_workflow_indicators(self) -> None:
        """Re-read the waiting/supervised/running state into the window chrome."""
        # refresh the supervised status chip
        self._update_supervised_status()

        hold = self.autolamella_ui.run_relay.hold
        self.attention.show_hold(hold)

        self.run_border.set_state(
            run_border_state(
                self.run_border.state,
                held=hold is not None,
                pending=self.autolamella_ui.run_relay.pending,
                running=self._running_border_state(self._current_task_name)
                if self.autolamella_ui.session.runner.is_running
                else None,
            )
        )

    def _rebuild_lamella_list(self):
        """Clear and repopulate the lamella list and card container from the current experiment."""
        if not hasattr(self, "workflow_tab"):
            return
        experiment = self.autolamella_ui.experiment if self.autolamella_ui else None
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
        experiment = self.autolamella_ui.experiment if self.autolamella_ui else None
        has_positions = experiment is not None and len(experiment.positions) > 0
        self.tab_widget.setTabEnabled(index, has_positions)
        self.tab_widget.setTabToolTip(
            index, "" if has_positions else "Add lamella positions to enable this tab"
        )

    def _on_lamella_move_to(self, lamella: "Lamella"):
        """Move the stage to the given lamella's milling position."""
        if self.autolamella_ui is None or self.autolamella_ui.experiment is None:
            return
        self.autolamella_ui.panel.experiment_tab.lamella_list.select(lamella.name)
        self.autolamella_ui.panel.experiment_tab.move_to_lamella_position()

    def _on_lamella_card_update_position(self, lamella: "Lamella"):
        """Update the stage position of the given lamella to the current stage position."""
        if self.autolamella_ui is None:
            return
        self.autolamella_ui.panel.experiment_tab.lamella_list.select(lamella.name)
        self.autolamella_ui.panel.experiment_tab.update_lamella_position_ui()

    def _on_lamella_edit(self, lamella: "Lamella"):
        """Switch to the Lamella tab and select the given lamella in the protocol editor."""
        if self.autolamella_ui is None or self.autolamella_ui.experiment is None:
            return
        self.autolamella_ui.panel.experiment_tab.lamella_list.select(lamella.name)

        # Select the lamella in the protocol editor
        self.lamella_tab.editor.select_lamella(lamella.name)

        # Switch to the Lamella tab
        for i in range(self.tab_widget.count()):
            if self.tab_widget.tabText(i) == "Lamella":
                self.tab_widget.setCurrentIndex(i)
                break

    def _on_lamella_defect_changed(self, lamella: "Lamella"):
        """Persist defect state change to disk and sync all widgets."""
        if self.autolamella_ui is None or self.autolamella_ui.experiment is None:
            return
        self.autolamella_ui.experiment.sign_verdict(lamella)
        self.autolamella_ui.experiment.save()
        # Sync defect icon across all widgets
        self.autolamella_ui.panel.experiment_tab.lamella_list.refresh_all()
        self.workflow_tab.lamella_workflow_widget.lamella_list.refresh_lamella(lamella)
        self.lamella_tab.cards.refresh_lamella(lamella)

    def _on_lamella_remove_requested(self, lamella: "Lamella"):
        """Remove the given lamella from the experiment after the list widget has already removed its row."""
        if self.autolamella_ui is None or self.autolamella_ui.experiment is None:
            return
        if not remove_lamella(self.autolamella_ui.experiment, lamella):
            return
        self.autolamella_ui.panel.experiment_tab.update_lamella_combobox(latest=True)
        self.autolamella_ui.update_ui()

    def _on_step_update(self, label: str) -> None:
        """Handle per-step update from the workflow worker thread."""
        self.workflow_tab.timeline.update_step(label)

    def _on_workflow_finished(self, cancelled: bool = False):
        """Handle workflow finished signal."""
        # The next run's items carry new ids, so the timeline rebuilds itself on
        # its first status update — nothing to reset here.
        # Resolve any outer row left in ACTIVE state (e.g. if workflow was cancelled)
        self.workflow_tab.timeline.finish_current_step(failed=cancelled)
        self.workflow_tab.timeline.clear_steps()
        self.hide_workflow_running()
        self.lamella_tab.editor.set_active_lamella_name(None)
        if getattr(self, "grids_tab", None) is not None:
            self.grids_tab.set_controls_enabled(True)
        self.workflow_tab.grid_workflow_widget.set_controls_enabled(True)
        self.workflow_tab.grid_workflow_widget.refresh()
        sample = self.autolamella_ui.panel.sample_widget
        if sample is not None:
            sample.set_controls_enabled(True)
            sample.refresh()
        self.workflow_controls.set_attention(None)
        self.workflow_tab.lamella_workflow_widget.lamella_list.refresh_all()
        self.lamella_tab.cards.refresh_all()
        self._say_how_the_run_ended()
        self.run_border.set_state("idle")

    def _show_run_failures(self) -> None:
        """The failure line's Show: the Workflow tab, with the run's
        first failed row selected and in view. The timeline is where the run is
        laid out, failures marked, and where they are queued again."""
        self.tab_widget.setCurrentWidget(self.workflow_tab)
        self.workflow_tab.timeline.show_first_failed()

    def _say_how_the_run_ended(self) -> None:
        """The run's line has gone (`hide_workflow_running`). A run with failures,
        or one that stalled, leaves a line that stays until it is dismissed or the
        next run starts; a clean one says so once, as a toast, and the line goes
        back to the instruction. A run with a summary was already announced by
        the "Run ended" toast (`_show_workflow_summary`), so this says nothing."""
        runner = self.autolamella_ui.session.runner
        note = runner.last_note
        failures = self._run_failures
        if failures:
            what, reason = failures[-1]
            n = len(failures)
            count = f"{n} of {self._run_total}" if self._run_total >= n else f"{n}"
            self.status_bar.show_failure(
                "Run finished", f"{count} failed · {what}: {reason}"
            )
        elif note:
            self.status_bar.show_failure("Run stalled", note)
        elif runner.last_summary is None or runner.last_summary.empty:
            notification_service.show_toast("Workflow finished.")

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
            self.autolamella_ui.session,
            self.autolamella_ui.panel.experiment_tab.add_new_lamella,
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
            self.autolamella_ui.panel.experiment_tab.lamella_list.select(lamella.name)
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
        experiment = getattr(self.autolamella_ui, "experiment", None)
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

    def _on_notification_service(
        self, message: str, notification_type: str, temporary: bool
    ) -> None:
        self.show_toast(message, notification_type, temporary=temporary)

    def _on_notification_service_action(
        self,
        message: str,
        notification_type: str,
        temporary: bool,
        action_text: str,
        callback,
    ) -> None:
        self.show_toast(
            message,
            notification_type,
            temporary=temporary,
            action=(action_text, callback),
        )

    def _confirm_close_during_workflow(self) -> bool:
        """Ask before closing ends a workflow in progress; True to go ahead.

        Only asked while one is running: any other close loses nothing, since
        pending edits are flushed on the way out. Confirming stops the run, the
        same as Stop Workflow, without waiting for it to wind down -- the worker
        may be waiting on this thread, and it is a daemon thread, so it cannot
        hold the process open.
        """
        ui = self.autolamella_ui
        if ui is None or not ui.session.runner.is_running:
            return True
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Workflow Running")
        box.setText("A workflow is running. Close anyway?")
        box.setInformativeText(
            "Closing stops the workflow and disconnects from the microscope."
        )
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.No)
        if box.exec_() != QMessageBox.Yes:
            return False
        ui.stop_task_workflow(actor=OPERATOR, via="closing the window")
        return True

    def closeEvent(self, event):
        """Flush what is still pending, then let the application go."""
        # Before anything else, so a cancelled close leaves the session untouched.
        if not self._confirm_close_during_workflow():
            event.ignore()
            return
        # The editor holds edits for a moment before writing them (FIB-683); this is
        # the last chance to get the final one onto disk.
        if getattr(self, "lamella_tab", None) is not None:
            try:
                self.lamella_tab.editor.flush_pending_save()
            except Exception as e:
                logging.warning(
                    f"Could not flush a pending experiment save on close: {e}"
                )
        # persist the FM working state (channels / camera transform / objective)
        if (
            self.autolamella_ui is not None
            and self.autolamella_ui.panel.fm_control_widget is not None
        ):
            try:
                self.autolamella_ui.panel.fm_control_widget.save_fm_configuration()
            except Exception as e:
                logging.warning(f"Could not save FM working state on close: {e}")
        # After the flush, so its edit is recorded: write what is queued to
        # events.jsonl and stop the recorder's writer thread. Nothing else does,
        # so a window closed without disconnecting leaves it running.
        if self.autolamella_ui is not None:
            try:
                self.autolamella_ui.session.stop_recording()
            except Exception as e:
                logging.warning(f"Could not close the event recorder on close: {e}")
        # Last, so everything above has the microscope it needs. Let the
        # instrument go rather than leave the client open until the process ends.
        if (
            self.autolamella_ui is not None
            and self.autolamella_ui.microscope is not None
        ):
            self.autolamella_ui.microscope.try_disconnect()
        if self.autolamella_ui is not None:
            self.autolamella_ui.session.changed.disconnect(self._hear_session)
        # TypeError: a second close finds it already disconnected -- close() sends
        # the close event again even when the window is hidden.
        try:
            notification_service._get_service().toast.disconnect(
                self._on_notification_service
            )
        except (RuntimeError, TypeError):
            pass
        super().closeEvent(event)
        # Force the event loop to exit even if another top-level window (e.g. a stray
        # matplotlib pyplot figure) would otherwise keep the app alive once this window
        # closes. quit() — not os._exit — so atexit / experiment autosave still run.
        app = QApplication.instance()
        if app is not None:
            app.quit()
