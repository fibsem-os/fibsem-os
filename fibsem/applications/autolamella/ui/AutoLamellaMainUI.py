from __future__ import annotations

import sys

try:
    sys.modules.pop("PySide6.QtCore")
except Exception:
    pass

import logging
import traceback
import warnings

import napari
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from superqt import ensure_main_thread
from superqt.iconify import QIconifyIcon

import fibsem
from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus
from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI, INSTRUCTIONS
from fibsem.applications.autolamella.workflows.tasks.tasks import get_task_supervision
from fibsem.ui import FibsemMinimapWidget
from fibsem.ui.stylesheets import (
    MILLING_PROGRESS_BAR_STYLESHEET,
    NAPARI_STYLE,
    RUN_WORKFLOW_BUTTON_STYLESHEET,
    STOP_WORKFLOW_BUTTON_STYLESHEET,
    SUPERVISION_STATUS_AUTOMATED_STYLESHEET,
    SUPERVISION_STATUS_SUPERVISED_STYLESHEET,
    USER_ATTENTION_BUTTON_STYLESHEET,
    STATUS_BAR_STYLESHEET,
    PRIMARY_BUTTON_STYLESHEET,
    SECONDARY_BUTTON_STYLESHEET,
)
from fibsem.ui.widgets.autolamella_lamella_protocol_editor import (
    AutoLamellaProtocolEditorWidget,
)
from fibsem.ui.widgets.autolamella_task_config_editor import (
    AutoLamellaProtocolTaskConfigEditor,
)
from fibsem.ui.widgets.lamella_card_widget import LamellaCardContainer
from fibsem.ui.widgets.lamella_workflow_widget import LamellaWorkflowWidget
from fibsem.ui.widgets.notifications import NotificationBell, ToastManager
# from fibsem.ui.widgets.task_history_table_widget import TaskHistoryTableWidget
from fibsem.utils import format_duration

# Suppress a specific upstream Napari/NumPy warning from shapes miter computation.
warnings.filterwarnings(
    "ignore",
    message=r"'where' used without 'out', expect unit?ialized memory in output\. If this is intentional, use out=None\.",
    category=UserWarning,
    module=r"napari\.layers\.shapes\._shapes_utils",
)

def play_notification_sound():
    """Play a notification sound to alert the user."""
    QApplication.beep()

class AutoLamellaSingleWindowUI(QMainWindow):
    """Main window for AutoLamella UI with embedded napari viewers."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"AutoLamella v{fibsem.__version__} ")
        self.resize(1600, 1000)

        # Apply napari-style dark theme
        self.setStyleSheet(NAPARI_STYLE)

        # Central tab widget
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        self.viewers: list[napari.Viewer] = []
        self.autolamella_ui: AutoLamellaUI
        self.minimap_widget: FibsemMinimapWidget
        self.minimap_viewer: napari.Viewer

        # Toast notification manager
        self.toast_manager = ToastManager(self)

        # User attention tracking
        self._user_interaction_sound_played = False  # Track if sound was played
        self._sound_enabled = False  # Toggle for notification sounds
        self._toasts_enabled = True  # Toggle for toast notifications

        # create menus, status bar, and tabs
        self._create_menu_bar()
        self._create_test_menu()
        self._create_status_bar()
        self.create_tabs()
        self._update_instructions()

        # Connect tab change to status bar update
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

    def _create_menu_bar(self):
        """Create the main menu bar."""
        menu_bar = self.menuBar()

        if menu_bar is None:
            raise RuntimeError("Failed to create menu bar for AutoLamella UI.")

        file_menu = menu_bar.addMenu("File")
        if file_menu is None:
            raise RuntimeError("Failed to create File menu in AutoLamella UI.")

        view_menu = menu_bar.addMenu("View")
        if view_menu is None:
            raise RuntimeError("Failed to create View menu in AutoLamella UI.")

        self.action_new_experiment = QAction("New Experiment", self)
        self.action_new_experiment.triggered.connect(self._on_new_experiment)

        self.action_load_experiment = QAction("Load Experiment", self)
        self.action_load_experiment.triggered.connect(self._on_load_experiment)

        self.action_open_experiment_directory = QAction("Open Experiment Directory", self)
        self.action_open_experiment_directory.triggered.connect(self._on_open_experiment_directory)

        self.action_load_protocol = QAction("Load Protocol", self)
        self.action_load_protocol.triggered.connect(self._on_load_protocol)
        self.action_save_protocol = QAction("Save Protocol", self)
        self.action_save_protocol.triggered.connect(self._on_save_protocol)
        self.action_exit = QAction("Exit", self)
        self.action_exit.triggered.connect(self.close) # type: ignore

        file_menu.addAction(self.action_new_experiment)
        file_menu.addAction(self.action_load_experiment)
        file_menu.addAction(self.action_open_experiment_directory)
        file_menu.addSeparator()
        file_menu.addAction(self.action_load_protocol)
        file_menu.addAction(self.action_save_protocol)
        file_menu.addSeparator()
        file_menu.addAction(self.action_exit)

        # View menu
        self.action_show_minimap = QAction("Show Minimap Widget", self)
        self.action_show_minimap.setCheckable(True)
        self.action_show_minimap.setChecked(True)
        self.action_show_minimap.triggered.connect(self._on_toggle_minimap_widget)

        self.action_toggle_layer_controls = QAction("Show Layer Controls", self)
        self.action_toggle_layer_controls.setCheckable(True)
        self.action_toggle_layer_controls.setChecked(True)
        self.action_toggle_layer_controls.triggered.connect(self._on_toggle_layer_controls)

        view_menu.addAction(self.action_show_minimap)
        view_menu.addAction(self.action_toggle_layer_controls)

        # add tools menu, reporting submenu
        tools_menu = menu_bar.addMenu("Tools")
        if tools_menu is None:
            raise RuntimeError("Failed to create Tools menu in AutoLamella UI.")
        reporting_menu = tools_menu.addMenu("Reporting")
        if reporting_menu is None:
            raise RuntimeError("Failed to create Reporting submenu in AutoLamella UI.")
        self.action_generate_report = QAction("Generate Report", self)
        self.action_generate_report.triggered.connect(self._on_generate_report)
        self.action_generate_overview_plot = QAction("Generate Overview Plot", self)
        self.action_generate_overview_plot.triggered.connect(self._on_generate_overview_plot)
        reporting_menu.addAction(self.action_generate_report)
        reporting_menu.addAction(self.action_generate_overview_plot)

        # add help menu
        help_menu = menu_bar.addMenu("Help")
        if help_menu is None:
            raise RuntimeError("Failed to create Help menu in AutoLamella UI.")
        self.action_about = QAction("About", self)
        self.action_about.triggered.connect(self._show_about_dialog)
        help_menu.addAction(self.action_about)

    def _create_test_menu(self):        
        """Create a test menu for toast notifications and sounds."""

        self.action_toast_info = QAction("Toast: Info", self)
        self.action_toast_info.triggered.connect(lambda: self.show_toast("This is an info message", "info"))

        self.action_toast_success = QAction("Toast: Success", self)
        self.action_toast_success.triggered.connect(lambda: self.show_toast("Operation completed successfully!", "success"))

        self.action_toast_warning = QAction("Toast: Warning", self)
        self.action_toast_warning.triggered.connect(lambda: self.show_toast("Warning: Check your settings", "warning"))

        self.action_toast_error = QAction("Toast: Error", self)
        self.action_toast_error.triggered.connect(lambda: self.show_toast("Error: Something went wrong", "error"))

        self.action_beep = QAction("Play Beep", self)
        self.action_beep.triggered.connect(play_notification_sound)

        self.action_sound_toggle = QAction("Sound Enabled", self)
        self.action_sound_toggle.setCheckable(True)
        self.action_sound_toggle.setChecked(self._sound_enabled)
        self.action_sound_toggle.triggered.connect(self._on_sound_toggle)

        self.action_toasts_toggle = QAction("Toasts Enabled", self)
        self.action_toasts_toggle.setCheckable(True)
        self.action_toasts_toggle.setChecked(self._toasts_enabled)
        self.action_toasts_toggle.triggered.connect(self._on_toasts_toggle)

        # add to menu bar
        menu_bar = self.menuBar()
        test_menu = menu_bar.addMenu("Test")                # type: ignore
        test_menu.addAction(self.action_toast_info)         # type: ignore
        test_menu.addAction(self.action_toast_success)      # type: ignore
        test_menu.addAction(self.action_toast_warning)      # type: ignore
        test_menu.addAction(self.action_toast_error)        # type: ignore
        test_menu.addSeparator()                            # type: ignore
        test_menu.addAction(self.action_beep)               # type: ignore
        test_menu.addAction(self.action_sound_toggle)       # type: ignore
        test_menu.addAction(self.action_toasts_toggle)      # type: ignore 

    def _on_sound_toggle(self, checked: bool):
        """Handle sound toggle."""
        self._sound_enabled = checked

    def _on_toasts_toggle(self, checked: bool):
        """Handle toasts toggle."""
        self._toasts_enabled = checked

    def show_toast(self, message: str, notification_type: str = "info", duration: int = 5000):
        """Show a toast notification."""
        if self._toasts_enabled:
            self.toast_manager.show_toast(message, notification_type, duration)
        elif self.toast_manager.notification_bell:
            # Still log to notification bell even when toasts are disabled
            self.toast_manager.notification_bell.add_notification(message, notification_type)

    def _on_new_experiment(self):
        """Handle New Experiment action."""
        if self.autolamella_ui is not None:
            self.autolamella_ui.create_experiment()

    def _on_load_experiment(self):
        """Handle Load Experiment action."""
        if self.autolamella_ui is not None:
            self.autolamella_ui.load_experiment()

    def _on_open_experiment_directory(self):
        """Handle Open Experiment Directory action."""
        if self.autolamella_ui is not None:
            self.autolamella_ui.action_open_experiment_directory.trigger()

    def _on_load_protocol(self):
        """Handle Load Protocol action."""
        if self.autolamella_ui is not None:
            self.autolamella_ui.load_protocol()

    def _on_save_protocol(self):
        """Handle Save Protocol action."""
        if self.autolamella_ui is not None:
            self.autolamella_ui.export_protocol_ui()

    def _show_about_dialog(self):
        """Show the About dialog."""
        if self.autolamella_ui is not None:
            self.autolamella_ui.actionInformation.trigger()

    def _on_toggle_minimap_widget(self, checked: bool):
        """Toggle the minimap plot dock widget visibility."""
        if self.autolamella_ui is not None and hasattr(self.autolamella_ui, 'minimap_plot_dock'):
            self.autolamella_ui.minimap_plot_dock.setVisible(checked)
            self.autolamella_ui.minimap_plot_dock.activateWindow()

    def _on_toggle_layer_controls(self, checked: bool):
        """Toggle the layer list and layer controls for all viewers."""
        for viewer in self.viewers:
            qt_viewer = viewer.window._qt_viewer
            qt_viewer.dockLayerList.setVisible(checked)
            qt_viewer.dockLayerControls.setVisible(checked)

    def _on_generate_report(self):
        """Handle Generate Report action."""
        if self.autolamella_ui is not None:
            self.autolamella_ui.actionGenerate_Report.trigger()
    
    def _on_generate_overview_plot(self):
        """Handle Generate Overview Plot action."""
        if self.autolamella_ui is not None:
            self.autolamella_ui.actionGenerate_Overview_Plot.trigger()

    def _create_status_bar(self):
        """Create the status bar."""
        self.status_bar = self.statusBar()
        if self.status_bar is None:
            raise RuntimeError("Failed to create status bar for AutoLamella UI.")
        self.status_bar.setStyleSheet(STATUS_BAR_STYLESHEET)
    
        # Add milling progress bar
        self.milling_progress_bar = QProgressBar(self.status_bar)
        self.milling_progress_bar.setMaximumWidth(400)
        self.milling_progress_bar.setMaximum(100)
        self.milling_progress_bar.setValue(0)
        self.milling_progress_bar.setTextVisible(True)
        self.milling_progress_bar.setStyleSheet(MILLING_PROGRESS_BAR_STYLESHEET)
        self.milling_progress_bar.hide()  # Hidden by default
        self.status_bar.addPermanentWidget(self.milling_progress_bar)

        # Add user attention button (shown when waiting for user interaction)
        self.user_attention_btn = QPushButton("Attention Required")
        self.user_attention_btn.setStyleSheet(USER_ATTENTION_BUTTON_STYLESHEET)
        self.user_attention_btn.hide()  # Hidden by default
        self.user_attention_btn.setToolTip("User Input Required - Click to go to Microscope tab")
        self.user_attention_btn.clicked.connect(self._on_user_attention_clicked)
        self.status_bar.addPermanentWidget(self.user_attention_btn)

        # Add supervised status chip (shown during workflow to indicate supervision mode)
        self._current_task_name = None  # Track current task for supervision toggle
        self.supervised_status_btn = QPushButton("Supervised")
        self.supervised_status_btn.setCursor(Qt.PointingHandCursor) # type: ignore
        self.supervised_status_btn.setToolTip("Click to toggle supervision")
        self.supervised_status_btn.clicked.connect(self._on_supervised_status_clicked)
        self.supervised_status_btn.hide()  # Hidden by default
        self.status_bar.addPermanentWidget(self.supervised_status_btn)

        # Add run workflow button (visible when workflow is not running)
        self.run_workflow_btn = QPushButton("Run Workflow")
        self.run_workflow_btn.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        self.run_workflow_btn.setEnabled(False)
        self.run_workflow_btn.setToolTip("Run the AutoLamella workflow.")
        self.run_workflow_btn.clicked.connect(self._on_run_workflow_clicked)
        self.status_bar.addPermanentWidget(self.run_workflow_btn)

        # Add stop workflow button
        self.stop_workflow_btn = QPushButton("Stop Workflow")
        self.stop_workflow_btn.setStyleSheet(STOP_WORKFLOW_BUTTON_STYLESHEET)
        self.stop_workflow_btn.hide()  # Hidden by default
        self.stop_workflow_btn.setToolTip("Stop the current workflow. You will be asked to confirm.")
        self.stop_workflow_btn.clicked.connect(self._on_stop_workflow_clicked)
        self.status_bar.addPermanentWidget(self.stop_workflow_btn)

    def _on_stop_workflow_clicked(self):
        """Handle stop workflow button click with confirmation."""
        reply = QMessageBox.question(
            self,
            "Stop Workflow",
            "Are you sure you want to stop the workflow?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes and self.autolamella_ui is not None:
            self.autolamella_ui.stop_task_workflow()

    def _on_user_attention_clicked(self):
        """Handle user attention button click - switch to Microscope tab."""
        self.tab_widget.setCurrentIndex(0)  # Microscope tab is index 0

    def _on_run_workflow_clicked(self):
        """Run the workflow using the lamella and task selections from the workflow widget."""
        ui = self.autolamella_ui
        if ui is None:
            return
        if ui.is_workflow_running:
            return
        if ui.microscope is None or ui.experiment is None or ui.experiment.task_protocol is None:
            return

        selected_tasks = self.lamella_workflow_widget.get_selected_tasks()
        selected_lamella = self.lamella_workflow_widget.get_selected_lamella()

        if not selected_tasks or not selected_lamella:
            return

        task_names = [t.name for t in selected_tasks]
        lamella_names = [lam.name for lam in selected_lamella]

        confirm_msg = (
            f"Run workflow for {len(lamella_names)} lamella "
            f"with {len(task_names)} task(s)?\n\n"
        )
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Run Workflow")
        dlg.setText(confirm_msg)
        dlg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        dlg.setDefaultButton(QMessageBox.No)
        dlg.button(QMessageBox.Yes).setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        dlg.button(QMessageBox.No).setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        if dlg.exec_() != QMessageBox.Yes:
            return

        ui._start_run_workflow_thread(task_names, lamella_names)

    def _on_workflow_selection_changed(self, _=None) -> None:
        """Enable the run button only when at least one lamella and one task are selected."""
        n_lam = len(self.lamella_workflow_widget.get_selected_lamella())
        n_task = len(self.lamella_workflow_widget.get_selected_tasks())
        valid = n_lam > 0 and n_task > 0
        self.run_workflow_btn.setEnabled(valid)
        if valid:
            self.run_workflow_btn.setToolTip(
                f"Run workflow: {n_lam} lamella, {n_task} task{'s' if n_task != 1 else ''}"
            )
        else:
            missing = []
            if n_lam == 0:
                missing.append("a lamella")
            if n_task == 0:
                missing.append("a task")
            self.run_workflow_btn.setToolTip(
                f"Select {' and '.join(missing)} to run the workflow"
            )

    def set_workflow_running(self, message: str | None = None):
        """Show stop button and update status message."""
        self.run_workflow_btn.hide()
        self.stop_workflow_btn.show()
        if message and self.status_bar is not None:
            self.status_bar.showMessage(message)

    def hide_workflow_running(self):
        """Hide the stop button and show run button."""
        self.stop_workflow_btn.hide()
        self.supervised_status_btn.hide()
        self.run_workflow_btn.show()

    def _update_supervised_status(self):
        """Update the supervised status chip for the current task."""
        task_name = self._current_task_name
        if task_name is None or self.autolamella_ui is None:
            return
        supervised = get_task_supervision(task_name, self.autolamella_ui)
        if supervised:
            self.supervised_status_btn.setIcon(QIconifyIcon("mdi:account-hard-hat", color="white"))
            self.supervised_status_btn.setText("Supervised")
            self.supervised_status_btn.setToolTip(f"{task_name} is running in supervised mode. Your input will be required. Click to toggle.")
            self.supervised_status_btn.setStyleSheet(SUPERVISION_STATUS_SUPERVISED_STYLESHEET)
        else:
            self.supervised_status_btn.setIcon(QIconifyIcon("mdi:robot", color="white"))
            self.supervised_status_btn.setText("Automated")
            self.supervised_status_btn.setToolTip(f"{task_name} is running in automated mode. Click to toggle.")
            self.supervised_status_btn.setStyleSheet(SUPERVISION_STATUS_AUTOMATED_STYLESHEET)
        self.supervised_status_btn.show()

    def _on_supervised_status_clicked(self):
        """Toggle supervision for the current task in the protocol."""
        if self._current_task_name is None or self.autolamella_ui is None:
            return
        protocol = self.autolamella_ui.protocol
        if protocol is None:
            return
        for task in protocol.workflow_config.tasks:
            if task.name == self._current_task_name:
                task.supervise = not task.supervise
                break
        self._update_supervised_status()
        # Refresh the workflow widget to reflect the toggled supervise state
        if hasattr(self, 'lamella_workflow_widget'):
            self.lamella_workflow_widget.workflow.refresh_all()

    def _update_instructions(self):
        """Update the status bar with the current instruction based on application state."""
        if self.autolamella_ui is None or self.status_bar is None:
            return
        is_connected = self.autolamella_ui.microscope is not None
        experiment = self.autolamella_ui.experiment
        is_experiment_loaded = experiment is not None
        has_positions = is_experiment_loaded and len(experiment.positions) > 0

        if not is_connected:
            msg = INSTRUCTIONS["NOT_CONNECTED"]
        elif not is_experiment_loaded:
            msg = INSTRUCTIONS["NO_EXPERIMENT"]
        elif not has_positions:
            msg = INSTRUCTIONS["NO_LAMELLA"]
        else:
            msg = INSTRUCTIONS["AUTOLAMELLA_READY"]

        self.status_bar.showMessage(msg)

    def _on_microscope_connected(self):
        """Handle microscope connection and connect milling progress signal."""
        if self.autolamella_ui is not None and self.autolamella_ui.microscope is not None:
            self.autolamella_ui.microscope.milling_progress_signal.connect(self._on_milling_progress)
        self.btn_create_experiment.setEnabled(True)
        self.btn_load_experiment.setEnabled(True)
        self._update_instructions()

    @ensure_main_thread
    def _on_milling_progress(self, progress: dict):
        """Handle milling progress updates from the microscope."""
        progress_info = progress.get("progress", None)
        if progress_info is None:
            return

        state = progress_info.get("state", None)

        if state == "start":
            msg = progress.get("msg", "Milling...")
            current_stage = progress_info.get("current_stage", 0)
            total_stages = progress_info.get("total_stages", 1)
            self.milling_progress_bar.setVisible(True)
            self.milling_progress_bar.setValue(0)
            self.milling_progress_bar.setFormat(msg)
            self.milling_progress_bar.setToolTip(f"Milling Stage: {current_stage + 1}/{total_stages}")

        elif state == "update":
            estimated_time = progress_info.get("estimated_time", None)
            remaining_time = progress_info.get("remaining_time", None)

            if remaining_time is not None and estimated_time is not None and estimated_time > 0:
                percent_complete = int((1 - (remaining_time / estimated_time)) * 100)
                self.milling_progress_bar.setValue(percent_complete)
                self.milling_progress_bar.setFormat(
                    f"Milling: {format_duration(remaining_time)} remaining"
                )

        elif state == "finished":
            self.milling_progress_bar.setVisible(False)

    def _on_tab_changed(self, index: int):
        """Handle tab change and update status bar."""
        self.status_bar.setStyleSheet(STATUS_BAR_STYLESHEET)    # type: ignore

    def _create_main_tab(self):
        """Create the main AutoLamella tab."""
        # Create the embedded viewer container
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create napari viewer for main UI
        self.main_viewer = napari.Viewer(show=False, title="AutoLamella Main")
        self.main_viewer.window._qt_window.menuBar().hide()
        self.main_viewer.window._qt_window.statusBar().hide()
        self.viewers.append(self.main_viewer)

        # Create the AutoLamellaUI widget
        self.autolamella_ui = AutoLamellaUI(viewer=self.main_viewer, parent_ui=self)

        # Connect to workflow update signal from AutoLamellaUI
        if self.autolamella_ui is not None:
            self.autolamella_ui.workflow_update_signal.connect(self._on_workflow_update)
            self.autolamella_ui.experiment_update_signal.connect(self._on_experiment_update)
            self.autolamella_ui._workflow_finished_signal.connect(self._on_workflow_finished)
            self.autolamella_ui.system_widget.connected_signal.connect(self._on_microscope_connected)

        # Add it as a dock widget to the viewer
        self.main_viewer.window.add_dock_widget(
            widget=self.autolamella_ui,
            area="right",
            add_vertical_stretch=True,
            name="AutoLamella"
        )

        # hide menu bar
        self.autolamella_ui.menuBar().setVisible(False)

        # Add the viewer's Qt window to our layout
        layout.addWidget(self.main_viewer.window._qt_window)
        self.tab_widget.addTab(container, QIconifyIcon("mdi:microscope", color="#d6d6d6"), "Microscope")

    def create_tabs(self):
        """Create the tabs for the AutoLamella UI."""
        self._create_main_tab()
        self.add_minimap_tab()
        self.add_protocol_editor_tab()
        self.add_lamella_tab()
        self.add_workflow_tab()
        self.add_lamella_cards_tab()

        # add notification button to tab bar
        self.create_notification_button()

    def _on_experiment_update(self):
        """Handle experiment update signal and propagate to tabs."""

        if self.autolamella_ui is None:
            return
        if self.autolamella_ui.experiment is None:
            return

        self.minimap_widget.set_experiment()
        self.task_widget.set_experiment(self.autolamella_ui.experiment)
        self.lamella_widget.set_experiment()
        experiment = self.autolamella_ui.experiment
        if experiment is not None and experiment.task_protocol is not None:
            self.lamella_workflow_widget.set_experiment(experiment)
            self.lamella_workflow_widget.set_workflow_config(experiment.task_protocol.workflow_config)
        # self.task_history_widget.set_experiment(self.autolamella_ui.experiment)

        # Set widget minimum widths (allows resize)
        self.autolamella_ui.setMinimumWidth(500)
        self.task_widget.setMinimumWidth(500)
        self.lamella_widget.setMinimumWidth(500)
        self.lamella_workflow_widget.setMinimumWidth(600)

        # Update experiment name label
        if self.autolamella_ui is not None and self.autolamella_ui.experiment is not None:
            self.experiment_name_label.setText(f"Experiment: {self.autolamella_ui.experiment.name}")
        else:
            self.experiment_name_label.setText("No Experiment")

        # Show run workflow button when experiment is loaded
        self.run_workflow_btn.show()

        # enable all the tabs
        for i in range(self.tab_widget.count()):
            self.tab_widget.setTabEnabled(i, True)

        self._update_instructions()

        # Rebuild lamella list and wire position events for the new experiment
        self._rebuild_lamella_list()
        self._on_workflow_selection_changed()  # evaluate after lamella are populated
        experiment = self.autolamella_ui.experiment if self.autolamella_ui else None
        if experiment is not self._lamella_list_experiment:
            # Disconnect from the old experiment's position events
            if self._lamella_list_experiment is not None:
                try:
                    self._lamella_list_experiment.positions.events.inserted.disconnect(self._rebuild_lamella_list)
                    self._lamella_list_experiment.positions.events.removed.disconnect(self._rebuild_lamella_list)
                except Exception:
                    pass
            # Connect to the new experiment's position events
            if experiment is not None:
                experiment.positions.events.inserted.connect(lambda *_: self._rebuild_lamella_list())
                experiment.positions.events.removed.connect(lambda *_: self._rebuild_lamella_list())
            self._lamella_list_experiment = experiment

    def create_notification_button(self):
        """Add buttons to the tab bar for adding Protocol Editor, Lamella, and Minimap tabs."""
        # Create button container widget
        button_widget = QWidget()
        button_layout = QHBoxLayout(button_widget)
        button_layout.setContentsMargins(5, 0, 5, 0)
        button_layout.setSpacing(5)

        # Create / Load experiment buttons
        self.btn_create_experiment = QPushButton("Create Experiment")
        self.btn_create_experiment.setToolTip("Create a new experiment")
        self.btn_create_experiment.setEnabled(False)
        self.btn_create_experiment.clicked.connect(self._on_new_experiment)

        self.btn_load_experiment = QPushButton("Load Experiment")
        self.btn_load_experiment.setToolTip("Load an existing experiment")
        self.btn_load_experiment.setEnabled(False)
        self.btn_load_experiment.clicked.connect(self._on_load_experiment)

        # Experiment name label
        self.experiment_name_label = QLabel("No Experiment")
        self.experiment_name_label.setStyleSheet("color: #d6d6d6; font-size: 12px;")

        # Notification bell
        self.notification_bell = NotificationBell(self)
        self.toast_manager.set_notification_bell(self.notification_bell)

        # Add widgets to layout
        button_layout.addWidget(self.experiment_name_label)
        button_layout.addWidget(self.btn_create_experiment)
        button_layout.addWidget(self.btn_load_experiment)
        button_layout.addWidget(self.notification_bell)

        # Add to tab widget corner
        self.tab_widget.setCornerWidget(button_widget)

    def add_protocol_editor_tab(self):
        """Add the protocol editor as a separate tab with its own viewer."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create separate napari viewer for protocol editor
        self.editor_viewer = napari.Viewer(show=False, title="Protocol Editor")
        self.editor_viewer.window._qt_window.menuBar().hide()
        self.editor_viewer.window._qt_window.statusBar().hide()
        self.viewers.append(self.editor_viewer)

        # Create the protocol editor widget
        self.task_widget = AutoLamellaProtocolTaskConfigEditor(
            viewer=self.editor_viewer,
            parent=self.autolamella_ui
        )
        self.editor_viewer.window.add_dock_widget(
            self.task_widget,
            area='right',
            name='Protocol Editor'
        )
        self.autolamella_ui.system_widget.connected_signal.connect(self.task_widget._on_microscope_connected)
        layout.addWidget(self.editor_viewer.window._qt_window)
        self.tab_widget.addTab(container, QIconifyIcon("mdi:file-document-edit", color="#d6d6d6"), "Protocol")

        # disable the tab by default
        self.tab_widget.setTabEnabled(self.tab_widget.indexOf(container), False)

    def add_lamella_tab(self):
        """Add the lamella editor as a separate tab with its own viewer."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create separate napari viewer for lamella editor
        self.lamella_viewer = napari.Viewer(show=False, title="Lamella Editor")
        self.lamella_viewer.window._qt_window.menuBar().hide()
        self.lamella_viewer.window._qt_window.statusBar().hide()
        self.viewers.append(self.lamella_viewer)

        # Create the lamella editor widget
        self.lamella_widget = AutoLamellaProtocolEditorWidget(
            viewer=self.lamella_viewer,
            parent=self.autolamella_ui
        )

        # Store reference in the protocol editor widget if it exists
        if hasattr(self.autolamella_ui, 'protocol_editor_widget') and self.autolamella_ui.protocol_editor_widget:
            self.autolamella_ui.protocol_editor_widget.lamella_widget = self.lamella_widget
        self.autolamella_ui.system_widget.connected_signal.connect(self.lamella_widget._on_microscope_connected)

        # Add to viewer dock
        self.lamella_viewer.window.add_dock_widget(
            self.lamella_widget,
            area='right',
            name='Lamella Editor'
        )

        layout.addWidget(self.lamella_viewer.window._qt_window)
        self.tab_widget.addTab(container, QIconifyIcon("mdi:layers", color="#d6d6d6"), "Lamella")

        # disable the tab by default
        index = self.tab_widget.indexOf(container)
        self.tab_widget.setTabEnabled(index, False)

    def add_workflow_tab(self):
        """Add the workflow tab with the combined lamella + workflow widget."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        self.lamella_workflow_widget = LamellaWorkflowWidget()
        self.lamella_workflow_widget.lamella_move_to_requested.connect(self._on_lamella_move_to)
        self.lamella_workflow_widget.lamella_edit_requested.connect(self._on_lamella_edit)
        self.lamella_workflow_widget.lamella_remove_requested.connect(self._on_lamella_remove_requested)
        self.lamella_workflow_widget.lamella_defect_changed.connect(self._on_lamella_defect_changed)

        # Alias so existing methods (_rebuild_lamella_list etc.) keep working unchanged
        self.lamella_list_widget = self.lamella_workflow_widget.lamella_list

        # Workflow task signals — each change persists the updated config to disk
        self.lamella_workflow_widget.task_supervised_changed.connect(self._save_workflow_config)
        self.lamella_workflow_widget.task_edited.connect(self._save_workflow_config)
        self.lamella_workflow_widget.task_remove_requested.connect(self._save_workflow_config)
        self.lamella_workflow_widget.task_order_changed.connect(self._save_workflow_config)
        self.lamella_workflow_widget.task_added.connect(self._save_workflow_config)
        self.lamella_workflow_widget.task_schedule_changed.connect(self._save_workflow_config)

        # Selection signals — update run button enabled state
        self.lamella_workflow_widget.lamella_selection_changed.connect(self._on_workflow_selection_changed)
        self.lamella_workflow_widget.task_selection_changed.connect(self._on_workflow_selection_changed)

        self.workflow_right_panel = QWidget()
        self.workflow_right_panel.setStyleSheet("background: #2b2d31;")

        splitter.addWidget(self.lamella_workflow_widget)
        splitter.addWidget(self.workflow_right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)
        self.tab_widget.addTab(container, QIconifyIcon("mdi:play-circle-outline", color="#d6d6d6"), "Workflow")

        # disable the workflow tab by default
        self.tab_widget.setTabEnabled(self.tab_widget.indexOf(container), False)

        # Track which experiment's position events we're connected to
        self._lamella_list_experiment = None

    def add_lamella_cards_tab(self):
        """Add the lamella card container as a separate tab."""
        from PyQt5.QtWidgets import QScrollArea

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        self.lamella_card_container = LamellaCardContainer()
        self.lamella_card_container.defect_changed.connect(self._on_lamella_defect_changed)

        card_scroll = QScrollArea()
        card_scroll.setWidget(self.lamella_card_container)
        card_scroll.setWidgetResizable(True)
        card_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        layout.addWidget(card_scroll)
        self.tab_widget.addTab(container, QIconifyIcon("mdi:card-multiple-outline", color="#d6d6d6"), "Lamella Cards")

        # disable the tab by default
        self.tab_widget.setTabEnabled(self.tab_widget.indexOf(container), False)

    def _save_workflow_config(self, *_args):
        """Persist the current task list to the experiment protocol after any task change."""
        if self.autolamella_ui is None or self.autolamella_ui.experiment is None:
            return
        protocol = self.autolamella_ui.experiment.task_protocol
        if protocol is None:
            return
        protocol.workflow_config.tasks[:] = self.lamella_workflow_widget.get_tasks()
        self.autolamella_ui._on_workflow_config_changed(protocol.workflow_config)

    def _on_workflow_update(self, info: dict):
        """Handle workflow update signal and update the workflow status bar."""
        status_msg = info.get("status", None)
        if status_msg is not None:
            task_name = status_msg.get("task_name", "Unknown Task")
            lamella_name = status_msg.get("lamella_name", "Unknown Lamella")
            current_lamella_index = status_msg.get("current_lamella_index", None)
            total_lamellae = status_msg.get("total_lamellas", None)
            current_task_index = status_msg.get("current_task_index", None)
            total_tasks = status_msg.get("total_tasks", None)
            error_msg = status_msg.get("error_message", None)
            msg = info.get("msg", "No message")
            status = status_msg.get("status", "info")

            txt = f"Workflow: {task_name} | {lamella_name}"
            if current_task_index is not None and total_tasks is not None:
                txt += f" | Task {current_task_index + 1}/{total_tasks}"
            if current_lamella_index is not None and total_lamellae is not None:
                txt += f" ({current_lamella_index + 1}/{total_lamellae})"

            if current_lamella_index is not None and total_lamellae is not None:
                self.set_workflow_running(txt)

            # update current task
            self._current_task_name = task_name

            # Show toast notification based on status
            msg_type = None
            if status is AutoLamellaTaskStatus.Completed:
                msg_type = "success"
            elif status is AutoLamellaTaskStatus.Failed:
                msg_type = "error"
                msg = error_msg if error_msg is not None else msg
            elif status is AutoLamellaTaskStatus.Skipped:
                msg_type = "warning"

            if msg_type is not None:
                self.show_toast(msg, msg_type)

            # Refresh lamella list and card container
            if hasattr(self, 'lamella_list_widget'):
                self.lamella_list_widget.refresh_all()
                self.lamella_card_container.refresh_all()

        # refresh the supervised status chip
        self._update_supervised_status()

        # Check if waiting for user response and update status bar
        if self.autolamella_ui is not None:
            if self.autolamella_ui.WAITING_FOR_USER_INTERACTION:
                # Show user attention button and change status bar color
                self.user_attention_btn.show()
                # Play notification sound once when entering waiting state
                if not self._user_interaction_sound_played and self._sound_enabled:
                    play_notification_sound()
                    self._user_interaction_sound_played = True
            else:
                # Hide user attention button and reset to original dark theme
                self.user_attention_btn.hide()
                self._user_interaction_sound_played = False  # Reset for next time

    def _rebuild_lamella_list(self):
        """Clear and repopulate the lamella list and card container from the current experiment."""
        if not hasattr(self, 'lamella_list_widget'):
            return
        experiment = self.autolamella_ui.experiment if self.autolamella_ui else None
        self.lamella_list_widget.clear()
        self.lamella_card_container.clear()
        if experiment is None:
            return
        for lamella in experiment.positions:
            self.lamella_list_widget.add_lamella(lamella)
            self.lamella_card_container.add_lamella(lamella)
        self._on_workflow_selection_changed()

    def _on_lamella_move_to(self, lamella):
        """Move the stage to the given lamella's milling position."""
        if self.autolamella_ui is None or self.autolamella_ui.experiment is None:
            return
        try:
            idx = self.autolamella_ui.experiment.positions.index(lamella)
        except ValueError:
            return
        self.autolamella_ui.comboBox_current_lamella.setCurrentIndex(idx)
        self.autolamella_ui.move_to_lamella_position()

    def _on_lamella_edit(self, lamella):
        """Switch to the Lamella tab and select the given lamella in the protocol editor."""
        if self.autolamella_ui is None or self.autolamella_ui.experiment is None:
            return
        try:
            idx = self.autolamella_ui.experiment.positions.index(lamella)
        except ValueError:
            return
        self.autolamella_ui.comboBox_current_lamella.setCurrentIndex(idx)

        # Select the lamella in the protocol editor combobox
        if hasattr(self, "lamella_widget"):
            cb = self.lamella_widget.comboBox_selected_lamella
            for i in range(cb.count()):
                if cb.itemData(i) is lamella or cb.itemData(i).name == lamella.name:
                    cb.setCurrentIndex(i)
                    break

        # Switch to the Lamella tab
        for i in range(self.tab_widget.count()):
            if self.tab_widget.tabText(i) == "Lamella":
                self.tab_widget.setCurrentIndex(i)
                break

    def _on_lamella_defect_changed(self, lamella):
        """Persist defect state change to disk."""
        if self.autolamella_ui is None or self.autolamella_ui.experiment is None:
            return
        self.autolamella_ui.experiment.save()

    def _on_lamella_remove_requested(self, lamella):
        """Remove the given lamella from the experiment after the list widget has already removed its row."""
        if self.autolamella_ui is None or self.autolamella_ui.experiment is None:
            return
        try:
            idx = self.autolamella_ui.experiment.positions.index(lamella)
        except ValueError:
            return
        self.autolamella_ui.experiment.positions.pop(idx)
        self.autolamella_ui.experiment.save()
        self.autolamella_ui.update_lamella_combobox(latest=True)
        self.autolamella_ui.update_ui()

    def _on_workflow_finished(self):
        """Handle workflow finished signal."""
        self.hide_workflow_running()
        self.user_attention_btn.hide()
        if hasattr(self, 'lamella_list_widget'):
            self.lamella_list_widget.refresh_all()
            self.lamella_card_container.refresh_all()
        if self.status_bar is not None:
            self.status_bar.showMessage("Workflow: Finished")
            self.status_bar.setStyleSheet(STATUS_BAR_STYLESHEET)

    def add_minimap_tab(self):
        """Add the minimap as a separate tab with its own viewer."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create separate napari viewer for minimap
        self.minimap_viewer = napari.Viewer(show=False, title="AutoLamella Minimap")
        self.minimap_viewer.window._qt_window.menuBar().hide()
        self.minimap_viewer.window._qt_window.statusBar().hide()
        self.viewers.append(self.minimap_viewer)

        # Create the minimap widget
        self.minimap_widget = FibsemMinimapWidget(
            viewer=self.minimap_viewer,
            parent=self.autolamella_ui
        )

        # Store reference in the parent AutoLamellaUI
        self.autolamella_ui.minimap_widget = self.minimap_widget
        self.autolamella_ui.viewer_minimap = self.minimap_viewer

        # Add to viewer dock
        self.minimap_viewer.window.add_dock_widget(
            self.minimap_widget,
            area='right',
            add_vertical_stretch=True,
            name='AutoLamella Overview'
        )
        layout.addWidget(self.minimap_viewer.window._qt_window)
        self.tab_widget.insertTab(1, container, QIconifyIcon("mdi:map", color="#d6d6d6"), "Overview")

        # disable the tab by default
        self.tab_widget.setTabEnabled(self.tab_widget.indexOf(container), False)

    def closeEvent(self, event):
        """Clean up viewers on close."""
        for viewer in self.viewers:
            try:
                viewer.close()
            except Exception:
                pass
        super().closeEvent(event)


def run_ui():
    """Run the AutoLamella embedded example."""
    app = QApplication.instance() or QApplication(sys.argv)
    window = AutoLamellaSingleWindowUI()
    window.show()
    app.exec_()


if __name__ == "__main__":
    run_ui()
