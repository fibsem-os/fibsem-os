from __future__ import annotations

import logging
import sys
import time
from typing import Optional

try:
    sys.modules.pop("PySide6.QtCore")
except Exception:
    pass

import warnings

import napari
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from superqt import ensure_main_thread
from fibsem.ui.icon import fibsem_icon

import fibsem
from fibsem.versioning import get_version_string
import fibsem.config as fibsem_cfg
from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus, Lamella
from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI, INSTRUCTIONS
from fibsem.applications.autolamella.workflows.tasks.queue import QueueOp, QueueResult
from fibsem.applications.autolamella.workflows.tasks.tasks import get_task_supervision
from fibsem.ui import FibsemMinimapWidget
from fibsem.ui.fm.widgets.fm_overview_widget import FMOverviewWidget
from fibsem.ui.qt.gc import install_main_thread_gc
from fibsem.ui.stylesheets import (
    MILLING_PROGRESS_BAR_STYLESHEET,
    NAPARI_STYLE,
    STOP_WORKFLOW_BUTTON_STYLESHEET,
    SUPERVISION_STATUS_AUTOMATED_STYLESHEET,
    SUPERVISION_STATUS_SUPERVISED_STYLESHEET,
    USER_ATTENTION_BUTTON_STYLESHEET,
    STATUS_BAR_STYLESHEET,
    PRIMARY_BUTTON_STYLESHEET,
    SECONDARY_BUTTON_STYLESHEET,
    GRAY_ICON_COLOR,
    DEFECT_ORANGE_COLOR,
    WORKFLOW_BORDER_STYLESHEET,
)
from fibsem.ui.widgets.progress_widget import FibsemProgressWidget, ProgressUpdate
from fibsem.ui.FibsemSpotBurnWidget import build_spot_burn_progress_update
from fibsem.ui import notification_service
from fibsem.ui.widgets.autolamella_lamella_protocol_editor import (
    AutoLamellaProtocolEditorWidget,
)
from fibsem.ui.widgets.autolamella_task_config_editor import (
    AutoLamellaProtocolTaskConfigEditor,
)
from fibsem.ui.widgets.lamella_card_widget import LamellaCardContainer
from fibsem.ui.widgets.lamella_task_image_widget import LamellaTaskImageWidget
from fibsem.ui.widgets.lamella_workflow_widget import LamellaWorkflowWidget
from fibsem.ui.widgets.notifications import NotificationBell, ToastManager
from fibsem.ui.widgets.workflow_timeline_widget import WorkflowProgressWidget
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


def confirm_run_workflow_dialog(
    lamella_names: list,
    task_names: list,
    parent=None,
) -> bool:
    """Show a confirmation dialog before starting the workflow. Returns True if confirmed."""
    dlg = QDialog(parent)
    dlg.setWindowTitle("Run Workflow")
    dlg.setMinimumWidth(600)

    layout = QVBoxLayout(dlg)
    layout.setSpacing(8)

    msg_label = QLabel(
        f"Run workflow for {len(lamella_names)} lamella with {len(task_names)} task(s)?"
    )
    layout.addWidget(msg_label)

    col_row = QHBoxLayout()
    col_row.setSpacing(8)

    detail_height = min(max(len(lamella_names), len(task_names)) * 20 + 16, 216)

    for heading, items in [("Lamella", lamella_names), ("Tasks", task_names)]:
        col = QVBoxLayout()
        col.setSpacing(4)
        col.addWidget(QLabel(f"<b>{heading}</b>"))
        te = QTextEdit()
        te.setPlainText("\n".join(f"• {n}" for n in items))
        te.setReadOnly(True)
        te.setFixedHeight(detail_height)
        col.addWidget(te)
        col_row.addLayout(col)

    layout.addLayout(col_row)

    btn_row = QHBoxLayout()
    btn_row.addStretch()
    yes_btn = QPushButton("Yes")
    no_btn = QPushButton("No")
    yes_btn.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
    no_btn.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
    yes_btn.clicked.connect(dlg.accept)
    no_btn.clicked.connect(dlg.reject)
    no_btn.setDefault(True)
    btn_row.addWidget(no_btn)
    btn_row.addWidget(yes_btn)
    layout.addLayout(btn_row)

    return dlg.exec_() == QDialog.Accepted


def confirm_add_to_queue_dialog(
    lamella_names: list,
    task_names: list,
    run_next: bool,
    already_queued: list,
    parent=None,
) -> bool:
    """Confirm adding work to a queue that is already running.

    ``already_queued`` is stated rather than acted on: a task that runs twice
    mills twice — the same mechanism that makes "Run again" useful — so the
    consequence is named and the operator decides. Silently adding four of the
    six pairs asked for would be its own surprise.
    """
    dlg = QDialog(parent)
    dlg.setWindowTitle("Add to Queue")
    dlg.setMinimumWidth(600)

    layout = QVBoxLayout(dlg)
    layout.setSpacing(8)

    total = len(lamella_names) * len(task_names)
    where = "to run next" if run_next else "at the end of the queue"
    layout.addWidget(QLabel(
        f"Add {total} task(s) for {len(lamella_names)} lamella {where}?"
    ))

    col_row = QHBoxLayout()
    col_row.setSpacing(8)
    detail_height = min(max(len(lamella_names), len(task_names)) * 20 + 16, 216)
    for heading, items in [("Lamella", lamella_names), ("Tasks", task_names)]:
        col = QVBoxLayout()
        col.setSpacing(4)
        col.addWidget(QLabel(f"<b>{heading}</b>"))
        te = QTextEdit()
        te.setPlainText("\n".join(f"• {n}" for n in items))
        te.setReadOnly(True)
        te.setFixedHeight(detail_height)
        col.addWidget(te)
        col_row.addLayout(col)
    layout.addLayout(col_row)

    if already_queued:
        warning = QLabel(
            f"<b>{len(already_queued)} already queued</b> and will be added again, "
            f"so those tasks will run twice:<br>"
            + "<br>".join(f"• {n}" for n in already_queued[:6])
            + ("<br>• …" if len(already_queued) > 6 else "")
        )
        warning.setWordWrap(True)
        warning.setStyleSheet(f"color: {DEFECT_ORANGE_COLOR};")
        layout.addWidget(warning)

    btn_row = QHBoxLayout()
    btn_row.addStretch()
    yes_btn = QPushButton(f"Add {total} task(s)")
    no_btn = QPushButton("Cancel")
    yes_btn.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
    no_btn.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
    yes_btn.clicked.connect(dlg.accept)
    no_btn.clicked.connect(dlg.reject)
    no_btn.setDefault(True)
    btn_row.addWidget(no_btn)
    btn_row.addWidget(yes_btn)
    layout.addLayout(btn_row)

    return dlg.exec_() == QDialog.Accepted


class AutoLamellaSingleWindowUI(QMainWindow):
    """Main window for AutoLamella UI with embedded napari viewers."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"AutoLamella v{get_version_string()} ")
        self.resize(1600, 1000)

        # Apply napari-style dark theme. Border state rules live here (on the parent)
        # so that setProperty + unpolish/polish on _border_frame re-evaluates them.
        self.setStyleSheet(NAPARI_STYLE + WORKFLOW_BORDER_STYLESHEET)

        # Central tab widget wrapped in a QFrame so the border renders reliably
        self.tab_widget = QTabWidget()
        self._border_frame = QFrame()
        self._border_frame.setObjectName("workflow_border_frame")
        self._border_frame.setProperty("borderState", "idle")
        _frame_layout = QVBoxLayout(self._border_frame)
        _frame_layout.setContentsMargins(0, 0, 0, 0)
        _frame_layout.setSpacing(0)
        _frame_layout.addWidget(self.tab_widget)
        self.setCentralWidget(self._border_frame)

        self.viewers: list[napari.Viewer] = []
        self.autolamella_ui: AutoLamellaUI
        self.minimap_widget: FibsemMinimapWidget
        self.minimap_viewer: napari.Viewer

        # Toast notification manager
        self.toast_manager = ToastManager(self)

        # Load user preferences
        self._preferences = fibsem_cfg.load_user_preferences()
        fibsem_cfg.apply_feature_flags(self._preferences)

        # User attention tracking
        self._user_interaction_sound_played = False  # Track if sound was played
        self._sound_enabled = self._preferences.display.sound_enabled
        self._toasts_enabled = self._preferences.display.toasts_enabled
        self._border_enabled = self._preferences.display.border_enabled
        self.dev_mode = self._preferences.display.dev_mode

        # create menus, status bar, and tabs
        self._create_menu_bar()
        self._create_test_menu()
        self._create_status_bar()
        self.create_tabs()
        self._apply_preferences()
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

        edit_menu = menu_bar.addMenu("Edit")
        if edit_menu is None:
            raise RuntimeError("Failed to create Edit menu in AutoLamella UI.")

        view_menu = menu_bar.addMenu("View")
        if view_menu is None:
            raise RuntimeError("Failed to create View menu in AutoLamella UI.")

        self.action_new_experiment = QAction("New Experiment", self)
        self.action_new_experiment.triggered.connect(self._on_new_experiment)

        self.action_load_experiment = QAction("Load Experiment", self)
        self.action_load_experiment.triggered.connect(self._on_load_experiment)

        self.action_open_experiment_directory = QAction(
            "Open Experiment Directory", self
        )
        self.action_open_experiment_directory.triggered.connect(
            self._on_open_experiment_directory
        )

        self.action_load_protocol = QAction("Load Protocol", self)
        self.action_load_protocol.triggered.connect(self._on_load_protocol)
        self.action_save_protocol = QAction("Save Protocol", self)
        self.action_save_protocol.triggered.connect(self._on_save_protocol)
        self.action_exit = QAction("Exit", self)
        self.action_exit.triggered.connect(self.close)  # type: ignore

        file_menu.addAction(self.action_new_experiment)
        file_menu.addAction(self.action_load_experiment)
        file_menu.addAction(self.action_open_experiment_directory)
        file_menu.addSeparator()
        file_menu.addAction(self.action_load_protocol)
        file_menu.addAction(self.action_save_protocol)
        file_menu.addSeparator()
        file_menu.addAction(self.action_exit)

        # Edit menu
        self.action_preferences = QAction("Preferences...", self)
        self.action_preferences.triggered.connect(self._on_open_preferences)
        edit_menu.addAction(self.action_preferences)

        # View menu
        self.action_show_minimap = QAction("Show Minimap Widget", self)
        self.action_show_minimap.setCheckable(True)
        self.action_show_minimap.setChecked(False)
        self.action_show_minimap.triggered.connect(self._on_toggle_minimap_widget)

        layer_controls_menu = view_menu.addMenu("Show Layer Controls")

        self.action_layer_controls_microscope = QAction("Microscope", self)
        self.action_layer_controls_microscope.setCheckable(True)
        self.action_layer_controls_microscope.setChecked(True)
        self.action_layer_controls_microscope.triggered.connect(
            lambda checked: self._on_toggle_viewer_layer_controls(checked, "microscope")
        )

        self.action_layer_controls_overview = QAction("Overview", self)
        self.action_layer_controls_overview.setCheckable(True)
        self.action_layer_controls_overview.setChecked(True)
        self.action_layer_controls_overview.triggered.connect(
            lambda checked: self._on_toggle_viewer_layer_controls(checked, "overview")
        )

        # No "Lamella Editor" entry: that tab renders on the matplotlib canvas now and
        # has no napari layer docks to show. The submenu goes entirely once the last
        # napari viewer does.
        layer_controls_menu.addAction(self.action_layer_controls_microscope)
        layer_controls_menu.addAction(self.action_layer_controls_overview)

        view_menu.addAction(self.action_show_minimap)

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
        self.action_generate_overview_plot.triggered.connect(
            self._on_generate_overview_plot
        )
        reporting_menu.addAction(self.action_generate_report)
        reporting_menu.addAction(self.action_generate_overview_plot)

        # user scripts (FIB-338). The menu itself is application-agnostic; this
        # supplies only the folder, the context, and how to notify.
        #
        # Built unconditionally and hidden by _apply_preferences when the
        # features.scripts_enabled flag is off, which is the default. Hidden rather
        # than absent so toggling the preference takes effect without a restart, the
        # same way the coincidence viewer and bug reporter do. Constructing the
        # controller costs nothing -- it adds two fixed actions and never touches the
        # scripts folder until the dialog is opened.
        from fibsem.applications.autolamella.scripting import get_scripts_directory
        from fibsem.ui.widgets.script_menu import ScriptMenuController

        self.scripts_menu = tools_menu.addMenu("Scripts")
        if self.scripts_menu is None:
            raise RuntimeError("Failed to create Scripts submenu in AutoLamella UI.")
        self.script_menu_controller = ScriptMenuController(
            menu=self.scripts_menu,
            scripts_directory=get_scripts_directory,
            context_factory=self._script_context,
            notify=self.show_toast,
            parent=self,
        )

        # Below the separator because it is the one entry here that acts on the
        # install rather than on the experiment: read-only, and the thing you open
        # when a Scripts entry above did not appear.
        tools_menu.addSeparator()
        self.action_show_plugins = QAction("Plugins...", self)
        self.action_show_plugins.setToolTip(
            "Show every registered pattern, strategy and task, and where it came from"
        )
        self.action_show_plugins.triggered.connect(self._on_show_plugins)
        tools_menu.addAction(self.action_show_plugins)

        # add help menu
        help_menu = menu_bar.addMenu("Help")
        if help_menu is None:
            raise RuntimeError("Failed to create Help menu in AutoLamella UI.")
        self.action_report_issue = QAction("Report an Issue...", self)
        self.action_report_issue.triggered.connect(self._on_report_issue)
        help_menu.addAction(self.action_report_issue)
        self.action_about = QAction("About", self)
        self.action_about.triggered.connect(self._show_about_dialog)
        help_menu.addAction(self.action_about)

        # add development menu
        dev_menu = menu_bar.addMenu("Development")
        if dev_menu is None:
            raise RuntimeError("Failed to create Development menu in AutoLamella UI.")
        self.action_print_hello = QAction("Print Hello", self)
        self.action_print_hello.triggered.connect(lambda: print("Hello"))
        dev_menu.addAction(self.action_print_hello)

        self._action_coincidence_separator = dev_menu.addSeparator()
        self.action_open_coincidence_viewer = QAction(
            "Open Coincidence Milling Viewer", self
        )
        self.action_open_coincidence_viewer.triggered.connect(
            self._open_coincidence_milling_viewer
        )
        dev_menu.addAction(self.action_open_coincidence_viewer)

        self._dev_menu = dev_menu
        self._dev_menu.menuAction().setVisible(self.dev_mode)

        action_open_fm_minimap = QAction("Open Fluorescence Minimap", self)
        action_open_fm_minimap.triggered.connect(self._open_fm_minimap_widget)
        dev_menu.addAction(action_open_fm_minimap)

        action_open_fm_image_viewer = QAction("Open Fluorescence Image Viewer", self)
        action_open_fm_image_viewer.triggered.connect(self._open_fm_image_viewer)
        dev_menu.addAction(action_open_fm_image_viewer)

        dev_menu.addSeparator()

        action_load_fm_configuration = QAction("Load Fluorescence Configuration", self)
        action_load_fm_configuration.triggered.connect(self._import_fm_configuration)
        dev_menu.addAction(action_load_fm_configuration)

        action_save_fm_configuration = QAction("Save Fluorescence Configuration", self)
        action_save_fm_configuration.triggered.connect(self._export_fm_configuration)
        dev_menu.addAction(action_save_fm_configuration)

        dev_menu.addSeparator()

        self.action_export_targeting_ml_data = QAction(
            "Export Targeting ML Data...", self
        )
        self.action_export_targeting_ml_data.triggered.connect(
            self._on_export_targeting_ml_data
        )
        dev_menu.addAction(self.action_export_targeting_ml_data)

    def _create_test_menu(self):
        """Create a test menu for toast notifications and sounds."""

        self.action_toast_info = QAction("Toast: Info", self)
        self.action_toast_info.triggered.connect(
            lambda: self.show_toast("This is an info message", "info")
        )

        self.action_toast_success = QAction("Toast: Success", self)
        self.action_toast_success.triggered.connect(
            lambda: self.show_toast("Operation completed successfully!", "success")
        )

        self.action_toast_warning = QAction("Toast: Warning", self)
        self.action_toast_warning.triggered.connect(
            lambda: self.show_toast("Warning: Check your settings", "warning")
        )

        self.action_toast_error = QAction("Toast: Error", self)
        self.action_toast_error.triggered.connect(
            lambda: self.show_toast("Error: Something went wrong", "error")
        )

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

        # Border state test actions
        self.action_border_toggle = QAction("Show Workflow Border", self)
        self.action_border_toggle.setCheckable(True)
        self.action_border_toggle.setChecked(self._border_enabled)
        self.action_border_toggle.triggered.connect(self._on_border_toggle)

        self.action_border_automated = QAction("Automated (green)", self)
        self.action_border_automated.triggered.connect(
            lambda: self._set_border_state("automated")
        )

        self.action_border_supervised = QAction("Supervised (blue)", self)
        self.action_border_supervised.triggered.connect(
            lambda: self._set_border_state("supervised")
        )

        self.action_border_waiting = QAction("Waiting for User (orange)", self)
        self.action_border_waiting.triggered.connect(
            lambda: self._set_border_state("waiting")
        )

        self.action_border_idle = QAction("Idle (no border)", self)
        self.action_border_idle.triggered.connect(
            lambda: self._set_border_state("idle")
        )
        self.action_border_agent = QAction("Agent (electric purple)", self)
        self.action_border_agent.triggered.connect(
            lambda: self._set_border_state("agent")
        )

        # add to menu bar
        menu_bar = self.menuBar()
        test_menu = menu_bar.addMenu("Test")  # type: ignore

        toast_menu = test_menu.addMenu("Toast")  # type: ignore
        toast_menu.addAction(self.action_toast_info)  # type: ignore
        toast_menu.addAction(self.action_toast_success)  # type: ignore
        toast_menu.addAction(self.action_toast_warning)  # type: ignore
        toast_menu.addAction(self.action_toast_error)  # type: ignore

        border_menu = test_menu.addMenu("Border State")  # type: ignore
        border_menu.addAction(self.action_border_toggle)  # type: ignore
        border_menu.addSeparator()  # type: ignore
        border_menu.addAction(self.action_border_automated)  # type: ignore
        border_menu.addAction(self.action_border_supervised)  # type: ignore
        border_menu.addAction(self.action_border_waiting)  # type: ignore
        border_menu.addAction(self.action_border_idle)  # type: ignore
        border_menu.addAction(self.action_border_agent)  # type: ignore

        test_menu.addSeparator()  # type: ignore
        test_menu.addAction(self.action_beep)  # type: ignore
        test_menu.addAction(self.action_sound_toggle)  # type: ignore
        test_menu.addAction(self.action_toasts_toggle)  # type: ignore

        self._test_menu = test_menu
        self._test_menu.menuAction().setVisible(self.dev_mode)

    def _on_sound_toggle(self, checked: bool):
        """Handle sound toggle."""
        self._sound_enabled = checked
        self._preferences.display.sound_enabled = checked
        fibsem_cfg.save_user_preferences(self._preferences)

    def _on_toasts_toggle(self, checked: bool):
        """Handle toasts toggle."""
        self._toasts_enabled = checked
        self._preferences.display.toasts_enabled = checked
        fibsem_cfg.save_user_preferences(self._preferences)

    def _on_border_toggle(self, checked: bool):
        """Handle workflow border toggle."""
        self._border_enabled = checked
        self._preferences.display.border_enabled = checked
        fibsem_cfg.save_user_preferences(self._preferences)
        self._set_border_state("idle")

    def _on_open_preferences(self):
        """Open the preferences dialog."""
        from fibsem.ui.widgets.preferences_dialog import PreferencesDialog

        dialog = PreferencesDialog(self._preferences, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            self._preferences = dialog.get_preferences()
            fibsem_cfg.save_user_preferences(self._preferences)
            fibsem_cfg.apply_feature_flags(self._preferences)
            self._apply_preferences()

    def _apply_preferences(self):
        """Apply current preferences to UI state."""
        d = self._preferences.display
        self._sound_enabled = d.sound_enabled
        self._toasts_enabled = d.toasts_enabled
        self._border_enabled = d.border_enabled
        self.dev_mode = d.dev_mode
        # Sync Test menu toggle actions
        self.action_sound_toggle.setChecked(d.sound_enabled)
        self.action_toasts_toggle.setChecked(d.toasts_enabled)
        self.action_border_toggle.setChecked(d.border_enabled)
        # Toggle dev/test menu visibility
        self._dev_menu.menuAction().setVisible(d.dev_mode)
        self._test_menu.menuAction().setVisible(d.dev_mode)
        # Toggle coincidence milling viewer action
        coincidence_enabled = self._preferences.features.coincidence_milling_enabled
        self.action_open_coincidence_viewer.setVisible(coincidence_enabled)
        self._action_coincidence_separator.setVisible(coincidence_enabled)
        # Toggle the "Report an Issue..." Help menu action
        self.action_report_issue.setVisible(
            self._preferences.features.bug_report_enabled
        )
        # Show or hide the FM Overview tab, building or dropping its widget to match
        self._apply_fm_overview_visibility()
        # Toggle Tools -> Scripts. Hiding the menu hides the whole feature: it is the
        # only route to the manager dialog, and the dialog is the only thing that runs
        # a script. If a script is mid-run, leave it visible -- taking away the only
        # Stop button while the microscope is moving would be worse than the flag
        # being briefly wrong.
        self.scripts_menu.menuAction().setVisible(
            self._preferences.features.scripts_enabled
            or self.script_menu_controller.runner.is_running
        )
        # Toggle the per-task schedule button in the workflow tab live, so a
        # scheduled-tasks flag change applies without restarting.
        if hasattr(self, "lamella_workflow_widget"):
            self.lamella_workflow_widget.workflow.enable_schedule_button(
                self._preferences.features.scheduled_tasks
            )

    #### USER SCRIPTS (FIB-338)

    def _script_context(self):
        """Build the context a script runs against, or explain why it cannot.

        Returns (context, reason). A reason means the menu entries are disabled and
        that string is shown instead.
        """
        from fibsem.applications.autolamella.scripting import ScriptContext

        experiment = self.autolamella_ui.experiment
        if experiment is None:
            return None, "Load an experiment to run scripts"
        if self.autolamella_ui.is_workflow_running:
            # tasks mutate lamella state on a worker thread, so a script reading
            # mid-workflow would get a torn snapshot.
            return None, "Unavailable while a workflow is running"

        return ScriptContext(
            experiment=experiment,
            log=logging.info,
            microscope=self.autolamella_ui.microscope,
        ), ""

    def show_toast(
        self,
        message: str,
        notification_type: str = "info",
        duration: int = 5000,
        temporary: bool = False,
    ):
        """Show a toast notification."""
        if self._toasts_enabled:
            self.toast_manager.show_toast(
                message, notification_type, duration, temporary=temporary
            )
        elif self.toast_manager.notification_bell and not temporary:
            # Still log to notification bell even when toasts are disabled
            self.toast_manager.notification_bell.add_notification(
                message, notification_type
            )

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
            self.autolamella_ui._open_experiment_directory()

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
            self.autolamella_ui.open_information_dialog()

    def _on_report_issue(self):
        """Open the Report an Issue dialog."""
        from fibsem.ui.widgets.bug_report_widget import open_bug_report_dialog

        experiment = getattr(self.autolamella_ui, "experiment", None)
        microscope = getattr(self.autolamella_ui, "microscope", None)
        open_bug_report_dialog(
            experiment=experiment, microscope=microscope, parent=self
        )

    def _on_show_plugins(self):
        """Open the read-only listing of registered extensions."""
        from fibsem.ui.widgets.plugins_dialog import PluginsDialog

        PluginsDialog(parent=self).exec_()

    def _on_toggle_minimap_widget(self, checked: bool):
        """Toggle the minimap plot dock widget visibility."""
        if self.autolamella_ui is not None and hasattr(
            self.autolamella_ui, "minimap_plot_dock"
        ):
            self.autolamella_ui.minimap_plot_dock.setVisible(checked)
            self.autolamella_ui.minimap_plot_dock.activateWindow()

    def _on_toggle_viewer_layer_controls(self, checked: bool, viewer_key: str):
        """Toggle the layer list and layer controls for a specific viewer.

        getattr, not attribute access: tabs move off napari one at a time, so a tab that
        has already migrated never sets its viewer attribute. A dict literal would
        dereference all three eagerly and raise AttributeError for *every* entry, not
        just the migrated one — and an exception escaping a Qt slot aborts the process
        under PyQt5 (FIB-329).
        """
        viewer_map = {
            "microscope": getattr(self, "main_viewer", None),
            "overview": getattr(self, "minimap_viewer", None),
        }
        viewer = viewer_map.get(viewer_key)
        if viewer is not None:
            qt_viewer = viewer.window._qt_viewer
            qt_viewer.dockLayerList.setVisible(checked)
            qt_viewer.dockLayerControls.setVisible(checked)

    def _on_generate_report(self):
        """Handle Generate Report action."""
        if self.autolamella_ui is not None:
            self.autolamella_ui.action_generate_report()

    def _on_generate_overview_plot(self):
        """Handle Generate Overview Plot action."""
        if self.autolamella_ui is not None:
            self.autolamella_ui.action_generate_overview_plot()

    def _on_export_targeting_ml_data(self):
        """Handle Export Targeting ML Data action."""
        if self.autolamella_ui is not None:
            self.autolamella_ui.export_targeting_ml_data()

    def _open_fm_minimap_widget(self):
        """Open the Fluorescence Minimap widget."""
        if self.autolamella_ui is not None:
            self.autolamella_ui.open_fm_minimap_widget()

    def _open_fm_image_viewer(self):
        """Open the Fluorescence Image Viewer widget."""
        if self.autolamella_ui is not None:
            self.autolamella_ui._open_fm_image_viewer()

    def _import_fm_configuration(self):
        """Load a fluorescence microscope configuration."""
        if self.autolamella_ui is not None:
            self.autolamella_ui.import_fm_configuration()

    def _export_fm_configuration(self):
        """Save the current fluorescence microscope configuration."""
        if self.autolamella_ui is not None:
            self.autolamella_ui.export_fm_configuration()

    def _open_coincidence_milling_viewer(self):
        """Open the Coincidence Milling Viewer dialog."""
        if self.autolamella_ui is not None:
            self.autolamella_ui._open_coincidence_milling_viewer()

    def _create_status_bar(self):
        """Create the status bar."""
        status_bar = self.statusBar()
        if status_bar is None:
            raise RuntimeError("Failed to create status bar for AutoLamella UI.")
        self.status_bar = status_bar
        self.status_bar.setStyleSheet(STATUS_BAR_STYLESHEET)

        # Add generic progress widget (tile acquisition, etc.)
        self.progress_widget = FibsemProgressWidget(self.status_bar)
        self.progress_widget.setMaximumWidth(400)
        self.status_bar.addPermanentWidget(self.progress_widget)

        # Add milling progress bar
        self.milling_progress_bar = QProgressBar(self.status_bar)
        self.milling_progress_bar.setMaximumWidth(400)
        self.milling_progress_bar.setMaximum(100)
        self.milling_progress_bar.setValue(0)
        self.milling_progress_bar.setTextVisible(True)
        self.milling_progress_bar.setAlignment(Qt.AlignCenter)
        self.milling_progress_bar.setStyleSheet(MILLING_PROGRESS_BAR_STYLESHEET)
        self.milling_progress_bar.hide()  # Hidden by default
        self.status_bar.addPermanentWidget(self.milling_progress_bar)

        # Add user attention button (shown when waiting for user interaction)
        self.user_attention_btn = QPushButton("Attention Required")
        self.user_attention_btn.setStyleSheet(USER_ATTENTION_BUTTON_STYLESHEET)
        self.user_attention_btn.setIcon(
            fibsem_icon("mdi:alert-circle", color=GRAY_ICON_COLOR)
        )
        self.user_attention_btn.hide()  # Hidden by default
        self.user_attention_btn.setToolTip(
            "User Input Required - Click to go to Microscope tab"
        )
        self.user_attention_btn.clicked.connect(self._on_user_attention_clicked)
        self.status_bar.addPermanentWidget(self.user_attention_btn)

        # Add supervised status chip (shown during workflow to indicate supervision mode)
        self._current_task_name = None  # Track current task for supervision toggle
        self.supervised_status_btn = QPushButton("Supervised")
        self.supervised_status_btn.setCursor(Qt.PointingHandCursor)  # type: ignore
        self.supervised_status_btn.setToolTip("Click to toggle supervision")
        self.supervised_status_btn.clicked.connect(self._on_supervised_status_clicked)
        self.supervised_status_btn.hide()  # Hidden by default
        self.status_bar.addPermanentWidget(self.supervised_status_btn)

        # Add run workflow button (visible when workflow is not running)
        self.run_workflow_btn = QPushButton("Run Workflow")
        self.run_workflow_btn.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        self.run_workflow_btn.setIcon(
            fibsem_icon("mdi:play-circle", color=GRAY_ICON_COLOR)
        )
        self.run_workflow_btn.setEnabled(False)
        self.run_workflow_btn.setToolTip("Run the AutoLamella workflow.")
        self.run_workflow_btn.clicked.connect(self._on_run_workflow_clicked)
        self.status_bar.addPermanentWidget(self.run_workflow_btn)

        # Add stop workflow button
        self.stop_workflow_btn = QPushButton("Stop Workflow")
        self.stop_workflow_btn.setStyleSheet(STOP_WORKFLOW_BUTTON_STYLESHEET)
        self.stop_workflow_btn.setIcon(
            fibsem_icon("mdi:stop-circle", color=GRAY_ICON_COLOR)
        )
        self.stop_workflow_btn.hide()  # Hidden by default
        self.stop_workflow_btn.setToolTip(
            "Stop the current workflow. You will be asked to confirm."
        )
        self.stop_workflow_btn.clicked.connect(self._on_stop_workflow_clicked)
        self.status_bar.addPermanentWidget(self.stop_workflow_btn)

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
            self.autolamella_ui.stop_task_workflow()
            self._set_border_state("stopped")

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
        if (
            ui.microscope is None
            or ui.experiment is None
            or ui.experiment.task_protocol is None
        ):
            return

        selected_tasks = self.lamella_workflow_widget.get_selected_tasks()
        selected_lamella = self.lamella_workflow_widget.get_selected_lamella()

        if not selected_tasks or not selected_lamella:
            return

        task_names = [t.name for t in selected_tasks]
        lamella_names = [lam.name for lam in selected_lamella]

        if not confirm_run_workflow_dialog(lamella_names, task_names, parent=self):
            return

        initial_state = "supervised" if selected_tasks[0].supervise else "automated"
        self._set_border_state(initial_state)
        ui._start_run_workflow_thread(task_names, lamella_names)
        # Show the Stop button immediately so the run is cancellable even while
        # waiting for a scheduled first task (before any task status arrives).
        self.set_workflow_running()
        # Clear selections after starting workflow
        self.lamella_workflow_widget.lamella_list._on_select_all(False)
        self.lamella_workflow_widget.workflow._on_select_all(False)

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

        # The timeline's Add button commits the same selection, so it follows the
        # same rule — there is nothing to add until something is ticked.
        if hasattr(self, "workflow_timeline"):
            if valid:
                tip = (f"Add to queue: {n_lam} lamella, "
                       f"{n_task} task{'s' if n_task != 1 else ''}")
            else:
                tip = f"Select {' and '.join(missing)} to add to the queue"
            self.workflow_timeline.set_add_enabled(valid, tip)

    def set_workflow_running(self, message: str | None = None):
        """Show stop button and update status message."""
        self.run_workflow_btn.hide()
        self.stop_workflow_btn.show()
        if message and self.status_bar is not None:
            self.status_bar.showMessage(message)
        self._set_minimap_workflow_enabled(False)
        if hasattr(self, "workflow_timeline"):
            self.workflow_timeline.set_actions_enabled(
                fibsem_cfg.FEATURE_EDIT_RUNNING_QUEUE_ENABLED
            )

    def hide_workflow_running(self):
        """Hide the stop button and show run button."""
        self.stop_workflow_btn.hide()
        self.supervised_status_btn.hide()
        self.run_workflow_btn.show()
        self._set_minimap_workflow_enabled(True)
        # The timeline stays on screen after a run, but there is no longer a
        # queue behind it — offering to reorder one would be a lie.
        if hasattr(self, "workflow_timeline"):
            self.workflow_timeline.set_actions_enabled(False)

    def _set_minimap_workflow_enabled(self, enabled: bool):
        """Enable/disable overview acquisition in both overview tabs during a workflow.

        A running workflow owns the instrument, so neither tab should be able to start
        competing for it. The FM tab keeps its Cancel button live regardless -- see
        `FMOverviewWidget.set_interactive` -- so a lock arriving mid-run cannot strand an
        acquisition with no way to stop it.
        """
        if hasattr(self, "minimap_widget"):
            self.minimap_widget.pushButton_run_tile_collection.setEnabled(enabled)
            self.minimap_widget.pushButton_load_image.setEnabled(enabled)
        if getattr(self, "fm_overview_widget", None) is not None:
            self.fm_overview_widget.set_interactive(enabled)

    def _set_border_state(self, state: str):
        """Update the tab widget border to reflect current workflow state.

        States: 'waiting', 'supervised', 'automated', 'idle'
        """
        if state == getattr(self, "_border_state", None):
            return
        self._border_state = state
        effective = state if self._border_enabled else "idle"
        self._border_frame.setProperty("borderState", effective)
        style = self._border_frame.style()
        if style is not None:
            style.unpolish(self._border_frame)
            style.polish(self._border_frame)
        self._border_frame.update()

    def _update_supervised_status(self) -> bool:
        """Update the supervised status chip for the current task."""
        task_name = self._current_task_name
        if task_name is None or self.autolamella_ui is None:
            return False
        supervised = get_task_supervision(task_name, self.autolamella_ui)
        if supervised:
            self.supervised_status_btn.setIcon(
                fibsem_icon("mdi:account-hard-hat", color="white")
            )
            self.supervised_status_btn.setText("Supervised")
            self.supervised_status_btn.setToolTip(
                f"{task_name} is running in supervised mode. Your input will be required. Click to toggle."
            )
            self.supervised_status_btn.setStyleSheet(
                SUPERVISION_STATUS_SUPERVISED_STYLESHEET
            )
        else:
            self.supervised_status_btn.setIcon(
                fibsem_icon("mdi:lightning-bolt", color="white")
            )
            self.supervised_status_btn.setText("Automated")
            self.supervised_status_btn.setToolTip(
                f"{task_name} is running in automated mode. Click to toggle."
            )
            self.supervised_status_btn.setStyleSheet(
                SUPERVISION_STATUS_AUTOMATED_STYLESHEET
            )
        self.supervised_status_btn.show()

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
                task.supervise = not task.supervise
                break
        supervised = self._update_supervised_status()
        if self.autolamella_ui.is_workflow_running:
            self._set_border_state("supervised" if supervised else "automated")
        # Refresh the workflow widget to reflect the toggled supervise state
        if hasattr(self, "lamella_workflow_widget"):
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
        # Before the signal wiring below, which returns early on a disconnect: the FM
        # overview tab has to hear about that case too, to let go of the old microscope.
        # Through the visibility path rather than straight to the builder, because
        # whether this system has a fluorescence detector at all is only known now.
        self._apply_fm_overview_visibility()
        if (
            self.autolamella_ui is not None
            and self.autolamella_ui.microscope is not None
        ):
            try:
                self.autolamella_ui.microscope.milling_progress_signal.disconnect(
                    self._on_milling_progress
                )
            except Exception:
                pass
            self.autolamella_ui.microscope.milling_progress_signal.connect(
                self._on_milling_progress
            )
            try:
                self.autolamella_ui.microscope.tiled_acquisition_signal.disconnect(
                    self._on_tile_acquisition_progress
                )
            except Exception:
                pass
            self.autolamella_ui.microscope.tiled_acquisition_signal.connect(
                self._on_tile_acquisition_progress
            )
            try:
                self.autolamella_ui.microscope.spot_burn_progress_signal.disconnect(
                    self._on_spot_burn_progress
                )
            except Exception:
                pass
            self.autolamella_ui.microscope.spot_burn_progress_signal.connect(
                self._on_spot_burn_progress
            )
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
            stage_name = progress_info.get("stage_name", f"Stage {current_stage + 1}")
            self.milling_progress_bar.setVisible(True)
            self.milling_progress_bar.setValue(0)
            self.milling_progress_bar.setFormat(msg)
            self.milling_progress_bar.setToolTip(
                f"Milling Stage: {current_stage + 1}/{total_stages} - {stage_name}"
            )

        elif state == "update":
            estimated_time = progress_info.get("estimated_time", None)
            remaining_time = progress_info.get("remaining_time", None)

            if (
                remaining_time is not None
                and estimated_time is not None
                and estimated_time > 0
            ):
                percent_complete = int((1 - (remaining_time / estimated_time)) * 100)
                self.milling_progress_bar.setValue(percent_complete)
                self.milling_progress_bar.setFormat(
                    f"Milling: {format_duration(remaining_time)} remaining"
                )

        elif state == "finished":
            self.milling_progress_bar.setVisible(False)

    @ensure_main_thread
    def _on_spot_burn_progress(self, ddict: dict) -> None:
        """Handle spot burn progress updates from the microscope (supervised + unsupervised)."""
        self.progress_widget.update_progress(build_spot_burn_progress_update(ddict))
        if ddict.get("finished"):
            # hide the Done/Failed state after a moment; reset_if_finished leaves the
            # widget alone if another operation has started rendering progress since
            QTimer.singleShot(2000, self.progress_widget.reset_if_finished)

    @ensure_main_thread
    def _on_tile_acquisition_progress(self, ddict: dict) -> None:
        """Handle tiled acquisition progress updates from the microscope."""
        counter = ddict.get("counter", 0)
        total = ddict.get("total", 1)
        msg = ddict.get("msg", "Collecting tiles")

        if ddict.get("finished"):
            self.progress_widget.update_progress(ProgressUpdate.done())
        elif counter >= total:
            self.progress_widget.update_progress(ProgressUpdate.indeterminate(msg))
        else:
            self.progress_widget.update_progress(
                ProgressUpdate.numeric(counter, total, msg)
            )

    def _on_tile_acquisition_finished(self, result: dict) -> None:
        self.progress_widget.reset()
        tiles = result.get("tiles", 0)
        total = result.get("total", 0)
        elapsed = result.get("elapsed", 0.0)
        cancelled = result.get("cancelled", False)
        error: Exception | None = result.get("error", None)

        tile_info = f"{tiles}/{total} tiles" if total else ""
        elapsed_info = f" in {format_duration(elapsed)}" if elapsed else ""

        if error is not None:
            if cancelled:
                self.show_toast(
                    f"Tile acquisition cancelled. {tile_info} collected.", "warning"
                )
            else:
                self.show_toast(
                    f"Tile acquisition failed. {tile_info} collected. {error}", "error"
                )
        else:
            self.show_toast(
                f"Tile acquisition complete. {tile_info}{elapsed_info}.", "success"
            )

    def _on_tab_changed(self, index: int):
        """Handle tab change and update status bar."""
        self.status_bar.setStyleSheet(STATUS_BAR_STYLESHEET)

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
        self.autolamella_ui.workflow_update_signal.connect(self._on_workflow_update)
        self.autolamella_ui.queue_changed_signal.connect(self._on_queue_changed)
        self.autolamella_ui.step_update_signal.connect(self._on_step_update)
        self.autolamella_ui.experiment_update_signal.connect(self._on_experiment_update)
        self.autolamella_ui._workflow_finished_signal.connect(
            self._on_workflow_finished
        )
        self.autolamella_ui._hook_toast_signal.connect(self.show_toast)
        notification_service._get_service().toast.connect(self._on_notification_service)
        self.autolamella_ui.system_widget.connected_signal.connect(
            self._on_microscope_connected
        )
        self.autolamella_ui.lamella_list.defect_changed.connect(
            self._on_lamella_defect_changed
        )
        self.autolamella_ui.lamella_list.lamella_selected.connect(
            self._on_experiment_lamella_selected
        )

        # hide menu bar
        self.autolamella_ui.menuBar().setVisible(False)
        self.autolamella_ui.setMinimumWidth(550)

        # Layout: napari viewer (left) | autolamella controls (right) via splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        splitter.addWidget(self.main_viewer.window._qt_window)
        splitter.addWidget(self.autolamella_ui)

        splitter.setSizes([700, 550])
        # set minimum width of right panel to 500
        splitter.widget(1).setMinimumWidth(500)
        layout.addWidget(splitter)
        self.tab_widget.addTab(
            container,
            fibsem_icon("mdi:microscope", color=GRAY_ICON_COLOR),
            "Microscope",
        )

    def create_tabs(self):
        """Create the tabs for the AutoLamella UI."""
        self._create_main_tab()
        self.add_minimap_tab()
        self.add_fm_overview_tab()
        self.add_protocol_editor_tab()
        self.add_lamella_editor_tab()
        self.add_workflow_tab()

        # add notification button to tab bar
        self.create_notification_button()

    def _on_experiment_update(self):
        """Handle experiment update signal and propagate to tabs."""

        if self.autolamella_ui is None:
            return
        if self.autolamella_ui.experiment is None:
            return

        self.minimap_widget.set_experiment()
        self._update_fm_overview_experiment()
        self.task_widget.set_experiment(self.autolamella_ui.experiment)
        self.lamella_widget.set_experiment()
        experiment = self.autolamella_ui.experiment
        if experiment is not None and experiment.task_protocol is not None:
            self.lamella_workflow_widget.set_experiment(experiment)
            self.lamella_workflow_widget.set_workflow_config(
                experiment.task_protocol.workflow_config
            )
            self.lamella_workflow_widget.set_options(experiment.task_protocol.options)

        # Set widget minimum widths (allows resize)
        self.autolamella_ui.setMinimumWidth(500)
        self.task_widget.setMinimumWidth(500)
        self.lamella_widget.setMinimumWidth(500)
        self.lamella_workflow_widget.setMinimumWidth(600)

        # Update experiment name label
        self.experiment_name_label.setText(
            f"Experiment: {self.autolamella_ui.experiment.name}"
        )
        self.btn_create_experiment.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        self.btn_load_experiment.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)

        # Show run workflow button when experiment is loaded
        self.run_workflow_btn.show()

        # enable all the tabs (except lamella tab, which is managed by _update_lamella_tab_enabled)
        lamella_tab_index = (
            self.tab_widget.indexOf(self._lamella_tab_container)
            if hasattr(self, "_lamella_tab_container")
            else -1
        )
        # The FM overview tab is enabled by the presence of a fluorescence detector, not
        # by an experiment, so loading one must not switch it on for a system that has
        # no FM -- it would open onto an empty container.
        fm_overview_tab_index = (
            self.tab_widget.indexOf(self._fm_overview_container)
            if getattr(self, "fm_overview_widget", None) is None
            and hasattr(self, "_fm_overview_container")
            else -1
        )
        for i in range(self.tab_widget.count()):
            if i in (lamella_tab_index, fm_overview_tab_index):
                continue
            self.tab_widget.setTabEnabled(i, True)

        self._update_instructions()

        # Rebuild lamella list and wire position events for the new experiment
        self._rebuild_lamella_list()
        self._on_workflow_selection_changed()  # evaluate after lamella are populated
        self.lamella_workflow_widget._update_summary()
        experiment = self.autolamella_ui.experiment if self.autolamella_ui else None
        if experiment is not self._lamella_list_experiment:
            # Disconnect from the old experiment's position events
            if self._lamella_list_experiment is not None:
                try:
                    self._lamella_list_experiment.positions.events.inserted.disconnect(
                        self._rebuild_lamella_list
                    )
                    self._lamella_list_experiment.positions.events.removed.disconnect(
                        self._rebuild_lamella_list
                    )
                except Exception:
                    pass
            # Connect to the new experiment's position events
            if experiment is not None:
                experiment.positions.events.inserted.connect(
                    lambda *_: self._rebuild_lamella_list()
                )
                experiment.positions.events.removed.connect(
                    lambda *_: self._rebuild_lamella_list()
                )
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
        self.btn_create_experiment.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        self.btn_create_experiment.clicked.connect(self._on_new_experiment)

        self.btn_load_experiment = QPushButton("Load Experiment")
        self.btn_load_experiment.setToolTip("Load an existing experiment")
        self.btn_load_experiment.setEnabled(False)
        self.btn_load_experiment.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
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
        container = QWidget(parent=self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create the protocol editor widget (viewer is created internally)
        self.task_widget = AutoLamellaProtocolTaskConfigEditor(
            parent=self.autolamella_ui
        )
        self.autolamella_ui.system_widget.connected_signal.connect(
            self.task_widget._on_microscope_connected
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
        """Consolidated Lamella tab: 1-column card strip (left) + Images/Protocol sub-tabs (right)."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        outer_splitter = QSplitter(Qt.Horizontal)
        outer_splitter.setChildrenCollapsible(True)

        # ── Left: 1-column card strip ──────────────────────────────────────
        self.lamella_card_container = LamellaCardContainer(columns=1)
        self.lamella_card_container.defect_changed.connect(
            self._on_lamella_defect_changed
        )
        self.lamella_card_container.lamella_selected.connect(
            self._on_lamella_card_selected
        )
        self.lamella_card_container.move_to_requested.connect(self._on_lamella_move_to)
        self.lamella_card_container.update_position_requested.connect(
            self._on_lamella_card_update_position
        )
        self.lamella_card_container.remove_requested.connect(
            self._on_lamella_remove_requested
        )

        card_scroll = QScrollArea()
        card_scroll.setWidget(self.lamella_card_container)
        card_scroll.setWidgetResizable(True)
        card_scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )
        card_scroll.setMaximumWidth(340)

        outer_splitter.addWidget(card_scroll)
        outer_splitter.setStretchFactor(0, 0)

        # ── Right: sub-tab widget ──────────────────────────────────────────
        right_tabs = QTabWidget()

        # Review tab
        self.lamella_task_image_widget = LamellaTaskImageWidget()

        # Protocol tab: matplotlib canvas (left) + editor (right). The editor owns its
        # own controller, built eagerly so the splitter can embed the canvas before a
        # microscope connects.
        self.lamella_widget = AutoLamellaProtocolEditorWidget(
            parent=self.autolamella_ui,
        )
        self.autolamella_ui.system_widget.connected_signal.connect(
            self.lamella_widget._on_microscope_connected
        )
        self.lamella_widget.setMinimumWidth(550)

        protocol_splitter = QSplitter(Qt.Horizontal)
        protocol_splitter.setChildrenCollapsible(False)
        protocol_splitter.addWidget(self.lamella_widget.view_controller.widget)
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.lamella_widget)
        scroll_area.setWidgetResizable(True)
        protocol_splitter.addWidget(scroll_area)
        protocol_splitter.setSizes([700, 550])

        right_tabs.addTab(protocol_splitter, "Protocol")
        right_tabs.addTab(self.lamella_task_image_widget, "Review")

        outer_splitter.addWidget(right_tabs)
        outer_splitter.setStretchFactor(1, 1)
        outer_splitter.setSizes([340, 99999])

        layout.addWidget(outer_splitter)
        self.tab_widget.addTab(
            container, fibsem_icon("mdi:layers", color=GRAY_ICON_COLOR), "Lamella"
        )
        self._lamella_tab_container = container

        index = self.tab_widget.indexOf(container)
        self.tab_widget.setTabEnabled(index, False)
        self.tab_widget.setTabToolTip(index, "Add lamella positions to enable this tab")

        self._on_lamella_card_selected(None)

    def add_workflow_tab(self):
        """Add the workflow tab with the combined lamella + workflow widget."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        self.lamella_workflow_widget = LamellaWorkflowWidget()
        self.lamella_workflow_widget.lamella_move_to_requested.connect(
            self._on_lamella_move_to
        )
        self.lamella_workflow_widget.lamella_edit_requested.connect(
            self._on_lamella_edit
        )
        self.lamella_workflow_widget.lamella_remove_requested.connect(
            self._on_lamella_remove_requested
        )
        self.lamella_workflow_widget.lamella_defect_changed.connect(
            self._on_lamella_defect_changed
        )

        # Alias so existing methods (_rebuild_lamella_list etc.) keep working unchanged
        self.lamella_list_widget = self.lamella_workflow_widget.lamella_list

        # Workflow task signals — each change persists the updated config to disk
        self.lamella_workflow_widget.task_supervised_changed.connect(
            self._save_workflow_config
        )
        self.lamella_workflow_widget.task_edited.connect(self._save_workflow_config)
        self.lamella_workflow_widget.task_remove_requested.connect(
            self._save_workflow_config
        )
        self.lamella_workflow_widget.task_order_changed.connect(
            self._save_workflow_config
        )
        self.lamella_workflow_widget.task_added.connect(self._save_workflow_config)

        # Workflow info signals — name/description/options changes also persist
        self.lamella_workflow_widget.workflow_name_changed.connect(
            self._save_workflow_config
        )
        self.lamella_workflow_widget.workflow_description_changed.connect(
            self._save_workflow_config
        )
        self.lamella_workflow_widget.workflow_options_changed.connect(
            self._save_workflow_config
        )

        # Selection signals — update run button enabled state
        self.lamella_workflow_widget.lamella_selection_changed.connect(
            self._on_workflow_selection_changed
        )
        self.lamella_workflow_widget.task_selection_changed.connect(
            self._on_workflow_selection_changed
        )

        self.workflow_right_panel = QWidget()
        self.workflow_right_panel.setStyleSheet("background: #2b2d31;")

        _rp_layout = QVBoxLayout(self.workflow_right_panel)
        _rp_layout.setContentsMargins(0, 0, 0, 0)
        _rp_layout.setSpacing(0)
        self.workflow_timeline = WorkflowProgressWidget()
        self.workflow_timeline.queue_action_requested.connect(self._on_queue_action)
        self.workflow_timeline.add_to_queue_requested.connect(self._on_add_to_queue)
        _rp_layout.addWidget(self.workflow_timeline)

        splitter.addWidget(self.lamella_workflow_widget)
        splitter.addWidget(self.workflow_right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)
        self.tab_widget.addTab(
            container,
            fibsem_icon("mdi:play-circle-outline", color=GRAY_ICON_COLOR),
            "Workflow",
        )

        # disable the workflow tab by default
        self.tab_widget.setTabEnabled(self.tab_widget.indexOf(container), False)

        # Track which experiment's position events we're connected to
        self._lamella_list_experiment = None

        # Guard against bidirectional selection sync loops
        self._syncing_selection = False

        # Connect protocol editor → workflow tab (deferred here since task_widget is created first)
        self.task_widget.workflow_config_changed.connect(
            self.lamella_workflow_widget.set_workflow_config
        )

    def _on_lamella_card_selected(self, lamella: Lamella | None):
        """Update task image panel and protocol editor for selected lamella card."""
        if not hasattr(self, "lamella_task_image_widget"):
            return

        self._selected_card_lamella = lamella
        self.lamella_task_image_widget.set_lamella(lamella)
        if lamella is not None:
            self.lamella_widget.select_lamella(lamella.name)
            if not self._syncing_selection:
                self._syncing_selection = True
                try:
                    self.autolamella_ui.lamella_list.select(lamella.name)
                    if hasattr(self, "minimap_widget"):
                        self.minimap_widget.lamella_list.select(lamella.name)
                finally:
                    self._syncing_selection = False

    def _on_experiment_lamella_selected(self, lamella):
        """Sync card container and minimap when experiment-tab list selection changes."""
        if getattr(self, "_syncing_selection", False) or lamella is None:
            return
        if not hasattr(self, "lamella_card_container"):
            return
        self._syncing_selection = True
        try:
            self.lamella_card_container.select_lamella(lamella.name)
            if hasattr(self, "minimap_widget"):
                self.minimap_widget.lamella_list.select(lamella.name)
        finally:
            self._syncing_selection = False

    def _on_minimap_lamella_selected(self, lamella):
        """Sync experiment list and card container when minimap list selection changes."""
        if getattr(self, "_syncing_selection", False) or lamella is None:
            return
        self._syncing_selection = True
        try:
            self.autolamella_ui.lamella_list.select(lamella.name)
            if hasattr(self, "lamella_card_container"):
                self.lamella_card_container.select_lamella(lamella.name)
        finally:
            self._syncing_selection = False

    def _save_workflow_config(self, *_args):
        """Persist the current task list to the experiment protocol after any task change."""
        if self.autolamella_ui is None or self.autolamella_ui.experiment is None:
            return
        protocol = self.autolamella_ui.experiment.task_protocol
        if protocol is None:
            return
        protocol.workflow_config.tasks[:] = self.lamella_workflow_widget.get_tasks()
        self.autolamella_ui._on_workflow_config_changed(protocol.workflow_config)

    def _on_add_to_queue(self, run_next: bool) -> None:
        """Add the left panel's current selection to the running queue.

        The selection UI is LamellaWorkflowWidget, which stays live during a run
        and is cleared when one is launched — so it is empty and free mid-run.
        Reusing it avoids a second lamella-and-task picker that would have to be
        kept in step with the first.
        """
        manager = getattr(self.autolamella_ui, "_task_manager", None)
        if manager is None:
            return

        lamellae = self.lamella_workflow_widget.get_selected_lamella()
        tasks = self.lamella_workflow_widget.get_selected_tasks()
        if not lamellae or not tasks:
            self._show_queue_message(
                "Select at least one lamella and one task to add to the queue."
            )
            return

        lamella_names = [lam.name for lam in lamellae]
        task_names = [t.name for t in tasks]

        pending = {(i.lamella_name, i.task_name) for i in manager.queue.pending}
        already = [f"{t} for {ln}" for t in task_names for ln in lamella_names
                   if (ln, t) in pending]

        if not confirm_add_to_queue_dialog(lamella_names, task_names, run_next,
                                           already, parent=self):
            return

        # A lamella created before a task joined the protocol has no config for
        # it, and run_task raises on that. Backfill from the base protocol first.
        experiment = self.autolamella_ui.experiment
        missing = [ln for ln, lam in zip(lamella_names, lamellae)
                   if any(t not in lam.task_config for t in task_names)]
        if missing and experiment is not None:
            experiment.apply_lamella_config(missing, task_names)

        # Task-outer, lamella-inner, matching how build_from_matrix lays out the
        # original queue, so added work interleaves the same way.
        #
        # For "run next" each item is anchored after the previous one rather than
        # to the front: front=True on every add would put each new item ahead of
        # the last, reversing the batch.
        added, anchor = 0, None
        for task_name in task_names:
            for lamella_name in lamella_names:
                if not run_next:
                    item = manager.queue.add(lamella_name, task_name)
                elif anchor is None:
                    item = manager.queue.add(lamella_name, task_name, front=True)
                else:
                    item = manager.queue.add(lamella_name, task_name, after=anchor)
                if item is not None:
                    anchor = item.id
                    added += 1

        manager.notify_queue_changed()
        where = "to run next" if run_next else "to the queue"
        self._show_queue_message(f"Added {added} task(s) {where}.")

        # Clear the selection, as starting a workflow does — the work is committed,
        # and leaving it ticked invites adding the same batch twice.
        self.lamella_workflow_widget.lamella_list._on_select_all(False)
        self.lamella_workflow_widget.workflow._on_select_all(False)

    def _on_queue_changed(self, info: dict) -> None:
        """The queue was edited between tasks — no task lifecycle to hang it off."""
        self.workflow_timeline.refresh_queue(info.get("queue_items", []))

    def _on_queue_action(self, action: str, item_id: str) -> None:
        """Apply a timeline row action to the live queue.

        The timeline emits an intent keyed on WorkItem.id and nothing more; the
        queue is reached from here because the widget is generic and has no
        business knowing what a queue is.
        """
        manager = getattr(self.autolamella_ui, "_task_manager", None)
        if manager is None:
            return

        queue = manager.queue
        item = next((i for i in queue.items if i.id == item_id), None)
        if item is None:
            self._show_queue_message("That task is no longer in the queue.")
            return
        label = f"{item.task_name} for {item.lamella_name}"

        if action == "stop_task":
            # Confirmed, unlike Remove: that edits a list, this interrupts an
            # operation already under way on the sample.
            # Built rather than QMessageBox.question(...): the difference from Stop
            # is the point of asking at all, and everything in the text slot renders
            # at question weight. The explanation belongs in the informative slot.
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Question)
            box.setWindowTitle("Stop Task")
            box.setText(f"Stop <b>{label}</b>?")
            box.setInformativeText(
                "It will be marked cancelled and the workflow will carry on with "
                "the next queued task. Use Stop to end the whole run instead."
            )
            box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            box.setDefaultButton(QMessageBox.No)
            if box.exec_() != QMessageBox.Yes:
                return
            # Two halves, as Stop Workflow has: tell the queue this task is being
            # abandoned, and bring the hardware to a halt. Without the second, the
            # task stops waiting for the mill but the mill keeps going.
            manager.stop_task()
            self.autolamella_ui.stop_current_operations()
            # No queue mutation here: the task unwinds on the worker thread and the
            # status it emits on the way out is what updates the timeline.
            self._show_queue_message(f"Stopping {label}...")
            return

        if action == "run_again":
            # A fresh item rather than a rewind: the original attempt still
            # happened and stays in the run record.
            added = queue.add(item.lamella_name, item.task_name, front=True)
            message = (f"Queued {label} to run next." if added is not None
                       else f"Could not queue {label}.")
        else:
            call = {
                "move_up":   lambda: queue.nudge(item_id, -1),
                "move_down": lambda: queue.nudge(item_id, +1),
                "run_next":  lambda: queue.move_to_front(item_id),
                "remove":    lambda: queue.remove(item_id),
            }.get(action)
            if call is None:
                logging.warning(f"Unknown queue action from the timeline: {action}")
                return
            message = self._queue_action_message(action, label, call())

        manager.notify_queue_changed()
        self._show_queue_message(message)

    @staticmethod
    def _queue_action_message(action: str, label: str, result: QueueResult) -> str:
        """Describe the outcome of a queue action, or "" to say nothing.

        The row may have started running between the menu opening and the click,
        which is the whole reason the queue returns a structured result rather
        than a bool — say so instead of appearing to do nothing.
        """
        if result.op is QueueOp.NO_OP:
            return ""  # already first or last; the user can see that
        if result.op is QueueOp.NOT_PENDING:
            return f"{label} is already running."
        if not result.ok:
            return f"{label} is no longer in the queue."
        return {
            "move_up":   f"Moved {label} up.",
            "move_down": f"Moved {label} down.",
            "run_next":  f"{label} will run next.",
            "remove":    f"Removed {label} from the queue.",
        }.get(action, "")

    def _show_queue_message(self, message: str) -> None:
        """Transient confirmation for a queue edit — no modal, no interruption."""
        if message and self.status_bar is not None:
            self.status_bar.showMessage(message, 4000)

    def _on_workflow_update(self, info: dict):
        """Handle workflow update signal and update the workflow status bar."""
        t0 = t1 = time.time()
        timings = {}

        # transient status-bar messages (e.g. scheduled-start countdown)
        status_bar_msg = info.get("status_bar", None)
        if status_bar_msg is not None and self.status_bar is not None:
            self.status_bar.showMessage(status_bar_msg)

        status_msg = info.get("status", None)
        if status_msg is not None:
            # No build-once guard: update_from_status reconciles rows against the
            # snapshot by WorkItem id, so first paint and every later change go
            # through the same path.
            self.workflow_timeline.update_from_status(status_msg)

            task_name = status_msg.get("task_name", "Unknown Task")
            lamella_name = status_msg.get("item_name", "Unknown Lamella")
            queue_position = status_msg.get("queue_position", None)
            queue_total = status_msg.get("queue_total", None)
            error_msg = status_msg.get("error_message", None)
            timestamp = status_msg.get("timestamp", None)
            task_duration = status_msg.get("task_duration", None)
            msg = info.get("msg", "No message")
            status = status_msg.get("status", "info")

            # Position in the live queue, not the launch matrix — stays correct
            # when the queue is added to or reordered mid-run.
            txt = f"Workflow: {task_name} | {lamella_name}"
            if queue_position is not None and queue_total is not None:
                txt += f" | {queue_position}/{queue_total}"

            self.set_workflow_running(txt)
            timings["set_workflow_running"] = time.time() - t1
            t1 = time.time()

            # update current task
            self._current_task_name = task_name

            # Lock editor when the active lamella/task is being processed
            if status is AutoLamellaTaskStatus.InProgress:
                self.lamella_widget.set_active_lamella_name(lamella_name, task_name)
            else:
                self.lamella_widget.set_active_lamella_name(None)

            # Refresh only the affected lamella if we can identify it
            lamella = None
            experiment = self.autolamella_ui.experiment
            if experiment is not None and lamella_name is not None:
                lamella = experiment.get_lamella_by_name(lamella_name)

            if lamella is not None:
                self.lamella_list_widget.refresh_lamella(lamella)
                timings["lamella_list.refresh_lamella"] = time.time() - t1
                t1 = time.time()
                self.lamella_card_container.refresh_lamella(lamella)
                timings["lamella_cards.refresh_lamella"] = time.time() - t1
                t1 = time.time()
            else:
                self.lamella_list_widget.refresh_all()
                timings["lamella_list.refresh_all"] = time.time() - t1
                t1 = time.time()
                self.lamella_card_container.refresh_all()
                timings["lamella_cards.refresh_all"] = time.time() - t1
                t1 = time.time()
            self._on_lamella_card_selected(
                getattr(self, "_selected_card_lamella", None)
            )
            timings["lamella_card_selected"] = time.time() - t1
            t1 = time.time()

        # Check if waiting for user response and update status bar
        if self.autolamella_ui is None:
            return

        # refresh the supervised status chip
        supervised = self._update_supervised_status()

        waiting = self.autolamella_ui.WAITING_FOR_USER_INTERACTION
        if waiting:
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
        t1 = time.time()

        # Update border to reflect current workflow state
        if self._border_state == "stopped":
            pass  # Keep red border until workflow finishes
        elif waiting:
            self._set_border_state("waiting")
        elif self.autolamella_ui.is_workflow_running:
            self._set_border_state("supervised" if supervised else "automated")
        else:
            self._set_border_state("idle")
        timings["set_border_state"] = time.time() - t1

    def _rebuild_lamella_list(self):
        """Clear and repopulate the lamella list and card container from the current experiment."""
        if not hasattr(self, "lamella_list_widget"):
            return
        experiment = self.autolamella_ui.experiment if self.autolamella_ui else None
        self.lamella_list_widget.clear()
        self.lamella_card_container.clear()
        self._on_lamella_card_selected(None)
        if experiment is None:
            return
        for lamella in experiment.positions:
            self.lamella_list_widget.add_lamella(lamella)
            self.lamella_card_container.add_lamella(lamella)
        self._on_workflow_selection_changed()
        self._update_lamella_tab_enabled()

    def _update_lamella_tab_enabled(self):
        """Enable or disable the Lamella tab based on whether positions exist."""
        if not hasattr(self, "_lamella_tab_container"):
            return
        index = self.tab_widget.indexOf(self._lamella_tab_container)
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
        self.autolamella_ui.lamella_list.select(lamella.name)
        self.autolamella_ui.move_to_lamella_position()

    def _on_lamella_card_update_position(self, lamella: "Lamella"):
        """Update the stage position of the given lamella to the current stage position."""
        if self.autolamella_ui is None:
            return
        self.autolamella_ui.lamella_list.select(lamella.name)
        self.autolamella_ui.update_lamella_position_ui()

    def _on_lamella_edit(self, lamella: "Lamella"):
        """Switch to the Lamella tab and select the given lamella in the protocol editor."""
        if self.autolamella_ui is None or self.autolamella_ui.experiment is None:
            return
        self.autolamella_ui.lamella_list.select(lamella.name)

        # Select the lamella in the protocol editor
        self.lamella_widget.select_lamella(lamella.name)

        # Switch to the Lamella tab
        for i in range(self.tab_widget.count()):
            if self.tab_widget.tabText(i) == "Lamella":
                self.tab_widget.setCurrentIndex(i)
                break

    def _on_lamella_defect_changed(self, lamella: "Lamella"):
        """Persist defect state change to disk and sync all widgets."""
        if self.autolamella_ui is None or self.autolamella_ui.experiment is None:
            return
        self.autolamella_ui.experiment.save()
        # Sync defect icon across all widgets
        self.autolamella_ui.lamella_list.refresh_all()
        self.lamella_list_widget.refresh_lamella(lamella)
        self.lamella_card_container.refresh_lamella(lamella)

    def _on_lamella_remove_requested(self, lamella: "Lamella"):
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

    def _on_step_update(self, label: str) -> None:
        """Handle per-step update from the workflow worker thread."""
        self.workflow_timeline.update_step(label)

    def _on_workflow_finished(self, cancelled: bool = False):
        """Handle workflow finished signal."""
        # The next run's items carry new ids, so the timeline rebuilds itself on
        # its first status update — nothing to reset here.
        # Resolve any outer row left in ACTIVE state (e.g. if workflow was cancelled)
        self.workflow_timeline.finish_current_step(failed=cancelled)
        self.workflow_timeline.clear_steps()
        self.hide_workflow_running()
        self.lamella_widget.set_active_lamella_name(None)
        self.user_attention_btn.hide()
        self.lamella_list_widget.refresh_all()
        self.lamella_card_container.refresh_all()
        if self.status_bar is not None:
            self.status_bar.showMessage("Workflow: Finished")
            self.status_bar.setStyleSheet(STATUS_BAR_STYLESHEET)
        self._set_border_state("idle")

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
            viewer=self.minimap_viewer, parent=self.autolamella_ui
        )
        self.minimap_widget.setMinimumWidth(500)
        self.minimap_widget._acquisition_finished.connect(
            self._on_tile_acquisition_finished
        )
        self.minimap_widget.lamella_list.lamella_selected.connect(
            self._on_minimap_lamella_selected
        )

        # Layout: napari viewer (left) | minimap controls (right) via splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        splitter.addWidget(self.minimap_viewer.window._qt_window)
        splitter.addWidget(self.minimap_widget)

        splitter.setSizes([700, 500])
        layout.addWidget(splitter)
        self.tab_widget.insertTab(
            1, container, fibsem_icon("mdi:map", color=GRAY_ICON_COLOR), "Overview"
        )

        # disable the tab by default
        self.tab_widget.setTabEnabled(self.tab_widget.indexOf(container), False)

    def add_fm_overview_tab(self):
        """Reserve the FM Overview tab, beside the beam one.

        Behind `features.fm_overview_tab` in user preferences, off by default: the
        rebuilt fluorescence overview UI is still being finished, and it sits beside the
        existing Overview tab while driving the same instrument.

        The tab is always created and hidden when off, rather than skipped: that makes
        the preference take effect the moment it is changed instead of at the next
        launch. The widget inside is still built and destroyed with the flag -- see
        :meth:`_apply_fm_overview_visibility` -- so a hidden tab costs nothing but the
        container.

        Separate from "Overview" deliberately, not as a step toward merging them: the
        two show different stage poses -- milling pose on the beam side, FM pose here --
        and relating them is what the correlation workflow is for.

        Only the container is made here. `FMOverviewWidget` requires a microscope with a
        fluorescence detector at construction -- it has no meaningful empty state, since
        every scale on its canvas comes from the camera -- and at this point there may
        be no microscope at all. The widget is built on connection instead, by
        :meth:`_build_fm_overview_widget`, and the tab stays disabled until then.
        """
        self._fm_overview_container = QWidget()
        layout = QVBoxLayout(self._fm_overview_container)
        layout.setContentsMargins(0, 0, 0, 0)
        self.fm_overview_widget: Optional[FMOverviewWidget] = None
        # The microscope the current widget was built for. A reconnection hands out a
        # new object, and the old widget would go on reading geometry from an instrument
        # nobody is driving (FIB-433), so the identity is worth keeping.
        self._fm_overview_microscope = None

        self.tab_widget.insertTab(
            2, self._fm_overview_container,
            fibsem_icon("mdi:microscope", color=GRAY_ICON_COLOR), "FM Overview",
        )
        self.tab_widget.setTabEnabled(
            self.tab_widget.indexOf(self._fm_overview_container), False
        )
        self._apply_fm_overview_visibility()

    def _apply_fm_overview_visibility(self):
        """Show or hide the FM Overview tab to match the preference, immediately.

        The widget inside is built and destroyed along with the tab rather than left
        behind hidden. It subscribes to the microscope's stage signal for its lifetime,
        so a hidden one would go on doing work on every poll for a feature that has been
        switched off — and would still be holding a psygnal reference to tear down later.

        Two different absences, deliberately handled differently:

        * **A connected microscope with no fluorescence detector** will never drive this
          tab, so it is hidden. Leaving it visible means a permanently greyed-out tab
          with nothing to say why, on every system without an FM.
        * **No microscope yet** is temporary, so the tab stays visible and disabled,
          which is what every other tab does while waiting for a connection.
        """
        if not hasattr(self, "_fm_overview_container"):
            return
        index = self.tab_widget.indexOf(self._fm_overview_container)
        microscope = (
            self.autolamella_ui.microscope if self.autolamella_ui is not None else None
        )
        has_no_fm = microscope is not None and microscope.fm is None
        self.tab_widget.setTabVisible(
            index, fibsem_cfg.FEATURE_FM_OVERVIEW_TAB_ENABLED and not has_no_fm
        )
        # Builds when the flag is on and there is an FM, tears down otherwise.
        self._build_fm_overview_widget()

    def _build_fm_overview_widget(self):
        """Put a widget in the FM Overview tab, once there is something to drive.

        Called on every connection. Rebuilds when the microscope object has changed,
        because the widget holds its microscope for life and would otherwise be talking
        to the previous one -- destructive of anything on the canvas, and the reason
        FIB-433 exists to do this without throwing the view away.
        """
        # `connected_signal` is subscribed in `_create_main_tab`, which runs before this
        # tab is reserved. Nothing can fire in between synchronously, but the ordering is
        # not this method's to rely on.
        if not hasattr(self, "_fm_overview_container"):
            return

        microscope = (
            self.autolamella_ui.microscope if self.autolamella_ui is not None else None
        )
        index = self.tab_widget.indexOf(self._fm_overview_container)

        if (
            not fibsem_cfg.FEATURE_FM_OVERVIEW_TAB_ENABLED
            or microscope is None
            or microscope.fm is None
        ):
            # Switched off, or no fluorescence detector. The tab stays in place but dead
            # rather than being removed, so the tab bar does not change shape between
            # systems; hiding it is `_apply_fm_overview_visibility`'s job.
            self._teardown_fm_overview_widget()
            self.tab_widget.setTabEnabled(index, False)
            return

        if self.fm_overview_widget is not None:
            if self._fm_overview_microscope is microscope:
                return
            self._teardown_fm_overview_widget()

        try:
            self.fm_overview_widget = FMOverviewWidget(microscope)
        except Exception as e:
            logging.error(f"Could not create the FM overview tab: {e}")
            self.tab_widget.setTabEnabled(index, False)
            return

        self._fm_overview_microscope = microscope
        self._fm_overview_container.layout().addWidget(self.fm_overview_widget)
        self.tab_widget.setTabEnabled(index, True)
        self._update_fm_overview_experiment()

    def _teardown_fm_overview_widget(self):
        """Drop the current FM overview widget, if there is one.

        `close()` rather than `deleteLater()` alone: the widget's `closeEvent` is what
        releases its psygnal subscriptions on the microscope, and those outlive the
        widget -- a stale one left connected emits into a torn-down Qt object.
        """
        if self.fm_overview_widget is None:
            return
        try:
            self.fm_overview_widget.close()
        except Exception as e:
            logging.debug(f"Error closing the FM overview widget: {e}")
        self._fm_overview_container.layout().removeWidget(self.fm_overview_widget)
        self.fm_overview_widget.deleteLater()
        self.fm_overview_widget = None
        self._fm_overview_microscope = None

    def _update_fm_overview_experiment(self):
        """Tell the FM overview tab where to save.

        Told rather than handed the experiment: the widget knows nothing about
        experiments, and keeping it that way is what lets it open standalone against a
        simulator and be constructed in a test from a microscope alone.
        """
        if getattr(self, "fm_overview_widget", None) is None:
            return
        experiment = (
            self.autolamella_ui.experiment if self.autolamella_ui is not None else None
        )
        self.fm_overview_widget.set_save_directory(
            str(experiment.path) if experiment is not None else None
        )

    def _on_notification_service(
        self, message: str, notification_type: str, temporary: bool
    ) -> None:
        self.show_toast(message, notification_type, temporary=temporary)

    def closeEvent(self, event):
        """Clean up viewers on close."""
        # persist the FM working state (channels / camera transform / objective)
        if self.autolamella_ui is not None and self.autolamella_ui.fm_control_widget is not None:
            try:
                self.autolamella_ui.fm_control_widget.save_fm_configuration()
            except Exception as e:
                logging.warning(f"Could not save FM working state on close: {e}")
        try:
            notification_service._get_service().toast.disconnect(
                self._on_notification_service
            )
        except RuntimeError:
            pass
        for viewer in self.viewers:
            try:
                viewer.close()
            except Exception:
                pass
        super().closeEvent(event)
        # Force the event loop to exit even if another top-level window (e.g. a stray
        # matplotlib pyplot figure) would otherwise keep the app alive once this window
        # closes. quit() — not os._exit — so atexit / experiment autosave still run.
        app = QApplication.instance()
        if app is not None:
            app.quit()


def _start_update_check() -> None:
    """Ask PyPI about newer releases on a worker thread, once, at startup.

    Startup rather than on dialog open: the about dialog reads the cached result,
    so it stays instant and offline-safe by construction, and the latency lands
    where nothing is waiting on it. A no-op for source installs and when the user
    has opted out, so the common developer case makes no network call at all.
    """
    from fibsem import update_check
    from fibsem.ui.qt.threading import thread_worker

    if not update_check.is_enabled():
        return

    @thread_worker
    def _check() -> None:
        update_check.refresh()

    _check().start()


def run_ui():
    """Run the AutoLamella embedded example."""
    import faulthandler
    import signal

    from fibsem.applications.autolamella.tools.bug_report import init_sentry

    # Ctrl+C: PyQt never runs Python's SIGINT handler while blocked in the C++ event
    # loop — and if the GUI thread wedges, no Python runs at all — so restore the OS
    # default disposition: SIGINT then terminates the process at the kernel level,
    # making the app always killable with Ctrl+C. (A hard stop, not a graceful one;
    # use Cmd+Q / File→Exit for a clean shutdown when the loop is responsive.)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    # Diagnostics: dump every thread's Python stack on a fatal signal, and on demand
    # via `kill -USR1 <pid>` — the fastest way to see *where* a frozen GUI thread is
    # stuck (it does not terminate the process, so you can dump repeatedly).
    faulthandler.enable()
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, all_threads=True)

    init_sentry()  # inert unless crash reporting is enabled in preferences
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    # Cyclic garbage must only ever be collected on this thread: worker-thread GC
    # finalizes Qt/vispy objects off the GUI thread and hard-crashes Windows GL
    # drivers (access violation in glDrawArrays). See fibsem/ui/qt/gc.py.
    gc_collector = install_main_thread_gc(parent=app)  # noqa: F841 — kept alive for app lifetime
    window = AutoLamellaSingleWindowUI()
    window.show()
    _start_update_check()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_ui()
