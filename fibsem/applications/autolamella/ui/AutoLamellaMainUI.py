from __future__ import annotations

import logging
import os
import sys
from dataclasses import replace
from typing import List, Optional, Tuple

try:
    sys.modules.pop("PySide6.QtCore")
except Exception:
    pass

from datetime import datetime

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QStackedWidget,
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
from fibsem.applications.autolamella.ui.attention import (
    AttentionRouter,
    play_notification_sound,
)
from fibsem.applications.autolamella.ui.autolamella_lamella_protocol_editor import (
    AutoLamellaProtocolEditorWidget,
)
from fibsem.applications.autolamella.ui.autolamella_task_config_editor import (
    AutoLamellaProtocolTaskConfigEditor,
)
from fibsem.applications.autolamella.ui.AutoLamellaUI import INSTRUCTIONS, AutoLamellaUI
from fibsem.applications.autolamella.ui.experiment_header import ExperimentHeader
from fibsem.applications.autolamella.ui.grids_tab_widget import GridsTabWidget
from fibsem.applications.autolamella.ui.lamella_card_widget import LamellaCardContainer
from fibsem.applications.autolamella.ui.lamella_task_image_widget import (
    LamellaTaskImageWidget,
)
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
from fibsem.structures import BeamType
from fibsem.tools.apps import AUTOLAMELLA, App, apps
from fibsem.ui import notification_service
from fibsem.ui.agent.chip_host import AgentChipHost
from fibsem.ui.icon import fibsem_icon
from fibsem.ui.stylesheets import (
    GRAY_ICON_COLOR,
    NAPARI_STYLE,
    STATUS_BAR_STYLESHEET,
    border_stylesheet,
)
from fibsem.ui.tokens import TEXT_MUTED_COLOR
from fibsem.ui.widgets.canvas.quad_view import MicroscopeViewController
from fibsem.ui.widgets.coincidence_menu import CoincidenceMenu
from fibsem.ui.widgets.connection_dialog import connect_to_microscope_dialog
from fibsem.ui.widgets.notifications import ToastManager
from fibsem.ui.widgets.run_border import RunBorder, run_border_state
from fibsem.ui.widgets.status_bar import FibsemStatusBar
from fibsem.versioning import get_version_string
from fibsem.workflows.attention import Attention
from fibsem.workflows.grid.naming import grid_rename_refusal, rename_grid
from fibsem.workflows.runner import GRIDS, RunRequest
from fibsem.workflows.status import (
    TaskStatus,
    WorkflowStatusEvent,
    WorkflowStatusUpdate,
)


def set_menu_icon(action: QAction, key: str) -> None:
    """Give a menu action an icon that actually renders.

    Qt turns menu icons off wholesale on macOS (AA_DontShowIconsInMenus), so an
    icon set the usual way shows on Windows and Linux and silently vanishes here.
    """
    action.setIcon(fibsem_icon(key, color=GRAY_ICON_COLOR))
    action.setIconVisibleInMenu(True)


class AutoLamellaSingleWindowUI(QMainWindow):
    """Main window for AutoLamella UI.

    No napari viewer of its own any more: every tab here renders on the matplotlib
    canvas. napari survives in this application only in labelling and segmentation,
    which open windows of their own (FIB-407).
    """

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

        # Read once, here, because both the menu entry and the header chip are gated
        # on it and the menu is built before the tabs are.
        self._connection_chip_enabled = self._preferences.features.connection_chip

        self._sound_enabled = self._preferences.display.sound_enabled
        self.run_border.set_on(self._preferences.display.border_enabled)
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

        # These three carry icons because they are also the tab-corner experiment
        # menu (see ExperimentHeader), where the icons do the work of
        # telling create from load at a glance. Qt hides action icons in menus on
        # macOS unless each action asks for them, hence set_menu_icon.
        # Connecting had no menu entry at all -- the Connection tab was the only
        # door, which is what made that tab impossible to remove (FIB-775). Built
        # unconditionally, offered only when the feature flag is on: see below.
        self.action_connect_microscope = QAction("Connect to Microscope...", self)
        set_menu_icon(self.action_connect_microscope, "mdi:connection")
        self.action_connect_microscope.triggered.connect(self._on_connect_microscope)

        self.action_new_experiment = QAction("New Experiment", self)
        set_menu_icon(self.action_new_experiment, "mdi:plus")
        self.action_new_experiment.triggered.connect(
            lambda: self.autolamella_ui.create_experiment()
        )

        self.action_load_experiment = QAction("Load Experiment", self)
        set_menu_icon(self.action_load_experiment, "mdi:folder-open")
        self.action_load_experiment.triggered.connect(
            lambda: self.autolamella_ui.load_experiment()
        )

        self.action_open_experiment_directory = QAction(
            "Open Experiment Directory", self
        )
        set_menu_icon(self.action_open_experiment_directory, "mdi:folder")
        self.action_open_experiment_directory.triggered.connect(
            lambda: self.autolamella_ui._open_experiment_directory()
        )

        self.action_export_image = QAction("Export Image...", self)
        set_menu_icon(self.action_export_image, "mdi:export")
        self.action_export_image.triggered.connect(self._on_export_image)

        self.action_load_protocol = QAction("Load Protocol", self)
        self.action_load_protocol.triggered.connect(
            lambda: self.autolamella_ui.load_protocol()
        )
        self.action_save_protocol = QAction("Save Protocol", self)
        self.action_save_protocol.triggered.connect(
            lambda: self.autolamella_ui.export_protocol_ui()
        )
        self.action_exit = QAction("Exit", self)
        self.action_exit.triggered.connect(self.close)  # type: ignore

        # Behind the same flag as the chip. The action is built either way -- it is
        # the thing the tab removal will depend on -- but nothing offers it until
        # the feature is switched on, so a release carrying this changes nothing.
        if self._connection_chip_enabled:
            file_menu.addAction(self.action_connect_microscope)
            file_menu.addSeparator()
        file_menu.addAction(self.action_new_experiment)
        file_menu.addAction(self.action_load_experiment)
        file_menu.addAction(self.action_open_experiment_directory)
        file_menu.addSeparator()
        file_menu.addAction(self.action_export_image)
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
        # Quad-view display controls. The F5/Esc shortcuts live on these QActions — one
        # source of truth for the menu item and its keybinding (Qt renders the shortcut
        # text in the menu automatically). They are scoped to the Microscope tab via
        # setShortcutContext + container.addAction (see _create_main_tab), so they only
        # fire when focus is inside that tab.
        view_menu.addSeparator()
        self.action_toggle_fullscreen = QAction("Full Screen", self)
        self.action_toggle_fullscreen.setCheckable(True)
        self.action_toggle_fullscreen.setShortcut(QKeySequence(Qt.Key_F5))
        self.action_toggle_fullscreen.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.action_toggle_fullscreen.triggered.connect(self._hotkey_toggle_fullscreen)
        view_menu.addAction(self.action_toggle_fullscreen)

        self.action_exit_fullscreen = QAction("Exit Full Screen", self)
        self.action_exit_fullscreen.setShortcut(QKeySequence(Qt.Key_Escape))
        self.action_exit_fullscreen.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.action_exit_fullscreen.triggered.connect(self._hotkey_exit_fullscreen)
        view_menu.addAction(self.action_exit_fullscreen)

        fullscreen_menu = view_menu.addMenu("Full Screen View")
        for label, key in (
            ("SEM", BeamType.ELECTRON),
            ("FIB", BeamType.ION),
            ("Fluorescence", "fm"),
        ):
            act = QAction(label, self)
            act.triggered.connect(
                lambda _checked=False, k=key: self.view_controller.set_fullscreen(k)
            )
            fullscreen_menu.addAction(act)

        # The standalone viewer, where a multi-channel image is composed: the
        # Microscope tab's fluorescence view shows one frame (FIB-942). Needs no
        # experiment; it is a file viewer.
        view_menu.addSeparator()
        self.action_open_fm_image_viewer = QAction("Fluorescence Image Viewer...", self)
        self.action_open_fm_image_viewer.triggered.connect(
            lambda: self.autolamella_ui._open_fm_image_viewer()
        )
        view_menu.addAction(self.action_open_fm_image_viewer)

        # keep the checkable / enabled state honest each time the menu opens
        view_menu.aboutToShow.connect(self._sync_view_menu)

        # Imaging hotkeys (Microscope tab; scoped via container.addAction). Live / auto
        # contrast / auto focus act on the selected beam — the view <-> radio sync keeps
        # dual_beam_widget.beam_type aligned with the selected view.
        imaging_menu = menu_bar.addMenu("Imaging")
        if imaging_menu is None:
            raise RuntimeError("Failed to create Imaging menu in AutoLamella UI.")
        self._imaging_actions: list = []  # added to the Microscope-tab container for scoping
        for label, key, handler in (
            ("Acquire", Qt.Key_F2, self._hotkey_acquire),
            ("Live Acquisition", Qt.Key_F6, self._hotkey_toggle_live),
            ("Auto Contrast", Qt.Key_F9, self._hotkey_autocontrast),
            ("Auto Focus", Qt.Key_F11, self._hotkey_autofocus),
        ):
            action = QAction(label, self)
            action.setShortcut(QKeySequence(key))
            action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
            action.triggered.connect(handler)
            imaging_menu.addAction(action)
            self._imaging_actions.append(action)

        # add tools menu, reporting submenu
        tools_menu = menu_bar.addMenu("Tools")
        if tools_menu is None:
            raise RuntimeError("Failed to create Tools menu in AutoLamella UI.")
        reporting_menu = tools_menu.addMenu("Reporting")
        if reporting_menu is None:
            raise RuntimeError("Failed to create Reporting submenu in AutoLamella UI.")
        self.action_generate_report = QAction("Generate Report", self)
        self.action_generate_report.triggered.connect(
            lambda: self.autolamella_ui.action_generate_report()
        )
        # Report v2 (FIB-1036): an HTML page built from events.jsonl. Offered
        # beside the PDF report until it has been checked against real runs.
        self.action_generate_report_v2 = QAction("Generate Report v2 (preview)", self)
        self.action_generate_report_v2.setToolTip(
            "Write an HTML report from the experiment's events.jsonl under its "
            "folder, and open it"
        )
        self.action_generate_report_v2.triggered.connect(self._on_generate_report_v2)
        self.action_generate_overview_plot = QAction("Generate Overview Plot", self)
        self.action_generate_overview_plot.triggered.connect(
            lambda: self.autolamella_ui.action_generate_overview_plot()
        )
        # The grid screening report (FIB-1057): what the Grids tab holds, as a PDF.
        self.action_generate_grid_report = QAction(
            "Generate Grid Screening Report", self
        )
        self.action_generate_grid_report.setToolTip(
            "Write a PDF of every grid's overviews and verdicts under the "
            "experiment folder, and open it"
        )
        self.action_generate_grid_report.triggered.connect(
            self._on_generate_grid_report
        )
        reporting_menu.addAction(self.action_generate_report)
        reporting_menu.addAction(self.action_generate_report_v2)
        reporting_menu.addAction(self.action_generate_grid_report)
        reporting_menu.addAction(self.action_generate_overview_plot)

        # user scripts (FIB-338). The menu itself is application-agnostic; this
        # supplies only the folder, the context, and how to notify.
        #
        # Constructing the controller costs nothing -- it adds two fixed actions and
        # never touches the scripts folder until the dialog is opened.
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

        # The embedded App Server's dialog: whether it runs, its services, the
        # dashboard. The agent's level and connecting one are the agent chip's
        # (FIB-1035). Always visible: it is where the server is turned on. Next
        # to Scripts (FIB-863).
        self.action_app_server = QAction("App Server...", self)
        self.action_app_server.setMenuRole(QAction.NoRole)
        self.action_app_server.setToolTip(
            "The App Server: whether it is running, its services, and the dashboard"
        )
        self.action_app_server.triggered.connect(self._on_app_server_dialog)
        tools_menu.addAction(self.action_app_server)

        # Below the separator because these act on the install rather than on the
        # experiment. The wizard needs a home here as well as on the connection tab:
        # the callout there appears once and is dismissible, so without this entry a
        # second microscope could never be set up through it.
        tools_menu.addSeparator()
        self.action_guided_setup = QAction("Guided Setup...", self)
        # NoRole stays even though "Guided Setup..." clears the trap by itself. Qt's
        # default MenuRole is TextHeuristicRole, and the heuristic moves any action
        # whose text *starts with* "about", "config", "preference", "options",
        # "setting" or "setup" into the application menu. That is how the earlier name,
        # "Setup Wizard...", displaced Edit -> Preferences and opened this instead --
        # nothing about the connection was wrong, the action was simply no longer in
        # the menu it had been added to. This name is safe; the next one might not be.
        self.action_guided_setup.setMenuRole(QAction.NoRole)
        self.action_guided_setup.setToolTip(
            "A guided walkthrough that configures fibsemOS to work with your microscope"
        )
        self.action_guided_setup.triggered.connect(self._on_guided_setup)
        tools_menu.addAction(self.action_guided_setup)

        self.action_show_plugins = QAction("Plugins...", self)
        self.action_show_plugins.setToolTip(
            "Show every registered pattern, strategy and task, and where it came from"
        )
        self.action_show_plugins.triggered.connect(self._on_show_plugins)
        tools_menu.addAction(self.action_show_plugins)

        # One entry per app the launcher knows (fibsem.tools.apps), plugins included.
        shortcut_menu = tools_menu.addMenu("Create Desktop Shortcut")
        self.shortcut_actions = {}
        for key, app in apps().items():
            action = QAction(f"{app.name}...", self)
            action.setToolTip(
                f"Place a shortcut on your desktop that launches {app.name} from "
                "this environment"
            )
            action.triggered.connect(
                lambda _checked=False, app=app: self._on_create_desktop_shortcut(app)
            )
            shortcut_menu.addAction(action)
            self.shortcut_actions[key] = action
        self.action_create_desktop_shortcut = self.shortcut_actions["autolamella"]

        # add help menu
        help_menu = menu_bar.addMenu("Help")
        if help_menu is None:
            raise RuntimeError("Failed to create Help menu in AutoLamella UI.")
        self.action_report_issue = QAction("Report an Issue...", self)
        self.action_report_issue.triggered.connect(self._on_report_issue)
        help_menu.addAction(self.action_report_issue)
        self.action_about = QAction("About", self)
        self.action_about.triggered.connect(
            lambda: self.autolamella_ui.open_information_dialog()
        )
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

        self.coincidence_menu = CoincidenceMenu(
            dev_menu,
            microscope=lambda: getattr(self.autolamella_ui, "microscope", None),
            experiment_path=lambda: getattr(
                getattr(self.autolamella_ui, "experiment", None), "path", None
            ),
            show_toast=self.show_toast,
            parent=self,
        )

        self._dev_menu = dev_menu
        self._dev_menu.menuAction().setVisible(self.dev_mode)

        action_load_fm_configuration = QAction("Load Fluorescence Configuration", self)
        action_load_fm_configuration.triggered.connect(
            lambda: self.autolamella_ui.import_fm_configuration()
        )
        dev_menu.addAction(action_load_fm_configuration)

        action_save_fm_configuration = QAction("Save Fluorescence Configuration", self)
        action_save_fm_configuration.triggered.connect(
            lambda: self.autolamella_ui.export_fm_configuration()
        )
        dev_menu.addAction(action_save_fm_configuration)

        dev_menu.addSeparator()

        self.action_export_targeting_ml_data = QAction(
            "Export Targeting ML Data...", self
        )
        self.action_export_targeting_ml_data.triggered.connect(
            lambda: self.autolamella_ui.export_targeting_ml_data()
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

        # Border state test actions
        self.action_border_toggle = QAction("Show Workflow Border", self)
        self.action_border_toggle.setCheckable(True)
        self.action_border_toggle.setChecked(self.run_border.on)
        self.action_border_toggle.triggered.connect(self._on_border_toggle)

        self.action_border_automated = QAction("Automated (green)", self)
        self.action_border_automated.triggered.connect(
            lambda: self.run_border.set_state("automated")
        )

        self.action_border_supervised = QAction("Supervised (blue)", self)
        self.action_border_supervised.triggered.connect(
            lambda: self.run_border.set_state("supervised")
        )

        self.action_border_waiting = QAction("Waiting for User (orange)", self)
        self.action_border_waiting.triggered.connect(
            lambda: self.run_border.set_state("waiting")
        )
        self.action_border_pending = QAction("Pending (grey)", self)
        self.action_border_pending.triggered.connect(
            lambda: self.run_border.set_state("pending")
        )
        self.action_border_stopping = QAction("Stopping (red)", self)
        self.action_border_stopping.triggered.connect(
            lambda: self.run_border.set_state("stopping")
        )

        self.action_border_idle = QAction("Idle (no border)", self)
        self.action_border_idle.triggered.connect(
            lambda: self.run_border.set_state("idle")
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
        border_menu.addAction(self.action_border_pending)  # type: ignore
        border_menu.addAction(self.action_border_stopping)  # type: ignore
        border_menu.addAction(self.action_border_idle)  # type: ignore

        test_menu.addSeparator()  # type: ignore
        test_menu.addAction(self.action_beep)  # type: ignore
        test_menu.addAction(self.action_sound_toggle)  # type: ignore

        self._test_menu = test_menu
        self._test_menu.menuAction().setVisible(self.dev_mode)

    def _on_sound_toggle(self, checked: bool):
        """Handle sound toggle."""
        self._sound_enabled = checked
        # Through the file, not this window's copy: another widget may have saved a
        # setting of its own since start-up (the view bars' fields, FIB-1186).
        self._preferences = fibsem_cfg.update_user_preferences(
            lambda p: setattr(p.display, "sound_enabled", checked)
        )

    def _on_border_toggle(self, checked: bool):
        """Handle workflow border toggle."""
        self._preferences = fibsem_cfg.update_user_preferences(
            lambda p: setattr(p.display, "border_enabled", checked)
        )
        self.run_border.set_on(checked)

    def _on_open_preferences(self):
        """Open the preferences dialog."""
        from fibsem.ui.widgets.preferences_dialog import PreferencesDialog

        # Fresh from disk, so saving the dialog's result does not put back anything
        # another widget has changed since start-up.
        self._preferences = fibsem_cfg.load_user_preferences()
        dialog = PreferencesDialog(self._preferences, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            self._preferences = dialog.get_preferences()
            fibsem_cfg.save_user_preferences(self._preferences)
            self._apply_preferences()

    def _apply_preferences(self):
        """Apply current preferences to UI state."""
        # The embedded App Server follows its saved switch (set in Tools → App
        # Server) against the connected microscope.
        if getattr(self, "autolamella_ui", None) is not None:
            self.autolamella_ui.sync_app_server_with_preference()
        d = self._preferences.display
        self._sound_enabled = d.sound_enabled
        self.run_border.set_on(d.border_enabled)
        self.dev_mode = d.dev_mode
        # The lamella strip's density. Guarded because _apply_preferences also runs
        # before the Lamella tab is built.
        if hasattr(self, "lamella_card_container"):
            self.lamella_card_container.set_mode(d.lamella_card_mode)
        # Sync Test menu toggle actions
        self.action_sound_toggle.setChecked(d.sound_enabled)
        self.action_border_toggle.setChecked(d.border_enabled)
        # Toggle dev/test menu visibility
        self._dev_menu.menuAction().setVisible(d.dev_mode)
        self._test_menu.menuAction().setVisible(d.dev_mode)
        # Toggle coincidence milling viewer action
        coincidence_enabled = self._preferences.features.coincidence_milling_enabled
        self.action_open_coincidence_viewer.setVisible(coincidence_enabled)
        self._action_coincidence_separator.setVisible(coincidence_enabled)
        # getattr because this also runs from the preferences dialog, and the tab it
        # reaches into is built by AutoLamellaUI rather than here.
        system_widget = getattr(self.autolamella_ui, "system_widget", None)
        if system_widget is not None:
            system_widget.refresh_first_run_offer(self._preferences)

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
        if self.autolamella_ui.session.runner.is_running:
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

    def _on_connect_microscope(self):
        """Connect from the dialog, then hand the session to the system widget.

        The widget stays the one place that owns the connection: everything else in
        the application follows its `connected_signal`, so routing through it means
        the dialog needs to know nothing about who cares.
        """
        if self.autolamella_ui is None:
            return

        system_widget = self.autolamella_ui.system_widget
        result = connect_to_microscope_dialog(
            parent=self,
            microscope=system_widget.microscope,
            settings=system_widget.settings,
            workflow_running=self.autolamella_ui.session.runner.is_running,
        )
        if not result.changed:
            return

        # Including a disconnect, where the new session is None. Handing that back
        # through the widget is what tells the rest of the application, which
        # follows its signals rather than knowing about this dialog.
        system_widget.microscope = result.microscope
        system_widget.settings = result.settings
        system_widget.update_ui()

    def _on_export_image(self):
        """Handle Export Image action: pick an image, starting in the experiment."""
        from fibsem.ui.widgets.image_export_dialog import open_image_export

        experiment = getattr(self.autolamella_ui, "experiment", None)
        start_dir = ""
        if experiment is not None and experiment.path is not None:
            start_dir = os.fspath(experiment.path)
        open_image_export(self, start_dir)

    def _on_report_issue(self):
        """Open the Report an Issue dialog."""
        from fibsem.ui.widgets.bug_report_widget import open_bug_report_dialog

        experiment = getattr(self.autolamella_ui, "experiment", None)
        microscope = getattr(self.autolamella_ui, "microscope", None)
        open_bug_report_dialog(
            experiment_path=str(experiment.path) if experiment is not None else None,
            microscope=microscope,
            parent=self,
        )

    def _on_guided_setup(self):
        """Run the microscope guided setup.

        Delegated to the connection tab rather than opened here, so that whatever the
        wizard saves is selected in the one combo box that decides what the next
        connection uses -- and so the live microscope is handed over, rather than the
        wizard opening a second client against the same instrument.
        """
        self.autolamella_ui.system_widget.run_guided_setup()

    def _on_show_plugins(self):
        """Open the read-only listing of registered extensions."""
        from fibsem.ui.widgets.plugins_dialog import PluginsDialog

        PluginsDialog(parent=self).exec_()

    def _on_create_desktop_shortcut(self, app: App = AUTOLAMELLA):
        """Create a launcher shortcut for ``app``, confirming overwrites.

        The location is chosen by the user, defaulting to the Desktop.
        """
        from pathlib import Path

        from PyQt5.QtWidgets import QFileDialog

        from fibsem.tools import desktop_shortcut

        chosen = QFileDialog.getExistingDirectory(
            self,
            "Choose Shortcut Location",
            str(desktop_shortcut.get_desktop_directory()),
        )
        if not chosen:
            return
        directory = Path(chosen)
        try:
            path = desktop_shortcut.create_desktop_shortcut(
                directory=directory, app=app
            )
        except FileExistsError as exc:
            reply = QMessageBox.question(
                self,
                "Create Desktop Shortcut",
                f"A shortcut already exists at:\n{exc}\n\nReplace it?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
            try:
                path = desktop_shortcut.create_desktop_shortcut(
                    overwrite=True, directory=directory, app=app
                )
            except Exception as exc2:
                self._show_shortcut_error(exc2)
                return
        except Exception as exc:
            self._show_shortcut_error(exc)
            return
        self.show_toast(f"Shortcut created: {path}", "info")

    def _show_shortcut_error(self, exc: Exception) -> None:
        logging.error(f"Failed to create desktop shortcut: {exc}")
        QMessageBox.warning(
            self,
            "Create Desktop Shortcut",
            f"Could not create the desktop shortcut:\n{exc}",
        )

    def _sync_view_menu(self) -> None:
        """Refresh the View menu's dynamic state right before it opens: reflect whether a
        view is full-screened (check + Exit-enabled)."""
        if getattr(self, "view_controller", None) is None:
            return
        fullscreen = self.view_controller.fullscreen is not None
        self.action_toggle_fullscreen.setChecked(fullscreen)
        self.action_exit_fullscreen.setEnabled(fullscreen)

    def _hotkey_toggle_fullscreen(self) -> None:
        """F5: toggle full screen for the selected view."""
        self.view_controller.toggle_fullscreen()

    def _hotkey_exit_fullscreen(self) -> None:
        """Esc: exit full screen (no-op when already showing the grid)."""
        self.view_controller.set_fullscreen(None)

    def _hotkey_acquire(self) -> None:
        """F2: acquire the selected view (SEM / FIB / FM), if a microscope is connected
        and an acquisition isn't already running."""
        view = self.view_controller.selected_view
        ui = self.autolamella_ui
        if view in (BeamType.ELECTRON, BeamType.ION):
            image_widget = getattr(ui, "image_widget", None)
            if image_widget is None:
                logging.info("F2 acquire: no image widget (microscope not connected)")
                return
            if getattr(image_widget, "is_acquiring", False):
                logging.info("F2 acquire: acquisition already in progress")
                return
            if view is BeamType.ELECTRON:
                image_widget.acquire_sem_image()
            else:
                image_widget.acquire_fib_image()
        elif view == "fm":
            fm_widget = getattr(ui, "fm_control_widget", None)
            if fm_widget is None:
                logging.info("F2 acquire: no fluorescence widget")
                return
            # `is_acquisition_active`, not `is_acquiring` — the latter does not exist on
            # this widget, so a getattr default of False would make the guard inert and
            # let F2 start a second FM acquisition mid-run (cf. FIB-441 / FIB-436).
            if fm_widget.is_acquisition_active:
                logging.info("F2 acquire: fluorescence acquisition already in progress")
                return
            fm_widget.acquire_image()

    def _selected_em_image_widget(self):
        """The image widget when a SEM/FIB view is selected, else None (FM / not
        connected). Backs the EM-only imaging hotkeys (live / auto contrast / auto focus)."""
        if self.view_controller.selected_view not in (BeamType.ELECTRON, BeamType.ION):
            return None
        return getattr(self.autolamella_ui, "image_widget", None)

    def _hotkey_toggle_live(self) -> None:
        """F6: toggle live acquisition on the selected SEM/FIB beam."""
        image_widget = self._selected_em_image_widget()
        if image_widget is None:
            logging.info("F6 live: only SEM/FIB support live acquisition")
            return
        image_widget.toggle_live_acquisition()

    def _hotkey_autocontrast(self) -> None:
        """F9: auto-contrast the selected SEM/FIB beam."""
        image_widget = self._selected_em_image_widget()
        if image_widget is None:
            logging.info("F9 autocontrast: only SEM/FIB supported")
            return
        image_widget.run_autocontrast()

    def _hotkey_autofocus(self) -> None:
        """F11: auto-focus the selected SEM/FIB beam."""
        image_widget = self._selected_em_image_widget()
        if image_widget is None:
            logging.info("F11 autofocus: only SEM/FIB supported")
            return
        image_widget.run_autofocus()

    def _on_generate_report_v2(self):
        """Tools → Reporting → Generate Report v2 (preview): write the page from
        the experiment's events.jsonl, under its folder, and open it; then
        print it to a PDF beside it.

        Both on workers: the page loads each lamella's final images for its
        thumbnails, and the PDF takes a browser a few seconds. Each says when it
        is done, or why it could not be.
        """
        experiment = getattr(self.autolamella_ui, "experiment", None)
        if experiment is None:
            self.show_toast("Open an experiment to report on.", "warning")
            return
        from fibsem.applications.autolamella.tools import report_v2
        from fibsem.ui.qt.threading import thread_worker

        missing = report_v2.no_record(experiment)  # recorded before the stream
        if missing is not None:
            self.show_toast(missing, "warning")
            return

        @thread_worker
        def _write():
            return report_v2.write_report(experiment)

        worker = _write()
        worker.returned.connect(self._on_report_v2_written)
        worker.errored.connect(
            lambda exc: self.show_toast(f"Could not write the report: {exc}", "error")
        )
        worker.start()

    def _on_report_v2_written(self, path) -> None:
        from PyQt5.QtCore import QUrl
        from PyQt5.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        self.show_toast(f"Report written: {path.name}", "success")
        self._print_report_v2(path)

    def _print_report_v2(self, path) -> None:
        from fibsem.applications.autolamella.tools import pdf_export
        from fibsem.ui.qt.threading import thread_worker

        @thread_worker
        def _print():
            return pdf_export.html_to_pdf(path)

        worker = _print()
        worker.returned.connect(
            lambda pdf: self.show_toast(f"PDF written: {pdf.name}", "success")
        )
        worker.errored.connect(
            lambda exc: self.show_toast(
                f"No PDF: {exc}. The page's Print button makes one.", "warning"
            )
        )
        worker.start()

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
            # The border goes red on workflow_stopping_signal, as for any stop.
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
        sample = getattr(self.autolamella_ui, "sample_widget", None)
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
        self.lamella_widget.flush_pending_save()
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
        sample = getattr(self.autolamella_ui, "sample_widget", None)
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

    def _on_microscope_connected(self):
        """Handle microscope connection: hand the microscope to what shows its
        progress."""
        # Before the signal wiring below, which returns early on a disconnect: both
        # overview modalities have to hear about that case too, to let go of the old
        # microscope. They hold it for life, so each has to be handed the new one. It
        # also re-answers whether the tab can be used, which is only knowable now --
        # whether this system has a fluorescence detector at all.
        self._refresh_overview_microscope()
        if getattr(self, "quad_overview_page", None) is not None:
            self.quad_overview_page.set_microscope(
                self.autolamella_ui.microscope if self.autolamella_ui else None
            )
        self._refresh_fm_placeholder()
        # Milling, tiles, spot burn: the bar listens for them itself (FIB-1188).
        self.status_bar.set_microscope(
            self.autolamella_ui.microscope if self.autolamella_ui else None
        )
        self.agent_host.activity.set_microscope(
            self.autolamella_ui.microscope if self.autolamella_ui else None
        )
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
        self.autolamella_ui.workflow_status_signal.connect(self._on_workflow_status)
        self.autolamella_ui.step_update_signal.connect(self._on_step_update)
        self.autolamella_ui.experiment_update_signal.connect(self._on_experiment_update)
        self.autolamella_ui._workflow_finished_signal.connect(
            self._on_workflow_finished
        )
        self.autolamella_ui.session.runner.starting.connect(self._on_run_starting)
        # Red while a stop unwinds, whoever asked for it (the agent's too).
        self.autolamella_ui.workflow_stopping_signal.connect(self._on_workflow_stopping)
        # A grid task records its overview as it finishes, and says so in a status.
        self.autolamella_ui.workflow_status_signal.connect(self._refresh_quad_overviews)
        self.autolamella_ui._workflow_finished_signal.connect(
            self._refresh_quad_overviews
        )
        self.autolamella_ui._hook_toast_signal.connect(self.show_toast)
        self.autolamella_ui.grids_changed.connect(self._refresh_after_naming)
        notification_service._get_service().toast.connect(self._on_notification_service)
        notification_service._get_service().toast_action.connect(
            self._on_notification_service_action
        )
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
            self.action_toggle_fullscreen,
            self.action_exit_fullscreen,
            *self._imaging_actions,
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
            prompts=self.autolamella_ui,
            tab_widget=self.tab_widget,
            controls=self.workflow_controls,
            status_bar=self.status_bar,
            timeline=self.workflow_tab.timeline,
            sound_enabled=lambda: self._sound_enabled,
            parent=self,
        )

        self.experiment_header = ExperimentHeader(
            chip_enabled=self._connection_chip_enabled,
            new_experiment=self.action_new_experiment,
            load_experiment=self.action_load_experiment,
            open_experiment_directory=self.action_open_experiment_directory,
            on_new_experiment=self.autolamella_ui.create_experiment,
            on_load_experiment=self.autolamella_ui.load_experiment,
            on_connect=self._on_connect_microscope,
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
        self.lamella_widget.set_experiment()
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
        self.lamella_widget.setMinimumWidth(500)
        self.workflow_tab.lamella_workflow_widget.setMinimumWidth(600)

        self._refresh_header()

        # Show run workflow button when experiment is loaded
        self.workflow_controls.run_btn.show()

        # enable all the tabs (except lamella tab, which is managed by _update_lamella_tab_enabled)
        lamella_tab_index = (
            self.tab_widget.indexOf(self._lamella_tab_container)
            if hasattr(self, "_lamella_tab_container")
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
        self.lamella_card_container = LamellaCardContainer(
            columns=1, mode=self._preferences.display.lamella_card_mode
        )
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

        # History tab: what was done to this lamella, task by task, with each
        # task's images.
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

        # With no card selected the editor has no lamella, so the page says so
        # rather than leaving the last one up to be edited (FIB-1370).
        protocol_empty = QLabel("Select a lamella card to edit its protocol.")
        protocol_empty.setStyleSheet(f"color: {TEXT_MUTED_COLOR}; font-size: 12px;")
        protocol_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._protocol_page = QStackedWidget()
        self._protocol_page.addWidget(protocol_empty)
        self._protocol_page.addWidget(protocol_splitter)

        right_tabs.addTab(self._protocol_page, "Protocol")
        right_tabs.addTab(self.lamella_task_image_widget, "History")

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

    def add_grids_tab(self):
        """The Grids tab, between Lamella and Workflow: the experiment's grid records."""
        self.grids_tab = GridsTabWidget()
        # The Positions view makes lamellae through the application widget, the
        # one path every lamella is made by.
        self.grids_tab.set_autolamella_ui(self.autolamella_ui)
        # Fires on disconnect too, with microscope None; the tab redraws its chips
        # from whatever stage there is.
        self.autolamella_ui.system_widget.connected_signal.connect(
            self._refresh_grids_tab_microscope
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
        sample = getattr(self.autolamella_ui, "sample_widget", None)
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
            getattr(self, "lamella_card_container", None),
            self.workflow_tab.lamella_workflow_widget.lamella_list,
            getattr(self.autolamella_ui, "lamella_list", None),
        ):
            if widget is not None:
                widget.set_grid_context(context)
        # The Overview canvases mark only the lamellae on the stage, so what
        # they draw changes with every load and unload too.
        self._refresh_overview_positions()

    def _refresh_sample_view(self) -> None:
        """Redraw Microscope → Sample from the stage. Looked up each time: the
        Sample view is rebuilt on every connect."""
        sample = getattr(self.autolamella_ui, "sample_widget", None)
        if sample is not None:
            sample.refresh()

    def add_workflow_tab(self):
        """The Workflow tab: what to run, the queue and its timeline."""
        ui = self.autolamella_ui
        self.workflow_tab = WorkflowTab(
            ui.session,
            controls=self.workflow_controls,
            flush_edits=self.lamella_widget.flush_pending_save,
            on_config_changed=ui._on_workflow_config_changed,
            stop_task=ui.stop_current_task,
            after_naming=self._refresh_after_naming,
            say=self._say_selection,
            show_toast=self.show_toast,
        )
        ui.queue_changed_signal.connect(self.workflow_tab.on_queue_changed)
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
            lambda _: self.lamella_widget._refresh_experiment_positions()
        )
        # And the Grid page → Workflow → Grids: a task added on the Protocol tab
        # gets its row in the run view without an inventory or a reload.
        self.task_widget.grid_protocol_changed.connect(grids.refresh)

    def _on_lamella_card_selected(self, lamella: Lamella | None):
        """Update task image panel and protocol editor for selected lamella card."""
        if not hasattr(self, "lamella_task_image_widget"):
            return

        self._selected_card_lamella = lamella
        self.overview_tab.set_selected(lamella)
        self.lamella_task_image_widget.set_lamella(lamella)
        self.lamella_widget.select_lamella(lamella.name if lamella else None)
        self._protocol_page.setCurrentIndex(0 if lamella is None else 1)
        if lamella is not None:
            if not self._syncing_selection:
                self._syncing_selection = True
                try:
                    self.autolamella_ui.lamella_list.select(lamella.name)
                finally:
                    self._syncing_selection = False

    def _on_experiment_lamella_selected(self, lamella):
        """Sync card container and overviews when experiment-tab list selection changes."""
        self.overview_tab.set_selected(lamella)
        self.quad_overview_page.set_selected(lamella)
        if getattr(self, "_syncing_selection", False) or lamella is None:
            return
        if not hasattr(self, "lamella_card_container"):
            return
        self._syncing_selection = True
        try:
            self.lamella_card_container.select_lamella(lamella.name)
        finally:
            self._syncing_selection = False

    def _on_workflow_status(self, event: "WorkflowStatusEvent"):
        """Handle a fire-and-forget status update, from workflow_status_signal.

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
            self.lamella_widget.set_active_lamella_name(lamella_name, task_name)
        else:
            self.lamella_widget.set_active_lamella_name(None)

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
            sample = getattr(self.autolamella_ui, "sample_widget", None)
            if sample is not None:
                sample.refresh()
        elif lamella is not None:
            self.workflow_tab.lamella_workflow_widget.lamella_list.refresh_lamella(
                lamella
            )
            self.lamella_card_container.refresh_lamella(lamella)
            self.autolamella_ui.lamella_list.refresh_lamella(lamella)
            # The Overview pages' lists carry the same status column; without
            # this they sat stale for the whole run (FIB-995).
            for lamella_list in self.overview_tab.lamella_lists():
                lamella_list.refresh_lamella(lamella)
        else:
            self.workflow_tab.lamella_workflow_widget.lamella_list.refresh_all()
            self.lamella_card_container.refresh_all()
            self.autolamella_ui.lamella_list.refresh_all()
            for lamella_list in self.overview_tab.lamella_lists():
                lamella_list.refresh_all()
        selected = getattr(self, "_selected_card_lamella", None)
        self._on_lamella_card_selected(selected)
        # Re-selecting does not redraw the History panel: it skips the lamella it
        # already shows. A task that has finished on that lamella has added to its
        # history, and this report comes after the task has returned -- its images
        # written and its history entry appended -- so rebuild it here (FIB-1111).
        if (
            selected is not None
            and lamella is not None
            and selected.id == lamella.id
            and status
            in (
                TaskStatus.Completed,
                TaskStatus.Failed,
                TaskStatus.Cancelled,
            )
            and hasattr(self, "lamella_task_image_widget")
        ):
            self.lamella_task_image_widget.refresh()

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

        hold = self.autolamella_ui.hold
        self.attention.show_hold(hold)

        self.run_border.set_state(
            run_border_state(
                self.run_border.state,
                held=hold is not None,
                pending=self.autolamella_ui.WORKFLOW_PENDING,
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
        selected = getattr(self, "_selected_card_lamella", None)
        self.lamella_card_container.clear()
        # The overview canvases are further displays of the same set, so they are
        # rebuilt here rather than from subscriptions of their own -- one handler for
        # "the lamellae changed" means the displays cannot end up disagreeing about what
        # the experiment holds. Before the `experiment is None` return, because an
        # experiment closing has to clear them as much as an experiment opening has to
        # fill them.
        self._refresh_overview_positions()
        if experiment is None:
            self.workflow_tab.lamella_workflow_widget.lamella_list.clear()
            self._on_lamella_card_selected(None)
            return
        self._refresh_grid_context()
        self.workflow_tab.lamella_workflow_widget.lamella_list.set_lamellae(
            list(experiment.positions)
        )
        for lamella in experiment.positions:
            self.lamella_card_container.add_lamella(lamella)
        still_there = next(
            (
                lamella
                for lamella in experiment.positions
                if selected is not None and lamella.id == selected.id
            ),
            None,
        )
        if still_there is not None:
            self.lamella_card_container.select_lamella(still_there.name)
        self._on_lamella_card_selected(still_there)
        self.workflow_tab.selection_changed()
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
        self.autolamella_ui.experiment.sign_verdict(lamella)
        self.autolamella_ui.experiment.save()
        # Sync defect icon across all widgets
        self.autolamella_ui.lamella_list.refresh_all()
        self.workflow_tab.lamella_workflow_widget.lamella_list.refresh_lamella(lamella)
        self.lamella_card_container.refresh_lamella(lamella)

    def _on_lamella_remove_requested(self, lamella: "Lamella"):
        """Remove the given lamella from the experiment after the list widget has already removed its row."""
        if self.autolamella_ui is None or self.autolamella_ui.experiment is None:
            return
        if not remove_lamella(self.autolamella_ui.experiment, lamella):
            return
        self.autolamella_ui.update_lamella_combobox(latest=True)
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
        self.lamella_widget.set_active_lamella_name(None)
        if getattr(self, "grids_tab", None) is not None:
            self.grids_tab.set_controls_enabled(True)
        self.workflow_tab.grid_workflow_widget.set_controls_enabled(True)
        self.workflow_tab.grid_workflow_widget.refresh()
        sample = getattr(self.autolamella_ui, "sample_widget", None)
        if sample is not None:
            sample.set_controls_enabled(True)
            sample.refresh()
        self.workflow_controls.set_attention(None)
        self.workflow_tab.lamella_workflow_widget.lamella_list.refresh_all()
        self.lamella_card_container.refresh_all()
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
        self.overview_tab = AutoLamellaOverviewContainerTab(self.autolamella_ui)
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
            self.autolamella_ui.lamella_list.select(lamella.name)
            if hasattr(self, "lamella_card_container"):
                self.lamella_card_container.select_lamella(lamella.name)
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
        if getattr(self, "lamella_widget", None) is not None:
            try:
                self.lamella_widget.flush_pending_save()
            except Exception as e:
                logging.warning(
                    f"Could not flush a pending experiment save on close: {e}"
                )
        # persist the FM working state (channels / camera transform / objective)
        if (
            self.autolamella_ui is not None
            and self.autolamella_ui.fm_control_widget is not None
        ):
            try:
                self.autolamella_ui.fm_control_widget.save_fm_configuration()
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
