"""The main window's menu bar: File, Edit, View, Imaging, Tools, Help, and the
Development and Test menus shown in dev mode. Each entry calls its target; the
imaging and full-screen keys are actions here, scoped to the Microscope tab by
the window.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QAction, QDialog, QMainWindow, QMessageBox

import fibsem.config as fibsem_cfg
from fibsem.structures import BeamType
from fibsem.tools.apps import AUTOLAMELLA, App, apps
from fibsem.ui.icon import fibsem_icon
from fibsem.ui.sound import play_notification_sound
from fibsem.ui.stylesheets import GRAY_ICON_COLOR
from fibsem.ui.widgets.coincidence_menu import CoincidenceMenu
from fibsem.ui.widgets.connection_dialog import connect_to_microscope_dialog
from fibsem.ui.widgets.run_border import RunBorder

if TYPE_CHECKING:
    from fibsem.applications.autolamella.ui.panel.panel import MicroscopePanel
    from fibsem.ui.FibsemSystemSetupWidget import FibsemSystemSetupWidget
    from fibsem.workflows.session import AppSession


def set_menu_icon(action: QAction, key: str) -> None:
    """Give a menu action an icon that actually renders.

    Qt turns menu icons off wholesale on macOS (AA_DontShowIconsInMenus), so an
    icon set the usual way shows on Windows and Linux and silently vanishes here.
    """
    action.setIcon(fibsem_icon(key, color=GRAY_ICON_COLOR))
    action.setIconVisibleInMenu(True)


@dataclass(frozen=True)
class MenuActions:
    """What the menu entries run, handed in by the window."""

    new_experiment: Callable[[], None]
    load_experiment: Callable[[], None]
    open_experiment_directory: Callable[[], None]
    load_protocol: Callable[[], None]
    save_protocol: Callable[[], None]
    fm_image_viewer: Callable[[], None]
    coincidence_viewer: Callable[[], None]
    generate_report: Callable[[], None]
    generate_overview_plot: Callable[[], None]
    grid_report: Callable[[], None]
    app_server: Callable[[], None]
    export_ml_data: Callable[[], None]
    about: Callable[[], None]


class MainMenuBar(QObject):
    # Preferences saved from the menu (the dialog or a Test-menu toggle).
    preferences_saved = pyqtSignal(object)

    def __init__(
        self,
        host: QMainWindow,
        *,
        session: AppSession,
        actions: MenuActions,
        panel: Callable[[], MicroscopePanel],
        system_widget: Callable[[], FibsemSystemSetupWidget],
        view_controller: Callable[[], Any],
        run_border: RunBorder,
        preferences: Any,
        show_toast: Callable[..., None],
        parent: Optional[QObject] = None,
    ) -> None:
        """Builds the menus on *host*'s menu bar; *preferences* are the ones
        read at start-up."""
        super().__init__(parent)
        self._host = host
        self._session = session
        self._actions = actions
        self._panel = panel
        self._system_widget = system_widget
        self._view_controller = view_controller
        self._run_border = run_border
        self._show_toast = show_toast
        # Read once: the File entry and the header chip are both gated on it.
        self.connection_chip_enabled = preferences.features.connection_chip
        self._create_menu_bar()
        self._create_test_menu()
        self.apply_preferences(preferences)

    def _create_menu_bar(self):
        """Create the main menu bar."""
        menu_bar = self._host.menuBar()

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
        self.action_connect_microscope = QAction("Connect to Microscope...", self._host)
        set_menu_icon(self.action_connect_microscope, "mdi:connection")
        self.action_connect_microscope.triggered.connect(self.connect_microscope)

        self.action_new_experiment = QAction("New Experiment", self._host)
        set_menu_icon(self.action_new_experiment, "mdi:plus")
        self.action_new_experiment.triggered.connect(self._actions.new_experiment)

        self.action_load_experiment = QAction("Load Experiment", self._host)
        set_menu_icon(self.action_load_experiment, "mdi:folder-open")
        self.action_load_experiment.triggered.connect(self._actions.load_experiment)

        self.action_open_experiment_directory = QAction(
            "Open Experiment Directory", self
        )
        set_menu_icon(self.action_open_experiment_directory, "mdi:folder")
        self.action_open_experiment_directory.triggered.connect(
            self._actions.open_experiment_directory
        )

        self.action_export_image = QAction("Export Image...", self._host)
        set_menu_icon(self.action_export_image, "mdi:export")
        self.action_export_image.triggered.connect(self._on_export_image)

        self.action_load_protocol = QAction("Load Protocol", self._host)
        self.action_load_protocol.triggered.connect(self._actions.load_protocol)
        self.action_save_protocol = QAction("Save Protocol", self._host)
        self.action_save_protocol.triggered.connect(self._actions.save_protocol)
        self.action_exit = QAction("Exit", self._host)
        self.action_exit.triggered.connect(self._host.close)

        # Behind the same flag as the chip. The action is built either way -- it is
        # the thing the tab removal will depend on -- but nothing offers it until
        # the feature is switched on, so a release carrying this changes nothing.
        if self.connection_chip_enabled:
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
        self.action_preferences = QAction("Preferences...", self._host)
        self.action_preferences.triggered.connect(self._on_open_preferences)
        edit_menu.addAction(self.action_preferences)

        # View menu
        # Quad-view display controls. The F5/Esc shortcuts live on these QActions — one
        # source of truth for the menu item and its keybinding (Qt renders the shortcut
        # text in the menu automatically). They are scoped to the Microscope tab via
        # setShortcutContext + container.addAction (see _create_main_tab), so they only
        # fire when focus is inside that tab.
        view_menu.addSeparator()
        self.action_toggle_fullscreen = QAction("Full Screen", self._host)
        self.action_toggle_fullscreen.setCheckable(True)
        self.action_toggle_fullscreen.setShortcut(QKeySequence(Qt.Key_F5))
        self.action_toggle_fullscreen.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.action_toggle_fullscreen.triggered.connect(self._hotkey_toggle_fullscreen)
        view_menu.addAction(self.action_toggle_fullscreen)

        self.action_exit_fullscreen = QAction("Exit Full Screen", self._host)
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
            act = QAction(label, self._host)
            act.triggered.connect(
                lambda _checked=False, k=key: self._view_controller().set_fullscreen(k)
            )
            fullscreen_menu.addAction(act)

        # The standalone viewer, where a multi-channel image is composed: the
        # Microscope tab's fluorescence view shows one frame (FIB-942). Needs no
        # experiment; it is a file viewer.
        view_menu.addSeparator()
        self.action_open_fm_image_viewer = QAction(
            "Fluorescence Image Viewer...", self._host
        )
        self.action_open_fm_image_viewer.triggered.connect(
            self._actions.fm_image_viewer
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
        self.imaging_actions: list = []  # added to the Microscope-tab container for scoping
        for label, key, handler in (
            ("Acquire", Qt.Key_F2, self._hotkey_acquire),
            ("Live Acquisition", Qt.Key_F6, self._hotkey_toggle_live),
            ("Auto Contrast", Qt.Key_F9, self._hotkey_autocontrast),
            ("Auto Focus", Qt.Key_F11, self._hotkey_autofocus),
        ):
            action = QAction(label, self._host)
            action.setShortcut(QKeySequence(key))
            action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
            action.triggered.connect(handler)
            imaging_menu.addAction(action)
            self.imaging_actions.append(action)

        # add tools menu, reporting submenu
        tools_menu = menu_bar.addMenu("Tools")
        if tools_menu is None:
            raise RuntimeError("Failed to create Tools menu in AutoLamella UI.")
        reporting_menu = tools_menu.addMenu("Reporting")
        if reporting_menu is None:
            raise RuntimeError("Failed to create Reporting submenu in AutoLamella UI.")
        self.action_generate_report = QAction("Generate Report", self._host)
        self.action_generate_report.triggered.connect(self._actions.generate_report)
        # Report v2 (FIB-1036): an HTML page built from events.jsonl. Offered
        # beside the PDF report until it has been checked against real runs.
        self.action_generate_report_v2 = QAction(
            "Generate Report v2 (preview)", self._host
        )
        self.action_generate_report_v2.setToolTip(
            "Write an HTML report from the experiment's events.jsonl under its "
            "folder, and open it"
        )
        self.action_generate_report_v2.triggered.connect(self._on_generate_report_v2)
        self.action_generate_overview_plot = QAction(
            "Generate Overview Plot", self._host
        )
        self.action_generate_overview_plot.triggered.connect(
            self._actions.generate_overview_plot
        )
        # The grid screening report (FIB-1057): what the Grids tab holds, as a PDF.
        self.action_generate_grid_report = QAction(
            "Generate Grid Screening Report", self
        )
        self.action_generate_grid_report.setToolTip(
            "Write a PDF of every grid's overviews and verdicts under the "
            "experiment folder, and open it"
        )
        self.action_generate_grid_report.triggered.connect(self._actions.grid_report)
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
            notify=self._show_toast,
            parent=self,
        )

        # The embedded App Server's dialog: whether it runs, its services, the
        # event log. The agent's level and connecting one are the agent chip's
        # (FIB-1035). Always visible: it is where the server is turned on. Next
        # to Scripts (FIB-863).
        self.action_app_server = QAction("App Server...", self._host)
        self.action_app_server.setMenuRole(QAction.NoRole)
        self.action_app_server.setToolTip(
            "The App Server: whether it is running, its services, and the event log"
        )
        self.action_app_server.triggered.connect(self._actions.app_server)
        tools_menu.addAction(self.action_app_server)

        # Below the separator because these act on the install rather than on the
        # experiment. The wizard needs a home here as well as on the connection tab:
        # the callout there appears once and is dismissible, so without this entry a
        # second microscope could never be set up through it.
        tools_menu.addSeparator()
        self.action_guided_setup = QAction("Guided Setup...", self._host)
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

        self.action_show_plugins = QAction("Plugins...", self._host)
        self.action_show_plugins.setToolTip(
            "Show every registered pattern, strategy and task, and where it came from"
        )
        self.action_show_plugins.triggered.connect(self._on_show_plugins)
        tools_menu.addAction(self.action_show_plugins)

        # One entry per app the launcher knows (fibsem.tools.apps), plugins included.
        shortcut_menu = tools_menu.addMenu("Create Desktop Shortcut")
        self.shortcut_actions = {}
        for key, app in apps().items():
            action = QAction(f"{app.name}...", self._host)
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
        self.action_report_issue = QAction("Report an Issue...", self._host)
        self.action_report_issue.triggered.connect(self._on_report_issue)
        help_menu.addAction(self.action_report_issue)
        self.action_about = QAction("About", self._host)
        self.action_about.triggered.connect(self._actions.about)
        help_menu.addAction(self.action_about)

        # add development menu
        dev_menu = menu_bar.addMenu("Development")
        if dev_menu is None:
            raise RuntimeError("Failed to create Development menu in AutoLamella UI.")
        self.action_print_hello = QAction("Print Hello", self._host)
        self.action_print_hello.triggered.connect(lambda: print("Hello"))
        dev_menu.addAction(self.action_print_hello)

        self._action_coincidence_separator = dev_menu.addSeparator()
        self.action_open_coincidence_viewer = QAction(
            "Open Coincidence Milling Viewer", self
        )
        self.action_open_coincidence_viewer.triggered.connect(
            self._actions.coincidence_viewer
        )
        dev_menu.addAction(self.action_open_coincidence_viewer)

        self.coincidence_menu = CoincidenceMenu(
            dev_menu,
            microscope=lambda: self._session.microscope,
            experiment_path=lambda: getattr(self._session.experiment, "path", None),
            show_toast=self._show_toast,
            parent=self,
        )

        self._dev_menu = dev_menu

        action_load_fm_configuration = QAction(
            "Load Fluorescence Configuration", self._host
        )
        action_load_fm_configuration.triggered.connect(
            lambda: self._panel().import_fm_configuration()
        )
        dev_menu.addAction(action_load_fm_configuration)

        action_save_fm_configuration = QAction(
            "Save Fluorescence Configuration", self._host
        )
        action_save_fm_configuration.triggered.connect(
            lambda: self._panel().export_fm_configuration()
        )
        dev_menu.addAction(action_save_fm_configuration)

        dev_menu.addSeparator()

        self.action_export_targeting_ml_data = QAction(
            "Export Targeting ML Data...", self
        )
        self.action_export_targeting_ml_data.triggered.connect(
            self._actions.export_ml_data
        )
        dev_menu.addAction(self.action_export_targeting_ml_data)

    def _create_test_menu(self):
        """Create a test menu for toast notifications and sounds."""

        self.action_toast_info = QAction("Toast: Info", self._host)
        self.action_toast_info.triggered.connect(
            lambda: self._show_toast("This is an info message", "info")
        )

        self.action_toast_success = QAction("Toast: Success", self._host)
        self.action_toast_success.triggered.connect(
            lambda: self._show_toast("Operation completed successfully!", "success")
        )

        self.action_toast_warning = QAction("Toast: Warning", self._host)
        self.action_toast_warning.triggered.connect(
            lambda: self._show_toast("Warning: Check your settings", "warning")
        )

        self.action_toast_error = QAction("Toast: Error", self._host)
        self.action_toast_error.triggered.connect(
            lambda: self._show_toast("Error: Something went wrong", "error")
        )

        self.action_beep = QAction("Play Beep", self._host)
        self.action_beep.triggered.connect(play_notification_sound)

        self.action_sound_toggle = QAction("Sound Enabled", self._host)
        self.action_sound_toggle.setCheckable(True)
        self.action_sound_toggle.triggered.connect(self._on_sound_toggle)

        # Border state test actions
        self.action_border_toggle = QAction("Show Workflow Border", self._host)
        self.action_border_toggle.setCheckable(True)
        self.action_border_toggle.triggered.connect(self._on_border_toggle)

        self.action_border_automated = QAction("Automated (green)", self._host)
        self.action_border_automated.triggered.connect(
            lambda: self._run_border.set_state("automated")
        )

        self.action_border_supervised = QAction("Supervised (blue)", self._host)
        self.action_border_supervised.triggered.connect(
            lambda: self._run_border.set_state("supervised")
        )

        self.action_border_waiting = QAction("Waiting for User (orange)", self._host)
        self.action_border_waiting.triggered.connect(
            lambda: self._run_border.set_state("waiting")
        )
        self.action_border_pending = QAction("Pending (grey)", self._host)
        self.action_border_pending.triggered.connect(
            lambda: self._run_border.set_state("pending")
        )
        self.action_border_stopping = QAction("Stopping (red)", self._host)
        self.action_border_stopping.triggered.connect(
            lambda: self._run_border.set_state("stopping")
        )

        self.action_border_idle = QAction("Idle (no border)", self._host)
        self.action_border_idle.triggered.connect(
            lambda: self._run_border.set_state("idle")
        )

        # add to menu bar
        test_menu = self._host.menuBar().addMenu("Test")  # type: ignore

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

    def _on_sound_toggle(self, checked: bool):
        # Through the file, not a copy held here: another widget may have saved a
        # setting of its own since start-up (the view bars' fields, FIB-1186).
        self.preferences_saved.emit(
            fibsem_cfg.update_user_preferences(
                lambda p: setattr(p.display, "sound_enabled", checked)
            )
        )

    def _on_border_toggle(self, checked: bool):
        self.preferences_saved.emit(
            fibsem_cfg.update_user_preferences(
                lambda p: setattr(p.display, "border_enabled", checked)
            )
        )

    def _on_open_preferences(self):
        from fibsem.ui.widgets.preferences_dialog import PreferencesDialog

        # Fresh from disk, so saving the dialog's result does not put back anything
        # another widget has changed since start-up.
        dialog = PreferencesDialog(
            fibsem_cfg.load_user_preferences(), parent=self._host
        )
        if dialog.exec_() == QDialog.Accepted:
            preferences = dialog.get_preferences()
            fibsem_cfg.save_user_preferences(preferences)
            self.preferences_saved.emit(preferences)

    def apply_preferences(self, preferences) -> None:
        """The toggles, the dev and test menus and the coincidence viewer entry
        follow *preferences*."""
        d = preferences.display
        self.action_sound_toggle.setChecked(d.sound_enabled)
        self.action_border_toggle.setChecked(d.border_enabled)
        self._dev_menu.menuAction().setVisible(d.dev_mode)
        self._test_menu.menuAction().setVisible(d.dev_mode)
        coincidence_enabled = preferences.features.coincidence_milling_enabled
        self.action_open_coincidence_viewer.setVisible(coincidence_enabled)
        self._action_coincidence_separator.setVisible(coincidence_enabled)

    #### USER SCRIPTS (FIB-338)

    def _script_context(self):
        """Build the context a script runs against, or explain why it cannot.

        Returns (context, reason). A reason means the menu entries are disabled and
        that string is shown instead.
        """
        from fibsem.applications.autolamella.scripting import ScriptContext

        experiment = self._session.experiment
        if experiment is None:
            return None, "Load an experiment to run scripts"
        if self._session.runner.is_running:
            # tasks mutate lamella state on a worker thread, so a script reading
            # mid-workflow would get a torn snapshot.
            return None, "Unavailable while a workflow is running"

        return ScriptContext(
            experiment=experiment,
            log=logging.info,
            microscope=self._session.microscope,
        ), ""

    def connect_microscope(self):
        """Connect from the dialog, then hand the session to the system widget.

        The widget stays the one place that owns the connection: everything else in
        the application follows its `connected_signal`, so routing through it means
        the dialog needs to know nothing about who cares.
        """
        system_widget = self._system_widget()
        result = connect_to_microscope_dialog(
            parent=self._host,
            microscope=system_widget.microscope,
            settings=system_widget.settings,
            workflow_running=self._session.runner.is_running,
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

        experiment = self._session.experiment
        start_dir = ""
        if experiment is not None and experiment.path is not None:
            start_dir = os.fspath(experiment.path)
        open_image_export(self._host, start_dir)

    def _on_report_issue(self):
        """Open the Report an Issue dialog."""
        from fibsem.ui.widgets.bug_report_widget import open_bug_report_dialog

        experiment = self._session.experiment
        microscope = self._session.microscope
        open_bug_report_dialog(
            experiment_path=str(experiment.path) if experiment is not None else None,
            microscope=microscope,
            parent=self._host,
        )

    def _on_guided_setup(self):
        """Run the microscope guided setup.

        Delegated to the connection tab rather than opened here, so that whatever the
        wizard saves is selected in the one combo box that decides what the next
        connection uses -- and so the live microscope is handed over, rather than the
        wizard opening a second client against the same instrument.
        """
        self._system_widget().run_guided_setup()

    def _on_show_plugins(self):
        """Open the read-only listing of registered extensions."""
        from fibsem.ui.widgets.plugins_dialog import PluginsDialog

        PluginsDialog(parent=self._host).exec_()

    def _on_create_desktop_shortcut(self, app: App = AUTOLAMELLA):
        """Create a launcher shortcut for ``app``, confirming overwrites.

        The location is chosen by the user, defaulting to the Desktop.
        """
        from pathlib import Path

        from PyQt5.QtWidgets import QFileDialog

        from fibsem.tools import desktop_shortcut

        chosen = QFileDialog.getExistingDirectory(
            self._host,
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
                self._host,
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
        self._show_toast(f"Shortcut created: {path}", "info")

    def _show_shortcut_error(self, exc: Exception) -> None:
        logging.error(f"Failed to create desktop shortcut: {exc}")
        QMessageBox.warning(
            self._host,
            "Create Desktop Shortcut",
            f"Could not create the desktop shortcut:\n{exc}",
        )

    def _sync_view_menu(self) -> None:
        """Refresh the View menu's dynamic state right before it opens: reflect whether a
        view is full-screened (check + Exit-enabled)."""
        if self._view_controller() is None:
            return
        fullscreen = self._view_controller().fullscreen is not None
        self.action_toggle_fullscreen.setChecked(fullscreen)
        self.action_exit_fullscreen.setEnabled(fullscreen)

    def _hotkey_toggle_fullscreen(self) -> None:
        """F5: toggle full screen for the selected view."""
        self._view_controller().toggle_fullscreen()

    def _hotkey_exit_fullscreen(self) -> None:
        """Esc: exit full screen (no-op when already showing the grid)."""
        self._view_controller().set_fullscreen(None)

    def _hotkey_acquire(self) -> None:
        """F2: acquire the selected view (SEM / FIB / FM), if a microscope is connected
        and an acquisition isn't already running."""
        view = self._view_controller().selected_view
        if view in (BeamType.ELECTRON, BeamType.ION):
            image_widget = self._panel().image_widget
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
            fm_widget = self._panel().fm_control_widget
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
        if self._view_controller().selected_view not in (
            BeamType.ELECTRON,
            BeamType.ION,
        ):
            return None
        return self._panel().image_widget

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
        experiment = self._session.experiment
        if experiment is None:
            self._show_toast("Open an experiment to report on.", "warning")
            return
        from fibsem.applications.autolamella.tools import report_v2
        from fibsem.ui.qt.threading import thread_worker

        missing = report_v2.no_record(experiment)  # recorded before the stream
        if missing is not None:
            self._show_toast(missing, "warning")
            return

        @thread_worker
        def _write():
            return report_v2.write_report(experiment)

        worker = _write()
        worker.returned.connect(self._on_report_v2_written)
        worker.errored.connect(
            lambda exc: self._show_toast(f"Could not write the report: {exc}", "error")
        )
        worker.start()

    def _on_report_v2_written(self, path) -> None:
        from PyQt5.QtCore import QUrl
        from PyQt5.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        self._show_toast(f"Report written: {path.name}", "success")
        self._print_report_v2(path)

    def _print_report_v2(self, path) -> None:
        from fibsem.applications.autolamella.tools import pdf_export
        from fibsem.ui.qt.threading import thread_worker

        @thread_worker
        def _print():
            return pdf_export.html_to_pdf(path)

        worker = _print()
        worker.returned.connect(
            lambda pdf: self._show_toast(f"PDF written: {pdf.name}", "success")
        )
        worker.errored.connect(
            lambda exc: self._show_toast(
                f"No PDF: {exc}. The page's Print button makes one.", "warning"
            )
        )
        worker.start()
