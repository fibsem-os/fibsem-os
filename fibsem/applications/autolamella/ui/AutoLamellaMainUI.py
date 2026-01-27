from __future__ import annotations

import sys

try:
    sys.modules.pop("PySide6.QtCore")
except Exception:
    pass

import logging
import traceback

import napari
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from superqt.iconify import QIconifyIcon

import fibsem
from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus
from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI
from fibsem.ui import FibsemMinimapWidget
from fibsem.ui.stylesheets import NAPARI_STYLE
from fibsem.ui.widgets.autolamella_lamella_protocol_editor import (
    AutoLamellaProtocolEditorWidget,
)
from fibsem.ui.widgets.autolamella_task_config_editor import (
    AutoLamellaProtocolTaskConfigEditor,
    AutoLamellaWorkflowWidget,
)
from fibsem.ui.widgets.notifications import NotificationBell, ToastManager


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

        self.viewers = []
        self.autolamella_ui = None
        self.minimap_widget: FibsemMinimapWidget
        self.minimap_viewer: napari.Viewer

        # Toast notification manager
        self.toast_manager = ToastManager(self)

        # Status bar pulse animation for user attention
        self._status_pulse_timer = QTimer(self)
        self._status_pulse_timer.timeout.connect(self._toggle_status_pulse)
        self._status_pulse_state = False  # False = original, True = light cyan
        self._status_pulse_enabled = False  # Toggle for pulse animation
        self._user_interaction_sound_played = False  # Track if sound was played
        self._sound_enabled = True  # Toggle for notification sounds

        # Create menu bar (includes notification bell)
        self._create_menu_bar()
        self._create_test_menu()

        # Create status bar
        self._create_status_bar()

        # Create tabs
        self.create_tabs()

        # Hide main menu bar from AutoLamellaUI viewer
        self.autolamella_ui.menuBar().setVisible(False)

        # Connect tab change to status bar update
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

    def _create_menu_bar(self):
        """Create the main menu bar."""
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("File")

        self.action_new_experiment = QAction("New Experiment", self)
        self.action_new_experiment.triggered.connect(self._on_new_experiment)
        file_menu.addAction(self.action_new_experiment)

        self.action_load_experiment = QAction("Load Experiment", self)
        self.action_load_experiment.triggered.connect(self._on_load_experiment)
        file_menu.addAction(self.action_load_experiment)

        file_menu.addSeparator()

        self.action_load_protocol = QAction("Load Protocol", self)
        self.action_load_protocol.triggered.connect(self._on_load_protocol)
        file_menu.addAction(self.action_load_protocol)

        self.action_save_protocol = QAction("Save Protocol", self)
        self.action_save_protocol.triggered.connect(self._on_save_protocol)
        file_menu.addAction(self.action_save_protocol)

        file_menu.addSeparator()

        self.action_exit = QAction("Exit", self)
        self.action_exit.triggered.connect(self.close)
        file_menu.addAction(self.action_exit)

        # add tools menu
        tools_menu = menu_bar.addMenu("Tools")
        # add reporting sub menu
        reporting_menu = tools_menu.addMenu("Reporting")
        self.action_generate_report = QAction("Generate Report", self, triggered=self._on_generate_report)
        self.action_generate_overview_plot = QAction("Generate Overview Plot", self, triggered=self._on_generate_overview_plot)
        reporting_menu.addAction(self.action_generate_report)
        reporting_menu.addAction(self.action_generate_overview_plot)

        # add help menu
        help_menu = menu_bar.addMenu("Help")
        self.action_about = QAction("About", self, triggered=self._show_about_dialog) # type: ignore
        help_menu.addAction(self.action_about)

    def _create_test_menu(self):        
        """Create a test menu for toast notifications and sounds."""
        menu_bar = self.menuBar()
        # Test menu for toast notifications
        test_menu = menu_bar.addMenu("Test")

        self.action_toast_info = QAction("Toast: Info", self)
        self.action_toast_info.triggered.connect(lambda: self.show_toast("This is an info message", "info"))
        test_menu.addAction(self.action_toast_info)

        self.action_toast_success = QAction("Toast: Success", self)
        self.action_toast_success.triggered.connect(lambda: self.show_toast("Operation completed successfully!", "success"))
        test_menu.addAction(self.action_toast_success)

        self.action_toast_warning = QAction("Toast: Warning", self)
        self.action_toast_warning.triggered.connect(lambda: self.show_toast("Warning: Check your settings", "warning"))
        test_menu.addAction(self.action_toast_warning)

        self.action_toast_error = QAction("Toast: Error", self)
        self.action_toast_error.triggered.connect(lambda: self.show_toast("Error: Something went wrong", "error"))
        test_menu.addAction(self.action_toast_error)

        test_menu.addSeparator()

        self.action_beep = QAction("Play Beep", self)
        self.action_beep.triggered.connect(play_notification_sound)
        test_menu.addAction(self.action_beep)

        self.action_sound_toggle = QAction("Sound Enabled", self)
        self.action_sound_toggle.setCheckable(True)
        self.action_sound_toggle.setChecked(True)
        self.action_sound_toggle.triggered.connect(self._on_sound_toggle)
        test_menu.addAction(self.action_sound_toggle)

        self.action_pulse_toggle = QAction("Pulse Animation Enabled", self)
        self.action_pulse_toggle.setCheckable(True)
        self.action_pulse_toggle.setChecked(True)
        self.action_pulse_toggle.triggered.connect(self._on_pulse_toggle)
        test_menu.addAction(self.action_pulse_toggle)

    def _on_sound_toggle(self, checked: bool):
        """Handle sound toggle."""
        self._sound_enabled = checked

    def _on_pulse_toggle(self, checked: bool):
        """Handle pulse animation toggle."""
        self._status_pulse_enabled = checked

    def show_toast(self, message: str, notification_type: str = "info", duration: int = 5000):
        """Show a toast notification."""
        self.toast_manager.show_toast(message, notification_type, duration)

    def _on_new_experiment(self):
        """Handle New Experiment action."""
        if self.autolamella_ui is not None:
            self.autolamella_ui.create_experiment()

    def _on_load_experiment(self):
        """Handle Load Experiment action."""
        if self.autolamella_ui is not None:
            self.autolamella_ui.load_experiment()

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
        self.status_bar.showMessage("Hello Microscope Control")

        # Add progress bar to status bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #3d4251;
                border-radius: 3px;
                text-align: center;
                background-color: #1e2027;
                color: #d6d6d6;
            }
            QProgressBar::chunk {
                background-color: #50a6ff;
            }
        """)
        self.progress_bar.hide()  # Hidden by default
        self.status_bar.addPermanentWidget(self.progress_bar)

        # Add stop workflow button
        self.stop_workflow_btn = QPushButton("Stop Workflow")
        self.stop_workflow_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 4px 12px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:pressed {
                background-color: #b71c1c;
            }
        """)
        self.stop_workflow_btn.hide()  # Hidden by default
        self.stop_workflow_btn.clicked.connect(self._on_stop_workflow_clicked)
        self.status_bar.addPermanentWidget(self.stop_workflow_btn)

    def _on_stop_workflow_clicked(self):
        """Handle stop workflow button click."""
        if self.autolamella_ui is not None:
            self.autolamella_ui.stop_task_workflow()

    def set_progress(self, value: int, message: str = None):
        """Show and update the progress bar."""
        self.progress_bar.show()
        self.stop_workflow_btn.show()
        self.progress_bar.setValue(value)
        if message:
            self.status_bar.showMessage(message)

    def hide_progress(self):
        """Hide the progress bar and stop button."""
        self.progress_bar.hide()
        self.stop_workflow_btn.hide()

    def _on_tab_changed(self, index: int):
        """Handle tab change and update status bar."""
        tab_name = self.tab_widget.tabText(index)
        self.status_bar.showMessage(f"Hello {tab_name}")

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

        # Add it as a dock widget to the viewer
        self.main_viewer.window.add_dock_widget(
            widget=self.autolamella_ui,
            area="right",
            add_vertical_stretch=True,
            name="AutoLamella"
        )

        # Add the viewer's Qt window to our layout
        layout.addWidget(self.main_viewer.window._qt_window)

        # Add status bar for this tab
        self.main_status_bar = QStatusBar()
        self.main_status_bar.showMessage("Microscope Control: Ready")
        layout.addWidget(self.main_status_bar)

        self.tab_widget.addTab(container, QIconifyIcon("mdi:microscope", color="#d6d6d6"), "Microscope")

    def create_tabs(self):
        """Create the tabs for the AutoLamella UI."""
        self._create_main_tab()
        self.add_minimap_tab()
        self.add_protocol_editor_tab()
        self.add_lamella_tab()
        self.add_workflow_tab()

        # add notification button to tab bar
        self.create_notification_button()

    def _on_experiment_update(self):
        """Handle experiment update signal and propagate to tabs."""

        self.minimap_widget.set_experiment()
        self.task_widget.set_experiment(self.autolamella_ui.experiment)
        self.lamella_widget.set_experiment()
        self.workflow_widget.set_experiment(self.autolamella_ui.experiment)

        # enable all the tabs
        for i in range(self.tab_widget.count()):
            self.tab_widget.setTabEnabled(i, True)

    def create_notification_button(self):
        """Add buttons to the tab bar for adding Protocol Editor, Lamella, and Minimap tabs."""
        # Create button container widget
        button_widget = QWidget()
        button_layout = QHBoxLayout(button_widget)
        button_layout.setContentsMargins(5, 0, 5, 0)
        button_layout.setSpacing(5)


        # Notification bell
        self.notification_bell = NotificationBell(self)
        self.toast_manager.set_notification_bell(self.notification_bell)
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

        # Add status bar for this tab
        self.editor_status_bar = QStatusBar()
        self.editor_status_bar.showMessage("Protocol Editor: No protocol loaded")
        layout.addWidget(self.editor_status_bar)

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

        # Add status bar for this tab
        self.lamella_status_bar = QStatusBar()
        self.lamella_status_bar.showMessage("Lamella: No lamella selected")
        layout.addWidget(self.lamella_status_bar)

        self.tab_widget.addTab(container, QIconifyIcon("mdi:layers", color="#d6d6d6"), "Lamella")

        # disable the tab by default
        index = self.tab_widget.indexOf(container)
        self.tab_widget.setTabEnabled(index, False)

    def add_workflow_tab(self):
        """Add an empty workflow tab with just a header."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header label
        header = QLabel("Workflow")
        header.setStyleSheet("color: #d6d6d6; font-size: 18px; font-weight: bold;")
        layout.addWidget(header)

        self.workflow_widget = AutoLamellaWorkflowWidget(
            experiment=self.autolamella_ui.experiment,
            parent=self.autolamella_ui
        )
        if self.autolamella_ui is not None and self.autolamella_ui.experiment is not None:
            self.workflow_widget.workflow_config_changed.connect(self.autolamella_ui._on_workflow_config_changed)
            self.workflow_widget.workflow_options_changed.connect(self.autolamella_ui._on_workflow_options_changed)
            self.workflow_widget.setStyleSheet(NAPARI_STYLE)

        # add horizontal layout for workflow widget
        grid_layout = QHBoxLayout()
        if hasattr(self, 'workflow_widget') and self.workflow_widget is not None:
            grid_layout.addWidget(self.workflow_widget)
            # add horizontal stretch
            grid_layout.addStretch()
        layout.addLayout(grid_layout)

        # Add stretch to push header to top
        layout.addStretch()

        # Add status bar for this tab
        self.workflow_status_bar = QStatusBar()
        self.workflow_status_bar.showMessage("Workflow: No workflow running")
        layout.addWidget(self.workflow_status_bar)

        self.tab_widget.addTab(container, QIconifyIcon("mdi:play-circle-outline", color="#d6d6d6"), "Workflow")

        # disable the workflow tab by default
        self.tab_widget.setTabEnabled(self.tab_widget.indexOf(container), False)

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
            if current_lamella_index is not None and total_lamellae is not None:
                txt += f" ({current_lamella_index + 1}/{total_lamellae})"
            if current_task_index is not None and total_tasks is not None:
                txt += f" | Task {current_task_index + 1}/{total_tasks}"

            if hasattr(self, 'workflow_status_bar'):
                self.workflow_status_bar.showMessage(txt)

            if current_lamella_index is not None and total_lamellae is not None:
                progress = int(((current_lamella_index + 1) / total_lamellae) * 100)
                self.set_progress(progress, txt)

            # Show toast notification based on status
            msg_type = "info"
            if status is AutoLamellaTaskStatus.Completed:
                msg_type = "success"
            elif status is AutoLamellaTaskStatus.Failed:
                msg_type = "error"
                msg = error_msg if error_msg is not None else msg
            elif status is AutoLamellaTaskStatus.Skipped:
                msg_type = "warning"
            # toast_msg = f"{task_name} - {lamella_name}: {msg}"
            self.show_toast(msg, msg_type)

        # Check if waiting for user response and start/stop pulse animation
        if self.autolamella_ui is not None:
            if self.autolamella_ui.WAITING_FOR_USER_INTERACTION:
                if self._status_pulse_enabled:
                    # Start pulsing animation if not already running
                    if not self._status_pulse_timer.isActive():
                        self._status_pulse_timer.start(500)  # Toggle every 500ms
                else:
                    # Show constant light cyan color (no pulse)
                    self._status_pulse_timer.stop()
                    self.status_bar.setStyleSheet("""
                        QStatusBar {
                            background-color: #4dd0e1;
                            color: #1e2027;
                            border-top: 1px solid #80deea;
                        }
                    """)
                # Play notification sound once when entering waiting state
                if not self._user_interaction_sound_played and self._sound_enabled:
                    play_notification_sound()
                    self._user_interaction_sound_played = True
            else:
                # Stop pulsing and reset to original dark theme
                self._status_pulse_timer.stop()
                self._status_pulse_state = False
                self._user_interaction_sound_played = False  # Reset for next time
                self.status_bar.setStyleSheet("""
                    QStatusBar {
                        background-color: #1e2027;
                        color: #d6d6d6;
                        border-top: 1px solid #3d4251;
                    }
                """)

    def _toggle_status_pulse(self):
        """Toggle status bar color for pulse animation."""
        self._status_pulse_state = not self._status_pulse_state
        if self._status_pulse_state:
            # Light cyan
            self.status_bar.setStyleSheet("""
                QStatusBar {
                    background-color: #4dd0e1;
                    color: #1e2027;
                    border-top: 1px solid #80deea;
                }
            """)
        else:
            # Original dark theme
            self.status_bar.setStyleSheet("""
                QStatusBar {
                    background-color: #1e2027;
                    color: #d6d6d6;
                    border-top: 1px solid #3d4251;
                }
            """)

    def _on_workflow_finished(self):
        """Handle workflow finished signal."""
        self.hide_progress()
        # Stop pulse animation if running
        self._status_pulse_timer.stop()
        self._status_pulse_state = False
        # Green for workflow complete
        self.status_bar.showMessage("Workflow: Finished")
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #4caf50;
                color: #1e2027;
                border-top: 1px solid #81c784;
            }
        """)
        if hasattr(self, 'workflow_status_bar'):
            self.workflow_status_bar.showMessage("Workflow: Finished")

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

        # Add status bar for this tab
        self.minimap_status_bar = QStatusBar()
        self.minimap_status_bar.showMessage("Minimap: No overview acquired")
        layout.addWidget(self.minimap_status_bar)

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
