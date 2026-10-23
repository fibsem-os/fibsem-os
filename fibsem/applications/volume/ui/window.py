"""``fibsem-volume-ui``: the volume app's window. The Microscope tab is the quad
view and the panel, with its Positions tab first; the Protocol tab is the
tasks; the Positions tab is a card per position and its slices. The App
Server, when it is on, serves the same ``VolumeAppContext`` as
``fibsem-volume``.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any, List, Optional

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QInputDialog,
    QMessageBox,
    QSplitter,
)

from fibsem.applications.volume import launch
from fibsem.applications.volume.experiment_setup import (
    create_experiment,
    load_for_adoption,
)
from fibsem.applications.volume.server import VolumeAppContext
from fibsem.applications.volume.ui.positions_panel import PositionsPanel
from fibsem.applications.volume.ui.positions_tab import PositionsTab
from fibsem.applications.volume.ui.protocol_tab import ProtocolTab
from fibsem.tools.app_identity import set_app_identity
from fibsem.tools.apps import VOLUME
from fibsem.ui import notification_service
from fibsem.ui.app_window import AppWindow
from fibsem.ui.icon import fibsem_icon, fibsemos_icon
from fibsem.ui.qt.gc import install_main_thread_gc
from fibsem.ui.run.attention import AttentionRouter
from fibsem.ui.run.view import RunView
from fibsem.ui.stylesheets import GRAY_ICON_COLOR
from fibsem.ui.widgets.canvas.quad_view import MicroscopeViewController
from fibsem.ui.widgets.microscope_panel import MicroscopePanel
from fibsem.workflows.experiment import EXPERIMENT_FILENAME
from fibsem.workflows.experiments import default_experiment_directory
from fibsem.workflows.protocol_files import record_recent
from fibsem.workflows.runner import ITEMS, RunRequest
from fibsem.workflows.session import EXPERIMENT, MICROSCOPE, PROTOCOL

if TYPE_CHECKING:
    from fibsem.tools.desktop_launch import Splash

__all__ = ["VolumeWindow", "main"]

# What to do next, on the status bar.
INSTRUCTIONS = {
    "NOT_CONNECTED": "Connect to the microscope to begin.",
    "NO_EXPERIMENT": "Create or load an experiment to begin (File).",
    "NO_POSITION": "Add a position where the microscope is, on the Positions tab.",
    "READY": "Ready to run. Tick the tasks to run on the Protocol tab.",
}
POSITIONS_REDRAW_MS = 300


class VolumeWindow(AppWindow):
    def __init__(self, session=None):
        super().__init__("Volume", session=session)
        self._create_menu()
        self._create_status_bar()
        self._create_tabs()
        self._update_instructions()
        self.sync_app_server_with_preference()
        self._catch_up()

    def app_context(self) -> VolumeAppContext:
        return VolumeAppContext(self.session)

    # --- building -------------------------------------------------------------

    def _create_menu(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        self.action_new_experiment = file_menu.addAction("New Experiment…")
        self.action_load_experiment = file_menu.addAction("Load Experiment…")
        file_menu.addSeparator()
        file_menu.addAction("Quit", self.close)
        tools_menu = menu.addMenu("Tools")
        tools_menu.addAction("Host…", self._on_app_server_dialog)
        self.action_new_experiment.triggered.connect(self.create_experiment)
        self.action_load_experiment.triggered.connect(self.load_experiment)

    def _create_tabs(self) -> None:
        self.panel = MicroscopePanel(
            self.session,
            experiment_tab=lambda panel: PositionsPanel(self.session, panel),
            view_controller=self._view_controller,
            overview_acquiring=lambda: False,
        )
        self.panel.setAutoFillBackground(True)
        self._attach_run(self.panel)
        self.panel.tabWidget.insertTab(0, self.system_widget, "Connection")
        self.panel.refresh()

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        self.view_controller = MicroscopeViewController(parent=self)
        self.status_bar.follow(self.view_controller)
        splitter.addWidget(self.view_controller.widget)
        splitter.addWidget(self.panel)
        splitter.setSizes([700, 550])
        self.panel.setMinimumWidth(500)
        self.tab_widget.addTab(
            splitter,
            fibsem_icon("mdi:microscope", color=GRAY_ICON_COLOR),
            "Microscope",
        )

        self.protocol_tab = ProtocolTab(self.session)
        self.tab_widget.addTab(
            self.protocol_tab,
            fibsem_icon("mdi:format-list-numbered", color=GRAY_ICON_COLOR),
            "Protocol",
        )

        self.positions_tab = PositionsTab(self.session)
        self.tab_widget.addTab(
            self.positions_tab,
            fibsem_icon("mdi:layers-outline", color=GRAY_ICON_COLOR),
            "Positions",
        )
        # A run's status comes in bursts: the cards are drawn once per burst.
        self._positions_redraw = QTimer(self)
        self._positions_redraw.setSingleShot(True)
        self._positions_redraw.setInterval(POSITIONS_REDRAW_MS)
        self._positions_redraw.timeout.connect(self.positions_tab.refresh)

        self.attention = AttentionRouter(
            prompts=self.proposals,
            tab_widget=self.tab_widget,
            controls=self.workflow_controls,
            status_bar=self.status_bar,
            timeline=None,
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
            show_failures=lambda: self.tab_widget.setCurrentWidget(self.protocol_tab),
            after_supervision=self.protocol_tab.refresh,
            reset_panel=self._reset_panel_after_run,
        )
        self.workflow_controls.run_clicked.connect(self.run)
        self.panel.experiment_tab.items_changed.connect(self._update_instructions)
        self.panel.experiment_tab.items_changed.connect(self._positions_redraw.start)
        self.run_relay.status.connect(lambda _event: self._positions_redraw.start())
        self.protocol_tab.tasks.selection_changed.connect(
            lambda _: self._update_instructions()
        )
        self.run_relay.finished.connect(lambda *_: self.panel.refresh())
        self.run_relay.finished.connect(lambda *_: self._positions_redraw.start())
        self._session_changed.connect(self._on_session_changed)
        self.session.changed.connect(self._hear_session)

    # --- the session ----------------------------------------------------------

    def _on_session_changed(self, what: str, detail: Any) -> None:
        if what == MICROSCOPE:
            self.status_bar.set_microscope(self.microscope)
            self.agent_host.activity.set_microscope(self.microscope)
            self.protocol_tab.set_microscope(self.microscope)
        elif what == EXPERIMENT:
            self._setup_experiment_connections()
            self.attention.listen(self.experiment)
            self.protocol_tab.refresh()
            self.positions_tab.refresh()
            self._refresh_title()
        elif what == PROTOCOL:
            self.protocol_tab.refresh()
        self._update_instructions()

    def flush_editors(self) -> None:
        self.protocol_tab.flush_pending_save()

    def _refresh_title(self) -> None:
        title = self.windowTitle().split(" — ")[0]
        if self.experiment is not None:
            title = f"{title} — {self.experiment.name}"
        self.setWindowTitle(title)

    def _update_instructions(self) -> None:
        """What to do next on the status bar, and whether Run can start."""
        experiment = self.experiment
        if self.microscope is None:
            key = "NOT_CONNECTED"
        elif experiment is None:
            key = "NO_EXPERIMENT"
        elif not experiment.positions:
            key = "NO_POSITION"
        else:
            key = "READY"
        self.status_bar.set_instruction(INSTRUCTIONS[key])
        ticked = bool(self.protocol_tab.selected_task_names())
        self.workflow_controls.set_run_enabled(
            key == "READY" and ticked,
            "Run the ticked tasks on every position."
            if ticked
            else "Tick the tasks to run on the Protocol tab.",
        )

    def _lock_for_run(self, running: bool) -> None:
        """A run owns the stage: no new position while it goes."""
        self.panel.experiment_tab.add_button.setEnabled(
            not running and self.experiment is not None
        )

    def _reset_panel_after_run(self) -> None:
        self.panel.reset_after_run(str(self.experiment.path))
        self.panel.refresh()
        self.panel.set_current_workflow_message(msg=None, show=False)

    # --- experiments and runs -------------------------------------------------

    def _experiments_folder(self) -> str:
        return self.session.experiments_folder or default_experiment_directory()

    def create_experiment(self) -> None:
        """New Experiment: a name, then the folder it goes in."""
        name, ok = QInputDialog.getText(self, "New Experiment", "Name")
        name = name.strip()
        if not ok or not name:
            return
        directory = QFileDialog.getExistingDirectory(
            self, "Folder for the experiment", self._experiments_folder()
        )
        if not directory:
            return
        if os.path.exists(os.path.join(directory, name)):
            QMessageBox.warning(
                self, "New Experiment", f"{os.path.join(directory, name)} exists."
            )
            return
        self.session.adopt(create_experiment(directory, name))

    def load_experiment(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Experiment",
            self._experiments_folder(),
            f"Experiment ({EXPERIMENT_FILENAME})",
        )
        if not path:
            return
        experiment, why = load_for_adoption(path)
        if experiment is None:
            QMessageBox.warning(self, "Load Experiment", f"Not loaded: {why}.")
            return
        self.session.adopt(experiment)
        record_recent(experiment)

    def run(self) -> None:
        """Run the ticked tasks on every position, after a confirmation."""
        session = self.session
        experiment = session.experiment
        if session.runner.is_running or experiment is None:
            return
        task_names = self.protocol_tab.selected_task_names()
        request = RunRequest(kind=ITEMS, task_names=tuple(task_names))
        checked = session.runner.check(experiment, request)
        if not checked.started:
            QMessageBox.warning(self, "Cannot run", checked.reason)
            return
        positions = len(checked.request.names)
        reply = QMessageBox.question(
            self,
            "Run Workflow",
            f"Run {len(task_names)} task{'s' if len(task_names) != 1 else ''} "
            f"on {positions} position{'s' if positions != 1 else ''}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return
        started = session.runner.start(experiment, request)
        if not started.started:
            notification_service.show_toast(started.reason, "warning")


def main(
    argv: Optional[List[str]] = None,
    splash: Optional[Splash] = None,
    session=None,
) -> None:
    """Open the volume window on *session* (a new one when None). *argv* is
    the launcher's common signature; the window takes no flags. *splash*
    closes once the window is up."""
    set_app_identity(VOLUME.name, windows_app_id=VOLUME.app_id)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(VOLUME.name)
    app.setStyle("Fusion")
    app.setWindowIcon(fibsemos_icon())
    # Cyclic garbage is only collected on this thread (fibsem/ui/qt/gc.py).
    gc_collector = install_main_thread_gc(parent=app)  # noqa: F841 - kept alive
    window = VolumeWindow(session=session)
    window.show()
    if splash is not None:
        splash.finish(window)
    code = app.exec_()
    window.session.stop_app_server()
    sys.exit(code)


def host(argv: Optional[List[str]] = None, splash: Optional[Splash] = None) -> int:
    """``fibsem-volume-ui``: the Volume host with the window. The host's
    flags connect the session first (none: the window's Connect does) and
    may name an experiment to open."""
    return launch.main(argv, window=lambda session: main([], splash, session))


if __name__ == "__main__":
    raise SystemExit(main())
