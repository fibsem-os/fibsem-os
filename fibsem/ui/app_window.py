"""What an app's main window is built on: its session, the run border round its
tabs, the status bar, the run's responder and executor, the connection and the
App Server. The app adds its tabs, its panel and its menus.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget

import fibsem.config as fibsem_cfg
from fibsem.acting import OPERATOR
from fibsem.microscope import FibsemMicroscope
from fibsem.structures.system import MicroscopeSettings
from fibsem.ui.agent.chip_host import AgentChipHost
from fibsem.ui.FibsemSystemSetupWidget import FibsemSystemSetupWidget
from fibsem.ui.qt.connections import connect_on_gui_thread
from fibsem.ui.qt.run_bridge import GuiExecutor, RunRelay
from fibsem.ui.run.proposals import ProposalPresenter
from fibsem.ui.run.responder import QtResponder
from fibsem.ui.stylesheets import NAPARI_STYLE, border_stylesheet
from fibsem.ui.widgets.notifications import ToastManager
from fibsem.ui.widgets.run_border import RunBorder
from fibsem.ui.widgets.status_bar import FibsemStatusBar
from fibsem.ui.widgets.workflow.controls import WorkflowControls
from fibsem.versioning import get_version_string
from fibsem.workflows.session import AppSession

if TYPE_CHECKING:
    from fibsem.server.app_context import AppContext
    from fibsem.ui.widgets.microscope_panel import MicroscopePanel

__all__ = ["AppWindow"]


class AppWindow(QMainWindow):
    """An app's main window. Subclasses build ``panel`` (a ``MicroscopePanel``),
    ``view_controller`` and ``run_view``, and say what the App Server serves
    (:meth:`app_context`)."""

    @property
    def microscope(self) -> Optional[FibsemMicroscope]:
        return self.session.microscope

    @microscope.setter
    def microscope(self, microscope: Optional[FibsemMicroscope]) -> None:
        self.session.microscope = microscope

    @property
    def experiment(self) -> Optional[Any]:
        return self.session.experiment

    @experiment.setter
    def experiment(self, experiment: Optional[Any]) -> None:
        self.session.experiment = experiment

    def __init__(self, title: str):
        super().__init__()
        # The microscope, the open experiment and the connection's event
        # stream are the session's; this window is one client of it.
        self.session = AppSession()
        self.settings: Optional[MicroscopeSettings] = None
        self.setWindowTitle(f"{title} v{get_version_string()} ")
        self.resize(1600, 1000)

        # Border state rules live here (on the parent) so that setProperty +
        # unpolish/polish on the RunBorder re-evaluates them.
        self.setStyleSheet(NAPARI_STYLE + border_stylesheet("workflow_border_frame"))

        self.tab_widget = QTabWidget()
        self.run_border = RunBorder("workflow_border_frame")
        self.run_border.set_content(self.tab_widget)
        self.setCentralWidget(self.run_border)

        self.toast_manager = ToastManager(self)
        self._preferences = fibsem_cfg.load_user_preferences()
        self.run_border.set_on(self._preferences.qt.border_enabled)

    def app_context(self, event_buffer) -> "AppContext":
        """What the App Server serves over this window's session."""
        raise NotImplementedError

    def show_toast(
        self,
        message: str,
        notification_type: str = "info",
        duration: int = 5000,
        temporary: bool = False,
        action=None,
        always: bool = False,
    ):
        """Show a toast notification, with an optional (text, callback) link.
        Every non-temporary message is kept in the notification bell too."""
        self.toast_manager.show_toast(
            message,
            notification_type,
            duration,
            temporary=temporary,
            action=action,
            always=always,
        )

    def _create_status_bar(self):
        """Create the status bar: the shared bar's line on the left, with
        the microscope's progress in it, and this window's workflow buttons on the
        right."""
        self.status_bar = FibsemStatusBar(self)
        self.setStatusBar(self.status_bar)
        # The agent's chip, beside the workflow buttons.
        self.agent_host = AgentChipHost(
            self.status_bar,
            session=lambda: self.session,
            view_controller=lambda: getattr(self, "view_controller", None),
            on_driving=self.run_border.set_agent_driving,
            parent=self,
        )

        # Attention Required, the supervision chip, Run and Stop: one component on the
        # right of the bar, which shows what it is told; the run view says what a
        # click does.
        self.workflow_controls = WorkflowControls(self.status_bar)
        self.status_bar.add_action(self.workflow_controls)

    def _follow_session(self) -> None:
        """Call `_on_session_changed` on the GUI thread for each of the
        session's changes, until the window closes."""
        self._stop_following_session = connect_on_gui_thread(
            self, self.session.changed, self._on_session_changed
        )

    def _on_session_changed(self, what: str, detail: Any = None) -> None:
        """The session's microscope, experiment or protocol changed."""

    def _attach_run(self, panel: "MicroscopePanel") -> None:
        """The session's run, responder and executor, shown on *panel*, and the
        connection widget, which owns the connection."""
        # Every run starts and ends through the session's runner, whoever
        # started it: the window hears the run's status, and redraws.
        self.run_relay = RunRelay(self.session.runner, parent=self)
        self.proposals = ProposalPresenter(
            self.session,
            panel,
            self.run_relay,
            view_controller=self._view_controller,
        )

        # The Qt side of the workflow's Responder seam: workflow code is handed
        # this one-method object, never the window itself.
        self.ui_responder = QtResponder(panel)
        self.session.responder = self.ui_responder
        # The operator decides the run's proposals here, for the session's life.
        self.session.attach_decider(OPERATOR)
        # Edits to the experiment run on this thread: its widgets listen to it.
        # The editors' pending edits are written first: an edit
        # from elsewhere to the same config then finds it changed.
        self.session.executor = GuiExecutor(self.flush_editors, parent=self)

        self.system_widget = FibsemSystemSetupWidget(parent=self)
        self.system_widget.connected_signal.connect(self.connect_to_microscope)
        self.system_widget.disconnected_signal.connect(self.disconnect_from_microscope)
        self.run_relay.status.connect(panel.handle_workflow_status)

    def flush_editors(self) -> None:
        """Write the edits the window's editors still hold, before
        anything else writes the experiment."""

    def _setup_experiment_connections(self) -> None:
        """Point the image settings at the experiment, and register the
        microscope's metadata with it. The panel follows the experiment itself."""
        if self.experiment is None:
            logging.warning("Cannot setup experiment connections: experiment is None")
            return
        if self.settings is not None:
            self.settings.image.path = self.experiment.path
        if self.microscope is not None:
            self.experiment.register_metadata(self.microscope)

    def connect_to_microscope(self):
        # The window hears the new microscope once its views here are built.
        with self.session.changed.paused():
            self.microscope = self.system_widget.microscope
            self.settings = self.system_widget.settings
            if self.experiment is not None:
                self.settings.image.path = self.experiment.path
            self._show_microscope()
            if self.experiment is not None:
                self._setup_experiment_connections()
            self._start_event_recorder()
            self._start_app_server()

    def _start_event_recorder(self) -> None:
        """Start recording events for this microscope connection, whatever else runs.

        This is the slot for the system widget's ``connected_signal``, which fires
        whenever that widget refreshes while connected -- not once per connection
        -- so it can run again for a microscope that already has a stream, and
        that microscope keeps it. A different microscope (the connection dialog can
        hand one back) gets a new stream, and the session stops a running App
        Server so it restarts on the new buffer.
        """
        self.session.start_recording()

    def _start_app_server(self) -> None:
        """Host the App Server over this session, if the preference asks for it.

        The agent starts at Watch; never raises — an optional observer
        must not be able to take the session down (see fibsem/server/hosting.py).
        """
        preference = fibsem_cfg.load_user_preferences().install.app_server
        if preference.enabled:
            if self.session.serve_microscope is None:
                # the operator's choice in Tools → App Server is per session,
                # like the agent level; it starts at the preference's default
                self.session.serve_microscope = preference.microscope
            self.session.start_app_server(self.app_context)

    def sync_app_server_with_preference(self) -> None:
        """Start or stop the embedded server to match the saved preference.

        Called when Tools → App Server saves the switch, so it acts now rather
        than at the next connect. Starting needs a connected microscope (the
        server wraps it); without one this is a no-op and the connect path
        picks the preference up as before.
        """
        host = self.session.app_server
        preference = fibsem_cfg.load_user_preferences().install.app_server
        if preference.enabled:
            if self.microscope is not None and (host is None or not host.running):
                self._start_app_server()
        elif host is not None and host.running:
            host.stop()

    def set_app_server_microscope(self, serve: bool) -> None:
        """Serve the microscope, or stop serving it, from Tools →
        App Server, which also saves the choice for the next launch.

        The services are fixed when the server starts, so a change restarts a
        running server. The agent's token and level are the session's and
        carry over, so a connected agent carries on.
        """
        serve = bool(serve)
        if serve == self.session.serve_microscope:
            return
        logging.info(
            "App Server: the microscope %s by the operator",
            "served" if serve else "no longer served",
        )
        if self.session.set_serve_microscope(serve) and self.microscope is not None:
            self._start_app_server()

    def _on_app_server_dialog(self) -> None:
        """Open Tools → App Server: status, services, clients, the event log."""
        from fibsem.ui.agent.app_server_dialog import (
            AppServerDialog,
        )

        preference = fibsem_cfg.load_user_preferences().install.app_server
        dialog = AppServerDialog(
            host_provider=lambda: self.session.app_server,
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
            lambda p: setattr(p.install.app_server, "enabled", bool(enabled))
        )
        self.sync_app_server_with_preference()

    def _set_app_server_microscope(self, serve: bool) -> None:
        """Tools → App Server: save whether the microscope is served, and act on it
        now (a running server restarts)."""
        self._preferences = fibsem_cfg.update_user_preferences(
            lambda p: setattr(p.install.app_server, "microscope", bool(serve))
        )
        self.set_app_server_microscope(serve)

    def disconnect_from_microscope(self):
        with self.session.changed.paused():
            self.session.stop_app_server()
            self.session.stop_recording()
            self.microscope = None
            self.settings = None
            self._show_microscope()

    def _show_microscope(self) -> None:
        """The panel's microscope tabs for the connected microscope, or none."""
        self.panel.show_microscope(self.microscope, self.settings)
        image_widget = self.panel.image_widget
        # The defaults' "Read from Acquire Tab" reads this tab.
        self.system_widget.set_current_imaging(
            image_widget._get_image_settings_from_ui
            if image_widget is not None
            else None
        )

    def _view_controller(self):
        """The quad view, once it is built."""
        return getattr(self, "view_controller", None)

    def _closing(self) -> None:
        """The app's own teardown as the window closes, after the edits are
        flushed and before the microscope is let go."""

    def closeEvent(self, event):
        """Flush what is still pending, then let the application go."""
        # Before anything else, so a cancelled close leaves the session untouched.
        if not self.run_view.confirm_close():
            event.ignore()
            return
        # The editors hold edits for a moment before writing them;
        # this is the last chance to get the final one onto disk.
        try:
            self.flush_editors()
        except Exception as e:
            logging.warning(f"Could not flush a pending experiment save on close: {e}")
        # persist the FM working state (channels / camera transform / objective)
        if self.panel.fm_control_widget is not None:
            try:
                self.panel.fm_control_widget.save_fm_configuration()
            except Exception as e:
                logging.warning(f"Could not save FM working state on close: {e}")
        self._closing()
        # After the flush, so its edit is recorded: write what is queued to
        # events.jsonl and stop the recorder's writer thread. Nothing else does,
        # so a window closed without disconnecting leaves it running.
        try:
            self.session.stop_recording()
        except Exception as e:
            logging.warning(f"Could not close the event recorder on close: {e}")
        # Last, so everything above has the microscope it needs. Let the
        # instrument go rather than leave the client open until the process ends.
        if self.microscope is not None:
            self.microscope.try_disconnect()
        self._stop_following_session()
        super().closeEvent(event)
        # Force the event loop to exit even if another top-level window (e.g. a
        # stray matplotlib pyplot figure) would otherwise keep the app alive once
        # this window closes. quit() — not os._exit — so atexit / experiment
        # autosave still run.
        app = QApplication.instance()
        if app is not None:
            app.quit()
