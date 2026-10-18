"""The running application, without Qt.

An :class:`AppSession` holds what a session of an application is: the
microscope it is connected to, the experiment it has open, the event stream of
that connection, whoever answers the run's questions, and the App Server over
them when one is hosted. A window attaches to
it as one client and reads and writes these through it; a headless server can
be another, so nothing here may need a widget.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from fibsem.acting import OPERATOR
from fibsem.workflows.event_recording import EventRecorder


class AppSession:
    """The microscope, the open experiment and the connection's event stream."""

    def __init__(self, responder: Optional[Any] = None) -> None:
        self.microscope: Optional[Any] = None
        self.experiment: Optional[Any] = None
        # Whoever answers the run's questions: the window's QtResponder, or a
        # server-side one when there is no window. The recorder records the
        # prompts it puts up.
        self.responder = responder
        # The connection's event stream (FIB-1031): made when a microscope
        # connects, whatever else runs, and recorded to events.jsonl in the
        # experiment's directory. None only while disconnected.
        self.recorder: Optional[EventRecorder] = None
        # The App Server over this session (FIB-845), while one is hosted.
        self.app_server: Optional[Any] = None
        # What outlives a restart of it: the agent's token and level
        # (``AuthConfig``), made when it first starts, and whether it serves
        # the microscope (FIB-1359), None until the client first says.
        self.agent_auth: Optional[Any] = None
        self.serve_microscope: Optional[bool] = None

    def adopt(self, experiment: Any) -> None:
        """Open ``experiment``: the session's, and the one its events are
        recorded to from now on."""
        self.experiment = experiment
        if self.recorder is not None:
            self.recorder.set_experiment(experiment.path, experiment)

    def start_recording(self) -> None:
        """Record events for the connected microscope.

        Keeps the stream it has for the same microscope: a client may ask again
        whenever it refreshes. A different microscope gets a new stream, and
        no microscope none. Never raises: the recording is a record of the
        connection, and failing to keep one must not stop the connection being
        used.
        """
        recorder = self.recorder
        if recorder is not None and recorder.microscope is self.microscope:
            return
        if recorder is not None:
            # A running server serves the old stream: stopped, so it is
            # started again on the new one rather than serve a closed one.
            self.stop_app_server()
        self.stop_recording()
        if self.microscope is None:
            return
        try:
            self.recorder = EventRecorder(
                self.microscope,
                responder=self.responder,
                experiment_path=self.experiment.path if self.experiment else None,
                # tasks and the agent mark their own calls; the rest are the
                # operator's
                default_actor=OPERATOR,
                experiment=self.experiment,
            )
        except Exception:
            logging.exception("event stream failed to start; continuing without it")
            self.recorder = None

    def stop_recording(self) -> None:
        """Close the event stream, whose writer thread would otherwise run
        until the process ends."""
        if self.recorder is not None:
            self.recorder.close()
            self.recorder = None

    def start_app_server(self, app_context: Callable[[Any], Any]) -> bool:
        """Host the App Server over this session, or keep the one running.

        ``app_context`` builds the application's ``/app`` view over an event
        buffer. Never raises (see ``fibsem.server.hosting``).
        """
        from fibsem.server import hosting

        if self.app_server is None:
            self.app_server = hosting.AppServerHost(
                self, app_context, serve_microscope=bool(self.serve_microscope)
            )
        return self.app_server.start()

    def stop_app_server(self) -> None:
        if self.app_server is not None:
            self.app_server.stop()

    def set_serve_microscope(self, serve: bool) -> bool:
        """Serve the microscope, or stop serving it. The routes are fixed when
        the server starts, so a running one is dropped; returns whether it was
        running, for the client to start it again."""
        self.serve_microscope = bool(serve)
        host = self.app_server
        if host is None or host.serve_microscope == self.serve_microscope:
            return False
        was_running = host.running
        host.stop()
        self.app_server = None
        return was_running
