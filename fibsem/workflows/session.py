"""The running application, without Qt.

An :class:`AppSession` holds what a session of an application is: the
microscope it is connected to, the experiment it has open, the event stream of
that connection, whoever answers the run's questions, where an edit to the
experiment runs, and the App Server over them when one is hosted. A window
attaches to it as one client and reads and writes these through it; a headless
server can be another, so nothing here may need a widget.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future
from typing import Any, Callable, Dict, Optional

from fibsem.acting import OPERATOR
from fibsem.imaging.latest import LatestImages
from fibsem.workflows.event_recording import EventRecorder

EditExecutor = Callable[[Callable[[], Any]], "Future[Any]"]


class LockedExecutor:
    """Runs each edit in the calling thread, one at a time: where an edit runs
    when no window owns the experiment."""

    def __init__(self) -> None:
        self._lock = threading.RLock()

    def __call__(self, call: Callable[[], Any]) -> "Future[Any]":
        outcome: "Future[Any]" = Future()
        with self._lock:
            try:
                outcome.set_result(call())
            except Exception as exc:  # noqa: BLE001 - the requester owns the failure
                outcome.set_exception(exc)
        return outcome


class AppSession:
    """The microscope, the open experiment and the connection's event stream."""

    def __init__(
        self,
        responder: Optional[Any] = None,
        executor: Optional[EditExecutor] = None,
    ) -> None:
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
        # Where an edit to the experiment runs: the window's GUI thread, whose
        # widgets listen to the experiment, or here under a lock.
        self.executor: EditExecutor = executor or LockedExecutor()
        # The latest image of each beam, as the window shows it: what the App
        # Server serves and a question's marker is placed on (FIB-1451).
        self.images = LatestImages()
        # The App Server over this session (FIB-845), while one is hosted.
        self.app_server: Optional[Any] = None
        # What outlives a restart of it: the agent's token and level
        # (``AuthConfig``), made when it first starts, and whether it serves
        # the microscope (FIB-1359), None until the client first says.
        self.agent_auth: Optional[Any] = None
        self.serve_microscope: Optional[bool] = None
        self._analysis_services: Optional[Dict[str, Any]] = None

    @property
    def analysis_services(self) -> Dict[str, Any]:
        """The analysis services (`fibsem.services.analysis`), made on first
        use and kept for the session: the App Server serves these and the
        widgets use the same ones, so a loaded model outlives a restart of it."""
        if self._analysis_services is None:
            from fibsem.services.analysis import analysis_services

            self._analysis_services = analysis_services()
        return self._analysis_services

    def run_edit(self, call: Callable[[], Any]) -> "Future[Any]":
        """Run ``call()`` where the experiment is edited; any thread. Resolves
        to its result, or its exception."""
        return self.executor(call)

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
