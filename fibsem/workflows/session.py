"""The running application, without Qt.

An :class:`AppSession` holds what a session of an application is: the
microscope it is connected to (connected and disconnected here, whoever asks),
the experiment it has open, the event stream of that connection, whoever
carries out the run's instructions in a window, where an edit to the
experiment runs, the run controller, and the App Server over them when one is
hosted. A window attaches to it as one client and reads and writes these
through it; the App Server is another, so nothing here may need a widget.
"""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import Future
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from psygnal import Signal

from fibsem import config as cfg
from fibsem.acting import AGENT, OPERATOR, ActorKind
from fibsem.ai.agent.session import AgentSession
from fibsem.devices.runs import Runs
from fibsem.imaging.latest import LatestImages
from fibsem.imaging.overview_session import OverviewSession
from fibsem.workflows.event_recording import EventRecorder
from fibsem.workflows.events import EventBuffer
from fibsem.workflows.runner import WorkflowRunner

EditExecutor = Callable[[Callable[[], Any]], "Future[Any]"]

# Where a session's plain experiments go, one folder per session that needed one.
SESSIONS_PATH = os.path.join(cfg.LOG_PATH, "sessions")

# What `AppSession.changed` names. The last three are the open experiment
# edited where no widget drew the edit (the App Server): whoever draws it redraws.
MICROSCOPE = "microscope"
EXPERIMENT = "experiment"
PROTOCOL = "protocol"
GRID_RECORDS = "grid_records"
# With what the edit was: its level, item_name, task_name and result.
CONFIG = "config"


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
    """The microscope, the open experiment and the connection's event stream.

    ``changed`` fires with ``MICROSCOPE`` when another microscope (or none) is
    connected, with ``EXPERIMENT`` when an experiment is opened
    (:meth:`adopt`), and with ``PROTOCOL``, ``GRID_RECORDS`` or ``CONFIG`` (and
    what the edit was) when the open experiment is edited from elsewhere, on
    the thread that changed it.
    """

    changed = Signal(str, object)

    def __init__(
        self,
        responder: Optional[Any] = None,
        executor: Optional[EditExecutor] = None,
    ) -> None:
        self._microscope: Optional[Any] = None
        # The connected microscope's settings (its configuration and image
        # settings), None while disconnected.
        self.settings: Optional[Any] = None
        # The configuration file the microscope was connected with.
        self.configuration_path: Optional[str] = None
        # What a connect is doing now ("Turning the beams on…"), and why the
        # last one failed; a connect from the App Server runs as a run here.
        self.connecting: Optional[str] = None
        self.connect_error: Optional[str] = None
        self.connect_runs = Runs(_Connecting())
        self._connect_lock = threading.Lock()
        self._experiment: Optional[Any] = None
        # Where what is done is recorded while no application's experiment is
        # open: a plain FibsemExperiment, made the first time the agent may act.
        self.plain_experiment: Optional[Any] = None
        # Whoever carries out the run's instructions (show these images, clear
        # the milling editor): the window's QtResponder, or none headless.
        self.responder = responder
        # The connection's event stream: made when a microscope
        # connects, whatever else runs, and recorded to events.jsonl in the
        # experiment's directory. None only while disconnected.
        self.recorder: Optional[EventRecorder] = None
        # The stream while there is no recorder: the agent's messages and
        # calls before a microscope connects.
        self._idle_buffer = EventBuffer()
        # Where an edit to the experiment runs: the window's GUI thread, whose
        # widgets listen to the experiment, or here under a lock.
        self.executor: EditExecutor = executor or LockedExecutor()
        # The latest image of each beam, as the window shows it: what the App
        # Server serves and a question's marker is placed on.
        self.images = LatestImages()
        # Each overview of the connected microscope, by kind.
        self._overviews: Dict[str, OverviewSession] = {}
        # Where every run starts, whoever asks.
        self.runner = WorkflowRunner(self)
        # The App Server over this session, while one is hosted.
        self.app_server: Optional[Any] = None
        # What outlives a restart of it: the agent's token and level
        # (``AuthConfig``), made when it first starts, and whether it serves
        # the microscope, None until the client first says.
        self.agent_auth: Optional[Any] = None
        self.serve_microscope: Optional[bool] = None
        # Where an agent creates and finds experiments this session; None is
        # the app's default (fibsem-autolamella-server --experiments sets it).
        self.experiments_folder: Optional[str] = None
        # The agent as the operator meets it: its level, its requests and the
        # operator's messages.
        self.agent = AgentSession(self)
        # The surfaces attached to decide a run's proposals (a window: the
        # operator), while they are attached.
        self._deciders: List[ActorKind] = []
        self._analysis_services: Optional[Dict[str, Any]] = None
        # The correlation (``CorrelationSession``), made under the lock the
        # first time any thread asks for it.
        self._correlation: Optional[Any] = None
        self._correlation_lock = threading.Lock()

    def overview(self, kind: str) -> OverviewSession:
        """The connected microscope's beam (``"beam"``) or fluorescence
        (``"fm"``) overview: what its tab shows and does. Made on first use and
        kept until the microscope changes. Raises ValueError with no microscope,
        or for an FM overview on one with no FM."""
        if self.microscope is None:
            raise ValueError("No microscope is connected.")
        overview = self._overviews.get(kind)
        if overview is None or overview.microscope is not self.microscope:
            overview = OverviewSession(self.microscope, kind)
            self._overviews[kind] = overview
        return overview

    @property
    def analysis_services(self) -> Dict[str, Any]:
        """The analysis services (`fibsem.services.analysis`), made on first
        use and kept for the session: the App Server serves these and the
        widgets use the same ones, so a loaded model outlives a restart of it."""
        if self._analysis_services is None:
            from fibsem.services.analysis import analysis_services

            self._analysis_services = analysis_services()
        return self._analysis_services

    def attach_decider(self, who: ActorKind) -> Callable[[], None]:
        """``who`` is here to decide a run's proposals, until the returned
        detach is called: the window attaches the operator."""
        self._deciders.append(who)

        def detach() -> None:
            try:
                self._deciders.remove(who)
            except ValueError:
                pass

        return detach

    def deciders(self) -> Tuple[ActorKind, ...]:
        """Who can decide a proposal now: each attached surface, and the agent
        while the App Server serves it at Drive and it is listening (heard in
        the last ``AGENT_LISTENING_S``). Empty means nobody: a supervised
        proposal still waits, for someone to come back."""
        from fibsem.ai.agent.levels import AGENT_LISTENING_S, AgentLevel

        who = list(dict.fromkeys(self._deciders))
        host = self.app_server
        if (
            host is not None
            and host.running
            and self.agent_auth is not None
            and self.agent_auth.level is AgentLevel.DRIVE
            and AGENT not in who
        ):
            seen = self.agent.seconds_since_seen()
            if seen is not None and seen < AGENT_LISTENING_S:
                who.append(AGENT)
        return tuple(who)

    @property
    def microscope(self) -> Optional[Any]:
        return self._microscope

    @microscope.setter
    def microscope(self, microscope: Optional[Any]) -> None:
        if microscope is self._microscope:
            return
        self._microscope = microscope
        self.images.follow(microscope)
        self.changed.emit(MICROSCOPE)

    def connect(
        self,
        configuration_path: Optional[str] = None,
        progress: Optional[Callable[[str], None]] = None,
        **options: Any,
    ) -> Any:
        """Connect the microscope *configuration_path* names (the default
        configuration when None), record its events, and say so on
        ``changed``. Runs on the calling thread; *progress* hears each step.
        *options* go to `utils.setup_session` (``manufacturer``,
        ``ip_address``, ``debug``). Raises ValueError while a microscope is
        connected or another connect is under way, and whatever connecting
        raised."""
        from fibsem import utils

        if not self._connect_lock.acquire(blocking=False):
            raise ValueError("A connect is already under way.")
        try:
            if self.microscope is not None:
                raise ValueError("A microscope is connected; disconnect it first.")
            self.connect_error = None

            def step(line: str) -> None:
                self.connecting = line
                if progress is not None:
                    progress(line)

            step("Connecting…")
            try:
                microscope, settings = utils.setup_session(
                    config_path=configuration_path, progress=step, **options
                )
            except Exception as error:
                self.connect_error = str(error) or type(error).__name__
                raise
            finally:
                self.connecting = None
            # Heard once everything here is the new microscope's.
            with self.changed.paused():
                self.settings = settings
                self.configuration_path = getattr(
                    microscope, "configuration_path", None
                ) or (str(configuration_path) if configuration_path else None)
                self.microscope = microscope
                self.start_recording()
            return microscope
        finally:
            self._connect_lock.release()

    def disconnect(self) -> Optional[str]:
        """Let the microscope go and stop recording its events; the App Server
        stays up. Refused (RuntimeError) while a run is on. Returns why the
        client did not close cleanly, or None: the session lets go of it
        either way."""
        if self.microscope is None:
            return None
        if self.runner.is_running:
            raise RuntimeError("A run is on: stop it before disconnecting.")
        microscope = self.microscope
        error = None
        try:
            microscope.disconnect()
        except Exception as e:  # noqa: BLE001 - a failed teardown still ends it
            logging.error(f"Could not cleanly disconnect the microscope: {e}")
            error = str(e)
        with self.changed.paused():
            self.stop_recording()
            self.settings = None
            self.configuration_path = None
            self.microscope = None
        return error

    @property
    def connection(self) -> Dict[str, Any]:
        """Whether a microscope is connected, with which configuration, what a
        connect is doing or why the last failed, and the configurations it can
        connect with."""
        from fibsem import config

        microscope = self.microscope
        info = (
            getattr(getattr(self.settings, "system", None), "info", None)
            if microscope is not None
            else None
        )
        path = self.configuration_path
        name = next(
            (
                known
                for known, entry in config.USER_CONFIGURATIONS.items()
                if path and entry.get("path") == path
            ),
            None,
        )
        run = self.connect_runs.latest
        return {
            "connected": microscope is not None,
            "connecting": self.connecting,
            "error": self.connect_error,
            "configuration": None
            if microscope is None
            else {
                "name": name,
                "path": path,
                "manufacturer": getattr(info, "manufacturer", None),
                "model": getattr(info, "model", None),
                "address": getattr(info, "ip_address", None),
            },
            "configurations": [
                {"name": known, "default": known == config.DEFAULT_CONFIGURATION_NAME}
                for known, entry in config.USER_CONFIGURATIONS.items()
                if entry.get("path")
            ],
            "run": run.to_dict() if run is not None else None,
        }

    @property
    def experiment(self) -> Optional[Any]:
        return self._experiment

    @experiment.setter
    def experiment(self, experiment: Optional[Any]) -> None:
        """The open experiment, edited where the session edits it
        (``ProposalDecisions.edit_executor``)."""
        self._experiment = experiment
        if experiment is not None:
            experiment.edit_executor = self.run_edit

    @property
    def correlation(self) -> Any:
        """The session's correlation (``CorrelationSession``), made when first
        asked for, so nothing here imports the correlation package until then.
        A new experiment starts a new one (see :meth:`adopt`)."""
        with self._correlation_lock:
            if self._correlation is None:
                from fibsem.correlation.session import CorrelationSession

                self._correlation = CorrelationSession()
            return self._correlation

    def run_edit(self, call: Callable[[], Any]) -> "Future[Any]":
        """Run ``call()`` where the experiment is edited; any thread. Resolves
        to its result, or its exception."""
        return self.executor(call)

    def adopt(self, experiment: Any) -> None:
        """Open ``experiment``: the session's, the one its events are recorded
        to and the one the process logs to from now on. The correlation belongs
        to the experiment it was made in, so the next one asked for is a fresh
        one."""
        with self._correlation_lock:
            self._correlation = None
        if self.recorder is not None:
            self.recorder.set_experiment(experiment.path, experiment)
        self.experiment = experiment
        experiment.configure_logging()
        logging.info(f"Logging to experiment {experiment.name} at {experiment.path}")
        self.changed.emit(EXPERIMENT)

    def recording_experiment(self) -> Any:
        """The experiment what is done now is recorded to: the open one, or else
        the session's plain one, made under the log folder the first time it is
        needed. Raises OSError when it cannot be made."""
        if self.experiment is not None:
            return self.experiment
        if self.plain_experiment is None:
            from fibsem.workflows.experiment import FibsemExperiment

            name = datetime.now().strftime("session-%Y-%m-%d-%H-%M-%S")
            self.plain_experiment = FibsemExperiment.create(SESSIONS_PATH, name)
            if self.recorder is not None:
                self.recorder.set_experiment(
                    self.plain_experiment.path, self.plain_experiment
                )
        return self.plain_experiment

    @property
    def event_buffer(self) -> EventBuffer:
        """The session's event stream: the recorder's while a microscope is
        connected, else the session's own. A reader tells them apart by the
        buffer's ``stream``."""
        recorder = self.recorder
        return recorder.buffer if recorder is not None else self._idle_buffer

    def record_event(self, kind: str, payload: Dict[str, Any]) -> None:
        """Put an event on the session's record, as whoever marked this
        thread."""
        self.event_buffer.append(kind, payload)

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
        self.stop_recording()
        if self.microscope is None:
            return
        recorded = self.experiment or self.plain_experiment
        try:
            self.recorder = EventRecorder(
                self.microscope,
                experiment_path=recorded.path if recorded else None,
                # tasks and the agent mark their own calls; the rest are the
                # operator's
                default_actor=OPERATOR,
                experiment=recorded,
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

    def start_app_server(self, app_context: Callable[[], Any]) -> bool:
        """Host the App Server over this session, or keep the one running.

        ``app_context`` builds the application's ``/app`` view over the
        session. Never raises (see ``fibsem.server.hosting``).
        """
        from fibsem.server import hosting

        if self.app_server is None:
            self.app_server = hosting.AppServerHost(
                self, app_context, serve_microscope=self.serve_microscope is not False
            )
        started = self.app_server.start()
        if started:
            self.agent.keep_pictures()
        return started

    def stop_app_server(self) -> None:
        self.agent.stop_pictures()
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


class _Connecting:
    """What the session's connect runs are on (`Runs`): the connection."""

    name = "connection"
    session = None
