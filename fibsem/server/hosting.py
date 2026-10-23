"""Embedded hosting: the App Server running inside an application's process.

This is the piece that turns the factory into the app's public interface: when
a microscope connects (and the App Server preference, ``install.app_server.enabled``, is on), the
app's session builds ``build_server(microscope, app_context=...)`` and runs it on a
daemon thread. One process, one microscope connection, one commander — the whole
reason embedded hosting exists. The microscope, the event stream
(``event_recording.EventRecorder``, which exists whether or not this runs), the
responder and the agent's token are the session's (``fibsem.workflows.session``);
the ``/app`` view over them is the application's own (AutoLamella's is
``AutoLamellaAppContext``).

Threading notes, because this file is where the server meets Qt:

* The server runs via ``uvicorn.Server`` on a plain daemon thread with signal
  handling disabled — ``uvicorn.run()`` wants the main thread, which belongs
  to Qt. The thread owns no Qt objects (main-thread-only GC: off-thread Qt
  finalization is the Windows crash class).
* Request handlers only ever touch the app through its app context facade, which returns plain-data snapshots — nothing here adds a new way
  for a server thread to reach into widgets.
* ``stop()`` asks uvicorn to exit (``should_exit``) and joins briefly; the
  thread is a daemon either way, so a wedged shutdown cannot hang app exit.

Failure posture: this is an optional observer, so it must never take the app
down. ``start()`` catches its own failures, logs them, and leaves the app
exactly as if the feature were off.
"""

import atexit
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, List, Optional

from fibsem.acting import AGENT
from fibsem.cancellation import NothingToStop
from fibsem.workflows.events import (
    EventBuffer,
    attach_microscope_taps,
    make_lifecycle_hook,
)

__all__ = ["AppServerHost", "chat_url", "event_log_url"]


def _stop_the_microscope(microscope) -> None:
    """Stop whatever the microscope is running: milling, a spot burn, an auto
    focus sweep, a deposition, live imaging. Each stop is safe when nothing is running (the
    milling service's says so, and is let be), and one failing never keeps the
    others from being tried."""
    stops = [microscope.stop_acquisition]
    # The milling service's stop, not the microscope's: it ends a milling task
    # as well as its pattern, so no later stage starts.
    for service in ("milling", "spot_burn", "imaging", "coating"):
        found = getattr(microscope, service, None)
        if found is not None:
            stops.append(found.stop)
    for stop in stops:
        try:
            stop()
        except NothingToStop:
            pass
        except Exception:
            logging.exception("Stop agent: %s failed", getattr(stop, "__name__", stop))


_START_TIMEOUT_S = 10.0


def event_log_url(host) -> str:
    """The web app's Events page with the read-only dashboard token, never
    the agent's, in the URL *fragment* — the one part of a URL a browser
    never sends anywhere, so it reaches the page's own script and nothing
    else (no server log, no referrer)."""
    return f"{host.url}/ui/events#token={host.auth.dashboard_token}"


def chat_url(host) -> str:
    """The web app's Agent page, with the operator token in the fragment."""
    return f"{host.url}/ui/agent#token={host.auth.operator_token}"


class AppServerHost:
    """Owns the embedded server's lifecycle: buffer, taps, thread, discovery.

    ``session`` is the application's ``AppSession``. ``app_context`` builds the
    app's ``/app`` view over an event buffer: the session's stream, or one the
    host feeds itself when the session records none.
    """

    def __init__(
        self,
        session,
        app_context: Callable[[EventBuffer], Any],
        host: str = "127.0.0.1",
        port: int = 8001,
        discovery_path=None,
        serve_microscope: bool = False,
    ):
        self._session = session
        self._app_context = app_context
        # Whether the /microscope routes are served. Fixed for the
        # host's life; a change builds a new one.
        self.serve_microscope = serve_microscope
        self._host = host
        self._port = port
        self._discovery_path = discovery_path
        # The operator token's file sits beside a given discovery file (tests),
        # else in ~/.fibsem.
        self._operator_token_path = (
            Path(discovery_path).parent / "operator-token.json"
            if discovery_path is not None
            else None
        )
        self._paired_path = (
            Path(discovery_path).parent / "paired.json"
            if discovery_path is not None
            else None
        )
        self._server = None
        # The running app: the window reads the agent's requests from it.
        self._app = None
        self._thread: Optional[threading.Thread] = None
        self._disposers: List[Callable[[], None]] = []
        self._wrote_discovery = False
        self.auth = None
        self.event_buffer: Optional[EventBuffer] = None
        self.lifecycle_hook = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def requests(self):
        """The agent's requests while the server runs, else None."""
        if not self.running or self._app is None:
            return None
        return getattr(self._app.state, "requests", None)

    @property
    def app_context(self):
        """The app's ``/app`` view while the server runs, else None."""
        if not self.running or self._app is None:
            return None
        return getattr(self._app.state, "app_context", None)

    def agent_seconds_since_seen(self) -> Optional[float]:
        """Seconds since the agent's token last made a request, or None if the
        server isn't running or nothing has connected yet. A live supervising
        agent long-polls the event stream, so this stays small while one is
        attached — staleness means nobody is on the other end."""
        if not self.running or self.auth is None:
            return None
        return self.auth.seconds_since_seen()

    def agent_client(self) -> Optional[str]:
        """The MCP client the agent's sidecar named ("claude-code/2.1.3"), and
        the paired computer it runs on ("claude-code/2.1.3 on mac-mini"), or
        None if the server isn't running or no sidecar has named one."""
        if not self.running or self.auth is None:
            return None
        client, computer = self.auth.agent_client, self.auth.agent_computer
        if computer is None:
            return client
        return f"{client} on {computer}" if client else computer

    def new_operator_token(self) -> Optional[str]:
        """Replace the operator token, on disk and on the running server: the
        old one stops working at once. None while the server isn't running."""
        from fibsem.server.operator_token import new_operator_token

        if self.auth is None:
            return None
        self.auth.operator_token = new_operator_token(self._operator_token_path)
        logging.info("New operator token made; the old one no longer works")
        return self.auth.operator_token

    def agent_session(self) -> Optional[dict]:
        """The agent's session as its harness last reported it (model, goal,
        budget, spend), or None if the server isn't running or none reported."""
        if not self.running or self.auth is None:
            return None
        return self.auth.agent_session

    def stop_agent(self, microscope) -> Optional[str]:
        """The operator's Stop agent: down to Watch at once, and stop
        what the agent is doing on the microscope.

        The agent's action is stopped when the agent is in control, or when it
        was the last to take control and nothing else has since: an action
        like start_milling lets go of control while the mill runs on. A run's
        or a script's action is never stopped here; they have their own Stop.
        Stopping takes no control, so nothing refuses it. A stage move has no
        stop and finishes.

        Returns the stopped action, or None when the agent was doing nothing.
        """
        if self.auth is not None:
            self.auth.stop_agent()
        control = getattr(microscope, "control", None)
        if control is None:
            return None
        current, last = control.current, control.last
        acting = current if current is not None else last
        if acting is None or acting.by != AGENT:
            return None
        _stop_the_microscope(microscope)
        return acting.action

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}"

    def _connections(self):
        """Asks to connect, allowed at once from this PC while the operator's
        setting says so; an Allow by the setting goes on the record as theirs."""
        from fibsem import config
        from fibsem.acting import OPERATOR
        from fibsem.server.connections import Connections
        from fibsem.workflows.actions import ACTION, action_record

        def save(on: bool) -> None:
            config.update_user_preferences(
                lambda p: setattr(p.install.app_server, "allow_local_agents", on)
            )

        connections = Connections(
            self.auth.pairing.tokens,
            always_allow=config.load_user_preferences().install.app_server.allow_local_agents,
            save_always_allow=save,
        )

        def record(computer, replaced) -> None:
            logging.info("Allowed %s to connect (always allowed)", computer.name)
            if self.event_buffer is None:
                return
            self.event_buffer.append(
                ACTION,
                action_record(
                    f"Allow {computer.name} to connect",
                    reason="Always allow agents logged in to this PC is on"
                    + (f"; replaces {replaced}" if replaced else ""),
                ),
                actor=OPERATOR,
            )

        connections.on_allowed_always(record)
        return connections

    def start(self) -> bool:
        """Serve the session's microscope. Returns False (and logs) rather than
        ever raising.

        The session's event stream (``event_recording.EventRecorder``) has its
        taps and prompt feed attached already and outlives the server. A
        session recording none gets a buffer the host builds and feeds itself.
        """
        try:
            return self._start()
        except Exception:
            logging.exception("App Server failed to start; continuing without it")
            self.stop()
            return False

    def _start(self) -> bool:
        import uvicorn

        from fibsem.ai.agent.levels import AgentLevel
        from fibsem.server import AuthConfig, build_server
        from fibsem.server.devices import WS_PROTOCOL
        from fibsem.server.discovery import (
            read_discovery_file,
            write_discovery_file,
        )

        if self.running:
            return True

        # One server per microscope/machine; a live one wins.
        existing = read_discovery_file(self._discovery_path)
        if existing is not None:
            logging.error(
                "App Server not started: another fibsem server is running "
                f"(pid {existing.get('pid')}, {existing.get('url')})"
            )
            return False

        # Watch until the operator raises the level from the Agent chip.
        # FIBSEM_AGENT_ARM_CONTROL=1 is a developer override to start at Drive:
        # set by whoever launches the app, never by a config file, and loudly
        # logged.
        arm_control = os.environ.get("FIBSEM_AGENT_ARM_CONTROL") == "1"
        if arm_control:
            logging.warning(
                "App Server: agent level DRIVE via FIBSEM_AGENT_ARM_CONTROL "
                "(developer override)"
            )
        # One AuthConfig per app session, kept on the session across server
        # restarts (a microscope reconnect, a microscope toggle): the agent's
        # token stays valid, so a connected sidecar carries on, and the level
        # the operator set stands. It starts at Watch at every launch and is
        # never written to disk. The override is Drive.
        session = self._session
        microscope = session.microscope
        if session.agent_auth is None:
            session.agent_auth = AuthConfig.generate(
                level=AgentLevel.DRIVE if arm_control else AgentLevel.WATCH
            )
        self.auth = session.agent_auth
        if self.auth.operator_token is None:
            from fibsem.server.operator_token import load_operator_token

            self.auth.operator_token = load_operator_token(self._operator_token_path)
        if self.auth.pairing is None:
            from fibsem.server.pairing import PairedTokens, Pairing

            self.auth.pairing = Pairing(PairedTokens(self._paired_path))
        if self.auth.connections is None:
            self.auth.connections = self._connections()
        recorder = session.recorder
        if recorder is not None:
            # The app's stream: already fed, and not ours to detach on stop.
            self.event_buffer = recorder.buffer
            self.lifecycle_hook = recorder.lifecycle_hook
        else:
            self.event_buffer = EventBuffer()
            self.lifecycle_hook = make_lifecycle_hook(self.event_buffer)
            self._disposers = attach_microscope_taps(self.event_buffer, microscope)

        app = build_server(
            microscope,
            app_context=self._app_context(self.event_buffer),
            auth=self.auth,
            serve_microscope=self.serve_microscope,
            services=session.analysis_services,
            operator=getattr(session, "agent", None),
        )
        self._app = app
        config = uvicorn.Config(
            app,
            host=self._host,
            port=self._port,
            log_level="warning",
            ws=WS_PROTOCOL,  # for /microscope/events
        )
        self._server = uvicorn.Server(config)
        # Signal handlers belong to the main thread, which belongs to Qt.
        self._server.install_signal_handlers = lambda: None  # type: ignore[method-assign]

        def run_server(server=self._server):
            # uvicorn sys.exit()s on startup failure (e.g. port in use); a bare
            # SystemExit escaping a thread is noise at best and, under a test
            # suite's thread-exception hooks, a failure. Contain and log.
            try:
                server.run()
            except SystemExit:
                logging.error("App Server exited at startup (port in use?)")
            except Exception:
                logging.exception("App Server crashed")

        self._thread = threading.Thread(
            target=run_server, name="fibsem-app-server", daemon=True
        )
        self._thread.start()

        deadline = time.monotonic() + _START_TIMEOUT_S
        while not self._server.started:
            if not self._thread.is_alive() or time.monotonic() > deadline:
                logging.error("App Server did not start (port in use?)")
                self.stop()
                return False
            time.sleep(0.05)

        write_discovery_file(
            self._host, self._port, self.auth, path=self._discovery_path
        )
        self._wrote_discovery = True
        atexit.register(self._remove_discovery)
        # Never a token: a log is read by more people than the discovery
        # file (0600).
        logging.info(
            "App Server on %s — agent level: %s", self.url, self.auth.level.value
        )
        return True

    def stop(self) -> None:
        for dispose in self._disposers:
            try:
                dispose()
            except Exception:
                pass
        self._disposers = []
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._thread = None
        self._server = None
        # A restart drops requests: say so, rather than leave one waiting.
        store = getattr(getattr(self._app, "state", None), "requests", None)
        if store is not None:
            store.cancel("the App Server stopped")
        self._app = None
        self._remove_discovery()

    def _remove_discovery(self) -> None:
        if not self._wrote_discovery:
            return
        from fibsem.server.discovery import remove_discovery_file

        remove_discovery_file(self._discovery_path)
        self._wrote_discovery = False
