"""Embedded hosting: the agent server running inside the AutoLamella process.

This is the piece that turns the factory into the app's public interface: when
a microscope connects (and the ``agent_server_enabled`` preference is on), the
app builds ``build_server(microscope, app_context=AgentContext(ui, buffer))``
and runs it on a daemon thread. One process, one microscope connection, one
commander — the whole reason embedded hosting exists.

Threading notes, because this file is where the server meets Qt:

* The server runs via ``uvicorn.Server`` on a plain daemon thread with signal
  handling disabled — ``uvicorn.run()`` wants the main thread, which belongs
  to Qt. The thread owns no Qt objects (main-thread-only GC: off-thread Qt
  finalization is the Windows crash class).
* Request handlers only ever touch the app through the ``AgentContext``
  facade, which returns plain-data snapshots — nothing here adds a new way
  for a server thread to reach into widgets.
* ``stop()`` asks uvicorn to exit (``should_exit``) and joins briefly; the
  thread is a daemon either way, so a wedged shutdown cannot hang app exit.

Failure posture: this is an optional observer, so it must never take the app
down. ``start()`` catches its own failures, logs them, and leaves the app
exactly as if the feature were off.
"""

import atexit
import logging
import threading
import time
from typing import Callable, List, Optional

from fibsem.applications.autolamella.server.context import AgentContext
from fibsem.applications.autolamella.server.events import (
    EventBuffer,
    attach_microscope_taps,
    make_lifecycle_hook,
)

__all__ = ["AgentServerHost"]

_START_TIMEOUT_S = 10.0


class AgentServerHost:
    """Owns the embedded server's lifecycle: buffer, taps, thread, discovery."""

    def __init__(
        self, ui, host: str = "127.0.0.1", port: int = 8001, discovery_path=None
    ):
        self._ui = ui
        self._host = host
        self._port = port
        self._discovery_path = discovery_path
        self._server = None
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
    def url(self) -> str:
        return f"http://{self._host}:{self._port}"

    def start(self, microscope) -> bool:
        """Start serving. Returns False (and logs) rather than ever raising."""
        try:
            return self._start(microscope)
        except Exception:
            logging.exception("agent server failed to start; continuing without it")
            self.stop()
            return False

    def _start(self, microscope) -> bool:
        import uvicorn

        from fibsem.server import AuthConfig, build_server
        from fibsem.server.discovery import (
            read_discovery_file,
            remove_discovery_file,
            write_discovery_file,
        )

        if self.running:
            return True

        # One server per microscope/machine; a live bench server wins.
        existing = read_discovery_file(self._discovery_path)
        if existing is not None:
            logging.error(
                "agent server not started: another fibsem server is running "
                f"(pid {existing.get('pid')}, {existing.get('url')})"
            )
            return False

        # Read-only until scopes are armed; arming arrives with the GUI dialog.
        self.auth = AuthConfig.generate()
        self.event_buffer = EventBuffer()
        self.lifecycle_hook = make_lifecycle_hook(self.event_buffer)
        self._disposers = attach_microscope_taps(self.event_buffer, microscope)

        app = build_server(
            microscope,
            app_context=AgentContext(self._ui, event_buffer=self.event_buffer),
            auth=self.auth,
        )
        config = uvicorn.Config(
            app,
            host=self._host,
            port=self._port,
            log_level="warning",
            # No websocket support: nothing serves WS, and uvicorn's WS protocol
            # import warns via websockets.legacy — which the test suite's
            # warnings-as-errors policy would turn fatal inside this thread.
            ws="none",
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
                logging.error("agent server exited at startup (port in use?)")
            except Exception:
                logging.exception("agent server crashed")

        self._thread = threading.Thread(
            target=run_server, name="fibsem-agent-server", daemon=True
        )
        self._thread.start()

        deadline = time.monotonic() + _START_TIMEOUT_S
        while not self._server.started:
            if not self._thread.is_alive() or time.monotonic() > deadline:
                logging.error("agent server did not start (port in use?)")
                self.stop()
                return False
            time.sleep(0.05)

        write_discovery_file(
            self._host, self._port, self.auth, path=self._discovery_path
        )
        self._wrote_discovery = True
        atexit.register(self._remove_discovery)
        logging.info("agent server on %s — scopes: read only", self.url)
        logging.info("agent server token: %s", self.auth.token)
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
        self._remove_discovery()

    def _remove_discovery(self) -> None:
        if not self._wrote_discovery:
            return
        from fibsem.server.discovery import remove_discovery_file

        remove_discovery_file(self._discovery_path)
        self._wrote_discovery = False
