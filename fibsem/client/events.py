"""A server's events websocket, held open on a thread of its own.

The Device Server's ``WS /events`` and the App Server's ``WS /microscope/events``
send the same messages (`fibsem.server.devices.stream_events`). A stream that drops
is opened again until the server is back, each wait longer, up to the heartbeat.
"""

import json
import logging
import threading
from typing import Any, Callable, Dict, Mapping, Optional

from fibsem.server import missing_server_extra

HEARTBEAT = 5.0  # seconds between pings; a server silent for as long again is gone
OPEN_TIMEOUT = 5.0


def require_websockets(what: str) -> None:
    """Raise, saying what to install, when websockets (in the [server] extra)
    is not installed: *what* needs it."""
    missing = missing_server_extra(("websockets",))
    if missing is not None:
        raise ImportError(f"{what} needs {missing}")


class EventStream:
    """*url*'s events, each handed to *on_event* as a dict on the stream's thread.

    *on_open* runs as each stream opens, before its first event, with whether it
    is the first try; *on_lost* when an open stream drops. *headers* go on the
    opening request (the App Server's bearer token: never in the URL).
    """

    def __init__(
        self,
        url: str,
        on_event: Callable[[Dict[str, Any]], None],
        on_open: Callable[[bool], None] = lambda first: None,
        on_lost: Callable[[], None] = lambda: None,
        headers: Optional[Mapping[str, str]] = None,
        heartbeat: float = HEARTBEAT,
    ):
        self.url = url
        self.heartbeat = heartbeat
        self.connected = False
        self._on_event, self._on_open, self._on_lost = on_event, on_open, on_lost
        self._headers = dict(headers or {})
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._socket: Any = None
        self.closing = threading.Event()

    @property
    def started(self) -> bool:
        return self._thread is not None

    def start(self, wait: Optional[float] = OPEN_TIMEOUT) -> None:
        """Start following, once. With *wait*, until the first try opened or failed,
        so no event is missed from here on."""
        if self._thread is None:
            require_websockets(f"following {self.url}")
            self._thread = threading.Thread(target=self._listen, daemon=True)
            self._thread.start()
        if wait:
            self._ready.wait(wait)

    def _listen(self) -> None:
        delay, first = 0.5, True
        while not self.closing.is_set():
            try:
                self._stream(first)
                delay = 0.5  # it was up: retry quickly next time
            except Exception as error:  # refused, dropped or timed out: all the same
                if self.connected or first:
                    logging.warning(f"events from {self.url}: {error}")
            finally:
                first = False
                if self.connected:
                    self.connected = False
                    self._on_lost()
                self._ready.set()
            if self.closing.wait(delay):
                return
            delay = min(delay * 2, self.heartbeat)

    def _stream(self, first: bool) -> None:
        from websockets.sync.client import connect

        with connect(
            self.url,
            additional_headers=self._headers,
            open_timeout=OPEN_TIMEOUT,
            ping_interval=self.heartbeat,
            ping_timeout=self.heartbeat,
            close_timeout=self.heartbeat,  # a silent server won't answer the close
        ) as socket:
            self._socket = socket
            self.connected = True
            self._on_open(first)
            self._ready.set()
            for message in socket:
                self._on_event(json.loads(message))

    def close(self) -> None:
        self.closing.set()
        if self._socket is not None:
            self._socket.close()
