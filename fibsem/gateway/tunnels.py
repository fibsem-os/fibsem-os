"""The hub's SSH tunnels: one per instrument, from this PC to the instrument's.

An instrument's gateway and App Server listen on that PC only, so the hub
reaches them through ``ssh -N -L``. The hub picks the local ports, keeps each
tunnel open, opens it again when it drops, and says why when it can't. It
never types a password: the key is set up once by a person (``BatchMode``),
and a host this PC hasn't met is refused until someone has connected by hand.
"""

from __future__ import annotations

import logging
import re
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from fibsem.util.process import end

__all__ = ["Tunnel", "TunnelDown", "Tunnels", "valid_destination"]

log = logging.getLogger(__name__)

# The ports an instrument PC serves on itself: its gateway and its App Server.
GATEWAY_PORT = 8010
APP_PORT = 8001
# How long a tunnel that failed waits before it is tried again.
RETRY_S = 15.0
# How long a new tunnel is given to accept a connection.
OPEN_WAIT_S = 8.0
END_WAIT_S = 3.0
# ``user@host`` or a name from the ssh configuration: never an option.
_DESTINATION = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.@-]{0,252}$")


class TunnelDown(OSError):
    """The tunnel to an instrument isn't open, with why."""


def valid_destination(destination: str) -> bool:
    return bool(_DESTINATION.match(destination))


def free_port() -> int:
    """A port on this PC nothing listens on now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _accepts(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def _spawn(command: List[str]) -> subprocess.Popen:
    return subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


@dataclass
class Tunnel:
    """One instrument's tunnel: the local ports that reach its gateway and
    its App Server while *process* runs."""

    destination: str
    gateway_port: int
    app_port: int
    process: Optional[subprocess.Popen] = None
    opened: Optional[float] = None
    failed: Optional[float] = None
    why: Optional[str] = None

    @property
    def open(self) -> bool:
        return self.process is not None and self.process.poll() is None

    @property
    def gateway_url(self) -> str:
        return f"http://127.0.0.1:{self.gateway_port}"

    @property
    def app_url(self) -> str:
        return f"http://127.0.0.1:{self.app_port}"


def command(tunnel: Tunnel, ssh: str = "ssh") -> List[str]:
    """The ssh command that holds *tunnel* open and does nothing else there."""
    return [
        ssh,
        "-N",
        "-o",
        "BatchMode=yes",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        "-L",
        f"127.0.0.1:{tunnel.gateway_port}:127.0.0.1:{GATEWAY_PORT}",
        "-L",
        f"127.0.0.1:{tunnel.app_port}:127.0.0.1:{APP_PORT}",
        "--",
        tunnel.destination,
    ]


class Tunnels:
    """The tunnels by instrument name. `ensure` gives an open one or raises
    `TunnelDown`; a tunnel that failed is tried again after `RETRY_S`."""

    def __init__(
        self,
        spawn: Callable[[List[str]], subprocess.Popen] = _spawn,
        port: Callable[[], int] = free_port,
        accepts: Callable[[int], bool] = _accepts,
        which: Callable[[str], Optional[str]] = shutil.which,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        open_wait_s: float = OPEN_WAIT_S,
    ) -> None:
        self._spawn = spawn
        self._port = port
        self._accepts = accepts
        self._which = which
        self._clock = clock
        self._sleep = sleep
        self._open_wait_s = open_wait_s
        self._lock = threading.Lock()
        self._locks: Dict[str, threading.Lock] = {}
        self._tunnels: Dict[str, Tunnel] = {}

    def get(self, name: str) -> Optional[Tunnel]:
        with self._lock:
            return self._tunnels.get(name)

    def ensure(self, name: str, destination: str) -> Tunnel:
        """The open tunnel to *name* at *destination*, opened now if it isn't."""
        with self._lock:
            lock = self._locks.setdefault(name, threading.Lock())
        with lock:
            tunnel = self.get(name)
            if tunnel is not None and tunnel.destination != destination:
                self._end(tunnel)
                tunnel = None
            if tunnel is not None and tunnel.open:
                return tunnel
            now = self._clock()
            if (
                tunnel is not None
                and tunnel.failed is not None
                and now - tunnel.failed < RETRY_S
            ):
                raise TunnelDown(tunnel.why or "The tunnel is down.")
            if tunnel is not None and tunnel.process is not None:
                # It was open and dropped: say so once, then open it again.
                log.warning("The tunnel to %s dropped; opening it again.", name)
            tunnel = self._open(name, destination)
            with self._lock:
                self._tunnels[name] = tunnel
            if not tunnel.open:
                raise TunnelDown(tunnel.why or "The tunnel is down.")
            return tunnel

    def _open(self, name: str, destination: str) -> Tunnel:
        gateway_port = app_port = self._port()
        while app_port == gateway_port:
            app_port = self._port()
        tunnel = Tunnel(destination, gateway_port, app_port)
        if not valid_destination(destination):
            return self._failed(tunnel, "Its ssh destination is not user@host.")
        ssh = self._which("ssh")
        if ssh is None:
            return self._failed(tunnel, "ssh isn't installed on this PC.")
        try:
            tunnel.process = self._spawn(command(tunnel, ssh))
        except OSError as error:
            return self._failed(tunnel, f"ssh didn't start: {error}")
        deadline = self._clock() + self._open_wait_s
        while tunnel.open and not self._accepts(tunnel.gateway_port):
            if self._clock() >= deadline:
                self._end(tunnel)
                return self._failed(
                    tunnel, f"ssh to {destination} didn't open within the wait."
                )
            self._sleep(0.1)
        if not tunnel.open:
            return self._failed(tunnel, _said(tunnel.process, destination))
        tunnel.opened = self._clock()
        log.info(
            "Opened the tunnel to %s (%s): gateway on %d, app on %d.",
            name,
            destination,
            tunnel.gateway_port,
            tunnel.app_port,
        )
        return tunnel

    def _failed(self, tunnel: Tunnel, why: str) -> Tunnel:
        tunnel.failed, tunnel.why = self._clock(), why
        return tunnel

    def _end(self, tunnel: Tunnel) -> None:
        if tunnel.process is not None and tunnel.process.poll() is None:
            end(tunnel.process, END_WAIT_S)

    def close(self, name: str) -> None:
        """Close *name*'s tunnel, if it has one."""
        with self._lock:
            tunnel = self._tunnels.pop(name, None)
        if tunnel is not None:
            self._end(tunnel)

    def close_all(self) -> None:
        with self._lock:
            names = list(self._tunnels)
        for name in names:
            self.close(name)


def _said(process: Optional[subprocess.Popen], destination: str) -> str:
    """Why ssh ended, in its own last words."""
    said = ""
    stderr = getattr(process, "stderr", None)
    if stderr is not None:
        try:
            said = stderr.read() or ""
        except (OSError, ValueError):
            said = ""
    lines = [line.strip() for line in said.splitlines() if line.strip()]
    last = lines[-1] if lines else "ssh ended without saying why."
    return f"ssh to {destination} failed: {last[:300]}"
