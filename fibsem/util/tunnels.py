"""SSH tunnels from this PC to others, kept open by name.

A server that listens on its own PC only is reached from another PC through
``ssh -N``: a forward (``-L``) brings one of its ports here, and a reverse
forward (``-R``) lends it one of ours. A tunnel forwards the ports it is
given and nothing else. `Tunnels` picks the local port, keeps each tunnel
open, opens it again when it drops, and says why when it can't. It never
types a password: the key is set up once by a person (``BatchMode``), and a
host this PC hasn't met is refused until someone has connected by hand.

Whoever needs a tunnel uses this: a hub to each instrument's gateway, a
gateway to a device's PC.
"""

from __future__ import annotations

import logging
import re
import shutil
import socket
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from fibsem.util.process import end

__all__ = ["Tunnel", "TunnelDown", "Tunnels", "valid_destination"]

log = logging.getLogger(__name__)

# How long a tunnel that failed waits before it is tried again.
RETRY_S = 15.0
# How long a new tunnel is given to accept a connection.
OPEN_WAIT_S = 8.0
END_WAIT_S = 3.0
# How long an ssh with no forward to this PC is watched before it counts as
# open: one whose forward was refused ends within it (ExitOnForwardFailure).
SETTLE_S = 1.5
# ``user@host`` or a name from the ssh configuration: never an option.
_DESTINATION = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.@-]{0,252}$")


class TunnelDown(OSError):
    """A tunnel isn't open, with why."""


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


# How many of ssh's last lines are kept to say why it ended.
LAST_WORDS = 20


def _listen(process: subprocess.Popen) -> None:
    """Read what ssh says as it says it, keeping its last lines.

    ssh writes a line each time a forward can't connect, which is every
    read while a gateway is down. Left unread they fill the pipe and ssh
    stops, so the tunnel hangs."""
    said: deque = deque(maxlen=LAST_WORDS)
    stderr = process.stderr

    def read() -> None:
        try:
            for line in stderr:  # type: ignore[union-attr]
                said.append(line)
        except (OSError, ValueError):
            pass
        finally:
            try:
                stderr.close()  # type: ignore[union-attr]
            except OSError:
                pass

    listener = threading.Thread(target=read, name="hub-ssh-said", daemon=True)
    process.last_words = said  # type: ignore[attr-defined]
    process.listener = listener  # type: ignore[attr-defined]
    listener.start()


def _spawn(command: List[str]) -> subprocess.Popen:
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    _listen(process)
    return process


@dataclass
class Tunnel:
    """One tunnel to *destination* while *process* runs.

    With a *target*, *port* on this PC reaches that port there. Each pair in
    *reverse* is (a port there, a port here): the first reaches the second.
    All of it is on 127.0.0.1 at both ends."""

    destination: str
    port: int
    target: Optional[int] = None
    reverse: Tuple[Tuple[int, int], ...] = ()
    process: Optional[subprocess.Popen] = None
    opened: Optional[float] = None
    failed: Optional[float] = None
    why: Optional[str] = None

    @property
    def open(self) -> bool:
        return self.process is not None and self.process.poll() is None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def forwards(tunnel: Tunnel) -> List[str]:
    """What *tunnel* forwards, as ssh's own options."""
    said: List[str] = []
    if tunnel.target is not None:
        said += ["-L", f"127.0.0.1:{tunnel.port}:127.0.0.1:{tunnel.target}"]
    for there, here in tunnel.reverse:
        said += ["-R", f"127.0.0.1:{there}:127.0.0.1:{here}"]
    return said


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
        *forwards(tunnel),
        "--",
        tunnel.destination,
    ]


class Tunnels:
    """The tunnels by name. `ensure` gives an open one or raises
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
        settle_s: float = SETTLE_S,
    ) -> None:
        self._spawn = spawn
        self._port = port
        self._accepts = accepts
        self._which = which
        self._clock = clock
        self._sleep = sleep
        self._open_wait_s = open_wait_s
        self._settle_s = settle_s
        self._lock = threading.Lock()
        self._locks: Dict[str, threading.Lock] = {}
        self._tunnels: Dict[str, Tunnel] = {}

    def get(self, name: str) -> Optional[Tunnel]:
        with self._lock:
            return self._tunnels.get(name)

    def ensure(
        self,
        name: str,
        destination: str,
        target: Optional[int] = None,
        port: Optional[int] = None,
        reverse: Sequence[Tuple[int, int]] = (),
    ) -> Tunnel:
        """The open tunnel *name* to *destination*, opened now if it isn't.

        *target* is the port there that this PC reaches, and *reverse* the
        (port there, port here) pairs lent the other way. A tunnel opened
        again keeps its port, and a first one takes *port* (as kept from an
        earlier run) when nothing else listens on it, so an address given
        out for it keeps reaching it. One asked for with other forwards than
        it has is closed and opened with those, on the same port."""
        reverse = tuple((int(there), int(here)) for there, here in reverse)
        with self._lock:
            lock = self._locks.setdefault(name, threading.Lock())
        with lock:
            tunnel = self.get(name)
            if tunnel is not None and tunnel.destination != destination:
                self._end(tunnel)
                tunnel = None
            if tunnel is not None:
                port = tunnel.port
            if tunnel is not None and (tunnel.target, tunnel.reverse) != (
                target,
                reverse,
            ):
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
            tunnel = self._open(name, destination, target, port, reverse)
            with self._lock:
                self._tunnels[name] = tunnel
            if not tunnel.open:
                raise TunnelDown(tunnel.why or "The tunnel is down.")
            return tunnel

    def _open(
        self,
        name: str,
        destination: str,
        target: Optional[int],
        port: Optional[int],
        reverse: Tuple[Tuple[int, int], ...],
    ) -> Tunnel:
        if port is None or self._accepts(port):
            port = self._port()
        tunnel = Tunnel(destination, port, target, reverse)
        if not valid_destination(destination):
            return self._failed(tunnel, "Its ssh destination is not user@host.")
        if target is None and not reverse:
            return self._failed(tunnel, "It was given nothing to forward.")
        ssh = self._which("ssh")
        if ssh is None:
            return self._failed(tunnel, "ssh isn't installed on this PC.")
        try:
            tunnel.process = self._spawn(command(tunnel, ssh))
        except OSError as error:
            return self._failed(tunnel, f"ssh didn't start: {error}")
        started = self._clock()
        deadline = started + self._open_wait_s
        while tunnel.open and not self._is_up(tunnel, started):
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
            "Opened the tunnel %s (%s): %s.",
            name,
            destination,
            " ".join(forwards(tunnel)),
        )
        return tunnel

    def _is_up(self, tunnel: Tunnel, started: float) -> bool:
        """Whether a running ssh has its forwards. One to this PC shows by
        its port answering. One with reverse forwards only can't be asked
        from here, so it is up once it has run for the settle time."""
        if tunnel.target is not None:
            return self._accepts(tunnel.port)
        return self._clock() - started >= self._settle_s

    def _failed(self, tunnel: Tunnel, why: str) -> Tunnel:
        tunnel.failed, tunnel.why = self._clock(), why
        return tunnel

    def _end(self, tunnel: Tunnel) -> None:
        if tunnel.process is not None and tunnel.process.poll() is None:
            end(tunnel.process, END_WAIT_S)

    def close(self, name: str) -> None:
        """Close the tunnel *name*, if there is one."""
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
    listener = getattr(process, "listener", None)
    if listener is not None:
        # ssh has ended, so its last line arrives at once.
        listener.join(timeout=1.0)
        said = "".join(getattr(process, "last_words", ()))
    elif stderr is not None:
        try:
            said = stderr.read() or ""
        except (OSError, ValueError):
            said = ""
    lines = [line.strip() for line in said.splitlines() if line.strip()]
    last = lines[-1] if lines else "ssh ended without saying why."
    return f"ssh to {destination} failed: {last[:300]}"
