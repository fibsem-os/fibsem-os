"""The Device Server the gateway runs on a device PC (a METEOR's Linux PC, an
AFM PC), from ``~/.fibsem/device-server.yaml``.

When the file exists the gateway starts the Device Server as it starts
(``python -m fibsem.gateway.child``, stopped by its stop file), with the
Python the file names under ``server.python`` (another environment on this PC,
for a vendor's SDK), else its own. A Device Server that ends is shown stopped
with its last log lines and is not started again by itself: Start on the
gateway's page starts it (Patrick, 25 Nov). The gateway reads its health with
this install's Device Server token: its program, version and api_version, and
whether each device it serves answers.
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from fibsem.gateway import child_secrets
from fibsem.server.discovery import DISCOVERY_DIR
from fibsem.server.operator_token import read_device_token

logger = logging.getLogger(__name__)

__all__ = ["CONFIG_FILE", "DeviceServerProcess", "DeviceServerState", "TARGET"]

TARGET = "fibsem.server.devices:serve_for_gateway"
CONFIG_FILE = DISCOVERY_DIR / "device-server.yaml"
DEFAULT_PORT = 8765
STOP_WAIT_S = 5.0
LOG_LINES = 20
HEALTH_TIMEOUT_S = 2.0

# What the gateway's page shows.
STOPPED, STARTING, RUNNING, NOT_SET_UP = "stopped", "starting", "running", "not_set_up"


@dataclass
class DeviceServerState:
    """The Device Server as the gateway's page shows it."""

    status: str
    config: str
    url: Optional[str] = None
    python: Optional[str] = None
    pid: Optional[int] = None
    since: Optional[float] = None
    # From its /health: program, version, api_version; None until it answers.
    health: Optional[Dict[str, Any]] = None
    # Each device it serves: {name, ok, detail}.
    devices: List[Dict[str, Any]] = field(default_factory=list)
    exit_code: Optional[int] = None
    log: Optional[str] = None
    log_tail: List[str] = field(default_factory=list)
    detail: Optional[str] = None


def _read_config(path: Path) -> Optional[dict]:
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return None
    return data if isinstance(data, dict) else {}


def _tail(path: Optional[Path], lines: int = LOG_LINES) -> List[str]:
    if path is None:
        return []
    try:
        with open(path, errors="replace") as f:
            return [line.rstrip("\n") for line in f.readlines()[-lines:]]
    except OSError:
        return []


def _get(url: str, route: str, token: Optional[str]) -> Any:
    import httpx

    r = httpx.get(
        f"{url}{route}",
        headers={"Authorization": f"Bearer {token}"} if token else {},
        timeout=HEALTH_TIMEOUT_S,
    )
    # A Device Server with a device down answers its health as 503.
    if r.status_code != 503:
        r.raise_for_status()
    return r.json()


class DeviceServerProcess:
    """The device PC's Device Server, run by the gateway: `ensure` starts it
    once when its file exists; `start` and `stop` are the page's buttons;
    `state` is what the page shows."""

    def __init__(
        self,
        python: str,
        config: Path = CONFIG_FILE,
        home: Optional[Path] = None,
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
        get: Callable[[str, str, Optional[str]], Any] = _get,
        token: Callable[[], Optional[str]] = read_device_token,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._python = python
        self._config = Path(config)
        self._home = Path(home) if home is not None else DISCOVERY_DIR
        self._popen = popen
        self._get = get
        self._token = token
        self._clock = clock
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._stop_file: Optional[Path] = None
        self._log: Optional[Path] = None
        self._since: Optional[float] = None
        self._tried = False
        self._stopped_here = False

    # -- the file ------------------------------------------------------------

    def _settings(self) -> Optional[dict]:
        """The file's ``server`` part ({} when it has none), or None when
        there is no file."""
        if not self._config.is_file():
            return None
        data = _read_config(self._config)
        server = (data or {}).get("server") or {}
        return server if isinstance(server, dict) else {}

    def _url(self, server: dict) -> str:
        host = server.get("host") or "127.0.0.1"
        if host in ("0.0.0.0", "::"):
            host = "127.0.0.1"
        return f"http://{host}:{int(server.get('port') or DEFAULT_PORT)}"

    # -- running it ----------------------------------------------------------

    def ensure(self) -> None:
        """Start it once, the first time the gateway looks, when its file
        exists. Never again by itself: a Device Server that ends stays
        stopped until Start."""
        if self._tried:
            return
        self._tried = True
        if self._settings() is not None:
            self.start()

    def start(self) -> DeviceServerState:
        """Start it (the page's Start); a running one is left as it is."""
        with self._lock:
            server = self._settings()
            if server is None:
                return self._state_locked()
            if self._proc is not None and self._proc.poll() is None:
                return self._state_locked()
            stamp = time.strftime("%Y%m%d-%H%M%S")
            logs = self._home / "logs" / "gateway"
            logs.mkdir(parents=True, exist_ok=True)
            self._log = logs / f"device-server-{stamp}.log"
            self._stop_file = self._home / f"device-server-stop-{stamp}"
            self._stop_file.unlink(missing_ok=True)
            python = str(server.get("python") or self._python)
            command = [
                python,
                "-m",
                "fibsem.gateway.child",
                str(self._stop_file),
                TARGET,
                "--config",
                str(self._config),
            ]
            self._stopped_here = False
            # A new secret at each start, as the App Server gets (step 6):
            # handed and kept, not yet checked by the Device Server, whose
            # app calls come straight through the tunnel (C1, 25 Nov).
            secret = child_secrets.make()
            try:
                with open(self._log, "w") as out:
                    self._proc = self._popen(
                        command,
                        stdout=out,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        env=child_secrets.environment(secret),
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
            except OSError as error:
                self._proc = None
                logger.warning(
                    "The gateway couldn't start the Device Server: %s", error
                )
                with open(self._log, "a") as out:
                    out.write(f"Couldn't start {python}: {error}\n")
                return self._state_locked()
            self._since = self._clock()
            child_secrets.keep(self._proc.pid, secret)
            logger.info("Gateway started the Device Server (pid %d)", self._proc.pid)
            return self._state_locked()

    def stop(self, wait_s: float = STOP_WAIT_S) -> DeviceServerState:
        """Stop the one the gateway started, by its stop file (it switches
        its devices off as it ends), else killed after *wait_s*."""
        with self._lock:
            proc, stop_file = self._proc, self._stop_file
            if proc is not None and proc.poll() is None:
                self._stopped_here = True
                if stop_file is not None:
                    stop_file.touch()
                try:
                    proc.wait(wait_s)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    try:
                        proc.wait(2.0)
                    except subprocess.TimeoutExpired:
                        pass
            if stop_file is not None:
                stop_file.unlink(missing_ok=True)
            if proc is not None and proc.poll() is not None:
                child_secrets.forget(proc.pid)
            return self._state_locked()

    close = stop

    # -- what the page shows -------------------------------------------------

    def listed(self) -> List[Dict[str, Any]]:
        """This PC's Device Server as one row of ``GET /gateway``'s
        ``device_servers``: none when this PC has no file."""
        state = self.state()
        if state.status == NOT_SET_UP:
            return []
        return [{"kind": "here", "name": "This PC", **asdict(state)}]

    def state(self) -> DeviceServerState:
        with self._lock:
            return self._state_locked()

    def _state_locked(self) -> DeviceServerState:
        server = self._settings()
        if server is None:
            return DeviceServerState(
                status=NOT_SET_UP,
                config=str(self._config),
                detail=f"No {self._config.name}: this PC serves no devices.",
            )
        url = self._url(server)
        python = str(server.get("python") or self._python)
        proc = self._proc
        running = proc is not None and proc.poll() is None
        state = DeviceServerState(
            status=STOPPED,
            config=str(self._config),
            url=url,
            python=python,
            pid=proc.pid if running else None,
            since=self._since if running else None,
            log=str(self._log) if self._log else None,
        )
        if not running:
            state.exit_code = proc.returncode if proc is not None else None
            state.log_tail = _tail(self._log)
            if proc is None and self._log is None:
                state.detail = "Not started."
            elif self._stopped_here:
                state.detail = "Stopped on this page."
            else:
                state.detail = "It ended: its last log lines are below."
            return state
        token = self._token()
        try:
            state.health = self._get(url, "/health", token)
        except Exception as error:  # noqa: BLE001 - not answering yet is a state
            state.status = STARTING
            state.detail = f"Not answering yet: {type(error).__name__}"
            state.log_tail = _tail(self._log, 5)
            return state
        state.status = RUNNING
        state.devices = _devices(state.health)
        return state


def _devices(health: Any) -> List[Dict[str, Any]]:
    """Each device in the Device Server's /health answer as {name, ok,
    detail}."""
    listed = health.get("devices") if isinstance(health, dict) else None
    if not isinstance(listed, dict):
        return []
    return [
        {
            "name": name,
            "ok": bool(v.get("ok", True)) if isinstance(v, dict) else True,
            "detail": v.get("detail") if isinstance(v, dict) else None,
        }
        for name, v in listed.items()
    ]
