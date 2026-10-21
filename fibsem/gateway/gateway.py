"""The gateway's modes and the one process each runs.

One mode at a time: **off** (nothing holds the microscope), **window** (the
app's Qt window, started here or claimed by a desktop launch) or **headless**
(the app's App Server with no window, `fibsem.gateway.child`). Every switch is
recorded with who made it and from where. A switch away from a running
workflow is refused unless the operator forces it, and an App Server the
gateway didn't start (a hand-run ``fibsem-autolamella-server``) blocks every
switch until it goes: two things never hold the microscope at once.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional

import fibsem
from fibsem.server.discovery import DISCOVERY_DIR, _pid_alive, read_discovery_file
from fibsem.server.operator_token import read_operator_token
from fibsem.tools.apps import App, apps

__all__ = [
    "GATEWAY_FILE",
    "GATEWAY_PORT",
    "HEADLESS",
    "MODES",
    "OFF",
    "WINDOW",
    "Gateway",
    "GatewayRefused",
    "GatewayState",
    "read_gateway_file",
]

OFF = "off"
WINDOW = "window"
HEADLESS = "headless"
MODES = (OFF, WINDOW, HEADLESS)

GATEWAY_PORT = 8010
GATEWAY_FILE = DISCOVERY_DIR / "gateway.json"
# Set on a window the gateway started, so its launch doesn't ask the gateway.
CHILD_ENV = "FIBSEM_GATEWAY"
STOP_WAIT_S = 60.0
POLL_S = 1.0
HISTORY_SHOWN = 50


class GatewayRefused(Exception):
    """A switch the gateway won't make, with the HTTP status the route answers."""

    def __init__(self, status: int, error_type: str, message: str):
        super().__init__(message)
        self.status = status
        self.error_type = error_type


@dataclass
class Holder:
    """An App Server the gateway didn't start."""

    pid: int
    url: str


@dataclass
class LastExit:
    mode: str
    app: Optional[str]
    code: Optional[int]
    at: float
    killed: bool = False
    log: Optional[str] = None


@dataclass
class GatewayApp:
    key: str
    name: str
    headless: bool
    """Whether it can run with no window."""


@dataclass
class GatewayState:
    mode: str
    version: str
    app: Optional[str] = None
    experiment: Optional[str] = None
    pid: Optional[int] = None
    since: Optional[float] = None
    who: Optional[str] = None
    via: Optional[str] = None
    switching: bool = False
    holder: Optional[Holder] = None
    last_exit: Optional[LastExit] = None
    apps: List[GatewayApp] = field(default_factory=list)


@dataclass
class Switch:
    """One line of the gateway's history."""

    at: float
    to_mode: str
    from_mode: str
    who: str
    via: str
    result: str
    app: Optional[str] = None
    experiment: Optional[str] = None


@dataclass
class History:
    switches: List[Switch] = field(default_factory=list)


def read_gateway_file(path: Optional[Path] = None) -> Optional[dict]:
    """The running gateway's ``{"url", "pid"}``, or None when none runs."""
    try:
        with open(path or GATEWAY_FILE) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or "url" not in data:
        return None
    return data if _pid_alive(data.get("pid")) else None


def _python() -> str:
    """The interpreter to run children with: ``python`` beside a ``pythonw``,
    whose children would have no standard streams for their log."""
    exe = Path(sys.executable)
    if exe.stem.lower() == "pythonw":
        console = exe.with_name("python" + exe.suffix)
        if console.exists():
            return str(console)
    return str(exe)


def workflow_running(url: str, token: Optional[str]) -> bool:
    """Whether the App Server at *url* is running a workflow. One that doesn't
    answer isn't."""
    import httpx

    try:
        r = httpx.get(
            f"{url}/app/workflow/status",
            headers={"Authorization": f"Bearer {token}"} if token else {},
            timeout=3.0,
        )
        return bool(r.json()["workflow"]["running"]) if r.status_code == 200 else False
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        return False


class Gateway:
    """The mode, its process, and the history of switches.

    *connection* is the flags the headless app connects with (`fibsem.server.
    headless.connection_parser`); *home* holds the history, logs and stop
    files."""

    def __init__(
        self,
        connection: Optional[List[str]] = None,
        home: Optional[Path] = None,
        app_registry: Callable[[], Dict[str, App]] = apps,
        discovery: Callable[[], Optional[dict]] = read_discovery_file,
        running: Callable[[str, Optional[str]], bool] = workflow_running,
        python: Optional[str] = None,
        stop_wait_s: float = STOP_WAIT_S,
    ):
        self._connection = list(connection or [])
        self._home = Path(home) if home is not None else DISCOVERY_DIR
        self._apps = app_registry
        self._discovery = discovery
        self._running = running
        self._python = python or _python()
        self._stop_wait_s = stop_wait_s
        self._lock = threading.RLock()
        self._busy = False
        self._proc: Optional[subprocess.Popen] = None
        self._claimed: Optional[int] = None
        self._stop_file: Optional[Path] = None
        self._log: Optional[Path] = None
        self._state = GatewayState(mode=OFF, version=fibsem.__version__)
        self._watcher: Optional[threading.Thread] = None
        self._closed = threading.Event()

    @property
    def history_file(self) -> Path:
        return self._home / "gateway-history.jsonl"

    # -- reading -----------------------------------------------------------

    def state(self) -> GatewayState:
        with self._lock:
            state = replace(self._state)
        state.holder = self._holder()
        state.apps = [
            GatewayApp(key=key, name=app.name, headless=app.serve is not None)
            for key, app in self._apps().items()
        ]
        return state

    def history(self, limit: int = HISTORY_SHOWN) -> History:
        try:
            lines = self.history_file.read_text().splitlines()
        except OSError:
            return History()
        switches = []
        for line in reversed(lines[-limit:]):
            try:
                switches.append(Switch(**json.loads(line)))
            except (ValueError, TypeError):
                continue
        return History(switches=switches)

    def _holder(self) -> Optional[Holder]:
        found = self._discovery()
        if found is None:
            return None
        pid = found.get("pid")
        with self._lock:
            ours = {self._state.pid, self._claimed}
        return None if pid in ours else Holder(pid=pid, url=found["url"])

    def _run_going(self) -> bool:
        found = self._discovery()
        return found is not None and self._running(found["url"], read_operator_token())

    # -- switching ---------------------------------------------------------

    def switch(
        self,
        mode: str,
        who: str,
        via: str,
        app: Optional[str] = None,
        experiment: Optional[str] = None,
        last: bool = False,
        force: bool = False,
    ) -> GatewayState:
        """Go to *mode*, stopping the current one first. Raises
        `GatewayRefused` when the switch can't be made."""
        if mode not in MODES:
            raise GatewayRefused(422, "bad_mode", f"Modes are {list(MODES)}.")
        if mode == OFF:
            return self.stop(who, via, force)
        chosen = self._app(app, mode)
        with self._take(who, via, mode, app, experiment):
            if self._state.mode != OFF:
                self._stop_current(force)
            holder = self._holder()
            if holder is not None:
                raise GatewayRefused(
                    409,
                    "held",
                    f"Another App Server (pid {holder.pid}) holds the microscope: "
                    "close it at the PC first.",
                )
            self._start(mode, key=app, app=chosen, experiment=experiment, last=last)
            self._record(OFF, mode, who, via, "started", app, experiment)
            with self._lock:
                self._state.who, self._state.via = who, via
            return self.state()

    def stop(self, who: str, via: str, force: bool = False) -> GatewayState:
        """Back to off. A window is closed only by the operator at the PC."""
        with self._take(who, via, OFF, None, None):
            before = self._state.mode
            if before == OFF:
                return self.state()
            app, experiment = self._state.app, self._state.experiment
            self._stop_current(force)
            self._record(before, OFF, who, via, "stopped", app, experiment)
            return self.state()

    def claim(self, pid: int, app: str, who: str, via: str) -> GatewayState:
        """A desktop launch takes window mode for its own process *pid*."""
        self._app(app, WINDOW)
        with self._take(who, via, WINDOW, app, None):
            if self._state.mode != OFF:
                raise GatewayRefused(
                    409, "not_off", f"The gateway is in {self._state.mode} mode."
                )
            with self._lock:
                self._claimed = pid
                self._state = GatewayState(
                    mode=WINDOW,
                    version=self._state.version,
                    app=app,
                    pid=pid,
                    since=time.time(),
                    who=who,
                    via=via,
                    last_exit=self._state.last_exit,
                )
            self._record(OFF, WINDOW, who, via, "claimed", app, None)
            return self.state()

    def _app(self, key: Optional[str], mode: str) -> App:
        found = self._apps().get(key or "")
        if found is None:
            raise GatewayRefused(422, "bad_app", f"No app {key!r}.")
        if mode == HEADLESS and found.serve is None:
            raise GatewayRefused(
                422, "no_headless", f"{found.name} has no headless mode."
            )
        return found

    def _take(self, who, via, mode, app, experiment):
        gateway = self

        class _Busy:
            def __enter__(self):
                with gateway._lock:
                    if gateway._busy:
                        raise GatewayRefused(
                            409, "busy", "The gateway is already switching."
                        )
                    gateway._busy = True
                    gateway._state.switching = True

            def __exit__(self, kind, error, tb):
                with gateway._lock:
                    gateway._busy = False
                    gateway._state.switching = False
                if isinstance(error, GatewayRefused):
                    gateway._record(
                        gateway._state.mode,
                        mode,
                        who,
                        via,
                        f"refused: {error}",
                        app,
                        experiment,
                    )
                return False

        return _Busy()

    def _start(
        self,
        mode: str,
        key: str,
        app: App,
        experiment: Optional[str],
        last: bool,
    ) -> None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        logs = self._home / "logs" / "gateway"
        logs.mkdir(parents=True, exist_ok=True)
        log = logs / f"{mode}-{key}-{stamp}.log"
        env = dict(os.environ)
        stop_file = None
        if mode == WINDOW:
            command = [self._python, "-m", "fibsem.tools.desktop_launch", key]
            env[CHILD_ENV] = WINDOW
        else:
            stop_file = self._home / f"gateway-stop-{stamp}"
            stop_file.unlink(missing_ok=True)
            # An app with neither starts with no experiment open, if it can.
            args = [experiment] if experiment else ["--last"] if last else []
            command = [
                self._python,
                "-m",
                "fibsem.gateway.child",
                str(stop_file),
                app.serve,
                *args,
                *self._connection,
            ]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        with open(log, "w") as out:
            proc = subprocess.Popen(
                command,
                stdout=out,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=env,
                creationflags=flags,
            )
        logging.info("Gateway started %s %s (pid %d)", mode, key, proc.pid)
        with self._lock:
            self._proc, self._stop_file, self._log = proc, stop_file, log
            self._claimed = None
            self._state = GatewayState(
                mode=mode,
                version=self._state.version,
                app=key,
                experiment=experiment or ("last" if last else None),
                pid=proc.pid,
                since=time.time(),
                last_exit=self._state.last_exit,
            )

    def _stop_current(self, force: bool) -> None:
        """Stop the current mode's process, or raise `GatewayRefused`."""
        mode = self._state.mode
        if mode == WINDOW:
            raise GatewayRefused(
                409,
                "window_open",
                "The app's window is open: it's closed at the PC, never remotely.",
            )
        if not force and self._run_going():
            raise GatewayRefused(
                409,
                "run_in_progress",
                "A workflow is running. Stop it first, or force the switch at the PC.",
            )
        with self._lock:
            proc, stop_file = self._proc, self._stop_file
        if proc is None:
            self._ended(None)
            return
        if stop_file is not None:
            stop_file.touch()
        killed = False
        try:
            proc.wait(self._stop_wait_s)
        except subprocess.TimeoutExpired:
            logging.warning("Gateway: pid %d didn't stop; killing it", proc.pid)
            proc.kill()
            proc.wait()
            killed = True
        self._ended(proc.returncode, killed)

    def _ended(self, code: Optional[int], killed: bool = False) -> None:
        with self._lock:
            state = self._state
            if self._stop_file is not None:
                self._stop_file.unlink(missing_ok=True)
            self._state = GatewayState(
                mode=OFF,
                version=state.version,
                last_exit=LastExit(
                    mode=state.mode,
                    app=state.app,
                    code=code,
                    at=time.time(),
                    killed=killed,
                    log=str(self._log) if self._log else None,
                ),
            )
            self._proc = self._stop_file = self._log = None
            self._claimed = None

    def _record(self, from_mode, to_mode, who, via, result, app, experiment) -> None:
        line = Switch(
            at=time.time(),
            to_mode=to_mode,
            from_mode=from_mode,
            who=who,
            via=via,
            result=result,
            app=app,
            experiment=experiment,
        )
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_file, "a") as f:
                f.write(json.dumps(asdict(line)) + "\n")
        except OSError:
            logging.exception("Gateway history not written")

    # -- watching ----------------------------------------------------------

    def poll(self) -> None:
        """Notice a process that ended on its own: back to off, with its exit."""
        with self._lock:
            if self._busy:
                return
            mode, proc, claimed = self._state.mode, self._proc, self._claimed
        if mode == OFF:
            return
        if proc is not None:
            code = proc.poll()
            if code is None:
                return
        elif claimed is not None and _pid_alive(claimed):
            return
        else:
            code = None
        with self._lock:
            app, experiment = self._state.app, self._state.experiment
        self._ended(code)
        result = "closed" if mode == WINDOW else f"exited ({code})"
        self._record(mode, OFF, "gateway", "gateway", result, app, experiment)

    def start(self) -> None:
        """Watch the running mode's process on a daemon thread."""

        def loop():
            while not self._closed.wait(POLL_S):
                try:
                    self.poll()
                except Exception:  # noqa: BLE001 - the watcher never dies
                    logging.exception("Gateway watcher failed")

        self._watcher = threading.Thread(target=loop, name="gateway", daemon=True)
        self._watcher.start()

    def close(self) -> None:
        """Stop watching. A headless app keeps running; a new gateway shows it
        as a holder until it ends."""
        self._closed.set()
