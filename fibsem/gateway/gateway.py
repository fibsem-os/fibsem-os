"""The gateway's app: **off** (nothing holds the microscope) or **running**,
the app's host with its window at this PC (started here or claimed by a
desktop launch) or without one (`fibsem.gateway.child`).

The host starts with no microscope unless the gateway was given connection
flags; the operator connects one in the window or on a page. Every start and
stop is recorded with who made it and from where. A stop during a workflow is
refused unless the operator forces it, and a host the gateway didn't start (a
hand-run ``fibsem-autolamella-server``) blocks every start until it goes: two
things never hold the microscope at once. Only the operator closes a window,
and a window that asks a question as it closes is left to answer at the PC.
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
from typing import Callable, Dict, List, Optional, Tuple

import fibsem
from fibsem.server.discovery import DISCOVERY_DIR, _pid_alive, read_discovery_file
from fibsem.server.operator_token import read_operator_token
from fibsem.tools.apps import App, apps
from fibsem.update_check import Newer, check_newer

__all__ = [
    "GATEWAY_FILE",
    "GATEWAY_PORT",
    "MODES",
    "OFF",
    "RUNNING",
    "Gateway",
    "GatewayRefused",
    "GatewayState",
    "read_gateway_file",
]

OFF = "off"
RUNNING = "running"
MODES = (OFF, RUNNING)
# What a switch in progress does: a start (a restart included) or a stop.
STARTING = "starting"
STOPPING = "stopping"

GATEWAY_PORT = 8010
GATEWAY_FILE = DISCOVERY_DIR / "gateway.json"
# Set on a window the gateway started, so its launch doesn't ask the gateway.
CHILD_ENV = "FIBSEM_GATEWAY"
STOP_WAIT_S = 60.0
POLL_S = 1.0
NEWER_EVERY_S = 24 * 3600.0
HISTORY_SHOWN = 50
EXPERIMENT_FILE = "experiment.yaml"


class GatewayRefused(Exception):
    """A start or stop the gateway won't make, with the HTTP status the route
    answers."""

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
    window: bool
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
class RecentExperiment:
    name: str
    folder: str
    app: Optional[str] = None


@dataclass
class GatewayState:
    mode: str
    version: str
    app: Optional[str] = None
    window: bool = False
    """Whether the running app shows its window at this PC."""
    experiment: Optional[str] = None
    pid: Optional[int] = None
    since: Optional[float] = None
    who: Optional[str] = None
    via: Optional[str] = None
    switching: bool = False
    """Whether the gateway is starting or stopping the app now."""
    switch: Optional[str] = None
    """Which: "starting" (a restart too) or "stopping"; None between switches."""
    holder: Optional[Holder] = None
    last_exit: Optional[LastExit] = None
    apps: List[GatewayApp] = field(default_factory=list)
    last_experiment: Optional[str] = None
    recent_experiments: List[RecentExperiment] = field(default_factory=list)
    """The recent experiments on this PC, to load one."""
    experiments_folder: Optional[str] = None
    """Where a new experiment goes."""
    connection: List[str] = field(default_factory=list)
    """The flags the app connects with as it starts; none: the operator connects."""
    configurations: List[GatewayConfiguration] = field(default_factory=list)
    """The microscope configurations saved on this PC, for a start to connect."""
    newer: Optional[Newer] = None
    """A newer release, or commits on main, than the code this gateway runs."""
    app_url: Optional[str] = None
    """The running App Server's address, for the link to the app's pages."""


@dataclass
class GatewayConfiguration:
    """A microscope configuration saved on this PC."""

    name: str
    default: bool = False


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
    window: Optional[bool] = None


@dataclass
class History:
    switches: List[Switch] = field(default_factory=list)


def close_file(pid: int, home: Path = DISCOVERY_DIR) -> Path:
    """The file that asks the window with *pid* to close
    (`fibsem.tools.desktop_launch.watch_for_close`)."""
    return home / f"gateway-close-{pid}"


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


def saved_configurations() -> List[dict]:
    """``fibsem.config.saved_configurations``: ``{name, path, default}`` each."""
    from fibsem.config import saved_configurations

    return saved_configurations()


def _check_experiment(experiment: Optional[str], create: bool) -> None:
    """Refuse an experiment folder the app could not open or make."""
    if create and not experiment:
        raise GatewayRefused(422, "no_folder", "Name the new experiment's folder.")
    if not experiment:
        return
    folder = Path(experiment)
    if create:
        if folder.exists():
            raise GatewayRefused(409, "exists", f"{folder} already exists.")
        if not folder.parent.is_dir():
            raise GatewayRefused(
                422, "no_parent", f"There is no folder {folder.parent}."
            )
    elif not (folder / EXPERIMENT_FILE).is_file():
        raise GatewayRefused(422, "no_experiment", f"No experiment in {folder}.")


def last_experiment() -> Optional[str]:
    """The folder of the last experiment opened on this PC, as ``--last`` opens it."""
    from fibsem.config import get_last_experiment_file

    path = get_last_experiment_file()
    return str(Path(path).parent) if path else None


# Each recent experiment's summary, kept until its file changes: parsing an
# experiment.yaml takes up to seconds and the page reads the state often.
_recent_kept: Dict[str, Tuple[Tuple[int, int], "RecentExperiment"]] = {}


def recent_experiments() -> List["RecentExperiment"]:
    """The recent experiments on this PC still on disk, newest first."""
    from fibsem.config import peek_experiment, recent_experiment_paths

    found = []
    for path in recent_experiment_paths():
        try:
            info = os.stat(path)
        except OSError:
            continue
        stamp = (info.st_mtime_ns, info.st_size)
        kept = _recent_kept.get(path)
        if kept is None or kept[0] != stamp:
            s = peek_experiment(path)
            kept = (stamp, RecentExperiment(s.name, str(Path(path).parent), s.app))
            _recent_kept[path] = kept
        found.append(kept[1])
    return found


def experiments_folder() -> str:
    """Where a new experiment goes: the preference, or the default folder."""
    from fibsem.workflows.experiments import default_experiment_directory

    return default_experiment_directory()


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
    """The app, its process, and the history of starts and stops.

    *connection* is the flags the app connects with as it starts
    (`fibsem.server.headless.connection_parser`); without them it starts with
    no microscope. *home* holds the history, logs and stop files."""

    def __init__(
        self,
        connection: Optional[List[str]] = None,
        home: Optional[Path] = None,
        app_registry: Callable[[], Dict[str, App]] = apps,
        last: Callable[[], Optional[str]] = last_experiment,
        recent: Callable[[], List[RecentExperiment]] = recent_experiments,
        new_folder: Callable[[], str] = experiments_folder,
        configurations: Callable[[], List[dict]] = saved_configurations,
        discovery: Callable[[], Optional[dict]] = read_discovery_file,
        running: Callable[[str, Optional[str]], bool] = workflow_running,
        python: Optional[str] = None,
        stop_wait_s: float = STOP_WAIT_S,
        newer: Callable[[], Optional[Newer]] = check_newer,
    ):
        self._connection = list(connection or [])
        self._home = Path(home) if home is not None else DISCOVERY_DIR
        self._apps = app_registry
        self._last = last
        self._recent = recent
        self._new_folder = new_folder
        self._configurations = configurations
        self._discovery = discovery
        self._running = running
        self._python = python or _python()
        self._stop_wait_s = stop_wait_s
        self._check_newer = newer
        self._newer: Optional[Newer] = None
        self._lock = threading.RLock()
        # The switch in progress (STARTING or STOPPING), else None: the one
        # place it is kept, so replacing the state mid-switch can't drop it.
        self._switch: Optional[str] = None
        self._proc: Optional[subprocess.Popen] = None
        self._claimed: Optional[int] = None
        self._stop_file: Optional[Path] = None
        self._log: Optional[Path] = None
        self._state = GatewayState(mode=OFF, version=fibsem.__version__)
        self._watcher: Optional[threading.Thread] = None
        self._closed = threading.Event()
        # Run before the app stops, once the stop is sure (the agent's end).
        self.before_stop: List[Callable[[], None]] = []
        # Called on the watcher's thread after each poll.
        self.watchers: List[Callable[[], None]] = []

    @property
    def history_file(self) -> Path:
        return self._home / "gateway-history.jsonl"

    # -- reading -----------------------------------------------------------

    def mode(self) -> GatewayState:
        """The state's mode and switch only, cheap to read every second."""
        with self._lock:
            state = replace(self._state)
            state.switch = self._switch
            state.switching = self._switch is not None
        return state

    def state(self) -> GatewayState:
        with self._lock:
            state = replace(self._state)
            state.switch = self._switch
            state.switching = self._switch is not None
        state.holder = self._holder()
        state.apps = [
            GatewayApp(key=key, name=app.name, headless=app.serve is not None)
            for key, app in self._apps().items()
        ]
        state.last_experiment = self._last()
        state.recent_experiments = self._recent()
        state.experiments_folder = self._new_folder()
        state.connection = list(self._connection)
        state.configurations = [
            GatewayConfiguration(name=c["name"], default=c["default"])
            for c in self._configurations()
        ]
        state.newer = self._newer
        found = self._discovery()
        state.app_url = found["url"] if found else None
        return state

    def check_newer(self) -> None:
        """Ask whether newer code exists (PyPI, or GitHub for a checkout)."""
        try:
            self._newer = self._check_newer()
        except Exception:  # noqa: BLE001 - offline is normal; never fail
            logging.debug("Update check failed", exc_info=True)

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

    # -- starting and stopping ---------------------------------------------

    def start_app(
        self,
        who: str,
        via: str,
        app: Optional[str] = None,
        window: bool = True,
        experiment: Optional[str] = None,
        last: bool = False,
        force: bool = False,
        connect: Optional[bool] = None,
        configuration: Optional[str] = None,
        create: bool = False,
    ) -> GatewayState:
        """Start *app*'s host, with its window at this PC or without, stopping
        the running one first. *experiment* is a folder to open, or with
        *create* a new one to make. *connect* connects *configuration* (the
        default one without) as it starts, or nothing when False; None keeps
        the gateway's own connection flags. Raises `GatewayRefused` when it
        can't."""
        chosen = self._app(app, window)
        _check_experiment(experiment, create)
        connection = self._start_connection(connect, configuration)
        with self._take(who, via, RUNNING, app, experiment, window):
            if self._state.mode != OFF:
                self._stop_current(force)
            holder = self._holder()
            if holder is not None:
                raise GatewayRefused(
                    409,
                    "held",
                    f"Another fibsem host (pid {holder.pid}) holds the microscope: "
                    "close it at the PC first.",
                )
            self._start(app, chosen, window, experiment, last, connection, create)
            self._record(OFF, RUNNING, who, via, "started", app, experiment, window)
            with self._lock:
                self._state.who, self._state.via = who, via
        return self.state()

    def stop(
        self, who: str, via: str, force: bool = False, window: bool = False
    ) -> GatewayState:
        """Back to off. *window* closes a window too, for the operator only."""
        with self._take(who, via, OFF, None, None):
            if self._state.mode != OFF:
                app, experiment = self._state.app, self._state.experiment
                closing = self._state.window
                self._stop_current(force, close_window=window)
                self._record(
                    RUNNING, OFF, who, via, "stopped", app, experiment, closing
                )
        # Outside the switch: the answer says it is over.
        return self.state()

    def claim(self, pid: int, app: str, who: str, via: str) -> GatewayState:
        """A desktop launch takes the app's place with its own process *pid*."""
        self._app(app, True)
        with self._take(who, via, RUNNING, app, None, True):
            if self._state.mode != OFF:
                raise GatewayRefused(
                    409, "not_off", "The gateway is already running the app."
                )
            with self._lock:
                self._claimed = pid
                self._state = GatewayState(
                    mode=RUNNING,
                    version=self._state.version,
                    app=app,
                    window=True,
                    pid=pid,
                    since=time.time(),
                    who=who,
                    via=via,
                    last_exit=self._state.last_exit,
                )
            self._record(OFF, RUNNING, who, via, "claimed", app, None, True)
        return self.state()

    def _app(self, key: Optional[str], window: bool) -> App:
        found = self._apps().get(key or "")
        if found is None:
            raise GatewayRefused(422, "bad_app", f"No app {key!r}.")
        if not window and found.serve is None:
            raise GatewayRefused(
                422, "no_headless", f"{found.name} runs only with its window."
            )
        return found

    def _start_connection(
        self, connect: Optional[bool], configuration: Optional[str]
    ) -> List[str]:
        """The connection flags for one start."""
        if connect is None:
            return list(self._connection)
        if not connect:
            return []
        saved = self._configurations()
        found = next(
            (
                c
                for c in saved
                if (c["name"] == configuration if configuration else c["default"])
            ),
            None,
        )
        if found is None:
            raise GatewayRefused(
                422,
                "unknown_configuration",
                f"No configuration {configuration or 'is the default'} on this PC.",
            )
        return ["--config", str(found["path"])]

    def _take(self, who, via, mode, app, experiment, window=None):
        gateway = self

        class _Busy:
            def __enter__(self):
                with gateway._lock:
                    if gateway._switch is not None:
                        raise GatewayRefused(
                            409,
                            "busy",
                            "The gateway is already starting or stopping the app.",
                        )
                    gateway._switch = STARTING if mode == RUNNING else STOPPING

            def __exit__(self, kind, error, tb):
                with gateway._lock:
                    gateway._switch = None
                if isinstance(error, GatewayRefused):
                    gateway._record(
                        gateway._state.mode,
                        mode,
                        who,
                        via,
                        f"refused: {error}",
                        app,
                        experiment,
                        window,
                    )
                return False

        return _Busy()

    def _start(
        self,
        key: str,
        app: App,
        window: bool,
        experiment: Optional[str],
        last: bool,
        connection: List[str],
        create: bool = False,
    ) -> None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        logs = self._home / "logs" / "gateway"
        logs.mkdir(parents=True, exist_ok=True)
        log = logs / f"{'window' if window else 'headless'}-{key}-{stamp}.log"
        env = dict(os.environ)
        stop_file = None
        # An app with neither starts with no experiment open, if it can.
        args = [experiment] if experiment else ["--last"] if last else []
        if experiment and create:
            args.append("--create")
        if window:
            # An app with a headless command has the host's flags, and hosts
            # whatever the preference; others none.
            host = [*args, *connection, "--host"] if app.serve else []
            command = [self._python, "-m", "fibsem.tools.desktop_launch", key, *host]
            env[CHILD_ENV] = "1"
        else:
            stop_file = self._home / f"gateway-stop-{stamp}"
            stop_file.unlink(missing_ok=True)
            command = [
                self._python,
                "-m",
                "fibsem.gateway.child",
                str(stop_file),
                app.serve,
                *args,
                *(connection or ["--no-connect"]),
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
        logging.info("Gateway started %s (window: %s, pid %d)", key, window, proc.pid)
        with self._lock:
            self._proc, self._stop_file, self._log = proc, stop_file, log
            self._claimed = None
            self._state = GatewayState(
                mode=RUNNING,
                version=self._state.version,
                app=key,
                window=window,
                experiment=experiment or ("last" if last else None),
                pid=proc.pid,
                since=time.time(),
                last_exit=self._state.last_exit,
            )

    def _stop_current(self, force: bool, close_window: bool = False) -> None:
        """Stop the running app's process, or raise `GatewayRefused`."""
        if self._state.window and not close_window:
            raise GatewayRefused(
                409,
                "window_open",
                "The app's window is open: only the operator closes it.",
            )
        if not force and self._run_going():
            raise GatewayRefused(
                409,
                "run_in_progress",
                "A workflow is running. Stop it first, or force the stop at the PC.",
            )
        for hook in self.before_stop:
            hook()
        if self._state.window:
            self._close_window()
            return
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

    def _close_window(self) -> None:
        """Ask the window to close as its own close button does, and wait for
        its process to end. Never killed: it may be asking a question."""
        with self._lock:
            proc, pid = self._proc, self._state.pid
        ask = close_file(pid, self._home)
        ask.touch()
        end = time.monotonic() + self._stop_wait_s

        def gone() -> bool:
            return proc.poll() is not None if proc is not None else not _pid_alive(pid)

        try:
            while not gone():
                if time.monotonic() > end:
                    raise GatewayRefused(
                        409,
                        "window_didnt_close",
                        "The window didn't close: it may be asking a question "
                        "at the PC.",
                    )
                time.sleep(0.2)
        finally:
            ask.unlink(missing_ok=True)
        self._ended(proc.returncode if proc is not None else None)

    def _ended(self, code: Optional[int], killed: bool = False) -> None:
        with self._lock:
            state = self._state
            if self._stop_file is not None:
                self._stop_file.unlink(missing_ok=True)
            self._state = GatewayState(
                mode=OFF,
                version=state.version,
                last_exit=LastExit(
                    window=state.window,
                    app=state.app,
                    code=code,
                    at=time.time(),
                    killed=killed,
                    log=str(self._log) if self._log else None,
                ),
            )
            self._proc = self._stop_file = self._log = None
            self._claimed = None

    def _record(
        self, from_mode, to_mode, who, via, result, app, experiment, window=None
    ) -> None:
        line = Switch(
            at=time.time(),
            to_mode=to_mode,
            from_mode=from_mode,
            who=who,
            via=via,
            result=result,
            app=app,
            experiment=experiment,
            window=window,
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
            if self._switch is not None:
                return
            mode, proc, claimed = self._state.mode, self._proc, self._claimed
            window = self._state.window
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
        result = "closed" if window else f"exited ({code})"
        self._record(
            RUNNING, OFF, "gateway", "gateway", result, app, experiment, window
        )

    def start(self) -> None:
        """Watch the running mode's process on a daemon thread."""

        def loop():
            while not self._closed.wait(POLL_S):
                for watch in [self.poll, *self.watchers]:
                    try:
                        watch()
                    except Exception:  # noqa: BLE001 - the watcher never dies
                        logging.exception("Gateway watcher failed")

        def newer():
            while True:
                self.check_newer()
                if self._closed.wait(NEWER_EVERY_S):
                    return

        self._watcher = threading.Thread(target=loop, name="gateway", daemon=True)
        self._watcher.start()
        threading.Thread(target=newer, name="gateway-newer", daemon=True).start()

    def close(self) -> None:
        """Stop watching. An app without its window keeps running; a new
        gateway shows it as a holder until it ends."""
        self._closed.set()
