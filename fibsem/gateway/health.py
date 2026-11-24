"""The health of what the gateway looks after on this PC: the App Server and
the microscope it holds, in one shape (`Health`) that the gateway's page and
a hub both read.

The gateway asks on a thread of its own, never on a page's request: the App
Server's ``/health`` every few seconds, timed, and its ``/microscope/health``
(the connection and the Safety Halt) less often. Neither reaches the
hardware: both are what the App Server already holds, so asking never takes
the microscope's locks or switches its imaging channel. Whether the hardware
itself answers is the microscope server's to say, once it is its own process.
It only reports: nothing is restarted (auto-restart is planned for later).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

from fibsem.server.discovery import read_discovery_file
from fibsem.server.operator_token import read_operator_token

logger = logging.getLogger(__name__)

__all__ = [
    "CONNECTING",
    "FAULT",
    "HALTED",
    "NOT_ANSWERING",
    "NOT_CONNECTED",
    "OK",
    "SLOW",
    "STARTING",
    "STOPPED",
    "UNKNOWN",
    "Health",
    "HealthWatch",
    "ServersHealth",
]

# What a piece is: one word, the same for every piece.
OK = "ok"
SLOW = "slow"
NOT_ANSWERING = "not_answering"
STOPPED = "stopped"
STARTING = "starting"
# The microscope's own.
NOT_CONNECTED = "not_connected"
CONNECTING = "connecting"
HALTED = "halted"
FAULT = "fault"
# The microscope's while the App Server doesn't answer: nobody can tell.
UNKNOWN = "unknown"

APP_EVERY_S = 5.0
MICROSCOPE_EVERY_S = 15.0
# An answer slower than this is "slow".
APP_SLOW_S = 1.0
MICROSCOPE_SLOW_S = 1.0
APP_TIMEOUT_S = 5.0
# An app the gateway started that hasn't answered yet, this soon after its
# start: starting, not stopped or hung. A cold start on Windows takes a while.
STARTING_MAX_S = 180.0
MICROSCOPE_TIMEOUT_S = 5.0
# A call on a device (not a long command, which may run for minutes) running
# longer than these: slow, then not answering. A call waiting on a lock another
# holds counts too, hence the long second limit.
CALL_SLOW_S = 120.0
CALL_HUNG_S = 600.0
# A driver error with no success after it, this recent: a fault.
ERROR_RECENT_S = 30 * 60.0


@dataclass
class Health:
    """One piece's health: *status* (one of the words above), why in a line
    when it isn't ok, when it was last asked, how long its answer took, and
    since when it has been in this status."""

    status: str
    detail: Optional[str] = None
    checked_at: Optional[float] = None
    answer_s: Optional[float] = None
    since: Optional[float] = None


@dataclass
class ServersHealth:
    app: Health
    microscope: Health
    heard: Dict[str, Any] = field(default_factory=dict)
    """When each device and service was last heard from, as the App Server
    answered it (``devices`` and ``services``), for a page that lists them."""


class Unreachable(Exception):
    """The App Server gave no answer: refused, timed out, or cut off."""


def _get(url: str, route: str, timeout: float) -> Tuple[int, Any]:
    """``GET route`` on the App Server at *url* as the operator: its status
    and JSON (None when not JSON)."""
    import httpx

    token = read_operator_token()
    try:
        r = httpx.get(
            f"{url}{route}",
            headers={"Authorization": f"Bearer {token}"} if token else {},
            timeout=timeout,
        )
    except httpx.HTTPError as error:
        raise Unreachable(str(error) or type(error).__name__) from None
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, None


def _gone(error: str, timeout: float) -> str:
    if "timed out" in error.lower() or "timeout" in error.lower():
        return f"No answer in {timeout:g} s."
    return f"No answer: {error}"


def _ago(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    return f"{seconds / 60:.0f} min"


def _calls(answer: Dict[str, Any], now: float) -> Tuple[str, Optional[str]]:
    """What the devices' and services' own calls say: a call running too long,
    else a recent driver error with no success since; OK otherwise."""
    parts: List[Tuple[str, Dict[str, Any]]] = [
        (name, heard)
        for group in ("devices", "services")
        for name, heard in sorted((answer.get(group) or {}).items())
        if isinstance(heard, dict)
    ]
    longest: Optional[Tuple[float, str]] = None
    for name, heard in parts:
        for call in heard.get("running") or []:
            if call.get("long"):
                continue
            age = now - float(call.get("since") or now)
            if longest is None or age > longest[0]:
                longest = (age, f"{name}.{call.get('call')}")
    if longest is not None and longest[0] > CALL_HUNG_S:
        return NOT_ANSWERING, f"{longest[1]} has run for {_ago(longest[0])}."
    for name, heard in parts:
        at = heard.get("last_error_at")
        ok = heard.get("last_ok")
        if at is None or now - float(at) > ERROR_RECENT_S:
            continue
        if ok is not None and float(ok) > float(at):
            continue
        call = heard.get("last_error_call") or "a call"
        return FAULT, (
            f"{name}.{call} failed {_ago(now - float(at))} ago: "
            f"{heard.get('last_error')}"
        )
    if longest is not None and longest[0] > CALL_SLOW_S:
        return SLOW, f"{longest[1]} has run for {_ago(longest[0])}."
    return OK, None


def microscope_health(
    answer: Dict[str, Any], now: Optional[float] = None
) -> Tuple[str, Optional[str]]:
    """The microscope's status and why, from ``/microscope/health``."""
    if answer.get("halted"):
        return HALTED, f"Safety Halt: {answer['halted']}"
    if answer.get("connecting"):
        return CONNECTING, str(answer["connecting"])
    if not answer.get("connected"):
        error = answer.get("error")
        if error:
            return FAULT, f"The last connect failed: {error}"
        return NOT_CONNECTED, None
    return _calls(answer, time.time() if now is None else now)


class HealthWatch:
    """Asks the App Server how it and its microscope are, on a thread.

    *mode* is the gateway's (`Gateway.mode`), to tell an app starting or
    stopping from one that hung; *discovery* finds the App Server running on
    this PC, whoever started it."""

    def __init__(
        self,
        mode: Callable[[], Any],
        discovery: Callable[[], Optional[dict]] = read_discovery_file,
        get: Callable[[str, str, float], Tuple[int, Any]] = _get,
        clock: Callable[[], float] = time.time,
        timer: Callable[[], float] = time.monotonic,
    ) -> None:
        self._mode = mode
        self._discovery = discovery
        self._get = get
        self._clock = clock
        self._timer = timer
        self._lock = threading.Lock()
        self._app = Health(status=STOPPED)
        self._microscope = Health(status=STOPPED)
        self._heard: Dict[str, Any] = {}
        # The gateway's run (pid, since) whose App Server has answered.
        self._answered: Optional[Tuple[Any, Any]] = None
        self._closed = threading.Event()

    # -- reading -----------------------------------------------------------

    def state(self) -> ServersHealth:
        with self._lock:
            return ServersHealth(
                app=replace(self._app),
                microscope=replace(self._microscope),
                heard=dict(self._heard),
            )

    # -- asking ------------------------------------------------------------

    def _set(self, which: str, status: str, detail=None, answer_s=None) -> None:
        now = self._clock()
        with self._lock:
            was: Health = getattr(self, which)
            since = was.since if was.status == status and was.since else now
            setattr(
                self,
                which,
                Health(
                    status=status,
                    detail=detail,
                    checked_at=now,
                    answer_s=None if answer_s is None else round(answer_s, 3),
                    since=since,
                ),
            )
            if status != was.status:
                log = (
                    logger.info if status in (OK, STOPPED, STARTING) else logger.warning
                )
                log(
                    {
                        "msg": "health changed",
                        "piece": which.strip("_"),
                        "from": was.status,
                        "to": status,
                        "detail": detail,
                    }
                )

    def _url(self) -> Optional[str]:
        try:
            found = self._discovery()
        except Exception:  # noqa: BLE001 - no file is no app
            logger.debug("could not read the App Server's file", exc_info=True)
            return None
        return found.get("url") if found else None

    def _switch(self) -> Optional[str]:
        try:
            return getattr(self._mode(), "switch", None)
        except Exception:  # noqa: BLE001 - the watch never fails on a read
            return None

    def _run_key(self) -> Optional[Tuple[Any, Any]]:
        """The app the gateway runs now, as (pid, since); None when off."""
        try:
            mode = self._mode()
        except Exception:  # noqa: BLE001 - the watch never fails on a read
            return None
        # "running" is the gateway's RUNNING (gateway.py imports this module).
        if getattr(mode, "mode", None) != "running" or not getattr(mode, "since", None):
            return None
        return getattr(mode, "pid", None), mode.since

    def _first_start(self) -> bool:
        """Whether the gateway started its app lately and that app hasn't
        answered yet: starting, between the launch and the first answer."""
        run = self._run_key()
        if run is None:
            return False
        with self._lock:
            if self._answered == run:
                return False
        try:
            return self._clock() - float(run[1]) < STARTING_MAX_S
        except (TypeError, ValueError):
            return False

    def check_app(self) -> None:
        """The App Server's ``/health``, timed."""
        url = self._url()
        if url is None:
            if self._switch() == STARTING or self._first_start():
                self._set("_app", STARTING, "The app is starting.")
            else:
                self._set("_app", STOPPED)
            return
        started = self._timer()
        try:
            status, _ = self._get(url, "/health", APP_TIMEOUT_S)
        except Unreachable as error:
            if self._switch() is not None:
                self._set("_app", STARTING, "The app is starting or stopping.")
            elif self._first_start():
                self._set("_app", STARTING, "The app is starting.")
            else:
                self._set("_app", NOT_ANSWERING, _gone(str(error), APP_TIMEOUT_S))
            return
        took = self._timer() - started
        run = self._run_key()
        if run is not None:
            with self._lock:
                self._answered = run
        if status >= 400:
            self._set("_app", NOT_ANSWERING, f"/health answered {status}.", took)
        elif took > APP_SLOW_S:
            self._set("_app", SLOW, f"/health took {took:.1f} s.", took)
        else:
            self._set("_app", OK, None, took)

    def check_microscope(self) -> None:
        """The App Server's ``/microscope/health``: the connection and the
        Safety Halt."""
        app = self.state().app.status
        if app in (STOPPED, STARTING):
            with self._lock:
                self._heard = {}
            self._set("_microscope", STOPPED, "The app isn't running.")
            return
        url = self._url()
        if url is None or app == NOT_ANSWERING:
            self._set("_microscope", UNKNOWN, "The app doesn't answer.")
            return
        started = self._timer()
        try:
            status, answer = self._get(url, "/microscope/health", MICROSCOPE_TIMEOUT_S)
        except Unreachable as error:
            self._set(
                "_microscope", NOT_ANSWERING, _gone(str(error), MICROSCOPE_TIMEOUT_S)
            )
            return
        took = self._timer() - started
        if status >= 400 or not isinstance(answer, dict):
            self._set(
                "_microscope",
                UNKNOWN,
                f"/microscope/health answered {status}.",
                took,
            )
            return
        try:
            found, detail = microscope_health(answer, self._clock())
        except (TypeError, ValueError, AttributeError) as error:
            found, detail = UNKNOWN, f"Couldn't read /microscope/health: {error}"
        with self._lock:
            self._heard = {
                group: answer.get(group) or {} for group in ("devices", "services")
            }
        if found == OK and took > MICROSCOPE_SLOW_S:
            found, detail = SLOW, f"/microscope/health took {took:.1f} s."
        self._set("_microscope", found, detail, took)

    # -- the thread --------------------------------------------------------

    def start(self) -> None:
        """Ask on a daemon thread until `close`."""

        def loop():
            due, app_was = 0.0, None
            while not self._closed.is_set():
                self._run(self.check_app)
                app = self.state().app.status
                # at once when the app comes up or goes, else every so often
                if app != app_was or self._timer() >= due:
                    self._run(self.check_microscope)
                    due, app_was = self._timer() + MICROSCOPE_EVERY_S, app
                self._closed.wait(APP_EVERY_S)

        threading.Thread(target=loop, name="gateway-health", daemon=True).start()

    @staticmethod
    def _run(check: Callable[[], None]) -> None:
        try:
            check()
        except Exception:  # noqa: BLE001 - the watch never dies
            logger.exception("Health check failed")

    def close(self) -> None:
        self._closed.set()
