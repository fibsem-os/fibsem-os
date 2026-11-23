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
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Optional, Tuple

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
MICROSCOPE_TIMEOUT_S = 5.0


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


def microscope_health(answer: Dict[str, Any]) -> Tuple[str, Optional[str]]:
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
    return OK, None


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
        self._closed = threading.Event()

    # -- reading -----------------------------------------------------------

    def state(self) -> ServersHealth:
        with self._lock:
            return ServersHealth(
                app=replace(self._app),
                microscope=replace(self._microscope),
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

    def check_app(self) -> None:
        """The App Server's ``/health``, timed."""
        url = self._url()
        if url is None:
            if self._switch() == STARTING:
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
            else:
                self._set("_app", NOT_ANSWERING, _gone(str(error), APP_TIMEOUT_S))
            return
        took = self._timer() - started
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
        found, detail = microscope_health(answer)
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
