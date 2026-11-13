"""The hub role: a gateway that is given a list of instruments.

Each instrument is another PC's gateway, named in ``~/.fibsem/hub.yaml`` with
where ssh connects to it: the hub opens a tunnel to each and picks the ports
(`fibsem.gateway.tunnels`). An instrument can instead give an address to use
as it is. The hub pairs with each one as a hub (the
code ``fibsem-gateway pair --hub`` shows there), keeps the token it answers,
and reads each gateway's health in the background: whether it answers, what it
runs and its agent. Nothing is read when the list is empty, and no instrument
waits on the hub: it only reads what a gateway already serves.
"""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

import yaml

from fibsem.gateway.gateway import GatewayRefused
from fibsem.gateway.tunnels import (
    Tunnel,
    TunnelDown,
    Tunnels,
    start_gateway_there,
    valid_destination,
)
from fibsem.server.discovery import DISCOVERY_DIR
from fibsem.util.files import write_atomic

__all__ = [
    "HUB_FILE",
    "Hub",
    "HubState",
    "Instrument",
    "InstrumentState",
]

log = logging.getLogger(__name__)

HUB_FILE = DISCOVERY_DIR / "hub.yaml"
# The token each instrument's gateway gave this hub, one file per instrument.
TOKENS_DIR = DISCOVERY_DIR / "hub"
# The local ports each instrument's tunnel last had, kept beside the tokens.
PORTS_FILE = "_ports.json"
CHECK_EVERY_S = 5.0
# One instrument's read: a PC that is off answers nothing, and the others
# are read beside it, not after it.
TIMEOUT_S = 3.0
# A call that acts there (pairing an agent, setting the level).
CALL_TIMEOUT_S = 15.0
MAX_INSTRUMENTS = 32
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]{0,39}$")
# What the hub keeps of an instrument's ``GET /gateway``.
_GATEWAY_KEYS = (
    "mode",
    "version",
    "app",
    "experiment",
    "switching",
    "since",
    "last_experiment",
)
# And of the agent running there (``GET /gateway/agent``).
_AGENT_KEYS = ("session_id", "state", "playbook", "goal", "level", "started")


@dataclass
class Instrument:
    """One instrument in the hub's list: its name there, and how this PC
    reaches its gateway. With *ssh* (``user@host``) the hub opens a tunnel to
    it and picks the ports; without, *url* is the address to use as it is."""

    name: str
    url: Optional[str] = None
    ssh: Optional[str] = None


@dataclass
class InstrumentState:
    """An instrument as the hub last read it."""

    name: str
    url: Optional[str] = None
    ssh: Optional[str] = None
    tunnel: Optional[Dict[str, Any]] = None
    """Its tunnel, when the hub opens one: open or not, the local ports."""
    paired: bool = False
    """Whether this hub holds a token for its gateway."""
    reachable: Optional[bool] = None
    """Whether its gateway answered the last read; None before the first."""
    since: Optional[float] = None
    """When it became reachable, or stopped being."""
    checked: Optional[float] = None
    why: Optional[str] = None
    """Why nothing more is known: it didn't answer, or it refused this hub."""
    gateway: Optional[Dict[str, Any]] = None
    """Off or running, the app and its experiment, the gateway's version."""
    agent: Optional[Dict[str, Any]] = None
    """The agent running there, if one is."""
    session: Optional[Dict[str, Any]] = None
    """The agent this hub runs for it on this PC, if it does."""
    microscope: Optional[Dict[str, Any]] = None
    """Whether its app has the microscope, and the configuration's name."""
    level_cap: Optional[str] = None
    """The highest level its PC lets a hub set; None for no limit."""


@dataclass
class HubState:
    hub: bool
    """Whether this gateway has instruments: it is a hub when it does."""
    instruments: List[InstrumentState] = field(default_factory=list)


def _fetch(
    url: str,
    route: str,
    token: Optional[str],
    timeout: float,
    method: str = "GET",
    body: Optional[dict] = None,
) -> Tuple[int, Any]:
    """``method route`` at *url*: its status and JSON answer. A gateway that
    doesn't answer raises ``OSError``."""
    import httpx

    try:
        response = httpx.request(
            method,
            f"{url}{route}",
            json=body,
            headers={"Authorization": f"Bearer {token}"} if token else {},
            timeout=timeout,
        )
    except httpx.HTTPError as error:
        raise OSError(str(error) or type(error).__name__) from None
    try:
        return response.status_code, response.json()
    except ValueError:
        return response.status_code, None


def _pair(url: str, code: str, name: str) -> Tuple[str, str]:
    """Pair with the gateway at *url*: the token and the role it answers.
    A refusal raises ``PermissionError`` with the gateway's words."""
    import httpx

    try:
        response = httpx.post(
            f"{url}/pair", json={"code": code, "name": name}, timeout=10.0
        )
    except httpx.HTTPError as error:
        raise OSError(str(error) or type(error).__name__) from None
    if response.status_code != 200:
        try:
            message = response.json()["detail"]["message"]
        except (ValueError, KeyError, TypeError):
            message = response.text
        raise PermissionError(message)
    answer = response.json()
    return answer["token"], answer.get("role", "operator")


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "instrument"


class Hub:
    """The instrument list, this hub's token for each, and their health."""

    def __init__(
        self,
        file: Optional[Path] = None,
        tokens: Optional[Path] = None,
        fetch: Callable[..., Tuple[int, Any]] = _fetch,
        pair: Callable[[str, str, str], Tuple[str, str]] = _pair,
        clock: Callable[[], float] = time.time,
        every_s: float = CHECK_EVERY_S,
        tunnels: Optional[Tunnels] = None,
        start_there: Callable[[str], None] = start_gateway_there,
    ) -> None:
        self._file = Path(file or HUB_FILE)
        self._tokens = Path(tokens or TOKENS_DIR)
        self._fetch = fetch
        self._pair = pair
        self._clock = clock
        self._every_s = every_s
        self._tunnels = tunnels or Tunnels()
        self._start_there = start_there
        self._lock = threading.Lock()
        self._read: Dict[str, InstrumentState] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # The agent this hub runs for an instrument (`HubAgents.session`).
        self.session: Callable[[str], Optional[Dict[str, Any]]] = lambda name: None

    # -- the list ------------------------------------------------------------

    def instruments(self) -> List[Instrument]:
        """The instruments in the hub's file; none when there is no file or it
        doesn't read."""
        try:
            saved = yaml.safe_load(self._file.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            return []
        rows = saved.get("instruments") if isinstance(saved, dict) else None
        found: List[Instrument] = []
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            name, url, ssh = row.get("name"), row.get("url"), row.get("ssh")
            url = url.rstrip("/") if isinstance(url, str) and url else None
            ssh = ssh if isinstance(ssh, str) and ssh else None
            if isinstance(name, str) and (url or ssh):
                found.append(Instrument(name, url, ssh))
        return found[:MAX_INSTRUMENTS]

    def instrument(self, name: str) -> Instrument:
        """The instrument named *name*, or `GatewayRefused`."""
        for instrument in self.instruments():
            if instrument.name == name:
                return instrument
        raise GatewayRefused(404, "no_instrument", f"No instrument named {name}.")

    def _write(self, instruments: List[Instrument]) -> None:
        rows = [
            {
                "name": i.name,
                **({"url": i.url} if i.url else {}),
                **({"ssh": i.ssh} if i.ssh else {}),
            }
            for i in instruments
        ]
        self._file.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(self._file, yaml.safe_dump({"instruments": rows}, sort_keys=False))

    def add(
        self, name: str, url: Optional[str] = None, ssh: Optional[str] = None
    ) -> HubState:
        """Add an instrument, or change how the one named *name* is reached:
        by a tunnel to *ssh*, or at *url* as it is."""
        name = name.strip()
        url = (url or "").strip().rstrip("/") or None
        ssh = (ssh or "").strip() or None
        if not _NAME.match(name):
            raise GatewayRefused(
                422,
                "bad_name",
                "Name it with up to 40 letters, digits, spaces, dots or dashes.",
            )
        if ssh is not None and not valid_destination(ssh):
            raise GatewayRefused(
                422, "bad_ssh", "Give where ssh connects to it, such as user@host."
            )
        split = urlsplit(url or "")
        if url is None and ssh is None:
            raise GatewayRefused(
                422,
                "no_address",
                "Give where ssh connects to it (user@host), or its gateway's address.",
            )
        if url is not None and (
            split.scheme not in ("http", "https") or not split.hostname
        ):
            raise GatewayRefused(
                422, "bad_url", "Give its gateway's address, such as http://host:8010."
            )
        instruments = self.instruments()
        kept = [i for i in instruments if i.name != name]
        if len(kept) >= MAX_INSTRUMENTS:
            raise GatewayRefused(
                409, "too_many", f"A hub lists up to {MAX_INSTRUMENTS} instruments."
            )
        if len(kept) == len(instruments):
            kept.append(Instrument(name, url, ssh))
        else:
            kept = [
                Instrument(name, url, ssh) if i.name == name else i for i in instruments
            ]
        self._write(kept)
        with self._lock:
            self._read.pop(name, None)
        return self.state()

    def remove(self, name: str) -> HubState:
        """Take *name* off the list, and drop this hub's token for it."""
        self.instrument(name)
        self._write([i for i in self.instruments() if i.name != name])
        try:
            self._token_file(name).unlink()
        except OSError:
            pass
        self._tunnels.close(name)
        with self._lock:
            self._read.pop(name, None)
        return self.state()

    # -- this hub's token for each instrument ----------------------------------

    def _token_file(self, name: str) -> Path:
        return self._tokens / f"{_slug(name)}.json"

    def token(self, name: str) -> Optional[str]:
        try:
            token = json.loads(self._token_file(name).read_text()).get("token")
        except (OSError, ValueError, AttributeError):
            return None
        return token if isinstance(token, str) and token else None

    def pair(self, name: str, code: str) -> HubState:
        """Pair this hub with *name*'s gateway using the *code* it shows, and
        keep the token."""
        instrument = self.instrument(name)
        try:
            token, role = self._pair(
                self.reach(instrument), str(code).strip(), socket.gethostname()
            )
        except PermissionError as refused:
            raise GatewayRefused(403, "pairing_refused", str(refused)) from None
        except OSError as error:
            raise GatewayRefused(
                503, "unreachable", f"{name}'s gateway didn't answer: {error}"
            ) from None
        path = self._token_file(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Created user-only, never readable by others even for a moment.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump({"name": name, "role": role, "token": token}, f, indent=2)
        os.chmod(path, 0o600)
        log.info("The hub paired with %s as %s.", name, role)
        self._check_one(instrument)
        return self.state()

    # -- reaching and reading an instrument -----------------------------------

    def reach(self, instrument: Instrument) -> str:
        """The address this PC reaches *instrument*'s gateway at, through
        its tunnel when it has one: the one place that decides how an
        instrument is reached. Raises `TunnelDown` when the tunnel isn't open."""
        if instrument.ssh:
            return self._open(instrument).gateway_url
        return instrument.url or ""

    def app_url(self, instrument: Instrument) -> Optional[str]:
        """The address this PC reaches *instrument*'s App Server at: the
        tunnel's other port. None for an instrument reached without one."""
        if instrument.ssh:
            return self._open(instrument).app_url
        return None

    def _open(self, instrument: Instrument) -> Tunnel:
        """*instrument*'s tunnel, on the ports it had in an earlier run when
        they are free; the ports it has are kept for the next."""
        path = self._tokens / PORTS_FILE
        try:
            kept = json.loads(path.read_text())
        except (OSError, ValueError):
            kept = {}
        before = kept.get(instrument.name) if isinstance(kept, dict) else None
        ports = (
            (before[0], before[1])
            if isinstance(before, list)
            and len(before) == 2
            and all(isinstance(port, int) for port in before)
            else None
        )
        tunnel = self._tunnels.ensure(instrument.name, instrument.ssh or "", ports)
        now = [tunnel.gateway_port, tunnel.app_port]
        if before != now:
            with self._lock:
                try:
                    kept = json.loads(path.read_text())
                except (OSError, ValueError):
                    kept = {}
                kept = kept if isinstance(kept, dict) else {}
                kept[instrument.name] = now
                path.parent.mkdir(parents=True, exist_ok=True)
                write_atomic(path, json.dumps(kept, indent=2))
        return tunnel

    def call(
        self, name: str, method: str, route: str, body: Optional[dict] = None
    ) -> Any:
        """``method route`` on *name*'s gateway as this hub: its answer, or
        its refusal in its own words as a `GatewayRefused`."""
        instrument = self.instrument(name)
        token = self.token(name)
        if token is None:
            raise GatewayRefused(
                409, "not_paired", f"Pair this hub with {name}'s gateway first."
            )
        try:
            status, answer = self._fetch(
                self.reach(instrument), route, token, CALL_TIMEOUT_S, method, body
            )
        except OSError as error:
            raise GatewayRefused(
                503, "unreachable", f"{name}'s gateway didn't answer: {error}"
            ) from None
        if 200 <= status < 300:
            return answer
        try:
            detail = answer["detail"]
            raise GatewayRefused(status, detail["error_type"], detail["message"])
        except (KeyError, TypeError):
            raise GatewayRefused(
                status, "refused", f"{name}'s gateway answered {status}."
            ) from None

    def start_gateway(self, name: str) -> None:
        """Start *name*'s gateway when it is down, by the task its
        ``fibsem-gateway install`` registered: only over ssh, and only in the
        session of someone logged on at that PC."""
        instrument = self.instrument(name)
        if not instrument.ssh:
            raise GatewayRefused(
                409,
                "no_ssh",
                f"{name} has no ssh destination, so this PC can't start its gateway.",
            )
        try:
            self._start_there(instrument.ssh)
        except OSError as error:
            raise GatewayRefused(
                503,
                "not_started",
                f"{name}'s gateway wasn't started: {error} It needs "
                "fibsem-gateway install run on that PC, and someone logged on.",
            ) from None
        log.info("The hub started %s's gateway.", name)

    def answers(self, url: str) -> bool:
        """Whether the server at *url* answers its ``/health``."""
        try:
            return self._fetch(url, "/health", None, TIMEOUT_S)[0] == 200
        except OSError:
            return False

    def _tunnel(self, instrument: Instrument) -> Optional[Dict[str, Any]]:
        tunnel = self._tunnels.get(instrument.name) if instrument.ssh else None
        if tunnel is None:
            return None
        return {
            "open": tunnel.open,
            "gateway_port": tunnel.gateway_port,
            "app_port": tunnel.app_port,
        }

    def _check_one(self, instrument: Instrument) -> None:
        now = self._clock()
        token = self.token(instrument.name)
        found = InstrumentState(
            instrument.name,
            instrument.url,
            instrument.ssh,
            paired=token is not None,
            checked=now,
        )
        try:
            url = self.reach(instrument)
            status, _ = self._fetch(url, "/health", None, TIMEOUT_S)
            found.reachable = status == 200
            if not found.reachable:
                found.why = f"Its gateway answered {status}."
            elif token is None:
                found.why = "Pair this hub with its gateway to read it."
            else:
                status, gateway = self._fetch(url, "/gateway", token, TIMEOUT_S)
                if status == 200 and isinstance(gateway, dict):
                    found.gateway = {
                        key: gateway.get(key)
                        for key in _GATEWAY_KEYS
                        if gateway.get(key) is not None
                    }
                    found.agent, found.level_cap = self._agent(url, token)
                    # The apps a hub can start: the ones that run without a window.
                    apps = gateway.get("apps")
                    found.gateway["apps"] = [
                        {"key": app.get("key"), "name": app.get("name")}
                        for app in (apps if isinstance(apps, list) else [])
                        if isinstance(app, dict) and app.get("headless")
                    ]
                    if found.gateway.get("mode") == "running":
                        found.microscope = self._microscope(url, token)
                elif status in (401, 403):
                    found.why = "Its gateway no longer knows this hub: pair again."
                else:
                    found.why = f"Its gateway answered {status}."
        except TunnelDown as down:
            found.reachable = False
            found.why = str(down)
        except OSError as error:
            found.reachable = False
            found.why = f"Its gateway didn't answer: {error}"
        found.tunnel = self._tunnel(instrument)
        with self._lock:
            before = self._read.get(instrument.name)
            same = before is not None and before.reachable == found.reachable
            found.since = before.since if same and before is not None else now
            self._read[instrument.name] = found

    def _agent(
        self, url: str, token: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """The agent running there, and the level its PC caps a hub at."""
        status, agent = self._fetch(url, "/gateway/agent", token, TIMEOUT_S)
        if status != 200 or not isinstance(agent, dict):
            return None, None
        running, cap = agent.get("running"), agent.get("hub_max_level")
        cap = cap if isinstance(cap, str) and cap else None
        if not isinstance(running, dict):
            return None, cap
        return {key: running[key] for key in _AGENT_KEYS if running.get(key)}, cap

    def _microscope(self, url: str, token: str) -> Optional[Dict[str, Any]]:
        """Whether the app there has the microscope; None while the app
        doesn't answer."""
        status, found = self._fetch(
            url, "/gateway/microscope/connection", token, TIMEOUT_S
        )
        if status != 200 or not isinstance(found, dict):
            return None
        configuration = found.get("configuration")
        about = configuration if isinstance(configuration, dict) else {}
        model = " ".join(
            part
            for part in (about.get("manufacturer"), about.get("model"))
            if isinstance(part, str) and part
        )
        name = about.get("name")
        return {
            "connected": bool(found.get("connected")),
            "connecting": bool(found.get("connecting")),
            **({"configuration": name} if isinstance(name, str) and name else {}),
            **({"model": model} if model else {}),
        }

    def check(self) -> None:
        """Read every instrument once, side by side."""
        instruments = self.instruments()
        names = {i.name for i in instruments}
        with self._lock:
            for gone in [name for name in self._read if name not in names]:
                del self._read[gone]
        for instrument in instruments:
            if not instrument.ssh:
                self._tunnels.close(instrument.name)
        if not instruments:
            return
        with ThreadPoolExecutor(max_workers=min(8, len(instruments))) as pool:
            for instrument in instruments:
                pool.submit(self._safely, instrument)

    def read(self, name: str) -> None:
        """Read *name* now, after something was done there."""
        self._safely(self.instrument(name))

    def _safely(self, instrument: Instrument) -> None:
        try:
            self._check_one(instrument)
        except Exception:  # noqa: BLE001 - one instrument never costs the others
            log.debug("Reading %s failed.", instrument.name, exc_info=True)

    def state(self) -> HubState:
        """The list with what was last read of each: never waits on one."""
        instruments = self.instruments()
        with self._lock:
            read = dict(self._read)
        rows = [
            read.get(i.name)
            or InstrumentState(
                i.name, i.url, i.ssh, paired=self.token(i.name) is not None
            )
            for i in instruments
        ]
        rows = [replace(row, session=self.session(row.name)) for row in rows]
        return HubState(hub=bool(instruments), instruments=rows)

    # -- the background read -----------------------------------------------------

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run, name="hub-health", daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        """Stop reading, and close every tunnel."""
        self._stop.set()
        self._tunnels.close_all()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.check()
            except Exception:  # noqa: BLE001 - the read goes on
                log.debug("The hub's read failed.", exc_info=True)
            self._stop.wait(self._every_s)
