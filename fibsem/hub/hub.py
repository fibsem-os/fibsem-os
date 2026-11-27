"""The hub's list of instruments, and what it reads from each.

Each instrument is another PC's gateway, named in ``~/.fibsem/hub.yaml`` with
where ssh connects to it: the hub opens a tunnel to each and picks the port
(`fibsem.util.tunnels`). An instrument can instead give an address to use
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

import yaml

from fibsem.gateway.gateway import GatewayRefused
from fibsem.gateway.server import GATEWAY_API_VERSION
from fibsem.server.discovery import DISCOVERY_DIR
from fibsem.util.files import write_atomic
from fibsem.util.tunnels import Tunnel, TunnelDown, Tunnels, valid_destination

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
# The port an instrument PC's gateway serves on, on that PC only: the one
# port a tunnel to an instrument forwards.
GATEWAY_PORT = 8010
CHECK_EVERY_S = 5.0
# One instrument's read: a PC that is off answers nothing, and the others
# are read beside it, not after it.
TIMEOUT_S = 3.0
# A page's read of one instrument, and a start or an end asked of it: a
# gateway answers a start once its agent is up.
READ_S = 10.0
ACT_S = 90.0
MAX_INSTRUMENTS = 32
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]{0,39}$")
# What the hub keeps of an instrument's ``GET /gateway``.
_GATEWAY_KEYS = ("mode", "version", "app", "experiment", "switching", "since")
# And of the agent running there (``GET /gateway/agent``).
_AGENT_KEYS = ("session_id", "state", "playbook", "goal", "level", "started")


# What the hub keeps of each piece's health (``GET /gateway``'s ``health``).
_HEALTH_PIECES = ("app", "microscope")
_HEALTH_KEYS = ("status", "detail", "since")


def _health(read: Any) -> Optional[Dict[str, Any]]:
    """The gateway's health of its App Server and microscope, cut to what a
    list shows. None when the gateway gave none (an older gateway)."""
    if not isinstance(read, dict):
        return None
    found = {
        piece: {key: read[piece][key] for key in _HEALTH_KEYS if read[piece].get(key)}
        for piece in _HEALTH_PIECES
        if isinstance(read.get(piece), dict) and read[piece].get("status")
    }
    return found or None


@dataclass
class Instrument:
    """One instrument in the hub's list: its name there, and how this PC
    reaches its gateway. With *ssh* (``user@host``) the hub opens a tunnel to
    it and picks the port; without, *url* is the address to use as it is."""

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
    """Its tunnel, when the hub opens one: open or not, the local port."""
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
    api_version: Optional[str] = None
    """The version of its gateway's own routes, as its ``/health`` says."""
    same_version: Optional[bool] = None
    """Whether this hub was written for that version (the same major and
    minor number); None when its gateway gave none. An agent runs on the hub
    PC only for an instrument whose gateway it was written for."""
    health: Optional[Dict[str, Any]] = None
    """How its App Server (``app``) and its microscope (``microscope``) are,
    as its gateway says: each a status, why in a line, and since when."""


@dataclass
class HubState:
    instruments: List[InstrumentState] = field(default_factory=list)


def _fetch(
    url: str, route: str, token: Optional[str], timeout: float
) -> Tuple[int, Any]:
    """``GET route`` at *url*: its status and JSON answer. A gateway that
    doesn't answer raises ``OSError``."""
    import httpx

    try:
        response = httpx.get(
            f"{url}{route}",
            headers={"Authorization": f"Bearer {token}"} if token else {},
            timeout=timeout,
        )
    except httpx.HTTPError as error:
        raise OSError(str(error) or type(error).__name__) from None
    try:
        return response.status_code, response.json()
    except ValueError:
        return response.status_code, None


def _send(
    url: str,
    method: str,
    route: str,
    token: str,
    body: Optional[Dict[str, Any]],
    timeout: float,
) -> Tuple[int, Any]:
    """*method* *route* at *url* as this hub: its status and JSON answer. A
    gateway that doesn't answer raises ``OSError``."""
    import httpx

    try:
        response = httpx.request(
            method,
            f"{url}{route}",
            headers={"Authorization": f"Bearer {token}"},
            json=body if method != "GET" else None,
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


def _same_version(theirs: Optional[str]) -> Optional[bool]:
    """Whether a gateway whose routes are version *theirs* is one this hub was
    written for: a change an older hub can't work with raises the minor
    number (`GATEWAY_API_VERSION`)."""
    if not theirs:
        return None
    return theirs.split(".")[:2] == GATEWAY_API_VERSION.split(".")[:2]


def _address(instrument: "Instrument") -> str:
    """Where *instrument*'s gateway is: what its token is issued for."""
    return f"ssh:{instrument.ssh}" if instrument.ssh else f"url:{instrument.url}"


def crosses_in_clear(instrument: "Instrument") -> bool:
    """Whether what this hub and *instrument*'s gateway say to each other
    crosses a network unprotected: plain http to another PC. A tunnel, https
    and an address on this PC do not."""
    if instrument.ssh:
        return False
    split = urlsplit(instrument.url or "")
    host = (split.hostname or "").lower()
    local = host == "localhost" or host == "::1" or host.startswith("127.")
    return split.scheme != "https" and not local


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "instrument"


class Hub:
    """The instrument list, this hub's token for each, and their health."""

    def __init__(
        self,
        file: Optional[Path] = None,
        tokens: Optional[Path] = None,
        fetch: Callable[[str, str, Optional[str], float], Tuple[int, Any]] = _fetch,
        pair: Callable[[str, str, str], Tuple[str, str]] = _pair,
        send: Callable[..., Tuple[int, Any]] = _send,
        clock: Callable[[], float] = time.time,
        every_s: float = CHECK_EVERY_S,
        tunnels: Optional[Tunnels] = None,
    ) -> None:
        self._file = Path(file or HUB_FILE)
        self._tokens = Path(tokens or TOKENS_DIR)
        self._fetch = fetch
        self._pair = pair
        self._send = send
        self._clock = clock
        self._every_s = every_s
        self._tunnels = tunnels or Tunnels()
        self._lock = threading.Lock()
        self._read: Dict[str, InstrumentState] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

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
        """The instrument named *name*; a `GatewayRefused` when none is."""
        return self._instrument(name)

    def _instrument(self, name: str) -> Instrument:
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
        for other in kept:
            # Each instrument's token is kept under its name in this form,
            # so two names that come to the same one would share a token.
            if _slug(other.name) == _slug(name):
                raise GatewayRefused(
                    409,
                    "name_taken",
                    f"That name is too close to {other.name}: pick another.",
                )
        if len(kept) >= MAX_INSTRUMENTS:
            raise GatewayRefused(
                409, "too_many", f"A hub lists up to {MAX_INSTRUMENTS} instruments."
            )
        if len(kept) == len(instruments):
            kept.append(Instrument(name, url, ssh))
        else:
            before = next(i for i in instruments if i.name == name)
            if _address(before) != _address(Instrument(name, url, ssh)):
                # Its token was issued at the old address, and stays there.
                try:
                    self._token_file(name).unlink()
                except OSError:
                    pass
                log.info("%s is reached somewhere new: pair with it again.", name)
            kept = [
                Instrument(name, url, ssh) if i.name == name else i for i in instruments
            ]
        self._write(kept)
        with self._lock:
            self._read.pop(name, None)
        return self.state()

    def remove(self, name: str) -> HubState:
        """Take *name* off the list, and drop this hub's token for it."""
        self._instrument(name)
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
        """This hub's token for *name*, while *name* is still reached where
        the token was issued. A token is never sent anywhere else: an
        instrument whose address changed is paired again."""
        try:
            kept = json.loads(self._token_file(name).read_text())
            token, issued = kept.get("token"), kept.get("address")
            if kept.get("name") != name:
                # Another instrument's file: a list written by hand can
                # hold two names that come to one file name.
                return None
        except (OSError, ValueError, AttributeError):
            return None
        now = next((_address(i) for i in self.instruments() if i.name == name), None)
        if now is None or issued != now:
            return None
        return token if isinstance(token, str) and token else None

    def pair(self, name: str, code: str) -> HubState:
        """Pair this hub with *name*'s gateway using the *code* it shows, and
        keep the token."""
        instrument = self._instrument(name)
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
            json.dump(
                {
                    "name": name,
                    "role": role,
                    "address": _address(instrument),
                    "token": token,
                },
                f,
                indent=2,
            )
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
            return self._open(instrument).url
        return instrument.url or ""

    def _open(self, instrument: Instrument) -> Tunnel:
        """*instrument*'s tunnel, on the port it had in an earlier run when
        it is free; the port it has is kept for the next."""
        path = self._tokens / PORTS_FILE
        try:
            kept = json.loads(path.read_text())
        except (OSError, ValueError):
            kept = {}
        before = kept.get(instrument.name) if isinstance(kept, dict) else None
        # A list is how it was kept while a tunnel also forwarded the app.
        first = before[0] if isinstance(before, list) and before else before
        port = first if isinstance(first, int) else None
        tunnel = self._tunnels.ensure(
            instrument.name, instrument.ssh or "", GATEWAY_PORT, port
        )
        now = tunnel.port
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

    def _tunnel(self, instrument: Instrument) -> Optional[Dict[str, Any]]:
        tunnel = self._tunnels.get(instrument.name) if instrument.ssh else None
        if tunnel is None:
            return None
        return {
            "open": tunnel.open,
            "gateway_port": tunnel.port,
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
            status, health = self._fetch(url, "/health", None, TIMEOUT_S)
            found.reachable = status == 200
            if isinstance(health, dict):
                found.api_version = health.get("api_version") or None
            found.same_version = _same_version(found.api_version)
            if not found.reachable:
                found.why = f"Its gateway answered {status}."
            elif token is None:
                found.why = "Pair this hub with its gateway to read it."
            else:
                self._read_gateway(found, url, token)
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

    def _read_gateway(self, found: InstrumentState, url: str, token: str) -> None:
        """What a gateway that answered says of itself and of its agent.
        Each read fails by itself: a slow one never makes an instrument
        that answered show as unreachable."""
        try:
            status, gateway = self._fetch(url, "/gateway", token, TIMEOUT_S)
        except OSError as error:
            found.why = f"Its gateway answered, then its state didn't: {error}"
            return
        if status in (401, 403):
            found.why = "Its gateway no longer knows this hub: pair again."
            return
        if status != 200 or not isinstance(gateway, dict):
            found.why = f"Its gateway answered {status}."
            return
        found.gateway = {
            key: gateway.get(key)
            for key in _GATEWAY_KEYS
            if gateway.get(key) is not None
        }
        found.health = _health(gateway.get("health"))
        try:
            found.agent = self._agent(url, token)
        except OSError as error:
            found.why = f"Its agent's state didn't answer: {error}"

    def _agent(self, url: str, token: str) -> Optional[Dict[str, Any]]:
        status, agent = self._fetch(url, "/gateway/agent", token, TIMEOUT_S)
        running = agent.get("running") if isinstance(agent, dict) else None
        if status != 200 or not isinstance(running, dict):
            return None
        return {key: running[key] for key in _AGENT_KEYS if running.get(key)}

    # -- asking an instrument's gateway ---------------------------------------

    def ask(
        self,
        name: str,
        method: str,
        route: str,
        body: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        read_again: bool = True,
    ) -> Any:
        """*name*'s gateway's answer to *route*, asked as this hub with the
        token that gateway gave it. The gateway decides what a hub may do: its
        refusal is passed on in its own words. After a change the instrument
        is read again at once, unless *read_again* is off."""
        instrument = self._instrument(name)
        token = self.token(name)
        if token is None:
            raise GatewayRefused(
                409, "not_paired", f"Pair this hub with {name}'s gateway first."
            )
        if timeout is None:
            timeout = READ_S if method == "GET" else ACT_S
        try:
            status, answer = self._send(
                self.reach(instrument), method, route, token, body, timeout
            )
        except TunnelDown as down:
            raise GatewayRefused(503, "unreachable", str(down)) from None
        except OSError as error:
            raise GatewayRefused(
                503, "unreachable", f"{name}'s gateway didn't answer: {error}"
            ) from None
        if status == 401:
            # Never a 401 of the hub's own: that would sign this page out.
            raise GatewayRefused(
                409,
                "not_paired",
                f"{name}'s gateway no longer knows this hub: pair again.",
            )
        if status >= 400:
            detail = answer.get("detail") if isinstance(answer, dict) else None
            detail = detail if isinstance(detail, dict) else {}
            raise GatewayRefused(
                status,
                str(detail.get("error_type") or detail.get("reason") or "refused"),
                str(detail.get("message") or f"{name}'s gateway answered {status}."),
            )
        if method != "GET" and read_again:
            # Its state changed: read it again now, not at the next round.
            self._safely(instrument)
        return answer

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
        return HubState(instruments=rows)

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
