"""The Device Servers the microscope reaches, as the SEM PC's gateway watches
them: read from the ``remote:`` entries of the configuration the app connects
with, so there is no second list to keep in step (Patrick, 25 Nov).

For each Device Server the gateway asks its ``/health`` (with the token this
PC was paired with there) and says whether it answers, its version and
api_version, and whether each of its devices answers. An entry that names a
``tunnel: user@host`` is reached through an SSH tunnel the gateway opens and
keeps open (`fibsem.util.tunnels`), as this PC's user with their own SSH key:
the gateway stores no secret and makes no key, and one that is missing shows
ssh's own words (Patrick, 25 Nov). A port that already answers, such as a
tunnel opened by hand, is used as it is.

Nothing here connects the microscope or starts anything on the device PC: it
only watches, in the background, and the answer is in ``GET /gateway``.
"""

from __future__ import annotations

import logging
import os
import re
import socket
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

from fibsem.client.pairing import device_server_token
from fibsem.codec.upgrades.configuration import upgrade
from fibsem.devices.wire import version_problem
from fibsem.gateway import peers
from fibsem.server.discovery import GATEWAY_PORT
from fibsem.structures.system import (
    CONFIGURED_DEVICES,
    DEVICE_TYPES,
    REMOTE_KEY,
    read_device_entries,
)
from fibsem.util.tunnels import TunnelDown, Tunnels, valid_destination

logger = logging.getLogger(__name__)

__all__ = [
    "DeviceServers",
    "EntryRefused",
    "RemoteDeviceServer",
    "add_entry",
    "configuration_file",
    "listed",
    "remove_entry",
]

# How often each Device Server is asked, and how long one may take.
EVERY_S = 10.0
HEALTH_TIMEOUT_S = 3.0
TUNNEL_KEY = "tunnel"
LOCAL = ("127.0.0.1", "localhost", "::1")

# What the page shows for each.
RUNNING = "running"
NOT_ANSWERING = "not_answering"
NOT_PAIRED = "not_paired"
TUNNEL_DOWN = "tunnel_down"
TOO_OLD_OR_NEW = "version_mismatch"
NOT_CHECKED = "not_checked"
# A pairing asked from here, waiting for the operator at the device PC.
ASKING = "asking"


@dataclass
class RemoteDeviceServer:
    """One Device Server the configuration names, and how it answered."""

    address: str
    port: int
    # The configuration's devices it serves.
    serves: List[str] = field(default_factory=list)
    # ``user@host`` when the gateway reaches it through an SSH tunnel.
    tunnel: Optional[str] = None
    status: str = NOT_CHECKED
    detail: Optional[str] = None
    health: Optional[Dict[str, Any]] = None
    devices: List[Dict[str, Any]] = field(default_factory=list)
    checked: Optional[float] = None
    # Paired through the gateways: its gateway's token is kept here, so
    # Start and Stop work (`fibsem.gateway.peers`).
    paired: bool = False
    # Pairing now: "asking" while its gateway waits for the operator there,
    # else how the last pairing ended; None when none was asked.
    pairing: Optional[str] = None

    @property
    def key(self) -> str:
        return f"{self.address}:{self.port}"

    @property
    def url(self) -> str:
        return f"http://{self.address}:{self.port}"

    def to_dict(self) -> dict:
        return {"kind": "remote", "name": self.tunnel or self.key, **asdict(self)}


def configuration_file(
    flags: List[str], chosen: Optional[str], saved: List[dict]
) -> Optional[Path]:
    """The configuration the app connects with: the ``--config`` in its
    *flags*, else the saved one the page chose (*chosen*), else the default
    saved one."""
    if "--config" in flags:
        at = flags.index("--config") + 1
        if at < len(flags):
            return Path(flags[at])
    found = next(
        (c for c in saved if (c["name"] == chosen if chosen else c.get("default"))),
        None,
    )
    if found is None and chosen:
        found = next((c for c in saved if c.get("default")), None)
    return Path(found["path"]) if found else None


def listed(path: Optional[Path]) -> Tuple[List[RemoteDeviceServer], Optional[str]]:
    """The Device Servers *path* names, one per address and port, with the
    devices each serves; and why none could be read, when so."""
    if path is None:
        return [], None
    try:
        with open(path) as f:
            settings = yaml.safe_load(f) or {}
        entries = read_device_entries(upgrade(settings))
    except (OSError, yaml.YAMLError, ValueError, TypeError) as error:
        return [], f"{Path(path).name}: {error}"
    servers: Dict[Tuple[str, int], RemoteDeviceServer] = {}
    for entry in entries.values():
        if entry.enabled is False or not entry.is_remote:
            continue
        try:
            address, port = entry.remote_address()
        except ValueError as error:
            return [], str(error)
        if address is None or port is None:
            continue
        server = servers.setdefault(
            (address, port), RemoteDeviceServer(address=address, port=port)
        )
        server.serves.append(entry.name)
        tunnel = entry.options.get(TUNNEL_KEY)
        if tunnel and not server.tunnel:
            server.tunnel = str(tunnel)
    return list(servers.values()), None


class EntryRefused(ValueError):
    """A Device Server entry that can't be added or removed, with why."""

    def __init__(self, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.status = status


_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_HOST = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]{0,252}$")


def _settings(text: str, name: str) -> Tuple[dict, list]:
    """The file's settings and its ``hardware.devices`` list. A file in an
    older shape is refused: it is opened and saved once in the app first."""
    try:
        settings = yaml.safe_load(text) or {}
    except yaml.YAMLError as error:
        raise EntryRefused(f"{name} can't be read: {error}") from None
    if not isinstance(settings, dict):
        raise EntryRefused(f"{name} is not a configuration.")
    hardware = settings.get("hardware")
    devices = hardware.get("devices") if isinstance(hardware, dict) else None
    if not isinstance(devices, list):
        raise EntryRefused(
            f"{name} has no list of devices (hardware.devices), or lists them in an "
            "older shape: open and save it in the app once, then add it here."
        )
    return settings, devices


def _entry_named(devices: list, device: str) -> Optional[dict]:
    for item in devices:
        if isinstance(item, dict) and (item.get("name") or item.get("type")) == device:
            return item
    return None


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _quiet(line: str) -> bool:
    """A blank or comment line, which never ends a block."""
    return not line.strip() or line.lstrip().startswith("#")


class _Devices:
    """Where ``hardware.devices`` and its entries are in the file's lines, so
    an edit changes one entry's lines and keeps every other line, comments
    and spacing as they are."""

    def __init__(self, lines: List[str], name: str) -> None:
        self.lines = lines
        hardware = next(
            (
                i
                for i, line in enumerate(lines)
                if re.match(r"^hardware:\s*(#.*)?$", line)
            ),
            None,
        )
        found = None
        if hardware is not None:
            for i in range(hardware + 1, len(lines)):
                if _quiet(lines[i]):
                    continue
                if _indent(lines[i]) == 0:
                    break
                if re.match(r"^\s+devices:\s*(#.*)?$", lines[i]):
                    found = i
                    break
        if found is None:
            raise EntryRefused(
                f"{name} writes its devices in a way this page can't edit: add the "
                "remote: line to the device's entry by hand."
            )
        self.key_indent = _indent(lines[found])
        end = len(lines)
        for i in range(found + 1, len(lines)):
            if (
                not _quiet(lines[i])
                and _indent(lines[i]) <= self.key_indent
                and not (
                    lines[i].lstrip().startswith("- ")
                    and _indent(lines[i]) == self.key_indent
                )
            ):
                end = i
                break
        self.start, self.end = found + 1, end
        dashes = [
            i for i in range(self.start, self.end) if lines[i].lstrip().startswith("- ")
        ]
        self.dash = _indent(lines[dashes[0]]) if dashes else self.key_indent + 2
        self.items = [i for i in dashes if _indent(lines[i]) == self.dash]

    def item(self, at: int) -> Tuple[int, int]:
        """The lines of the entry starting at *at*: to its last line that
        isn't blank or a comment."""
        later = [i for i in self.items if i > at]
        stop = later[0] if later else self.end
        while stop - 1 > at and _quiet(self.lines[stop - 1]):
            stop -= 1
        return at, stop

    def last_line(self) -> int:
        stop = self.end
        while stop - 1 >= self.start and _quiet(self.lines[stop - 1]):
            stop -= 1
        return stop


def _edited(text: str, name: str, device: str, values: Dict[str, Optional[str]]) -> str:
    """*text* with *device*'s entry given *values* (None removes a key), the
    entry added when there is none, and dropped when only its name is left.
    Refused when the result isn't exactly that change."""
    settings, devices = _settings(text, name)
    entry = _entry_named(devices, device)
    expected = [dict(item) if isinstance(item, dict) else item for item in devices]
    target = _entry_named(expected, device)
    if target is None:
        target = {"name": device}
        expected.append(target)
    for key, value in values.items():
        if value is None:
            target.pop(key, None)
        else:
            target[key] = value
    if set(target) <= {"name"}:
        expected.remove(target)

    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    found = _Devices(lines, name)
    at = None
    if entry is not None:
        index = next(i for i, item in enumerate(devices) if item is entry)
        if index >= len(found.items):
            raise EntryRefused(f"{name}'s entry for {device} can't be found to edit.")
        at = found.items[index]
    keys = " " * (found.dash + 2)
    if at is None:
        new = [f"{' ' * found.dash}- name: {device}\n"] + [
            f"{keys}{key}: {value}\n" for key, value in values.items() if value
        ]
        stop = found.last_line()
        lines[stop:stop] = new
    else:
        first, stop = found.item(at)
        if "{" in lines[first]:
            raise EntryRefused(
                f"{name} writes {device}'s entry on one line: edit it by hand."
            )
        region = lines[first:stop]
        for key, value in values.items():
            match = [
                i
                for i, line in enumerate(region)
                if i > 0
                and re.match(rf"^{keys}{key}:", line)
                or i == 0
                and re.match(rf"^\s*- {key}:", line)
            ]
            if match and value is None:
                if match[0] == 0:
                    raise EntryRefused(f"{name}: {device}'s entry starts with {key}.")
                del region[match[0]]
            elif match:
                head = "- " if match[0] == 0 else ""
                lead = " " * found.dash if match[0] == 0 else keys
                region[match[0]] = f"{lead}{head}{key}: {value}\n"
            elif value is not None:
                region.insert(1, f"{keys}{key}: {value}\n")
        if set(target) <= {"name"} and target not in expected:
            region = []
        lines[first:stop] = region
    result = "".join(lines)
    after, listed_ = _settings(result, name)
    settings["hardware"] = dict(settings["hardware"], devices=expected)
    if after != settings:
        raise EntryRefused(
            f"{name} couldn't be edited safely: add the remote: line to {device}'s "
            "entry by hand."
        )
    return result


def _write(path: Path, text: str) -> None:
    """Write *text* to *path*, keeping the file as it was beside it
    (``<name>.bak``) the first time."""
    backup = path.with_name(path.name + ".bak")
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text)
    os.replace(partial, path)


def _read(path: Path) -> str:
    try:
        return path.read_text()
    except OSError as error:
        raise EntryRefused(f"{path.name} can't be read: {error}") from None


def add_entry(
    path: Optional[Path],
    device: str,
    address: str,
    port: int,
    tunnel: Optional[str] = None,
) -> None:
    """Point *device* at the Device Server at *address*:*port* in the
    configuration *path*: its entry gets ``remote:`` (and ``tunnel:``), or a
    new entry is added. Only that entry's lines change: every other line,
    comments included, stays as it is."""
    if path is None:
        raise EntryRefused("No configuration is chosen to add it to.", 409)
    device, address = (device or "").strip(), (address or "").strip()
    tunnel = (tunnel or "").strip() or None
    if not _NAME.match(device):
        raise EntryRefused(
            f"A device is named with letters and digits, not {device!r}."
        )
    if tunnel is not None:
        if not valid_destination(tunnel):
            raise EntryRefused(f"A tunnel is user@host, not {tunnel!r}.")
        address = address or "127.0.0.1"
        if address not in LOCAL:
            raise EntryRefused(
                "Through a tunnel the Device Server is reached on this PC: leave the "
                "address as 127.0.0.1."
            )
    if not _HOST.match(address):
        raise EntryRefused(f"An address is a host name or IP address, not {address!r}.")
    try:
        port = int(port)
    except (TypeError, ValueError):
        raise EntryRefused(f"A port is a number, not {port!r}.") from None
    if not 0 < port < 65536:
        raise EntryRefused(f"A port is 1 to 65535, not {port}.")
    text = _read(path)
    _, devices = _settings(text, path.name)
    if _entry_named(devices, device) is None and not (
        device in CONFIGURED_DEVICES or device in DEVICE_TYPES
    ):
        raise EntryRefused(
            f"{path.name} has no entry for {device!r}, and it is no device type: "
            f"name a device it has, or a type ({', '.join(DEVICE_TYPES)})."
        )
    _write(
        path,
        _edited(
            text,
            path.name,
            device,
            {REMOTE_KEY: f"{address}:{port}", TUNNEL_KEY: tunnel},
        ),
    )


def remove_entry(path: Optional[Path], device: str) -> None:
    """Take *device* off its Device Server in the configuration *path*: its
    ``remote:`` and ``tunnel:`` lines go, and an entry left with nothing but
    its name goes too. Every other line stays as it is."""
    if path is None:
        raise EntryRefused("No configuration is chosen.", 409)
    text = _read(path)
    _, devices = _settings(text, path.name)
    entry = _entry_named(devices, device)
    if entry is None or entry.get(REMOTE_KEY) is None:
        raise EntryRefused(f"{path.name} has no Device Server for {device!r}.", 404)
    _write(path, _edited(text, path.name, device, {REMOTE_KEY: None, TUNNEL_KEY: None}))


def _get(url: str, token: Optional[str]) -> Tuple[int, Any]:
    import httpx

    r = httpx.get(
        f"{url}/health",
        headers={"Authorization": f"Bearer {token}"} if token else {},
        timeout=HEALTH_TIMEOUT_S,
    )
    try:
        body = r.json()
    except ValueError:
        body = None
    return r.status_code, body


def _post(
    url: str, route: str, body: Optional[dict], token: Optional[str], timeout: float
) -> Tuple[int, Any]:
    import httpx

    r = httpx.post(
        f"{url}{route}",
        json=body or {},
        headers={"Authorization": f"Bearer {token}"} if token else {},
        timeout=timeout,
    )
    try:
        answer = r.json()
    except ValueError:
        answer = None
    return r.status_code, answer


class PairRefused(RuntimeError):
    """A pairing or a start that can't be done, with why."""

    def __init__(self, message: str, status: int = 409) -> None:
        super().__init__(message)
        self.status = status


def _accepts(port: int) -> bool:
    import socket

    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


class DeviceServers:
    """The Device Servers of the configuration the app connects with, asked
    every `EVERY_S` in the background. `state` is what ``GET /gateway`` and
    the page show."""

    def __init__(
        self,
        configuration: Callable[[], Optional[Path]],
        tunnels: Optional[Tunnels] = None,
        get: Callable[[str, Optional[str]], Tuple[int, Any]] = _get,
        token: Callable[[str], Optional[str]] = device_server_token,
        accepts: Callable[[int], bool] = _accepts,
        clock: Callable[[], float] = time.time,
        every_s: float = EVERY_S,
        post: Callable[..., Tuple[int, Any]] = _post,
        name: Callable[[], str] = socket.gethostname,
        pair_lapse_s: float = peers.ASK_LAPSE_S,
    ) -> None:
        self._configuration = configuration
        self._post = post
        self._name = name
        self._pair_lapse_s = pair_lapse_s
        # How each pairing asked from here stands, by Device Server.
        self._pairings: Dict[str, str] = {}
        self._tunnels = tunnels if tunnels is not None else Tunnels()
        self._get = get
        self._token = token
        self._accepts = accepts
        self._clock = clock
        self._every_s = every_s
        self._lock = threading.Lock()
        self._servers: List[RemoteDeviceServer] = []
        self._problem: Optional[str] = None
        self._file: Optional[Path] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- watching --------------------------------------------------------------

    def ensure(self) -> None:
        """Start watching, once (a gateway watcher)."""
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run, name="gateway-device-servers", daemon=True
            )
            self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.check()
            except Exception:  # noqa: BLE001 - watching never ends the gateway
                logger.exception("Checking the Device Servers failed")
            self._stop.wait(self._every_s)

    def close(self) -> None:
        self._stop.set()
        self._tunnels.close_all()

    def check(self) -> List[RemoteDeviceServer]:
        """Read the configuration again and ask each Device Server now."""
        path = self._configuration()
        servers, problem = listed(path)
        names = {s.key for s in servers}
        for old in self._servers:
            if old.tunnel and old.key not in names:
                self._tunnels.close(old.key)
        for server in servers:
            self._ask(server)
        with self._lock:
            self._servers, self._problem, self._file = servers, problem, path
        return servers

    def _ask(self, server: RemoteDeviceServer) -> None:
        server.checked = self._clock()
        server.paired = peers.kept(server.url) is not None
        server.pairing = self._pairings.get(server.key)
        if server.tunnel:
            why = self._through_tunnel(server)
            if why is not None:
                server.status, server.detail = TUNNEL_DOWN, why
                return
        token = self._token(server.url)
        try:
            status, body = self._get(server.url, token)
        except Exception as error:  # noqa: BLE001 - not answering is a state
            server.status = NOT_ANSWERING
            server.detail = (
                f"It doesn't answer at {server.key}: {type(error).__name__}."
            )
            return
        if status in (401, 403):
            server.status = NOT_PAIRED
            server.detail = (
                f"This PC is not paired with the Device Server at {server.key}."
            )
            return
        if status not in (200, 503) or not isinstance(body, dict):
            server.status = NOT_ANSWERING
            server.detail = f"It answered {status} at {server.key}."
            return
        server.health = {
            k: body.get(k) for k in ("ok", "program", "version", "api_version")
        }
        server.devices = _devices(body)
        refused, why = version_problem(body, server.key)
        server.status = TOO_OLD_OR_NEW if refused else RUNNING
        server.detail = why or None

    def _through_tunnel(self, server: RemoteDeviceServer) -> Optional[str]:
        """Open (or keep) the tunnel that brings the Device Server to its
        port here: None once it is up, else why not."""
        destination = server.tunnel or ""
        if server.address not in LOCAL:
            return (
                f"A tunnel reaches its Device Server on this PC's own port: write "
                f"remote: 127.0.0.1:{server.port}, not {server.key}."
            )
        if not valid_destination(destination):
            return f"tunnel is user@host, not {destination!r}."
        current = self._tunnels.get(server.key)
        if (current is None or not current.open) and self._accepts(server.port):
            # Something here already answers on it: a tunnel opened by hand.
            return None
        try:
            self._tunnels.ensure(
                server.key, destination, target=server.port, port=server.port
            )
        except TunnelDown as down:
            return f"The tunnel to {destination} is down: {down}"
        return None

    # -- the page's edits ------------------------------------------------------

    def add(
        self, device: str, address: str, port: int, tunnel: Optional[str] = None
    ) -> Optional[Path]:
        """`add_entry` in the configuration the app connects with, then asked
        again at once (in the background)."""
        path = self._configuration()
        add_entry(path, device, address, port, tunnel)
        self._again()
        return path

    def remove(self, device: str) -> Optional[Path]:
        """`remove_entry`; a Device Server left serving nothing has both its
        tokens forgotten here."""
        path = self._configuration()
        before, _ = listed(path)
        remove_entry(path, device)
        after, _ = listed(path)
        left = {s.key for s in after}
        for server in before:
            if device in server.serves and server.key not in left:
                peers.forget(server.url)
                self._pairings.pop(server.key, None)
        self._again()
        return path

    # -- pairing through the gateways (C1) -------------------------------------

    def _server(self, key: str) -> RemoteDeviceServer:
        servers, _ = listed(self._configuration())
        found = next((s for s in servers if s.key == key), None)
        if found is None:
            raise PairRefused(f"The configuration has no Device Server at {key}.", 404)
        return found

    def _gateway_url(self, server: RemoteDeviceServer) -> str:
        """The device PC's gateway, through a tunnel of its own: it listens on
        its own PC only, so there is no reaching it without one."""
        if not server.tunnel:
            raise PairRefused(
                "Pairing goes through the device PC's gateway, which only its own PC "
                "reaches: give this Device Server an SSH tunnel (user@host) first."
            )
        try:
            tunnel = self._tunnels.ensure(
                f"{server.key} gateway", server.tunnel, target=GATEWAY_PORT
            )
        except TunnelDown as down:
            raise PairRefused(
                f"The tunnel to {server.tunnel} is down: {down}"
            ) from None
        return tunnel.url

    def pair(self, key: str) -> None:
        """Ask the Device Server's gateway to pair, in the background: the
        operator there allows it, and both tokens are kept here."""
        server = self._server(key)
        if self._pairings.get(key) == ASKING:
            return
        url = self._gateway_url(server)
        self._pairings[key] = ASKING
        threading.Thread(target=self._pairing, args=(server, url), daemon=True).start()
        self._again()

    def _pairing(self, server: RemoteDeviceServer, url: str) -> None:
        key, ask_id = server.key, None
        until = self._clock() + self._pair_lapse_s
        outcome = "It lapsed: nobody allowed it at the device PC."
        while not self._stop.is_set() and self._clock() < until:
            try:
                status, answer = self._post(
                    url,
                    "/gateway/peers/ask",
                    {"name": self._name(), "id": ask_id, "wait_s": 20},
                    None,
                    30.0,
                )
            except Exception as error:  # noqa: BLE001 - said on the row
                outcome = f"Its gateway didn't answer: {type(error).__name__}."
                break
            if status != 200 or not isinstance(answer, dict):
                message = (
                    (answer or {}).get("detail", {}) if isinstance(answer, dict) else {}
                )
                outcome = (
                    message.get("message") if isinstance(message, dict) else None
                ) or f"Its gateway answered {status}."
                break
            state = answer.get("state")
            if state == "waiting":
                ask_id = answer.get("id")
                continue
            if state == "allowed":
                peers.keep(server.url, answer["device_token"], answer["gateway_token"])
                outcome = "Paired."
                logger.info("Paired with the Device Server at %s", key)
            else:
                outcome = answer.get("message") or f"Not paired ({state})."
            break
        self._pairings[key] = outcome
        self._check_quietly()

    def switch(self, key: str, act: str) -> dict:
        """Start or stop the Device Server through its PC's gateway, with
        the token pairing kept. Its answer, as that gateway gives it."""
        if act not in ("start", "stop"):
            raise PairRefused(f"Start or stop, not {act!r}.", 422)
        server = self._server(key)
        token = peers.kept(server.url)
        if token is None:
            raise PairRefused("Pair with it first: Pair, then Allow at the device PC.")
        url = self._gateway_url(server)
        try:
            status, answer = self._post(
                url, f"/gateway/device-server/{act}", None, token, 15.0
            )
        except Exception as error:  # noqa: BLE001
            raise PairRefused(
                f"Its gateway didn't answer: {type(error).__name__}.", 502
            ) from None
        if status in (401, 403):
            raise PairRefused(
                "Its gateway refused: it was unpaired there. Pair again.", 403
            )
        if status != 200 or not isinstance(answer, dict):
            raise PairRefused(f"Its gateway answered {status}.", 502)
        self._again()
        return answer

    def _again(self) -> None:
        threading.Thread(target=self._check_quietly, daemon=True).start()

    def _check_quietly(self) -> None:
        try:
            self.check()
        except Exception:  # noqa: BLE001
            logger.exception("Checking the Device Servers failed")

    def file(self) -> Optional[str]:
        path = self._configuration()
        return str(path) if path is not None else None

    # -- what the page shows ---------------------------------------------------

    def state(self) -> List[dict]:
        with self._lock:
            return [s.to_dict() for s in self._servers]

    def problem(self) -> Optional[str]:
        """Why the configuration's Device Servers couldn't be read, if so."""
        with self._lock:
            return self._problem


def _devices(health: Dict[str, Any]) -> List[Dict[str, Any]]:
    listed_ = health.get("devices")
    if not isinstance(listed_, dict):
        return []
    return [
        {
            "name": name,
            "ok": bool(v.get("ok", True)) if isinstance(v, dict) else True,
            "detail": v.get("detail") if isinstance(v, dict) else None,
        }
        for name, v in listed_.items()
    ]
