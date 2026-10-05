"""The remote driver: devices that live on another computer, used as if they were local.

The coordinator side of ``fibsem.server.devices``. ``RemoteDevice`` is
a mixin: combined with a device type it gives that type's parameters, backed by HTTP.

    class RemoteBeam(RemoteDevice, Beam): ...

At ``connect()`` it asks the server what the device has and binds exactly that: a
parameter the server doesn't list stays absent, as on any backend. Its type and unit
must match the local declaration, or connect raises. Limits, choices and settable
come from the server. After that:

- ``get_value()`` is a live read over the network. It raises
  ``RemoteDeviceUnreachable`` when the server can't be reached, so a guard fails
  closed and never trusts a stale value;
- ``set_value()`` checks locally with the server's metadata, then the server checks
  again and writes;
- ``cached`` needs no network: the event stream keeps it current, including changes
  made on the far side (the far computer's own UI, another client);
- commands run on the server with ``call_command``.

The event stream carries a heartbeat: when the server stops answering pings (a
pulled cable, a frozen PC), ``DeviceClient.disconnected`` fires within two heartbeats,
without waiting for the next read. ``DeviceClient.health()`` asks the server how its
devices are. The client keeps retrying; when the server is back it reads every
parameter's metadata and value again, so anything that changed in the gap is
signalled, and then fires ``DeviceClient.reconnected``.

A server that isn't running yet need not stop the coordinator connecting: with
``offline=True``, ``connect_remote_fm`` builds the FM's devices unbound, and the same
retry loop binds them when the server first answers, then fires ``reconnected``.
Until then each device's ``online`` is False and its parameters are absent.

A command that returns an array (a camera frame, an FM channel) comes back as
``np.save`` bytes rather than JSON, and arrives as the same array.

``RemoteFM``, ``RemoteCamera``, ``RemoteLightSource``, ``RemoteFilterSet`` and
``RemoteObjective`` are the FM's parts from ``fibsem.devices.fm``; ``connect_remote_fm``
builds them from what a server has.

Not yet: the one-commander lease, and a beam's ``acquire`` (a
``FibsemImage`` with its metadata).
"""

from __future__ import annotations

import io
import json
import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import requests
from psygnal import Signal

from fibsem.devices.beam import Beam
from fibsem.devices.core import (
    Device,
    ParameterMetadata,
    ParameterReadOnly,
    ParameterUnavailable,
    command,
)
from fibsem.devices.fm import FM, Camera, FilterSet, LightSource, Objective
from fibsem.devices.wire import (
    FRAME_METADATA_HEADER,
    NPY_MEDIA_TYPE,
    Frame,
    from_wire,
    to_wire,
)
from fibsem.structures import BeamType, FibsemRectangle, Point, RangeLimit

READ_TIMEOUT = 5.0
HEARTBEAT = 5.0  # seconds between pings; a server silent for as long again is gone
WRITE_TIMEOUT = 60.0  # a plasma gas change takes a while


class RemoteDeviceError(RuntimeError):
    """The server answered, but with an error that has no local meaning."""


class RemoteDeviceUnreachable(ConnectionError):
    """The server could not be reached. Nothing about the device is known right now."""


_ERRORS: Dict[str, Callable[[str], Exception]] = {
    "ParameterUnavailable": ParameterUnavailable,
    "ParameterReadOnly": ParameterReadOnly,
    "TypeError": TypeError,
    "ValueError": ValueError,
}


def _limits(payload: Any) -> Any:
    """The inverse of the server's limits: a RangeLimit, or one per field."""
    if payload is None:
        return None
    if set(payload) == {"min", "max"}:
        return RangeLimit.from_dict(payload)
    return {name: RangeLimit.from_dict(limit) for name, limit in payload.items()}


def _metadata(payload: Dict[str, Any], type_: type) -> ParameterMetadata:
    choices = payload.get("choices")
    return ParameterMetadata(
        limits=_limits(payload.get("limits")),
        choices=None if choices is None else [from_wire(type_, c) for c in choices],
        settable=payload.get("settable", True),
    )


class DeviceClient:
    """One connection to one device server, shared by all its remote devices."""

    disconnected = Signal()
    """The event stream closed: the server stopped or the network dropped."""

    reconnected = Signal()
    """The server is back, or answered for the first time. Metadata and values were
    read again, and every parameter that changed in the gap emitted ``changed``;
    devices built offline are bound."""

    def __init__(self, host: str, port: int, heartbeat: float = HEARTBEAT):
        self.heartbeat = heartbeat
        self.base_url = f"http://{host}:{port}"
        self.events_url = f"ws://{host}:{port}/events"
        self._session = requests.Session()
        self._devices: Dict[str, Device] = {}
        self._events: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._socket: Any = None
        self._closing = threading.Event()
        # Values this client wrote whose change event hasn't come back yet. The
        # event can arrive before or after the write returns, so it is matched by
        # value, once, rather than by timing.
        self._echoes: Dict[Tuple[str, str], List[Any]] = {}
        self._echoes_lock = threading.Lock()
        self.connected = False

    # -- requests -------------------------------------------------------------------

    def request(self, method: str, path: str, timeout: float, **kwargs: Any) -> Any:
        try:
            response = self._session.request(
                method, f"{self.base_url}/{path}", timeout=timeout, **kwargs
            )
        except requests.RequestException as error:
            # One line saying what is unreachable and why; the urllib3 chain under a
            # refused connection is five tracebacks deep and says nothing more.
            reason = type(error).__name__
            if isinstance(error, requests.Timeout):
                reason = f"no answer within {timeout} s"
            elif isinstance(error, requests.ConnectionError):
                reason = "connection refused or dropped"
            raise RemoteDeviceUnreachable(
                f"{method} {self.base_url}/{path}: {reason}"
            ) from None
        if response.ok:
            if response.headers.get("content-type", "").startswith(NPY_MEDIA_TYPE):
                data = np.load(io.BytesIO(response.content), allow_pickle=False)
                metadata = response.headers.get(FRAME_METADATA_HEADER)
                if metadata is not None:
                    return Frame(data, json.loads(metadata))
                return data
            return response.json()
        try:
            detail = response.json().get("detail")
        except ValueError:  # not JSON: an unhandled server error
            detail = response.text
        if isinstance(detail, dict) and detail.get("error") in _ERRORS:
            raise _ERRORS[detail["error"]](detail["detail"])
        raise RemoteDeviceError(f"{method} {path}: {response.status_code} {detail}")

    def health(self) -> Dict[str, Any]:
        """The server's own report: is it up, and can each device reach its hardware.

        For a status display or a check before a long run. It is not a guard: a
        device healthy a second ago says nothing about the next read, so guards
        keep reading live.
        """
        return self.request("GET", "health", READ_TIMEOUT)

    def describe(self) -> Dict[str, Any]:
        return self.request("GET", "devices", READ_TIMEOUT)

    # -- events ---------------------------------------------------------------------

    def register(self, device: Device, wait: bool = True) -> None:
        """Follow the device's events. The first device starts the event stream; with
        ``wait``, until it is open, so no change is missed from here on."""
        self._devices[device.name] = device
        if self._events is None:
            self._events = threading.Thread(target=self._listen, daemon=True)
            self._events.start()
            if wait:
                self._ready.wait(READ_TIMEOUT)

    def _listen(self) -> None:
        """Hold the event stream open; when it drops, retry until the server is back."""
        delay, first = 0.5, True
        while not self._closing.is_set():
            try:
                self._stream(resync=not first)
                delay = 0.5  # it was up: retry quickly next time
            except Exception as error:  # refused, dropped or timed out: all the same
                if self.connected or first:
                    logging.warning(f"device events from {self.base_url}: {error}")
            finally:
                first = False
                if self.connected:
                    self.connected = False
                    self.disconnected.emit()
                self._ready.set()
            if self._closing.wait(delay):
                return
            delay = min(delay * 2, self.heartbeat)

    def _stream(self, resync: bool) -> None:
        from websockets.sync.client import connect

        with connect(
            self.events_url,
            open_timeout=READ_TIMEOUT,
            ping_interval=self.heartbeat,
            ping_timeout=self.heartbeat,
            close_timeout=self.heartbeat,  # a silent server won't answer the close
        ) as socket:
            self._socket = socket
            bound = self._bind_offline()
            if resync:
                self._resync()
            self.connected = True
            self._ready.set()
            if resync or bound:
                logging.info(f"device events from {self.base_url} resumed")
                self.reconnected.emit()
            for message in socket:
                self._dispatch(json.loads(message))

    def _bind_offline(self) -> bool:
        """Bind the devices built while the server was down. True if there were any.

        Runs with the event stream already open, so no change falls between a
        device's first read and its events.
        """
        offline = [
            device
            for device in self._devices.values()
            if isinstance(device, RemoteDevice) and not device.online
        ]
        if not offline:
            return False
        descriptions = self.describe()
        for device in offline:
            description = descriptions.get(device.name)
            if description is None:
                logging.warning(f"{self.base_url} serves no '{device.name}'")
                continue
            try:
                device._bind_from(description)
                device._prime()
            except Exception as error:  # a type mismatch: stays offline, and says so
                logging.error(f"{device.name} at {self.base_url}: {error}")
        return True

    def _resync(self) -> None:
        """After a gap, events may have been missed: read metadata and values again."""
        for device in self._devices.values():
            for param in device.parameters.values():
                try:
                    param.refresh_metadata()
                    param.get_value()  # emits changed if it moved while we were away
                except Exception as error:
                    logging.warning(
                        f"{device.name}.{param.name} after reconnect: {error}"
                    )

    def _dispatch(self, event: Dict[str, Any]) -> None:
        device = self._devices.get(event["device"])
        param = device.parameters.get(event["parameter"]) if device else None
        if param is None:
            return
        if event["kind"] == "changed":
            if self._take_echo((event["device"], event["parameter"]), event["value"]):
                return  # our own write: its write path signals it
            param.report(from_wire(param.type, event["value"]))  # only if news
        elif event["kind"] == "metadata":
            param.refresh_metadata()

    def _expect_echo(self, key: Tuple[str, str], value: Any) -> None:
        with self._echoes_lock:
            self._echoes.setdefault(key, []).append(value)

    def _take_echo(self, key: Tuple[str, str], value: Any) -> bool:
        with self._echoes_lock:
            pending = self._echoes.get(key, [])
            for i, expected in enumerate(pending):
                if expected == value:
                    del pending[i]
                    return True
            return False

    def close(self) -> None:
        self._closing.set()
        if self._socket is not None:
            self._socket.close()
        self._session.close()


class RemoteDevice(Device):
    """Parameters and commands of a device served by ``fibsem.server.devices``.

    Mix it in before a device type. The device's ``name`` is its name on the server.
    """

    def __init__(self, *args: Any, client: DeviceClient, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.client = client
        self.server_commands: frozenset = frozenset()
        """The commands the server has for this device, known once bound. A server
        older than this client may lack some."""
        self.online = False
        """Bound to what the server has. False while a device built offline waits for
        its server; its parameters are absent until then."""

    def connect(self, description: Optional[Dict[str, Any]] = None) -> Device:
        if description is None:
            description = self.client.request(
                "GET", f"devices/{self.name}", READ_TIMEOUT
            )
        self._bind_from(description)
        self.client.register(self)
        self._prime()
        return self

    def connect_offline(self) -> Device:
        """Follow a server that isn't answering yet: nothing is bound until it does."""
        self.client.register(self, wait=False)
        return self

    def _bind_from(self, description: Dict[str, Any]) -> None:
        declared = self.declared_parameters()
        for name, info in description["parameters"].items():
            spec = declared.get(name)
            if spec is None:
                logging.debug(
                    f"{self.name}: the server has '{name}', not declared here"
                )
                continue
            if (info["type"], info["unit"]) != (spec.type.__name__, spec.unit):
                raise TypeError(
                    f"{self.name}.{name} is {info['type']} in {info['unit']!r} on the "
                    f"server, but is declared {spec.type.__name__} in {spec.unit!r}"
                )
            self.bind(
                name,
                read=self._reader(name),
                write=self._writer(name) if info["settable"] else None,
                metadata=self._metadata_reader(name, first=info),
            )
        self.server_commands = frozenset(description.get("commands", {}))
        self.online = True

    def _prime(self) -> None:
        # Fill the cache once, after the event stream is open so no change falls in
        # between: displays read ``cached`` and must not start empty.
        for param in self.parameters.values():
            param.get_value()

    def call_command(self, command: str, **kwargs: Any) -> Any:
        """Run one of the device's commands on the server. Structured arguments
        (a ``Point``) cross the wire like parameter values do."""
        path = f"devices/{self.name}/commands/{command}"
        body = {"kwargs": {name: to_wire(value) for name, value in kwargs.items()}}
        answer = self.client.request("POST", path, WRITE_TIMEOUT, json=body)
        if isinstance(answer, (np.ndarray, Frame)):
            return answer
        return answer["result"]

    def _reader(self, name: str) -> Callable[[], Any]:
        path = f"devices/{self.name}/{name}"
        type_ = self.declared_parameters()[name].type
        return lambda: from_wire(
            type_, self.client.request("GET", path, READ_TIMEOUT)["value"]
        )

    def _writer(self, name: str) -> Callable[[Any], None]:
        path = f"devices/{self.name}/{name}"

        def write(value: Any) -> None:
            key, wire = (self.name, name), to_wire(value)
            self.client._expect_echo(key, wire)  # matched in the form it comes back
            try:
                self.client.request("PUT", path, WRITE_TIMEOUT, json={"value": wire})
            except Exception:
                self.client._take_echo(key, wire)  # no write, so no echo
                raise

        return write

    def _metadata_reader(
        self, name: str, first: Dict[str, Any]
    ) -> Callable[[], ParameterMetadata]:
        """The description already carries the metadata, so binding costs no request;
        a later refresh (a dependency changed, a reconnect) asks the server again."""
        path = f"devices/{self.name}/{name}/metadata"
        pending = [first]
        type_ = self.declared_parameters()[name].type

        def read() -> ParameterMetadata:
            if pending:
                return _metadata(pending.pop(), type_)
            return _metadata(self.client.request("GET", path, READ_TIMEOUT), type_)

        return read


class RemoteBeam(RemoteDevice, Beam):
    """A beam on another computer. ``blank`` and ``unblank`` work unchanged: they set
    ``blanked``, which is remote. The scan-area commands run on the server; the local
    command then reads ``scanning_mode`` back, as on any backend."""

    def _spot(self, point: Point) -> None:
        self.call_command("spot", point=point)

    def _reduced_area(self, area: FibsemRectangle) -> None:
        self.call_command("reduced_area", area=area)

    def _full_frame(self) -> None:
        self.call_command("full_frame")

    @command(available=lambda beam: False)
    def acquire(self, image_settings: Any = None) -> Any:
        """Not yet: a FibsemImage needs its metadata sent too."""
        raise NotImplementedError("remote beam acquisition is not supported yet")


def connect_remote_beams(
    host: str, port: int, client: Optional[DeviceClient] = None
) -> Dict[BeamType, Beam]:
    """Build ``beams[BeamType]`` from the beams a device server has."""
    client = client if client is not None else DeviceClient(host, port)
    beams: Dict[BeamType, Beam] = {}
    for name, description in client.describe().items():
        beam_type = BeamType.__members__.get(name.upper())
        if beam_type is None:
            continue
        beams[beam_type] = RemoteBeam(beam_type, client=client).connect(description)
    return beams


class RemoteCamera(RemoteDevice, Camera):
    def _acquire(self) -> np.ndarray:
        return self.call_command("acquire")


class RemoteLightSource(RemoteDevice, LightSource):
    pass


class RemoteFilterSet(RemoteDevice, FilterSet):
    pass


class RemoteObjective(RemoteDevice, Objective):
    """Moves run on the server; position and state follow through the events."""

    def _insert(self) -> None:
        self.call_command("insert")

    def _retract(self) -> None:
        self.call_command("retract")

    def _move_absolute(self, position: float) -> None:
        self.call_command("move_absolute", position=position)

    def _move_relative(self, delta: float) -> None:
        self.call_command("move_relative", delta=delta)


class RemoteFM(RemoteDevice, FM):
    """Channel acquisition runs on the FM's computer, in one call."""

    # The server's FM watches live view; a second watchdog here would only stop it
    # when this process stops asking, which the server notices anyway.
    live_timeout = None

    def _start_live(self, channel: Optional[Dict[str, Any]]) -> None:
        # A server from before live view: frames are still pulled, each one a whole
        # acquisition, as before.
        if "start_live" in self.server_commands:
            self.call_command("start_live", channel=channel)

    def _stop_live(self) -> None:
        if "stop_live" in self.server_commands:
            self.call_command("stop_live")

    def _acquire_channel(self, channel: Optional[Dict[str, Any]]) -> np.ndarray:
        return self.call_command("acquire_channel", channel=channel)

    def _acquire_frame(self, channel: Optional[Dict[str, Any]]) -> Frame:
        if "acquire_frame" not in self.server_commands:
            # A server from before acquire_frame: the frame alone, and the client
            # reads the metadata as it used to.
            return Frame(self._acquire_channel(channel), {})
        return self.call_command("acquire_frame", channel=channel)


REMOTE_FM_PARTS = {
    "fm": RemoteFM,
    "camera": RemoteCamera,
    "light_source": RemoteLightSource,
    "filter_set": RemoteFilterSet,
    "objective": RemoteObjective,
}


def connect_remote_fm(
    host: str,
    port: int,
    client: Optional[DeviceClient] = None,
    offline: bool = False,
) -> Dict[str, Device]:
    """The FM's group and parts a device server has, by name.

    If the server can't be reached this raises ``RemoteDeviceUnreachable``, unless
    ``offline``: then every part is built unbound, and binds when the server answers.
    """
    client = client if client is not None else DeviceClient(host, port)
    try:
        descriptions = client.describe()
    except RemoteDeviceUnreachable as error:
        if not offline:
            raise
        logging.warning(
            f"The fluorescence microscope is not reachable ({error}). It will connect "
            "when its device server starts."
        )
        return {
            name: part(name=name, client=client).connect_offline()
            for name, part in REMOTE_FM_PARTS.items()
        }
    return {
        name: REMOTE_FM_PARTS[name](name=name, client=client).connect(description)
        for name, description in descriptions.items()
        if name in REMOTE_FM_PARTS
    }
