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
``np.save`` bytes rather than JSON, and arrives as the same array. A beam's image
comes back as a TIFF with its metadata, and arrives as the same ``FibsemImage``.

``RemoteFM``, ``RemoteCamera``, ``RemoteLightSource``, ``RemoteFilterSet`` and
``RemoteObjective`` are the FM's parts from ``fibsem.devices.fm``; ``connect_remote_fm``
builds them from what a server has.

``RemoteStage``, ``RemoteChamber`` and ``RemoteManipulator`` move, pump and vent on the
server, and wait for as long as that takes. What they say about themselves beyond
their parameters (the stage's frame, shuttle and the device each orientation is at;
the needle's axes and named positions) comes from the device's ``facts`` in its
description, and a stage's poses from its ``pose_table`` query. A move
is checked against the stage limits here and again on the server.

``RemoteScanner`` scans a frame on the server, in one call, and knows its limits from
its facts. A plugin driver's device is made with the class its ``DriverEntry`` names
for the type (``remote_devices``) when that driver is installed here too, so its own
parameters cross the wire; otherwise with the type's class above (`remote_class`).

Not yet: the one-commander lease.
"""

from __future__ import annotations

import importlib
import logging
import math
import threading
from copy import deepcopy
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
)

import httpx
import numpy as np
from psygnal import Signal

from fibsem.client.events import HEARTBEAT, EventStream
from fibsem.client.transport import ServerError, ServerUnreachable, Transport
from fibsem.codec import decode
from fibsem.devices.beam import Beam
from fibsem.devices.chamber import Chamber
from fibsem.devices.chamber_camera import ChamberCamera
from fibsem.devices.core import (
    Device,
    ParameterMetadata,
    command,
)
from fibsem.devices.entries import REMOTE_DRIVER
from fibsem.devices.fm import FM, Camera, FilterSet, LightSource, Objective
from fibsem.devices.manipulator import Manipulator
from fibsem.devices.scanner import Scanner
from fibsem.devices.stage import Stage
from fibsem.devices.wire import (
    Frame,
    to_wire,
)
from fibsem.drivers.registry import DeviceBuilder, get_driver
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemManipulatorPosition,
    FibsemRectangle,
    FibsemStagePosition,
    ImageSettings,
    Point,
    RangeLimit,
)

if TYPE_CHECKING:
    from fibsem.drivers.registry import BuildContext
    from fibsem.fm.remote import RemoteFluorescenceMicroscope
    from fibsem.structures import DeviceEntry

READ_TIMEOUT = 5.0
# Commands that stop something, which carry a reason (fibsem.server.server).
_STOPS = frozenset(("stop", "stop_live", "cancel"))
WRITE_TIMEOUT = 60.0  # a plasma gas change takes a while
UNTIL_DONE = (READ_TIMEOUT, None)
"""For a move, a pump or a vent: it takes as long as it takes, and ends by answering
or by the connection dropping, so no read timeout."""


def _timeout(timeout: Any) -> Any:
    """A (connect, read) pair as httpx takes it."""
    if isinstance(timeout, tuple):
        connect, read = timeout
        return httpx.Timeout(read, connect=connect)
    return timeout


class RemoteDeviceError(RuntimeError):
    """The server answered, but with an error that has no local meaning."""


class RemoteDeviceUnreachable(ConnectionError):
    """The server could not be reached. Nothing about the device is known right now."""


def _range(payload: Dict[str, Any]) -> RangeLimit:
    """An end sent as None is unbounded: JSON has no infinity."""
    low, high = payload["min"], payload["max"]
    return RangeLimit(
        min=-math.inf if low is None else low, max=math.inf if high is None else high
    )


def _limits(payload: Any) -> Any:
    """The inverse of the server's limits: a RangeLimit, or one per field."""
    if payload is None:
        return None
    if set(payload) == {"min", "max"}:
        return _range(payload)
    return {name: _range(limit) for name, limit in payload.items()}


def _metadata(payload: Dict[str, Any], type_: type) -> ParameterMetadata:
    choices = payload.get("choices")
    return ParameterMetadata(
        limits=_limits(payload.get("limits")),
        choices=None if choices is None else [decode(type_, c) for c in choices],
        settable=payload.get("settable", True),
        native_max=payload.get("native_max"),
        native_unit=payload.get("native_unit"),
    )


class DeviceClient:
    """One connection to one device server, shared by all its remote devices."""

    disconnected = Signal()
    """The event stream closed: the server stopped or the network dropped."""

    reconnected = Signal()
    """The server is back, or answered for the first time. Metadata and values were
    read again, and every parameter that changed in the gap emitted ``changed``;
    devices built offline are bound."""

    def __init__(
        self,
        host: str,
        port: int,
        heartbeat: float = HEARTBEAT,
        token: Optional[str] = None,
        events: bool = True,
        root: str = "",
    ):
        """*token*, when given, goes on every HTTP request as a bearer token, for
        the App Server, which serves the same routes behind one under *root*
        (``"/microscope"``, ``fibsem.server.server``), and on opening its event
        stream at ``{root}/events``.

        *events* False never opens the event stream. Reads, writes and commands
        work the same; but nothing signals a change made on the far side,
        ``cached`` is only as current as this client's last read or write,
        metadata (limits, choices) only as current as its last
        ``refresh_metadata()``, and ``connected``, ``disconnected`` and
        ``reconnected`` never change or fire.
        """
        self.base_url = f"http://{host}:{port}{root}"
        self.events = events
        self._transport = Transport(self.base_url, token, discover=False)
        self._devices: Dict[str, Device] = {}
        self._events = EventStream(
            f"ws://{host}:{port}{root}/events",
            self._dispatch,
            on_open=self._opened,
            on_lost=self.disconnected.emit,
            headers=None if token is None else {"Authorization": f"Bearer {token}"},
            heartbeat=heartbeat,
        )
        # Values this client wrote whose change event hasn't come back yet. The
        # event can arrive before or after the write returns, so it is matched by
        # value, once, rather than by timing.
        self._echoes: Dict[Tuple[str, str], List[Any]] = {}
        self._echoes_lock = threading.Lock()

    @property
    def connected(self) -> bool:
        """The event stream is open."""
        return self._events.connected

    # -- requests -------------------------------------------------------------------

    def request(self, method: str, path: str, timeout: Any, json: Any = None) -> Any:
        """The answer as data (`Transport.fetch`). *timeout* is seconds, or
        (connect, read) with None for no read limit."""
        try:
            return self._transport.fetch(
                method, f"/{path}", json, timeout=_timeout(timeout)
            )
        except ServerUnreachable as error:
            # One line saying what is unreachable and why.
            reason = "connection refused or dropped"
            if error.timed_out:
                reason = f"no answer within {timeout} s"
            raise RemoteDeviceUnreachable(
                f"{method} {self.base_url}/{path}: {reason}"
            ) from None
        except ServerError as error:
            raise RemoteDeviceError(str(error)) from None

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
        if self.events:
            self._events.start(wait=READ_TIMEOUT if wait else None)

    def _opened(self, first: bool) -> None:
        bound = self._bind_offline()
        if not first:
            self._resync()
        if not first or bound:
            logging.info(f"device events from {self.base_url} resumed")
            self.reconnected.emit()

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
        device = self._devices.get(event.get("device"))
        param = device.parameters.get(event["parameter"]) if device else None
        if param is None:
            return
        if event["kind"] == "changed":
            if self._take_echo((event["device"], event["parameter"]), event["value"]):
                return  # our own write: its write path signals it
            param.report(decode(param.type, event["value"]))  # only if news
        elif event["kind"] == "metadata":
            param.refresh_metadata()

    def _expect_echo(self, key: Tuple[str, str], value: Any) -> None:
        if not self.events:
            return  # no stream, so no echo will come back to take it
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
        self._events.close()
        self._transport.close()


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
        self.check_roles()
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
        self.check_roles()
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
        self._apply_facts(description.get("facts", {}))

    def _apply_facts(self, facts: Dict[str, Any]) -> None:
        """Take on what the server's device says about itself (``Device.facts``)."""

    def _prime(self) -> None:
        # Fill the cache once, after the event stream is open so no change falls in
        # between: displays read ``cached`` and must not start empty.
        for param in self.parameters.values():
            param.get_value()

    def call_command(
        self, command: str, timeout: Any = WRITE_TIMEOUT, **kwargs: Any
    ) -> Any:
        """Run one of the device's commands on the server. Structured arguments
        (a ``Point``) cross the wire like parameter values do. ``timeout`` is the
        requests timeout: seconds, or (connect, read)."""
        path = f"devices/{self.name}/commands/{command}"
        body = {"kwargs": {name: to_wire(value) for name, value in kwargs.items()}}
        if command in _STOPS:
            # The App Server records why with every stop; a device server ignores it.
            body["reason"] = f"{self.name} {command} from a remote microscope"
        answer = self.client.request("POST", path, timeout, json=body)
        if isinstance(answer, (np.ndarray, Frame, FibsemImage, list)):
            return answer
        return answer["result"]

    def call_query(self, query: str, timeout: Any = READ_TIMEOUT, **kwargs: Any) -> Any:
        """Ask one of the device's queries on the server: a read that takes
        arguments and moves nothing."""
        path = f"devices/{self.name}/queries/{query}"
        body = {"kwargs": {name: to_wire(value) for name, value in kwargs.items()}}
        return self.client.request("POST", path, timeout, json=body)["result"]

    def _reader(self, name: str) -> Callable[[], Any]:
        path = f"devices/{self.name}/{name}"
        type_ = self.declared_parameters()[name].type
        return lambda: decode(
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

    _installed_gas: Optional[str] = None

    def _apply_facts(self, facts: Dict[str, Any]) -> None:
        self._installed_gas = facts.get("installed_plasma_gas")

    def installed_plasma_gas(self) -> Optional[str]:
        return self._installed_gas

    @command
    def acquire(self, image_settings: Optional[ImageSettings] = None) -> FibsemImage:
        """Acquire on the server. The image comes back whole, as a TIFF with its
        metadata, stamped and saved as the server does it."""
        image = self.call_command(
            "acquire", timeout=UNTIL_DONE, image_settings=image_settings
        )
        if image_settings is not None:
            self._last_image_settings = image_settings
        self.image_acquired.emit(image)
        return image


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

    runs_elsewhere = True

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

    def _acquire_z_stack(
        self,
        channels: List[Dict[str, Any]],
        positions: List[float],
        order: str,
        restore_position: Optional[float],
    ) -> List[Frame]:
        if "acquire_z_stack" not in self.server_commands:
            # A server from before the command: the stack runs here, one move and one
            # frame per request, as before.
            return super()._acquire_z_stack(
                channels, positions, order, restore_position
            )
        # A whole stack takes as long as its exposures and moves; it ends by
        # answering, or by the connection dropping, so no read timeout.
        return self.call_command(
            "acquire_z_stack",
            timeout=(READ_TIMEOUT, None),
            channels=channels,
            positions=positions,
            order=order,
            restore_position=restore_position,
        )

    def _cancelled(self) -> None:
        if "cancel" in self.server_commands:
            self.call_command("cancel")


class RemoteStage(RemoteDevice, Stage):
    """A stage on another computer. The limit check runs here and again on the
    server, which moves and answers when the stage has stopped."""

    _shuttle = False
    _mirrors = False
    _devices_at_pose: Dict[str, Optional[str]] = {}

    def _apply_facts(self, facts: Dict[str, Any]) -> None:
        self.frame = facts.get("frame", self.frame)
        self._shuttle = bool(facts.get("has_builtin_shuttle", False))
        self._mirrors = bool(facts.get("mirrors_back_view", False))
        self._devices_at_pose = dict(facts.get("device_at_pose", {}))
        self._poses: Dict[Tuple[float, float, float], Any] = {}
        if "position" in self.parameters:
            self._build_axes()

    def has_builtin_shuttle(self) -> bool:
        return self._shuttle

    def mirrors_back_view(self) -> bool:
        return self._mirrors

    def device_at_pose(self, orientation: str) -> Optional[str]:
        return self._devices_at_pose.get(orientation)

    def poses(
        self, rotation_reference: float, shuttle_pre_tilt: float, fib_column_tilt: float
    ) -> Dict[str, FibsemStagePosition]:
        """The server's stage's poses, asked once per geometry: they are a pure
        function of it, and orientation checks ask often."""
        key = (rotation_reference, shuttle_pre_tilt, fib_column_tilt)
        if key not in self._poses:
            answer = self.call_query(
                "pose_table",
                timeout=READ_TIMEOUT,
                rotation_reference=rotation_reference,
                shuttle_pre_tilt=shuttle_pre_tilt,
                fib_column_tilt=fib_column_tilt,
            )
            self._poses[key] = {
                name: FibsemStagePosition.from_dict(pose)
                for name, pose in answer.items()
            }
        return {name: deepcopy(pose) for name, pose in self._poses[key].items()}

    def _move_absolute(self, position: FibsemStagePosition) -> None:
        self.call_command("move_absolute", timeout=UNTIL_DONE, position=position)

    def _move_relative(self, delta: FibsemStagePosition) -> None:
        self.call_command("move_relative", timeout=UNTIL_DONE, delta=delta)

    def _home(self) -> None:
        self.call_command("home", timeout=UNTIL_DONE)

    def _link(self) -> None:
        self.call_command("link")


class RemoteChamber(RemoteDevice, Chamber):
    """Pumping and venting run on the server, for as long as they take."""

    def _pump(self) -> None:
        self.call_command("pump", timeout=UNTIL_DONE)

    def _vent(self) -> None:
        self.call_command("vent", timeout=UNTIL_DONE)


class RemoteChamberCamera(RemoteDevice, ChamberCamera):
    """The frame comes back as the array the server's camera gave."""

    def _acquire(self) -> np.ndarray:
        return self.call_command("acquire")


class RemoteManipulator(RemoteDevice, Manipulator):
    """A needle on another computer. Its axes come from the limits of the server's
    ``position``, and its named positions are read when it binds; a saved position is read from the server each time, since where a named
    position is can change (Thermo gives it in stage coordinates)."""

    _named: Tuple[str, ...] = ()
    _installed: Optional[bool] = None

    def _apply_facts(self, facts: Dict[str, Any]) -> None:
        self._named = tuple(facts.get("named_positions", ()))
        self._installed = facts.get("is_installed")
        self._build_axes()

    def is_installed(self) -> Optional[bool]:
        return self._installed

    def named_positions(self) -> List[str]:
        return list(self._named)

    def saved_position(self, name: str) -> FibsemManipulatorPosition:
        path = f"devices/{self.name}"
        facts = self.client.request("GET", path, READ_TIMEOUT).get("facts", {})
        saved = facts.get("saved_positions", {})
        if name not in saved:
            raise ValueError(f"{self.name} has no saved position '{name}'")
        return FibsemManipulatorPosition.from_dict(saved[name])

    def _insert(self, name: str) -> None:
        self.call_command("insert", timeout=UNTIL_DONE, name=name)

    def _retract(self) -> None:
        self.call_command("retract", timeout=UNTIL_DONE)

    def _move_absolute(self, position: FibsemManipulatorPosition) -> None:
        self.call_command("move_absolute", timeout=UNTIL_DONE, position=position)

    def _move_relative(self, delta: FibsemManipulatorPosition) -> None:
        self.call_command("move_relative", timeout=UNTIL_DONE, delta=delta)

    def _stop(self) -> None:
        self.call_command("stop")


class RemoteScanner(RemoteDevice, Scanner):
    """A scan generator on another computer. A frame is scanned there, in one call,
    and comes back as the array. Its limits come from its facts, so a frame outside
    them is refused here before the request, and a beam it is bound to narrows to
    them as it would to a local one. ``facts()`` is what the server's device says."""

    _served_facts: Dict[str, Any] = {}

    def _apply_facts(self, facts: Dict[str, Any]) -> None:
        self._served_facts = deepcopy(facts)

    def facts(self) -> Dict[str, Any]:
        return deepcopy(self._served_facts)

    def _dwell_time_limits(self) -> Optional[RangeLimit]:
        limits = self._served_facts.get("dwell_time_limits")
        return None if limits is None else RangeLimit(*limits)

    def _resolution_limits(self) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        limits = self._served_facts.get("resolution_limits")
        if limits is None:
            return None
        low, high = limits
        return (int(low[0]), int(low[1])), (int(high[0]), int(high[1]))

    def _acquire(self, resolution: Tuple[int, int], dwell_time: float) -> np.ndarray:
        return self.call_command(
            "acquire", timeout=UNTIL_DONE, resolution=resolution, dwell_time=dwell_time
        )

    def _stop(self) -> None:
        self.call_command("stop")


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
    owned = client is None
    client = client if client is not None else DeviceClient(host, port)
    try:
        descriptions = client.describe()
    except RemoteDeviceUnreachable as error:
        if not offline:
            if owned:
                client.close()
            raise
        logging.warning(
            f"The fluorescence microscope is not reachable ({error}). It will connect "
            "when its device server starts."
        )
        devices = _remote_fm_devices(REMOTE_FM_PARTS, client)
        return {name: device.connect_offline() for name, device in devices.items()}
    devices = _remote_fm_devices(
        [name for name in descriptions if name in REMOTE_FM_PARTS], client
    )
    if not devices and owned:  # no FM there, and nothing to close the client later
        client.close()
    return {
        name: device.connect(descriptions[name]) for name, device in devices.items()
    }


def _remote_fm_devices(
    names: Iterable[str], client: DeviceClient
) -> Dict[str, RemoteDevice]:
    """The named remote FM devices, unconnected, with the group's roles filled by the
    parts beside it."""
    devices = {name: REMOTE_FM_PARTS[name](name=name, client=client) for name in names}
    group = devices.get("fm")
    if group is not None:
        group.fill_roles(
            **{name: device for name, device in devices.items() if name != "fm"}
        )
    return devices


# The remote device for each entry type a ``driver: remote`` entry can have. The FM's
# group is not one: an ``fm`` entry is the whole FM (``build_remote_fm``).
REMOTE_DEVICE_TYPES: Dict[str, type] = {
    "beam": RemoteBeam,
    **{name: part for name, part in REMOTE_FM_PARTS.items() if name != "fm"},
    "stage": RemoteStage,
    "chamber": RemoteChamber,
    "chamber_camera": RemoteChamberCamera,
    "manipulator": RemoteManipulator,
    "scan_generator": RemoteScanner,
}

# The types whose device has one name, its type's, on every backend.
_NAMED_AS_TYPE = ("stage", "chamber", "chamber_camera", "manipulator")


def remote_class(entry_type: str, description: Mapping[str, Any]) -> Optional[type]:
    """The class a remote copy of the served device *description* is made with, for an
    entry of *entry_type*; None if there is none.

    The class the server's driver names for the type (``DriverEntry.remote_devices``)
    when that driver is installed here, else the type's own (`REMOTE_DEVICE_TYPES`).
    Falling back loses whatever the driver's device has beyond its type's interface,
    and says so. A served device built as another type is an error.
    """
    name = description.get("name")
    served_type = description.get("type")
    if served_type is not None and served_type != entry_type:
        raise ValueError(
            f"'{name}' is a {served_type} on the device server, not a {entry_type}"
        )
    interface = REMOTE_DEVICE_TYPES.get(entry_type)
    driver = description.get("driver")
    if not driver or driver == REMOTE_DRIVER:
        return interface
    try:
        target = get_driver(driver).remote_devices.get(entry_type)
    except NotImplementedError:
        why = f"its driver '{driver}' is not installed here"
    else:
        if target is not None:
            module, _, attribute = target.partition(":")
            return getattr(importlib.import_module(module), attribute)
        why = f"its driver '{driver}' names no remote class for a {entry_type}"
    if interface is None:
        raise ValueError(f"there is no remote '{entry_type}' device: {why}")
    lost = sorted(
        set(description.get("parameters", {})) - set(interface.declared_parameters())
    )
    if lost:
        logging.warning(
            f"'{name}' is reached as a plain {entry_type} ({interface.__name__}), as "
            f"{why}. Not available here: {', '.join(lost)}. Install the driver's "
            "plugin on this computer too to use them."
        )
    return interface


def build_remote_device(entry: "DeviceEntry", context: "BuildContext") -> Device:
    """The device a ``driver: remote`` entry names, on the device server at its
    ``address`` and ``port``, made with `remote_class`.

    The entry's name is the device's name on the server. Entries at one address share
    one ``DeviceClient``, and the server is asked what it has once. A server that does
    not answer, or has no device of that name, fails the build: the device is left out
    with a warning, or connect fails when the entry is ``required``.
    """
    address, port = entry.options.get("address"), entry.options.get("port")
    if address is None or port is None:
        raise ValueError("a remote device needs an `address` and a `port`")
    client, descriptions = _server(context, str(address), int(port))
    description = descriptions.get(entry.name)
    if description is None:
        raise ValueError(
            f"the device server at {address}:{port} has no device '{entry.name}' "
            f"(it has {', '.join(descriptions) or 'none'})"
        )
    device_class = remote_class(entry.type, description)
    if device_class is None:
        raise ValueError(f"the remote driver has no '{entry.type}' device")
    if issubclass(device_class, RemoteBeam):
        beam_type = BeamType.__members__.get(entry.name.upper())
        if beam_type is None:
            raise ValueError("a beam is named 'electron' or 'ion'")
        device: RemoteDevice = RemoteBeam(beam_type, client=client)
    elif entry.type in _NAMED_AS_TYPE:
        if entry.name != entry.type:
            raise ValueError(f"a {entry.type} is named '{entry.type}'")
        device = device_class(client=client)
    else:
        device = device_class(name=entry.name, client=client)
    return device.connect(description)


def build_remote_fm(
    entry: "DeviceEntry", context: "BuildContext"
) -> "RemoteFluorescenceMicroscope":
    """The FM API over the FM on its own PC, which an ``fm`` entry with
    ``driver: remote`` names at its ``address`` and ``port``.

    A server that isn't answering yet gives an FM that is offline: its reads fail
    closed, and it comes online by itself when the server starts (FIB-1086), so the
    beams never wait for the FM's PC. With ``required: true`` an unreachable server
    fails the build instead. The objective calibration is pushed again each time the
    FM comes (back) online, since a restarted FM computer starts from its own
    defaults.
    """
    from fibsem.fm.remote import RemoteFluorescenceMicroscope

    if entry.options.get("mount_transform") not in (None, "none"):
        logging.warning(
            "The configuration states the FM's mount_transform, but this FM is "
            "remote: its server states its own (--mount-transform), and that is "
            "what is used."
        )
    address, port = entry.options.get("address"), entry.options.get("port")
    if address is None or port is None:
        raise ValueError(
            "the FM is configured as remote but has no `address` and `port`"
        )
    microscope = context.microscope
    try:
        fm = RemoteFluorescenceMicroscope.connect(
            str(address), int(port), parent=microscope, offline=not entry.required
        )
    except Exception as error:
        raise ConnectionError(
            f"could not connect to the FM at {address}:{port}: {error}"
        ) from error
    context.on_disconnect(fm.client.close)
    apply_calibration = getattr(microscope, "_apply_fluorescence_calibration", None)
    if apply_calibration is not None:
        fm.client.reconnected.connect(apply_calibration)
    return fm


def _server(
    context: "BuildContext", address: str, port: int
) -> Tuple[DeviceClient, Dict[str, Any]]:
    """The client for the server at *address*:*port* and what the server has, made
    the first time this connect asks for it. A server that did not answer is not
    asked again in this connect."""
    servers = context.shared.setdefault(REMOTE_DRIVER, {})
    key = (address, port)
    if key not in servers:
        client = DeviceClient(address, port)
        try:
            servers[key] = (client, client.describe())
        except Exception as error:
            client.close()
            servers[key] = error
        else:
            # one client per server, shared by its devices: closed at disconnect
            context.on_disconnect(client.close)
    server = servers[key]
    if isinstance(server, Exception):
        raise server
    return server


class _RemoteBuilders(Mapping[str, DeviceBuilder]):
    """A builder for every type: one with no remote class here may have one in the
    server's driver, which is known only once the server answers (`remote_class`).
    Lists the types the remote driver has a class for."""

    _device = DeviceBuilder("fibsem.drivers.remote.devices:build_remote_device")
    _fm = DeviceBuilder("fibsem.drivers.remote.devices:build_remote_fm")

    def __getitem__(self, device_type: str) -> DeviceBuilder:
        return self._fm if device_type == "fm" else self._device

    def __iter__(self) -> Iterator[str]:
        return iter([*REMOTE_DEVICE_TYPES, "fm"])

    def __len__(self) -> int:
        return len(REMOTE_DEVICE_TYPES) + 1


DEVICE_BUILDERS: Mapping[str, DeviceBuilder] = _RemoteBuilders()
"""How the remote driver builds each type, as the registry gives a driver's builders
(``fibsem.drivers.registry.device_builder``)."""
