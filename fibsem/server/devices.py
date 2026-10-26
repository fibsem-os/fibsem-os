"""A device server: devices on this computer, reachable from a coordinator on another.

The far side of a remote device (the METEOR PC, say). It wraps any
``fibsem.devices.Device`` and exposes what the device already describes about itself:

    GET  /health                               up, and each device's hardware reachable
    GET  /devices                              every device: parameters and commands
    GET  /devices/{device}                     one device
    GET  /devices/{device}/{parameter}         a live read       -> {"value": ...}
    PUT  /devices/{device}/{parameter}         {"value": ...}    -> {"value": written}
    GET  /devices/{device}/{parameter}/metadata                  -> limits, choices, settable
    POST /devices/{device}/commands/{command}  {"kwargs": {...}} -> {"result": ...},
                                               TIFF for a beam's FibsemImage,
                                               or np.save bytes for an array, with a
                                               frame's metadata as JSON in the
                                               X-Frame-Metadata header; with
                                               "answer": "preview", any image as
                                               {"result": {summary, path, preview}}
    POST /devices/{device}/queries/{query}     {"kwargs": {...}} -> {"result": ...},
                                               a read that moves nothing
    WS   /events                               {"device", "parameter", "kind", "value"},
                                               with pings as a heartbeat
    WS   /devices/{beam}/live                  each live frame the beam acquires, as
                                               ``frames_to_bytes`` writes one: the
                                               latest, a slow client skipping frames

The coordinator side is ``fibsem.drivers.remote.devices``. A write runs the device's
``set_value``, so the server checks every value itself, whatever the client did.
Errors keep their meaning across the wire: the client raises the same exception
types a local device would.

The same routes are mounted in the App Server (``fibsem.server.server``) under
``/microscope/devices`` over ``microscope.devices``, behind its bearer token and level,
with the events at ``WS /microscope/events``. Served here on their own, every
route and stream needs this install's token or a paired computer's
(`fibsem.server.device_auth`); there is no one-commander lease.

Try it on one computer:

    python -m fibsem.server.devices --port 8765        # terminal 1: Demo beams
    python -m fibsem.server.devices --serve stage chamber manipulator  # or the rest
    python -m fibsem.server.devices --serve fm         # or a simulated FM's parts
    python -m fibsem.server.devices --config devices.yaml  # or what a file lists

On a METEOR's Linux PC, serving its real FM to the PC that drives the beams (see
INSTALLATION.md, "Delmic METEOR"), from a file naming the FM and where to serve it:

    # ~/.fibsem/device-server.yaml
    info: {manufacturer: Odemis}
    server: {host: 127.0.0.1, port: 8765}
    hardware:
      devices:
        - {name: fm, type: fm, mount_transform: none}

    python -m fibsem.server.devices --config ~/.fibsem/device-server.yaml --pair

    from fibsem.drivers.remote.devices import connect_remote_beams   # terminal 2
    beams = connect_remote_beams("127.0.0.1", 8765)
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import threading
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import uvicorn
from fastapi import (
    APIRouter,
    Body,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.encoders import jsonable_encoder
from fastapi.responses import Response

from fibsem.ai.agent.calls import current_call
from fibsem.cancellation import NothingToStop, OperationCancelledError
from fibsem.codec import decode, encode
from fibsem.devices.core import (
    Device,
    DeviceFault,
    ParameterMetadata,
    ParameterReadOnly,
    ParameterUnavailable,
    ResourceBusy,
)
from fibsem.devices.entries import (
    build_device_entries,
    disconnect_devices,
    resolve_device_entries,
)
from fibsem.devices.fm import FM
from fibsem.devices.runs import RUNNING, Run
from fibsem.devices.sample_loader import GridExchangeError
from fibsem.devices.wire import (
    FRAME_METADATA_HEADER,
    FRAMES_MEDIA_TYPE,
    NPY_MEDIA_TYPE,
    Frame,
    command_kwargs,
    decode_argument,
    decode_kwargs,
    frames_to_bytes,
    to_wire,
    type_hints,
    type_path,
)
from fibsem.drivers.registry import REMOTE_DRIVER
from fibsem.fm.microscope import FluorescenceMicroscope
from fibsem.schema import arguments_schema
from fibsem.server import operator_token
from fibsem.server.auth import refusal_error, require_operator
from fibsem.server.device_auth import DeviceAuth
from fibsem.server.images import (
    TIFF_MEDIA_TYPE,
    image_answer,
    is_image_list,
    is_image_result,
    tiff_bytes,
)
from fibsem.server.operator_token import read_device_token
from fibsem.server.pairing_routes import build_pairing_router
from fibsem.server.responses import (
    CommandDescription,
    DeviceDescription,
    DeviceHealth,
    Health,
    MetadataAnswer,
    ParameterDescription,
    QueryDescription,
    ResultAnswer,
    RunAnswer,
    RunEnvelope,
    ValueAnswer,
    Values,
    answer,
)
from fibsem.server.uvicorn_server import Server
from fibsem.structures.image import FibsemImage
from fibsem.structures.params import Limits
from fibsem.structures.system import read_device_entries

# Before uvicorn 0.54 its default websockets implementation imports websockets.legacy,
# which websockets 14+ deprecates on import; under the test suite's warnings-as-errors
# that kills the server thread. The sans-I/O one serves /events without it.
WS_PROTOCOL = "websockets-sansio"

"""A command that returns an array (an image) answers with ``np.save`` bytes."""

# An error keeps its type across the wire; anything else is a plain failure. These
# are refusals and outcomes, not faults: each is logged as one line, no traceback.
ERROR_STATUS = {
    ParameterUnavailable: 404,
    ParameterReadOnly: 409,
    OperationCancelledError: 409,  # stopped: a sweep, a mill
    ResourceBusy: 409,  # held by someone else: a scanner live on the other beam
    NothingToStop: 409,  # a stop with nothing running
    GridExchangeError: 409,  # a grid not in the inventory, or the loader refused
    TypeError: 422,
    ValueError: 422,  # an argument that does not decode names its parameter
    NotImplementedError: 501,  # a command this driver has no hook for (a needle's stop)
    DeviceFault: 502,  # the hardware or its controller refused
}


def refusal(error: Exception, what: str) -> HTTPException:
    """The answer to an expected error (one of ``ERROR_STATUS``), in the shape every
    App Server refusal has: ``error_type`` and ``message``, with ``parameter`` for
    an argument that did not decode and ``status: cancelled`` for a stop.
    Logged as one line, no traceback."""
    status = next(s for t, s in ERROR_STATUS.items() if isinstance(error, t))
    detail = {"error_type": type(error).__name__, "message": str(error)}
    if isinstance(error, OperationCancelledError):
        detail["status"] = "cancelled"
    parameter = getattr(error, "parameter", None)
    if parameter is not None:
        detail["parameter"] = parameter
    device = getattr(error, "device", None) if isinstance(error, DeviceFault) else None
    if device is not None:
        detail["device"] = device
    if isinstance(error, ResourceBusy):
        detail.update(resource=error.resource, holder=error.holder)
    logging.info(f"{what}: {status} {detail['error_type']}: {error}")
    return refusal_error(status, **detail)


def unexpected(error: Exception, what: str) -> HTTPException:
    """The answer to any other error: 500, in the same shape as a refusal, so a
    client reads every failure one way. The traceback goes to the server's log,
    not the answer."""
    logging.error(f"{what}: {type(error).__name__}: {error}", exc_info=error)
    return refusal_error(500, type(error).__name__, str(error))


def answer_for_error(error: Exception, what: str) -> HTTPException:
    """The answer to any error: a refusal for an expected one, else unexpected."""
    if isinstance(error, tuple(ERROR_STATUS)):
        return refusal(error, what)
    return unexpected(error, what)


def _choices_to_wire(choices: Any) -> Any:
    return None if choices is None else [to_wire(choice) for choice in choices]


def _limits(limits: Any) -> Optional[Limits]:
    """``_limits_to_dict``'s output as the limits it describes."""
    return None if limits is None else decode(Limits, limits)


def _facts(device: Device) -> Dict[str, Any]:
    facts = getattr(device, "facts", None)  # a service has none
    if facts is None:
        return {}
    try:
        return jsonable_encoder(facts())
    except Exception as error:  # a description without facts beats none at all
        logging.warning(f"{device.name}: facts: {type(error).__name__}: {error}")
        return {}


# Why the server refuses a write or a command on a device, or None when it
# doesn't: (device, "write" or "command", name, the command's arguments).
Refuse = Callable[[Device, str, str, Mapping[str, Any]], Optional[str]]


def not_allowed(message: str) -> HTTPException:
    return refusal_error(403, "not_allowed", message)


def never_allowed(device: Device, kind: str, name: str, why: str) -> str:
    """Why the agent may not use a write or a command the blocklist names: at
    any level, so the App Server says it before the level does."""
    if kind == "write":
        return (
            f"The agent may not set {device.name}.{name}, at any level: {why}. "
            "Its value stays readable."
        )
    return f"The agent may not run {device.name}.{name}, at any level: {why}."


def describe_device(
    device: Device, refuse: Optional[Refuse] = None
) -> DeviceDescription:
    """A device's parameters, commands, queries and facts. With *refuse*, each
    parameter and command also says whether the agent may use it (``agent``):
    a parameter it may read but never set is ``settable`` and ``agent: false``."""

    def agent(kind: str, name: str, settable: bool = True) -> Optional[bool]:
        if refuse is None:
            return None
        # A parameter nobody can set is only read, and reads are never refused.
        return not settable or refuse(device, kind, name, {}) is None

    return DeviceDescription(
        name=device.name,
        class_=type(device).__name__,
        driver=getattr(device, "driver", None),
        type=getattr(device, "entry_type", None),
        parameters={
            name: ParameterDescription(
                **{
                    **info,
                    "limits": _limits(info["limits"]),
                    "choices": _choices_to_wire(info["choices"]),
                },
                type_path=_type_path(device, name),
                agent=agent("write", name, info["settable"]),
            )
            for name, info in device.describe().items()
        },
        commands={
            name: CommandDescription(
                signature=info.signature,
                arguments=_arguments(device, name),
                returns=_returns(device, name),
                available=info.available,
                agent=agent("command", name),
                long=info.long,
            )
            for name, info in device.commands.items()
        },
        queries={
            name: QueryDescription(
                signature=info.signature,
                arguments=_arguments(device, name),
                returns=_returns(device, name),
                available=info.available,
            )
            for name, info in getattr(device, "queries", {}).items()
        },
        facts=_facts(device),
    )


def _arguments(device: Device, name: str) -> Dict[str, Any]:
    """A command's or query's arguments as JSON Schema (``fibsem.schema``): what an
    agent sends, with the units, SI limits and choices the GUI reads."""
    try:
        return arguments_schema(getattr(type(device), name))
    except Exception as error:  # a schema is a help, never a reason to fail a listing
        logging.warning(f"{device.name}.{name}: no argument schema: {error!r}")
        return {}


def _type_path(device: Device, parameter: str) -> Optional[str]:
    """A parameter's type as its path (`type_path`), so a client reads its value
    as that type."""
    try:
        return type_path(device.declared_parameters()[parameter].type)
    except Exception as error:  # a type is a help, never a reason to fail a listing
        logging.warning(f"{device.name}.{parameter}: no type path: {error!r}")
        return None


def _returns(device: Device, name: str) -> Optional[str]:
    """What a command or query returns, as its type's path (`type_path`), so a
    client reads its result as that type. None when it declares nothing."""
    hint = type_hints(getattr(type(device), name)).get("return")
    try:
        return None if hint is None else type_path(hint)
    except Exception as error:  # a type is a help, never a reason to fail a listing
        logging.warning(f"{device.name}.{name}: no return type: {error!r}")
        return None


def metadata_payload(metadata: ParameterMetadata) -> MetadataAnswer:
    return MetadataAnswer(
        limits=metadata.limits,
        choices=_choices_to_wire(metadata.choices),
        settable=metadata.settable,
        native_max=metadata.native_max,
        native_unit=metadata.native_unit,
    )


def device_health(device: Device) -> DeviceHealth:
    """A driver that can tell whether its hardware answers defines ``check_health()``,
    returning None when fine or a reason when not. Without one, a device is as healthy
    as the server serving it."""
    check = getattr(device, "check_health", None)
    if check is None:
        return DeviceHealth(ok=True)
    try:
        problem = check()
    except Exception as error:
        problem = f"{type(error).__name__}: {error}"
    return DeviceHealth(ok=problem is None, detail=problem)


# How many events may wait for one client before its stream is closed.
STREAM_BACKLOG = 1000


async def stream_events(
    websocket: WebSocket,
    sources: Mapping[str, Mapping[str, Any]],
    allowed: Callable[[], bool] = lambda: True,
    backlog: int = STREAM_BACKLOG,
) -> None:
    """Send *websocket* the signals of every device and service in *sources* until
    it closes, each as ``{noun: name, "parameter", "kind", "value"}``.

    *sources* maps a noun (``"device"``, ``"service"``) to its map by name, read as
    the stream opens. Kinds ``changed`` and ``metadata`` are a parameter's;
    ``signal`` is one of the source's ``remote_signals``, named in ``parameter``,
    its value the signal's argument (a list when it has several). Signals are
    emitted on any thread. The stream closes once *allowed* answers False, or
    once more than *backlog* events wait for a client that isn't reading.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=backlog)
    behind = asyncio.Event()
    connected: List[Tuple[Any, Callable[..., None]]] = []

    def offer(event: Dict[str, Any]) -> None:
        if queue.full():
            behind.set()
        else:
            queue.put_nowait(event)

    def follow(noun, source, signal, kind, as_wire, name=None) -> None:
        def publish(parameter: str, *args: Any) -> None:
            try:
                event = {
                    noun: source.name,
                    "parameter": parameter,
                    "kind": kind,
                    "value": jsonable_encoder(as_wire(*args)),
                }
                loop.call_soon_threadsafe(offer, event)
            except Exception:  # noqa: BLE001 - the emitter must not pay for it
                logging.debug(
                    f"could not send {source.name}.{parameter}", exc_info=True
                )

        # A parameter's signals carry its name; a remote signal is named here.
        callback = publish if name is None else partial(publish, name)
        signal.connect(callback)
        connected.append((signal, callback))

    for noun, by_name in sources.items():
        for source in list(by_name.values()):
            follow(noun, source, source.changed, "changed", to_wire)
            follow(
                noun,
                source,
                source.metadata_changed,
                "metadata",
                lambda metadata: encode(metadata_payload(metadata)),
            )
            for name in getattr(source, "remote_signals", ()):
                follow(
                    noun, source, getattr(source, name), "signal", _signal_value, name
                )
    await websocket.accept()
    closed = asyncio.ensure_future(_until_closed(websocket))
    # A send to a client that isn't reading never finishes, so falling behind
    # interrupts it.
    stop = {closed, asyncio.ensure_future(behind.wait())}
    try:
        while allowed() and not behind.is_set():
            event = asyncio.ensure_future(queue.get())
            await asyncio.wait(
                {event, *stop}, timeout=1.0, return_when=asyncio.FIRST_COMPLETED
            )
            if closed.done():
                event.cancel()
                return
            if not event.done():
                event.cancel()
                continue
            sending = asyncio.ensure_future(
                websocket.send_text(json.dumps(event.result()))
            )
            await asyncio.wait({sending, *stop}, return_when=asyncio.FIRST_COMPLETED)
            if closed.done():
                sending.cancel()
                return
            if sending.done():
                sending.result()  # a disconnect raises here
            else:
                sending.cancel()
        # Too far behind: try again later, and read again on reopening. Otherwise
        # a policy violation: refused from now on.
        await websocket.close(code=1013 if behind.is_set() else 1008)
    except WebSocketDisconnect:
        pass
    finally:
        for task in stop:
            task.cancel()
        for signal, callback in connected:
            signal.disconnect(callback)


async def stream_live(
    websocket: WebSocket, beam: Any, allowed: Callable[[], bool] = lambda: True
) -> None:
    """Send *websocket* each frame *beam*'s live view acquires (``live_frame``),
    raw, with its metadata (`live_frame_bytes`), until it closes. Only the latest
    frame waits: one that arrives while the last is sending replaces any waiting,
    so a slow client gets fewer frames, never older ones. It closes once
    *allowed* answers False."""
    loop = asyncio.get_running_loop()
    latest: List[Any] = [None]
    fresh = asyncio.Event()

    def take(image: Any) -> None:
        latest[0] = image
        loop.call_soon_threadsafe(fresh.set)

    beam.live_frame.connect(take)
    await websocket.accept()
    closed = asyncio.ensure_future(_until_closed(websocket))
    try:
        while allowed():
            waiting = asyncio.ensure_future(fresh.wait())
            await asyncio.wait(
                {waiting, closed}, timeout=1.0, return_when=asyncio.FIRST_COMPLETED
            )
            if closed.done():
                waiting.cancel()
                return
            if not waiting.done():
                waiting.cancel()
                continue
            fresh.clear()
            image, latest[0] = latest[0], None
            if image is None:
                continue
            sending = asyncio.ensure_future(
                websocket.send_bytes(live_frame_bytes(image))
            )
            await asyncio.wait({sending, closed}, return_when=asyncio.FIRST_COMPLETED)
            if closed.done():
                sending.cancel()
                return
            sending.result()  # a disconnect raises here
        await websocket.close(code=1008)
    except WebSocketDisconnect:
        pass
    finally:
        closed.cancel()
        beam.live_frame.disconnect(take)


def live_frame_bytes(image: FibsemImage) -> bytes:
    """A live frame as one ``frames_to_bytes`` frame: its data and its metadata."""
    metadata = image.metadata.to_dict() if image.metadata is not None else {}
    return frames_to_bytes([Frame(image.data, metadata)])


def _signal_value(*args: Any) -> Any:
    return to_wire(args[0] if len(args) == 1 else list(args))


async def _until_closed(websocket: WebSocket) -> None:
    """Clients only listen, so anything received is the close."""
    try:
        while True:
            await websocket.receive_text()
    except (WebSocketDisconnect, RuntimeError):
        return


# How long ``GET .../run`` may wait for a run to end: under the sidecar's 120 s
# timeout, so a wait answers before the call gives up.
RUN_WAIT_CAP_S = 60.0

# A command or query with no arguments may be posted with no body.
_NO_BODY = Body(default={})


# What a command answers that is not JSON: a beam's image, a z-stack, an array.
BINARY_ANSWERS = (TIFF_MEDIA_TYPE, FRAMES_MEDIA_TYPE, NPY_MEDIA_TYPE)


def build_device_router(
    devices: Callable[[], Mapping[str, Device]],
    read: Sequence[Any] = (),
    write: Sequence[Any] = (),
    command: Sequence[Any] = (),
    prefix: str = "/devices",
    noun: str = "device",
    on_image: Optional[Callable[[FibsemImage], None]] = None,
    refuse: Optional[Refuse] = None,
    long_commands: bool = False,
    adopt_control: Optional[Callable[[Request], Optional[Callable[[], None]]]] = None,
    paths: Optional[Callable[[Mapping[str, Any]], Dict[str, Any]]] = None,
    listing: Optional[Sequence[Any]] = None,
) -> APIRouter:
    """The ``/devices`` routes over whatever *devices* returns, by name.

    *devices* is called per request, so a map that changes (an FM reconnecting) is
    served as it is now. A service has the device shape, so the App Server serves
    the microscope's services with the same routes under ``prefix="/services"``
    (*noun* names them in a 404). *read*, *write* and *command* are FastAPI
    dependencies for the reads, the parameter writes and the commands: the device
    server passes none, the App Server its token, level and hardware lock. A query
    moves nothing, so it takes the *read* dependencies, not the hardware lock.
    *on_image* is given each image a command acquired, before it is answered: the
    App Server shows it in the app's views. *refuse* says why a write
    or a command is never allowed here (403 ``not_allowed``), and the listings
    mark those (``agent: false``): the App Server's blocklist. *paths* reads a
    command's or query's relative paths where they belong (the App Server: in the
    open experiment), before *refuse* checks them. *listing*, when given,
    gates the listing of every device in place of *read*.

    With *long_commands*, a command marked ``long`` answers at once with its run
    and runs on, and ``GET {prefix}/{name}/run`` reads the service's latest run
    (`fibsem.devices.runs`), whoever started it: this route, a task or the
    operator. *adopt_control* keeps the control of the microscope the request
    took until the run ends. Without, every command answers when it ends: the device
    server, whose client (the remote driver) waits for it.
    """
    router = APIRouter()

    def lookup(name: str) -> Device:
        try:
            return devices()[name]
        except KeyError:
            raise refusal_error(404, "not_found", f"no {noun} '{name}'") from None

    def parameter_of(device: str, name: str) -> Any:
        d = lookup(device)
        if name not in d.declared_parameters():
            raise refusal_error(
                404, "not_found", f"'{device}' declares no parameter '{name}'"
            )
        return getattr(d, name)  # ParameterUnavailable when this backend lacks it

    def check(d: Device, kind: str, name: str, kwargs: Mapping[str, Any]) -> None:
        why = refuse(d, kind, name, kwargs) if refuse is not None else None
        if why is None:
            return
        if refuse(d, kind, name, {}) is None:
            # Refused for its arguments (a path outside the experiment), not itself.
            raise not_allowed(f"{d.name}.{name} refused: {why}.")
        raise not_allowed(never_allowed(d, kind, name, why))

    def run(call: Any, what: str) -> Any:
        try:
            return call()
        except HTTPException:
            raise
        except Exception as error:
            raise answer_for_error(error, what) from None

    @router.get(
        prefix,
        dependencies=list(read if listing is None else listing),
        response_model=Dict[str, DeviceDescription],
    )
    def list_devices() -> Response:
        return answer(
            {name: describe_device(d, refuse) for name, d in devices().items()},
            Dict[str, DeviceDescription],
        )

    @router.get(
        prefix + "/{device}",
        dependencies=list(read),
        response_model=DeviceDescription,
    )
    def get_device(device: str) -> Response:
        return answer(describe_device(lookup(device), refuse))

    if long_commands:
        # Before /{parameter}, which would otherwise take "run" as a name.
        @router.get(
            prefix + "/{device}/run",
            dependencies=list(read),
            response_model=RunEnvelope,
        )
        def read_run(
            device: str, run_id: Optional[str] = None, wait: float = 0.0
        ) -> Response:
            """The latest run of a long command, whoever started it, waiting up
            to *wait* seconds (at most a minute) for it to end; with *run_id*,
            for that run."""
            runs = lookup(device).runs
            found = runs.wait(run_id, min(max(wait, 0.0), RUN_WAIT_CAP_S))
            if found is None:
                raise refusal_error(
                    404, "not_found", f"{noun} {device} has run no long command"
                )
            return answer(RunEnvelope(run_answer(device, found)))

    # Before /{parameter}, which would otherwise take "values" as a name.
    @router.get(
        prefix + "/{device}/values", dependencies=list(read), response_model=Values
    )
    def read_values(device: str) -> Response:
        """Every parameter's value in one call; those this backend lacks are
        named apart, not failed."""
        d = lookup(device)
        values, unavailable = {}, []
        for name in d.declared_parameters():
            try:
                value = getattr(d, name).get_value()
            except ParameterUnavailable:
                unavailable.append(name)
                continue
            except Exception as error:
                raise answer_for_error(error, f"{noun} {device}.{name}") from None
            values[name] = to_wire(value)
        return answer(Values(values, unavailable))

    @router.get(
        prefix + "/{device}/{parameter}",
        dependencies=list(read),
        response_model=ValueAnswer,
    )
    def read_parameter(device: str, parameter: str) -> Response:
        value = run(
            lambda: parameter_of(device, parameter).get_value(),
            f"{noun} {device}.{parameter}",
        )
        return answer(ValueAnswer(to_wire(value)))

    @router.put(
        prefix + "/{device}/{parameter}",
        dependencies=list(write),
        response_model=ValueAnswer,
    )
    def write_parameter(device: str, parameter: str, body: Dict[str, Any]) -> Response:
        what = f"{noun} {device}.{parameter}"
        param = run(lambda: parameter_of(device, parameter), what)
        check(lookup(device), "write", parameter, {})
        value = run(lambda: decode_argument(parameter, param.type, body["value"]), what)
        return answer(ValueAnswer(to_wire(run(lambda: param.set_value(value), what))))

    @router.get(
        prefix + "/{device}/{parameter}/metadata",
        dependencies=list(read),
        response_model=MetadataAnswer,
    )
    def metadata(device: str, parameter: str) -> Response:
        return answer(
            metadata_payload(
                run(
                    lambda: parameter_of(device, parameter).metadata,
                    f"{noun} {device}.{parameter}",
                )
            )
        )

    def shown(result: Any) -> Any:
        """*result*, each image in it given to *on_image* first."""
        if on_image is not None:
            for image in _images_in(result):
                on_image(image)
        return result

    def json_answer(result: Any, preview: bool) -> Any:
        """A command's result as JSON. An image list, or with *preview* any
        image, answers what an agent reads: a summary, its saved path and a
        small JPEG (image_answer). So does an image under ``image`` in a dict
        (a tiled overview's answer)."""
        if (preview and is_image_result(result)) or is_image_list(result):
            return image_answer(result)
        if isinstance(result, dict) and isinstance(result.get("image"), FibsemImage):
            return json_answer(
                {**result, "image": image_answer(result["image"])}, preview
            )
        # A structure with to_dict answers that (an autofocus result, its summary).
        return jsonable_encoder(to_wire(result))

    @router.post(
        prefix + "/{device}/commands/{command}",
        dependencies=list(command),
        response_model=ResultAnswer,
        responses={200: {"content": {media: {} for media in BINARY_ANSWERS}}},
    )
    def call(
        request: Request, device: str, command: str, body: Dict[str, Any] = _NO_BODY
    ) -> Any:
        d = lookup(device)
        if command not in d.commands:
            raise refusal_error(
                404, "not_found", f"'{device}' has no command '{command}'"
            )
        method = getattr(d, command)
        what = f"{noun} {device}.{command}"
        kwargs = run(lambda: command_kwargs(method, body), what)
        if paths is not None:
            kwargs = paths(kwargs)
        check(d, "command", command, kwargs)
        if long_commands and d.commands[command].long:
            release = adopt_control(request) if adopt_control is not None else None
            agent_call = current_call()
            labels = {}
            if agent_call is not None:
                # What the agent's call was, for the run's events to say.
                labels = {
                    key: value
                    for key, value in (
                        ("action", agent_call.action),
                        ("level", agent_call.level),
                        ("evidence", agent_call.evidence),
                    )
                    if value
                }
            started = d.runs.start(
                command,
                lambda: shown(method(**kwargs)),
                before_end=release,
                labels=labels,
            )
            if agent_call is not None:
                agent_call.run_id = started.run_id  # its run's events say how it went
            return answer(ResultAnswer(started.to_dict()))
        result = shown(run(lambda: method(**kwargs), what))
        headers = {}
        # "answer": "preview" asks for what an agent reads. Otherwise the data
        # itself, for programs.
        preview = body.get("answer") == "preview"
        if (preview and is_image_result(result)) or is_image_list(result):
            return answer(ResultAnswer(json_answer(result, preview)))
        if isinstance(result, FibsemImage):  # a beam's image: TIFF
            return Response(tiff_bytes(result), media_type=TIFF_MEDIA_TYPE)
        if (
            isinstance(result, list)
            and result
            and all(isinstance(item, Frame) for item in result)
        ):  # several images, a z-stack: one archive
            return Response(frames_to_bytes(result), media_type=FRAMES_MEDIA_TYPE)
        if isinstance(result, Frame):  # the image, its metadata in a header
            headers[FRAME_METADATA_HEADER] = json.dumps(
                jsonable_encoder(result.metadata)
            )
            result = result.data
        if isinstance(result, np.ndarray):  # an image: binary, not JSON
            buffer = io.BytesIO()
            np.save(buffer, result, allow_pickle=False)
            return Response(
                buffer.getvalue(), media_type=NPY_MEDIA_TYPE, headers=headers
            )
        return answer(ResultAnswer(json_answer(result, preview)))

    # Each ended run's answer, made once (its images' previews, and its
    # error's answer, which logs it): the latest per device.
    answers: Dict[str, Tuple[str, RunAnswer]] = {}
    answers_lock = threading.Lock()

    def run_answer(device: str, found: Run) -> RunAnswer:
        """*found* as the route answers it: its result as a command's answer
        would carry it, and its error as a refusal does."""
        if found.state == RUNNING:
            return RunAnswer(**found.to_dict())
        with answers_lock:
            made = answers.get(device)
            if made is not None and made[0] == found.run_id:
                return made[1]
            ended = RunAnswer(**found.to_dict())
            ended.result = json_answer(found.result, preview=True)
            if found.exception is not None:
                what = f"{noun} {device}.{found.command}"
                refused = answer_for_error(found.exception, what)
                ended.error, ended.status_code = refused.detail, refused.status_code
            answers[device] = (found.run_id, ended)
            return ended

    @router.post(
        prefix + "/{device}/queries/{query}",
        dependencies=list(read),
        response_model=ResultAnswer,
    )
    def ask(device: str, query: str, body: Dict[str, Any] = _NO_BODY) -> Response:
        d = lookup(device)
        if query not in getattr(d, "queries", {}):
            raise refusal_error(404, "not_found", f"'{device}' has no query '{query}'")
        method = getattr(d, query)
        what = f"{noun} {device}.{query}"
        kwargs = run(lambda: decode_kwargs(method, body.get("kwargs", {})), what)
        if paths is not None:
            kwargs = paths(kwargs)
        check(d, "query", query, kwargs)
        result = run(lambda: method(**kwargs), what)
        # A query is asked by an agent: an image (a pattern preview) comes back
        # as one it reads.
        return answer(ResultAnswer(json_answer(result, preview=True)))

    return router


def _images_in(result: Any) -> List[FibsemImage]:
    """Every FibsemImage a command returned: one, or the ones in its lists."""
    if isinstance(result, FibsemImage):
        return [result]
    if isinstance(result, (list, tuple)):
        return [image for item in result for image in _images_in(item)]
    return []


def live_source(devices: Mapping[str, Device], name: str) -> Any:
    """The device *name* names, if it has a live view; else 404."""
    device = devices.get(name)
    if device is None or not hasattr(device, "live_frame"):
        raise refusal_error(404, "not_found", f"no beam '{name}' with a live view")
    return device


def build_device_app(
    devices: Iterable[Device], auth: Optional[DeviceAuth] = None
) -> FastAPI:
    """The server's routes over these devices, looked up by ``device.name``,
    behind *auth*: this install's token and paired computers by default."""
    by_name: Dict[str, Device] = {device.name: device for device in devices}
    auth = auth if auth is not None else DeviceAuth.from_disk()
    app = FastAPI(title="fibsem devices")
    app.state.auth = auth
    token = Depends(require_operator())
    app.include_router(build_pairing_router(auth))

    @app.get("/health", dependencies=[token], response_model=Health)
    def health() -> Response:
        """Up, and for each device whether its driver can reach the hardware."""
        devices = {name: device_health(d) for name, d in by_name.items()}
        return answer(Health(all(d.ok for d in devices.values()), devices))

    app.include_router(build_device_router(lambda: by_name), dependencies=[token])

    @app.websocket("/events", dependencies=[token])
    async def events(websocket: WebSocket) -> None:
        await stream_events(websocket, {"device": by_name})

    @app.websocket("/devices/{device}/live", dependencies=[token])
    async def live(websocket: WebSocket, device: str) -> None:
        await stream_live(websocket, live_source(by_name, device))

    return app


class DeviceServer:
    """Runs the app with uvicorn on a background thread, for tests and scripts."""

    def __init__(
        self,
        devices: Iterable[Device],
        host: str = "127.0.0.1",
        port: int = 0,
        auth: Optional[DeviceAuth] = None,
    ):
        config = uvicorn.Config(
            build_device_app(devices, auth),
            host=host,
            port=port,
            log_level="warning",
            timeout_graceful_shutdown=1,
            ws=WS_PROTOCOL,
        )
        self.app = config.app
        self._server = Server(config)
        self._thread: Optional[threading.Thread] = None

    @property
    def port(self) -> int:
        """The port actually bound (useful with ``port=0``)."""
        return self._server.servers[0].sockets[0].getsockname()[1]

    def start(self) -> "DeviceServer":
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        while not self._server.started:
            if not self._thread.is_alive():
                raise RuntimeError("device server failed to start")
            threading.Event().wait(0.01)
        return self

    def stop(self) -> None:
        self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)


DEMO_PARTS = ("beams", "stage", "chamber", "manipulator")
"""What of the Demo microscope a device server can serve."""


def demo_devices(parts: Sequence[str] = ("beams",)) -> List[Device]:
    """The Demo microscope's beams, standing in for real hardware, and its stage,
    chamber and manipulator when *parts* names them."""
    from fibsem import utils

    microscope, _ = utils.setup_session(manufacturer="Demo")
    devices: List[Device] = []
    for part in parts:
        if part == "beams":
            devices += microscope.beams.values()
        else:
            devices.append(microscope.devices[part])
    return devices


SERVED_FM_DRIVERS = {"fm": "Demo", "odemis-fm": "Odemis"}
"""The FM ``--serve`` names, and the driver whose fm entry each builds."""

HOST_FM_KEYS = (
    "address",
    "port",
    "origin",
    "range",
    "available_orientations",
    "pose",
    "focus_position",
    "limit_position",
    "camera_tilt",
)
"""fm entry keys the host's remote entry owns, refused in a Device Server's file."""


def served_fm_settings(
    serve: str, mount_transform: Optional[str] = None
) -> Dict[str, Any]:
    """The one-entry file that ``--serve fm`` or ``--serve odemis-fm`` stands for."""
    entry: Dict[str, Any] = {
        "name": "fm",
        "type": "fm",
        "driver": SERVED_FM_DRIVERS[serve],
        "required": True,
    }
    if mount_transform is not None:
        entry["mount_transform"] = mount_transform
    return {"hardware": {"devices": [entry]}}


def read_server_settings(path: str) -> Dict[str, Any]:
    """A Device Server's file: a microscope configuration's ``info`` and
    ``hardware.devices``, and a ``server:`` block with its ``host`` and ``port``."""
    import yaml

    from fibsem.codec.upgrades.configuration import upgrade

    with open(path) as f:
        return upgrade(yaml.safe_load(f) or {})


def _refuse_unservable(entries: Iterable[Any]) -> None:
    """Refuse an entry a Device Server can't serve, before anything is built."""
    for entry in entries:
        if entry.driver == REMOTE_DRIVER:
            raise ValueError(
                f"'{entry.name}' is a remote entry: a Device Server serves the devices "
                "it builds itself. Put remote entries in the host's configuration."
            )
        host_keys = [key for key in HOST_FM_KEYS if key in entry.options]
        if entry.type == "fm" and host_keys:
            raise ValueError(
                f"'{entry.name}' sets {', '.join(host_keys)}, which belong on the "
                "host's remote fm entry, not the Device Server's."
            )


def configured_devices(
    source: Union[str, Mapping[str, Any]],
) -> Tuple[Dict[str, Device], List[Callable[[], None]]]:
    """The devices a configuration's ``hardware.devices`` lists, each built by its own
    driver with no microscope (``BuildContext.microscope`` is None), as a unit's PC
    serves them; and what their builders registered to close
    (`fibsem.devices.entries.disconnect_devices` closes both).

    *source* is a file, or its settings already read. It has the shape of a
    microscope configuration, so entries copy across. An entry names its ``driver``,
    unless ``info.manufacturer`` names one for all of them. An entry that can't be
    built is left out with a warning, unless it is ``required: true``; one the
    server can't serve is refused (`_refuse_unservable`). An ``fm`` entry is served
    as the FM's parts and group, which the host's remote FM mirrors. Roles are not
    bound: each device is served on its own.
    """
    settings = read_server_settings(source) if isinstance(source, str) else source
    configured = read_device_entries(settings)
    _refuse_unservable(configured.values())
    manufacturer = (settings.get("info") or {}).get("manufacturer")
    resolved = resolve_device_entries([], configured, manufacturer)
    cleanups: List[Callable[[], None]] = []
    built: Dict[str, Device] = {}
    for name, device in build_device_entries(resolved, None, cleanups=cleanups).items():
        if isinstance(device, FluorescenceMicroscope):
            built.update(device.devices)
        else:
            built[name] = device
    return built, cleanups


def switch_off_fms(devices: Iterable[Device]) -> None:
    """Every FM group among *devices* idle with its light off (`FM.switch_off`), as
    the server binds them and as it stops. A failure is logged, not raised: the
    server still starts, and still closes the rest."""
    for device in devices:
        if isinstance(device, FM):
            try:
                device.switch_off()
            except Exception as e:
                logging.error(f"{device.name}: could not switch the light off: {e}")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Serve devices over HTTP.")
    parser.add_argument(
        "--host", default=None, help="default the file's server.host, else 127.0.0.1"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="default the file's server.port, else 8765",
    )
    parser.add_argument(
        "--serve",
        nargs="+",
        choices=(*DEMO_PARTS, *SERVED_FM_DRIVERS),
        default=None,
        help="beams, stage, chamber, manipulator: the Demo microscope's; fm: a "
        "simulated FM's parts; odemis-fm: the FM of the METEOR this computer runs "
        "odemis for (each the one fm entry of a file). Default beams, unless "
        "--config is given",
    )
    parser.add_argument(
        "--config",
        help="a Device Server file (~/.fibsem/device-server.yaml): its "
        "hardware.devices entries, each built and served with its own driver, and "
        "its server host and port",
    )
    parser.add_argument(
        "--mount-transform",
        choices=("none", "flip-x", "flip-y", "flip-xy"),
        default=None,
        help="fm, odemis-fm: the flip that puts the camera's frames into the "
        "stage's axes; in a file, the fm entry's mount_transform",
    )
    parser.add_argument(
        "--token-file",
        default=None,
        help="where this install's token is kept (default "
        "~/.fibsem/device-server-token.json); made on first start",
    )
    parser.add_argument(
        "--pair",
        action="store_true",
        help="show a pairing code for another computer (5 minutes, works once)",
    )
    args = parser.parse_args(argv)
    if args.serve is None:
        args.serve = [] if args.config else ["beams"]
    if args.mount_transform is not None:
        logging.warning(
            "--mount-transform still works, but belongs in the Device Server's file: "
            f"mount_transform: {args.mount_transform} on its fm entry."
        )
    served: List[Device] = []
    built: Dict[str, Device] = {}
    cleanups: List[Callable[[], None]] = []
    sources: List[Union[str, Mapping[str, Any]]] = []
    settings: Mapping[str, Any] = {}
    if args.config:
        try:
            settings = read_server_settings(args.config)
        except Exception as e:
            logging.error(f"{args.config}: {e}")
            raise SystemExit(1) from e
        sources.append(settings)
    sources += [
        served_fm_settings(serve, args.mount_transform)
        for serve in SERVED_FM_DRIVERS
        if serve in args.serve
    ]
    for source in sources:
        try:
            more, closes = configured_devices(source)
        except Exception as e:
            logging.error(f"{args.config if source is settings else 'FM'}: {e}")
            disconnect_devices(built, cleanups)
            raise SystemExit(1) from e
        built.update(more)
        cleanups += closes
    served += built.values()
    demo_parts = [part for part in DEMO_PARTS if part in args.serve]
    if demo_parts:
        served += demo_devices(demo_parts)
    if not served:
        logging.error("Nothing to serve: no device was built.")
        raise SystemExit(1)
    devices: Mapping[str, Device] = {d.name: d for d in served}
    switch_off_fms(devices.values())
    server = settings.get("server") or {}
    host = args.host or server.get("host") or "127.0.0.1"
    port = args.port or server.get("port") or 8765
    token_file = (
        Path(args.token_file) if args.token_file else operator_token.DEVICE_TOKEN_FILE
    )
    if read_device_token(token_file) is None:
        print(f"Made this Device Server's token, kept in {token_file}.")
    auth = DeviceAuth.from_disk(token_file)
    if args.pair:
        code = auth.pairing.open()
        print(
            f"Pairing code: {code} (5 minutes, works once). On the other computer "
            f"run: fibsem-cli pair-device <this computer's address>:{port} {code}"
        )
    logging.info(f"serving {sorted(devices)} on http://{host}:{port}")
    try:
        uvicorn.run(
            build_device_app(devices.values(), auth),
            host=host,
            port=port,
            ws=WS_PROTOCOL,
            # Ctrl-C or SIGTERM: stop waiting on a running acquisition, so the light
            # goes off below rather than when it ends.
            timeout_graceful_shutdown=5,
        )
    finally:
        switch_off_fms(devices.values())
        disconnect_devices(built, cleanups)


if __name__ == "__main__":
    main()
