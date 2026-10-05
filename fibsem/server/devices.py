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
                                               or np.save bytes for an image, with a
                                               frame's metadata as JSON in the
                                               X-Frame-Metadata header
    WS   /events                               {"device", "parameter", "kind", "value"},
                                               with pings as a heartbeat

The coordinator side is ``fibsem.devices.drivers.remote``. A write runs the device's
``set_value``, so the server checks every value itself, whatever the client did.
Errors keep their meaning across the wire: the client raises the same exception
types a local device would.

Not yet: authentication (``fibsem.server`` has bearer tokens; this
would be mounted there), the one-commander lease.

Try it on one computer:

    python -m fibsem.server.devices --port 8765        # terminal 1: Demo beams
    python -m fibsem.server.devices --serve fm         # or a simulated FM's parts

On a METEOR's Linux PC, serving its real FM to the PC that drives the beams (see
INSTALLATION.md, "Delmic METEOR"):

    python -m fibsem.server.devices --serve odemis-fm --host 0.0.0.0

    from fibsem.devices.drivers.remote import connect_remote_beams   # terminal 2
    beams = connect_remote_beams("127.0.0.1", 8765)
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import threading
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from fastapi.responses import Response

from fibsem.devices.core import (
    Device,
    ParameterMetadata,
    ParameterReadOnly,
    ParameterUnavailable,
    _limits_to_dict,
)
from fibsem.devices.wire import (
    FRAME_METADATA_HEADER,
    FRAMES_MEDIA_TYPE,
    NPY_MEDIA_TYPE,
    Frame,
    decode_kwargs,
    frames_to_bytes,
    from_wire,
    to_wire,
)

"""A command that returns an array (an image) answers with ``np.save`` bytes."""

# An error keeps its type across the wire; anything else is a plain failure.
ERROR_STATUS = {
    ParameterUnavailable: 404,
    ParameterReadOnly: 409,
    TypeError: 422,
    ValueError: 422,
}


def _choices_to_wire(choices: Any) -> Any:
    return None if choices is None else [to_wire(choice) for choice in choices]


def describe_device(device: Device) -> Dict[str, Any]:
    return {
        "name": device.name,
        "class": type(device).__name__,
        "parameters": {
            name: {**info, "choices": _choices_to_wire(info["choices"])}
            for name, info in device.describe().items()
        },
        "commands": {
            name: {"signature": info.signature, "available": info.available}
            for name, info in device.commands.items()
        },
    }


def metadata_payload(metadata: ParameterMetadata) -> Dict[str, Any]:
    return {
        "limits": _limits_to_dict(metadata.limits),
        "choices": _choices_to_wire(metadata.choices),
        "settable": metadata.settable,
        "native_max": metadata.native_max,
        "native_unit": metadata.native_unit,
    }


def device_health(device: Device) -> Dict[str, Any]:
    """A driver that can tell whether its hardware answers defines ``check_health()``,
    returning None when fine or a reason when not. Without one, a device is as healthy
    as the server serving it."""
    check = getattr(device, "check_health", None)
    if check is None:
        return {"ok": True, "detail": None}
    try:
        problem = check()
    except Exception as error:
        problem = f"{type(error).__name__}: {error}"
    return {"ok": problem is None, "detail": problem}


class _EventHub:
    """Fans device signals, emitted on any thread, out to every open websocket."""

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._queues: Set[asyncio.Queue] = set()

    def attach(self, devices: Iterable[Device]) -> None:
        for device in devices:
            device.changed.connect(
                lambda name, value, d=device: self.publish(d, name, "changed", value)
            )
            device.metadata_changed.connect(
                lambda name, meta, d=device: self.publish(
                    d, name, "metadata", metadata_payload(meta)
                )
            )

    def publish(self, device: Device, parameter: str, kind: str, value: Any) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        event = {
            "device": device.name,
            "parameter": parameter,
            "kind": kind,
            "value": jsonable_encoder(to_wire(value)),
        }
        for queue in list(self._queues):
            loop.call_soon_threadsafe(queue.put_nowait, event)

    async def stream(self, websocket: WebSocket) -> None:
        self._loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        self._queues.add(queue)
        await websocket.accept()
        closed = asyncio.ensure_future(self._until_closed(websocket))
        try:
            while True:
                event = asyncio.ensure_future(queue.get())
                await asyncio.wait({event, closed}, return_when=asyncio.FIRST_COMPLETED)
                if closed.done():
                    event.cancel()
                    return
                await websocket.send_text(json.dumps(event.result()))
        except WebSocketDisconnect:
            pass
        finally:
            closed.cancel()
            self._queues.discard(queue)

    @staticmethod
    async def _until_closed(websocket: WebSocket) -> None:
        """Clients only listen, so anything received is the close."""
        try:
            while True:
                await websocket.receive_text()
        except (WebSocketDisconnect, RuntimeError):
            return


def build_device_app(devices: Iterable[Device]) -> FastAPI:
    """The server's routes over these devices, looked up by ``device.name``."""
    by_name: Dict[str, Device] = {device.name: device for device in devices}
    hub = _EventHub()
    hub.attach(by_name.values())
    app = FastAPI(title="fibsem devices")

    def lookup(name: str) -> Device:
        try:
            return by_name[name]
        except KeyError:
            raise HTTPException(404, f"no device '{name}'") from None

    def parameter_of(device: str, name: str) -> Any:
        d = lookup(device)
        if name not in d.declared_parameters():
            raise HTTPException(404, f"'{device}' declares no parameter '{name}'")
        return getattr(d, name)  # ParameterUnavailable when this backend lacks it

    def run(call: Any) -> Any:
        try:
            return call()
        except HTTPException:
            raise
        except tuple(ERROR_STATUS) as error:
            status = next(s for t, s in ERROR_STATUS.items() if isinstance(error, t))
            raise HTTPException(
                status, {"error": type(error).__name__, "detail": str(error)}
            ) from None

    @app.get("/health")
    def health() -> Dict[str, Any]:
        """Up, and for each device whether its driver can reach the hardware."""
        devices = {name: device_health(d) for name, d in by_name.items()}
        return {"ok": all(d["ok"] for d in devices.values()), "devices": devices}

    @app.get("/devices")
    def list_devices() -> Dict[str, Any]:
        return {name: describe_device(d) for name, d in by_name.items()}

    @app.get("/devices/{device}")
    def get_device(device: str) -> Dict[str, Any]:
        return describe_device(lookup(device))

    @app.get("/devices/{device}/{parameter}")
    def read(device: str, parameter: str) -> Dict[str, Any]:
        value = run(lambda: parameter_of(device, parameter).get_value())
        return {"value": to_wire(value)}

    @app.put("/devices/{device}/{parameter}")
    def write(device: str, parameter: str, body: Dict[str, Any]) -> Dict[str, Any]:
        param = run(lambda: parameter_of(device, parameter))
        value = from_wire(param.type, body["value"])
        return {"value": to_wire(run(lambda: param.set_value(value)))}

    @app.get("/devices/{device}/{parameter}/metadata")
    def metadata(device: str, parameter: str) -> Dict[str, Any]:
        return metadata_payload(run(lambda: parameter_of(device, parameter).metadata))

    @app.post("/devices/{device}/commands/{command}")
    def call(device: str, command: str, body: Dict[str, Any]) -> Any:
        d = lookup(device)
        if command not in d.commands:
            raise HTTPException(404, f"'{device}' has no command '{command}'")
        method = getattr(d, command)
        kwargs = decode_kwargs(method, body.get("kwargs", {}))
        result = run(lambda: method(**kwargs))
        headers = {}
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
        return {"result": jsonable_encoder(result)}

    @app.websocket("/events")
    async def events(websocket: WebSocket) -> None:
        await hub.stream(websocket)

    return app


class DeviceServer:
    """Runs the app with uvicorn on a background thread, for tests and scripts."""

    def __init__(
        self, devices: Iterable[Device], host: str = "127.0.0.1", port: int = 0
    ):
        config = uvicorn.Config(
            build_device_app(devices),
            host=host,
            port=port,
            log_level="warning",
            timeout_graceful_shutdown=1,
        )
        self.app = config.app
        self._server = uvicorn.Server(config)
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


def demo_devices() -> List[Device]:
    """The Demo microscope's beams, standing in for real hardware."""
    from fibsem import utils

    microscope, _ = utils.setup_session(manufacturer="Demo")
    return list(microscope.beams.values())


def demo_fm_devices() -> List[Device]:
    """The simulated FM's parts and group, as a METEOR PC would serve its FM."""
    from fibsem.devices.drivers.fm import bind_fm_devices
    from fibsem.fm.microscope import FluorescenceMicroscope

    return list(bind_fm_devices(FluorescenceMicroscope()).values())


def odemis_fm_devices(mount_transform: str = "none") -> List[Device]:
    """The METEOR's FM, through the odemis backend on this computer (FIB-1095).

    odemis is reachable only from the computer running it, over unix sockets, which
    is why the FM is served from there rather than driven from the beams' PC.
    ``mount_transform`` is how its camera is mounted, which the camera reports, so
    every client sees the frames the same way round.

    Raises:
        RuntimeError: odemis is not installed here, or its backend did not answer;
            the message says which, and what to check.
    """
    from fibsem.devices.fm import mount_transform_from_name

    transform = mount_transform_from_name(mount_transform)
    try:
        import fibsem.fm.odemis  # noqa: F401
        from fibsem.devices.drivers.odemis_fm import bind_odemis_fm
    except ImportError as e:
        raise RuntimeError(
            f"odemis cannot be imported here ({e}). Serve the odemis FM from the "
            "METEOR PC that runs odemis."
        ) from e
    try:
        return list(bind_odemis_fm(parent=None, mount_transform=transform).values())
    except Exception as e:
        raise RuntimeError(
            f"The odemis backend did not answer ({type(e).__name__}: {e}). Check "
            "that odemis is running (odemis-start) and that this user is in the "
            "'odemis' group."
        ) from e


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Serve devices over HTTP.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--serve",
        nargs="+",
        choices=("beams", "fm", "odemis-fm"),
        default=["beams"],
        help="beams: the Demo microscope's beams; fm: a simulated FM's parts; "
        "odemis-fm: the FM of the METEOR this computer runs odemis for",
    )
    parser.add_argument(
        "--mount-transform",
        choices=("none", "flip-x", "flip-y", "flip-xy"),
        default="none",
        help="odemis-fm: the flip that puts the camera's frames into the stage's "
        "axes, from how it is mounted",
    )
    args = parser.parse_args(argv)
    served: List[Device] = []
    if "beams" in args.serve:
        served += demo_devices()
    if "fm" in args.serve:
        served += demo_fm_devices()
    if "odemis-fm" in args.serve:
        try:
            served += odemis_fm_devices(args.mount_transform)
        except RuntimeError as e:
            logging.error(e)
            raise SystemExit(1) from e
    devices: Mapping[str, Device] = {d.name: d for d in served}
    logging.info(f"serving {sorted(devices)} on http://{args.host}:{args.port}")
    uvicorn.run(build_device_app(devices.values()), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
