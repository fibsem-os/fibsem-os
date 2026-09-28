"""A device server: devices on this computer, reachable from a coordinator on another.

The far side of a remote device (the METEOR PC, say). It wraps any
``fibsem.devices.Device`` and exposes what the device already describes about itself:

    GET  /health                               up, and each device's hardware reachable
    GET  /access                               the scopes this server has armed
    GET  /devices                              every device: parameters and commands
    GET  /devices/{device}                     one device
    GET  /devices/{device}/{parameter}         a live read       -> {"value": ...}
    PUT  /devices/{device}/{parameter}         {"value": ...}    -> {"value": written}
    GET  /devices/{device}/{parameter}/metadata                  -> limits, choices, settable
    POST /devices/{device}/commands/{command}  {"kwargs": {...}} -> {"result": ...},
                                               or np.save bytes for an image
    WS   /events                               {"device", "parameter", "kind", "value"},
                                               with pings as a heartbeat
    POST /pair                                 {"code": ...}     -> {"token": ...}

The coordinator side is ``fibsem.devices.drivers.remote``. A write runs the device's
``set_value``, so the server checks every value itself, whatever the client did.
Errors keep their meaning across the wire: the client raises the same exception
types a local device would.

Every route but ``/pair`` needs the server's bearer token (``fibsem.server.auth``),
the websocket included. A valid token can read; writes and commands also need the
``hardware`` scope, which only ``--arm-hardware`` arms, so a server started without
it is a read-only mirror. The token lives in a file (``fibsem.server.pairing``), made
the first time, so it survives restarts. ``--pair`` opens a pairing window and shows
its code, which a coordinator trades for the token at ``POST /pair``.

Plain HTTP: the token crosses the network in the clear, which suits a direct cable or
a private instrument network, not a shared one. Not yet: the one-commander lease.

Try it on one computer:

    python -m fibsem.server.devices --arm-hardware       # terminal 1: Demo beams
    python -m fibsem.server.devices --serve fm --arm-hardware   # or a simulated FM

    from fibsem.devices.drivers.remote import DeviceClient, connect_remote_beams
    from fibsem.server.pairing import read_token             # terminal 2, same file
    client = DeviceClient("127.0.0.1", 8765, token=read_token())
    beams = connect_remote_beams("127.0.0.1", 8765, client=client)
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import io
import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set

import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from fastapi.responses import Response

from fibsem.devices.core import (
    Device,
    ParameterMetadata,
    ParameterReadOnly,
    ParameterUnavailable,
    _limits_to_dict,
)
from fibsem.devices.wire import from_wire, to_wire
from fibsem.server.auth import AuthConfig, Scope, require_scope
from fibsem.server.pairing import DEFAULT_TOKEN_FILE, Pairing, load_or_create_token

NPY_MEDIA_TYPE = "application/x-npy"
"""A command that returns an array (an image) answers with ``np.save`` bytes."""

# An error keeps its type across the wire; anything else is a plain failure.
ERROR_STATUS = {
    ParameterUnavailable: 404,
    ParameterReadOnly: 409,
    TypeError: 422,
    ValueError: 422,
}


def describe_device(device: Device) -> Dict[str, Any]:
    return {
        "name": device.name,
        "class": type(device).__name__,
        "parameters": device.describe(),
        "commands": {
            name: {"signature": info.signature, "available": info.available}
            for name, info in device.commands.items()
        },
    }


def metadata_payload(metadata: ParameterMetadata) -> Dict[str, Any]:
    return {
        "limits": _limits_to_dict(metadata.limits),
        "choices": list(metadata.choices) if metadata.choices is not None else None,
        "settable": metadata.settable,
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


def _websocket_authorized(websocket: WebSocket, auth: AuthConfig) -> bool:
    header = websocket.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return False
    if not hmac.compare_digest(header[len("Bearer ") :], auth.token):
        return False
    auth.mark_seen()
    return True


def build_device_app(
    devices: Iterable[Device],
    auth: Optional[AuthConfig] = None,
    pairing: Optional[Pairing] = None,
) -> FastAPI:
    """The server's routes over these devices, looked up by ``device.name``.

    Without ``auth`` the server makes a token of its own and arms nothing but
    ``read``: it fails closed, as ``fibsem.server`` does.
    """
    by_name: Dict[str, Device] = {device.name: device for device in devices}
    hub = _EventHub()
    hub.attach(by_name.values())
    app = FastAPI(title="fibsem devices")
    app.state.auth = auth if auth is not None else AuthConfig.generate()
    app.state.pairing = pairing if pairing is not None else Pairing()
    reader = [Depends(require_scope(Scope.READ))]
    commander = [Depends(require_scope(Scope.HARDWARE))]

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

    @app.get("/health", dependencies=reader)
    def health() -> Dict[str, Any]:
        """Up, and for each device whether its driver can reach the hardware."""
        devices = {name: device_health(d) for name, d in by_name.items()}
        return {"ok": all(d["ok"] for d in devices.values()), "devices": devices}

    @app.get("/access", dependencies=reader)
    def access() -> Dict[str, Any]:
        """What this token may do here: ``read``, and whatever the hosting armed."""
        return {"scopes": [s.value for s in app.state.auth.armed_scopes()]}

    @app.post("/pair")
    def pair(body: Dict[str, Any]) -> Dict[str, Any]:
        """The token, for the code shown in the open pairing window. No token needed:
        the code is the proof that someone can see the device computer's screen."""
        if not app.state.pairing.redeem(str(body.get("code", ""))):
            raise HTTPException(
                403,
                {
                    "error_type": "pairing_refused",
                    "message": "Wrong or expired code, or no pairing window open.",
                },
            )
        logging.info("A coordinator paired with this device server")
        return {"token": app.state.auth.token}

    @app.get("/devices", dependencies=reader)
    def list_devices() -> Dict[str, Any]:
        return {name: describe_device(d) for name, d in by_name.items()}

    @app.get("/devices/{device}", dependencies=reader)
    def get_device(device: str) -> Dict[str, Any]:
        return describe_device(lookup(device))

    @app.get("/devices/{device}/{parameter}", dependencies=reader)
    def read(device: str, parameter: str) -> Dict[str, Any]:
        value = run(lambda: parameter_of(device, parameter).get_value())
        return {"value": to_wire(value)}

    @app.put("/devices/{device}/{parameter}", dependencies=commander)
    def write(device: str, parameter: str, body: Dict[str, Any]) -> Dict[str, Any]:
        param = run(lambda: parameter_of(device, parameter))
        value = from_wire(param.type, body["value"])
        return {"value": to_wire(run(lambda: param.set_value(value)))}

    @app.get("/devices/{device}/{parameter}/metadata", dependencies=reader)
    def metadata(device: str, parameter: str) -> Dict[str, Any]:
        return metadata_payload(run(lambda: parameter_of(device, parameter).metadata))

    @app.post("/devices/{device}/commands/{command}", dependencies=commander)
    def call(device: str, command: str, body: Dict[str, Any]) -> Any:
        d = lookup(device)
        if command not in d.commands:
            raise HTTPException(404, f"'{device}' has no command '{command}'")
        result = run(lambda: getattr(d, command)(**body.get("kwargs", {})))
        if isinstance(result, np.ndarray):  # an image: binary, not JSON
            buffer = io.BytesIO()
            np.save(buffer, result, allow_pickle=False)
            return Response(buffer.getvalue(), media_type=NPY_MEDIA_TYPE)
        return {"result": jsonable_encoder(result)}

    @app.websocket("/events")
    async def events(websocket: WebSocket) -> None:
        if not _websocket_authorized(websocket, app.state.auth):
            # Closing before accepting refuses the handshake (HTTP 403).
            await websocket.close(code=1008)
            return
        await hub.stream(websocket)

    return app


class DeviceServer:
    """Runs the app with uvicorn on a background thread, for tests and scripts."""

    def __init__(
        self,
        devices: Iterable[Device],
        host: str = "127.0.0.1",
        port: int = 0,
        auth: Optional[AuthConfig] = None,
    ):
        config = uvicorn.Config(
            build_device_app(devices, auth=auth),
            host=host,
            port=port,
            log_level="warning",
            timeout_graceful_shutdown=1,
        )
        self.app = config.app
        self.auth: AuthConfig = self.app.state.auth
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
    from fibsem.devices.drivers.demo import bind_demo_beams

    microscope, _ = utils.setup_session(manufacturer="Demo")
    return list(bind_demo_beams(microscope).values())


def demo_fm_devices() -> List[Device]:
    """The simulated FM's parts and group, as a METEOR PC would serve its FM."""
    from fibsem.devices.drivers.fm import bind_fm_devices
    from fibsem.fm.microscope import FluorescenceMicroscope

    return list(bind_fm_devices(FluorescenceMicroscope()).values())


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Serve simulated devices over HTTP.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--serve",
        nargs="+",
        choices=("beams", "fm"),
        default=["beams"],
        help="beams: the Demo microscope's beams; fm: a simulated FM's parts",
    )
    parser.add_argument(
        "--token-file",
        default=str(DEFAULT_TOKEN_FILE),
        help="the server's token; made the first time, kept after",
    )
    parser.add_argument(
        "--arm-hardware",
        action="store_true",
        help="allow writes and commands; without it the server is read-only",
    )
    parser.add_argument(
        "--pair",
        action="store_true",
        help="open a pairing window and show its code, for a coordinator to pair with",
    )
    args = parser.parse_args(argv)
    auth = AuthConfig.generate(
        arm_hardware=args.arm_hardware, token=load_or_create_token(args.token_file)
    )
    pairing = Pairing()
    served: List[Device] = []
    if "beams" in args.serve:
        served += demo_devices()
    if "fm" in args.serve:
        served += demo_fm_devices()
    devices: Mapping[str, Device] = {d.name: d for d in served}
    access = "read and control" if args.arm_hardware else "read only"
    print(f"Serving {sorted(devices)} on http://{args.host}:{args.port} ({access}).")
    print(f"Token file: {Path(args.token_file).expanduser()}")
    if args.pair:
        code = pairing.open()
        print(
            f"Pairing code: {code} (works once, for {pairing.seconds:.0f} s). Enter "
            "it on the microscope PC with:\n"
            f"    python -m fibsem.server.pairing pair <this PC's address> "
            f"{args.port} {code}"
        )
    uvicorn.run(
        build_device_app(devices.values(), auth=auth, pairing=pairing),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
