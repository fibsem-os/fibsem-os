"""The fibsem server: one server codebase per microscope, hosted two ways (FIB-852).

``build_server`` constructs the FastAPI app around an already-connected
``FibsemMicroscope``:

- **Bench hosting** (this module's CLI): the server process owns the microscope
  connection and mounts the microscope router only.
- **Embedded hosting** (the AutoLamella app): the app passes its live microscope
  and, once FIB-846 lands, an ``app_context`` that additionally mounts the
  app-level router.

Every route requires a bearer token. Routes are split into two scopes: ``read``
(observation; granted to any valid token) and ``hardware`` (commands and
mutations; armed explicitly by the hosting). Hardware commands are additionally
serialized by a per-microscope lock — a second concurrent command is refused
with a structured ``409 busy`` rather than interleaved or queued.

Usage:

    from fibsem.server import FibsemServer
    server = FibsemServer.from_session(manufacturer="Demo", ip_address="localhost")
    server.run()   # token is generated and logged; read-only unless armed

Or as a script:

    python -m fibsem.server.server --manufacturer Demo --arm-hardware
"""

import atexit
import logging
import math
import os
import threading
from typing import Optional

import uvicorn
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from fibsem import utils
from fibsem.acting import AGENT, acting
from fibsem.microscope import FibsemMicroscope
from fibsem.server.app_routes import (
    build_app_config_router,
    build_app_control_router,
    build_app_router,
)
from fibsem.server.auth import AuthConfig, Scope, command_slot, require_scope
from fibsem.server.devices import build_device_router
from fibsem.server.discovery import (
    DISCOVERY_FILE,
    read_discovery_file,
    remove_discovery_file,
    write_discovery_file,
)
from fibsem.server.images import TIFF_MEDIA_TYPE, preview_payload, tiff_bytes
from fibsem.server.models import (
    AcquireImageRequest,
    BeamSettingsRequest,
    BeamSystemSettingsRequest,
    BeamTypeRequest,
    DetectorSettingsRequest,
    DrawPatternsRequest,
    FinishMillingRequest,
    ImageSettingsRequest,
    IsCloseToMillingAngleRequest,
    MicroscopeStateRequest,
    MillingAngleFromPositionRequest,
    MillingAngleRequest,
    MillingSettingsRequest,
    MoveToMillingAngleRequest,
    OrientationRequest,
    ProjectStableMoveRequest,
    StableMoveRequest,
    StagePositionRequest,
    StagePositionResponse,
    VerticalMoveRequest,
)
from fibsem.structures import (
    BeamSettings,
    BeamSystemSettings,
    BeamType,
    FibsemBitmapSettings,
    FibsemCircleSettings,
    FibsemDetectorSettings,
    FibsemLineSettings,
    FibsemMillingSettings,
    FibsemPatternSettings,
    FibsemPolygonSettings,
    FibsemRectangleSettings,
    FibsemStagePosition,
    ImageSettings,
    MicroscopeState,
)

API_VERSION = "0.2.0"

_PATTERN_CLASSES = {
    "Rectangle": FibsemRectangleSettings,
    "Line": FibsemLineSettings,
    "Circle": FibsemCircleSettings,
    "Bitmap": FibsemBitmapSettings,
    "Polygon": FibsemPolygonSettings,
}


def _pattern_from_dict(d: dict) -> FibsemPatternSettings:
    type_name = d.get("type")
    if type_name not in _PATTERN_CLASSES:
        raise ValueError(
            f"Unknown pattern type: {type_name!r}. Available: {list(_PATTERN_CLASSES)}"
        )
    return _PATTERN_CLASSES[type_name].from_dict(d)


def _image_response(image) -> Response:
    return Response(content=tiff_bytes(image), media_type=TIFF_MEDIA_TYPE)


def _beam_type(value: str) -> BeamType:
    try:
        return BeamType[value.upper()]
    except KeyError:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown beam_type: {value!r}. Use 'ELECTRON' or 'ION'.",
        )


# Device commands that stop something. Like POST /stop_milling, stopping is always
# allowed: read scope, and no wait for the command slot the running move holds.
STOP_COMMANDS = frozenset(("stop", "stop_live"))

_read_scope = require_scope(Scope.READ)
_hardware_scope = require_scope(Scope.HARDWARE)


def _device_command_slot(request: Request, command: str):
    """A device command's gate: a stop needs only a token, anything else the
    hardware scope and the command slot."""
    if command in STOP_COMMANDS:
        _read_scope(request)
        yield
        return
    _hardware_scope(request)
    yield from command_slot(request)


class _AgentActs:
    """Every request is the agent acting, for the experiment's record (FIB-1062).

    Pure ASGI middleware. The mark is a context variable, and a route handler --
    a sync function, which FastAPI runs in a worker thread -- runs with a copy
    of the request's context, so the mark reaches whatever the handler calls.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        with acting(AGENT):
            await self.app(scope, receive, send)


def build_server(
    microscope: FibsemMicroscope,
    app_context=None,
    auth: Optional[AuthConfig] = None,
) -> FastAPI:
    """Build the server app around an already-connected microscope.

    ``auth`` defaults to a generated token with only the ``read`` scope armed.
    ``app_context`` is anything satisfying app_routes.AppContext (structurally
    -- the application's AgentContext in practice); passing one mounts the
    app-level router and flips ``routers.app`` in /capabilities.
    """
    if auth is None:
        auth = AuthConfig.generate()

    app = FastAPI(title="fibsem server", version=API_VERSION)
    app.add_middleware(_AgentActs)
    app.state.auth = auth
    app.state.microscope = microscope
    # One hardware command in flight per microscope; see auth.command_slot.
    app.state.command_lock = threading.Lock()

    @app.exception_handler(Exception)
    def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # The remote caller must be able to tell a bad request from a fallen-over
        # microscope; FastAPI's default 500 hides everything.
        return JSONResponse(
            status_code=500,
            content={"detail": {"error_type": type(exc).__name__, "message": str(exc)}},
        )

    read = APIRouter(dependencies=[Depends(require_scope(Scope.READ))])
    # Hardware routes also take the command slot, so two commands never interleave.
    hw = APIRouter(
        dependencies=[Depends(require_scope(Scope.HARDWARE)), Depends(command_slot)]
    )

    # --- Health / capabilities ---

    @app.get("/health")
    def health():
        # Unauthenticated liveness probe; everything informative lives in /capabilities.
        return {"status": "ok"}

    @app.get("/dashboard")
    def dashboard():
        # The monitor page. Served unauthenticated like /health — the file is
        # static and carries no session data; every API call the page makes
        # needs the bearer token, which reaches it via the URL fragment (never
        # sent to the server) or a paste. Renders app panels only when
        # /capabilities says an application is hosted.
        import pkgutil

        from fastapi.responses import HTMLResponse

        page = pkgutil.get_data("fibsem.server", "dashboard.html")
        if page is None:  # pragma: no cover - packaging defect, not runtime state
            raise HTTPException(status_code=404, detail="dashboard.html not packaged")
        return HTMLResponse(page.decode("utf-8"))

    @read.get("/capabilities")
    def capabilities():
        return {
            "api_version": API_VERSION,
            "manufacturer": type(microscope).__name__,
            "routers": {
                "microscope": True,
                "devices": True,
                "app": app_context is not None,
            },
            "scopes": {s.value: auth.is_armed(s) for s in Scope},
        }

    @read.get("/system")
    def get_system():
        return {
            "system": microscope.system.to_dict(),
            # Clients from before the flag was retired require this key; nothing reads
            # it now. Drop it a release later.
            "stage_is_compustage": microscope._fm_is_a_pose(),
            # What is fitted is not in the configuration dict -- the instrument
            # answered it at connect -- so it travels beside it.
            "fitted": {"manipulator": microscope.is_available("manipulator")},
        }

    # --- Image acquisition ---

    @hw.post("/acquire_image")
    def acquire_image(body: AcquireImageRequest) -> Response:
        bt = _beam_type(body.beam_type)
        image_settings = (
            ImageSettings.from_dict(body.image_settings)
            if body.image_settings
            else None
        )
        return _image_response(
            microscope.acquire_image(image_settings=image_settings, beam_type=bt)
        )

    # hardware, not read: on Thermo this switches the active imaging channel
    # (shared-view state) before pulling the frame from the vendor server.
    @hw.post("/last_image")
    def last_image(body: BeamTypeRequest) -> Response:
        return _image_response(
            microscope.last_image(beam_type=_beam_type(body.beam_type))
        )

    @hw.post("/acquire_chamber_image")
    def acquire_chamber_image() -> Response:
        return _image_response(microscope.acquire_chamber_image())

    # --- Preview renditions (agent/browser-sized JPEG instead of full TIFF) ---

    @hw.post("/acquire_image_preview")
    def acquire_image_preview(body: AcquireImageRequest):
        bt = _beam_type(body.beam_type)
        image_settings = (
            ImageSettings.from_dict(body.image_settings)
            if body.image_settings
            else None
        )
        image = microscope.acquire_image(image_settings=image_settings, beam_type=bt)
        return preview_payload(image)

    @hw.post("/last_image_preview")
    def last_image_preview(body: BeamTypeRequest):
        image = microscope.last_image(beam_type=_beam_type(body.beam_type))
        return preview_payload(image)

    @hw.post("/autocontrast")
    def autocontrast(body: BeamTypeRequest):
        microscope.autocontrast(beam_type=_beam_type(body.beam_type))
        return {"status": "ok"}

    @hw.post("/auto_focus")
    def auto_focus(body: BeamTypeRequest):
        microscope.auto_focus(beam_type=_beam_type(body.beam_type))
        return {"status": "ok"}

    # --- Stage movement ---

    @read.get("/stage_position")
    def get_stage_position():
        return {"position": microscope.get_stage_position().to_dict()}

    @read.get("/stage_orientation")
    def get_stage_orientation():
        return {"orientation": microscope.get_stage_orientation()}

    @hw.post("/move_stage_absolute", response_model=StagePositionResponse)
    def move_stage_absolute(body: StagePositionRequest):
        result = microscope.move_stage_absolute(
            FibsemStagePosition.from_dict(body.position)
        )
        return StagePositionResponse(position=result.to_dict())

    @hw.post("/move_stage_relative", response_model=StagePositionResponse)
    def move_stage_relative(body: StagePositionRequest):
        result = microscope.move_stage_relative(
            FibsemStagePosition.from_dict(body.position)
        )
        return StagePositionResponse(position=result.to_dict())

    @hw.post("/stable_move", response_model=StagePositionResponse)
    def stable_move(body: StableMoveRequest):
        result = microscope.stable_move(
            dx=body.dx, dy=body.dy, beam_type=_beam_type(body.beam_type)
        )
        return StagePositionResponse(position=result.to_dict())

    # read, not hardware: pure geometry from an explicit base position, no motion.
    @read.post("/project_stable_move", response_model=StagePositionResponse)
    def project_stable_move(body: ProjectStableMoveRequest):
        base_position = FibsemStagePosition.from_dict(body.base_position)
        result = microscope.project_stable_move(
            dx=body.dx,
            dy=body.dy,
            beam_type=_beam_type(body.beam_type),
            base_position=base_position,
        )
        return StagePositionResponse(position=result.to_dict())

    @hw.post("/vertical_move", response_model=StagePositionResponse)
    def vertical_move(body: VerticalMoveRequest):
        result = microscope.vertical_move(dy=body.dy, dx=body.dx)
        return StagePositionResponse(position=result.to_dict())

    @hw.post("/safe_absolute_stage_movement")
    def safe_absolute_stage_movement(body: StagePositionRequest):
        microscope.safe_absolute_stage_movement(
            FibsemStagePosition.from_dict(body.position)
        )
        return {"status": "ok"}

    @hw.post("/move_to_orientation")
    def move_to_orientation(body: OrientationRequest):
        result = microscope.move_to_orientation(body.orientation)
        return StagePositionResponse(position=result.to_dict())

    # --- Microscope state ---

    @read.get("/microscope_state")
    def get_microscope_state():
        return {"microscope_state": microscope.get_microscope_state().to_dict()}

    @hw.put("/microscope_state")
    def set_microscope_state(body: MicroscopeStateRequest):
        microscope.set_microscope_state(
            MicroscopeState.from_dict(body.microscope_state)
        )
        return {"status": "ok"}

    # --- A beam's settings, as one group ---
    # Single parameters are /devices/{electron|ion}/{parameter}; these read and
    # write a whole group at once. The beam is the path's, whatever the body says.

    @read.get("/beams/{beam}/imaging_settings")
    def get_imaging_settings(beam: str):
        return {
            "image_settings": microscope.get_imaging_settings(
                _beam_type(beam)
            ).to_dict()
        }

    @hw.put("/beams/{beam}/imaging_settings")
    def set_imaging_settings(beam: str, body: ImageSettingsRequest):
        settings = ImageSettings.from_dict(body.image_settings)
        settings.beam_type = _beam_type(beam)
        microscope.set_imaging_settings(settings)
        return {"status": "ok"}

    @read.get("/beams/{beam}/beam_settings")
    def get_beam_settings(beam: str):
        return {
            "beam_settings": microscope.get_beam_settings(_beam_type(beam)).to_dict()
        }

    @hw.put("/beams/{beam}/beam_settings")
    def set_beam_settings(beam: str, body: BeamSettingsRequest):
        settings = BeamSettings.from_dict(body.beam_settings)
        settings.beam_type = _beam_type(beam)
        microscope.set_beam_settings(settings)
        return {"status": "ok"}

    @read.get("/beams/{beam}/beam_system_settings")
    def get_beam_system_settings(beam: str):
        return {
            "beam_system_settings": microscope.get_beam_system_settings(
                _beam_type(beam)
            ).to_dict()
        }

    @hw.put("/beams/{beam}/beam_system_settings")
    def set_beam_system_settings(beam: str, body: BeamSystemSettingsRequest):
        settings = BeamSystemSettings.from_dict(body.beam_system_settings)
        settings.beam_type = _beam_type(beam)
        microscope.set_beam_system_settings(settings)
        return {"status": "ok"}

    @read.get("/beams/{beam}/detector_settings")
    def get_detector_settings(beam: str):
        return {
            "detector_settings": microscope.get_detector_settings(
                _beam_type(beam)
            ).to_dict()
        }

    @hw.put("/beams/{beam}/detector_settings")
    def set_detector_settings(beam: str, body: DetectorSettingsRequest):
        microscope.set_detector_settings(
            FibsemDetectorSettings.from_dict(body.detector_settings),
            beam_type=_beam_type(beam),
        )
        return {"status": "ok"}

    # --- Milling angle ---
    # The HTTP boundary speaks DEGREES everywhere (fields named *_deg).
    # The ABC's move_to_milling_angle takes radians (FIB-853); converted here.

    @read.get("/milling_angle")
    def get_milling_angle():
        return {"milling_angle_deg": microscope.get_current_milling_angle()}

    @read.post("/milling_angle/from_position")
    def get_milling_angle_from_position(body: MillingAngleFromPositionRequest):
        position = (
            FibsemStagePosition.from_dict(body.stage_position)
            if body.stage_position
            else None
        )
        return {
            "milling_angle_deg": microscope.get_current_milling_angle(
                stage_position=position
            )
        }

    @hw.post("/milling_angle/set")
    def set_milling_angle(body: MillingAngleRequest):
        microscope.set_milling_angle(body.milling_angle_deg)
        return {"status": "ok"}

    @hw.post("/milling_angle/move")
    def move_to_milling_angle(body: MoveToMillingAngleRequest):
        rotation = (
            math.radians(body.rotation_deg) if body.rotation_deg is not None else None
        )
        success = microscope.move_to_milling_angle(
            math.radians(body.milling_angle_deg), rotation=rotation
        )
        return {
            "success": success,
            "milling_angle_deg": microscope.get_current_milling_angle(),
        }

    @read.post("/milling_angle/is_close")
    def is_close_to_milling_angle(body: IsCloseToMillingAngleRequest):
        return {
            "is_close": microscope.is_close_to_milling_angle(
                body.milling_angle_deg, atol=body.atol_deg
            )
        }

    # --- Milling ---

    @hw.post("/setup_milling")
    def setup_milling(body: MillingSettingsRequest):
        microscope.setup_milling(
            mill_settings=FibsemMillingSettings.from_dict(body.mill_settings)
        )
        return {"status": "ok"}

    @hw.post("/draw_patterns")
    def draw_patterns(body: DrawPatternsRequest):
        try:
            patterns = [_pattern_from_dict(p) for p in body.patterns]
        except (KeyError, ValueError) as e:
            raise HTTPException(status_code=422, detail=str(e))
        microscope.draw_patterns(patterns)
        return {"status": "ok"}

    @hw.post("/run_milling")
    def run_milling():
        microscope.run_milling()
        return {"status": "ok"}

    @hw.post("/start_milling")
    def start_milling():
        microscope.start_milling()
        return {"status": "ok"}

    # read scope, and no command slot: an emergency stop must never be blocked
    # by arming or by the in-flight command it exists to interrupt.
    @read.post("/stop_milling")
    def stop_milling():
        microscope.stop_milling()
        return {"status": "ok"}

    @hw.post("/pause_milling")
    def pause_milling():
        microscope.pause_milling()
        return {"status": "ok"}

    @hw.post("/resume_milling")
    def resume_milling():
        microscope.resume_milling()
        return {"status": "ok"}

    @hw.post("/finish_milling")
    def finish_milling(body: FinishMillingRequest):
        microscope.finish_milling(
            imaging_current=body.imaging_current,
            imaging_voltage=body.imaging_voltage,
        )
        return {"status": "ok"}

    @hw.post("/clear_patterns")
    def clear_patterns():
        microscope.clear_patterns()
        return {"status": "ok"}

    @read.get("/milling_state")
    def get_milling_state():
        return {"state": microscope.get_milling_state().name}

    @read.get("/estimate_milling_time")
    def estimate_milling_time():
        return {"seconds": microscope.estimate_milling_time()}

    app.include_router(read)
    app.include_router(hw)
    # Every device's parameters and commands, as the device server serves them:
    # reads are read scope, writes and commands hardware scope and the command slot.
    app.include_router(
        build_device_router(
            lambda: microscope.devices,
            read=[Depends(require_scope(Scope.READ))],
            write=[Depends(require_scope(Scope.HARDWARE)), Depends(command_slot)],
            command=[Depends(_device_command_slot)],
        )
    )
    if app_context is not None:
        # Read scope applied here so auth stays in one place; the router itself
        # is a thin pass-through over the context's JSON-able snapshots.
        app.include_router(
            build_app_router(app_context),
            dependencies=[Depends(require_scope(Scope.READ))],
        )
        # Acting on the session (answering prompts) is control scope — armed by
        # the hosting, never by default; unarmed callers get 403 scope_not_armed.
        app.include_router(
            build_app_control_router(app_context),
            dependencies=[Depends(require_scope(Scope.CONTROL))],
        )
        app.include_router(
            build_app_config_router(app_context),
            dependencies=[Depends(require_scope(Scope.CONFIGURE))],
        )
    return app


class FibsemServer:
    """Bench hosting: own the microscope connection and serve it.

    Binds localhost by default; exposing on the LAN is an explicit choice.
    """

    def __init__(
        self,
        microscope: FibsemMicroscope,
        host: str = "127.0.0.1",
        port: int = 8001,
        auth: Optional[AuthConfig] = None,
    ):
        self.microscope = microscope
        self.host = host
        self.port = port
        self.auth = auth or AuthConfig.generate()
        self.app = build_server(microscope, auth=self.auth)

    def run(self):
        # One server per microscope/machine: the discovery file is the guard.
        existing = read_discovery_file()
        if existing is not None and existing.get("pid") != os.getpid():
            raise RuntimeError(
                f"A fibsem server already appears to be running "
                f"(pid {existing.get('pid')}, {existing.get('url')}). "
                "One server per microscope: stop it first, or delete "
                f"{DISCOVERY_FILE} if it is stale."
            )
        armed = ", ".join(s.value for s in self.auth.armed_scopes())
        logging.info(
            "fibsem server on http://%s:%s — scopes armed: %s",
            self.host,
            self.port,
            armed,
        )
        logging.info("bearer token: %s", self.auth.token)
        write_discovery_file(self.host, self.port, self.auth)
        atexit.register(remove_discovery_file)
        try:
            uvicorn.run(self.app, host=self.host, port=self.port)
        finally:
            remove_discovery_file()

    @classmethod
    def from_session(
        cls,
        manufacturer: Optional[str] = None,
        ip_address: Optional[str] = None,
        config_path: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 8001,
        auth: Optional[AuthConfig] = None,
    ) -> "FibsemServer":
        microscope, _ = utils.setup_session(
            manufacturer=manufacturer, ip_address=ip_address, config_path=config_path
        )
        return cls(microscope, host=host, port=port, auth=auth)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Start a fibsem microscope server (bench hosting)"
    )
    parser.add_argument(
        "--manufacturer",
        default=None,
        help="Microscope manufacturer (default: from config)",
    )
    parser.add_argument(
        "--ip-address",
        default=None,
        help="Microscope IP address (default: from config)",
    )
    parser.add_argument(
        "--config", default=None, help="Path to a microscope configuration file"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1; use 0.0.0.0 to expose on the LAN)",
    )
    parser.add_argument(
        "--port", type=int, default=8001, help="Server port (default: 8001)"
    )
    parser.add_argument(
        "--token", default=None, help="Bearer token (default: generated and logged)"
    )
    parser.add_argument(
        "--arm-hardware",
        action="store_true",
        help="Arm the hardware scope (moves, acquisition, milling)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    server = FibsemServer.from_session(
        manufacturer=args.manufacturer,
        ip_address=args.ip_address,
        config_path=args.config,
        host=args.host,
        port=args.port,
        auth=AuthConfig.generate(arm_hardware=args.arm_hardware, token=args.token),
    )
    server.run()
