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
import io
import logging
import math
import os
import threading
from typing import Optional

import tifffile as tff
import uvicorn
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from fibsem import utils
from fibsem.microscope import FibsemMicroscope
from fibsem.server.app_routes import (
    build_app_config_router,
    build_app_control_router,
    build_app_router,
)
from fibsem.server.auth import AuthConfig, Scope, command_slot, require_scope
from fibsem.server.discovery import (
    DISCOVERY_FILE,
    read_discovery_file,
    remove_discovery_file,
    write_discovery_file,
)
from fibsem.server.images import preview_payload
from fibsem.server.models import (
    AcquireImageRequest,
    AvailableValuesRequest,
    BeamSettingsRequest,
    BeamSystemSettingsRequest,
    BeamTypeRequest,
    DetectorSettingsRequest,
    DrawPatternsRequest,
    FinishMillingRequest,
    FlatToBeamRequest,
    FloatBeamRequest,
    ImageSettingsRequest,
    IsCloseToMillingAngleRequest,
    MicroscopeStateRequest,
    MillingAngleFromPositionRequest,
    MillingAngleRequest,
    MillingSettingsRequest,
    MoveToMillingAngleRequest,
    PointBeamRequest,
    ProjectStableMoveRequest,
    ResolutionBeamRequest,
    RunMillingRequest,
    StableMoveRequest,
    StagePositionRequest,
    StagePositionResponse,
    StringBeamRequest,
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
    Point,
)

API_VERSION = "0.1.0"

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
    buf = io.BytesIO()
    metadata = image.metadata.to_dict() if image.metadata is not None else None
    tff.imwrite(buf, image.data, metadata=metadata)
    return Response(content=buf.getvalue(), media_type="image/tiff")


def _beam_type(value: str) -> BeamType:
    try:
        return BeamType[value.upper()]
    except KeyError:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown beam_type: {value!r}. Use 'ELECTRON' or 'ION'.",
        )


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
            "routers": {"microscope": True, "app": app_context is not None},
            "scopes": {s.value: auth.is_armed(s) for s in Scope},
        }

    @read.get("/system")
    def get_system():
        return {
            "system": microscope.system.to_dict(),
            "stage_is_compustage": microscope.stage_is_compustage,
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

    @hw.post("/move_flat_to_beam")
    def move_flat_to_beam(body: FlatToBeamRequest):
        microscope.move_flat_to_beam(beam_type=_beam_type(body.beam_type))
        return {"status": "ok"}

    # --- Microscope state ---

    @read.get("/microscope_state")
    def get_microscope_state():
        return {"microscope_state": microscope.get_microscope_state().to_dict()}

    @hw.post("/microscope_state")
    def set_microscope_state(body: MicroscopeStateRequest):
        microscope.set_microscope_state(
            MicroscopeState.from_dict(body.microscope_state)
        )
        return {"status": "ok"}

    # --- Imaging settings ---

    @read.post("/imaging_settings/get")
    def get_imaging_settings(body: BeamTypeRequest):
        return {
            "image_settings": microscope.get_imaging_settings(
                _beam_type(body.beam_type)
            ).to_dict()
        }

    @hw.post("/imaging_settings/set")
    def set_imaging_settings(body: ImageSettingsRequest):
        microscope.set_imaging_settings(ImageSettings.from_dict(body.image_settings))
        return {"status": "ok"}

    # --- Beam settings ---

    @read.post("/beam_settings/get")
    def get_beam_settings(body: BeamTypeRequest):
        return {
            "beam_settings": microscope.get_beam_settings(
                _beam_type(body.beam_type)
            ).to_dict()
        }

    @hw.post("/beam_settings/set")
    def set_beam_settings(body: BeamSettingsRequest):
        microscope.set_beam_settings(BeamSettings.from_dict(body.beam_settings))
        return {"status": "ok"}

    @read.post("/beam_system_settings/get")
    def get_beam_system_settings(body: BeamTypeRequest):
        return {
            "beam_system_settings": microscope.get_beam_system_settings(
                _beam_type(body.beam_type)
            ).to_dict()
        }

    @hw.post("/beam_system_settings/set")
    def set_beam_system_settings(body: BeamSystemSettingsRequest):
        microscope.set_beam_system_settings(
            BeamSystemSettings.from_dict(body.beam_system_settings)
        )
        return {"status": "ok"}

    # --- Detector settings ---

    @read.post("/detector_settings/get")
    def get_detector_settings(body: BeamTypeRequest):
        return {
            "detector_settings": microscope.get_detector_settings(
                _beam_type(body.beam_type)
            ).to_dict()
        }

    @hw.post("/detector_settings/set")
    def set_detector_settings(body: DetectorSettingsRequest):
        microscope.set_detector_settings(
            FibsemDetectorSettings.from_dict(body.detector_settings),
            beam_type=_beam_type(body.beam_type),
        )
        return {"status": "ok"}

    # --- Individual beam getters / setters ---

    @read.post("/beam_current/get")
    def get_beam_current(body: BeamTypeRequest):
        return {"value": microscope.get_beam_current(_beam_type(body.beam_type))}

    @hw.post("/beam_current/set")
    def set_beam_current(body: FloatBeamRequest):
        return {
            "value": microscope.set_beam_current(body.value, _beam_type(body.beam_type))
        }

    @read.post("/beam_voltage/get")
    def get_beam_voltage(body: BeamTypeRequest):
        return {"value": microscope.get_beam_voltage(_beam_type(body.beam_type))}

    @hw.post("/beam_voltage/set")
    def set_beam_voltage(body: FloatBeamRequest):
        return {
            "value": microscope.set_beam_voltage(body.value, _beam_type(body.beam_type))
        }

    @read.post("/field_of_view/get")
    def get_field_of_view(body: BeamTypeRequest):
        return {"value": microscope.get_field_of_view(_beam_type(body.beam_type))}

    @hw.post("/field_of_view/set")
    def set_field_of_view(body: FloatBeamRequest):
        return {
            "value": microscope.set_field_of_view(
                body.value, _beam_type(body.beam_type)
            )
        }

    @read.post("/working_distance/get")
    def get_working_distance(body: BeamTypeRequest):
        return {"value": microscope.get_working_distance(_beam_type(body.beam_type))}

    @hw.post("/working_distance/set")
    def set_working_distance(body: FloatBeamRequest):
        return {
            "value": microscope.set_working_distance(
                body.value, _beam_type(body.beam_type)
            )
        }

    @read.post("/dwell_time/get")
    def get_dwell_time(body: BeamTypeRequest):
        return {"value": microscope.get_dwell_time(_beam_type(body.beam_type))}

    @hw.post("/dwell_time/set")
    def set_dwell_time(body: FloatBeamRequest):
        return {
            "value": microscope.set_dwell_time(body.value, _beam_type(body.beam_type))
        }

    @read.post("/resolution/get")
    def get_resolution(body: BeamTypeRequest):
        return {"value": list(microscope.get_resolution(_beam_type(body.beam_type)))}

    @hw.post("/resolution/set")
    def set_resolution(body: ResolutionBeamRequest):
        return {
            "value": list(
                microscope.set_resolution(body.value, _beam_type(body.beam_type))
            )
        }

    @read.post("/scan_rotation/get")
    def get_scan_rotation(body: BeamTypeRequest):
        return {"value": microscope.get_scan_rotation(_beam_type(body.beam_type))}

    @hw.post("/scan_rotation/set")
    def set_scan_rotation(body: FloatBeamRequest):
        return {
            "value": microscope.set_scan_rotation(
                body.value, _beam_type(body.beam_type)
            )
        }

    @read.post("/stigmation/get")
    def get_stigmation(body: BeamTypeRequest):
        return {
            "value": microscope.get_stigmation(_beam_type(body.beam_type)).to_dict()
        }

    @hw.post("/stigmation/set")
    def set_stigmation(body: PointBeamRequest):
        result = microscope.set_stigmation(
            Point.from_dict(body.value), _beam_type(body.beam_type)
        )
        return {"value": result.to_dict()}

    @read.post("/beam_shift/get")
    def get_beam_shift(body: BeamTypeRequest):
        return {
            "value": microscope.get_beam_shift(_beam_type(body.beam_type)).to_dict()
        }

    @hw.post("/beam_shift/set")
    def set_beam_shift(body: PointBeamRequest):
        result = microscope.set_beam_shift(
            Point.from_dict(body.value), _beam_type(body.beam_type)
        )
        return {"value": result.to_dict()}

    # --- Detector individual getters / setters ---

    @read.post("/detector_type/get")
    def get_detector_type(body: BeamTypeRequest):
        return {"value": microscope.get_detector_type(_beam_type(body.beam_type))}

    @hw.post("/detector_type/set")
    def set_detector_type(body: StringBeamRequest):
        return {
            "value": microscope.set_detector_type(
                body.value, _beam_type(body.beam_type)
            )
        }

    @read.post("/detector_mode/get")
    def get_detector_mode(body: BeamTypeRequest):
        return {"value": microscope.get_detector_mode(_beam_type(body.beam_type))}

    @hw.post("/detector_mode/set")
    def set_detector_mode(body: StringBeamRequest):
        return {
            "value": microscope.set_detector_mode(
                body.value, _beam_type(body.beam_type)
            )
        }

    @read.post("/detector_contrast/get")
    def get_detector_contrast(body: BeamTypeRequest):
        return {"value": microscope.get_detector_contrast(_beam_type(body.beam_type))}

    @hw.post("/detector_contrast/set")
    def set_detector_contrast(body: FloatBeamRequest):
        return {
            "value": microscope.set_detector_contrast(
                body.value, _beam_type(body.beam_type)
            )
        }

    @read.post("/detector_brightness/get")
    def get_detector_brightness(body: BeamTypeRequest):
        return {"value": microscope.get_detector_brightness(_beam_type(body.beam_type))}

    @hw.post("/detector_brightness/set")
    def set_detector_brightness(body: FloatBeamRequest):
        return {
            "value": microscope.set_detector_brightness(
                body.value, _beam_type(body.beam_type)
            )
        }

    # --- Available values ---

    @read.post("/available_values")
    def get_available_values(body: AvailableValuesRequest):
        beam_type = _beam_type(body.beam_type) if body.beam_type else None
        return {
            "values": microscope.get_available_values(body.key, beam_type=beam_type)
        }

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
    def run_milling(body: RunMillingRequest):
        microscope.run_milling(
            milling_current=body.milling_current,
            milling_voltage=body.milling_voltage,
            asynch=body.asynch,
        )
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
