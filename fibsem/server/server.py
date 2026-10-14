"""The App Server: the fibsem server, embedded in the AutoLamella app (FIB-852).

``build_server`` constructs the FastAPI app around an already-connected
``FibsemMicroscope``: the app passes its live microscope and an ``app_context``
that additionally mounts the app-level router.

Every route requires a bearer token, and an agent level (``auth``): reads at
Watch, stops at Assist, acting at Drive. A hardware command also takes control
of the microscope (FIB-1270) for as long as it runs; when a run, a script or
another command is in control it is refused with a structured ``409
microscope_busy`` naming who, rather than interleaved or queued.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from fastapi.routing import APIRoute

from fibsem.acting import AGENT, acting
from fibsem.devices.chamber import Chamber
from fibsem.microscope import FibsemMicroscope
from fibsem.server.app_routes import (
    build_app_config_router,
    build_app_control_router,
    build_app_router,
    build_app_stop_router,
    build_app_suggest_router,
)
from fibsem.server.auth import (
    AgentLevel,
    AuthConfig,
    action_name,
    require_level,
    require_monitoring,
    take_control,
)
from fibsem.server.devices import ERROR_STATUS, build_device_router, refusal
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
    MicroscopeStateRequest,
    MillingSettingsRequest,
    OrientationRequest,
    StableMoveRequest,
    StagePositionRequest,
    StagePositionResponse,
    VerticalMoveRequest,
)
from fibsem.server.requests import RequestStore, build_request_router
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


def _show_in_app(microscope: FibsemMicroscope, image) -> None:
    """Send an image a route acquired to the microscope's acquisition signal, as
    the workflow's one-shots do, so the app's SEM/FIB views show it (FIB-1375).

    Before this, an agent's image went back to the agent only: the app kept the
    frame from before the agent's move, and a question raised next showed that.
    """
    metadata = getattr(image, "metadata", None)
    beam_type = getattr(metadata, "beam_type", None) if metadata is not None else None
    if beam_type is BeamType.ELECTRON:
        microscope.sem_acquisition_signal.emit(image)
    elif beam_type is BeamType.ION:
        microscope.fib_acquisition_signal.emit(image)


def _beam_type(value: str) -> BeamType:
    try:
        return BeamType[value.upper()]
    except KeyError:
        raise HTTPException(
            status_code=422,
            detail={
                "error_type": "ValueError",
                "message": f"Unknown beam_type: {value!r}. Use 'ELECTRON' or 'ION'.",
                "parameter": "beam_type",
            },
        )


class _RefusingRoute(APIRoute):
    """A microscope route whose expected errors (a stage limit, a bad value, a
    stopped mill) are refusals, as on the device routes: a 4xx in the one refusal
    shape, logged as one line, not a 500 with a traceback (FIB-1377)."""

    def get_route_handler(self):
        handler = super().get_route_handler()

        async def refusing(request: Request) -> Response:
            try:
                return await handler(request)
            except tuple(ERROR_STATUS) as error:
                raise refusal(error, action_name(request)) from None

        return refusing


# Device and service commands that stop something. Like POST /stop_milling,
# no control taken, so never refused by the running move.
STOP_COMMANDS = frozenset(("stop", "stop_live"))

_watch = require_level(AgentLevel.WATCH)
_drive = require_level(AgentLevel.DRIVE)


def _served_devices(microscope: FibsemMicroscope) -> dict:
    """The devices the server serves: all but the chamber. Pumping and venting
    are the operator's, at the microscope, so the chamber is not served at all."""
    return {
        name: device
        for name, device in microscope.devices.items()
        if not isinstance(device, Chamber)
    }


def _device_command_gate(request: Request, command: str):
    """A device command's gate: a stop needs only a token, anything else Drive
    and control of the microscope."""
    if command in STOP_COMMANDS:
        _watch(request)
        yield
        return
    _drive(request)
    yield from take_control(request)


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
    devices: bool = True,
) -> FastAPI:
    """Build the server app around an already-connected microscope.

    ``auth`` defaults to a generated token at Watch.
    ``app_context`` is anything satisfying app_routes.AppContext (structurally
    -- the application's AutoLamellaAppContext in practice); passing one mounts the
    app-level router and flips ``routers.app`` in /capabilities.

    ``devices`` is the App Server's devices service (FIB-1035): every route
    that reaches the instrument -- the microscope routes here and the
    ``/devices/...`` router, reads included, since a read can reach the
    instrument too. False leaves them unregistered (404), and /capabilities
    reports both routers off. The App Server passes its preference, which is
    off by default.
    """
    if auth is None:
        auth = AuthConfig.generate()

    app = FastAPI(title="fibsem server", version=API_VERSION)
    app.add_middleware(_AgentActs)
    app.state.auth = auth
    app.state.microscope = microscope

    @app.exception_handler(Exception)
    def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # The remote caller must be able to tell a bad request from a fallen-over
        # microscope; FastAPI's default 500 hides everything.
        return JSONResponse(
            status_code=500,
            content={"detail": {"error_type": type(exc).__name__, "message": str(exc)}},
        )

    # /capabilities, which every client reads first, whatever is mounted, the
    # dashboard included.
    base = APIRouter(dependencies=[Depends(require_monitoring())])
    read = APIRouter(dependencies=[Depends(_watch)], route_class=_RefusingRoute)
    # Stopping: at Assist and Drive, with no request. Never takes control: a
    # stop must not be refused by the mill it stops.
    stop = APIRouter(
        dependencies=[Depends(require_level(AgentLevel.ASSIST))],
        route_class=_RefusingRoute,
    )
    # Hardware routes also take control, so a command never interleaves with a
    # run, a script or another command.
    hw = APIRouter(
        dependencies=[Depends(_drive), Depends(take_control)],
        route_class=_RefusingRoute,
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

    @base.get("/capabilities")
    def capabilities():
        return {
            "api_version": API_VERSION,
            "manufacturer": microscope.manufacturer,
            "routers": {
                "microscope": devices,
                "devices": devices,
                "app": app_context is not None,
            },
            # The agent level (FIB-1035): what the agent may do right now.
            "level": auth.level.value,
        }

    @read.get("/system")
    def get_system():
        return {
            "system": microscope.system.to_dict(),
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
        image = microscope.acquire_image(image_settings=image_settings, beam_type=bt)
        _show_in_app(microscope, image)
        return _image_response(image)

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
        _show_in_app(microscope, image)
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
        except KeyError as e:  # a field a pattern needs; refused as a bad value
            raise ValueError(f"A pattern has a bad or missing field: {e}") from e
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

    # Whoever may stop (stop, control or hardware), and no control taken: a stop must
    # never be refused by the in-flight command it exists to interrupt.
    @stop.post("/stop_milling")
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

    app.include_router(base)
    if devices:
        app.include_router(read)
        app.include_router(stop)
        app.include_router(hw)
        # Every device's parameters and commands but the chamber's, as the
        # device server serves them: reads at Watch, writes and commands at
        # Drive with control of the microscope.
        app.include_router(
            build_device_router(
                lambda: _served_devices(microscope),
                read=[Depends(_watch)],
                write=[Depends(_drive), Depends(take_control)],
                command=[Depends(_device_command_gate)],
                on_image=lambda image: _show_in_app(microscope, image),
            )
        )
        # The microscope's services (milling, stage movement, ...) have the
        # device shape, so the same routes serve them, under /services, with the
        # same gates: a service's stop is a stop like a device's.
        app.include_router(
            build_device_router(
                lambda: microscope.services,
                read=[Depends(_watch)],
                write=[Depends(_drive), Depends(take_control)],
                command=[Depends(_device_command_gate)],
                prefix="/services",
                noun="service",
            )
        )
    else:
        # Off, the same paths answer why they are missing (FIB-1359): a bare
        # "Not Found" reads as a wrong path, not as the operator's choice.
        app.include_router(_devices_off_router(read, stop, hw))
    if app_context is not None:
        # The gate applied here so auth stays in one place; the router itself
        # is a thin pass-through over the context's JSON-able snapshots. These
        # are the monitoring routes, so the dashboard token may GET them too.
        app.include_router(
            build_app_router(app_context),
            dependencies=[Depends(require_monitoring())],
        )
        # Acting on the session (answering prompts, editing configuration):
        # Drive, or an approved request at Assist.
        app.include_router(
            build_app_control_router(app_context), dependencies=[Depends(_drive)]
        )
        app.include_router(
            build_app_config_router(app_context), dependencies=[Depends(_drive)]
        )
        # Stopping the workflow: Assist and Drive. Never waits for approval.
        app.include_router(
            build_app_stop_router(app_context),
            dependencies=[Depends(require_level(AgentLevel.ASSIST))],
        )
        # Suggesting an answer to the run's question: Assist and Drive, with no
        # request, since the operator's click is what answers.
        app.include_router(
            build_app_suggest_router(app_context),
            dependencies=[Depends(require_level(AgentLevel.ASSIST))],
        )
        # The agent's requests (FIB-1283): actions the operator approves in the
        # app, run through the routes above. The app decides them through
        # app.state.requests.
        mounted = {"base", "app"} | ({"microscope", "devices"} if devices else set())
        app.state.requests = RequestStore(
            app,
            auth,
            frozenset(mounted),
            emit=getattr(app_context, "record_event", None),
        )
        app.include_router(build_request_router(app.state.requests))
    return app


DEVICES_OFF_MESSAGE = (
    "The microscope's devices aren't served: the operator turns them on in "
    "Tools → App Server."
)


def _devices_off(request: Request) -> None:
    raise HTTPException(
        status_code=404,
        detail={"error_type": "devices_off", "message": DEVICES_OFF_MESSAGE},
    )


def _devices_off_router(*routers: APIRouter) -> APIRouter:
    """The instrument routes' paths, each answering a 404 that says the devices
    are off. Behind the token like the routes they stand in for."""
    off = APIRouter(dependencies=[Depends(_watch)])
    paths = {route.path for router in routers for route in router.routes}
    for prefix in ("/devices", "/services"):
        paths.update((prefix, prefix + "/{rest:path}"))
    for path in sorted(paths):
        off.add_api_route(
            path,
            _devices_off,
            methods=["GET", "POST", "PUT", "DELETE"],
            include_in_schema=False,
        )
    return off
