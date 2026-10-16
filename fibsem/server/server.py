"""The App Server: the fibsem server, embedded in the AutoLamella app (FIB-852).

``build_server`` constructs the FastAPI app around an already-connected
``FibsemMicroscope``: the app passes its live microscope and an ``app_context``
that additionally mounts the app-level router.

Every route requires a bearer token, and an agent level (``auth``): reads at
Watch, acting at Drive. The agent stops its own action at Assist or Drive,
anything else only at Drive, and never at Watch; every stop carries a reason. A hardware command also takes control
of the microscope (FIB-1270) for as long as it runs; when a run, a script or
another command is in control it is refused with a structured ``409
microscope_busy`` naming who, rather than interleaved or queued.
"""

import logging
from typing import Any, Callable, Mapping, Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import JSONResponse, Response
from fastapi.routing import APIRoute
from starlette.exceptions import HTTPException as StarletteHTTPException

from fibsem import __version__
from fibsem.acting import AGENT, acting
from fibsem.microscope import FibsemMicroscope
from fibsem.server import allowlist
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
    adopt_control,
    drive_is_lent,
    require_level,
    require_monitoring,
    take_control,
)
from fibsem.server.devices import ERROR_STATUS, build_device_router, refusal
from fibsem.server.discovery import fibsem_folder
from fibsem.server.models import MicroscopeStateRequest
from fibsem.server.requests import RequestStore, build_request_router
from fibsem.server.runs import Runs
from fibsem.structures import BeamType, MicroscopeState

API_VERSION = "0.3.0"


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


# Device and service commands that stop something (an FM z-stack stops through
# `imaging.stop`). No control taken, so never refused by the running move or mill.
STOP_COMMANDS = frozenset(("stop", "stop_live"))

# The instrument half of the API; /app is the application's.
MICROSCOPE = "/microscope"

_watch = require_level(AgentLevel.WATCH)
_assist = require_level(AgentLevel.ASSIST)
_drive = require_level(AgentLevel.DRIVE)


def _refuse(app_context) -> Callable[[Any, str, str, Mapping[str, Any]], Optional[str]]:
    """The App Server's allowlist (`fibsem.server.allowlist`): what the agent may
    never touch, and a path it may only give inside the open experiment, to a
    command or a query."""

    def experiment_path() -> Optional[str]:
        return getattr(app_context, "experiment_path", None)

    def refuse(device, kind: str, name: str, kwargs: Mapping[str, Any]):
        return allowlist.blocked(device, kind, name) or (
            allowlist.path_outside(kwargs, experiment_path)
            if kind in ("command", "query")
            else None
        )

    return refuse


def _not_allowed(microscope: FibsemMicroscope) -> Callable[[str, str], Optional[str]]:
    """Why a route call is on the allowlist's block list (a device or service
    write or command), or None: for a request, before it is filed. Its
    arguments are checked when it runs, as for any call."""

    def why(method: str, path: str) -> Optional[str]:
        parts = path.strip("/").split("/")
        if len(parts) < 4 or "/" + parts[0] != MICROSCOPE:
            return None
        if method == "PUT" and len(parts) == 4:
            kind, name = "write", parts[3]
        elif method == "POST" and len(parts) == 5 and parts[3] == "commands":
            kind, name = "command", parts[4]
        else:
            return None
        if parts[1] == "devices":
            devices = microscope.devices
        elif parts[1] == "services":
            devices = microscope.services
        else:
            return None
        device = devices.get(parts[2])
        return None if device is None else allowlist.blocked(device, kind, name)

    return why


def _device_command_gate(request: Request, command: str):
    """A device command's gate: a stop by the stop rule (:func:`_stop_gate`),
    anything else Drive and control of the microscope."""
    if command in STOP_COMMANDS:
        _stop_gate(request)
        yield
        return
    _drive(request)
    yield from take_control(request)


def _agents_own_action(request: Request) -> bool:
    """Whether what is running now is the agent's own: its command holds control
    of the microscope (FIB-1270). Control is one at a time, so while the agent
    holds it, nothing anyone else started is running through it."""
    in_control = request.app.state.microscope.control.current
    return in_control is not None and in_control.by == AGENT


def _stop_gate(request: Request) -> None:
    """The stop rule (Patrick, 2026-10-14): the agent stops its own action at the
    levels it acts at, Assist and Drive, and anything else (a workflow, the
    operator's mill) only at Drive, or in a request the operator approved. Watch
    never stops. A wrong stop isn't free: a mill stopped partway can spoil a
    lamella, and a stopped run loses hours."""
    if _agents_own_action(request):
        _assist(request)
    else:
        _drive(request)


async def _stop_reason(request: Request):
    """Every agent stop carries a reason: refused without one, and recorded as
    ``agent_stop`` (what was stopped, why, whether it was the agent's own) once
    it has stopped, for the operator to read."""
    command = request.path_params.get("command")
    if command is not None and command not in STOP_COMMANDS:
        yield
        return
    try:
        body = await request.json()
    except ValueError:
        body = None
    reason = body.get("reason") if isinstance(body, dict) else None
    if not isinstance(reason, str) or not reason.strip():
        raise HTTPException(
            status_code=422,
            detail={
                "error_type": "missing_reason",
                "message": "Pass a reason with a stop: the operator reads why "
                "the agent stopped it.",
            },
        )
    own = _agents_own_action(request)
    yield
    action = action_name(request) if command is not None else "workflow"
    request.app.state.microscope.record_event(
        "agent_stop", {"action": action, "reason": reason.strip(), "own": own}
    )


# The refusals of the agent's permissions: its level, Assist's ask-first, Off,
# a stop, and the allowlist. The 409s and 422s are the call's own trouble, and
# the routes log those (``devices.refusal``).
AGENT_REFUSALS = ("level", "ask_first", "agent_off", "agent_stopped", "not_allowed")


def _record_refusal(request: Request, exc: StarletteHTTPException) -> None:
    """An agent's action refused for its permissions, as ``agent_refused`` on the
    record and one log line (FIB-1402): an agent trying to act without asking is
    what the operator should be able to see. Reads stay out of the record, as do
    the calls an approved request makes. Never raises."""
    try:
        detail = exc.detail if isinstance(exc.detail, dict) else {}
        error_type = detail.get("error_type")
        if (
            exc.status_code != 403
            or error_type not in AGENT_REFUSALS
            or request.method == "GET"
            or drive_is_lent()
        ):
            return
        auth: AuthConfig = request.app.state.auth
        payload = {
            # a request refused at filing names the call it carried
            "action": detail.get("call") or action_name(request),
            "route": f"{request.method} {request.url.path}",
            "error_type": error_type,
            "level": auth.level.value,
        }
        logging.info(
            f"agent refused: {payload['action']} ({payload['route']}): "
            f"403 {error_type} at {payload['level']}"
        )
        request.app.state.microscope.record_event("agent_refused", payload)
    except Exception:  # noqa: BLE001 - recording must not change the answer
        logging.debug("could not record an agent refusal", exc_info=True)


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
    serve_microscope: bool = True,
) -> FastAPI:
    """Build the server app around an already-connected microscope.

    ``auth`` defaults to a generated token at Watch.
    ``app_context`` is anything satisfying app_routes.AppContext (structurally
    -- the application's AutoLamellaAppContext in practice); passing one mounts the
    ``/app`` routes and flips ``routers.app`` in /capabilities.

    ``serve_microscope`` is the App Server's microscope setting (FIB-1035):
    every route under ``/microscope``, reads included, since a read reaches the
    instrument too. False leaves each answering 404 ``microscope_off``, and
    /capabilities reports ``routers.microscope`` off. The App Server passes its
    preference, which is off by default.
    """
    if auth is None:
        auth = AuthConfig.generate()

    app = FastAPI(title="fibsem server", version=API_VERSION)
    app.add_middleware(_AgentActs)
    app.state.auth = auth
    app.state.microscope = microscope

    @app.exception_handler(StarletteHTTPException)
    async def _refused(request: Request, exc: StarletteHTTPException) -> Response:
        _record_refusal(request, exc)
        return await http_exception_handler(request, exc)

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
    # The whole instrument: who it is, and its state across stage and beams.
    whole = APIRouter(prefix=MICROSCOPE, route_class=_RefusingRoute)

    # --- Health / capabilities ---

    @app.get("/health")
    def health():
        # Unauthenticated liveness probe; everything informative lives in /capabilities.
        return {"status": "ok"}

    def _page(filename: str):
        import pkgutil

        from fastapi.responses import HTMLResponse

        page = pkgutil.get_data("fibsem.server", filename)
        if page is None:  # pragma: no cover - packaging defect, not runtime state
            raise HTTPException(status_code=404, detail=f"{filename} not packaged")
        return HTMLResponse(page.decode("utf-8"))

    @app.get("/dashboard")
    def dashboard():
        # The monitor page. Served unauthenticated like /health — the file is
        # static and carries no session data; every API call the page makes
        # needs the bearer token, which reaches it via the URL fragment (never
        # sent to the server) or a paste. Renders app panels only when
        # /capabilities says an application is hosted.
        return _page("dashboard.html")

    @app.get("/dashboard/log")
    def dashboard_log():
        # The event stream as one filterable table; served like /dashboard.
        return _page("dashboard_log.html")

    @base.get("/capabilities")
    def capabilities():
        return {
            "api_version": API_VERSION,
            "routers": {"microscope": serve_microscope, "app": app_context is not None},
            # The agent level (FIB-1035): what the agent may do right now.
            "level": auth.level.value,
            # The code the server runs, so a sidecar can tell it imports
            # other code (a worktree on the app's PYTHONPATH, FIB-1429).
            "fibsem": {"version": __version__, "path": fibsem_folder()},
        }

    # --- The microscope as a whole ---

    @whole.get("", dependencies=[Depends(_watch)])
    def get_microscope():
        info = microscope.system.info
        return {
            "manufacturer": microscope.manufacturer,
            "model": info.model,
            "serial_number": info.serial_number,
            "software_version": info.software_version,
            # What is fitted is not in the configuration dict -- the instrument
            # answered it at connect -- so it travels beside it.
            "fitted": {"manipulator": microscope.is_available("manipulator")},
            "system": microscope.system.to_dict(),
        }

    @whole.get("/state", dependencies=[Depends(_watch)])
    def get_microscope_state():
        return {"microscope_state": microscope.get_microscope_state().to_dict()}

    # Restores beams and stage in their set order: Drive, and control of the
    # microscope, like any command.
    @whole.put("/state", dependencies=[Depends(_drive), Depends(take_control)])
    def set_microscope_state(body: MicroscopeStateRequest):
        microscope.set_microscope_state(
            MicroscopeState.from_dict(body.microscope_state)
        )
        return {"status": "ok"}

    app.include_router(base)
    if serve_microscope:
        app.include_router(whole)
        refuse = _refuse(app_context)
        # Every device's parameters and commands, as the device server serves
        # them: reads at Watch, writes and commands at Drive with control of the
        # microscope. The chamber's state and pressure are read here; pumping
        # and venting are the operator's, so the allowlist refuses them.
        app.include_router(
            build_device_router(
                lambda: microscope.devices,
                read=[Depends(_watch)],
                write=[Depends(_drive), Depends(take_control)],
                command=[Depends(_device_command_gate), Depends(_stop_reason)],
                prefix=MICROSCOPE + "/devices",
                refuse=refuse,
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
                command=[Depends(_device_command_gate), Depends(_stop_reason)],
                prefix=MICROSCOPE + "/services",
                refuse=refuse,
                noun="service",
                on_image=lambda image: _show_in_app(microscope, image),
                runs=Runs(record=microscope.record_event),
                adopt_control=adopt_control,
            )
        )
    else:
        # Off, the same paths answer why they are missing (FIB-1359): a bare
        # "Not Found" reads as a wrong path, not as the operator's choice.
        app.include_router(_microscope_off_router())
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
        # Stopping the workflow: never the agent's own, so Drive, or a request
        # the operator approved at Assist; with a reason.
        app.include_router(
            build_app_stop_router(app_context),
            dependencies=[Depends(_drive), Depends(_stop_reason)],
        )
        # Suggesting an answer to the run's question: Assist and Drive, with no
        # request, since the operator's click is what answers.
        app.include_router(
            build_app_suggest_router(app_context), dependencies=[Depends(_assist)]
        )
        # The agent's requests (FIB-1283): actions the operator approves in the
        # app, run through the routes above. The app decides them through
        # app.state.requests.
        mounted = {"base", "app"} | ({"microscope"} if serve_microscope else set())
        app.state.requests = RequestStore(
            app,
            auth,
            frozenset(mounted),
            emit=getattr(app_context, "record_event", None),
            describe=getattr(app_context, "describe_action", None),
            not_allowed=_not_allowed(microscope) if serve_microscope else None,
            image_refused=lambda path: allowlist.path_outside(
                {"image_path": path},
                lambda: getattr(app_context, "experiment_path", None),
            ),
        )
        app.include_router(build_request_router(app.state.requests))
    return app


MICROSCOPE_OFF_MESSAGE = (
    "The microscope isn't served: the operator turns it on in Tools → App Server."
)


def _microscope_off(request: Request) -> None:
    raise HTTPException(
        status_code=404,
        detail={"error_type": "microscope_off", "message": MICROSCOPE_OFF_MESSAGE},
    )


def _microscope_off_router() -> APIRouter:
    """Every path under /microscope, answering a 404 that says the microscope
    is not served. Behind the token like the routes they stand in for."""
    off = APIRouter(dependencies=[Depends(_watch)])
    for path in (MICROSCOPE, MICROSCOPE + "/{rest:path}"):
        off.add_api_route(
            path,
            _microscope_off,
            methods=["GET", "POST", "PUT", "DELETE"],
            include_in_schema=False,
        )
    return off
