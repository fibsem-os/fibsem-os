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

import hmac
import json
import logging
from typing import Any, Callable, Mapping, Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import JSONResponse, Response
from fastapi.routing import APIRoute
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException as StarletteHTTPException

from fibsem import __version__
from fibsem.acting import AGENT, acting
from fibsem.ai.agent import allowlist
from fibsem.ai.agent.calls import calling, current_call, from_headers
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
from fibsem.structures import BeamType, MicroscopeState
from fibsem.workflows.actions import ACTION, action_record

API_VERSION = "0.3.0"


def _show_in_app(microscope: FibsemMicroscope, image) -> None:
    """Show an image a route acquired in the app's SEM/FIB views, as the
    workflow's one-shots do (FIB-1375).

    Before this, an agent's image went back to the agent only: the app kept the
    frame from before the agent's move, and a question raised next showed that.
    """
    metadata = getattr(image, "metadata", None)
    beam_type = getattr(metadata, "beam_type", None) if metadata is not None else None
    imaging = getattr(microscope, "imaging", None)
    if beam_type in (BeamType.ELECTRON, BeamType.ION) and imaging is not None:
        imaging.show(image)


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
# `imaging.stop`; milling's `end` ends the stage keeping what it milled). No
# control taken, so never refused by the running move or mill.
STOP_COMMANDS = frozenset(("stop", "stop_live", "end"))

# The instrument half of the API; /app is the application's.
MICROSCOPE = "/microscope"
# The analysis services, which need no microscope (`fibsem.services.analysis`).
SERVICES = "/services"

_watch = require_level(AgentLevel.WATCH)
_assist = require_level(AgentLevel.ASSIST)
_drive = require_level(AgentLevel.DRIVE)


def _refuse(app_context) -> Callable[[Any, str, str, Mapping[str, Any]], Optional[str]]:
    """The App Server's allowlist (`fibsem.ai.agent.allowlist`): what the agent may
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


# The app's stop routes and what each stops, as the operator's own stops record it.
_APP_STOPS = {"/app/workflow/stop": "workflow", "/app/workflow/stop_task": "task"}


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
    if command is not None:
        action = action_name(request)
    else:
        action = _APP_STOPS.get(request.url.path, "workflow")
    call = current_call()
    if call is not None:
        call.recorded = True
    owner = request.path_params.get("service") or request.path_params.get("device")
    target = _STOPPED.get(owner or "workflow", f"the {_words(owner)}")
    verb = "Stop"
    if command == "end":
        target, verb = "the milling stage", "End"
    request.app.state.microscope.record_event(
        ACTION,
        action_record(
            f"{verb} {target}",
            reason=reason.strip(),
            stops=action,
            target=target,
            own=own,
            level=request.app.state.auth.level.value,
        ),
    )


# What a stop stops, as the operator says it: "Stopped the mill: <why>".
_STOPPED = {
    "workflow": "the run",
    "milling": "the mill",
    "imaging": "imaging",
    "fluorescence": "the fluorescence acquisition",
}


def _words(name) -> str:
    return str(name or "").replace("_", " ")


# What an app route refused at Watch was for, where its last word says too
# little ("Tried to requests").
_REFUSED_ACTIONS = {
    "/app/agent/requests": "file a request",
    "/app/workflow/proposals/suggest": "suggest on a proposal",
}


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
        call = current_call()
        payload = action_record(
            # a request refused at filing names the call it carried
            detail.get("call")
            or _REFUSED_ACTIONS.get(request.url.path.rstrip("/"))
            or action_name(request),
            ok=False,
            refused=True,
            route=f"{request.method} {request.url.path}",
            error_type=error_type,
            level=auth.level.value,
        )
        if call is not None and call.action and not detail.get("call"):
            # The call as the card says it (FIB-1426), when the sidecar named it.
            payload["line"] = call.action
        logging.info(
            f"agent refused: {payload['what']} ({payload['route']}): "
            f"403 {error_type} at {payload['level']}"
        )
        request.app.state.microscope.record_event(ACTION, payload)
    except Exception:  # noqa: BLE001 - recording must not change the answer
        logging.debug("could not record an agent refusal", exc_info=True)


# Routes that put the agent's call on the record themselves: its requests, its
# notes, its suggestions. A stop says so as it stops (`_stop_reason`).
_RECORDS_ITSELF = (
    "/app/agent/",
    "/app/experiment/notes",
    "/app/workflow/proposals/suggest",
)
# POST routes that only read: a query, and the grid run's plan. Not the agent
# acting, and the request store reads them to describe each request.
_POSTED_READS = ("/queries/", "/app/workflow/grids/plan")


class _AgentActs:
    """Every request is the agent acting, for the experiment's record (FIB-1062).

    Pure ASGI middleware. The mark is a context variable, and a route handler --
    a sync function, which FastAPI runs in a worker thread -- runs with a copy
    of the request's context, so the mark reaches whatever the handler calls.

    An acting call is also the agent's call (`fibsem.ai.agent.calls`), named by
    the sidecar's headers, and once answered is recorded as ``agent_called``
    (FIB-1426): what it did in the operator's words, how it ended, and at which
    level. Not a read, an approved request's call (the request's own events say
    those), a refusal for the agent's permissions (``agent_refused``), a long
    command (its run's events), a POST that only reads (a query), or a route
    that records itself.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        lent = current_call() if drive_is_lent() else None
        if scope["type"] != "http" or scope.get("method") == "GET" or lent:
            with acting(AGENT):
                await self.app(scope, receive, send)
            return
        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers") or []
        }
        answer: dict = {"status": 500, "body": b""}

        async def sending(message) -> None:
            if message["type"] == "http.response.start":
                answer["status"] = message["status"]
            elif message["type"] == "http.response.body" and answer["status"] >= 400:
                answer["body"] += message.get("body", b"")[:65536]
            await send(message)

        call = from_headers(headers)
        call.action = await self._described(scope, call) or call.action
        with acting(AGENT), calling(call):
            call.level = self._level(scope)
            try:
                await self.app(scope, receive, sending)
            finally:
                self._record(scope, call, answer["status"], answer["body"])

    @staticmethod
    async def _described(scope, call) -> Optional[str]:
        """*call* in the card's words, read from the state it changes; None
        without a named tool or a request store, or when describing fails."""
        requests = getattr(getattr(scope.get("app"), "state", None), "requests", None)
        if not call.tool or requests is None:
            return None
        try:
            return await run_in_threadpool(
                requests.describe_call, call.tool, call.arguments, scope.get("path", "")
            )
        except Exception:  # noqa: BLE001 - the plain line stands in
            logging.debug("could not describe an agent call", exc_info=True)
            return None

    @staticmethod
    def _level(scope) -> Optional[str]:
        auth = getattr(getattr(scope.get("app"), "state", None), "auth", None)
        return auth.level.value if auth is not None else None

    @staticmethod
    def _record(scope, call, status: int, body: bytes) -> None:
        """Put *call* on the record as ``agent_called``. Never raises."""
        try:
            path = scope.get("path", "")
            auth = scope["app"].state.auth
            token = Headers(scope=scope).get("authorization", "")
            if (
                not hmac.compare_digest(token, f"Bearer {auth.token}")
                or status in (401, 403)
                or call.run_id is not None
                or call.recorded
                or path.startswith(_RECORDS_ITSELF)
                or any(read in path for read in _POSTED_READS)
            ):
                return
            request = Request(scope)
            payload = action_record(
                call.action or action_name(request),
                ok=status < 400,
                route=f"{scope.get('method')} {path}",
                status=status,
                level=call.level,
            )
            if call.tool:
                payload["tool"] = call.tool
            if call.arguments:
                payload["arguments"] = call.arguments
            if status >= 400:
                try:
                    detail = json.loads(body or b"{}").get("detail")
                except (ValueError, AttributeError):
                    detail = None
                if isinstance(detail, dict) and detail.get("error_type"):
                    payload["error_type"] = detail["error_type"]
                # Why, in the route's words: an exception's name is not a reason.
                message = detail.get("message") if isinstance(detail, dict) else detail
                if isinstance(message, str) and message.strip():
                    payload["message"] = message.strip()
            scope["app"].state.microscope.record_event(ACTION, payload)
        except Exception:  # noqa: BLE001 - recording must not change the answer
            logging.debug("could not record an agent call", exc_info=True)


def build_server(
    microscope: FibsemMicroscope,
    app_context=None,
    auth: Optional[AuthConfig] = None,
    serve_microscope: bool = True,
    services: Optional[Mapping[str, Any]] = None,
) -> FastAPI:
    """Build the server app around an already-connected microscope.

    ``auth`` defaults to a generated token at Watch.
    ``app_context`` is the application's view (AutoLamellaAppContext in practice);
    passing one mounts the ``/app`` routes and flips ``routers.app`` in /capabilities.

    ``serve_microscope`` is the App Server's microscope setting (FIB-1035):
    every route under ``/microscope``, reads included, since a read reaches the
    instrument too. False leaves each answering 404 ``microscope_off``, and
    /capabilities reports ``routers.microscope`` off. The App Server passes its
    preference, which is off by default.

    ``services`` are the analysis services (`fibsem.services.analysis`), served
    under ``/services`` whatever ``serve_microscope`` is, since they use no
    device: by default a new copy of each.
    """
    if auth is None:
        auth = AuthConfig.generate()

    app = FastAPI(title="fibsem server", version=API_VERSION)
    app.add_middleware(_AgentActs)
    app.state.auth = auth
    app.state.microscope = microscope
    if services is None:
        from fibsem.services.analysis import analysis_services

        services = analysis_services()
    app.state.services = services

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
            "routers": {
                "microscope": serve_microscope,
                "services": True,
                "app": app_context is not None,
            },
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
    refuse = _refuse(app_context)
    # The analysis services read saved data and use no device, so they are
    # served with or without the microscope: reads and queries at Watch, a
    # command at Drive with no control taken, and the same allowlist.
    app.include_router(
        build_device_router(
            lambda: services,
            read=[Depends(_watch)],
            write=[Depends(_drive)],
            command=[Depends(_drive)],
            prefix=SERVICES,
            refuse=refuse,
            noun="service",
        )
    )
    if serve_microscope:
        app.include_router(whole)
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
                long_commands=True,
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
        # Suggesting on a pending proposal: Assist and Drive, with no request,
        # since the operator's approval is what decides.
        app.include_router(
            build_app_suggest_router(app_context), dependencies=[Depends(_assist)]
        )
        # The agent's requests (FIB-1283): actions the operator approves in the
        # app, run through the routes above. The app decides them through
        # app.state.requests.
        mounted = {"base", "app", "services"} | (
            {"microscope"} if serve_microscope else set()
        )
        app.state.requests = RequestStore(
            app,
            auth,
            frozenset(mounted),
            emit=getattr(app_context, "record_event", None),
            describe=getattr(app_context, "describe_action", None),
            check=getattr(app_context, "check_action", None),
            not_allowed=_not_allowed(microscope) if serve_microscope else None,
            experiment_path=lambda: getattr(app_context, "experiment_path", None),
            experiment=getattr(app_context, "recording_experiment", None),
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
