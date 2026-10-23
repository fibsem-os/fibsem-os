"""fibsem-mcp: the MCP sidecar that connects an agent to a fibsem server.

Runs on the agent host (the machine running Claude Code / Claude Desktop) and
proxies MCP tool calls to a fibsem server over HTTP with a bearer token. It
owns no microscope connection and enforces nothing itself — the server is the
security boundary. It registers every tool the server's mounted routers serve,
whatever the agent's level: the level changes mid-session, and the server's
403 says what is needed (FIB-1035).

Connection resolution, in order: ``--url``/``--token`` arguments, the
``FIBSEM_SERVER_URL``/``FIBSEM_SERVER_TOKEN`` environment variables, the token
``fibsem-mcp pair`` kept, then the same-machine discovery file a running server
maintains.

Register with Claude Code:

    claude mcp add -s user fibsem -- fibsem-mcp

From another computer, pair once with the code Connect an agent shows (the url
defaults to the SSH tunnel's end, http://localhost:8001):

    fibsem-mcp pair 1234 [--url http://microscope-pc:8001]

Requires the [mcp] extra (Python 3.10+): pip install fibsem[mcp]

``fibsem-mcp watch`` prints a line per app event worth waking an agent for,
and needs nothing beyond fibsem itself (see :mod:`fibsem.ai.agent.mcp.watch`).
"""

import argparse
import base64
import json
import math
import os
import re
import sys
from contextvars import ContextVar
from typing import Annotated, Any, Dict, List, Optional, Tuple
from urllib.parse import quote, unquote

from fibsem.ai.agent.calls import headers_for
from fibsem.ai.agent.catalog import (
    CATALOG,
    EVIDENCE_PARAM,
    tools_for_capabilities,
)
from fibsem.client import tools
from fibsem.client.transport import ServerUnreachable, Transport

# How long start-up looks for the server before answering the MCP client anyway:
# a client gives a server only so long to start (FIB-1416).
_PROBE_S = 2.0


_NO_SERVER_MSG = (
    "fibsem-mcp: no server found. Pass --url and --token, set "
    "FIBSEM_SERVER_URL/FIBSEM_SERVER_TOKEN, or start a fibsem server on "
    "this machine (it writes ~/.fibsem/app-server.json)."
)


def _product(name, version):
    """One User-Agent product token, ``name/version``, with anything a token
    can't carry made a hyphen."""

    def clean(text):
        return re.sub(r"[^A-Za-z0-9!#$%&'*+.^_`|~-]+", "-", str(text)).strip("-")

    name = clean(name) or "unknown"
    version = clean(version)
    return f"{name}/{version}" if version else name


def _user_agent(client_info=None):
    """``fibsem-mcp/<version>``, then the MCP client it serves when known --
    how the app names the agent beside "last heard"."""
    from fibsem import __version__

    agent = _product("fibsem-mcp", __version__)
    name = getattr(client_info, "name", None)
    if name:
        agent += " " + _product(name, getattr(client_info, "version", None))
    return agent


def _connect(url, token, user_agent=None):
    """A connection to the server, found as `fibsem.client.transport.resolve`
    finds it, or None when none is known (yet). It follows a restarted
    server itself."""
    try:
        return Transport(url, token, user_agent=user_agent or _user_agent())
    except ServerUnreachable:
        return None


def _connect_with_retry(
    url, token, wait_s, sleep_fn=None, connect=_connect, user_agent=None
):
    """Find and reach a server, retrying until the deadline.

    MCP clients launch the sidecar at session start, often before the human has
    started the server — dying instantly turns a harmless ordering mistake into
    a CONNECTION_CLOSED (learned in the very first live session). So: keep
    looking (the discovery file appears when the server starts) and re-trying
    /capabilities until ``wait_s`` runs out.

    Returns (transport, capabilities). Raises `ServerUnreachable` with guidance
    on timeout. *user_agent* names the MCP client when it has said who it is,
    so an ask to connect names it too.
    """
    import time

    sleep_fn = sleep_fn or time.sleep
    deadline = time.monotonic() + wait_s
    waiting_reported = False
    while True:
        transport = (
            connect(url, token, user_agent=user_agent)
            if user_agent
            else connect(url, token)
        )
        if transport is not None:
            capabilities, err = _call(transport, "GET", "/capabilities")
            if err is None:
                return transport, capabilities
            transport.close()
            if "connect_waiting" in err or "connect_denied" in err:
                # The operator hasn't allowed this agent: say so, don't retry.
                raise ServerUnreachable(f"fibsem-mcp: {err}")
        if time.monotonic() >= deadline:
            if transport is None:
                raise ServerUnreachable(_NO_SERVER_MSG)
            raise ServerUnreachable(
                f"fibsem-mcp: cannot reach the server at {transport.url} — "
                "is it running, and is the token current?"
            )
        if not waiting_reported:
            print(
                f"fibsem-mcp: waiting up to {wait_s:.0f}s for a fibsem server...",
                file=sys.stderr,
            )
            waiting_reported = True
        sleep_fn(1.0)


class _LazyClient:
    """The server's connection, made on the first tool call rather than at
    start-up (FIB-1416).

    An MCP client gives a server only so long to answer its handshake, and the
    App Server may be slow to answer, or not started yet, while the client
    launches all its servers at once. So the sidecar answers the handshake
    first, and the wait for a server (``--wait``) happens in the first call,
    whose answer says what was wrong if none turns up.
    """

    def __init__(self, connect):
        import threading

        self._connect = connect
        self._client = None
        self._user_agent = None
        self._lock = threading.Lock()

    def _connected(self):
        with self._lock:
            if self._client is None:
                client, capabilities = self._connect()
                _say_connected(client.url, capabilities)
                if self._user_agent is not None:
                    client.user_agent = self._user_agent
                self._client = client
            return self._client

    @property
    def named_user_agent(self) -> Optional[str]:
        """The User-Agent naming the MCP client, once it has said who it is."""
        return self._user_agent

    @property
    def connected(self) -> bool:
        """Whether a server was found yet, so a call would not wait for one."""
        return self._client is not None

    def request(self, method, path, **kwargs):
        return self._connected().request(method, path, **kwargs)

    def set_user_agent(self, user_agent: str) -> None:
        with self._lock:
            self._user_agent = user_agent
            if self._client is not None:
                self._client.user_agent = user_agent

    def close(self):
        if self._client is not None:
            self._client.close()


def _connected_lines(base_url, capabilities):
    """What ``--check`` and the first call print: where the sidecar connected,
    the agent level when the server says one, and a warning when the app runs
    other ``fibsem`` code than this sidecar (FIB-1429)."""
    from fibsem.server.discovery import code_mismatch

    level = capabilities.get("level")
    line = f"fibsem-mcp: connected to {base_url}"
    lines = [f"{line} (agent level {level})" if level else line]
    warning = code_mismatch(capabilities, "fibsem-mcp")
    if warning:
        lines.append(warning)
    return lines


def _say_connected(base_url, capabilities):
    for line in _connected_lines(base_url, capabilities):
        print(line, file=sys.stderr)


def _probe(url, token, connect=_connect):
    """The server's /capabilities if it answers within ``_PROBE_S``, so the
    tools can match it; None otherwise, and the sidecar starts anyway."""
    transport = connect(url, token)
    if transport is None:
        return None
    # Asking to connect waits on the operator: the first call asks.
    transport.asks = False
    try:
        resp = transport.request("GET", "/capabilities", timeout=_PROBE_S)
        return resp.json() if resp.status_code == 200 else None
    except Exception:  # noqa: BLE001 - refused, timed out: start without it
        return None
    finally:
        transport.close()


# The tool running and its arguments, for the server's words (`_flag_failures`).
_calling: "ContextVar[Optional[Tuple[str, dict]]]" = ContextVar(
    "fibsem_mcp_calling", default=None
)

# The evidence the running tool's call cites, by id, once the sidecar has kept
# it (`_with_evidence`).
_citing: "ContextVar[Optional[str]]" = ContextVar("fibsem_mcp_citing", default=None)

# fibsem.structures.MillingState by value, without importing it: the sidecar
# stays light. tests/server/test_sidecar.py pins the two in step.
_MILLING_STATES = ("IDLE", "RUNNING", "STOPPING", "PAUSED", "ERROR", "UNKNOWN")


class _Failure(str):
    """An error answer, as text: a refusal (4xx, "refused (403): ...") or the
    server's own error (5xx, "failed (500): ..."; unlike a refusal, the call may
    have done part of its work: a stage move that failed partway has moved). A
    tool that answers one is flagged an error to the MCP client, with this text
    (`_flag_failures`), so the agent does not read a refusal as a result."""


def _pop_previews(data) -> List[bytes]:
    """Take every base64 preview out of *data*, in place, and answer the JPEGs.

    Each preview's place in the JSON keeps ``image_index``, its position among
    the JPEGs answered.
    """
    images = []

    def walk(node):
        if isinstance(node, dict):
            b64 = node.pop("image_b64_jpeg", None)
            if isinstance(b64, str):
                node["image_index"] = len(images)
                images.append(base64.b64decode(b64))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(data)
    return images


EXPERIMENT_URI = "experiment://"


def _with_blobs(content):
    """An experiment resource's JSON *content*, its previews moved out into
    image/jpeg contents after it (as :func:`_pop_previews` places them)."""
    from mcp.server.lowlevel.helper_types import ReadResourceContents

    data = json.loads(content.content)
    images = _pop_previews(data)
    yield ReadResourceContents(content=json.dumps(data), mime_type=content.mime_type)
    for jpeg in images:
        yield ReadResourceContents(content=jpeg, mime_type="image/jpeg")


def _refusal(resp):
    try:
        detail = resp.json().get("detail", resp.text)
    except (ValueError, AttributeError):
        detail = resp.text
    said = "failed" if resp.status_code >= 500 else "refused"
    return _Failure(f"{said} ({resp.status_code}): {json.dumps(detail)}")


def _call(client, method, path, payload=None):
    """Call the server; refusals come back as readable text, not exceptions.

    Agents act sensibly on a structured refusal string; a raw traceback for a
    403 teaches them nothing. Both come back as a `_Failure`, flagged an error
    with that text; a server error says "failed", not "refused", because the
    call may have done part of its work.
    """
    # A read's arguments are its query, as the FibsemClient sends them.
    kwargs = {"params": payload} if method == "GET" and payload else {"json": payload}
    tool = _calling.get()
    if tool is not None and method != "GET":
        # The server describes the call for the operator from these (FIB-1426).
        kwargs["headers"] = headers_for(*tool, evidence=_citing.get())
    try:
        resp = client.request(method, path, **kwargs)
    except ServerUnreachable as exc:
        return None, _Failure(str(exc))
    return _answer(resp)


def _answer(resp):
    """A response as (data, err): its JSON, or its refusal as `_Failure` text."""
    if resp.status_code >= 400:
        return None, _refusal(resp)
    try:
        return resp.json(), None
    except ValueError:
        # Data no tool asked for as a preview: nothing an agent can read as text.
        return None, (
            f"answered {len(resp.content)} bytes that are not JSON; "
            "use beam_acquire_image to see images"
        )


def _name_the_client(mcp, client):
    """Once the MCP client has said who it is (clientInfo, in initialize),
    name it in the User-Agent of every request after, so the app can show
    which agent it is hearing from.

    Middleware is provisional in the mcp SDK; if it is missing or changes
    shape, the sidecar still works and the app shows "Agent"."""
    middleware = getattr(mcp, "middleware", None)
    if not isinstance(middleware, list):
        return
    named = []

    async def name_client(ctx, call_next):
        if not named:
            try:
                info = ctx.session.client_params.client_info
            except AttributeError:
                info = None
            if info is not None and hasattr(client, "set_user_agent"):
                client.set_user_agent(_user_agent(info))
                named.append(True)
        return await call_next(ctx)

    middleware.append(name_client)


def build_sidecar(client, capabilities):
    """Build the MCP server over an httpx-compatible client.

    ``capabilities`` is the full /capabilities payload; a tool registers when
    its router is mounted (an app tool exists only when the server hosts an
    application), whatever the agent's level. The server enforces the level.
    None (no server answered at start-up) registers every tool.
    """
    from mcp.server.mcpserver import Image, MCPServer
    from mcp.types import ToolAnnotations

    offered = {
        t.name
        for t in (
            CATALOG if capabilities is None else tools_for_capabilities(capabilities)
        )
    }
    by_name = {t.name: t for t in CATALOG}
    # The open experiment's files as resources, beside the tools that read them.
    files_offered = "experiment_read_file" in offered

    class Sidecar(MCPServer):
        """The MCP server: resources/list adds the experiment's top level, and
        an experiment resource's previews are read as images."""

        async def list_resources(self):
            listed = await super().list_resources()
            if not files_offered:
                return listed
            import anyio

            return listed + await anyio.to_thread.run_sync(_experiment_resources)

        async def read_resource(self, uri, context=None):
            contents = await super().read_resource(uri, context)
            if not str(uri).startswith(EXPERIMENT_URI):
                return contents
            return [part for content in contents for part in _with_blobs(content)]

    mcp = Sidecar("fibsem")
    _name_the_client(mcp, client)

    # The tool's name is positional-only: a tool may take an argument called
    # `name` (an overview's file name) and pass it on in **arguments.
    def _route(tool_name, /, **arguments):
        """Call a tool's route as the catalog maps its arguments: (data, err)."""
        try:
            resp = _send(tool_name, arguments)
        except ValueError as exc:
            # Refused before it reached the server: an error, as a 422 would be.
            return None, _Failure(f"refused: {exc}")
        except ServerUnreachable as exc:
            return None, _Failure(str(exc))
        return _answer(resp)

    def _send(tool_name, arguments):
        """A tool's call (`tools.send`): its response, the headers naming the
        agent's own call (FIB-1426)."""
        tool = _calling.get()
        headers = headers_for(*tool, evidence=_citing.get()) if tool else {}
        return tools.send(client, tool_name, arguments, headers=headers)

    def _tool(tool_name, /, **arguments):
        """A tool's answer, or its refusal."""
        data, err = _route(tool_name, **arguments)
        return err if err else data

    def _routed(tool_name, key=None, /, **arguments):
        """A routed tool's answer, its route's result under *key* when given; an
        image in it (a command answered with a preview) as MCP image content."""
        data, err = _route(tool_name, **arguments)
        if err:
            return err
        return _with_images({key: data["result"]} if key else data)

    def _with_images(data):
        """Move every base64 preview in *data* out into MCP image content.

        An image as base64 text is unreadable to the agent and can overflow a
        tool result (one ran to 63k characters). Each preview's place in the JSON keeps ``image_index``,
        its position among the images that follow.
        """
        images = [Image(data=jpeg, format="jpeg") for jpeg in _pop_previews(data)]
        if not images:
            return data
        return [json.dumps(data), *images]

    # Explicit signatures so FastMCP derives a real parameter schema per tool.
    def server_get_capabilities():
        return _tool("server_get_capabilities")

    def server_set_constraints(constraints: List[Dict[str, Any]]):
        return _tool("server_set_constraints", constraints=constraints)

    def server_set_checklist(checklist: List[Dict[str, Any]]):
        return _tool("server_set_checklist", checklist=checklist)

    def server_report_session(
        session_id: Optional[str] = None,
        model: Optional[str] = None,
        goal: Optional[str] = None,
        budget_usd: Optional[float] = None,
        spent_usd: Optional[float] = None,
        turns: Optional[int] = None,
        ended: Optional[bool] = None,
        reason: Optional[str] = None,
    ):
        return _tool(
            "server_report_session",
            **_given(
                session_id=session_id,
                model=model,
                goal=goal,
                budget_usd=budget_usd,
                spent_usd=spent_usd,
                turns=turns,
                ended=ended,
                reason=reason,
            ),
        )

    def microscope_get_info():
        return _tool("microscope_get_info")

    def stage_get_position():
        data, err = _route("stage_get_position")
        return err if err else {"position": data["value"]}

    def _service_call(name, key):
        """A stage movement query as a named tool: *key* holds its result."""
        data, err = _route(name)
        return err if err else {key: data["result"]}

    def stage_get_orientation():
        return _service_call("stage_get_orientation", "orientation")

    def microscope_get_state():
        return _tool("microscope_get_state")

    def stage_get_current_milling_angle():
        return _service_call("stage_get_current_milling_angle", "milling_angle_deg")

    def stage_get_geometry():
        return _service_call("stage_get_geometry", "geometry")

    def milling_get_state():
        data, err = _route("milling_get_state")
        if err:
            return err
        # The wire carries the enum's number; the agent reads its name.
        value = data["value"]
        names = _MILLING_STATES
        return {
            "state": names[value]
            if isinstance(value, int) and 0 <= value < len(names)
            else value
        }

    def milling_estimate_time(stages: Optional[list] = None):
        # Answers seconds, milling_current and warning itself.
        data, err = _route("milling_estimate_time", stages=stages)
        return err if err else data["result"]

    def milling_preview_patterns(image_path: str, stages: Optional[list] = None):
        return _routed("milling_preview_patterns", image_path=image_path, stages=stages)

    def alignment_align_to_reference(
        reference_path: str,
        steps: int = 3,
        subsystem: str = "beam-shift",
        use_autofocus: bool = False,
        method: str = "cross-correlation",
        validate: bool = True,
        use_autocontrast: bool = False,
        path: Optional[str] = None,
    ):
        return _routed(
            "alignment_align_to_reference",
            "run",
            reference_path=reference_path,
            steps=steps,
            subsystem=subsystem,
            use_autofocus=use_autofocus,
            method=method,
            validate=validate,
            use_autocontrast=use_autocontrast,
            path=path,
        )

    def alignment_compare_methods(
        reference_path: str, image_path: str, agreement_px: float = 2.0
    ):
        return _routed(
            "alignment_compare_methods",
            "comparison",
            reference_path=reference_path,
            image_path=image_path,
            agreement_px=agreement_px,
        )

    def milling_check_stages(stages: list, aspect: float = 2 / 3):
        return _routed("milling_check_stages", "stages", stages=stages, aspect=aspect)

    def milling_list_options():
        return _routed("milling_list_options", "options")

    def alignment_measure_shift(
        reference_path: str, image_path: str, method: str = "cross-correlation"
    ):
        return _routed(
            "alignment_measure_shift",
            "shift",
            reference_path=reference_path,
            image_path=image_path,
            method=method,
        )

    def alignment_check_coincidence(
        x: float = 0.0,
        y: float = 0.0,
        hfw: float = 150e-6,
        autocontrast: bool = True,
        path: Optional[str] = None,
        beam: str = "ELECTRON",
    ):
        return _routed(
            "alignment_check_coincidence",
            "coincidence",
            x=x,
            y=y,
            hfw=hfw,
            autocontrast=autocontrast,
            path=path,
            beam=beam,
        )

    def alignment_measure_coincidence(
        sem_path: str,
        fib_path: str,
        x: float = 0.0,
        y: float = 0.0,
        beam: str = "ELECTRON",
    ):
        return _routed(
            "alignment_measure_coincidence",
            "coincidence",
            sem_path=sem_path,
            fib_path=fib_path,
            x=x,
            y=y,
            beam=beam,
        )

    def correlation_correlate(
        fib_image_path: str,
        fm_image_path: str,
        fib_points: list,
        fm_points: list,
        poi_points: list,
        lamella_path: str = "",
        fm_surface_point: Optional[List[float]] = None,
        fib_surface_point: Optional[List[float]] = None,
        ri: Optional[Dict[str, float]] = None,
        ri_factor: Optional[float] = None,
    ):
        arguments = dict(
            fib_image_path=fib_image_path,
            fm_image_path=fm_image_path,
            fib_points=fib_points,
            fm_points=fm_points,
            poi_points=poi_points,
        )
        optional = dict(
            lamella_path=lamella_path or None,
            fm_surface_point=fm_surface_point,
            fib_surface_point=fib_surface_point,
            ri=ri,
            ri_factor=ri_factor,
        )
        arguments.update({k: v for k, v in optional.items() if v is not None})
        return _routed("correlation_correlate", "correlation", **arguments)

    def correlation_find_spot_burns(lamella_path: str, fib_image_path: str = ""):
        arguments = dict(lamella_path=lamella_path)
        if fib_image_path:
            arguments["fib_image_path"] = fib_image_path
        return _routed("correlation_find_spot_burns", "spot_burns", **arguments)

    def correlation_refine_point(
        image_path: str,
        x: float,
        y: float,
        z: float = 0.0,
        beam: str = "fib",
        method: str = "",
        channel: int = 0,
    ):
        return _routed(
            "correlation_refine_point",
            "point",
            image_path=image_path,
            x=x,
            y=y,
            z=z,
            beam=beam,
            method=method,
            channel=channel,
        )

    def correlation_predict(run_path: str, fm_points: list):
        return _routed(
            "correlation_predict", "prediction", run_path=run_path, fm_points=fm_points
        )

    def milling_stop(reason: str):
        data, err = _route("milling_stop", reason=reason)
        return err if err else {"status": "ok"}

    def milling_end(reason: str):
        data, err = _route("milling_end", reason=reason)
        return err if err else {"status": "ok"}

    def milling_pause():
        data, err = _route("milling_pause")
        return err if err else {"status": "ok"}

    def milling_resume():
        data, err = _route("milling_resume")
        return err if err else {"status": "ok"}

    def milling_follow_readings(since: int = 0, timeout: float = 0.0):
        return _routed("milling_follow_readings", since=since, timeout=timeout)

    def beam_acquire_image(
        beam_type: str = "ELECTRON",
        image_settings: Optional[dict] = None,
        path: Optional[str] = None,
        filename: Optional[str] = None,
    ):
        data, err = _route(
            "beam_acquire_image",
            beam_type=beam_type,
            image_settings=image_settings,
            path=path,
            filename=filename,
        )
        return err if err else _with_images(data["result"])

    def stage_move_relative(dx: float = 0.0, dy: float = 0.0, dz: float = 0.0):
        return _routed("stage_move_relative", "position", dx=dx, dy=dy, dz=dz)

    def stage_stable_move(
        dx: float, dy: float, beam_type: str = "ELECTRON", static_wd: bool = False
    ):
        return _routed(
            "stage_stable_move",
            "position",
            dx=dx,
            dy=dy,
            beam_type=beam_type,
            static_wd=static_wd,
        )

    def stage_get_saved_positions():
        return _service_call("stage_get_saved_positions", "saved_positions")

    def stage_save_position(name: str):
        return _routed("stage_save_position", "position", name=name)

    def stage_move_to_saved_position(name: str):
        return _routed("stage_move_to_saved_position", "position", name=name)

    def stage_remove_saved_position(name: str):
        return _routed("stage_remove_saved_position", name=name)

    def stage_move_to_device(device: str, orientation: Optional[str] = None):
        return _routed("stage_move_to_device", device=device, orientation=orientation)

    def stage_fm_stable_move(dx: float, dy: float):
        return _routed("stage_fm_stable_move", "position", dx=dx, dy=dy)

    def stage_vertical_move(
        dy: float,
        dx: float = 0.0,
        beam_type: Optional[str] = None,
        relaxation: Optional[float] = None,
    ):
        return _routed(
            "stage_vertical_move",
            "position",
            dy=dy,
            dx=dx,
            beam_type=beam_type,
            relaxation=relaxation,
        )

    def stage_move_to_orientation(orientation: str):
        return _routed("stage_move_to_orientation", "position", orientation=orientation)

    def stage_move_absolute(x: float, y: float, z: float, r: float, t: float):
        return _routed("stage_move_absolute", "position", x=x, y=y, z=z, r=r, t=t)

    def stage_move_to_milling_angle(
        milling_angle_deg: float, rotation_deg: Optional[float] = None
    ):
        return _routed(
            "stage_move_to_milling_angle",
            "success",
            milling_angle_deg=milling_angle_deg,
            rotation_deg=rotation_deg,
        )

    def device_list():
        return _tool("device_list")

    def device_get_parameter(device: str, parameter: str):
        return _routed("device_get_parameter", device=device, parameter=parameter)

    def device_set_parameter(device: str, parameter: str, value: Any):
        return _routed(
            "device_set_parameter", device=device, parameter=parameter, value=value
        )

    def device_call_command(
        device: str,
        command: str,
        kwargs: Optional[dict] = None,
        reason: Optional[str] = None,
    ):
        return _routed(
            "device_call_command",
            device=device,
            command=command,
            kwargs=kwargs,
            reason=reason,
        )

    def device_call_query(device: str, query: str, kwargs: Optional[dict] = None):
        return _routed("device_call_query", device=device, query=query, kwargs=kwargs)

    def microscope_set_state(microscope_state: dict):
        return _routed("microscope_set_state", microscope_state=dict(microscope_state))

    def service_list():
        return _tool("service_list")

    def service_get_parameter(service: str, parameter: str):
        return _routed("service_get_parameter", service=service, parameter=parameter)

    def service_set_parameter(service: str, parameter: str, value: Any):
        return _routed(
            "service_set_parameter", service=service, parameter=parameter, value=value
        )

    def service_call_command(
        service: str,
        command: str,
        kwargs: Optional[dict] = None,
        reason: Optional[str] = None,
    ):
        return _routed(
            "service_call_command",
            service=service,
            command=command,
            kwargs=kwargs,
            reason=reason,
        )

    def service_wait_for_run(
        service: str, run_id: Optional[str] = None, wait: float = 60.0
    ):
        data, err = _route(
            "service_wait_for_run", service=service, run_id=run_id, wait=wait
        )
        return err if err else _with_images(data)

    def service_call_query(service: str, query: str, kwargs: Optional[dict] = None):
        return _routed(
            "service_call_query", service=service, query=query, kwargs=kwargs
        )

    def beam_auto_contrast(
        beam_type: str = "ELECTRON", reduced_area: Optional[dict] = None
    ):
        return _routed(
            "beam_auto_contrast", beam_type=beam_type, reduced_area=reduced_area
        )

    def beam_auto_focus(
        beam_type: str = "ELECTRON",
        method: str = "sweep",
        hfw: Optional[float] = None,
        settings: Optional[dict] = None,
        path: Optional[str] = None,
        name: Optional[str] = None,
    ):
        return _routed(
            "beam_auto_focus",
            "result",
            beam_type=beam_type,
            method=method,
            hfw=hfw,
            settings=settings,
            path=path,
            name=name,
        )

    def fm_auto_focus(
        channel_name: Optional[str] = None,
        settings: Optional[dict] = None,
        channel: Optional[dict] = None,
        path: Optional[str] = None,
        name: str = "autofocus",
    ):
        return _routed(
            "fm_auto_focus",
            "run",
            channel_name=channel_name,
            settings=settings,
            channel=channel,
            path=path,
            name=name,
        )

    def beam_acquire_overview(
        settings: dict,
        path: str,
        name: str = "overview",
        centre_position: Optional[dict] = None,
    ):
        return _routed(
            "beam_acquire_overview",
            "run",
            settings=settings,
            path=path,
            name=name,
            centre_position=centre_position,
        )

    def beam_estimate_overview(settings: dict):
        return _routed("beam_estimate_overview", "estimate", settings=settings)

    def beam_acquire_focus_stack(
        image_settings: dict, path: str, n_steps: int = 3, auto_focus: bool = True
    ):
        return _routed(
            "beam_acquire_focus_stack",
            "run",
            image_settings=image_settings,
            path=path,
            n_steps=n_steps,
            auto_focus=auto_focus,
        )

    def beam_turn_on(beam_type: str):
        return _routed("beam_turn_on", beam_type=beam_type)

    def beam_turn_off(beam_type: str):
        return _routed("beam_turn_off", beam_type=beam_type)

    def fm_insert_objective(tilt: Optional[float] = None):
        return _routed("fm_insert_objective", tilt=tilt)

    def fm_retract_objective():
        return _routed("fm_retract_objective")

    def fm_save_focus_position():
        return _routed("fm_save_focus_position")

    def fm_acquire_image(
        channels: List[dict],
        path: str,
        name: str = "image",
        zparams: Optional[dict] = None,
    ):
        return _routed(
            "fm_acquire_image",
            "run",
            channels=channels,
            path=path,
            name=name,
            zparams=zparams,
        )

    def grid_get_inventory():
        return _routed("grid_get_inventory", "inventory")

    def grid_get_current():
        return _routed("grid_get_current", "current")

    def grid_get_holder_slots():
        return _routed("grid_get_holder_slots", "slots")

    def grid_run_inventory():
        return _routed("grid_run_inventory", "run")

    def grid_load(grid_name: str):
        return _routed("grid_load", "run", grid_name=grid_name)

    def grid_unload():
        return _routed("grid_unload", "run")

    def grid_move_to(grid_name: str):
        return _routed("grid_move_to", "position", grid_name=grid_name)

    def grid_move_to_slot(slot_name: str):
        return _routed("grid_move_to_slot", "position", slot_name=slot_name)

    def milling_run_stage(stage: dict, path: Optional[str] = None):
        return _routed("milling_run_stage", "run", stage=stage, path=path)

    def milling_run_spot_burn(settings: dict):
        return _routed("milling_run_spot_burn", "run", settings=settings)

    def milling_estimate_spot_burn(settings: dict):
        return _routed("milling_estimate_spot_burn", "seconds", settings=settings)

    def coating_deposit(seconds: float):
        return _routed("coating_deposit", "run", seconds=seconds)

    def coating_sputter(
        seconds: float,
        current: Optional[float] = None,
        grids: Optional[List[int]] = None,
    ):
        return _routed(
            "coating_sputter", "run", seconds=seconds, current=current, grids=grids
        )

    def milling_run_task(config: dict, path: Optional[str] = None):
        return _routed("milling_run_task", "run", config=config, path=path)

    def fm_acquire_overview(
        channels: List[dict],
        parameters: dict,
        path: str,
        name: str = "overview",
        zparams: Optional[dict] = None,
        autofocus_settings: Optional[dict] = None,
        centre_position: Optional[dict] = None,
    ):
        return _routed(
            "fm_acquire_overview",
            "run",
            channels=channels,
            parameters=parameters,
            path=path,
            name=name,
            zparams=zparams,
            autofocus_settings=autofocus_settings,
            centre_position=centre_position,
        )

    def fm_estimate_overview(
        channels: List[dict],
        parameters: dict,
        zparams: Optional[dict] = None,
        autofocus_settings: Optional[dict] = None,
    ):
        return _routed(
            "fm_estimate_overview",
            "estimate",
            channels=channels,
            parameters=parameters,
            zparams=zparams,
            autofocus_settings=autofocus_settings,
        )

    def beam_auto_brightness_contrast(
        beam_type: str = "ELECTRON",
        settings: Optional[dict] = None,
        path: Optional[str] = None,
        name: Optional[str] = None,
    ):
        return _routed(
            "beam_auto_brightness_contrast",
            "result",
            beam_type=beam_type,
            settings=settings,
            path=path,
            name=name,
        )

    def _given(**params):
        """The *params* that are given (not None): a body that leaves out the rest."""
        return {k: v for k, v in params.items() if v is not None}

    def app_get_status():
        return _tool("app_get_status")

    def workflow_get_queue():
        return _tool("workflow_get_queue")

    def item_list():
        return _tool("item_list")

    def experiment_get_task_history():
        return _tool("experiment_get_task_history")

    def workflow_get_summary():
        return _tool("workflow_get_summary")

    def protocol_get():
        return _tool("protocol_get")

    def app_get_log(
        task_id: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        level: Optional[str] = None,
        lines: int = 200,
    ):
        return _tool(
            "app_get_log",
            task_id=task_id,
            since=since,
            until=until,
            level=level,
            lines=lines,
        )

    def item_list_outputs(item_name: str):
        return _routed("item_list_outputs", item_name=item_name)

    def item_get_history(
        item_name: str,
        show: str = "all",
        status: str = "all",
        task: Optional[str] = None,
        decided_by: str = "anyone",
        order: str = "oldest",
    ):
        arguments = dict(show=show, status=status, decided_by=decided_by, order=order)
        if task:
            arguments["task"] = task
        return _routed("item_get_history", item_name=item_name, **arguments)

    def item_get_stack(item_name: str, role: str):
        return _routed("item_get_stack", item_name=item_name, role=role)

    def _image_header(resp):
        # The image's scale and state ride on the JPEG answer as a header
        # (app_routes.IMAGE_METADATA_HEADER); absent for a non-fibsem image.
        raw = resp.headers.get("X-Image-Metadata")
        if not raw:
            return {}
        try:
            return {"metadata": json.loads(raw)}
        except ValueError:
            return {}

    def item_get_output_image(item_name: str, filename: str):
        # The route answers image/jpeg for the dashboard's <img>, so this reads
        # the bytes itself rather than through _answer.
        resp = _send(
            "item_get_output_image", dict(item_name=item_name, filename=filename)
        )
        if resp.status_code >= 400:
            return _refusal(resp)
        meta = {"item_name": item_name, "filename": filename, **_image_header(resp)}
        return [json.dumps(meta), Image(data=resp.content, format="jpeg")]

    def item_measure_thickness(
        item_name: str,
        path: str,
        width_um: Optional[float] = None,
        height_um: Optional[float] = None,
    ):
        return _routed(
            "item_measure_thickness",
            item_name=item_name,
            path=path,
            **_given(width_um=width_um, height_um=height_um),
        )

    def experiment_get_image_metadata(path: str):
        return _tool("experiment_get_image_metadata", path=path)

    def experiment_get_image_stage_position(
        path: str,
        x: Optional[float] = None,
        y: Optional[float] = None,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
    ):
        return _tool(
            "experiment_get_image_stage_position", path=path, x=x, y=y, fx=fx, fy=fy
        )

    def experiment_render_markers(path: str, grid_name: Optional[str] = None):
        return _routed(
            "experiment_render_markers", path=path, **_given(grid_name=grid_name)
        )

    def experiment_draw_image(
        path: str,
        marks: Optional[dict] = None,
        crosshair: Optional[bool] = None,
        metadata_bar: Optional[bool] = None,
        channels: Optional[List[str]] = None,
    ):
        return _routed(
            "experiment_draw_image",
            path=path,
            **_given(
                marks=dict(marks) if marks is not None else None,
                crosshair=crosshair,
                metadata_bar=metadata_bar,
                channels=list(channels) if channels is not None else None,
            ),
        )

    def experiment_draw_pair(
        before: str,
        after: str,
        marks: Optional[dict] = None,
        metadata_bar: Optional[bool] = None,
        channels: Optional[List[str]] = None,
    ):
        return _routed(
            "experiment_draw_pair",
            before=before,
            after=after,
            **_given(
                marks=dict(marks) if marks is not None else None,
                metadata_bar=metadata_bar,
                channels=list(channels) if channels is not None else None,
            ),
        )

    def experiment_get_image_pixels(path: str, stage_positions: list):
        return _tool(
            "experiment_get_image_pixels",
            path=path,
            stage_positions=[dict(p) for p in stage_positions],
        )

    def _region(values):
        return None if values is None else ",".join(str(float(v)) for v in values)

    def _read_file(
        path, box=None, fraction=None, z=None, channels=None, experiment=None
    ):
        """The file route's answer for *path*: (data, err)."""
        return _route(
            "experiment_read_file",
            path=str(path),
            box=_region(box),
            fraction=_region(fraction),
            z=z,
            channels=",".join(channels) if channels else None,
            experiment=experiment,
        )

    def experiment_list_files(
        dir: Optional[str] = None, experiment: Optional[str] = None
    ):
        return _tool("experiment_list_files", dir=dir, experiment=experiment)

    def experiment_read_file(
        path: str,
        z: Optional[int] = None,
        channels: Optional[List[str]] = None,
        experiment: Optional[str] = None,
    ):
        data, err = _read_file(path, z=z, channels=channels, experiment=experiment)
        return err if err else _with_images(data)

    def experiment_crop_image(
        path: str,
        box: Optional[list] = None,
        fraction: Optional[list] = None,
        z: Optional[int] = None,
        channels: Optional[List[str]] = None,
        experiment: Optional[str] = None,
    ):
        data, err = _read_file(
            path,
            box=box,
            fraction=fraction,
            z=z,
            channels=channels,
            experiment=experiment,
        )
        return err if err else _with_images(data)

    def experiment_fetch_file(path: str, experiment: Optional[str] = None):
        try:
            resp = _send(
                "experiment_fetch_file",
                dict(path=str(path), raw="true", experiment=experiment),
            )
        except ServerUnreachable as exc:
            return _Failure(str(exc))
        if resp.status_code >= 400:
            return _refusal(resp)
        return _save_fetched(
            unquote(resp.headers.get("X-Experiment", "experiment")),
            unquote(resp.headers.get("X-Experiment-Path", os.path.basename(path))),
            resp.content,
        )

    def _experiment_resources():
        """The experiment's top-level files and folders, as resources: not every
        tile under them, which a read of a folder lists."""
        from mcp.types import Resource

        if capabilities is None and not getattr(client, "connected", True):
            # No server answered at start-up: never wait for one here; the
            # first tool call does.
            return []
        try:
            data, err = _route("experiment_list_files")
        except Exception:  # noqa: BLE001 - no server: a listing with no files
            return []
        if err:
            return []  # no experiment open, or no app
        return [
            Resource(
                uri=EXPERIMENT_URI + quote(entry["path"]),
                name=entry["name"],
                description=f"{entry['kind']} in the open experiment",
                mime_type="application/json",
                size=entry["size"],
            )
            for entry in data["entries"]
        ]

    if files_offered:
        from mcp.server.mcpserver.exceptions import ResourceError

        @mcp.resource(
            EXPERIMENT_URI + "{+path}",
            name="experiment_file",
            title="Experiment file",
            description=by_name["experiment_read_file"].description,
            mime_type="application/json",
        )
        def experiment_file(path: str):
            data, err = _read_file(path)
            if err:
                raise ResourceError(err)
            return data

    def item_get(item_name: str):
        return _routed("item_get", item_name=item_name)

    def grid_list():
        return _tool("grid_list")

    def grid_get(grid_name: str):
        return _routed("grid_get", grid_name=grid_name)

    def grid_get_output_image(grid_name: str, filename: str):
        # A JPEG, like an item's output image.
        resp = _send(
            "grid_get_output_image", dict(grid_name=grid_name, filename=filename)
        )
        if resp.status_code >= 400:
            return _refusal(resp)
        meta = {"grid_name": grid_name, "filename": filename, **_image_header(resp)}
        return [json.dumps(meta), Image(data=resp.content, format="jpeg")]

    def grid_get_stage_position(
        grid_name: str,
        filename: str,
        x: Optional[float] = None,
        y: Optional[float] = None,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
    ):
        return _tool(
            "grid_get_stage_position",
            grid_name=grid_name,
            filename=filename,
            x=x,
            y=y,
            fx=fx,
            fy=fy,
        )

    def grid_get_markers(grid_name: str, filename: str):
        return _routed("grid_get_markers", grid_name=grid_name, filename=filename)

    def protocol_get_task_config(task_name: str):
        return _routed("protocol_get_task_config", task_name=task_name)

    def item_get_task_config(item_name: str, task_name: str):
        return _routed("item_get_task_config", item_name=item_name, task_name=task_name)

    def experiment_add_note(text: str, item_name: Optional[str] = None):
        return _tool("experiment_add_note", text=str(text), item_name=item_name)

    def experiment_message_operator(
        text: str, item_name: Optional[str] = None, reply_to: Optional[str] = None
    ):
        body = {"text": str(text), "item_name": item_name}
        if reply_to:
            body["reply_to"] = str(reply_to)
        return _tool("experiment_message_operator", **body)

    def request_create(actions: list, reason: str):
        body = {"actions": list(actions), "reason": str(reason)}
        return _tool("request_create", **body)

    def request_withdraw(request_id: str, reason: str):
        return _routed("request_withdraw", request_id=request_id, reason=str(reason))

    def request_get(request_id: str):
        # An applied image action carries its preview: an image, not base64 text.
        return _routed("request_get", request_id=request_id)

    def request_list():
        data, err = _route("request_list")
        return err if err else _with_images(data)

    def item_add(
        name: Optional[str] = None,
        stage_position: Optional[dict] = None,
        overview: Optional[dict] = None,
    ):
        return _tool(
            "item_add", name=name, stage_position=stage_position, overview=overview
        )

    def item_remove(item_name: str):
        return _routed("item_remove", item_name=item_name)

    def item_move_to_objective_position(item_name: str):
        return _routed("item_move_to_objective_position", item_name=item_name)

    def item_move_to_pose(item_name: str, pose: str):
        return _routed("item_move_to_pose", item_name=item_name, pose=pose)

    def item_set_pose(item_name: str, pose: str):
        return _routed("item_set_pose", item_name=item_name, pose=pose)

    def item_apply_correlation(item_name: str, run_path: str):
        return _routed("item_apply_correlation", item_name=item_name, run_path=run_path)

    def item_set_objective_position(item_name: str):
        return _routed("item_set_objective_position", item_name=item_name)

    def item_set_objective_position_on_all(item_name: str):
        return _routed("item_set_objective_position_on_all", item_name=item_name)

    def item_set_position(
        item_name: str,
        stage_position: Optional[dict] = None,
        overview: Optional[dict] = None,
    ):
        return _routed(
            "item_set_position",
            item_name=item_name,
            stage_position=stage_position,
            overview=overview,
        )

    def item_apply_protocol(item_name: str, task_names: Optional[List[str]] = None):
        return _routed(
            "item_apply_protocol",
            item_name=item_name,
            task_names=list(task_names) if task_names else None,
        )

    def item_reorder_milling_stages(
        item_name: str, task_name: str, milling_key: str, order: list, version: str
    ):
        return _routed(
            "item_reorder_milling_stages",
            item_name=item_name,
            task_name=task_name,
            milling_key=milling_key,
            order=list(order),
            version=version,
        )

    def protocol_reorder_milling_stages(
        task_name: str, milling_key: str, order: list, version: str
    ):
        return _routed(
            "protocol_reorder_milling_stages",
            task_name=task_name,
            milling_key=milling_key,
            order=list(order),
            version=version,
        )

    def protocol_set_schedule(task_name: str, scheduled_at: Optional[str] = None):
        return _tool(
            "protocol_set_schedule", task_name=task_name, scheduled_at=scheduled_at
        )

    def item_update(item_name: str, patch: dict, version: str):
        return _routed(
            "item_update", item_name=item_name, patch=dict(patch), version=str(version)
        )

    def protocol_update_task_config(task_name: str, patch: dict, version: str):
        return _routed(
            "protocol_update_task_config",
            task_name=task_name,
            patch=dict(patch),
            version=str(version),
        )

    def item_update_task_config(
        item_name: str, task_name: str, patch: dict, version: str
    ):
        return _routed(
            "item_update_task_config",
            item_name=item_name,
            task_name=task_name,
            patch=dict(patch),
            version=str(version),
        )

    def experiment_list_recent():
        return _tool("experiment_list_recent")

    def experiment_list_all(
        instrument: Optional[str] = None,
        operator: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
    ):
        return _tool(
            "experiment_list_all",
            instrument=instrument,
            operator=operator,
            since=since,
            until=until,
        )

    def protocol_list():
        return _tool("protocol_list")

    def experiment_create(
        name: str, protocol: Optional[str] = None, description: Optional[str] = None
    ):
        return _tool(
            "experiment_create", name=name, protocol=protocol, description=description
        )

    def grid_add():
        return _tool("grid_add")

    def grid_rename(grid_name: str, name: str):
        return _routed("grid_rename", grid_name=grid_name, name=name)

    def grid_set_quality(grid_name: str, verdict: str, reason: str = ""):
        return _routed(
            "grid_set_quality", grid_name=grid_name, verdict=verdict, reason=reason
        )

    def grid_set_note(grid_name: str, note: str):
        return _routed("grid_set_note", grid_name=grid_name, note=note)

    def grid_remove(grid_name: str):
        return _routed("grid_remove", grid_name=grid_name)

    def protocol_add_task(task_type: str, task_name: str):
        return _tool("protocol_add_task", task_type=task_type, task_name=task_name)

    def protocol_remove_task(task_name: str):
        return _routed("protocol_remove_task", task_name=task_name)

    def protocol_reorder_tasks(order: list):
        return _tool("protocol_reorder_tasks", order=list(order))

    def protocol_set_task_requirements(
        task_name: str, requires: Optional[list] = None, required: Optional[bool] = None
    ):
        return _routed(
            "protocol_set_task_requirements",
            task_name=task_name,
            **_given(
                requires=None if requires is None else list(requires),
                required=None if required is None else bool(required),
            ),
        )

    def protocol_set_lamella_defaults(
        use_petname: Optional[bool] = None,
        name_prefix: Optional[str] = None,
        poi: Optional[dict] = None,
        alignment_area: Optional[dict] = None,
    ):
        return _tool(
            "protocol_set_lamella_defaults",
            **_given(
                use_petname=use_petname,
                name_prefix=name_prefix,
                poi=poi,
                alignment_area=alignment_area,
            ),
        )

    def protocol_save(name: str, overwrite: bool = False):
        return _tool("protocol_save", name=name, overwrite=bool(overwrite))

    def protocol_get_grid_task_config(task_name: str):
        return _routed("protocol_get_grid_task_config", task_name=task_name)

    def protocol_update_grid_task_config(task_name: str, patch: dict, version: str):
        return _routed(
            "protocol_update_grid_task_config",
            task_name=task_name,
            patch=patch,
            version=version,
        )

    def protocol_reorder_grid_tasks(order: list):
        return _tool("protocol_reorder_grid_tasks", order=list(order))

    def protocol_set_grid_task_requirements(task_name: str, requires: list):
        return _routed(
            "protocol_set_grid_task_requirements",
            task_name=task_name,
            requires=list(requires),
        )

    def experiment_load(path: str):
        return _tool("experiment_load", path=path)

    def protocol_load(protocol: str):
        return _tool("protocol_load", protocol=protocol)

    def app_get_events(since: int = 0, timeout: float = 0.0):
        return _tool("app_get_events", since=since, timeout=timeout)

    def workflow_stop(reason: str):
        return _tool("workflow_stop", reason=reason)

    def workflow_stop_task(item_name: str, task_name: str, reason: str):
        return _tool(
            "workflow_stop_task",
            item_name=item_name,
            task_name=task_name,
            reason=reason,
        )

    def workflow_queue_grids(task_names: list, grid_names: list):
        return _tool(
            "workflow_queue_grids",
            task_names=list(task_names),
            grid_names=list(grid_names),
        )

    def workflow_start(task_names: list, item_names: list = None):  # noqa: RUF013
        payload = {"task_names": list(task_names)}
        if item_names is not None:
            payload["item_names"] = list(item_names)
        return _tool("workflow_start", **payload)

    def workflow_plan(
        task_names: list,
        item_names: list = None,  # noqa: RUF013
        add: bool = False,
        run_next: bool = False,
    ):
        payload = {
            "task_names": list(task_names),
            "add": bool(add),
            "run_next": bool(run_next),
        }
        if item_names is not None:
            payload["item_names"] = list(item_names)
        return _tool("workflow_plan", **payload)

    def workflow_plan_grids(
        task_names: list, grid_names: list = None, screen_all: bool = False
    ):  # noqa: RUF013
        payload = {"task_names": list(task_names), "screen_all": bool(screen_all)}
        if grid_names is not None:
            payload["grid_names"] = list(grid_names)
        return _tool("workflow_plan_grids", **payload)

    def workflow_start_grids(
        task_names: list,
        grid_names: list = None,  # noqa: RUF013
        screen_all: bool = False,
        unload_at_end: bool = False,
        report_at_end: bool = False,
    ):
        payload = {
            "task_names": list(task_names),
            "screen_all": bool(screen_all),
            "unload_at_end": bool(unload_at_end),
            "report_at_end": bool(report_at_end),
        }
        if grid_names is not None:
            payload["grid_names"] = list(grid_names)
        return _tool("workflow_start_grids", **payload)

    def workflow_set_options(
        show_run_summary: Optional[bool] = None,
        turn_beams_off: Optional[bool] = None,
    ):
        payload: dict = {}
        if show_run_summary is not None:
            payload["show_run_summary"] = bool(show_run_summary)
        if turn_beams_off is not None:
            payload["turn_beams_off"] = bool(turn_beams_off)
        return _tool("workflow_set_options", **payload)

    def protocol_set_supervision(task_name: str, supervise: bool):
        return _tool(
            "protocol_set_supervision", task_name=task_name, supervise=bool(supervise)
        )

    def workflow_requeue(item_name: str, task_name: str, front: bool = False):
        return _tool(
            "workflow_requeue",
            item_name=item_name,
            task_name=task_name,
            front=bool(front),
        )

    def workflow_edit_queue(item_name: str, task_name: str, action: str):
        return _tool(
            "workflow_edit_queue",
            item_name=item_name,
            task_name=task_name,
            action=action,
        )

    def app_get_display_images():
        data, err = _route("app_get_display_images")
        if err:
            return err
        # The previews travel as MCP images so the agent sees them directly;
        # everything else (availability, timestamps) rides along as JSON.
        content = []
        for key in ("sem", "fib"):
            entry = data.get(key)
            if isinstance(entry, dict) and "image_b64_jpeg" in entry:
                jpeg = base64.b64decode(entry.pop("image_b64_jpeg"))
                entry["beam"] = key
                content.append(Image(data=jpeg, format="jpeg"))
        content.insert(0, json.dumps(data))
        return content

    def proposal_list():
        data, err = _route("proposal_list")
        return err if err else _with_images(data)

    def proposal_decide(
        item_id: str,
        task_name: str,
        outcome: str,
        proposal_id: str = "",
        values: Optional[dict] = None,
        reason: str = "",
        actor: str = "",
        task_id: str = "",
        action: str = "",
        patch: Optional[dict] = None,
    ):
        body = {
            "item_id": str(item_id),
            "task_name": str(task_name),
            "outcome": str(outcome),
            "reason": str(reason),
            "actor": str(actor),
        }
        # What the decision is on. The id is the one to pass; the run is
        # accepted in its place, for a caller written before proposals had ids.
        if proposal_id:
            body["proposal_id"] = str(proposal_id)
        if task_id:
            body["task_id"] = str(task_id)
        if values is not None:
            body["values"] = dict(values)
        if action:
            body["action"] = str(action)
        if patch is not None:
            body["patch"] = dict(patch)
        return _tool("proposal_decide", **body)

    def proposal_suggest(
        item_id: str,
        task_name: str,
        proposal_id: str,
        reason: str,
        outcome: str = "Confirmed",
        values: Optional[dict] = None,
        patch: Optional[dict] = None,
        action: str = "",
        actor: str = "",
    ):
        body = {
            "item_id": str(item_id),
            "task_name": str(task_name),
            "proposal_id": str(proposal_id),
            "reason": str(reason),
            "outcome": str(outcome),
            "actor": str(actor),
        }
        if values is not None:
            body["values"] = dict(values)
        if patch is not None:
            body["patch"] = dict(patch)
        if action:
            body["action"] = str(action)
        return _tool("proposal_suggest", **body)

    implementations = {
        fn.__name__: fn
        for fn in (
            server_get_capabilities,
            microscope_get_info,
            stage_get_position,
            stage_get_orientation,
            microscope_get_state,
            stage_get_current_milling_angle,
            stage_get_geometry,
            milling_get_state,
            milling_estimate_time,
            milling_preview_patterns,
            alignment_align_to_reference,
            alignment_measure_shift,
            alignment_compare_methods,
            correlation_correlate,
            correlation_refine_point,
            correlation_predict,
            correlation_find_spot_burns,
            milling_check_stages,
            milling_list_options,
            device_call_query,
            microscope_set_state,
            experiment_get_image_stage_position,
            experiment_render_markers,
            experiment_draw_image,
            experiment_draw_pair,
            experiment_get_image_pixels,
            grid_set_quality,
            grid_set_note,
            grid_remove,
            protocol_add_task,
            protocol_remove_task,
            protocol_reorder_tasks,
            protocol_set_task_requirements,
            protocol_set_lamella_defaults,
            protocol_save,
            protocol_get_grid_task_config,
            protocol_update_grid_task_config,
            protocol_reorder_grid_tasks,
            protocol_set_grid_task_requirements,
            workflow_stop_task,
            workflow_queue_grids,
            item_move_to_objective_position,
            alignment_check_coincidence,
            alignment_measure_coincidence,
            milling_stop,
            milling_end,
            milling_pause,
            milling_resume,
            milling_follow_readings,
            beam_acquire_image,
            stage_move_relative,
            stage_stable_move,
            stage_fm_stable_move,
            stage_get_saved_positions,
            stage_save_position,
            stage_move_to_saved_position,
            stage_remove_saved_position,
            stage_move_to_device,
            stage_vertical_move,
            stage_move_to_orientation,
            stage_move_absolute,
            stage_move_to_milling_angle,
            device_list,
            device_get_parameter,
            device_set_parameter,
            device_call_command,
            service_list,
            service_get_parameter,
            service_set_parameter,
            service_call_command,
            service_call_query,
            service_wait_for_run,
            beam_auto_contrast,
            beam_auto_focus,
            fm_auto_focus,
            beam_acquire_overview,
            beam_estimate_overview,
            beam_acquire_focus_stack,
            beam_turn_on,
            beam_turn_off,
            fm_insert_objective,
            fm_retract_objective,
            fm_save_focus_position,
            fm_acquire_image,
            milling_run_stage,
            milling_run_task,
            milling_run_spot_burn,
            milling_estimate_spot_burn,
            coating_deposit,
            coating_sputter,
            fm_acquire_overview,
            fm_estimate_overview,
            beam_auto_brightness_contrast,
            app_get_status,
            workflow_get_queue,
            item_list,
            experiment_get_task_history,
            workflow_get_summary,
            app_get_log,
            protocol_get,
            item_list_outputs,
            item_get_history,
            item_get_stack,
            item_get_output_image,
            item_measure_thickness,
            experiment_get_image_metadata,
            experiment_list_files,
            experiment_read_file,
            experiment_crop_image,
            experiment_fetch_file,
            item_get,
            grid_list,
            grid_get,
            grid_get_output_image,
            grid_get_markers,
            grid_get_stage_position,
            grid_add,
            grid_rename,
            grid_get_inventory,
            grid_get_current,
            grid_get_holder_slots,
            grid_run_inventory,
            grid_load,
            grid_unload,
            grid_move_to,
            grid_move_to_slot,
            protocol_get_task_config,
            item_get_task_config,
            item_update_task_config,
            protocol_update_task_config,
            item_update,
            protocol_set_schedule,
            item_apply_protocol,
            item_add,
            item_remove,
            item_move_to_pose,
            item_set_pose,
            item_apply_correlation,
            item_set_objective_position,
            item_set_objective_position_on_all,
            item_set_position,
            experiment_add_note,
            experiment_message_operator,
            server_report_session,
            server_set_constraints,
            server_set_checklist,
            request_create,
            request_withdraw,
            request_get,
            request_list,
            item_reorder_milling_stages,
            protocol_reorder_milling_stages,
            experiment_list_recent,
            experiment_list_all,
            protocol_list,
            experiment_create,
            experiment_load,
            protocol_load,
            app_get_events,
            app_get_display_images,
            proposal_list,
            proposal_decide,
            proposal_suggest,
            workflow_stop,
            workflow_plan,
            workflow_plan_grids,
            workflow_start_grids,
            protocol_set_supervision,
            workflow_set_options,
            workflow_requeue,
            workflow_edit_queue,
            workflow_start,
        )
    }
    # The catalog is the contract: refuse to start with an implementation gap.
    missing = {t.name for t in CATALOG} - set(implementations)
    if missing:
        raise RuntimeError(f"sidecar is missing implementations for: {sorted(missing)}")

    for name, fn in implementations.items():
        if name in offered:
            spec = by_name[name]
            params = dict(spec.params)
            if spec.takes_evidence:
                fn = _with_evidence(fn, client)
                params["evidence"] = EVIDENCE_PARAM
            mcp.add_tool(
                _described(_flag_failures(fn, name), params),
                name=name,
                title=spec.title,
                description=spec.full_description,
                annotations=ToolAnnotations(
                    title=spec.title,
                    read_only_hint=spec.read_only,
                    destructive_hint=spec.destructive,
                    idempotent_hint=spec.idempotent,
                    open_world_hint=False,
                ),
            )
    return mcp


# Where experiment_fetch_file keeps what it fetches, under the agent's working
# folder (FIBSEM_AGENT_WORKDIR, else where the sidecar runs).
FETCHED = "fetched"


def _workdir() -> str:
    """The agent's working folder: FIBSEM_AGENT_WORKDIR, else where the sidecar
    runs."""
    return os.path.realpath(os.environ.get("FIBSEM_AGENT_WORKDIR") or os.getcwd())


def _save_fetched(experiment: str, relative: str, content: bytes):
    """Keep a fetched file at fetched/<experiment>/<its path in the
    experiment>, under the working folder: its local path and size."""
    root = os.path.join(_workdir(), FETCHED)
    parts = [experiment, *relative.split("/")]
    target = os.path.realpath(os.path.join(root, *parts))
    if os.path.commonpath([root, target]) != root or target == root:
        return _Failure(f"refused: not a path to keep a file at: {relative}")
    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as f:
            f.write(content)
    except OSError as exc:
        return _Failure(f"failed: could not save {target}: {exc}")
    return {
        "saved": target,
        "size": len(content),
        "experiment": experiment,
        "path": relative,
    }


def _described(fn, params):
    """*fn*, each argument carrying the catalog's words for it
    (`ToolSpec.params`), so the client's input schema describes it."""
    import inspect

    from pydantic import Field

    signature = inspect.signature(fn, eval_str=True)
    fn.__signature__ = signature.replace(
        parameters=[
            parameter.replace(
                annotation=Annotated[
                    parameter.annotation, Field(description=params[parameter.name])
                ]
            )
            if parameter.name in params
            else parameter
            for parameter in signature.parameters.values()
        ]
    )
    return fn


def _evidence_body(evidence):
    """The body that keeps *evidence* (the tool's ``evidence`` argument): a
    ``file`` in the agent's working folder is read and sent as an upload. A
    `_Failure` when it is outside that folder or can't be read."""
    if not isinstance(evidence, dict):
        return _Failure(
            'refused (422): {"error_type": "invalid", "message": "evidence is '
            '{image or file, marks?, source?, item_name?}"}'
        )
    body = {k: v for k, v in evidence.items() if k != "file"}
    path = evidence.get("file")
    if path:
        root = _workdir()
        target = os.path.realpath(os.path.expanduser(str(path)))
        if os.path.commonpath([root, target]) != root:
            return _Failure(
                "refused (422): "
                + json.dumps(
                    {
                        "error_type": "invalid",
                        "message": f"evidence file {path} is outside your working "
                        f"folder ({root}): give a file there, or an experiment "
                        "image as image.",
                    }
                )
            )
        try:
            with open(target, "rb") as f:
                body["upload"] = base64.b64encode(f.read()).decode("ascii")
        except OSError as exc:
            return _Failure(
                "refused (422): "
                + json.dumps(
                    {
                        "error_type": "invalid",
                        "message": f"evidence file {path} can't be read: {exc}",
                    }
                )
            )
    return body


def _with_evidence(fn, client):
    """*fn*, taking the optional ``evidence`` an acting tool takes: kept in the
    experiment first (``POST /app/experiment/evidence``), then cited by id on
    the call's requests (`calls.EVIDENCE_HEADER`)."""
    import functools
    import inspect

    signature = inspect.signature(fn, eval_str=True)

    @functools.wraps(fn)
    def tool(*args, evidence: Optional[dict] = None, **kwargs):
        if not evidence:
            return fn(*args, **kwargs)
        body = _evidence_body(evidence)
        if isinstance(body, _Failure):
            return body
        saved, err = _call(client, "POST", "/app/experiment/evidence", body)
        if err:
            return err
        token = _citing.set(saved.get("evidence_id"))
        try:
            return fn(*args, **kwargs)
        finally:
            _citing.reset(token)

    tool.__signature__ = signature.replace(
        parameters=[
            *signature.parameters.values(),
            inspect.Parameter(
                "evidence",
                inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=Optional[dict],
            ),
        ]
    )
    return tool


def _flag_failures(fn, name=None):
    """*fn*, with a server error it answers (`_Failure`) raised as the MCP
    ToolError, so the call is flagged an error rather than read as a refusal.
    Any other exception is raised as a ToolError carrying its type and message:
    the SDK would otherwise tell the agent only "Error executing tool X".

    While it runs, its acting calls name the tool *name* and its arguments to
    the server (`_call`)."""
    import functools

    from mcp.server.mcpserver.exceptions import ToolError

    @functools.wraps(fn)
    def tool(*args, **kwargs):
        # The arguments the server describes: the evidence is cited apart.
        arguments = {k: v for k, v in kwargs.items() if k != "evidence"}
        token = _calling.set((name, arguments) if name else None)
        try:
            answer = fn(*args, **kwargs)
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc
        finally:
            _calling.reset(token)
        if isinstance(answer, _Failure):
            raise ToolError(str(answer))
        return answer

    return tool


def _watch_argv(argv):
    """``argv`` minus the ``watch`` subcommand, or None when it is not one.

    ``watch`` takes the same --url/--token as the sidecar, so they may come
    before it (``fibsem-mcp --url ... watch --help``) and still reach it.
    """
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "watch":
            return argv[:i] + argv[i + 1 :]
        if arg in ("--url", "--token"):
            i += 2
        elif arg.startswith(("--url=", "--token=")):
            i += 1
        else:
            return None
    return None


def _pair(argv) -> int:
    """``fibsem-mcp pair <code>``: pair this computer's agent, once, by hand."""
    from fibsem.client.pairing import DEFAULT_URL, pair

    parser = argparse.ArgumentParser(
        prog="fibsem-mcp pair",
        description="Pair this computer's agent with an App Server, using the "
        "code its Connect an agent shows. Later launches need no token.",
    )
    parser.add_argument("code")
    parser.add_argument(
        "--url", default=DEFAULT_URL, help=f"App Server URL (default {DEFAULT_URL})"
    )
    parser.add_argument(
        "--name", default=None, help="this computer's name (default: its host name)"
    )
    args = parser.parse_args(argv)
    try:
        _, role = pair(args.code, args.url, args.name)
    except (PermissionError, OSError) as exc:
        print(f"fibsem-mcp: {exc}", file=sys.stderr)
        return 1
    print(
        f"fibsem-mcp: paired with {args.url} as {'an agent' if role == 'agent' else 'the operator'}."
    )
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["pair"]:
        sys.exit(_pair(argv[1:]))
    watch_argv = _watch_argv(argv)
    if watch_argv is not None:
        # Before the mcp import: watching needs only HTTP (FIB-1269).
        from fibsem.ai.agent.mcp.watch import main as watch_main

        sys.exit(watch_main(watch_argv))
    try:
        import mcp  # noqa: F401
    except ImportError:
        sys.exit(
            "fibsem-mcp needs the [mcp] extra (Python 3.10+): pip install fibsem[mcp]"
        )
    parser = argparse.ArgumentParser(
        description="MCP sidecar for a fibsem server",
        epilog="fibsem-mcp watch: print one line per app event worth waking "
        "an agent for (fibsem-mcp watch --help). fibsem-mcp pair <code>: pair "
        "this computer's agent (fibsem-mcp pair --help).",
    )
    parser.add_argument(
        "--url", default=None, help="Server URL, e.g. http://127.0.0.1:8001"
    )
    parser.add_argument(
        "--token", default=None, help="Bearer token (from the server log or dialog)"
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=30.0,
        help="seconds the first tool call keeps retrying for a server before "
        "giving up (default 30)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="print connection status and exit (does not start the MCP server)",
    )
    args = parser.parse_args(argv)

    if args.check:
        try:
            client, capabilities = _connect_with_retry(args.url, args.token, 0.0)
        except ServerUnreachable as exc:
            sys.exit(str(exc))
        _say_connected(client.url, capabilities)
        return
    # Answer the MCP client's handshake first, and wait for a server in the
    # first call (FIB-1416).
    capabilities = _probe(args.url, args.token)
    if capabilities is None:
        print(
            "fibsem-mcp: no fibsem server answered yet; the first tool call "
            f"waits up to {args.wait:.0f}s for one.",
            file=sys.stderr,
        )
    wait_s = max(0.0, args.wait)
    # The first connection names the MCP client, so an ask to connect does.
    client = _LazyClient(
        lambda: _connect_with_retry(
            args.url, args.token, wait_s, user_agent=client.named_user_agent
        )
    )
    build_sidecar(client, capabilities).run()


if __name__ == "__main__":
    main()
