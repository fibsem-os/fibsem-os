"""fibsem-mcp: the MCP sidecar that connects an agent to a fibsem server.

Runs on the agent host (the machine running Claude Code / Claude Desktop) and
proxies MCP tool calls to a fibsem server over HTTP with a bearer token. It
owns no microscope connection and enforces nothing itself — the server is the
security boundary. It registers every tool the server's mounted routers serve,
whatever the agent's level: the level changes mid-session, and the server's
403 says what is needed.

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
import os
import re
import sys
from contextvars import ContextVar
from typing import Annotated, Dict, List, Optional, Tuple, Union
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
# a client gives a server only so long to start.
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
    start-up.

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
    other ``fibsem`` code than this sidecar."""
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
        # The server describes the call for the operator from these.
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
        agent's own call."""
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

    def stage_get_position():
        data, err = _route("stage_get_position")
        return err if err else {"position": data["value"]}

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

    def microscope_set_state(microscope_state: dict):
        return _routed("microscope_set_state", microscope_state=dict(microscope_state))

    def service_wait_for_run(
        service: str, run_id: Optional[str] = None, wait: float = 60.0
    ):
        data, err = _route(
            "service_wait_for_run", service=service, run_id=run_id, wait=wait
        )
        return err if err else _with_images(data)

    def _given(**params):
        """The *params* that are given (not None): a body that leaves out the rest."""
        return {k: v for k, v in params.items() if v is not None}

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

    def experiment_get_image_stage_position(
        path: str,
        x: Optional[float] = None,
        y: Optional[float] = None,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        crop: Optional[Union[str, List[float]]] = None,
    ):
        if isinstance(crop, (list, tuple)):
            crop = ",".join(str(v) for v in crop)
        return _tool(
            "experiment_get_image_stage_position",
            path=path,
            x=x,
            y=y,
            fx=fx,
            fy=fy,
            crop=crop,
        )

    def experiment_render_markers(
        path: str,
        grid_name: Optional[str] = None,
        fov: Optional[bool] = None,
        current: Optional[bool] = None,
    ):
        return _routed(
            "experiment_render_markers",
            path=path,
            **_given(grid_name=grid_name, fov=fov, current=current),
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

    def item_draw_milling(
        item_name: str,
        task_names: List[str],
        image_path: Optional[str] = None,
        background: Optional[str] = None,
        scan_direction: Optional[bool] = None,
        alignment_area: Optional[bool] = None,
        metadata_bar: Optional[bool] = None,
    ):
        return _routed(
            "item_draw_milling",
            item_name=item_name,
            task_names=[task_names]
            if isinstance(task_names, str)
            else list(task_names),
            **_given(
                image_path=image_path,
                background=background,
                scan_direction=scan_direction,
                alignment_area=alignment_area,
                metadata_bar=metadata_bar,
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
        path,
        box=None,
        fraction=None,
        z=None,
        channels=None,
        experiment=None,
        cells=None,
    ):
        """The file route's answer for *path*: (data, err). *cells* is true
        (show the cells) or the cells' names (crop them)."""
        if isinstance(cells, (list, tuple)):
            cells = ",".join(str(name) for name in cells)
        elif cells is not None:
            cells = "true" if cells else None
        return _route(
            "experiment_read_file",
            path=str(path),
            box=_region(box),
            fraction=_region(fraction),
            z=z,
            channels=",".join(channels) if channels else None,
            cells=cells,
            experiment=experiment,
        )

    def experiment_read_file(
        path: str,
        z: Optional[int] = None,
        channels: Optional[List[str]] = None,
        cells: Optional[bool] = None,
        experiment: Optional[str] = None,
    ):
        data, err = _read_file(
            path, z=z, channels=channels, experiment=experiment, cells=cells
        )
        return err if err else _with_images(data)

    def experiment_crop_image(
        path: str,
        box: Optional[list] = None,
        fraction: Optional[list] = None,
        cells: Optional[List[str]] = None,
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
            cells=list(cells) if cells is not None else None,
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

    def grid_get_output_image(
        grid_name: str, filename: str, cells: Optional[bool] = None
    ):
        # A JPEG, like an item's output image; its cells ride in the metadata.
        resp = _send(
            "grid_get_output_image",
            dict(
                grid_name=grid_name,
                filename=filename,
                **_given(cells="true" if cells else None),
            ),
        )
        if resp.status_code >= 400:
            return _refusal(resp)
        meta = {"grid_name": grid_name, "filename": filename, **_image_header(resp)}
        return [json.dumps(meta), Image(data=resp.content, format="jpeg")]

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

    def request_list():
        data, err = _route("request_list")
        return err if err else _with_images(data)

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

    def protocol_reorder_grid_tasks(order: list):
        return _tool("protocol_reorder_grid_tasks", order=list(order))

    def protocol_set_grid_task_requirements(task_name: str, requires: list):
        return _routed(
            "protocol_set_grid_task_requirements",
            task_name=task_name,
            requires=list(requires),
        )

    def workflow_queue_grids(task_names: list, grid_names: list):
        return _tool(
            "workflow_queue_grids",
            task_names=list(task_names),
            grid_names=list(grid_names),
        )

    def workflow_start(task_names: list, item_names: Optional[list] = None):
        payload = {"task_names": list(task_names)}
        if item_names is not None:
            payload["item_names"] = list(item_names)
        return _tool("workflow_start", **payload)

    def workflow_plan(
        task_names: list,
        item_names: Optional[list] = None,
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
        task_names: list, grid_names: Optional[list] = None, screen_all: bool = False
    ):
        payload = {"task_names": list(task_names), "screen_all": bool(screen_all)}
        if grid_names is not None:
            payload["grid_names"] = list(grid_names)
        return _tool("workflow_plan_grids", **payload)

    def workflow_start_grids(
        task_names: list,
        grid_names: Optional[list] = None,
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

    def _answered(spec):
        """A tool that answers its route's answer as it stands, or its
        ``result`` under `ToolSpec.result_key`; a preview in it as an image."""

        def tool(**arguments):
            data, err = _route(spec.name, **arguments)
            if err:
                return err
            key = spec.result_key
            return _with_images({key: data["result"]} if key else data)

        tool.__name__ = spec.name
        tool.__signature__ = spec.signature
        return tool

    # The tools that reshape their answer or do more than call their route;
    # every other tool is its catalog entry (`_answered`).
    overrides = {
        fn.__name__: fn
        for fn in (
            stage_get_position,
            milling_get_state,
            milling_estimate_time,
            correlation_correlate,
            correlation_find_spot_burns,
            microscope_set_state,
            experiment_get_image_stage_position,
            experiment_render_markers,
            experiment_draw_image,
            experiment_draw_pair,
            item_draw_milling,
            experiment_get_image_pixels,
            protocol_reorder_tasks,
            protocol_set_task_requirements,
            protocol_set_lamella_defaults,
            protocol_save,
            protocol_reorder_grid_tasks,
            protocol_set_grid_task_requirements,
            workflow_queue_grids,
            milling_stop,
            milling_end,
            milling_pause,
            milling_resume,
            beam_acquire_image,
            service_wait_for_run,
            item_get_history,
            item_get_output_image,
            item_measure_thickness,
            experiment_read_file,
            experiment_crop_image,
            experiment_fetch_file,
            grid_get_output_image,
            item_update_task_config,
            protocol_update_task_config,
            item_update,
            item_apply_protocol,
            experiment_add_note,
            experiment_message_operator,
            server_report_session,
            request_create,
            request_withdraw,
            request_list,
            item_reorder_milling_stages,
            protocol_reorder_milling_stages,
            app_get_display_images,
            proposal_list,
            proposal_decide,
            proposal_suggest,
            workflow_plan,
            workflow_plan_grids,
            workflow_start_grids,
            protocol_set_supervision,
            workflow_set_options,
            workflow_requeue,
            workflow_start,
        )
    }
    unknown = set(overrides) - set(by_name)
    if unknown:
        raise RuntimeError(
            f"sidecar overrides tools not in the catalog: {sorted(unknown)}"
        )
    implementations = {
        spec.name: overrides.get(spec.name) or _answered(spec) for spec in CATALOG
    }

    for name, fn in implementations.items():
        if name in offered:
            spec = by_name[name]
            params = {n: a.doc for n, a in spec.arguments.items()}
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
        # Before the mcp import: watching needs only HTTP.
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
    # first call.
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
