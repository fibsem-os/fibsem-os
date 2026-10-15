"""fibsem-mcp: the MCP sidecar that connects an agent to a fibsem server.

Runs on the agent host (the machine running Claude Code / Claude Desktop) and
proxies MCP tool calls to a fibsem server over HTTP with a bearer token. It
owns no microscope connection and enforces nothing itself — the server is the
security boundary. It registers every tool the server's mounted routers serve,
whatever the agent's level: the level changes mid-session, and the server's
403 says what is needed (FIB-1035).

Connection resolution, in order: ``--url``/``--token`` arguments, the
``FIBSEM_SERVER_URL``/``FIBSEM_SERVER_TOKEN`` environment variables, then the
same-machine discovery file a running server maintains.

Register with Claude Code:

    claude mcp add -s user fibsem -- fibsem-mcp
    # or explicitly:
    claude mcp add -s user fibsem -- fibsem-mcp --url http://microscope-pc:8001 --token <token>

Requires the [mcp] extra (Python 3.10+): pip install fibsem[mcp]

``fibsem-mcp watch`` prints a line per app event worth waking an agent for,
and needs nothing beyond fibsem itself (see :mod:`fibsem.mcp.watch`).
"""

import argparse
import base64
import json
import math
import os
import re
import sys
from typing import Any, List, Optional, Tuple
from urllib.parse import quote, urlencode

from fibsem.server.catalog import CATALOG, route_call, tools_for_capabilities

_TIMEOUT_S = 120.0  # acquisitions and moves on real hardware are slow
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


def _try_resolve(url, token):
    # type: (Optional[str], Optional[str]) -> Optional[Tuple[str, str]]
    """One resolution attempt; None when no server is known (yet)."""
    url = url or os.environ.get("FIBSEM_SERVER_URL")
    token = token or os.environ.get("FIBSEM_SERVER_TOKEN")
    if url and token:
        return url, token
    from fibsem.server.discovery import read_discovery_file

    found = read_discovery_file()
    if found is not None:
        return url or found["url"], token or found["token"]
    return None


class ServerUnreachable(Exception):
    """No server found, or none answering, within the wait: the message says
    which and what to do."""


class _ReconnectingClient:
    """An httpx-style client that follows a restarted server.

    A server restart (a microscope reconnect, a devices toggle) can bring a
    new address or token; the sidecar would otherwise answer 401 until the MCP
    client restarted it. On a 401 or a failed connection, re-resolve once --
    the discovery file names the server now running -- and retry with the
    fresh client. Explicit ``--url``/``--token`` re-resolve to themselves, so
    they just fail as before.
    """

    def __init__(self, client_factory, url, token, client, resolved):
        self._factory = client_factory
        self._url = url
        self._token = token
        self._client = client
        self._resolved = resolved
        self._user_agent = None

    def set_user_agent(self, user_agent: str) -> None:
        """Send *user_agent* from now on, through a reconnect too."""
        self._user_agent = user_agent
        self._apply_user_agent()

    def _apply_user_agent(self) -> None:
        headers = getattr(self._client, "headers", None)
        if self._user_agent is not None and headers is not None:
            headers["User-Agent"] = self._user_agent

    def request(self, method, path, **kwargs):
        try:
            resp = self._client.request(method, path, **kwargs)
        except Exception:
            if not self._reresolve():
                raise
            return self._client.request(method, path, **kwargs)
        if resp.status_code == 401 and self._reresolve():
            return self._client.request(method, path, **kwargs)
        return resp

    def _reresolve(self) -> bool:
        fresh = _try_resolve(self._url, self._token)
        if fresh is None or fresh == self._resolved:
            return False
        self._client.close()
        self._client = self._factory(fresh[0], fresh[1])
        self._resolved = fresh
        self._apply_user_agent()
        return True

    def close(self):
        self._client.close()

    def __getattr__(self, name):  # base_url and the like
        return getattr(self._client, name)


def _connect_with_retry(client_factory, url, token, wait_s, sleep_fn=None):
    """Resolve + reach a server, retrying until the deadline.

    MCP clients launch the sidecar at session start, often before the human has
    started the server — dying instantly turns a harmless ordering mistake into
    a CONNECTION_CLOSED (learned in the very first live session). So: keep
    re-resolving (the discovery file appears when the server starts) and
    re-trying /capabilities until ``wait_s`` runs out.

    Returns (client, capabilities), the client a :class:`_ReconnectingClient`.
    Raises :class:`ServerUnreachable` with guidance on timeout.
    """
    import time

    sleep_fn = sleep_fn or time.sleep
    deadline = time.monotonic() + wait_s
    waiting_reported = False
    while True:
        resolved = _try_resolve(url, token)
        if resolved is not None:
            client = client_factory(resolved[0], resolved[1])
            try:
                capabilities, err = _call(client, "GET", "/capabilities")
            except Exception as e:  # connection refused, DNS, ...
                capabilities, err = None, str(e)
            if err is None:
                return (
                    _ReconnectingClient(client_factory, url, token, client, resolved),
                    capabilities,
                )
            client.close()
        if time.monotonic() >= deadline:
            if resolved is None:
                raise ServerUnreachable(_NO_SERVER_MSG)
            raise ServerUnreachable(
                f"fibsem-mcp: cannot reach the server at {resolved[0]} — "
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
    """The server's client, connected on the first tool call rather than at
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
                print(
                    f"fibsem-mcp: connected to {client.base_url} "
                    f"(agent level {capabilities.get('level')})",
                    file=sys.stderr,
                )
                if self._user_agent is not None:
                    client.set_user_agent(self._user_agent)
                self._client = client
            return self._client

    def request(self, method, path, **kwargs):
        return self._connected().request(method, path, **kwargs)

    def set_user_agent(self, user_agent: str) -> None:
        with self._lock:
            self._user_agent = user_agent
            if self._client is not None:
                self._client.set_user_agent(user_agent)

    def close(self):
        if self._client is not None:
            self._client.close()


def _probe(client_factory, url, token):
    """The server's /capabilities if it answers within ``_PROBE_S``, so the
    tools can match it; None otherwise, and the sidecar starts anyway."""
    resolved = _try_resolve(url, token)
    if resolved is None:
        return None
    client = client_factory(resolved[0], resolved[1])
    try:
        resp = client.request("GET", "/capabilities", timeout=_PROBE_S)
        return resp.json() if resp.status_code == 200 else None
    except Exception:  # noqa: BLE001 - refused, timed out: start without it
        return None
    finally:
        client.close()


# fibsem.structures.MillingState by value, without importing it: the sidecar
# stays light. tests/server/test_sidecar.py pins the two in step.
_MILLING_STATES = ("IDLE", "RUNNING", "STOPPING", "PAUSED", "ERROR", "UNKNOWN")


class _Failure(str):
    """The server's own error (5xx), as text: unlike a refusal, the call may have
    done part of its work (a stage move that failed partway has moved). A tool
    that answers one is flagged an error (`_flag_failures`)."""


def _refusal(resp):
    try:
        detail = resp.json().get("detail", resp.text)
    except (ValueError, AttributeError):
        detail = resp.text
    if resp.status_code >= 500:
        return _Failure(f"failed ({resp.status_code}): {json.dumps(detail)}")
    return f"refused ({resp.status_code}): {json.dumps(detail)}"


def _call(client, method, path, payload=None):
    """Call the server; refusals come back as readable text, not exceptions.

    Agents act sensibly on a structured refusal string; a raw traceback for a
    403 teaches them nothing. A server error comes back as a `_Failure`: it
    says "failed", not "refused", because the call may have done part of its work.
    """
    try:
        resp = client.request(method, path, json=payload)
    except ServerUnreachable as exc:
        return None, str(exc)
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

    mcp = MCPServer("fibsem")
    _name_the_client(mcp, client)
    offered = {
        t.name
        for t in (
            CATALOG if capabilities is None else tools_for_capabilities(capabilities)
        )
    }
    by_name = {t.name: t for t in CATALOG}

    def _json_tool(name, payload_fn=None):
        spec = by_name[name]

        def tool(**kwargs):
            payload = payload_fn(**kwargs) if payload_fn else (kwargs or None)
            data, err = _call(client, spec.method, spec.path, payload)
            return err if err else data

        return spec, tool

    # The tool's name is positional-only: a tool may take an argument called
    # `name` (an overview's file name) and pass it on in **arguments.
    def _route(tool_name, /, **arguments):
        """Call a tool's route as the catalog maps its arguments: (data, err)."""
        try:
            method, path, body = route_call(by_name[tool_name], arguments)
        except ValueError as exc:
            return None, f"refused: {exc}"
        return _call(client, method, path, body)

    def _routed(tool_name, key=None, /, **arguments):
        """A routed tool's answer, its route's result under *key* when given; an
        image in it (a command answered with a preview) as MCP image content."""
        data, err = _route(tool_name, **arguments)
        if err:
            return err
        return _with_images({key: data["result"]} if key else data)

    def _with_images(data):
        """Move every base64 preview in *data* out into MCP image content.

        A question's context image as base64 text is unreadable to the agent
        and can overflow a tool result (a PickPOI prompt ran to 63k
        characters). Each preview's place in the JSON keeps ``image_index``,
        its position among the images that follow.
        """
        images = []

        def walk(node):
            if isinstance(node, dict):
                b64 = node.pop("image_b64_jpeg", None)
                if isinstance(b64, str):
                    node["image_index"] = len(images)
                    images.append(Image(data=base64.b64decode(b64), format="jpeg"))
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(data)
        if not images:
            return data
        return [json.dumps(data), *images]

    # Explicit signatures so FastMCP derives a real parameter schema per tool.
    def server_get_capabilities():
        return _json_tool("server_get_capabilities")[1]()

    def microscope_get_info():
        return _json_tool("microscope_get_info")[1]()

    def stage_get_position():
        data, err = _call(client, "GET", by_name["stage_get_position"].path)
        return err if err else {"position": data["value"]}

    def _service_call(name, kwargs, key):
        """A stage movement query or command as a named tool: *key* holds its result."""
        data, err = _call(client, "POST", by_name[name].path, {"kwargs": kwargs})
        return err if err else {key: data["result"]}

    def stage_get_orientation():
        return _service_call("stage_get_orientation", {}, "orientation")

    def microscope_get_state():
        return _json_tool("microscope_get_state")[1]()

    def stage_get_milling_angle():
        return _service_call("stage_get_milling_angle", {}, "milling_angle_deg")

    def stage_get_geometry():
        return _service_call("stage_get_geometry", {}, "geometry")

    def milling_get_state():
        data, err = _call(client, "GET", by_name["milling_get_state"].path)
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

    def milling_estimate_time():
        return _service_call("milling_estimate_time", {}, "seconds")

    def milling_preview_patterns(image_path: str, stages: Optional[list] = None):
        return _routed("milling_preview_patterns", image_path=image_path, stages=stages)

    def alignment_align_to_reference(
        reference_path: str, steps: int = 3, subsystem: str = "beam-shift"
    ):
        return _routed(
            "alignment_align_to_reference",
            "run",
            reference_path=reference_path,
            steps=steps,
            subsystem=subsystem,
        )

    def alignment_measure_shift(reference_path: str, image_path: str):
        return _routed(
            "alignment_measure_shift",
            "shift",
            reference_path=reference_path,
            image_path=image_path,
        )

    def milling_stop(reason: str):
        data, err = _route("milling_stop", reason=reason)
        return err if err else {"status": "ok"}

    def beam_acquire_image(
        beam_type: str = "ELECTRON", image_settings: Optional[dict] = None
    ):
        data, err = _route(
            "beam_acquire_image", beam_type=beam_type, image_settings=image_settings
        )
        return err if err else _with_images(data["result"])

    def stage_move_relative(dx: float = 0.0, dy: float = 0.0, dz: float = 0.0):
        return _routed("stage_move_relative", "position", dx=dx, dy=dy, dz=dz)

    def stage_stable_move(dx: float, dy: float, beam_type: str = "ELECTRON"):
        return _routed(
            "stage_stable_move", "position", dx=dx, dy=dy, beam_type=beam_type
        )

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
        return _json_tool("device_list")[1]()

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

    def service_list():
        return _json_tool("service_list")[1]()

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
        data, err = _call(
            client,
            "GET",
            f"/microscope/services/{quote(service, safe='')}/run?"
            + _query(wait=wait, run_id=run_id),
        )
        return err if err else _with_images(data)

    def service_call_query(service: str, query: str, kwargs: Optional[dict] = None):
        return _routed(
            "service_call_query", service=service, query=query, kwargs=kwargs
        )

    def beam_auto_contrast(beam_type: str = "ELECTRON"):
        return _routed("beam_auto_contrast", beam_type=beam_type)

    def beam_auto_focus(beam_type: str = "ELECTRON"):
        return _routed("beam_auto_focus", "result", beam_type=beam_type)

    def fm_auto_focus(channel_name: Optional[str] = None, path: Optional[str] = None):
        return _routed("fm_auto_focus", "run", channel_name=channel_name, path=path)

    def beam_acquire_overview(settings: dict, path: str, name: str = "overview"):
        return _routed(
            "beam_acquire_overview", "run", settings=settings, path=path, name=name
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

    def milling_run_stage(stage: dict, path: Optional[str] = None):
        return _routed("milling_run_stage", "run", stage=stage, path=path)

    def milling_run_task(config: dict, path: Optional[str] = None):
        return _routed("milling_run_task", "run", config=config, path=path)

    def fm_acquire_overview(
        channels: List[dict],
        parameters: dict,
        path: str,
        name: str = "overview",
        zparams: Optional[dict] = None,
    ):
        return _routed(
            "fm_acquire_overview",
            "run",
            channels=channels,
            parameters=parameters,
            path=path,
            name=name,
            zparams=zparams,
        )

    def fm_estimate_overview(
        channels: List[dict], parameters: dict, zparams: Optional[dict] = None
    ):
        return _routed(
            "fm_estimate_overview",
            "estimate",
            channels=channels,
            parameters=parameters,
            zparams=zparams,
        )

    def beam_auto_brightness_contrast(beam_type: str = "ELECTRON"):
        return _routed("beam_auto_brightness_contrast", "result", beam_type=beam_type)

    def _query(**params):
        """A query string of the *params* that are given (not None)."""
        return urlencode({k: v for k, v in params.items() if v is not None})

    def _app_get(path):
        data, err = _call(client, "GET", path)
        return err if err else data

    def _app_post(path, body=None):
        data, err = _call(client, "POST", path, body)
        return err if err else data

    def app_get_status():
        return _app_get("/app/workflow/status")

    def workflow_get_queue():
        return _app_get("/app/workflow/queue")

    def item_list():
        return _app_get("/app/experiment")

    def experiment_get_task_history():
        return _app_get("/app/experiment/task_history")

    def workflow_get_run_summary():
        return _app_get("/app/workflow/run_summary")

    def protocol_get():
        return _app_get("/app/experiment/protocol")

    def app_get_log(
        task_id: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        level: Optional[str] = None,
        lines: int = 200,
    ):
        return _app_get(
            "/app/log?"
            + _query(
                task_id=task_id, since=since, until=until, level=level, lines=lines
            )
        )

    def item_list_outputs(item_name: str):
        return _app_get(f"/app/experiment/items/{item_name}/outputs")

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
        # the bytes itself rather than through _call. Basenames carry spaces.
        path = f"/app/experiment/items/{quote(item_name, safe='')}/outputs/{quote(filename, safe='')}"
        resp = client.request("GET", path)
        if resp.status_code >= 400:
            return _refusal(resp)
        meta = {"item_name": item_name, "filename": filename, **_image_header(resp)}
        return [json.dumps(meta), Image(data=resp.content, format="jpeg")]

    def experiment_get_image_metadata(path: str):
        return _app_get("/app/experiment/images/metadata?" + urlencode({"path": path}))

    def item_get(item_name: str):
        return _app_get(f"/app/experiment/items/{item_name}")

    def grid_list():
        return _app_get("/app/experiment/grids")

    def grid_get(grid_name: str):
        return _app_get(f"/app/experiment/grids/{grid_name}")

    def grid_get_output_image(grid_name: str, filename: str):
        # A JPEG, like an item's output image.
        path = f"/app/experiment/grids/{quote(grid_name, safe='')}/outputs/{quote(filename, safe='')}"
        resp = client.request("GET", path)
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
        query = {"x": x, "y": y, "fx": fx, "fy": fy}
        return _app_get(
            f"/app/experiment/grids/{quote(grid_name, safe='')}/outputs/"
            f"{quote(filename, safe='')}/stage_position?"
            + urlencode({k: v for k, v in query.items() if v is not None})
        )

    def grid_get_markers(grid_name: str, filename: str):
        return _app_get(f"/app/experiment/grids/{grid_name}/outputs/{filename}/markers")

    def protocol_get_task_config(task_name: str):
        return _app_get(f"/app/experiment/protocol/task_config/{task_name}")

    def item_get_task_config(item_name: str, task_name: str):
        return _app_get(f"/app/experiment/items/{item_name}/task_config/{task_name}")

    def experiment_add_note(text: str, item_name=None):
        return _app_post(
            "/app/agent/notes", {"text": str(text), "item_name": item_name}
        )

    def request_create(actions: list, reason: str):
        return _app_post(
            "/app/agent/requests", {"actions": list(actions), "reason": str(reason)}
        )

    def request_withdraw(request_id: str, reason: str):
        data, err = _call(
            client,
            "POST",
            f"/app/agent/requests/{request_id}/withdraw",
            {"reason": str(reason)},
        )
        return err if err else data

    def request_get(request_id: str):
        # An applied image action carries its preview: an image, not base64 text.
        data, err = _call(client, "GET", f"/app/agent/requests/{request_id}")
        return err if err else _with_images(data)

    def request_list():
        data, err = _call(client, "GET", "/app/agent/requests")
        return err if err else _with_images(data)

    def item_add(
        name: Optional[str] = None,
        stage_position: Optional[dict] = None,
        overview: Optional[dict] = None,
    ):
        return _app_post(
            "/app/experiment/items",
            {"name": name, "stage_position": stage_position, "overview": overview},
        )

    def item_remove(item_name: str):
        data, err = _call(
            client, "DELETE", f"/app/experiment/items/{quote(item_name, safe='')}"
        )
        return err if err else data

    def item_move_to_pose(item_name: str, pose: str):
        return _app_post(
            f"/app/experiment/items/{quote(item_name, safe='')}/move_to", {"pose": pose}
        )

    def item_set_pose(item_name: str, pose_name: str):
        return _app_post(
            f"/app/experiment/items/{quote(item_name, safe='')}"
            f"/poses/{quote(pose_name, safe='')}"
        )

    def item_set_objective_position(item_name: str):
        return _app_post(
            f"/app/experiment/items/{quote(item_name, safe='')}/objective_position"
        )

    def item_apply_protocol(item_name: str, task_names=None):
        return _app_post(
            f"/app/experiment/items/{item_name}/apply_protocol",
            {"task_names": list(task_names) if task_names else None},
        )

    def item_reorder_milling_stages(
        item_name: str, task_name: str, milling_key: str, order: list, version: str
    ):
        return _app_post(
            f"/app/experiment/items/{item_name}/task_config/{task_name}/stages/reorder",
            {"milling_key": milling_key, "order": list(order), "version": version},
        )

    def protocol_reorder_milling_stages(
        task_name: str, milling_key: str, order: list, version: str
    ):
        return _app_post(
            f"/app/experiment/protocol/task_config/{task_name}/stages/reorder",
            {"milling_key": milling_key, "order": list(order), "version": version},
        )

    def protocol_set_schedule(task_name: str, scheduled_at=None):
        return _app_post(
            "/app/workflow/schedule",
            {"task_name": task_name, "scheduled_at": scheduled_at},
        )

    def item_update(item_name: str, patch: dict, version: str):
        return _app_post(
            f"/app/experiment/items/{item_name}",
            {"patch": dict(patch), "version": str(version)},
        )

    def protocol_update_task_config(task_name: str, patch: dict, version: str):
        return _app_post(
            f"/app/experiment/protocol/task_config/{task_name}",
            {"patch": dict(patch), "version": str(version)},
        )

    def item_update_task_config(
        item_name: str, task_name: str, patch: dict, version: str
    ):
        return _app_post(
            f"/app/experiment/items/{item_name}/task_config/{task_name}",
            {"patch": dict(patch), "version": str(version)},
        )

    def experiment_list_recent():
        return _app_get("/app/experiment/recent")

    def protocol_list():
        return _app_get("/app/experiment/protocols")

    def experiment_create(
        name: str, protocol: Optional[str] = None, description: Optional[str] = None
    ):
        return _app_post(
            "/app/experiment/create",
            {"name": name, "protocol": protocol, "description": description},
        )

    def experiment_load(path: str):
        return _app_post("/app/experiment/load", {"path": path})

    def protocol_load(protocol: str):
        return _app_post("/app/experiment/protocol/load", {"protocol": protocol})

    def app_get_events(since: int = 0, timeout: float = 0.0):
        return _app_get(f"/app/events?since={since}&timeout={timeout}")

    def workflow_stop(reason: str):
        return _app_post("/app/workflow/stop", {"reason": reason})

    def workflow_start(task_names: list, item_names: list = None):  # noqa: RUF013
        payload = {"task_names": list(task_names)}
        if item_names is not None:
            payload["item_names"] = list(item_names)
        return _app_post("/app/workflow/start", payload)

    def workflow_plan_grids(
        task_names: list, grid_names: list = None, screen_all: bool = False
    ):  # noqa: RUF013
        payload = {"task_names": list(task_names), "screen_all": bool(screen_all)}
        if grid_names is not None:
            payload["grid_names"] = list(grid_names)
        return _app_post("/app/workflow/grids/plan", payload)

    def workflow_start_grids(
        task_names: list, grid_names: list = None, screen_all: bool = False
    ):  # noqa: RUF013
        payload = {"task_names": list(task_names), "screen_all": bool(screen_all)}
        if grid_names is not None:
            payload["grid_names"] = list(grid_names)
        return _app_post("/app/workflow/grids/start", payload)

    def workflow_set_options(show_run_summary: bool):
        payload = {"show_run_summary": bool(show_run_summary)}
        return _app_post("/app/workflow/options", payload)

    def protocol_set_supervision(task_name: str, supervise: bool):
        payload = {"task_name": task_name, "supervise": bool(supervise)}
        return _app_post("/app/agent/supervision", payload)

    def workflow_requeue(item_name: str, task_name: str, front: bool = False):
        return _app_post(
            "/app/workflow/queue/requeue",
            {"item_name": item_name, "task_name": task_name, "front": bool(front)},
        )

    def app_get_display_images():
        data, err = _call(client, "GET", "/app/experiment/images", None)
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

    def prompt_get():
        data, err = _call(client, "GET", "/app/workflow/prompt")
        return err if err else _with_images(data)

    def review_list():
        data, err = _call(client, "GET", "/app/workflow/reviews")
        return err if err else _with_images(data)

    def review_decide(
        item_id: str,
        task_name: str,
        outcome: str,
        proposal_id: str = "",
        values: Optional[dict] = None,
        reason: str = "",
        author: str = "",
        task_id: str = "",
    ):
        body = {
            "item_id": str(item_id),
            "task_name": str(task_name),
            "outcome": str(outcome),
            "reason": str(reason),
            "author": str(author),
        }
        # What the decision is on. The id is the one to pass; the run is
        # accepted in its place, for a caller written before proposals had ids.
        if proposal_id:
            body["proposal_id"] = str(proposal_id)
        if task_id:
            body["task_id"] = str(task_id)
        if values is not None:
            body["values"] = dict(values)
        return _app_post("/app/workflow/decide", body)

    def prompt_suggest_answer(
        response: bool, question_id: int, reason: str, value: Optional[dict] = None
    ):
        body = {
            "response": bool(response),
            "question_id": int(question_id),
            "reason": str(reason),
        }
        if value is not None:
            body["value"] = dict(value)
        return _app_post("/app/workflow/prompt/suggest", body)

    def prompt_answer(response: bool, question_id: int, value: Optional[dict] = None):
        body = {"response": bool(response), "question_id": int(question_id)}
        if value is not None:
            body["value"] = dict(value)
        return _app_post("/app/workflow/prompt/answer", body)

    implementations = {
        fn.__name__: fn
        for fn in (
            server_get_capabilities,
            microscope_get_info,
            stage_get_position,
            stage_get_orientation,
            microscope_get_state,
            stage_get_milling_angle,
            stage_get_geometry,
            milling_get_state,
            milling_estimate_time,
            milling_preview_patterns,
            alignment_align_to_reference,
            alignment_measure_shift,
            milling_stop,
            beam_acquire_image,
            stage_move_relative,
            stage_stable_move,
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
            fm_acquire_image,
            milling_run_stage,
            milling_run_task,
            fm_acquire_overview,
            fm_estimate_overview,
            beam_auto_brightness_contrast,
            app_get_status,
            workflow_get_queue,
            item_list,
            experiment_get_task_history,
            workflow_get_run_summary,
            app_get_log,
            protocol_get,
            item_list_outputs,
            item_get_output_image,
            experiment_get_image_metadata,
            item_get,
            grid_list,
            grid_get,
            grid_get_output_image,
            grid_get_markers,
            grid_get_stage_position,
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
            item_set_objective_position,
            experiment_add_note,
            request_create,
            request_withdraw,
            request_get,
            request_list,
            item_reorder_milling_stages,
            protocol_reorder_milling_stages,
            experiment_list_recent,
            protocol_list,
            experiment_create,
            experiment_load,
            protocol_load,
            app_get_events,
            app_get_display_images,
            prompt_get,
            review_list,
            review_decide,
            prompt_answer,
            prompt_suggest_answer,
            workflow_stop,
            workflow_plan_grids,
            workflow_start_grids,
            protocol_set_supervision,
            workflow_set_options,
            workflow_requeue,
            workflow_start,
        )
    }
    # The catalog is the contract: refuse to start with an implementation gap.
    missing = {t.name for t in CATALOG} - set(implementations)
    if missing:
        raise RuntimeError(f"sidecar is missing implementations for: {sorted(missing)}")

    for name, fn in implementations.items():
        if name in offered:
            mcp.add_tool(
                _flag_failures(fn),
                name=name,
                description=by_name[name].full_description,
            )
    return mcp


def _flag_failures(fn):
    """*fn*, with a server error it answers (`_Failure`) raised as the MCP
    ToolError, so the call is flagged an error rather than read as a refusal.
    Any other exception is raised as a ToolError carrying its type and message:
    the SDK would otherwise tell the agent only "Error executing tool X"."""
    import functools

    from mcp.server.mcpserver.exceptions import ToolError

    @functools.wraps(fn)
    def tool(*args, **kwargs):
        try:
            answer = fn(*args, **kwargs)
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc
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


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    watch_argv = _watch_argv(argv)
    if watch_argv is not None:
        # Before the mcp import: watching needs only HTTP (FIB-1269).
        from fibsem.mcp.watch import main as watch_main

        sys.exit(watch_main(watch_argv))
    try:
        import mcp  # noqa: F401
    except ImportError:
        sys.exit(
            "fibsem-mcp needs the [mcp] extra (Python 3.10+): pip install fibsem[mcp]"
        )
    import httpx

    parser = argparse.ArgumentParser(
        description="MCP sidecar for a fibsem server",
        epilog="fibsem-mcp watch: print one line per app event worth waking "
        "an agent for (fibsem-mcp watch --help).",
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

    def client_factory(url, token):
        return httpx.Client(
            base_url=url,
            headers={"Authorization": f"Bearer {token}", "User-Agent": _user_agent()},
            timeout=_TIMEOUT_S,
        )

    if args.check:
        try:
            client, capabilities = _connect_with_retry(
                client_factory, args.url, args.token, 0.0
            )
        except ServerUnreachable as exc:
            sys.exit(str(exc))
        print(
            f"fibsem-mcp: connected to {client.base_url} "
            f"(agent level {capabilities.get('level')})",
            file=sys.stderr,
        )
        return
    # Answer the MCP client's handshake first, and wait for a server in the
    # first call (FIB-1416).
    capabilities = _probe(client_factory, args.url, args.token)
    if capabilities is None:
        print(
            "fibsem-mcp: no fibsem server answered yet; the first tool call "
            f"waits up to {args.wait:.0f}s for one.",
            file=sys.stderr,
        )
    wait_s = max(0.0, args.wait)
    client = _LazyClient(
        lambda: _connect_with_retry(client_factory, args.url, args.token, wait_s)
    )
    build_sidecar(client, capabilities).run()


if __name__ == "__main__":
    main()
