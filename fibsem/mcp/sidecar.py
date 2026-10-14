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
    claude mcp add -s user fibsem -- fibsem-mcp --url http://hydra-support:8001 --token <token>

Requires the [mcp] extra (Python 3.10+): pip install fibsem[mcp]

``fibsem-mcp watch`` prints a line per app event worth waking an agent for,
and needs nothing beyond fibsem itself (see :mod:`fibsem.mcp.watch`).
"""

import argparse
import base64
import json
import os
import re
import sys
from typing import Any, Optional, Tuple
from urllib.parse import quote, urlencode

from fibsem.server.catalog import CATALOG, tools_for_capabilities

_TIMEOUT_S = 120.0  # acquisitions and moves on real hardware are slow


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
    Exits with guidance on timeout.
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
                sys.exit(_NO_SERVER_MSG)
            sys.exit(
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


def _refusal(resp):
    try:
        detail = resp.json().get("detail", resp.text)
    except ValueError:
        detail = resp.text
    return f"refused ({resp.status_code}): {json.dumps(detail)}"


def _call(client, method, path, payload=None):
    """Call the server; refusals come back as readable text, not exceptions.

    Agents act sensibly on a structured refusal string; a raw traceback for a
    403 teaches them nothing.
    """
    resp = client.request(method, path, json=payload)
    if resp.status_code >= 400:
        return None, _refusal(resp)
    try:
        return resp.json(), None
    except ValueError:
        # An image or array from a device command: nothing an agent can read as
        # text. The preview tools are how an agent looks at an image.
        return None, (
            f"answered {len(resp.content)} bytes that are not JSON; "
            "use acquire_image_preview to see images"
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
    application), whatever the agent's level; on a bench server, which has no
    level, hardware tools only when the hardware scope is armed. The server
    enforces regardless.
    """
    from mcp.server.mcpserver import Image, MCPServer

    mcp = MCPServer("fibsem")
    _name_the_client(mcp, client)
    armed = {t.name for t in tools_for_capabilities(capabilities)}
    by_name = {t.name: t for t in CATALOG}

    def _json_tool(name, payload_fn=None):
        spec = by_name[name]

        def tool(**kwargs):
            payload = payload_fn(**kwargs) if payload_fn else (kwargs or None)
            data, err = _call(client, spec.method, spec.path, payload)
            return err if err else data

        return spec, tool

    def _image_tool(name):
        spec = by_name[name]

        def tool(beam_type="ELECTRON"):
            data, err = _call(client, spec.method, spec.path, {"beam_type": beam_type})
            if err:
                return err
            jpeg = base64.b64decode(data.pop("image_b64_jpeg"))
            return [json.dumps(data), Image(data=jpeg, format="jpeg")]

        return spec, tool

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
    def get_capabilities():
        return _json_tool("get_capabilities")[1]()

    def get_system_info():
        return _json_tool("get_system_info")[1]()

    def get_stage_position():
        return _json_tool("get_stage_position")[1]()

    def get_stage_orientation():
        return _json_tool("get_stage_orientation")[1]()

    def get_microscope_state():
        return _json_tool("get_microscope_state")[1]()

    def get_milling_angle():
        return _json_tool("get_milling_angle")[1]()

    def get_milling_state():
        return _json_tool("get_milling_state")[1]()

    def estimate_milling_time():
        return _json_tool("estimate_milling_time")[1]()

    def stop_milling():
        return _json_tool("stop_milling")[1]()

    def acquire_image_preview(beam_type: str = "ELECTRON"):
        return _image_tool("acquire_image_preview")[1](beam_type=beam_type)

    def move_stage_relative(dx: float = 0.0, dy: float = 0.0, dz: float = 0.0):
        position = {
            "name": None,
            "x": dx,
            "y": dy,
            "z": dz,
            "r": 0.0,
            "t": 0.0,
            "coordinate_system": "RAW",
        }
        data, err = _call(
            client, "POST", "/move_stage_relative", {"position": position}
        )
        return err if err else data

    def move_stage_absolute(x: float, y: float, z: float, r: float, t: float):
        position = {
            "name": None,
            "x": x,
            "y": y,
            "z": z,
            "r": r,
            "t": t,
            "coordinate_system": "RAW",
        }
        data, err = _call(
            client, "POST", "/move_stage_absolute", {"position": position}
        )
        return err if err else data

    def stable_move(dx: float, dy: float, beam_type: str = "ELECTRON"):
        data, err = _call(
            client,
            "POST",
            "/stable_move",
            {"dx": float(dx), "dy": float(dy), "beam_type": str(beam_type)},
        )
        return err if err else data

    def vertical_move(dy: float, dx: float = 0.0):
        data, err = _call(
            client, "POST", "/vertical_move", {"dy": float(dy), "dx": float(dx)}
        )
        return err if err else data

    def move_to_orientation(orientation: str):
        data, err = _call(
            client, "POST", "/move_to_orientation", {"orientation": str(orientation)}
        )
        return err if err else data

    def safe_move_stage_absolute(x: float, y: float, z: float, r: float, t: float):
        position = {
            "name": None,
            "x": x,
            "y": y,
            "z": z,
            "r": r,
            "t": t,
            "coordinate_system": "RAW",
        }
        data, err = _call(
            client, "POST", "/safe_absolute_stage_movement", {"position": position}
        )
        return err if err else data

    def move_to_milling_angle(milling_angle_deg: float):
        data, err = _call(
            client,
            "POST",
            "/milling_angle/move",
            {"milling_angle_deg": milling_angle_deg},
        )
        return err if err else data

    def list_devices():
        return _json_tool("list_devices")[1]()

    def get_parameter(device: str, parameter: str):
        data, err = _call(client, "GET", f"/devices/{device}/{parameter}")
        return err if err else data

    def set_parameter(device: str, parameter: str, value: Any):
        data, err = _call(
            client, "PUT", f"/devices/{device}/{parameter}", {"value": value}
        )
        return err if err else data

    def call_command(device: str, command: str, kwargs: Optional[dict] = None):
        data, err = _call(
            client,
            "POST",
            f"/devices/{device}/commands/{command}",
            {"kwargs": dict(kwargs or {})},
        )
        return err if err else data

    def list_services():
        return _json_tool("list_services")[1]()

    def get_service_parameter(service: str, parameter: str):
        data, err = _call(client, "GET", f"/services/{service}/{parameter}")
        return err if err else data

    def call_service_command(service: str, command: str, kwargs: Optional[dict] = None):
        data, err = _call(
            client,
            "POST",
            f"/services/{service}/commands/{command}",
            {"kwargs": dict(kwargs or {})},
        )
        return err if err else data

    def call_service_query(service: str, query: str, kwargs: Optional[dict] = None):
        data, err = _call(
            client,
            "POST",
            f"/services/{service}/queries/{query}",
            {"kwargs": dict(kwargs or {})},
        )
        return err if err else data

    def autocontrast(beam_type: str = "ELECTRON"):
        data, err = _call(client, "POST", "/autocontrast", {"beam_type": beam_type})
        return err if err else data

    def _app_get(path):
        data, err = _call(client, "GET", path)
        return err if err else data

    def get_app_status():
        return _app_get("/app/status")

    def get_app_queue():
        return _app_get("/app/queue")

    def get_experiment_summary():
        return _app_get("/app/experiment_summary")

    def get_task_history():
        return _app_get("/app/task_history")

    def get_run_summary():
        return _app_get("/app/run_summary")

    def get_protocol():
        return _app_get("/app/protocol")

    def get_log(
        task_id: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        level: Optional[str] = None,
        lines: int = 200,
    ):
        query = {
            "task_id": task_id,
            "since": since,
            "until": until,
            "level": level,
            "lines": lines,
        }
        return _app_get(
            "/app/log?" + urlencode({k: v for k, v in query.items() if v is not None})
        )

    def get_task_outputs(item_name: str):
        return _app_get(f"/app/task_outputs/{item_name}")

    def get_item_output_image(item_name: str, filename: str):
        # The route answers image/jpeg for the dashboard's <img>, so this reads
        # the bytes itself rather than through _call. Basenames carry spaces.
        path = (
            f"/app/items/{quote(item_name, safe='')}/outputs/{quote(filename, safe='')}"
        )
        resp = client.request("GET", path)
        if resp.status_code >= 400:
            return _refusal(resp)
        meta = {"item_name": item_name, "filename": filename}
        return [json.dumps(meta), Image(data=resp.content, format="jpeg")]

    def get_item_detail(item_name: str):
        return _app_get(f"/app/items/{item_name}")

    def get_grids():
        return _app_get("/app/grids")

    def get_grid_detail(grid_name: str):
        return _app_get(f"/app/grids/{grid_name}")

    def get_grid_markers(grid_name: str, filename: str):
        return _app_get(f"/app/grids/{grid_name}/outputs/{filename}/markers")

    def get_protocol_task_config(task_name: str):
        return _app_get(f"/app/protocol/task_config/{task_name}")

    def get_item_task_config(item_name: str, task_name: str):
        return _app_get(f"/app/items/{item_name}/task_config/{task_name}")

    def add_note(text: str, item_name=None):
        data, err = _call(
            client,
            "POST",
            "/app/agent/notes",
            {"text": str(text), "item_name": item_name},
        )
        return err if err else data

    def apply_protocol_to_item(item_name: str, task_names=None):
        data, err = _call(
            client,
            "POST",
            f"/app/items/{item_name}/apply_protocol",
            {"task_names": list(task_names) if task_names else None},
        )
        return err if err else data

    def reorder_item_milling_stages(
        item_name: str, task_name: str, milling_key: str, order: list, version: str
    ):
        data, err = _call(
            client,
            "POST",
            f"/app/items/{item_name}/task_config/{task_name}/stages/reorder",
            {"milling_key": milling_key, "order": list(order), "version": version},
        )
        return err if err else data

    def reorder_protocol_milling_stages(
        task_name: str, milling_key: str, order: list, version: str
    ):
        data, err = _call(
            client,
            "POST",
            f"/app/protocol/task_config/{task_name}/stages/reorder",
            {"milling_key": milling_key, "order": list(order), "version": version},
        )
        return err if err else data

    def set_task_schedule(task_name: str, scheduled_at=None):
        data, err = _call(
            client,
            "POST",
            "/app/workflow/schedule",
            {"task_name": task_name, "scheduled_at": scheduled_at},
        )
        return err if err else data

    def update_item_detail(item_name: str, patch: dict, version: str):
        data, err = _call(
            client,
            "POST",
            f"/app/items/{item_name}",
            {"patch": dict(patch), "version": str(version)},
        )
        return err if err else data

    def update_protocol_task_config(task_name: str, patch: dict, version: str):
        data, err = _call(
            client,
            "POST",
            f"/app/protocol/task_config/{task_name}",
            {"patch": dict(patch), "version": str(version)},
        )
        return err if err else data

    def update_item_task_config(
        item_name: str, task_name: str, patch: dict, version: str
    ):
        data, err = _call(
            client,
            "POST",
            f"/app/items/{item_name}/task_config/{task_name}",
            {"patch": dict(patch), "version": str(version)},
        )
        return err if err else data

    def list_recent_experiments():
        return _app_get("/app/recent_experiments")

    def get_events(since: int = 0, timeout: float = 0.0):
        return _app_get(f"/app/events?since={since}&timeout={timeout}")

    def stop_workflow():
        data, err = _call(client, "POST", "/app/workflow/stop", None)
        return err if err else data

    def start_workflow(task_names: list, item_names: list = None):  # noqa: RUF013
        payload = {"task_names": list(task_names)}
        if item_names is not None:
            payload["item_names"] = list(item_names)
        data, err = _call(client, "POST", "/app/workflow/start", payload)
        return err if err else data

    def plan_grid_workflow(
        task_names: list, grid_names: list = None, screen_all: bool = False
    ):  # noqa: RUF013
        payload = {"task_names": list(task_names), "screen_all": bool(screen_all)}
        if grid_names is not None:
            payload["grid_names"] = list(grid_names)
        data, err = _call(client, "POST", "/app/workflow/grids/plan", payload)
        return err if err else data

    def start_grid_workflow(
        task_names: list, grid_names: list = None, screen_all: bool = False
    ):  # noqa: RUF013
        payload = {"task_names": list(task_names), "screen_all": bool(screen_all)}
        if grid_names is not None:
            payload["grid_names"] = list(grid_names)
        data, err = _call(client, "POST", "/app/workflow/grids/start", payload)
        return err if err else data

    def set_workflow_options(show_run_summary: bool):
        payload = {"show_run_summary": bool(show_run_summary)}
        data, err = _call(client, "POST", "/app/workflow/options", payload)
        return err if err else data

    def set_task_supervision(task_name: str, supervise: bool):
        payload = {"task_name": task_name, "supervise": bool(supervise)}
        data, err = _call(client, "POST", "/app/supervision", payload)
        return err if err else data

    def requeue_task(item_name: str, task_name: str, front: bool = False):
        data, err = _call(
            client,
            "POST",
            "/app/queue/requeue",
            {"item_name": item_name, "task_name": task_name, "front": bool(front)},
        )
        return err if err else data

    def get_display_images():
        data, err = _call(client, "GET", "/app/images", None)
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

    def get_pending_prompt():
        data, err = _call(client, "GET", "/app/prompt")
        return err if err else _with_images(data)

    def get_pending_reviews():
        data, err = _call(client, "GET", "/app/reviews")
        return err if err else _with_images(data)

    def decide_review(
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
        data, err = _call(client, "POST", "/app/decide", body)
        return err if err else data

    def answer_prompt(response: bool, question_id: int, value: Optional[dict] = None):
        body = {"response": bool(response), "question_id": int(question_id)}
        if value is not None:
            body["value"] = dict(value)
        data, err = _call(client, "POST", "/app/prompt/answer", body)
        return err if err else data

    implementations = {
        fn.__name__: fn
        for fn in (
            get_capabilities,
            get_system_info,
            get_stage_position,
            get_stage_orientation,
            get_microscope_state,
            get_milling_angle,
            get_milling_state,
            estimate_milling_time,
            stop_milling,
            acquire_image_preview,
            move_stage_relative,
            move_stage_absolute,
            stable_move,
            vertical_move,
            move_to_orientation,
            safe_move_stage_absolute,
            move_to_milling_angle,
            list_devices,
            get_parameter,
            set_parameter,
            call_command,
            list_services,
            get_service_parameter,
            call_service_command,
            call_service_query,
            autocontrast,
            get_app_status,
            get_app_queue,
            get_experiment_summary,
            get_task_history,
            get_run_summary,
            get_log,
            get_protocol,
            get_task_outputs,
            get_item_output_image,
            get_item_detail,
            get_grids,
            get_grid_detail,
            get_grid_markers,
            get_protocol_task_config,
            get_item_task_config,
            update_item_task_config,
            update_protocol_task_config,
            update_item_detail,
            set_task_schedule,
            apply_protocol_to_item,
            add_note,
            reorder_item_milling_stages,
            reorder_protocol_milling_stages,
            list_recent_experiments,
            get_events,
            get_display_images,
            get_pending_prompt,
            get_pending_reviews,
            decide_review,
            answer_prompt,
            stop_workflow,
            plan_grid_workflow,
            start_grid_workflow,
            set_task_supervision,
            set_workflow_options,
            requeue_task,
            start_workflow,
        )
    }
    # The catalog is the contract: refuse to start with an implementation gap.
    missing = {t.name for t in CATALOG} - set(implementations)
    if missing:
        raise RuntimeError(f"sidecar is missing implementations for: {sorted(missing)}")

    for name, fn in implementations.items():
        if name in armed:
            mcp.add_tool(fn, name=name, description=by_name[name].full_description)
    return mcp


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
        help="seconds to keep retrying for a server before giving up (default 30)",
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

    wait_s = 0.0 if args.check else max(0.0, args.wait)
    client, capabilities = _connect_with_retry(
        client_factory, args.url, args.token, wait_s
    )
    print(
        f"fibsem-mcp: connected to {client.base_url} "
        f"({capabilities.get('manufacturer')}, agent level "
        f"{capabilities.get('level')})",
        file=sys.stderr,
    )
    if args.check:
        return
    build_sidecar(client, capabilities).run()


if __name__ == "__main__":
    main()
