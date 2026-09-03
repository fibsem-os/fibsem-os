"""fibsem-mcp: the MCP sidecar that connects an agent to a fibsem server.

Runs on the agent host (the machine running Claude Code / Claude Desktop) and
proxies MCP tool calls to a fibsem server over HTTP with a bearer token. It
owns no microscope connection and enforces nothing itself — the server is the
security boundary; the sidecar merely registers the tools the server's
``/capabilities`` says are available.

Connection resolution, in order: ``--url``/``--token`` arguments, the
``FIBSEM_SERVER_URL``/``FIBSEM_SERVER_TOKEN`` environment variables, then the
same-machine discovery file a running server maintains.

Register with Claude Code:

    claude mcp add fibsem -- fibsem-mcp
    # or explicitly:
    claude mcp add fibsem -- fibsem-mcp --url http://hydra-support:8001 --token <token>

Requires the [mcp] extra (Python 3.10+): pip install fibsem[mcp]
"""

import argparse
import base64
import json
import os
import sys
from typing import Optional, Tuple

from fibsem.server.catalog import CATALOG, tools_for_capabilities

_TIMEOUT_S = 120.0  # acquisitions and moves on real hardware are slow


_NO_SERVER_MSG = (
    "fibsem-mcp: no server found. Pass --url and --token, set "
    "FIBSEM_SERVER_URL/FIBSEM_SERVER_TOKEN, or start a fibsem server on "
    "this machine (it writes ~/.fibsem/agent-server.json)."
)


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


def _connect_with_retry(client_factory, url, token, wait_s, sleep_fn=None):
    """Resolve + reach a server, retrying until the deadline.

    MCP clients launch the sidecar at session start, often before the human has
    started the server — dying instantly turns a harmless ordering mistake into
    a CONNECTION_CLOSED (learned in the very first live session). So: keep
    re-resolving (the discovery file appears when the server starts) and
    re-trying /capabilities until ``wait_s`` runs out.

    Returns (client, capabilities). Exits with guidance on timeout.
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
                return client, capabilities
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


def _call(client, method, path, payload=None):
    """Call the server; refusals come back as readable text, not exceptions.

    Agents act sensibly on a structured refusal string; a raw traceback for a
    403 teaches them nothing.
    """
    resp = client.request(method, path, json=payload)
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        return None, f"refused ({resp.status_code}): {json.dumps(detail)}"
    return resp.json(), None


def build_sidecar(client, capabilities):
    """Build the MCP server over an httpx-compatible client.

    ``capabilities`` is the full /capabilities payload; a tool registers only
    when its scope is armed AND its router is mounted (an app tool exists only
    when the server hosts an application). The server enforces regardless.
    """
    from mcp.server.mcpserver import Image, MCPServer

    mcp = MCPServer("fibsem")
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

    def last_image_preview(beam_type: str = "ELECTRON"):
        return _image_tool("last_image_preview")[1](beam_type=beam_type)

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

    def move_to_milling_angle(milling_angle_deg: float):
        data, err = _call(
            client,
            "POST",
            "/milling_angle/move",
            {"milling_angle_deg": milling_angle_deg},
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

    def get_task_outputs(item_name: str):
        return _app_get(f"/app/task_outputs/{item_name}")

    def get_item_detail(item_name: str):
        return _app_get(f"/app/items/{item_name}")

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

    def set_task_supervision(task_name: str, supervise: bool, supervisor: str = None):  # noqa: RUF013
        payload = {"task_name": task_name, "supervise": bool(supervise)}
        if supervisor is not None:
            payload["supervisor"] = supervisor
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
        return _app_get("/app/prompt")

    def answer_prompt(response: bool, nonce: int, value: Optional[dict] = None):
        body = {"response": bool(response), "nonce": int(nonce)}
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
            last_image_preview,
            move_stage_relative,
            move_stage_absolute,
            move_to_milling_angle,
            autocontrast,
            get_app_status,
            get_app_queue,
            get_experiment_summary,
            get_task_history,
            get_run_summary,
            get_protocol,
            get_task_outputs,
            get_item_detail,
            get_protocol_task_config,
            get_item_task_config,
            update_item_task_config,
            update_protocol_task_config,
            update_item_detail,
            set_task_schedule,
            apply_protocol_to_item,
            add_note,
            list_recent_experiments,
            get_events,
            get_display_images,
            get_pending_prompt,
            answer_prompt,
            stop_workflow,
            set_task_supervision,
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
            mcp.add_tool(fn, name=name, description=by_name[name].description)
    return mcp


def main(argv=None):
    try:
        import mcp  # noqa: F401
    except ImportError:
        sys.exit(
            "fibsem-mcp needs the [mcp] extra (Python 3.10+): pip install fibsem[mcp]"
        )
    import httpx

    parser = argparse.ArgumentParser(description="MCP sidecar for a fibsem server")
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
            headers={"Authorization": f"Bearer {token}"},
            timeout=_TIMEOUT_S,
        )

    wait_s = 0.0 if args.check else max(0.0, args.wait)
    client, capabilities = _connect_with_retry(
        client_factory, args.url, args.token, wait_s
    )
    print(
        f"fibsem-mcp: connected to {client.base_url} "
        f"({capabilities.get('manufacturer')}, scopes {capabilities.get('scopes')})",
        file=sys.stderr,
    )
    if args.check:
        return
    build_sidecar(client, capabilities).run()


if __name__ == "__main__":
    main()
