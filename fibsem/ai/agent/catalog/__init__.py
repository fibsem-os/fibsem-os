"""The tool catalog: the single source of truth for what agents can do.

The sidecar (``fibsem-mcp``) builds its MCP tool list from this module, and a
contract test asserts every entry maps onto a real route on the server app —
so the two surfaces cannot drift apart. Importable everywhere, including the
py3.8 floor: what a tool needs from the rest of fibsem it imports when called.

A tool lives in the module of its group (stage and beam tools in microscope and
imaging, item tools in items, ...), with its route call (``call``) and the line
its action says in an agent's request (``line``); `describe` says a request's
actions from them.

Levels mirror the server's enforcement (fibsem/server/auth.py). The sidecar
registers every tool the server's mounted routers serve, whatever the agent's
level: the level changes mid-session, and an MCP client may never
re-read its tool list, so the server's 403 is the gate and each description
names the level it needs.
"""

import re
from typing import Any, Callable, Dict, Optional, Tuple

from fibsem.ai.agent.catalog import (
    app,
    coating,
    grids,
    imaging,
    items,
    microscope,
    milling,
)
from fibsem.ai.agent.catalog.spec import EVIDENCE_PARAM, GROUP_TITLES, ToolSpec

__all__ = [
    "CATALOG",
    "EVIDENCE_PARAM",
    "GROUP_TITLES",
    "NOT_TOOLS",
    "ToolSpec",
    "route_call",
    "tools_for_capabilities",
]

CATALOG: Tuple[ToolSpec, ...] = (
    microscope.TOOLS
    + imaging.TOOLS
    + milling.TOOLS
    + coating.TOOLS
    + grids.TOOLS
    + items.TOOLS
    + app.TOOLS
)

# Every route the server serves is a tool above or here, with why it is not one
# (tests/server/test_catalog_discovery.py). A new route fails that test until
# someone decides. Path parameter names don't matter.
NOT_TOOLS: Dict[str, str] = {
    "GET /health": "Open liveness check, no token; the agent's first read is server_get_capabilities.",
    "GET /app/agent/session": "The agent's own report, read back by the app and the dashboard; the agent knows what it reported.",
    "GET /app/agent/transcript": "What fibsem-agent printed, read back by the web page; the agent has its own context.",
    "POST /app/agent/transcript": "fibsem-agent's runner sends what it printed; not the agent's to call.",
    "GET /dashboard/log": "The operator's event log page, HTML; app_get_events is the agent's read.",
    "GET /chat": "A redirect to the web app's Agent page, GET /ui/agent.",
    "GET /ui": "The operator's web app, HTML; it calls the operator's routes.",
    "GET /ui/{}": "The web app's pages and assets, served like GET /ui.",
    "GET /pair": "Pairing a computer with the operator's code: a person's computer, not the agent.",
    "POST /pair": "Pairing a computer with the operator's code: a person's computer, not the agent.",
    "POST /connect": "An agent on this PC asking to connect, with the discovery file's token: the client asks itself before any tool runs.",
    "POST /pair/request": "A browser asks the headless server for a pairing code: a person, not the agent.",
    "GET /pairing": "The operator's pairing (operator token only): the agent's token is refused.",
    "POST /pairing": "The operator opens pairing (operator token only): the agent's token is refused.",
    "DELETE /pairing": "The operator closes pairing (operator token only): the agent's token is refused.",
    "DELETE /pairing/computers/{computer_id}": "The operator unpairs a computer (operator token only).",
    "PATCH /pairing/computers/{computer_id}": "The operator sets what a paired hub may do (operator token only).",
    "PUT /pairing/always_allow_local": "The operator's setting to allow agents logged in to this PC without a click (operator token only).",
    "POST /app/experiment/evidence": "Every acting tool's evidence argument: the sidecar keeps it here first, then the call cites it.",
    "GET /app/events/history": "The event log page's read of the run before the live buffer; an agent reads a finished run from its experiment folder.",
    "GET /microscope/devices/{device}": "device_list carries every device's description.",
    "GET /microscope/devices/{device}/{parameter}/metadata": "device_list carries limits, choices and settable.",
    "GET /microscope/services/{service}": "service_list carries every service's description.",
    "GET /microscope/services/{service}/{parameter}/metadata": "service_list carries limits, choices and settable.",
    "GET /microscope/devices/{device}/values": "Not decided yet; device_get_parameter reads one value at a time.",
    "GET /microscope/services/{service}/values": "Not decided yet; service_get_parameter reads one value at a time.",
    "GET /microscope/control": "A remote app's control lease, held by its client (RemoteMicroscope.control); each agent call takes control itself.",
    "POST /microscope/control": "A remote app's control lease, held by its client (RemoteMicroscope.control); each agent call takes control itself.",
    "PUT /microscope/control/{lease}": "A remote app's control lease, held by its client (RemoteMicroscope.control); each agent call takes control itself.",
    "DELETE /microscope/control/{lease}": "A remote app's control lease, held by its client (RemoteMicroscope.control); each agent call takes control itself.",
    "POST /microscope/control/{lease}/lend": "A remote app's control lease, held by its client (RemoteMicroscope.control); each agent call takes control itself.",
    "DELETE /microscope/control/{lease}/lend": "A remote app's control lease, held by its client (RemoteMicroscope.control); each agent call takes control itself.",
    # The analysis services (fibsem.services.analysis): each query the agent
    # uses is a named tool (correlation_correlate, ...); the generic routes are
    # for scripts and the browser monitor.
    "GET /services": "The analysis services' listing; their queries are named tools.",
    "GET /services/{service}": "An analysis service's description; its queries are named tools.",
    "GET /services/{service}/values": "An analysis service's settings; none is an agent's to read yet.",
    "GET /services/{service}/{parameter}": "An analysis service's setting; none is an agent's to read yet.",
    "PUT /services/{service}/{parameter}": "An analysis service's setting; none is an agent's to change yet.",
    "GET /services/{service}/{parameter}/metadata": "An analysis service's setting limits; none is an agent's yet.",
    "POST /services/{service}/commands/{command}": "No analysis service has a command; each query is a named tool.",
    "POST /services/{service}/queries/{query}": "Each analysis service query the agent uses is a named tool (correlation_correlate, ...).",
}


def tools_for_capabilities(capabilities: Dict) -> Tuple[ToolSpec, ...]:
    """Catalog entries to offer given a full /capabilities payload.

    Every tool whatever the agent's level: the level changes mid-session and
    the server's 403 is the gate. The microscope's tools too while
    its routes are off: the operator can turn them on mid-session, and until
    then each answers 404 ``microscope_off`` saying so, where tools fixed at
    start would need the agent to reconnect. Only the app's tools depend on
    the routers and parts: a server hosts an application or not for its whole
    life.
    """
    routers = dict(capabilities.get("routers", {}), base=True, microscope=True)
    # A server from before the parts lists none and serves them all.
    parts = capabilities.get("parts")
    return tuple(
        t
        for t in CATALOG
        if routers.get(t.router, False)
        and (not t.part or parts is None or t.part in parts)
    )


# --- From a tool call to its route call ---
# The sidecar makes each tool's call this way, and an approved request makes the
# same call, so a request takes the arguments the tool does.

_PATH_PARAM = re.compile(r"{(\w+)}")


def route_call(
    spec: ToolSpec,
    arguments: Dict[str, Any],
    encode: Callable[[str], str] = lambda value: value,
) -> Tuple[str, str, Optional[dict]]:
    """The route call (method, path, body) a tool call stands for: the path's
    parameters from the arguments, the rest as the JSON body. Raises ValueError
    for a missing path parameter or a malformed argument.

    ``encode`` is applied to each path parameter: a client sending the path
    over HTTP percent-encodes them; a call into the ASGI app passes the decoded
    path and leaves them as they are."""
    args = dict(arguments)
    shape = spec.call
    if shape is not None:
        try:
            args, body = shape(args)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{spec.name}: bad or missing argument {exc}") from None
    names = _PATH_PARAM.findall(spec.path)
    missing = [n for n in names if not isinstance(args.get(n), str) or not args[n]]
    if missing:
        raise ValueError(f"{spec.name} needs {', '.join(missing)} (strings)")
    path = spec.path
    for name in names:
        path = path.replace("{" + name + "}", encode(args.pop(name)))
    if shape is None:
        body = args
    if not body:
        # A POST with no arguments still sends its (empty) object: a route
        # that reads a body refuses none at all.
        body = {} if spec.method in ("POST", "PUT") else None
    return spec.method, path, body
