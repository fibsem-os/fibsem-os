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
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, Optional, Tuple

from fibsem.ai.agent.catalog import (
    analysis,
    app,
    coating,
    grids,
    imaging,
    items,
    knowledge,
    microscope,
    milling,
)
from fibsem.ai.agent.catalog.spec import EVIDENCE_PARAM, GROUP_TITLES, ToolSpec

__all__ = [
    "CATALOG",
    "EVIDENCE_PARAM",
    "INSTRUMENT_ENV",
    "GROUP_TITLES",
    "NOT_TOOLS",
    "Devices",
    "ToolSpec",
    "needs",
    "route_call",
    "tools_for_capabilities",
]

# Where fibsem-agent tells the sidecar its playbook's instrument kind, so the
# sidecar offers the tools on that kind's devices (`Devices`).
INSTRUMENT_ENV = "FIBSEM_AGENT_INSTRUMENT"

CATALOG: Tuple[ToolSpec, ...] = (
    microscope.TOOLS
    + imaging.TOOLS
    + milling.TOOLS
    + coating.TOOLS
    + grids.TOOLS
    + items.TOOLS
    + app.TOOLS
    + analysis.TOOLS
    + knowledge.TOOLS
)

# Every route the server serves is a tool above or here, with why it is not one
# (tests/server/test_catalog_discovery.py). A new route fails that test until
# someone decides. Path parameter names don't matter.
NOT_TOOLS: Dict[str, str] = {
    "GET /app/experiment/replay": "The web page's replay timeline; app_get_events and item_get_history give the agent the same record.",
    "GET /app/experiment/replay/{index}": "The web page's replay scene, images and overlays at one recorded action; the agent reads the record and images directly.",
    "GET /health": "Open liveness check, no token; the agent's first read is server_get_capabilities.",
    "GET /app/agent/session": "The agent's own report, read back by the app and the dashboard; the agent knows what it reported.",
    "GET /app/knowledge/{section}/{id}/images/{name}": "One knowledge image as a file, for the web page; knowledge_get answers the agent its images as pictures.",
    "DELETE /app/knowledge/{section}/{id}": "Removing a knowledge entry is the operator's only (operator token).",
    "POST /app/knowledge/{section}/{id}/keep": "Keeping an entry the agent wrote is the operator's only (operator token).",
    "POST /app/knowledge/{section}/{id}/resolve": "Marking a fault fixed is the operator's only (operator token); the agent rewrites the entry with what the operator says.",
    "GET /app/library/{kind}": "The web page's Knowledge window lists the shipped and the lab's playbooks, skills and subagents; the agent loads its own when it starts.",
    "GET /app/library/{kind}/{name}": "One playbook, skill or subagent's file for the Knowledge window; the agent has the skills it loaded.",
    "GET /app/library/options": "The choices the Knowledge window's playbook and subagent forms offer; the agent doesn't edit them.",
    "POST /app/library/{kind}/{name}": "Making a new playbook, skill or subagent is the operator's only (operator token).",
    "PUT /app/library/{kind}/{name}": "Editing playbooks, skills and subagents is the operator's only (operator token); an edit to a shipped one saves the lab's copy.",
    "DELETE /app/library/{kind}/{name}": "Removing the lab's own copy is the operator's only (operator token).",
    "GET /app/agent/transcript": "What fibsem-agent printed, read back by the web page; the agent has its own context.",
    "POST /app/agent/transcript": "fibsem-agent's harness sends what it printed; not the agent's to call.",
    "PUT /app/agent/plan": "The operator edits the whole plan from the Agent page; the agent writes it a section at a time (plan_set_section).",
    "POST /app/agent/read": "fibsem-agent's harness, or fibsem-mcp watch, says which of the operator's messages the agent has read; not the agent's to call.",
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
    "POST /app/experiment/upload": "The operator adding a file from the web page (operator token only); the agent reads it with experiment_read_file.",
    "GET /app/experiment/planes/{path}": "Raw pixel planes the web viewer draws; the agent reads images as pictures with experiment_read_file.",
    "POST /app/experiment/evidence": "Every acting tool's evidence argument: the sidecar keeps it here first, then the call cites it.",
    "GET /app/events/history": "The web Events page's read of the run before the live buffer; an agent reads a finished run from its experiment folder.",
    "GET /app/experiment/export/image/options": "The web image export's form; an export is a file for a person, the agent reads images through experiment_read_file.",
    "POST /app/experiment/export/image": "An image exported for a person to download; the agent reads images through experiment_read_file.",
    "GET /app/experiment/export/animation/frames": "The web animation export's frames; an animation is a file for a person.",
    "POST /app/experiment/export/animation": "A lamella's animation for a person to download; the agent reads its images through experiment_read_file.",
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


_DEVICE_PATH = re.compile(r"^/microscope/devices/(\w+)(?:/|$)")
_SERVICE_PATH = re.compile(r"^/microscope/services/(\w+)(?:/|$)")
_MICROSCOPE_PATHS = ("/microscope", "/microscope/state")


def needs(tool: ToolSpec) -> FrozenSet[str]:
    """What an instrument must serve for *tool* to be offered: its own
    ``needs``, else what its path reaches. A named device ("device:stage") or
    service ("service:milling"), "microscope" for the FIB-SEM as a whole, and
    nothing for a tool on any device or service ({device}), the app or the
    analysis services."""
    if tool.needs:
        return frozenset(tool.needs)
    for pattern, kind in ((_DEVICE_PATH, "device"), (_SERVICE_PATH, "service")):
        match = pattern.match(tool.path)
        if match:
            return frozenset({f"{kind}:{match.group(1)}"})
    return frozenset({"microscope"}) if tool.path in _MICROSCOPE_PATHS else frozenset()


@dataclass(frozen=True)
class Devices:
    """What a kind of instrument has, as its ``devices.yaml`` says
    (`fibsem.ai.agent.instruments.devices`): a FIB-SEM, which gets every
    tool, or its devices and services by name."""

    microscope: bool = False
    devices: FrozenSet[str] = field(default_factory=frozenset)
    services: FrozenSet[str] = field(default_factory=frozenset)

    def offers(self, tool: ToolSpec) -> bool:
        if self.microscope:
            return True
        have = {f"device:{d}" for d in self.devices} | {
            f"service:{s}" for s in self.services
        }
        return needs(tool) <= have


def tools_for_capabilities(
    capabilities: Dict, devices: Optional[Devices] = None
) -> Tuple[ToolSpec, ...]:
    """Catalog entries to offer given a full /capabilities payload.

    Every tool whatever the agent's level: the level changes mid-session and
    the server's 403 is the gate. The microscope's tools too while
    its routes are off: the operator can turn them on mid-session, and until
    then each answers 404 ``microscope_off`` saying so, where tools fixed at
    start would need the agent to reconnect. Only the app's tools depend on
    the routers and parts: a server hosts an application or not for its whole
    life.

    With *devices*, what the playbook's instrument kind has, only the tools
    on them: an instrument doesn't change mid-session, and the agent may
    start before it is connected. None offers every tool.
    """
    routers = dict(capabilities.get("routers", {}), base=True, microscope=True)
    # A server from before the parts lists none and serves them all.
    parts = capabilities.get("parts")
    return tuple(
        t
        for t in CATALOG
        if routers.get(t.router, False)
        and (not t.part or parts is None or t.part in parts)
        and (devices is None or devices.offers(t))
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
