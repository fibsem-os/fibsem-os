"""The agent's tools as route calls: one tool call, its route as the catalog maps it
(`fibsem.ai.agent.catalog`), sent over a transport."""

from typing import Any, Dict, Optional
from urllib.parse import quote

import httpx

from fibsem.ai.agent.calls import headers_for
from fibsem.ai.agent.catalog import CATALOG, route_call

TOOLS = {spec.name: spec for spec in CATALOG}


def send(
    transport: Any,
    tool: str,
    arguments: Dict[str, Any],
    evidence: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
) -> httpx.Response:
    """The server's answer to *tool* with *arguments*, whatever its status.
    An acting call names its tool, and the *evidence* it cites, to the server,
    or sends *headers* in their place. Raises ValueError for arguments its route
    cannot take."""
    method, path, body = route(tool, arguments)
    if method == "GET":  # a read's other arguments are its query
        query = {k: v for k, v in (body or {}).items() if v is not None}
        return transport.request(method, path, params=query or None)
    if headers is None:
        headers = headers_for(tool, arguments, evidence=evidence)
    return transport.request(method, path, json=body, headers=headers)


def route(tool: str, arguments: Dict[str, Any]):
    """*tool*'s route call (method, path, body), its path parameters quoted."""
    spec = TOOLS.get(tool)
    if spec is None:
        raise AttributeError(f"no tool '{tool}'")
    return route_call(spec, arguments, encode=lambda value: quote(value, safe=""))
