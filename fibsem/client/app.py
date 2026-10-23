"""The App Server's ``/app``: the session of the app it hosts, by the parts it serves.

Its calls are the agent's tools (`fibsem.ai.agent.catalog`), so a script, an app off
the PC and the agent make the same calls, with the same arguments.
"""

import logging
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import httpx

from fibsem.ai.agent.catalog import CATALOG, ToolSpec, tools_for_capabilities
from fibsem.client import tools
from fibsem.client.transport import Transport, answer_of
from fibsem.codec import decode
from fibsem.devices.wire import type_from_path
from fibsem.server.responses import answer_class

_APP_TOOLS = {spec.name: spec for spec in CATALOG if spec.router == "app"}


class App:
    """The app the server hosts: its core and the optional parts ``/capabilities``
    lists, read once on first use (`refresh` reads them again).

    Each of its tools is a method: ``app.proposal_list()``,
    ``app.item_get(item_name="Lamella-01")``, answering the class its route
    declares, as the app writes it (an AutoLamella item is a ``LamellaDetail``).
    An acting call names its tool to the server, which says it in the operator's
    words.
    """

    def __init__(self, transport: Transport):
        self._transport = transport
        self._capabilities: Optional[Dict[str, Any]] = None
        self._tools: Dict[str, ToolSpec] = {}
        self._answer_types: Dict[type, type] = {}

    def refresh(self) -> None:
        capabilities = self._transport.fetch("GET", "/capabilities")
        self._capabilities = capabilities
        self._tools = {
            spec.name: spec
            for spec in tools_for_capabilities(capabilities)
            if spec.router == "app"
        }
        self._answer_types = _answer_types(capabilities.get("answer_types") or {})

    def _read(self) -> Dict[str, Any]:
        if self._capabilities is None:
            self.refresh()
        return self._capabilities

    @property
    def served(self) -> bool:
        """Whether the server hosts an app."""
        return bool(self._read().get("routers", {}).get("app"))

    @property
    def name(self) -> str:
        """The app, as the operator's window names it."""
        return self._read().get("app_name", "")

    @property
    def parts(self) -> List[str]:
        """The optional parts it serves (`fibsem.server.parts`)."""
        return list(self._read().get("parts") or [])

    @property
    def tools(self) -> Dict[str, ToolSpec]:
        """Its tools by name: the core's and its parts'."""
        self._read()
        return self._tools

    def call(self, tool: str, /, **arguments: Any) -> Any:
        """Call *tool* with its arguments: the answer as the class its route
        declares, a refusal raised as the exception it names. A key the class
        lacks is an error, not dropped: the route and its class have drifted."""
        spec = self.tools.get(tool)
        if spec is None:
            raise AttributeError(self._missing(tool))
        data = self.fetch(tool, **arguments)
        if not spec.returns:
            return data
        return decode(answer_class(spec.answers, self._answer_types), data, strict=True)

    def fetch(self, tool: str, /, **arguments: Any) -> Any:
        """*tool*'s answer as the server sent it (JSON data), a refusal raised."""
        method, path = self._route(tool, arguments)[:2]
        return answer_of(self.send(tool, arguments), f"{method} {path}")

    def send(
        self,
        tool: str,
        arguments: Dict[str, Any],
        evidence: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> httpx.Response:
        """The server's answer to *tool* with *arguments*, whatever its status
        (`tools.send`)."""
        if tool not in _APP_TOOLS:
            raise AttributeError(f"no app tool '{tool}'")
        return tools.send(self._transport, tool, arguments, evidence, headers)

    @staticmethod
    def _route(tool: str, arguments: Dict[str, Any]):
        if tool not in _APP_TOOLS:
            raise AttributeError(f"no app tool '{tool}'")
        return tools.route(tool, arguments)

    def _missing(self, tool: str) -> str:
        if not self.served:
            return "the server hosts no app"
        return f"{self.name} has no tool '{tool}' (parts served: {self.parts})"

    def __getattr__(self, name: str) -> Callable[..., Any]:
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self.tools:
            raise AttributeError(self._missing(name))
        return partial(self.call, name)

    def __dir__(self) -> List[str]:
        return sorted({*super().__dir__(), *self.tools})

    def __repr__(self) -> str:
        return f"<App {self.name!r} parts={self.parts}>"


def _answer_types(paths: Dict[str, str]) -> Dict[type, type]:
    """The app's own answer classes /capabilities names (by type path), each
    for the base class it stands for. One this fibsem can't import, or that
    isn't the base's subclass, is left out: its answers read as the base."""
    found = {}
    for base_path, own_path in paths.items():
        try:
            base, own = type_from_path(base_path), type_from_path(own_path)
        except (ValueError, ImportError, AttributeError):
            logging.debug("no answer class %s here", own_path, exc_info=True)
            continue
        if isinstance(own, type) and isinstance(base, type) and issubclass(own, base):
            found[base] = own
    return found
