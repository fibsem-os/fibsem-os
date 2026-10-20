"""The App Server's ``/app``: the session of the app it hosts, by the parts it serves.

Its calls are the agent's tools (`fibsem.ai.agent.catalog`), so a script, an app off
the PC and the agent make the same calls, with the same arguments.
"""

from functools import partial
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import quote

from fibsem.ai.agent.calls import headers_for
from fibsem.ai.agent.catalog import ToolSpec, route_call, tools_for_capabilities
from fibsem.client.transport import Transport


class App:
    """The app the server hosts: its core and the optional parts ``/capabilities``
    lists, read once on first use (`refresh` reads them again).

    Each of its tools is a method: ``app.proposal_list()``,
    ``app.item_get_detail(item_name="Lamella-01")``. An acting call names its tool
    to the server, which says it in the operator's words.
    """

    def __init__(self, transport: Transport):
        self._transport = transport
        self._capabilities: Optional[Dict[str, Any]] = None
        self._tools: Dict[str, ToolSpec] = {}

    def refresh(self) -> None:
        capabilities = self._transport.fetch("GET", "/capabilities")
        self._capabilities = capabilities
        self._tools = {
            spec.name: spec
            for spec in tools_for_capabilities(capabilities)
            if spec.router == "app"
        }

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
        """Call *tool* with its arguments: the answer as data, a refusal raised
        as the exception it names."""
        spec = self.tools.get(tool)
        if spec is None:
            raise AttributeError(self._missing(tool))
        method, path, body = route_call(
            spec, arguments, encode=lambda value: quote(value, safe="")
        )
        if method == "GET":  # a read's other arguments are its query
            return self._transport.fetch(method, path, params=body)
        return self._transport.fetch(
            method, path, body, headers=headers_for(tool, arguments)
        )

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
