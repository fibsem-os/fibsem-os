"""The fibsem MCP on the gateway's port: ``/mcp/fibsem``, and its watch at
``/mcp/fibsem/watch`` (`fibsem.ai.agent.mcp.served`).

An agent on another computer then needs only an MCP client and the SSH
tunnel, not fibsem installed. The MCP runs as a child of the gateway, as the
Agent Server does (`fibsem.server.agent.process`), never inside the gateway's
own loop: a stuck tool can't slow the pass-through or Stop. It is started the
first time something calls ``/mcp``, so a gateway nobody uses it on runs no
more than before, then kept running and started again if it ends. The gateway
stops it when it closes.

Every call is passed on as the App Server's are (`fibsem.gateway.passthrough`),
with the secret the gateway started it with. A connect token is answered
here, 403 ``connect_first``. The MCP calls the gateway's own port with the
caller's token, so each tool call is checked as any agent's call is.

Without the ``mcp`` extra installed on this PC, ``/mcp`` answers 503
``no_mcp``.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
from typing import Callable, Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from fibsem.gateway import child_secrets
from fibsem.gateway.passthrough import PassThrough
from fibsem.server import discovery
from fibsem.server.agent.process import AgentHostProcess

logger = logging.getLogger(__name__)

__all__ = ["NO_MCP", "GatewayMcp"]

TARGET = "fibsem.ai.agent.mcp.served:serve_for_gateway"
NO_MCP = "no_mcp"
# How long the first call waits for the MCP to start.
START_WAIT_S = 30.0


def _found() -> Optional[dict]:
    return discovery.read_discovery_file(discovery.DISCOVERY_DIR / "mcp-server.json")


def mcp_installed() -> bool:
    """Whether this PC has the ``mcp`` extra the served MCP needs."""
    return importlib.util.find_spec("mcp") is not None


def _refused(status: int, message: str) -> JSONResponse:
    return JSONResponse(
        {"detail": {"error_type": NO_MCP, "message": message}}, status_code=status
    )


class GatewayMcp:
    """The gateway's MCP: started by the first call to ``/mcp`` (`forward`),
    kept running by `ensure` (a gateway watcher), stopped by `close`."""

    def __init__(
        self,
        python: str,
        own_url: str,
        process: Optional[AgentHostProcess] = None,
        is_connect: Callable[[Optional[str]], bool] = lambda token: False,
        installed: Callable[[], bool] = mcp_installed,
        transport=None,
        start_wait_s: float = START_WAIT_S,
        poll_s: float = 0.2,
    ) -> None:
        self._process = process or AgentHostProcess(
            python,
            found=_found,
            args=("--gateway", own_url),
            target=TARGET,
            name="mcp",
            what="MCP",
        )
        self._installed = installed
        self._wanted = False
        self._start_wait_s = start_wait_s
        self._poll_s = poll_s
        self._passthrough = PassThrough(
            self.url,
            transport=transport,
            is_connect=is_connect,
            secret=self._secret,
            read_timeout_s=None,
        )

    def url(self) -> Optional[str]:
        found = self._process.found()
        return found.get("url") if found else None

    def _secret(self) -> Optional[str]:
        return child_secrets.for_server(self._process.found())

    @property
    def wanted(self) -> bool:
        """Whether something has called ``/mcp`` since the gateway started."""
        return self._wanted

    def ensure(self) -> None:
        """Start the MCP again if it ended, once something has used it."""
        if self._wanted:
            self._process.ensure()

    async def _started(self) -> bool:
        """Start the MCP if none runs, and wait up to `START_WAIT_S` for it."""
        self._wanted = True
        loop = asyncio.get_running_loop()
        waited = 0.0
        while self.url() is None:
            if waited >= self._start_wait_s:
                return False
            await loop.run_in_executor(None, self._process.ensure)
            await asyncio.sleep(self._poll_s)
            waited += self._poll_s
        return True

    async def forward(self, request: Request):
        if not self._installed():
            return _refused(
                503,
                "This PC has no MCP: install fibsem with the mcp extra "
                '(pip install "fibsem[mcp]") and restart the gateway.',
            )
        if not await self._started():
            return _refused(
                503, "The gateway's MCP didn't start: see its log in logs/gateway."
            )
        return await self._passthrough.forward(request)

    def router(self) -> APIRouter:
        router = APIRouter()
        methods = ["GET", "POST", "DELETE"]
        router.add_api_route(
            "/mcp/{rest:path}", self.forward, methods=methods, include_in_schema=False
        )
        return router

    async def aclose(self) -> None:
        await self._passthrough.close()

    def close(self) -> None:
        self._process.close()
