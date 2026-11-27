"""An instrument's Agent page, served by the Hub.

The page a person opens at the Hub (``/ui/hub/agent/<instrument>``) is the
same Agent page a microscope PC serves. Its calls come to the Hub under
``/hub/instruments/<name>/`` and go on to that instrument's gateway, through
its tunnel, with the token that gateway gave this Hub; the answer comes back
unchanged and streamed, as the gateway's own pass-through does it
(`fibsem.gateway.passthrough`, whose pieces this uses). The browser talks to
the Hub only, so nobody pairs it with the gateway.

The gateway decides what a Hub may ask (`fibsem.gateway.hub_remote`): reads, a
message, an answer, Stop agent, and the level within what the microscope PC
allows. Its refusal comes back in its own words. A gateway that no longer
knows this Hub answers 401: that is passed back as 409 ``not_paired``, never
as a 401 of the Hub's own, which would sign the page out.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import httpx
from fastapi import Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from fibsem.gateway.gateway import GatewayRefused
from fibsem.gateway.hub_remote import act_name
from fibsem.gateway.passthrough import (
    CONNECT_TIMEOUT_S,
    PASSED,
    READ_TIMEOUT_S,
    ROOT_PASSED,
    WAIT_HEADER,
    call_timeout,
    passed_headers,
    streamed,
)
from fibsem.hub.hub import Hub
from fibsem.util.tunnels import TunnelDown

logger = logging.getLogger(__name__)

__all__ = ["AgentPageRelay", "is_passed"]


def is_passed(route: str) -> bool:
    """Whether *route* (no leading slash) is one a gateway passes on to its
    app: the Agent page's own."""
    first = route.split("/", 1)[0]
    return first in PASSED or route in ROOT_PASSED


def _refused(status: int, error_type: str, message: str) -> JSONResponse:
    return JSONResponse(
        {"detail": {"error_type": error_type, "message": message}}, status_code=status
    )


class AgentPageRelay:
    """Passes the Agent page's calls for an instrument on to its gateway.
    *on_act* is told of each thing done to the agent: the instrument, what,
    and who at the Hub."""

    def __init__(
        self,
        hub: Hub,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        on_act: Optional[Callable[[str, str, str], None]] = None,
    ) -> None:
        self._hub = hub
        self._on_act = on_act
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(READ_TIMEOUT_S, connect=CONNECT_TIMEOUT_S),
            transport=transport,
        )

    async def close(self) -> None:
        await self._client.aclose()

    def _where(self, name: str):
        """*name*'s gateway's address and this Hub's token for it."""
        instrument = self._hub.instrument(name)
        token = self._hub.token(name)
        if token is None:
            raise GatewayRefused(
                409, "not_paired", f"Pair this hub with {name}'s gateway first."
            )
        return self._hub.reach(instrument).rstrip("/"), token

    async def forward(self, name: str, route: str, request: Request, who: str):
        try:
            # Off the event loop: opening a tunnel can take seconds.
            url, token = await run_in_threadpool(self._where, name)
        except GatewayRefused as refusal:
            return _refused(refusal.status, refusal.error_type, str(refusal))
        except TunnelDown as down:
            return _refused(503, "unreachable", str(down))
        path = "/" + route
        headers = [
            (k, v) for k, v in passed_headers(request) if k.lower() != "authorization"
        ]
        headers.append(("authorization", f"Bearer {token}"))
        sent = self._client.build_request(
            request.method,
            httpx.URL(f"{url}{path}").copy_with(
                query=request.url.query.encode("utf-8") or None
            ),
            headers=headers,
            content=request.stream(),
            timeout=call_timeout(request.headers.get(WAIT_HEADER)),
        )
        try:
            answer = await self._client.send(sent, stream=True)
        except httpx.HTTPError as error:
            return _refused(
                503, "unreachable", f"{name}'s gateway didn't answer: {error!r}"
            )
        if answer.status_code == 401:
            await answer.aclose()
            return _refused(
                409,
                "not_paired",
                f"{name}'s gateway no longer knows this hub: pair again.",
            )
        did = act_name(request.method, path)
        if did is not None and answer.is_success and self._on_act is not None:
            try:
                self._on_act(name, did, who)
            except Exception:  # noqa: BLE001 - a record never fails the call
                logger.warning("could not record a passed call", exc_info=True)
        return streamed(answer)
