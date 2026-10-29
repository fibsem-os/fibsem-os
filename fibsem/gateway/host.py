"""The gateway's calls to the fibsem host on this PC, as the operator: the
host's answer, or its refusal as a `GatewayRefused`. `HostMicroscope` is the
host's connect and disconnect, and the link to its Agent page, for the
gateway's page."""

from __future__ import annotations

from typing import Any, Callable, Optional

from fibsem.gateway.gateway import GatewayRefused
from fibsem.server.discovery import read_discovery_file
from fibsem.server.operator_token import read_operator_token

__all__ = ["HostMicroscope", "call_host", "microscope_connected"]


def call_host(
    url: str, method: str, route: str, body: Optional[dict] = None, timeout=10.0
) -> Any:
    """``method route`` on the host at *url*, its JSON answer."""
    import httpx

    token = read_operator_token()
    try:
        r = httpx.request(
            method,
            f"{url}{route}",
            json=body,
            headers={"Authorization": f"Bearer {token}"} if token else {},
            timeout=timeout,
        )
    except httpx.HTTPError as error:
        raise GatewayRefused(503, "no_app", f"The app didn't answer: {error}")
    if r.is_success:
        return r.json()
    try:
        detail = r.json()["detail"]
        raise GatewayRefused(r.status_code, detail["error_type"], detail["message"])
    except (ValueError, KeyError, TypeError):
        raise GatewayRefused(r.status_code, "app_refused", r.text) from None


def microscope_connected(url: str) -> bool:
    return bool(call_host(url, "GET", "/microscope/connection")["connected"])


class HostMicroscope:
    """The host's ``/microscope/connection``, ``connect`` and ``disconnect``,
    found by its discovery file."""

    def __init__(
        self,
        discovery: Callable[[], Optional[dict]] = read_discovery_file,
        call: Callable[..., Any] = call_host,
        token: Callable[[], Optional[str]] = read_operator_token,
    ) -> None:
        self._discovery = discovery
        self._call = call
        self._token = token

    def _url(self) -> str:
        found = self._discovery()
        if found is None:
            raise GatewayRefused(409, "no_app", "Start the app first.")
        return found["url"]

    def connection(self) -> dict:
        return self._call(self._url(), "GET", "/microscope/connection")

    def connect(self, configuration: Optional[str]) -> dict:
        body = {"configuration": configuration} if configuration else {}
        return self._call(self._url(), "POST", "/microscope/connect", body)

    def disconnect(self) -> dict:
        return self._call(self._url(), "POST", "/microscope/disconnect", {})

    def agent_page(self) -> str:
        """The host's Agent page with the operator token in the fragment, the
        link the window opens (`fibsem.server.hosting.chat_url`)."""
        return f"{self._url()}/ui/agent#token={self._token() or ''}"
