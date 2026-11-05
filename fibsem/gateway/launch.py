"""Open the gateway's page at this PC: the "Agent Gateway" desktop shortcut
and ``fibsem-gateway open`` run it.

It starts the gateway when none runs, then opens its page in the browser signed
in as the operator, as the tray's Open does. Nothing else starts: the operator
picks the microscope, the app and the agent on the page.
"""

from __future__ import annotations

import sys
import time
import webbrowser
from typing import Any, Callable, Optional

from fibsem.gateway.gateway import read_gateway_file
from fibsem.gateway.main import call, start_gateway
from fibsem.server.operator_token import read_operator_token
from fibsem.tools.apps import GATEWAY
from fibsem.unreleased import refuse

__all__ = ["OpenFailed", "gateway_page", "open_gateway", "open_gateway_page"]

GATEWAY_WAIT_S = 60.0
POLL_S = 1.0


class OpenFailed(Exception):
    """Why the page didn't open, in words for the operator."""


def _answers(ask: Callable[..., Any]) -> bool:
    try:
        return ask("GET", "/gateway", timeout=10.0) is not None
    except (OSError, ValueError):
        return False


def gateway_page(
    ask: Callable[..., Any] = call,
    start: Callable[[], None] = start_gateway,
    found: Callable[[], Optional[dict]] = read_gateway_file,
    token: Callable[[], Optional[str]] = read_operator_token,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """The gateway's page, signed in as the operator; the gateway started
    first when none runs. Raises `OpenFailed`."""
    if not _answers(ask):
        start()
        deadline = clock() + GATEWAY_WAIT_S
        while not _answers(ask):
            if clock() > deadline:
                raise OpenFailed(
                    "The gateway didn't start. Run `fibsem-gateway` in a terminal "
                    "to see why."
                )
            sleep(POLL_S)
    gateway = found()
    if gateway is None:
        raise OpenFailed("The gateway stopped.")
    return f"{gateway['url']}/ui/gateway#token={token() or ''}"


def open_gateway(open_page: Callable[[str], Any] = webbrowser.open) -> int:
    """`gateway_page` in the browser; a failure printed and shown in a dialog,
    since a desktop shortcut has no console to read."""
    try:
        open_page(gateway_page())
    except OpenFailed as failed:
        print(failed)
        _show(str(failed))
        return 1
    return 0


def _show(text: str) -> None:
    """*text* in a dialog, when Qt is there."""
    from fibsem.tools.desktop_launch import _say

    try:
        _say(GATEWAY, text)
    except ImportError:
        pass


def open_gateway_page() -> None:
    """``fibsem-gateway-open``: the "Agent Gateway" desktop shortcut."""
    refuse("fibsem-gateway-open")
    sys.exit(open_gateway())
