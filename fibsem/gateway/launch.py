"""One launch for the operator: the gateway, the app with its window, the
microscope connected, ``fibsem-agent`` at Watch, and the app's Agent page open
in the browser. ``fibsem-gateway launch`` and the "AutoLamella with the agent"
desktop shortcut run it.

Each step goes through the gateway's own routes, as its page does, and a step
already done is left as it is, so a second launch only opens the page. A step
that fails stops the launch and says why; the steps before it stay done. The
level is never raised here: the operator raises it on the Agent page.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import webbrowser
from typing import Any, Callable, Optional, Tuple
from urllib.error import HTTPError

from fibsem.ai.agent.profiles import DEFAULT_PROFILE
from fibsem.gateway.gateway import OFF
from fibsem.gateway.main import call, start_gateway
from fibsem.tools.apps import apps

__all__ = [
    "Launch",
    "LaunchFailed",
    "autolamella",
    "launch",
    "launch_app",
    "launch_options",
]

GATEWAY_WAIT_S = 60.0
APP_WAIT_S = 180.0
CONNECT_WAIT_S = 300.0
POLL_S = 1.0


class LaunchFailed(Exception):
    """A step that didn't happen, and why, in words for the operator."""


def _refusal(error: HTTPError) -> Tuple[Optional[str], str]:
    """A refusal's error type and message (its body reads once)."""
    try:
        detail = json.loads(error.read().decode("utf-8"))["detail"]
        return detail["error_type"], detail["message"]
    except (ValueError, KeyError, TypeError):
        return None, f"the gateway answered {error.code}"


class Launch:
    """The steps, over *call* (`fibsem.gateway.main.call`'s shape)."""

    def __init__(
        self,
        call: Callable[..., Any] = call,
        say: Callable[[str], None] = print,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        start_gateway: Callable[[], None] = start_gateway,
    ) -> None:
        self._call = call
        self._start_gateway = start_gateway
        self._say = say
        self._clock = clock
        self._sleep = sleep

    def _ask(self, step: str, method: str, route: str, body=None, timeout=10.0):
        try:
            answer = self._call(method, route, body, timeout=timeout)
        except HTTPError as error:
            raise LaunchFailed(f"{step}: {_refusal(error)[1]}") from None
        except (OSError, ValueError) as error:
            raise LaunchFailed(
                f"{step}: the gateway didn't answer ({error})."
            ) from None
        if answer is None:
            raise LaunchFailed(f"{step}: the gateway stopped.")
        return answer

    def gateway(self) -> None:
        """The gateway running on this PC, started when none is."""
        if self._ask_quietly() is not None:
            return
        self._say("Starting the gateway…")
        self._start_gateway()
        deadline = self._clock() + GATEWAY_WAIT_S
        while self._clock() < deadline:
            self._sleep(POLL_S)
            if self._ask_quietly() is not None:
                return
        raise LaunchFailed(
            "Start the gateway: it didn't answer. Run `fibsem-gateway` in a "
            "terminal to see why."
        )

    def _ask_quietly(self) -> Optional[dict]:
        try:
            return self._call("GET", "/gateway", timeout=10.0)
        except (OSError, ValueError):
            return None

    def app(self, key: str) -> None:
        """The app running, with its window when the gateway starts it."""
        state = self._ask("Start the app", "GET", "/gateway")
        if state["mode"] == OFF:
            self._say("Starting the app…")
            self._ask(
                "Start the app",
                "POST",
                "/gateway/start",
                {"app": key, "window": True, "via": "desktop launch"},
                timeout=30.0,
            )
        elif state.get("app") != key:
            raise LaunchFailed(
                f"Start the app: {state.get('app')} is running. Stop it on the "
                "gateway's page first."
            )

    def _connection(self, deadline: float) -> dict:
        """The host's connection, once the host answers."""
        while True:
            try:
                return self._call("GET", "/gateway/microscope/connection", timeout=10.0)
            except HTTPError as error:
                error_type, message = _refusal(error)
                if error_type != "no_app" or self._clock() > deadline:
                    raise LaunchFailed(f"Wait for the app: {message}") from None
            except (OSError, ValueError) as error:
                raise LaunchFailed(f"Wait for the app: {error}") from None
            self._sleep(POLL_S)

    def microscope(self, configuration: Optional[str]) -> None:
        """A microscope connected: *configuration*, or the default one."""
        connection = self._connection(self._clock() + APP_WAIT_S)
        if connection["connected"]:
            return
        if not connection.get("connecting"):
            self._say("Connecting the microscope…")
            body = {"configuration": configuration} if configuration else {}
            self._ask(
                "Connect the microscope", "POST", "/gateway/microscope/connect", body
            )
        deadline = self._clock() + CONNECT_WAIT_S
        while True:
            self._sleep(POLL_S)
            connection = self._connection(deadline)
            if connection["connected"]:
                return
            run = connection.get("run") or {}
            if not connection.get("connecting") and run.get("state") != "running":
                error = connection.get("error") or "it didn't connect"
                raise LaunchFailed(f"Connect the microscope: {error}")
            if self._clock() > deadline:
                raise LaunchFailed("Connect the microscope: it is still connecting.")

    def agent(self, profile: str) -> None:
        """``fibsem-agent`` running; a new one starts at Watch."""
        state = self._ask("Start the agent", "GET", "/gateway/agent")
        if state.get("running") is not None:
            return
        self._say("Starting the agent at Watch…")
        self._ask(
            "Start the agent",
            "POST",
            "/gateway/agent/start",
            {"profile": profile, "level": "watch", "via": "desktop launch"},
            timeout=60.0,
        )

    def page(self) -> str:
        """The app's Agent page, signed in as the operator."""
        return self._ask("Open the Agent page", "GET", "/gateway/app/agent-page")["url"]


def launch(
    key: str = "autolamella",
    configuration: Optional[str] = None,
    profile: str = DEFAULT_PROFILE,
    steps: Optional[Launch] = None,
    open_page: Callable[[str], Any] = webbrowser.open,
) -> None:
    """The gateway, the app, the microscope, the agent at Watch, then the
    Agent page in the browser. Raises `LaunchFailed` at the first step that
    fails."""
    steps = steps or Launch()
    steps.gateway()
    steps.app(key)
    steps.microscope(configuration)
    steps.agent(profile)
    open_page(steps.page())


def launch_options() -> argparse.ArgumentParser:
    """``--configuration`` and ``--profile``; no level, which stays at Watch."""
    options = argparse.ArgumentParser(add_help=False)
    options.add_argument(
        "--configuration", help="The microscope configuration (the default without)."
    )
    options.add_argument(
        "--profile", default=DEFAULT_PROFILE, help="The agent profile."
    )
    return options


def launch_app(
    key: str, configuration: Optional[str] = None, profile: str = DEFAULT_PROFILE
) -> int:
    """`launch`, with a failure also shown in a dialog: a desktop shortcut has
    no console to read."""
    try:
        launch(key, configuration, profile)
    except LaunchFailed as failed:
        print(failed)
        _show(apps()[key], str(failed))
        return 1
    return 0


def _show(app, text: str) -> None:
    """*text* in a dialog, when Qt is there."""
    from fibsem.tools.desktop_launch import _say

    try:
        _say(app, text)
    except ImportError:
        pass


def autolamella(argv: Optional[list] = None) -> None:
    """``fibsem-autolamella-agent``: the "AutoLamella with the agent" shortcut."""
    parser = argparse.ArgumentParser(
        prog="fibsem-autolamella-agent",
        description="Start AutoLamella, connect the microscope and start the agent "
        "at Watch, then open the Agent page.",
        parents=[launch_options()],
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    sys.exit(launch_app("autolamella", args.configuration, args.profile))
