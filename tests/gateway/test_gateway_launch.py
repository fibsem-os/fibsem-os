"""The one launch (`fibsem.gateway.launch`): the gateway, the app with its
window, the microscope, the agent at Watch and its page, each step through the
gateway's routes and skipped when it is already done."""

import io
import json
from urllib.error import HTTPError

import pytest

from fibsem.gateway import main as gateway_main
from fibsem.gateway.gateway import OFF, RUNNING
from fibsem.gateway.launch import Launch, LaunchFailed, launch

PAGE = "http://127.0.0.1:8001/ui/agent#token=operator"


def _refused(status, error_type, message):
    body = json.dumps({"detail": {"error_type": error_type, "message": message}})
    return HTTPError("http://gateway", status, message, {}, io.BytesIO(body.encode()))


class FakeGateway:
    """The gateway's routes as `fibsem.gateway.main.call` reaches them: an app
    that takes a poll to come up, and a connect that takes a poll to finish."""

    def __init__(self, up=True, mode=OFF, app=None, connected=False, agent=None):
        self.up = up
        self.mode, self.app = mode, app
        self.host_up = mode == RUNNING
        self.connected, self.connecting = connected, False
        self.connect_error = None
        self.agent = agent
        self.calls = []

    def __call__(self, method, route, body=None, timeout=3.0):
        self.calls.append((method, route, body))
        if not self.up:
            return None
        if route == "/gateway":
            return {"mode": self.mode, "app": self.app}
        if route == "/gateway/start":
            self.mode, self.app = RUNNING, body["app"]
            return {"mode": self.mode}
        if route == "/gateway/microscope/connection":
            if not self.host_up:
                self.host_up = self.mode == RUNNING
                raise _refused(409, "no_app", "Start the app first.")
            if self.connecting:
                self.connecting = False
                self.connected = self.connect_error is None
                return {"connected": False, "connecting": "Connecting", "error": None}
            return {
                "connected": self.connected,
                "connecting": None,
                "error": self.connect_error,
            }
        if route == "/gateway/microscope/connect":
            self.connecting = True
            return {"connected": False}
        if route == "/gateway/agent":
            return {"running": self.agent}
        if route == "/gateway/agent/start":
            self.agent = {"level": body["level"]}
            return {"running": self.agent}
        if route == "/gateway/app/agent-page":
            return {"url": PAGE}
        raise AssertionError(route)

    def posts(self):
        return [(route, body) for method, route, body in self.calls if method == "POST"]


def _launch(gateway, configuration=None, started=None):
    opened = []
    steps = Launch(
        call=gateway,
        say=lambda text: None,
        sleep=lambda seconds: None,
        start_gateway=started or (lambda: None),
    )
    launch(
        "autolamella", configuration, "default", steps=steps, open_page=opened.append
    )
    return opened


def test_a_launch_from_off_ends_on_the_agent_page_at_watch():
    gateway = FakeGateway()
    assert _launch(gateway, configuration="arctis") == [PAGE]
    assert gateway.posts() == [
        (
            "/gateway/start",
            {"app": "autolamella", "window": True, "via": "desktop launch"},
        ),
        ("/gateway/microscope/connect", {"configuration": "arctis"}),
        (
            "/gateway/agent/start",
            {"profile": "default", "level": "watch", "via": "desktop launch"},
        ),
    ]


def test_a_second_launch_only_opens_the_page():
    gateway = FakeGateway(
        mode=RUNNING, app="autolamella", connected=True, agent={"level": "drive"}
    )
    assert _launch(gateway) == [PAGE]
    assert gateway.posts() == []


def test_the_default_configuration_connects_when_none_is_named():
    gateway = FakeGateway(mode=RUNNING, app="autolamella")
    _launch(gateway)
    assert ("/gateway/microscope/connect", {}) in gateway.posts()


def test_no_gateway_starts_one_and_carries_on():
    gateway = FakeGateway(up=False)
    started = []

    def start():
        started.append(True)
        gateway.up = True

    assert _launch(gateway, started=start) == [PAGE]
    assert started == [True]


def test_a_connect_that_fails_stops_before_the_agent():
    gateway = FakeGateway(mode=RUNNING, app="autolamella")
    gateway.connect_error = "The Thermo server didn't answer."
    with pytest.raises(LaunchFailed, match="Connect the microscope: The Thermo server"):
        _launch(gateway)
    assert all(route != "/gateway/agent/start" for route, _ in gateway.posts())


def test_another_app_running_is_left_alone():
    gateway = FakeGateway(mode=RUNNING, app="volume")
    with pytest.raises(LaunchFailed, match="volume is running"):
        _launch(gateway)
    assert gateway.posts() == []


def test_a_refused_agent_start_says_why():
    gateway = FakeGateway(mode=RUNNING, app="autolamella", connected=True)

    def call(method, route, body=None, timeout=3.0):
        if route == "/gateway/agent/start":
            raise _refused(409, "no_api_key", "Set ANTHROPIC_API_KEY first.")
        return gateway(method, route, body, timeout)

    steps = Launch(call=call, say=lambda text: None, sleep=lambda seconds: None)
    with pytest.raises(LaunchFailed, match="Start the agent: Set ANTHROPIC_API_KEY"):
        launch(steps=steps, open_page=lambda url: None)


def test_the_command_has_no_level_to_raise():
    with pytest.raises(SystemExit):
        gateway_main.main(["launch", "--level", "drive"])


def test_the_shortcut_runs_the_agent_launch(tmp_path, monkeypatch):
    from fibsem.tools import desktop_shortcut

    monkeypatch.setattr(desktop_shortcut, "get_desktop_directory", lambda: tmp_path)
    monkeypatch.setattr(
        desktop_shortcut, "find_entry_point", lambda name: tmp_path / name
    )
    written = []
    monkeypatch.setattr(
        desktop_shortcut, "_write_windows_lnk", lambda *a: written.append(a)
    )
    assert gateway_main.agent_shortcut() == 0
    shortcut = next(tmp_path.iterdir()) if not written else written[0][0]
    assert "AutoLamella with the agent" in shortcut.name
    if not written:
        assert "fibsem-autolamella-agent" in shortcut.read_text()
