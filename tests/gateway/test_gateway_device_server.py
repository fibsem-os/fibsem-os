"""The Device Server the gateway runs on a device PC: started once when
device-server.yaml exists (with the Python the file names), never started
again by itself after it ends, shown stopped with its last log lines, and
started and stopped on the gateway's page by the operator only."""

import json
import os
import subprocess
import sys

import httpx
import pytest
from fastapi.testclient import TestClient

from fibsem.gateway.device_server import (
    NOT_SET_UP,
    RUNNING,
    STARTING,
    STOPPED,
    TARGET,
    DeviceServerProcess,
)
from fibsem.gateway.server import GatewayAuth, build_gateway_app
from fibsem.server.pairing import SWITCH_MODES
from tests.gateway.test_gateway_routes import (
    OPERATOR_TOKEN,
    FakeGateway,
    _as,
    _pair_hub,
)


class Proc:
    def __init__(self, command, **kwargs):
        self.command = command
        self.pid = 5150
        self.returncode = None
        self.killed = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("device server", timeout)
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


HEALTH = {
    "ok": False,
    "devices": {"stage": {"ok": True}, "laser": {"ok": False, "detail": "off"}},
    "program": "fibsem-device-server",
    "version": "0.6.0",
    "api_version": "0.1.0",
}


def _made(tmp_path, text="server:\n  port: 8799\n", answer=HEALTH):
    config = tmp_path / "device-server.yaml"
    if text is not None:
        config.write_text(text)
    started, asked = [], []

    def popen(command, **kwargs):
        started.append(Proc(command, **kwargs))
        return started[-1]

    def get(url, route, token):
        asked.append((url, route, token))
        if isinstance(answer, Exception):
            raise answer
        return answer

    server = DeviceServerProcess(
        sys.executable,
        config=config,
        home=tmp_path,
        popen=popen,
        get=get,
        token=lambda: "device-token",
        clock=lambda: 1000.0,
    )
    return server, started, asked


def test_no_file_means_no_device_server(tmp_path):
    server, started, _ = _made(tmp_path, text=None)
    server.ensure()
    assert started == []
    assert server.state().status == NOT_SET_UP
    assert server.start().status == NOT_SET_UP and started == []


def test_the_gateway_starts_it_once_with_the_files_python(tmp_path):
    server, started, asked = _made(
        tmp_path, text="server:\n  port: 8799\n  python: /opt/vendor/python\n"
    )
    server.ensure()
    server.ensure()
    assert len(started) == 1
    command = started[0].command
    assert command[0] == "/opt/vendor/python"
    assert command[1:3] == ["-m", "fibsem.gateway.child"] and command[4] == TARGET
    assert command[5:] == ["--config", str(tmp_path / "device-server.yaml")]
    state = server.state()
    assert state.status == RUNNING and state.url == "http://127.0.0.1:8799"
    assert state.health["api_version"] == "0.1.0"
    assert state.devices == [
        {"name": "stage", "ok": True, "detail": None},
        {"name": "laser", "ok": False, "detail": "off"},
    ]
    assert asked[-1] == ("http://127.0.0.1:8799", "/health", "device-token")


def test_without_python_in_the_file_it_runs_the_gateways_own(tmp_path):
    server, started, _ = _made(tmp_path)
    server.ensure()
    assert started[0].command[0] == sys.executable


def test_not_answering_yet_is_starting(tmp_path):
    server, _, _ = _made(tmp_path, answer=httpx.ConnectError("refused"))
    server.ensure()
    state = server.state()
    assert state.status == STARTING and "ConnectError" in state.detail


def test_one_that_ends_stays_stopped_with_its_last_log_lines(tmp_path):
    server, started, _ = _made(tmp_path)
    server.ensure()
    log = server.state().log
    with open(log, "a") as f:
        f.write("starting\nNo module named 'vendor_sdk'\n")
    started[0].returncode = 1
    server.ensure()
    assert len(started) == 1  # never started again by itself
    state = server.state()
    assert state.status == STOPPED and state.exit_code == 1
    assert state.log_tail[-1] == "No module named 'vendor_sdk'"
    assert state.detail.startswith("It ended")
    server.start()  # the page's Start
    assert len(started) == 2


def test_stop_uses_its_stop_file_then_kills(tmp_path):
    server, started, _ = _made(tmp_path)
    server.ensure()
    proc = started[0]
    stop_file = proc.command[3]
    seen = []

    def wait(timeout=None):
        seen.append(os.path.exists(stop_file))
        proc.returncode = 0
        return 0

    proc.wait = wait
    state = server.stop()
    assert seen == [True] and not proc.killed
    assert state.status == STOPPED and state.detail == "Stopped on this page."
    assert not os.path.exists(stop_file)


def test_one_that_ignores_its_stop_file_is_killed(tmp_path):
    server, started, _ = _made(tmp_path)
    server.ensure()
    server.stop(wait_s=0.01)
    assert started[0].killed


def test_a_python_that_wont_start_is_said_in_its_log(tmp_path):
    server, _, _ = _made(tmp_path, text="server:\n  python: /no/such/python\n")

    def refuse(command, **kwargs):
        raise FileNotFoundError(2, "No such file", command[0])

    server._popen = refuse
    state = server.start()
    assert state.status == STOPPED
    assert "/no/such/python" in "\n".join(state.log_tail)


@pytest.fixture
def routes(tmp_path):
    token_file = tmp_path / "operator-token.json"
    token_file.write_text(json.dumps({"token": OPERATOR_TOKEN}))
    auth = GatewayAuth.from_disk(token_file, tmp_path / "gateway-paired.json")
    server, started, _ = _made(tmp_path)
    gateway = FakeGateway()
    client = TestClient(build_gateway_app(gateway, auth, device_server=server))
    return client, started, gateway


def test_the_page_starts_and_stops_it(routes):
    client, started, gateway = routes
    r = client.get("/gateway/device-server", headers=_as(OPERATOR_TOKEN))
    assert r.status_code == 200 and r.json()["status"] == STOPPED
    r = client.post("/gateway/device-server/start", headers=_as(OPERATOR_TOKEN))
    assert r.json()["status"] == RUNNING and len(started) == 1
    started[0].returncode = 0
    started[0].wait = lambda timeout=None: 0
    r = client.post("/gateway/device-server/stop", headers=_as(OPERATOR_TOKEN))
    assert r.json()["status"] == STOPPED
    assert [a[2] for a in gateway.access] == [
        "start the Device Server",
        "stop the Device Server",
    ]


def test_a_hub_reads_it_but_does_not_start_it(routes):
    client, started, _ = routes
    hub = _pair_hub(client, [SWITCH_MODES])  # may start apps, not this
    assert (
        client.get("/gateway/device-server", headers=_as(hub["token"])).status_code
        == 200
    )
    r = client.post("/gateway/device-server/start", headers=_as(hub["token"]))
    assert r.status_code == 403 and started == []
    assert client.get("/gateway/device-server").status_code == 401
