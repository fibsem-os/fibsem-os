"""The App Server over a session with no microscope: /microscope answers 409
not_connected, and the operator connects and disconnects through the session."""

import time

import pytest
from fastapi.testclient import TestClient

from fibsem.ai.agent.levels import AgentLevel
from fibsem.applications.autolamella.server.context import (
    AutoLamellaAppContext,
)
from fibsem.server import AuthConfig, build_server
from fibsem.workflows.session import AppSession

AGENT = "agent-token"
OPERATOR = "operator-token"


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setenv("FIBSEM_SIM_NO_DELAY", "1")
    session = AppSession()
    auth = AuthConfig.generate(level=AgentLevel.DRIVE, token=AGENT)
    auth.operator_token = OPERATOR
    app = build_server(
        session=session,
        app_context=AutoLamellaAppContext(session),
        auth=auth,
        operator=session.agent,
    )
    with TestClient(app) as client:
        client.session = session
        yield client
    session.disconnect()


def _as(token):
    return {"Authorization": f"Bearer {token}"}


def _connected(client, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = client.get("/microscope/connection", headers=_as(OPERATOR)).json()
        if state["run"] and state["run"]["state"] != "running":
            return state
        time.sleep(0.05)
    raise AssertionError("the connect did not end")


def test_before_a_connect_the_microscope_routes_say_not_connected(server):
    assert server.get("/capabilities", headers=_as(AGENT)).json()["connected"] is False
    for path in ("/microscope", "/microscope/devices", "/microscope/control"):
        answer = server.get(path, headers=_as(AGENT))
        assert answer.status_code == 409, path
        assert answer.json()["detail"]["error_type"] == "not_connected"
    # The rest of the server answers as ever.
    assert server.get("/app/workflow/status", headers=_as(AGENT)).status_code == 200


def test_the_operator_connects_by_configuration_and_disconnects(server):
    state = server.get("/microscope/connection", headers=_as(OPERATOR)).json()
    assert state["connected"] is False
    names = [c["name"] for c in state["configurations"]]
    default = next(c["name"] for c in state["configurations"] if c["default"])
    assert default in names

    answer = server.post(
        "/microscope/connect", headers=_as(OPERATOR), json={"configuration": default}
    )
    assert answer.status_code == 202, answer.text

    state = _connected(server)
    assert state["run"]["state"] == "done", state
    assert state["connected"] is True
    assert state["configuration"]["name"] == default
    assert server.get("/microscope", headers=_as(AGENT)).status_code == 200
    again = server.post("/microscope/connect", headers=_as(OPERATOR), json={})
    assert again.status_code == 409

    answer = server.post("/microscope/disconnect", headers=_as(OPERATOR))
    assert answer.status_code == 200 and answer.json()["connected"] is False
    assert server.session.microscope is None
    assert server.get("/microscope", headers=_as(AGENT)).status_code == 409


def test_the_agent_can_never_connect_or_disconnect(server):
    for method, path in (
        ("GET", "/microscope/connection"),
        ("POST", "/microscope/connect"),
        ("POST", "/microscope/disconnect"),
    ):
        answer = server.request(method, path, headers=_as(AGENT))
        assert answer.status_code == 403, path
    assert server.session.microscope is None


def test_an_unknown_configuration_is_refused(server):
    answer = server.post(
        "/microscope/connect", headers=_as(OPERATOR), json={"configuration": "nope"}
    )
    assert answer.status_code == 404


def test_disconnect_is_refused_while_a_run_is_on(server, monkeypatch):
    session = server.session
    session.connect(manufacturer="Demo", ip_address="localhost")
    monkeypatch.setattr(type(session.runner), "is_running", property(lambda _: True))

    answer = server.post("/microscope/disconnect", headers=_as(OPERATOR))

    assert answer.status_code == 409
    assert answer.json()["detail"]["error_type"] == "run_on"
    assert session.microscope is not None
    monkeypatch.undo()  # before the fixture disconnects


def test_health_says_the_connection_and_the_halt_without_the_hardware(server):
    before = server.get("/microscope/health", headers=_as(OPERATOR))
    assert before.status_code == 200, before.text
    assert before.json() == {
        "connected": False,
        "connecting": None,
        "error": None,
        "halted": None,
        "devices": {},
        "services": {},
    }
    server.post("/microscope/connect", headers=_as(OPERATOR), json={})
    _connected(server)

    # Asking never reaches a device: no read, no claim of the imaging channel.
    devices = server.session.microscope.devices.values()
    for device in devices:
        device.check_health = lambda: pytest.fail("health read the hardware")
    with server.session.microscope.resources.claim("imaging_channel"):
        health = server.get("/microscope/health", headers=_as(OPERATOR)).json()
    assert health["connected"] is True and health["halted"] is None
    # When each device was last heard from, from the calls made: none yet here.
    assert "stage" in health["devices"]
    server.session.microscope.get_stage_position()
    health = server.get("/microscope/health", headers=_as(OPERATOR)).json()
    stage = health["devices"]["stage"]
    assert stage["last_ok"] is not None and stage["running"] == []

    halt = next(iter(server.session.microscope.devices.values())).safety_halt
    halt.set("door open")
    try:
        health = server.get("/microscope/health", headers=_as(OPERATOR)).json()
        assert health["halted"] == "door open"
    finally:
        halt.clear()


def test_the_agent_cannot_read_health(server):
    assert server.get("/microscope/health", headers=_as(AGENT)).status_code in (
        401,
        403,
    )
