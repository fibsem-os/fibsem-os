"""With the agent not released (fibsem.unreleased off), an App Server serves no
agent: the level is Off and stays there, nothing pairs or asks to connect as an
agent, and a client is the operator. The agent and gateway commands refuse."""

import socket

import httpx
import pytest

from fibsem import unreleased
from fibsem.ai.agent.levels import AgentLevel
from fibsem.applications.autolamella.server.context import AutoLamellaAppContext
from fibsem.server import AuthConfig
from fibsem.server.hosting import AppServerHost
from fibsem.workflows.session import AppSession


@pytest.fixture
def off(monkeypatch):
    monkeypatch.setenv(unreleased.ENV, "0")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def host(off, tmp_path, microscope):
    session = AppSession()
    session.microscope = microscope
    session.start_recording()
    server_host = AppServerHost(
        session,
        lambda: AutoLamellaAppContext(session),
        port=_free_port(),
        discovery_path=tmp_path / "agent-server.json",
    )
    assert server_host.start()
    yield server_host
    server_host.stop()
    session.stop_recording()


def _call(host, method, path, token, json=None):
    return httpx.request(
        method,
        f"{host.url}{path}",
        headers={"Authorization": f"Bearer {token}"},
        json=json,
        timeout=10,
    )


def test_the_switch_reads_the_environment_first(monkeypatch):
    monkeypatch.setenv(unreleased.ENV, "1")
    assert unreleased.enabled()
    monkeypatch.setenv(unreleased.ENV, "0")
    assert not unreleased.enabled()


def test_a_release_install_is_off(monkeypatch):
    monkeypatch.delenv(unreleased.ENV, raising=False)
    monkeypatch.setattr(unreleased, "_source_checkout_root", lambda: None)
    assert not unreleased.enabled()


def test_a_checkout_is_on(monkeypatch):
    monkeypatch.delenv(unreleased.ENV, raising=False)
    assert unreleased.enabled()


@pytest.mark.parametrize(
    ("main", "args"),
    [
        ("fibsem.ai.agent.harness:main", ([],)),
        ("fibsem.ai.agent.mcp.sidecar:main", ([],)),
        ("fibsem.gateway.main:main", ([],)),
        ("fibsem.gateway.launch:open_gateway_page", ()),
    ],
)
def test_the_agent_and_gateway_commands_refuse(off, main, args):
    import importlib

    module, function = main.split(":")
    with pytest.raises(SystemExit, match="not part of this release"):
        getattr(importlib.import_module(module), function)(*args)


def test_the_level_stays_off():
    auth = AuthConfig.generate(level=AgentLevel.DRIVE, agent_allowed=False)
    assert auth.level is AgentLevel.OFF
    auth.set_level(AgentLevel.DRIVE)
    auth.stop_agent()
    assert auth.level is AgentLevel.OFF and not auth.stopped


def test_the_server_serves_no_agent(host):
    operator = host.auth.operator_token
    caps = _call(host, "GET", "/capabilities", operator).json()
    assert caps["agent"] is False and caps["level"] == "off"

    refused = _call(host, "GET", "/capabilities", host.auth.token)
    assert refused.status_code == 403

    level = _call(host, "PUT", "/app/agent/level", operator, {"level": "drive"})
    assert level.status_code == 409
    assert host.auth.level is AgentLevel.OFF

    pair = _call(host, "POST", "/pairing", operator, {"role": "agent"})
    assert pair.status_code == 422
    assert _call(host, "POST", "/pairing", operator, {"role": "operator"}).is_success

    ask = _call(host, "POST", "/connect", host.auth.connect_token, {})
    assert ask.status_code == 404


def test_a_client_is_the_operator(off, monkeypatch):
    from fibsem.client import transport

    monkeypatch.setenv(transport.SERVER_URL_ENV, "http://127.0.0.1:1")
    monkeypatch.setenv(transport.OPERATOR_TOKEN_ENV, "the-operator")
    monkeypatch.delenv(transport.SERVER_TOKEN_ENV, raising=False)
    assert transport.resolve() == ("http://127.0.0.1:1", "the-operator")
