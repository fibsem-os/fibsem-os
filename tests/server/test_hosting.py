"""AppServerHost over a real socket: lifecycle, discovery, events, refusal."""

import os
import socket

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fibsem import utils  # noqa: E402
from fibsem.applications.autolamella.server.context import (  # noqa: E402
    AutoLamellaAppContext,
)
from fibsem.server.hosting import AppServerHost  # noqa: E402
from fibsem.workflows.session import AppSession  # noqa: E402


def _session(microscope):
    session = AppSession()
    session.microscope = microscope
    session.start_recording()
    return session


def _host(session, **kwargs):
    return AppServerHost(session, lambda: AutoLamellaAppContext(session), **kwargs)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def microscope():
    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    return microscope


@pytest.fixture
def host(tmp_path, microscope):
    session = _session(microscope)
    server_host = _host(
        session, port=_free_port(), discovery_path=tmp_path / "agent-server.json"
    )
    yield server_host
    server_host.stop()
    session.stop_recording()


def _get(server_host, path):
    return httpx.get(
        f"{server_host.url}{path}",
        headers={"Authorization": f"Bearer {server_host.auth.token}"},
        timeout=10,
    )


def test_start_serves_read_only_with_the_app_router(host, microscope):
    assert host.start() is True
    assert host.running
    body = _get(host, "/capabilities").json()
    assert body["routers"]["app"] is True
    assert body["level"] == "watch"
    assert body["level"] == "watch"  # every launch starts watching (FIB-1035)
    # idempotent: a second start is a no-op, not a second server
    assert host.start() is True


def test_discovery_file_lives_and_dies_with_the_server(host, microscope, tmp_path):
    from fibsem.server.discovery import read_discovery_file

    path = tmp_path / "agent-server.json"
    host.start()
    found = read_discovery_file(path)
    assert found is not None and found["url"] == host.url
    host.stop()
    assert not host.running
    assert read_discovery_file(path) is None
    with pytest.raises(httpx.ConnectError):
        httpx.get(f"{host.url}/health", timeout=2)


def test_refuses_to_start_beside_a_live_server(host, microscope, tmp_path):
    from fibsem.server import AuthConfig
    from fibsem.server.discovery import write_discovery_file

    # A live "other server" (this pid, so the liveness check passes).
    write_discovery_file(
        "127.0.0.1", 9, AuthConfig(token="other"), path=tmp_path / "agent-server.json"
    )
    assert host.start() is False
    assert not host.running


def test_lifecycle_events_reach_the_agent_through_the_hook(host, microscope):
    from fibsem.hooks import HookContext, HookEvent, HookManager

    host.start()
    # The runner registers this hook per run (run_hooks); simulate one firing.
    manager = HookManager()
    manager.register(host._session.recorder.lifecycle_hook)
    manager.fire(
        HookContext(event=HookEvent.TASK_STARTED.value, task_name="Rough Milling")
    )
    events = _get(host, "/app/events?since=0").json()["events"]
    assert any(e["kind"] == "task_started" for e in events)


def test_the_server_serves_the_sessions_stream(host, microscope):
    host.start()
    microscope.get_stage_position()  # ensure cache exists
    from fibsem.structures import FibsemStagePosition

    microscope.move_stage_relative(
        FibsemStagePosition(x=1e-6, y=0, z=0, r=0, t=0, coordinate_system="RAW")
    )
    events = _get(host, "/app/events?since=0").json()["events"]
    assert any(e["kind"] == "stage_position_changed" for e in events)
    assert host.event_buffer is host._session.recorder.buffer


def test_start_failure_is_contained_not_raised(microscope, tmp_path):
    # A port another socket already holds — start must fail closed. (Not port 1:
    # root can bind it, and a server left there broke the remote FM tests.)
    with socket.socket() as taken:
        # On Windows the server's SO_REUSEADDR would otherwise share the port.
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            taken.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        server_host = _host(
            _session(microscope),
            port=taken.getsockname()[1],
            discovery_path=tmp_path / "agent-server.json",
        )
        try:
            assert server_host.start() is False
            assert not server_host.running
        finally:
            server_host.stop()


def test_env_override_starts_at_drive(microscope, tmp_path, monkeypatch):
    monkeypatch.setenv("FIBSEM_AGENT_ARM_CONTROL", "1")
    server_host = _host(
        _session(microscope),
        port=_free_port(),
        discovery_path=tmp_path / "agent-server.json",
    )
    try:
        assert server_host.start() is True
        body = _get(server_host, "/capabilities").json()
        assert body["level"] == "drive"
    finally:
        server_host.stop()


def test_a_restart_keeps_the_token_and_the_level(microscope, tmp_path):
    """A microscope toggle restarts the server; the
    agent's token, the level the operator set and the analysis services (a
    loaded model) survive it, for the app session only (FIB-1035 app test #5
    and the reconnect question)."""
    from fibsem.ai.agent.levels import AgentLevel

    session = _session(microscope)
    first = _host(
        session,
        port=_free_port(),
        discovery_path=tmp_path / "agent-server.json",
    )
    assert first.start()
    first.auth.set_level(AgentLevel.DRIVE)
    token = first.auth.token
    services = first._app.state.services
    assert services is session.analysis_services
    first.stop()
    second = _host(
        session,
        port=_free_port(),
        discovery_path=tmp_path / "agent-server.json",
    )
    try:
        assert second.start()
        body = _get(second, "/capabilities").json()
        assert second.auth.token == token
        assert body["level"] == "drive"
        assert second._app.state.services is services
    finally:
        second.stop()
    # A new app session -- a new window -- starts at Watch with a new token.
    fresh = _host(
        _session(microscope),
        port=_free_port(),
        discovery_path=tmp_path / "agent-server.json",
    )
    try:
        assert fresh.start()
        assert fresh.auth.token != token
        assert _get(fresh, "/capabilities").json()["level"] == "watch"
    finally:
        fresh.stop()


def test_the_token_is_never_logged(host, microscope, caplog):
    import logging

    with caplog.at_level(logging.DEBUG):
        assert host.start()
    assert host.auth.token not in caplog.text
    assert host.auth.dashboard_token not in caplog.text


def test_a_host_with_no_microscope_serves_and_follows_a_connect(tmp_path, monkeypatch):
    """The server starts before any microscope: /microscope answers 409
    not_connected, the operator connects through the session, and the same
    server then serves the microscope and its events."""
    monkeypatch.setenv("FIBSEM_SIM_NO_DELAY", "1")
    session = AppSession()
    host = _host(
        session, port=_free_port(), discovery_path=tmp_path / "agent-server.json"
    )
    try:
        assert host.start()
        assert _get(host, "/capabilities").json()["connected"] is False
        refused = _get(host, "/microscope")
        assert refused.status_code == 409
        assert refused.json()["detail"]["error_type"] == "not_connected"
        assert _get(host, "/app/workflow/status").status_code == 200
        before = host.event_buffer

        session.connect(manufacturer="Demo", ip_address="localhost")

        assert _get(host, "/capabilities").json()["connected"] is True
        assert _get(host, "/microscope").status_code == 200
        assert host.event_buffer is session.recorder.buffer is not before
        first = session.recorder
        assert session.disconnect() is None
        assert session.recorder is None
        assert _get(host, "/microscope").status_code == 409
        session.connect(manufacturer="Demo", ip_address="localhost")
        assert session.recorder is not None and session.recorder is not first
        assert host.running
    finally:
        host.stop()
        session.disconnect()
