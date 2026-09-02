"""AgentServerHost over a real socket: lifecycle, discovery, events, refusal."""

import os
import socket

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fibsem import utils  # noqa: E402
from fibsem.applications.autolamella.server.hosting import AgentServerHost  # noqa: E402


class Host:
    experiment = None
    microscope = None
    _task_manager = None
    is_workflow_running = False


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
    ui = Host()
    ui.microscope = microscope
    server_host = AgentServerHost(
        ui, port=_free_port(), discovery_path=tmp_path / "agent-server.json"
    )
    yield server_host
    server_host.stop()


def _get(server_host, path):
    return httpx.get(
        f"{server_host.url}{path}",
        headers={"Authorization": f"Bearer {server_host.auth.token}"},
        timeout=10,
    )


def test_start_serves_read_only_with_the_app_router(host, microscope):
    assert host.start(microscope) is True
    assert host.running
    body = _get(host, "/capabilities").json()
    assert body["routers"]["app"] is True
    assert body["scopes"] == {"read": True, "control": False, "configure": False, "hardware": False}
    # idempotent: a second start is a no-op, not a second server
    assert host.start(microscope) is True


def test_discovery_file_lives_and_dies_with_the_server(host, microscope, tmp_path):
    from fibsem.server.discovery import read_discovery_file

    path = tmp_path / "agent-server.json"
    host.start(microscope)
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
    assert host.start(microscope) is False
    assert not host.running


def test_lifecycle_events_reach_the_agent_through_the_hook(host, microscope):
    from fibsem.hooks import HookContext, HookEvent, HookManager

    host.start(microscope)
    # The app registers this hook per run (setup_hooks); simulate one firing.
    manager = HookManager()
    manager.register(host.lifecycle_hook)
    manager.fire(
        HookContext(event=HookEvent.TASK_STARTED.value, task_name="Rough Milling")
    )
    events = _get(host, "/app/events?since=0").json()["events"]
    assert any(e["kind"] == "task_started" for e in events)


def test_microscope_taps_are_live_and_detach_on_stop(host, microscope):
    host.start(microscope)
    microscope.get_stage_position()  # ensure cache exists
    from fibsem.structures import FibsemStagePosition

    microscope.move_stage_relative(
        FibsemStagePosition(x=1e-6, y=0, z=0, r=0, t=0, coordinate_system="RAW")
    )
    events = _get(host, "/app/events?since=0").json()["events"]
    assert any(e["kind"] == "stage_position_changed" for e in events)
    buffer = host.event_buffer
    host.stop()
    before = buffer.events_since(0)["latest_seq"]
    microscope.move_stage_relative(
        FibsemStagePosition(x=1e-6, y=0, z=0, r=0, t=0, coordinate_system="RAW")
    )
    assert buffer.events_since(0)["latest_seq"] == before  # taps detached


def test_start_failure_is_contained_not_raised(microscope, tmp_path):
    ui = Host()
    # Port 1 is unbindable without privileges — start must fail closed.
    server_host = AgentServerHost(
        ui, port=1, discovery_path=tmp_path / "agent-server.json"
    )
    assert server_host.start(microscope) is False
    assert not server_host.running


def test_env_override_arms_control_scope(microscope, tmp_path, monkeypatch):
    monkeypatch.setenv("FIBSEM_AGENT_ARM_CONTROL", "1")
    ui = Host()
    server_host = AgentServerHost(
        ui, port=_free_port(), discovery_path=tmp_path / "agent-server.json"
    )
    try:
        assert server_host.start(microscope) is True
        scopes = _get(server_host, "/capabilities").json()["scopes"]
        assert scopes["control"] is True
        assert scopes["hardware"] is False  # the override never arms hardware
    finally:
        server_host.stop()
