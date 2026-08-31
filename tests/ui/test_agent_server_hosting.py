"""The embedded agent server through the real window: preference-gated start on
microscope connect, per-run hook registration, stop on disconnect."""

import os
import socket

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fibsem import config as fibsem_cfg
from fibsem.applications.autolamella.server import hosting
from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def ui(qapp):
    window = AutoLamellaUI(parent_ui=None)
    try:
        yield window
    finally:
        if window._agent_server_host is not None:
            window._agent_server_host.stop()
        window.close()
        window.deleteLater()
        qapp.processEvents()


@pytest.fixture
def agent_server_enabled(monkeypatch, tmp_path):
    """Flip the preference on and confine the host to a test port + tmp file."""
    preferences = fibsem_cfg.UserPreferences()
    preferences.features.agent_server_enabled = True
    monkeypatch.setattr(fibsem_cfg, "load_user_preferences", lambda: preferences)

    real_host = hosting.AgentServerHost
    port = _free_port()

    def confined(ui_object, **kwargs):
        return real_host(
            ui_object, port=port, discovery_path=tmp_path / "agent-server.json"
        )

    monkeypatch.setattr(hosting, "AgentServerHost", confined)
    return preferences


def test_connect_starts_a_live_read_only_server(ui, agent_server_enabled):
    ui.system_widget.connect_to_microscope()
    ui.connect_to_microscope()

    host = ui._agent_server_host
    assert host is not None and host.running
    body = httpx.get(
        f"{host.url}/capabilities",
        headers={"Authorization": f"Bearer {host.auth.token}"},
        timeout=10,
    ).json()
    assert body["routers"]["app"] is True
    assert body["scopes"]["hardware"] is False

    # Per-run hook registration: the manager built for a run carries the feed.
    manager = ui.setup_hooks()
    assert host.lifecycle_hook in manager._hooks

    ui.disconnect_from_microscope()
    assert not host.running


def _confine_and_control_preferences(monkeypatch, tmp_path):
    """A mutable preferences object the tests flip, host confined to a test port."""
    preferences = fibsem_cfg.UserPreferences()
    monkeypatch.setattr(fibsem_cfg, "load_user_preferences", lambda: preferences)
    real_host = hosting.AgentServerHost
    port = _free_port()
    monkeypatch.setattr(
        hosting,
        "AgentServerHost",
        lambda ui_object, **kwargs: real_host(
            ui_object, port=port, discovery_path=tmp_path / "agent-server.json"
        ),
    )
    return preferences


def test_saving_the_preference_mid_session_starts_and_stops_the_server(
    ui, monkeypatch, tmp_path
):
    preferences = _confine_and_control_preferences(monkeypatch, tmp_path)
    ui.system_widget.connect_to_microscope()
    ui.connect_to_microscope()
    assert ui._agent_server_host is None  # flag off at connect, as before

    # The user ticks the box and saves: the server starts now, not next connect.
    preferences.features.agent_server_enabled = True
    ui.sync_agent_server_with_preference()
    host = ui._agent_server_host
    assert host is not None and host.running

    # Already matching: syncing again must not restart or double-bind.
    ui.sync_agent_server_with_preference()
    assert ui._agent_server_host is host and host.running

    # Unticking stops it just as immediately.
    preferences.features.agent_server_enabled = False
    ui.sync_agent_server_with_preference()
    assert not host.running


def test_syncing_without_a_microscope_waits_for_connect(ui, monkeypatch, tmp_path):
    preferences = _confine_and_control_preferences(monkeypatch, tmp_path)
    preferences.features.agent_server_enabled = True
    ui.sync_agent_server_with_preference()
    # Nothing to serve yet: the connect path picks the preference up, as before.
    assert ui._agent_server_host is None


def test_default_preference_hosts_nothing(ui, monkeypatch):
    # Pinned to actual defaults: this machine's preference file may have the
    # flag on (it survives in the worktree), and that must not fake a failure.
    monkeypatch.setattr(
        fibsem_cfg, "load_user_preferences", lambda: fibsem_cfg.UserPreferences()
    )
    ui.system_widget.connect_to_microscope()
    ui.connect_to_microscope()
    assert ui._agent_server_host is None
    manager = ui.setup_hooks()
    assert manager is not None  # and built without any agent involvement
