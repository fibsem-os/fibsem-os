"""The embedded App Server through the real window: preference-gated start
with or without a microscope, serving whatever the session connects."""

import os
import socket

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fibsem import config as fibsem_cfg
from fibsem.applications.autolamella.ui.window import AutoLamellaWindow
from fibsem.server import hosting
from fibsem.workflows.event_recording import recorder_for
from tests.fixtures.main_window import close_window


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def ui(qapp):
    window = AutoLamellaWindow()
    try:
        yield window
    finally:
        window.session.stop_app_server()
        window.session.stop_recording()
        close_window(window)
        window.deleteLater()
        qapp.processEvents()


@pytest.fixture
def app_server_enabled(monkeypatch, tmp_path):
    """Flip the preference on and confine the host to a test port + tmp file."""
    preferences = fibsem_cfg.UserPreferences()
    preferences.app_server.enabled = True
    monkeypatch.setattr(fibsem_cfg, "load_user_preferences", lambda: preferences)

    real_host = hosting.AppServerHost
    port = _free_port()

    def confined(session, app_context, **kwargs):
        return real_host(
            session,
            app_context,
            port=port,
            discovery_path=tmp_path / "agent-server.json",
            serve_microscope=kwargs.get("serve_microscope", True),
        )

    monkeypatch.setattr(hosting, "AppServerHost", confined)
    return preferences


def _get(host, path):
    return httpx.get(
        f"{host.url}{path}",
        headers={"Authorization": f"Bearer {host.auth.token}"},
        timeout=10,
    )


def test_the_server_starts_before_a_microscope_and_follows_the_connect(
    ui, app_server_enabled
):
    ui.sync_app_server_with_preference()
    host = ui.session.app_server
    assert host is not None and host.running
    body = _get(host, "/capabilities").json()
    assert body["routers"]["app"] is True
    assert body["level"] == "watch"
    assert body["connected"] is False
    assert _get(host, "/microscope").status_code == 409

    ui.system_widget.connect_to_microscope()

    assert ui.session.app_server is host and host.running
    assert _get(host, "/capabilities").json()["connected"] is True
    assert _get(host, "/microscope").status_code == 200
    # The feed is the recorder's, which a run finds and registers for itself
    # (FIB-1044): not registered a second time by the app's own hooks.
    assert recorder_for(ui.microscope) is ui.session.recorder
    hooks = ui.session.runner.run_hooks("autolamella")._hooks
    assert ui.session.recorder.lifecycle_hook not in hooks

    ui.system_widget.connect_to_microscope()  # connected: disconnects
    assert ui.microscope is None and host.running
    assert _get(host, "/microscope").status_code == 409


def test_a_connect_from_elsewhere_reaches_the_window(ui, qapp):
    """The window follows the session, whoever connected: here the session
    itself, as the App Server's route does."""
    ui.session.connect(manufacturer="Demo", ip_address="localhost")
    qapp.processEvents()

    assert ui.system_widget._label_status_title.text() == "Microscope Connected"
    assert ui.panel.image_widget is not None
    assert ui.settings is ui.session.settings

    ui.session.disconnect()
    qapp.processEvents()
    assert ui.system_widget._label_status_title.text() == "Not Connected"
    assert ui.panel.image_widget is None


def _confine_and_control_preferences(monkeypatch, tmp_path):
    """A mutable preferences object the tests flip, host confined to a test port."""
    preferences = fibsem_cfg.UserPreferences()
    monkeypatch.setattr(fibsem_cfg, "load_user_preferences", lambda: preferences)
    real_host = hosting.AppServerHost
    port = _free_port()
    monkeypatch.setattr(
        hosting,
        "AppServerHost",
        lambda session, app_context, **kwargs: real_host(
            session,
            app_context,
            port=port,
            discovery_path=tmp_path / "agent-server.json",
            serve_microscope=kwargs.get("serve_microscope", True),
        ),
    )
    return preferences


def test_saving_the_preference_mid_session_starts_and_stops_the_server(
    ui, monkeypatch, tmp_path
):
    preferences = _confine_and_control_preferences(monkeypatch, tmp_path)
    ui.sync_app_server_with_preference()
    assert ui.session.app_server is None  # flag off

    # The user ticks the box and saves: the server starts now, not next connect.
    preferences.app_server.enabled = True
    ui.sync_app_server_with_preference()
    host = ui.session.app_server
    assert host is not None and host.running

    # Already matching: syncing again must not restart or double-bind.
    ui.sync_app_server_with_preference()
    assert ui.session.app_server is host and host.running

    # Unticking stops it just as immediately.
    preferences.app_server.enabled = False
    ui.sync_app_server_with_preference()
    assert not host.running


def _routers(host) -> dict:
    return httpx.get(
        f"{host.url}/capabilities",
        headers={"Authorization": f"Bearer {host.auth.token}"},
        timeout=10,
    ).json()["routers"]


def test_the_microscope_is_served_by_default_and_restarts_when_changed(
    ui, monkeypatch, tmp_path
):
    preferences = _confine_and_control_preferences(monkeypatch, tmp_path)
    preferences.app_server.enabled = True
    ui.sync_app_server_with_preference()
    first = ui.session.app_server
    assert first is not None and first.running
    assert _routers(first) == {"microscope": True, "services": True, "app": True}

    # Turning the microscope off in Tools → App Server rebuilds the server
    # without the instrument routes, keeping the agent's token (FIB-1359).
    ui.set_app_server_microscope(False)
    second = ui.session.app_server
    assert second is not first and not first.running and second.running
    assert second.auth is first.auth
    assert _routers(second) == {"microscope": False, "services": True, "app": True}

    # Unchanged: no restart.
    ui.set_app_server_microscope(False)
    assert ui.session.app_server is second and second.running

    # Syncing only starts or stops the server; the microscope is applied by
    # set_app_server_microscope, which Tools → App Server calls after saving it.
    preferences.app_server.microscope = True
    ui.sync_app_server_with_preference()
    assert ui.session.app_server is second and not second.serve_microscope


def test_default_preference_hosts_nothing(ui, monkeypatch):
    # Pinned to actual defaults: this machine's preference file may have the
    # flag on (it survives in the worktree), and that must not fake a failure.
    monkeypatch.setattr(
        fibsem_cfg, "load_user_preferences", lambda: fibsem_cfg.UserPreferences()
    )
    ui.system_widget.connect_to_microscope()
    assert ui.session.app_server is None
    manager = ui.session.runner.run_hooks("autolamella")
    assert manager is not None  # and built without any agent involvement


def test_the_event_stream_runs_without_the_app_server(ui, monkeypatch):
    """The stream is the app's (FIB-1031): connect starts it, server or not."""
    monkeypatch.setattr(
        fibsem_cfg, "load_user_preferences", lambda: fibsem_cfg.UserPreferences()
    )
    ui.system_widget.connect_to_microscope()
    recorder = ui.session.recorder
    assert recorder is not None and ui.session.app_server is None
    # A run finds this recorder and registers its lifecycle hook (FIB-1044).
    assert recorder_for(ui.microscope) is recorder
    # In the app, a call no task or agent marked is the operator's (FIB-1062).
    assert recorder.default_actor == "operator"

    ui.system_widget.connect_to_microscope()  # connected: disconnects
    assert ui.session.recorder is None
    assert recorder_for(recorder.microscope) is None
    assert not recorder.writer.alive


def test_the_app_server_reads_the_session_stream(ui, app_server_enabled):
    ui.sync_app_server_with_preference()
    ui.system_widget.connect_to_microscope()
    host, recorder = ui.session.app_server, ui.session.recorder
    assert host.running
    assert host.event_buffer is recorder.buffer
    # Shared, so not registered here: the run registers it, once (FIB-1044).
    hooks = ui.session.runner.run_hooks("autolamella")._hooks
    assert sum(hook is recorder.lifecycle_hook for hook in hooks) == 0


def test_tools_app_server_saves_both_switches(monkeypatch):
    """The switches in Tools → App Server are saved, so the next launch starts the
    same way, and act on the session now."""
    from types import SimpleNamespace

    from fibsem.applications.autolamella.ui.window import AutoLamellaWindow

    saved = fibsem_cfg.UserPreferences()
    monkeypatch.setattr(fibsem_cfg, "load_user_preferences", lambda: saved)
    monkeypatch.setattr(fibsem_cfg, "save_user_preferences", lambda p: None)
    calls = []
    window = SimpleNamespace(
        sync_app_server_with_preference=lambda: calls.append("sync"),
        set_app_server_microscope=lambda on: calls.append(("microscope", on)),
    )

    AutoLamellaWindow._set_app_server_microscope(window, True)
    AutoLamellaWindow._set_app_server_enabled(window, True)

    assert saved.app_server.microscope and saved.app_server.enabled
    assert window._preferences is saved
    assert calls == [("microscope", True), "sync"]
