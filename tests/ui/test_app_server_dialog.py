"""Tools → App Server and Connect an agent (FIB-1035): the server's page and the
agent's page."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("fastapi")

from fibsem.applications.autolamella.ui.app_server_dialog import (  # noqa: E402
    AppServerDialog,
)
from fibsem.applications.autolamella.ui.connect_agent_dialog import (  # noqa: E402
    ConnectAgentDialog,
    agent_setup_text,
    sidecar_path,
)
from fibsem.server.auth import AgentLevel, AuthConfig  # noqa: E402


class _Host:
    running = True
    url = "http://127.0.0.1:8001"

    def __init__(self, serve_microscope=False, age=None):
        self.auth = AuthConfig.generate(token="tok")
        self.auth.set_level(AgentLevel.WATCH)
        self.serve_microscope = serve_microscope
        self._age = age

    def agent_seconds_since_seen(self):
        return self._age


@pytest.fixture
def host():
    return _Host()


# --- Tools → App Server ------------------------------------------------------


def test_the_dialog_shows_status_services_and_the_agent(qapp):
    dialog = AppServerDialog(lambda: _Host(serve_microscope=True, age=5.4))
    assert dialog._status_label.text() == "Running on http://127.0.0.1:8001"
    assert dialog._services_label.text() == "Services: monitoring · microscope on"
    assert dialog._agent_label.text() == "Agent · last heard 5 s ago"
    assert dialog._btn_dashboard.isEnabled()
    dialog.deleteLater()


def test_the_dialog_serves_the_microscope_this_session(qapp):
    """FIB-1359: the microscope toggle is here, per session, not in Preferences."""
    hosts = [_Host(serve_microscope=False)]
    chosen = []

    def set_microscope(on):
        chosen.append(on)
        hosts.append(_Host(serve_microscope=on))  # the restarted server

    dialog = AppServerDialog(lambda: hosts[-1], set_microscope=set_microscope)
    assert dialog._chk_microscope.isEnabled() and not dialog._chk_microscope.isChecked()
    dialog._chk_microscope.setChecked(True)
    assert chosen == [True]
    assert dialog._services_label.text() == "Services: monitoring · microscope on"
    # A refresh that reads the box back sets nothing.
    dialog.refresh()
    assert chosen == [True]
    dialog.deleteLater()


def test_the_microscope_toggle_waits_for_a_running_server(qapp):
    dialog = AppServerDialog(lambda: None, set_microscope=lambda on: None)
    assert not dialog._chk_microscope.isEnabled()
    dialog.deleteLater()


def test_the_dialog_has_no_token_and_no_arming(qapp, host):
    # The agent's level is the chip's; its token is Connect an agent's.
    dialog = AppServerDialog(lambda: host)
    for name in ("_token_edit", "_chk_control", "_chk_configure", "_chk_hardware"):
        assert not hasattr(dialog, name)
    dialog.deleteLater()


def test_no_server_disables_the_dashboard(qapp):
    dialog = AppServerDialog(lambda: None)
    assert not dialog._btn_dashboard.isEnabled()
    assert not dialog._btn_copy_link.isEnabled()
    assert "Not running" in dialog._status_label.text()
    dialog.deleteLater()


def test_the_dashboard_link_carries_the_dashboard_token(qapp, host):
    # The fragment never leaves the browser — no server log, no referrer.
    url = AppServerDialog.dashboard_url(host)
    assert url == f"http://127.0.0.1:8001/dashboard#token={host.auth.dashboard_token}"
    assert host.auth.dashboard_token not in (None, "tok")


def test_agent_text_reports_when_the_agent_was_last_heard_from():
    assert AppServerDialog.agent_text(_Host(age=None)) == "Agent · not heard from yet"
    assert AppServerDialog.agent_text(_Host(age=200)) == "Agent · last heard 3 min ago"


# --- Connect an agent --------------------------------------------------------


def test_connect_shows_the_level_read_only_and_the_token_hidden(qapp, host, tmp_path):
    from PyQt5.QtWidgets import QLineEdit

    dialog = ConnectAgentDialog(lambda: host, sidecar=tmp_path / "fibsem-mcp")
    assert dialog._level_label.text() == "Agent: Watching"
    host.auth.set_level(AgentLevel.DRIVE)
    dialog.refresh()
    assert dialog._level_label.text() == "Agent: Driving"
    assert dialog._token_edit.text() == "tok"
    assert dialog._token_edit.echoMode() == QLineEdit.Password
    dialog._btn_reveal.setChecked(True)
    assert dialog._token_edit.echoMode() == QLineEdit.Normal
    dialog.deleteLater()


def test_connect_without_a_server_has_no_token(qapp, tmp_path):
    dialog = ConnectAgentDialog(lambda: None, sidecar=tmp_path / "fibsem-mcp")
    assert dialog._token_edit.text() == ""
    assert not dialog._btn_copy_token.isEnabled()
    dialog.deleteLater()


def test_the_agent_text_registers_for_every_project_by_absolute_path(tmp_path):
    path = tmp_path / "env" / "bin" / "fibsem-mcp"
    text = agent_setup_text(path, "http://127.0.0.1:8002", platform="darwin")
    assert f'claude mcp add -s user fibsem -- "{path}"' in text
    assert f'"{path}" --check' in text
    assert '"connected to http://127.0.0.1:8002"' in text
    assert "get_capabilities" in text
    # Desktop users merge one entry into mcpServers; they never replace the file.
    entry = text.split('"fibsem": ', 1)[1].split(" under", 1)[0]
    assert json.loads(entry) == {"command": str(path)}
    assert "Library/Application Support/Claude/claude_desktop_config.json" in text
    assert "don't replace the file" in text


def test_the_agent_text_names_the_windows_desktop_config():
    text = agent_setup_text(r"C:\env\Scripts\fibsem-mcp.exe", platform="win32")
    assert r"%APPDATA%\Claude\claude_desktop_config.json" in text
    assert '"connected to http://127.0.0.1:8001"' in text
    entry = text.split('"fibsem": ', 1)[1].split(" under", 1)[0]
    assert json.loads(entry) == {"command": r"C:\env\Scripts\fibsem-mcp.exe"}


def test_copy_for_your_agent_fills_in_the_running_server(qapp, host, tmp_path):
    from PyQt5.QtWidgets import QApplication

    path = tmp_path / "fibsem-mcp"
    dialog = ConnectAgentDialog(lambda: host, sidecar=path)
    assert not hasattr(dialog, "_btn_copy_code")
    dialog._btn_copy_setup.click()
    copied = QApplication.clipboard().text()
    assert copied == agent_setup_text(path, host.url)
    assert host.auth.token not in copied
    dialog.deleteLater()


@pytest.mark.skipif(os.name == "nt", reason="the POSIX layout")
def test_the_sidecar_is_found_beside_the_interpreter(tmp_path):
    bin_dir = tmp_path / "env" / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python"
    assert sidecar_path(str(python)) is None
    (bin_dir / "fibsem-mcp").write_text("#!/bin/sh\n")
    assert sidecar_path(str(python)) == bin_dir / "fibsem-mcp"


def test_a_missing_sidecar_says_how_to_install_it(qapp, host, monkeypatch):
    from fibsem.applications.autolamella.ui import connect_agent_dialog

    monkeypatch.setattr(connect_agent_dialog, "sidecar_path", lambda: None)
    dialog = ConnectAgentDialog(lambda: host)
    assert not dialog._btn_copy_setup.isEnabled()
    dialog.deleteLater()
