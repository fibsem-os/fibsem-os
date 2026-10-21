"""Tools → App Server and Connect an agent (FIB-1035): the server's page and the
agent's page."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("fastapi")

from fibsem.ai.agent.levels import AgentLevel  # noqa: E402
from fibsem.server.agent_setup import agent_setup_text, sidecar_path  # noqa: E402
from fibsem.server.auth import AuthConfig  # noqa: E402
from fibsem.server.discovery import dev_checkout  # noqa: E402
from fibsem.server.hosting import event_log_url  # noqa: E402
from fibsem.ui.agent.app_server_dialog import (  # noqa: E402
    AppServerDialog,
)
from fibsem.ui.agent.connect_agent_dialog import ConnectAgentDialog  # noqa: E402


class _Host:
    running = True
    url = "http://127.0.0.1:8001"

    def __init__(self, serve_microscope=False, age=None):
        self.auth = AuthConfig.generate(token="tok")
        self.auth.set_level(AgentLevel.WATCH)
        self.auth.operator_token = "operator"
        self.serve_microscope = serve_microscope
        self._age = age

    def new_operator_token(self):
        self.auth.operator_token = "renewed"
        return self.auth.operator_token

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
    assert dialog._btn_event_log.isEnabled()
    dialog.deleteLater()


def test_the_operator_token_is_copied_and_renewed(qapp, host):
    from PyQt5.QtWidgets import QApplication

    dialog = AppServerDialog(lambda: host)
    dialog._btn_copy_operator.click()
    assert QApplication.clipboard().text() == "operator"
    dialog._btn_new_operator.click()
    assert host.auth.operator_token == "renewed"
    assert QApplication.clipboard().text() == "renewed"
    assert dialog._btn_copy_operator.text() == "New token copied"
    dialog.deleteLater()


class _Session:
    """The window's side of the two switches: what they set, as the fake host."""

    def __init__(self, enabled=False, serve_microscope=False):
        self.calls = []
        self.enabled = enabled
        self.serve_microscope = serve_microscope

    def host(self):
        return _Host(self.serve_microscope) if self.enabled else None

    def set_enabled(self, on):
        self.calls.append(("enabled", on))
        self.enabled = on

    def set_microscope(self, on):
        self.calls.append(("microscope", on))
        self.serve_microscope = on

    def dialog(self):
        return AppServerDialog(
            self.host,
            set_enabled=self.set_enabled,
            set_microscope=self.set_microscope,
            enabled=self.enabled,
            serve_microscope=self.serve_microscope,
        )


def _settle(dialog):
    dialog._settle_timer.timeout.emit()


def test_the_server_is_turned_on_here(qapp):
    """One place for both switches: the server is turned on here, not in Preferences."""
    session = _Session()
    dialog = session.dialog()

    assert not dialog._chk_enabled.isChecked()
    assert not dialog._chk_microscope.isEnabled()  # nothing to serve while off
    assert dialog._status_label.text() == "Not running."

    dialog._chk_enabled.setChecked(True)
    assert dialog._chk_microscope.isEnabled()
    assert session.calls == []  # waits for the switches to settle
    _settle(dialog)
    assert session.calls == [("enabled", True)]
    assert dialog._status_label.text() == "Running on http://127.0.0.1:8001"

    dialog._chk_microscope.setChecked(True)
    _settle(dialog)
    assert session.calls[-1] == ("microscope", True)
    assert dialog._services_label.text() == "Services: monitoring · microscope on"
    dialog.refresh()  # a refresh changes nothing
    assert len(session.calls) == 2
    dialog.deleteLater()


def test_the_switches_open_as_saved(qapp):
    dialog = _Session(enabled=True, serve_microscope=True).dialog()
    assert dialog._chk_enabled.isChecked() and dialog._chk_microscope.isChecked()
    dialog.deleteLater()


def test_turning_both_on_serves_the_microscope_from_the_start(qapp):
    """The microscope is applied first, so the server starts once, serving it."""
    session = _Session()
    dialog = session.dialog()
    dialog._chk_enabled.setChecked(True)
    dialog._chk_microscope.setChecked(True)
    _settle(dialog)
    assert session.calls == [("microscope", True), ("enabled", True)]
    dialog.deleteLater()


def test_a_double_click_restarts_once_or_not_at_all(qapp):
    """FIB-1409: each toggle restarted the server; only the settled state applies."""
    session = _Session(enabled=True)
    dialog = session.dialog()
    box = dialog._chk_microscope
    box.setChecked(True)
    box.setChecked(False)
    _settle(dialog)
    assert session.calls == []  # back where it started: no restart

    box.setChecked(True)
    box.setChecked(False)
    box.setChecked(True)
    _settle(dialog)
    assert session.calls == [("microscope", True)]
    dialog.deleteLater()


def test_closing_the_dialog_applies_a_waiting_change(qapp):
    session = _Session(enabled=True)
    dialog = session.dialog()
    dialog.show()
    dialog._chk_microscope.setChecked(True)
    dialog.close()
    assert session.calls == [("microscope", True)]
    dialog.deleteLater()


def test_the_dialog_has_no_token_and_no_arming(qapp, host):
    # The agent's level is the chip's; its token is Connect an agent's.
    dialog = AppServerDialog(lambda: host)
    for name in ("_token_edit", "_chk_control", "_chk_configure", "_chk_hardware"):
        assert not hasattr(dialog, name)
    dialog.deleteLater()


def test_no_server_disables_the_event_log(qapp):
    dialog = AppServerDialog(lambda: None)
    assert not dialog._btn_event_log.isEnabled()
    assert "Not running" in dialog._status_label.text()
    dialog.deleteLater()


def test_the_event_log_link_carries_the_dashboard_token(qapp, host):
    url = event_log_url(host)
    assert url == (
        f"http://127.0.0.1:8001/dashboard/log#token={host.auth.dashboard_token}"
    )


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
    assert "server_get_capabilities" in text
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


def test_the_agent_text_says_what_to_do_when_fibsem_is_registered(tmp_path):
    text = agent_setup_text(tmp_path / "fibsem-mcp")
    assert "claude mcp remove -s user fibsem" in text


def test_a_worktree_on_pythonpath_goes_to_the_sidecar_too(tmp_path):
    """Registered bare, the sidecar imports the environment's fibsem, not the
    worktree the app runs (FIB-1429)."""
    path = tmp_path / "env" / "bin" / "fibsem-mcp"
    worktree = str(tmp_path / "worktree")
    text = agent_setup_text(path, platform="darwin", pythonpath=worktree)
    assert (
        f'claude mcp add -s user -e PYTHONPATH="{worktree}" fibsem -- "{path}"' in text
    )
    assert f'PYTHONPATH="{worktree}" "{path}" --check' in text
    entry = text.split('"fibsem": ', 1)[1].split(" under", 1)[0]
    assert json.loads(entry) == {
        "command": str(path),
        "env": {"PYTHONPATH": worktree},
    }


def test_the_running_checkout_is_named_only_when_it_came_from_pythonpath():
    from pathlib import Path

    from fibsem.server.discovery import dev_checkout, fibsem_folder

    checkout = str(Path(fibsem_folder()).parent)
    assert dev_checkout({"PYTHONPATH": ""}) is None
    assert dev_checkout({"PYTHONPATH": os.pathsep.join(["/elsewhere", checkout])}) == (
        checkout
    )


def test_copy_for_your_agent_fills_in_the_running_server(qapp, host, tmp_path):
    from PyQt5.QtWidgets import QApplication

    path = tmp_path / "fibsem-mcp"
    dialog = ConnectAgentDialog(lambda: host, sidecar=path)
    assert not hasattr(dialog, "_btn_copy_code")
    dialog._btn_copy_setup.click()
    copied = QApplication.clipboard().text()
    assert copied == agent_setup_text(path, host.url, pythonpath=dev_checkout())
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
    from fibsem.ui.agent import connect_agent_dialog

    monkeypatch.setattr(connect_agent_dialog, "sidecar_path", lambda: None)
    dialog = ConnectAgentDialog(lambda: host)
    assert not dialog._btn_copy_setup.isEnabled()
    dialog.deleteLater()


def test_a_computer_is_paired_and_unpaired_from_the_dialog(qapp, host, tmp_path):
    from fibsem.server.pairing import PairedTokens, Pairing

    dialog = AppServerDialog(lambda: host)
    assert not dialog._btn_pair.isEnabled()
    host.auth.pairing = Pairing(PairedTokens(tmp_path / "paired.json"))
    dialog.refresh()
    assert not dialog._pair_code.isVisibleTo(dialog)
    dialog._btn_pair.click()
    code = host.auth.pairing.state()["code"]
    assert dialog._pair_code.text() == code
    assert dialog._pair_left.text().endswith("left")
    assert dialog._btn_pair.text() == "Stop pairing"
    host.auth.pairing.pair(code, "lab laptop")
    dialog.refresh()
    assert not dialog._pair_code.isVisibleTo(dialog)
    assert dialog._btn_pair.text() == "Pair a computer"
    row = dialog._computers.itemAt(0).widget().layout()
    assert row.itemAt(0).widget().text() == "lab laptop"
    row.itemAt(2).widget().click()
    assert host.auth.pairing.tokens.computers() == []
    dialog.deleteLater()
