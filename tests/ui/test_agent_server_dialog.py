"""Tools → Agent Server: arming is a live, session-only act on the running
server's AuthConfig — the token never changes, and nothing here persists."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("fastapi")

from fibsem.applications.autolamella.ui.agent_server_dialog import AgentServerDialog
from fibsem.server.auth import AuthConfig, Scope


class _Host:
    running = True
    url = "http://127.0.0.1:8001"

    def __init__(self, arm_control=False):
        self.auth = AuthConfig.generate(arm_control=arm_control, token="tok")


@pytest.fixture
def host():
    return _Host()


def test_arming_flips_the_live_auth_and_keeps_the_token(qapp, host):
    dialog = AgentServerDialog(lambda: host)
    assert not host.auth.is_armed(Scope.CONTROL)

    dialog._chk_control.setChecked(True)
    assert host.auth.is_armed(Scope.CONTROL)
    dialog._chk_control.setChecked(False)
    assert not host.auth.is_armed(Scope.CONTROL)
    assert host.auth.token == "tok"  # arming never re-mints the token
    dialog.deleteLater()


def test_refresh_mirrors_armed_state_without_rearming(qapp):
    host = _Host(arm_control=True)
    dialog = AgentServerDialog(lambda: host)
    assert dialog._chk_control.isChecked()
    # Mirroring must not have *set* anything: disarm externally, refresh follows.
    host.auth.set_armed(Scope.CONTROL, False)
    dialog.refresh()
    assert not dialog._chk_control.isChecked()
    assert not host.auth.is_armed(Scope.CONTROL)
    dialog.deleteLater()


def test_no_server_disables_everything(qapp):
    dialog = AgentServerDialog(lambda: None)
    assert not dialog._chk_control.isEnabled()
    assert not dialog._token_edit.isEnabled()
    assert not dialog._btn_dashboard.isEnabled()
    assert dialog._token_edit.text() == ""
    assert "Not running" in dialog._status_label.text()
    dialog.deleteLater()


def test_dashboard_url_puts_the_token_in_the_fragment(qapp, host):
    # The fragment never leaves the browser — no server log, no referrer —
    # which is the whole reason the button can carry the token at all.
    dialog = AgentServerDialog(lambda: host)
    assert dialog._btn_dashboard.isEnabled()
    assert dialog.dashboard_url(host) == "http://127.0.0.1:8001/dashboard#token=tok"
    dialog.deleteLater()


def test_hardware_stays_visibly_unclimbable(qapp, host):
    dialog = AgentServerDialog(lambda: host)
    assert not dialog._chk_hardware.isEnabled()
    assert not dialog._chk_hardware.isChecked()
    dialog.deleteLater()


def test_the_token_starts_hidden(qapp, host):
    from PyQt5.QtWidgets import QLineEdit

    dialog = AgentServerDialog(lambda: host)
    assert dialog._token_edit.echoMode() == QLineEdit.Password
    dialog._btn_reveal.setChecked(True)
    assert dialog._token_edit.echoMode() == QLineEdit.Normal
    dialog.deleteLater()


def test_status_text_reports_when_the_agent_was_last_heard_from():
    """The presence signal's face: the status line says when the agent's
    token last made a request, in units a human reads at a glance."""
    from fibsem.applications.autolamella.ui.agent_server_dialog import (
        AgentServerDialog,
    )

    class _Host:
        url = "http://127.0.0.1:8001"

        def __init__(self, age):
            self._age = age

        def agent_seconds_since_seen(self):
            return self._age

    assert AgentServerDialog._status_text(_Host(None)).endswith(
        "nothing has connected yet"
    )
    assert "last heard from 5 s ago" in AgentServerDialog._status_text(_Host(5.4))
    assert "last heard from 3 min ago" in AgentServerDialog._status_text(_Host(200))
