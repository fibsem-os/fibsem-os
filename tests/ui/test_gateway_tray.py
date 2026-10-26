"""The gateway's tray: its menu follows the app, and its icon's colour."""

import pytest

pytest.importorskip("PyQt5")

from fibsem.gateway.gateway import (  # noqa: E402
    OFF,
    RUNNING,
    Gateway,
    GatewayState,
)
from fibsem.gateway.server import GatewayAuth  # noqa: E402
from fibsem.gateway.ui.tray import GatewayTray, mode_colour  # noqa: E402
from fibsem.theme.tokens import PRIMARY_COLOR, SEMANTIC_WARNING_COLOR  # noqa: E402


def _texts(menu):
    found = []
    for action in menu.actions():
        if action.isSeparator():
            continue
        found.append((action.text(), action.isEnabled()))
        if action.menu() is not None:
            found += [("  " + t, e) for t, e in _texts(action.menu())]
    return found


@pytest.fixture
def tray(qapp, tmp_path):
    gateway = Gateway(home=tmp_path, discovery=lambda: None)
    auth = GatewayAuth.from_disk(tmp_path / "token.json", tmp_path / "paired.json")
    tray = GatewayTray(gateway, auth, 8010)
    yield tray, gateway
    tray._timer.stop()
    tray.hide()


def test_off_offers_both_starts(tray):
    tray, _ = tray
    tray._build_menu()
    texts = dict(_texts(tray.menu))
    assert texts["App: Off"] is False
    assert texts["Start with the window"] and texts["  AutoLamella"]
    assert texts["Start without a window"] and texts["    Last experiment"]
    assert texts["Stop"] is False
    assert "fibsemOS Gateway: Off" == tray.toolTip()


def test_a_window_is_stopped_at_the_window(tray):
    tray, gateway = tray
    gateway._state.mode, gateway._state.window = RUNNING, True
    tray._build_menu()
    texts = dict(_texts(tray.menu))
    assert texts["App: Running with the window"] is False
    assert texts["Start with the window"] is False
    assert texts["Start without a window"] is False
    assert texts["Stop (close the window)"] is False


def test_an_app_without_its_window_is_stopped_from_the_tray(tray):
    tray, gateway = tray
    gateway._state.mode = RUNNING
    tray._build_menu()
    texts = dict(_texts(tray.menu))
    assert texts["App: Running without a window"] is False
    assert texts["Stop"] is True


def test_pairing_a_hub_opens_a_code_with_its_grant(tray, monkeypatch):
    from fibsem.gateway.ui import tray as tray_module

    shown = []
    monkeypatch.setattr(
        tray_module.QMessageBox, "information", lambda *args: shown.append(args[2])
    )
    tray, _ = tray
    tray._pair("hub", ("switch_modes",))
    state = tray._auth.pairing.state()
    assert (state["role"], state["grants"]) == ("hub", ["switch_modes"])
    assert state["code"] in shown[0]


def test_the_icon_colour_follows_the_app():
    def state(mode, window=False, switching=False):
        return GatewayState(mode, "test", window=window, switching=switching)

    assert mode_colour(state(RUNNING, window=True)) == PRIMARY_COLOR
    assert mode_colour(state(RUNNING, switching=True)) == SEMANTIC_WARNING_COLOR
    assert mode_colour(state(OFF)) != mode_colour(state(RUNNING))
