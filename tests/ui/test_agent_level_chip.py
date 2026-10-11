"""The agent chip (FIB-1035): the agent's level in the status bar, set in two
clicks, shown whenever the App Server runs."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.ui.agent_level_chip import (  # noqa: E402
    AgentLevelChip,
    last_heard_text,
)
from fibsem.server.levels import AgentLevel  # noqa: E402


def _menu_names(chip):
    return [a.text() for a in chip.menu_.actions() if a.isCheckable()]


def test_the_chip_starts_watching_and_offers_off_watch_drive(qapp):
    chip = AgentLevelChip()
    assert chip.level is AgentLevel.WATCH
    assert chip.text() == "Agent: Watching"
    # Assist joins with its approvals (FIB-1283).
    assert _menu_names(chip) == ["Off", "Watch", "Drive"]
    checked = [a.text() for a in chip.menu_.actions() if a.isChecked()]
    assert checked == ["Watch"]
    chip.deleteLater()


@pytest.mark.parametrize(
    "level, text",
    [
        (AgentLevel.OFF, "Agent: Off"),
        (AgentLevel.WATCH, "Agent: Watching"),
        (AgentLevel.DRIVE, "Agent: Driving"),
    ],
)
def test_each_level_has_its_word_and_tooltip(qapp, level, text):
    chip = AgentLevelChip()
    chip.set_level(level)
    assert chip.text() == text
    assert "agent token" in chip.toolTip()
    chip.deleteLater()


def test_choosing_a_level_says_so_and_choosing_the_current_one_does_not(qapp):
    chip = AgentLevelChip()
    chosen = []
    chip.level_chosen.connect(chosen.append)
    by_name = {a.text(): a for a in chip.menu_.actions()}
    by_name["Drive"].trigger()
    assert chosen == [AgentLevel.DRIVE]
    # The chip shows what the window tells it, not what was clicked.
    assert chip.level is AgentLevel.WATCH
    by_name["Watch"].trigger()
    assert chosen == [AgentLevel.DRIVE]
    chip.deleteLater()


def test_the_menu_offers_to_connect_an_agent(qapp):
    chip = AgentLevelChip()
    clicked = []
    chip.connect_clicked.connect(lambda: clicked.append(True))
    assert chip.menu_.actions()[-1].text() == "Connect an agent…"
    chip.connect_action.trigger()
    assert clicked == [True]
    chip.deleteLater()


def test_the_menu_says_when_the_agent_was_last_heard_from(qapp):
    seconds = [None]
    chip = AgentLevelChip(last_heard=lambda: seconds[0])
    chip.refresh_last_heard()
    assert chip.last_heard_action.text() == "Agent · not heard from yet"
    seconds[0] = 4.6
    chip.menu_.aboutToShow.emit()
    assert chip.last_heard_action.text() == "Agent · last heard 4 s ago"
    assert last_heard_text(200) == "Agent · last heard 3 min ago"
    chip.deleteLater()


# --- in the window ----------------------------------------------------------


@pytest.fixture(scope="module")
def main_ui(qapp):
    from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module

    window = module.AutoLamellaSingleWindowUI()
    yield window
    original_quit = qapp.quit
    qapp.quit = lambda: None
    try:
        window.close()
    finally:
        qapp.quit = original_quit


class _Host:
    def __init__(self):
        from fibsem.server import AuthConfig

        self.running = True
        self.auth = AuthConfig.generate()
        self.auth.set_level(AgentLevel.WATCH)

    def agent_seconds_since_seen(self):
        return self.auth.seconds_since_seen()


@pytest.fixture
def host(main_ui):
    pytest.importorskip("fastapi")
    ui = main_ui.autolamella_ui
    previous = ui._app_server_host
    ui._app_server_host = _Host()
    yield ui._app_server_host
    ui._app_server_host = previous
    main_ui._refresh_agent_chip()


def test_the_chip_is_hidden_without_a_running_app_server(main_ui):
    main_ui.autolamella_ui._app_server_host = None
    main_ui._refresh_agent_chip()
    assert main_ui.agent_chip.isHidden()


def test_the_chip_shows_the_running_server_s_level_and_sets_it(main_ui, host):
    main_ui._refresh_agent_chip()
    assert not main_ui.agent_chip.isHidden()
    assert main_ui.agent_chip.text() == "Agent: Watching"

    main_ui.agent_chip.level_chosen.emit(AgentLevel.DRIVE)
    assert host.auth.level is AgentLevel.DRIVE
    assert main_ui.agent_chip.text() == "Agent: Driving"

    main_ui.agent_chip.level_chosen.emit(AgentLevel.OFF)
    assert host.auth.level is AgentLevel.OFF
    # Off is still shown: the chip is how the agent comes back.
    assert not main_ui.agent_chip.isHidden()
    assert main_ui.agent_chip.text() == "Agent: Off"

    host.running = False
    main_ui._refresh_agent_chip()
    assert main_ui.agent_chip.isHidden()
