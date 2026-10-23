"""The agent chip (FIB-1035): the agent's level in the status bar, set in two
clicks, shown whenever the App Server runs."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.ai.agent.levels import AgentLevel  # noqa: E402
from fibsem.ui.agent.card import AgentCard  # noqa: E402
from fibsem.ui.agent.level_chip import AgentLevelChip, last_heard_text  # noqa: E402


def _labels(widget):
    from PyQt5.QtWidgets import QLabel

    return [
        label.text()
        for label in widget.findChildren(QLabel)
        if label.isVisibleTo(widget)
    ]


def test_the_chip_starts_watching_and_its_card_offers_every_level(qapp):
    chip = AgentLevelChip()
    assert chip.level is AgentLevel.WATCH
    assert chip.text() == "Agent: Watching"
    actions = chip.card.level_menu.actions_
    assert [a.text() for a in actions.values()] == ["Off", "Watch", "Assist", "Drive"]
    assert [a.text() for a in actions.values() if a.isChecked()] == ["Watch"]
    chip.deleteLater()


@pytest.mark.parametrize(
    "level, text",
    [
        (AgentLevel.OFF, "Agent: Off"),
        (AgentLevel.WATCH, "Agent: Watching"),
        (AgentLevel.ASSIST, "Agent: Assisting"),
        (AgentLevel.DRIVE, "Agent: Driving"),
    ],
)
def test_each_level_has_its_word_and_tooltip(qapp, level, text):
    chip = AgentLevelChip()
    chip.set_level(level)
    assert chip.text() == text
    assert "agent token" in chip.toolTip()
    chip.deleteLater()


def test_the_chip_keeps_one_width_at_every_level(qapp):
    chip = AgentLevelChip()
    widths = set()
    for level in AgentLevel:
        chip.set_level(level)
        widths.add(chip.width())
    chip.set_request(_request("applying", results=[{"index": 0, "ok": True}]))
    widths.add(chip.width())
    assert len(widths) == 1
    assert chip.minimumWidth() == chip.maximumWidth()
    chip.deleteLater()


def test_a_click_opens_the_card_on_its_own_and_another_closes_it(qapp):
    chip = AgentLevelChip()
    chip.show()
    chip.click()
    assert chip.popup.isVisible()
    assert chip.card.isVisibleTo(chip.popup)
    chip.click()
    assert not chip.popup.isVisible()
    chip.deleteLater()


def test_choosing_a_level_says_so_and_choosing_the_current_one_does_not(qapp):
    chip = AgentLevelChip()
    chosen = []
    chip.level_chosen.connect(chosen.append)
    menu = chip.card.level_menu
    menu.choose(AgentLevel.DRIVE)
    assert chosen == [AgentLevel.DRIVE]
    # The chip and its menu show what the window tells them, not the click.
    assert chip.level is AgentLevel.WATCH
    assert menu.actions_[AgentLevel.WATCH].isChecked()
    menu.choose(AgentLevel.WATCH)
    assert chosen == [AgentLevel.DRIVE]
    chip.deleteLater()


def test_stop_agent_shows_whenever_the_agent_can_act(qapp):
    chip = AgentLevelChip()
    stopped = []
    chip.stop_clicked.connect(lambda: stopped.append(True))
    for level, visible in (
        (AgentLevel.OFF, False),
        (AgentLevel.WATCH, False),
        (AgentLevel.ASSIST, True),
        (AgentLevel.DRIVE, True),
    ):
        chip.set_level(level)
        assert chip.card.stop_button.isVisibleTo(chip.card) is visible, level
    chip.card.stop_button.click()
    assert stopped == [True]
    chip.deleteLater()


def test_the_card_links_to_the_agent_page_and_the_event_log(qapp):
    seconds = [None]
    chip = AgentLevelChip(last_heard=lambda: seconds[0])
    logs, pages = [], []
    chip.event_log_clicked.connect(lambda: logs.append(True))
    chip.chat_clicked.connect(lambda: pages.append(True))
    chip.refresh_last_heard()
    card = chip.card
    assert card.heard_label.text() == "Agent · not heard from yet"
    card.chat_link.click()
    card.event_log_link.click()
    assert pages == [True] and logs == [True]
    seconds[0] = 4.6
    chip.refresh_last_heard()
    assert card.heard_label.text() == "Agent · last heard 4 s ago"
    chip.deleteLater()


def test_the_card_has_no_way_to_answer_or_message_the_agent(qapp):
    from PyQt5.QtWidgets import QLineEdit, QPlainTextEdit, QPushButton

    card = AgentCard()
    card.set_request(_request())
    card.set_suggestion(_SUGGESTION)
    card.set_asking(_ASKING)
    assert not card.findChildren((QLineEdit, QPlainTextEdit))
    words = {button.text().strip() for button in card.findChildren(QPushButton)}
    assert not words & {"Approve", "Reject…", "Allow", "Deny…", "Decline…", "Send"}
    card.deleteLater()


def test_the_last_heard_line():
    assert last_heard_text(None) == "Agent · not heard from yet"
    assert last_heard_text(200) == "Agent · last heard 3 min ago"
    assert last_heard_text(5, "claude-code/2.1.3") == "claude-code · last heard 5 s ago"


def test_the_session_line_shows_what_the_agent_reported(qapp):
    from fibsem.ai.agent.session import session_text

    reported = {"model": "claude-opus", "goal": "Mill 01", "budget_usd": 20.0}
    assert session_text(None) is None
    assert session_text({"client": "claude-code"}) is None
    assert session_text({**reported, "spent_usd": 2.1}) == (
        "claude-opus · Mill 01 · $2.10 of $20"
    )
    assert session_text({"goal": "Mill 01", "spent_usd": 3}) == "Mill 01 · $3 spent"
    ended = {**reported, "ended": True, "reason": "its budget is spent"}
    assert session_text(ended) == "claude-opus · Mill 01 · ended: its budget is spent"
    # At 80% of the budget the spend leads, so a narrow card keeps it.
    assert session_text({**reported, "spent_usd": 16.4}) == (
        "$16.40 of $20, 82% used · claude-opus · Mill 01"
    )

    chip = AgentLevelChip(agent_session=lambda: "claude-opus · Mill 01")
    assert not chip.card.session_label.isVisibleTo(chip.card)
    chip.refresh_last_heard()
    assert chip.card.session_label.isVisibleTo(chip.card)
    assert chip.card.session_label.text() == "claude-opus · Mill 01"
    chip.card.set_session(None)
    assert not chip.card.session_label.isVisibleTo(chip.card)
    chip.deleteLater()


def test_the_session_line_takes_the_warning_colour_when_the_budget_is_low(qapp):
    from fibsem.ai.agent.session import budget_low
    from fibsem.ui.tokens import TEXT_MUTED_COLOR, WARN_COLOR

    assert not budget_low({"spent_usd": 3.9, "budget_usd": 5})
    assert budget_low({"spent_usd": 4.0, "budget_usd": 5})
    assert not budget_low({"spent_usd": 4.0})
    assert not budget_low({"spent_usd": 4.0, "budget_usd": 0})

    low = [True]
    chip = AgentLevelChip(
        agent_session=lambda: "$4.10 of $5, 82% used", agent_budget_low=lambda: low[0]
    )
    chip.refresh_last_heard()
    assert WARN_COLOR in chip.card.session_label.styleSheet()
    low[0] = False
    chip.refresh_last_heard()
    assert TEXT_MUTED_COLOR in chip.card.session_label.styleSheet()
    chip.deleteLater()


def test_with_nothing_waiting_the_card_says_what_the_agent_is_doing(qapp):
    card = AgentCard()
    card.set_now("stable move")
    assert "stable move" in _labels(card)
    assert not card.waiting_row.isVisibleTo(card)
    card.deleteLater()


def test_now_shows_while_a_request_waits_too(qapp):
    card = AgentCard()
    card.set_request(_request())
    assert not card.now_row.isVisibleTo(card)  # nothing runs, a request waits
    card.set_now("Milling 'trench' · stage 2 of 6")
    assert card.now_row.isVisibleTo(card)
    assert card.waiting_row.isVisibleTo(card)
    assert card.now_label.text() == "Milling 'trench' · stage 2 of 6"
    card.set_request(None)
    card.set_now(None)
    assert not card.now_row.isVisibleTo(card)
    card.deleteLater()


# --- the agent's request (FIB-1283, FIB-1404) -----------------------------------


def _request(status="pending", results=(), request_id="ab12"):
    return {
        "request_id": request_id,
        "status": status,
        "reason": "the trench floor is high on 01",
        "actions": [
            {
                "tool": "item_update_task_config",
                "what": "01 · Rough Milling",
                "change": "Rough Mill 01 · trench depth 0.6 µm → 0.8 µm",
                "details": {
                    "columns": ["Setting", "Now", "New"],
                    "rows": [["Rough Mill 01 · trench depth", "0.6 µm", "0.8 µm"]],
                },
            },
            {
                "tool": "workflow_requeue",
                "what": "01",
                "change": "Queue Rough Milling to run again",
            },
            {
                "tool": "stage_move_relative",
                "what": "Stage",
                "change": "Move 1 µm in x",
            },
        ],
        "results": list(results),
    }


def test_a_waiting_request_turns_the_chip_and_is_one_line_on_the_card(qapp):
    chip = AgentLevelChip()
    chip.set_level(AgentLevel.ASSIST)
    chip.set_request(_request())
    # One request waits at a time: no count.
    assert chip.text() == "Agent asks"
    card = chip.card
    assert card.waiting_word.text() == "Asks"
    assert card.waiting_label.text() == (
        "Rough Mill 01 · trench depth 0.6 µm → 0.8 µm, and 2 more"
    )
    opened = []
    chip.chat_clicked.connect(lambda: opened.append(True))
    card.answer_link.click()
    assert opened == [True]
    chip.set_request(None)
    assert chip.text() == "Agent: Assisting"
    assert not card.waiting_row.isVisibleTo(card)
    chip.deleteLater()


def test_a_request_being_applied_waits_for_no_one(qapp):
    chip = AgentLevelChip()
    chip.set_level(AgentLevel.ASSIST)
    chip.set_request(_request("applying", results=[{"index": 0, "ok": True}]))
    assert chip.text() == "Agent: running 2/3"
    assert not chip.card.waiting_row.isVisibleTo(chip.card)
    chip.deleteLater()


_SUGGESTION = {
    "proposal": "01-causal-pony · Setup Lamella Position · State: Is the lamella centred?",
    "decision": "Continue",
    "reason": "the trench is centred on 01",
}


def test_a_suggestion_turns_the_chip_and_is_one_line_on_the_card(qapp):
    chip = AgentLevelChip()
    chip.set_level(AgentLevel.ASSIST)
    chip.set_suggestion(_SUGGESTION)
    assert chip.text() == "Agent suggests"
    # In the level's look, not amber: the run's Attention Required is.
    from fibsem.ui.agent.style import LEVEL_STYLES

    assert chip.styleSheet() == LEVEL_STYLES[AgentLevel.ASSIST]
    card = chip.card
    assert (card.waiting_word.text(), card.waiting_label.text()) == (
        "Suggests",
        "Continue",
    )
    chip.set_suggestion(None)
    assert chip.text() == "Agent: Assisting"
    assert not card.waiting_row.isVisibleTo(card)
    chip.deleteLater()


def test_a_request_shows_before_a_suggestion(qapp):
    chip = AgentLevelChip()
    chip.set_suggestion(_SUGGESTION)
    chip.set_request(_request())
    assert chip.text() == "Agent asks"
    assert chip.card.waiting_word.text() == "Asks"
    chip.set_request(None)
    assert chip.text() == "Agent suggests"
    assert chip.card.waiting_word.text() == "Suggests"
    chip.deleteLater()


_ASKING = {
    "id": "a1b2",
    "client": "claude-code 2.1.3",
    "origin": "this PC",
    "user": "microscope",
    "asked_at": 0.0,
    "name": "claude-code on this PC",
    "if_allowed": "It can read the session, and acts only at the level you set: "
    "now Watch, so it can look but not act.",
}


def test_an_agent_asking_to_connect_turns_the_chip_and_shows_first(qapp):
    chip = AgentLevelChip()
    chip.set_request(_request())
    chip.set_asking(_ASKING)
    card = chip.card
    # Before anything else: nothing else reaches it until it is allowed.
    assert chip.text() == "Agent asks"
    assert (card.waiting_word.text(), card.waiting_label.text()) == (
        "Connect",
        "claude-code 2.1.3 asks to connect",
    )
    chip.set_asking(None)
    assert card.waiting_word.text() == "Asks"
    chip.deleteLater()


# --- in the window ----------------------------------------------------------


@pytest.fixture(scope="module")
def main_ui(qapp):
    from fibsem.applications.autolamella.ui import window as module

    window = module.AutoLamellaWindow()
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
        self.requests = _Store()

    def agent_seconds_since_seen(self):
        return self.auth.seconds_since_seen()

    def stop_agent(self, microscope):
        from fibsem.server.hosting import AppServerHost

        return AppServerHost.stop_agent(self, microscope)


class _Store:
    """The request store's surface the window uses, holding finished dicts."""

    def __init__(self):
        self.requests = []
        self.decided = []
        self.listeners = []

    def recent(self):
        from fibsem.server.requests import AgentRequest

        out = []
        for d in self.requests:
            r = AgentRequest(id=d["request_id"], actions=[], reason=d["reason"])
            r.status, r.results = d["status"], d["results"]
            r.to_dict = lambda d=d: dict(d)
            out.append(r)
        return out

    def subscribe(self, listener):
        self.listeners.append(listener)
        return lambda: self.listeners.remove(listener)

    def approve(self, request_id):
        self.decided.append(("approve", request_id))
        self.requests[0]["status"] = "applying"

    def reject(self, request_id, reason=""):
        self.decided.append(("reject", request_id, reason))
        self.requests[0]["status"] = "rejected"


@pytest.fixture
def host(main_ui):
    pytest.importorskip("fastapi")
    ui = main_ui
    previous = ui.session.app_server, ui.session.agent_auth
    ui.session.app_server = _Host()
    # As the real host does, the session holds the auth the host checks.
    ui.session.agent_auth = ui.session.app_server.auth
    yield ui.session.app_server
    ui.session.app_server, ui.session.agent_auth = previous
    main_ui.agent_host.refresh()


def test_the_chip_is_hidden_without_a_running_app_server(main_ui):
    main_ui.session.app_server = None
    main_ui.agent_host.refresh()
    assert main_ui.agent_host.chip.isHidden()


def test_the_chip_shows_the_running_server_s_level_and_sets_it(main_ui, host):
    main_ui.agent_host.refresh()
    assert not main_ui.agent_host.chip.isHidden()
    assert main_ui.agent_host.chip.text() == "Agent: Watching"

    main_ui.agent_host.chip.level_chosen.emit(AgentLevel.DRIVE)
    assert host.auth.level is AgentLevel.DRIVE
    assert main_ui.agent_host.chip.text() == "Agent: Driving"

    main_ui.agent_host.chip.level_chosen.emit(AgentLevel.OFF)
    assert host.auth.level is AgentLevel.OFF
    # Off is still shown: the chip is how the agent comes back.
    assert not main_ui.agent_host.chip.isHidden()
    assert main_ui.agent_host.chip.text() == "Agent: Off"

    host.running = False
    main_ui.agent_host.refresh()
    assert main_ui.agent_host.chip.isHidden()


def test_stop_agent_drops_to_watch_and_is_the_operator_s_on_the_record(main_ui, host):
    from fibsem import utils
    from fibsem.workflows.event_recording import EventRecorder

    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    ui = main_ui
    previous = ui.session.recorder
    recorder = EventRecorder(microscope, default_actor="operator")
    ui.session.recorder = recorder
    try:
        main_ui.agent_host.chip.level_chosen.emit(AgentLevel.DRIVE)
        main_ui.agent_host.chip.stop_clicked.emit()
        assert host.auth.level is AgentLevel.WATCH and host.auth.stopped
        assert main_ui.agent_host.chip.text() == "Agent: Watching"

        events = recorder.buffer.events_since(0)["events"]
        kinds = [(e["kind"], e["payload"]) for e in events]
        assert kinds == [
            ("agent_level_set", {"previous": "watch", "level": "drive"}),
            (
                "action",
                {
                    "what": "Stop the agent",
                    "ok": True,
                    "reason": "Stopped by the operator",
                    "stops": "agent",
                    "previous": "drive",
                    "level": "watch",
                },
            ),
        ]
        assert all(e["actor"] == "operator" for e in events)
    finally:
        ui.session.recorder = previous
        recorder.close()
        microscope.disconnect()


def test_the_window_shows_the_request_as_it_changes(main_ui, host):
    host.auth.set_level(AgentLevel.ASSIST)
    host.requests.requests = [_request()]
    main_ui.agent_host.refresh()
    chip = main_ui.agent_host.chip
    assert chip.text() == "Agent asks"
    # A change from the store's worker thread reaches the chip without a poll.
    assert len(host.requests.listeners) == 1
    host.requests.requests[0]["status"] = "rejected"
    main_ui.agent_host.refresh()
    assert chip.text() == "Agent: Assisting"


def test_the_window_shows_an_agent_asking_to_connect(main_ui, host, tmp_path):
    from fibsem.server.connections import Connections
    from fibsem.server.pairing import PairedTokens

    host.auth.connections = Connections(PairedTokens(tmp_path / "paired.json"))
    waiting = host.auth.connections.ask("claude-code 2.1.3", "this PC", "lab", 0)
    main_ui.agent_host.refresh()
    chip = main_ui.agent_host.chip
    assert chip.text() == "Agent asks"
    assert chip.asking["id"] == waiting["id"]
    host.auth.connections.deny(waiting["id"], "not now")
    main_ui.agent_host.refresh()
    assert chip.asking is None
    assert chip.text() == "Agent: Watching"


def test_how_a_request_ended_is_no_toast(main_ui, host, monkeypatch):
    from fibsem.ui import notification_service

    shown = []
    for name in ("show", "show_toast"):
        monkeypatch.setattr(
            notification_service, name, lambda message, *a, **kw: shown.append(message)
        )
    host.auth.set_level(AgentLevel.ASSIST)
    main_ui.agent_host.refresh()
    host.requests.requests = [
        _request("failed", results=[{"index": 0, "ok": False}], request_id="ef56")
    ]
    main_ui.agent_host.refresh()
    assert shown == []


# --- the level as a menu (FIB-1418) ---------------------------------------------


def test_the_level_is_drawn_as_the_chip_and_opens_a_menu_of_the_four(qapp):
    from PyQt5.QtWidgets import QLabel

    from fibsem.ui.agent.style import LEVEL_STYLES

    card = AgentCard()
    card.set_level(AgentLevel.ASSIST)
    menu = card.level_menu
    assert menu.text() == "Assisting  ▾"
    assert menu.styleSheet() == LEVEL_STYLES[AgentLevel.ASSIST]
    # A popover, not a drawer: the card keeps its height.
    assert menu.menu() is menu.menu_
    rows = [action.defaultWidget() for action in menu.actions_.values()]
    shown = [label.text() for row in rows for label in row.findChildren(QLabel)]
    assert {"No access", "Read only", "Asks first", "Full control"} <= set(shown)
    chosen = []
    card.level_chosen.connect(chosen.append)
    menu.choose(AgentLevel.DRIVE)
    # The level stays as the window says until it answers.
    assert chosen == [AgentLevel.DRIVE]
    assert menu.text() == "Assisting  ▾"
    assert menu.actions_[AgentLevel.ASSIST].isChecked()
    card.deleteLater()


def test_the_level_and_stop_agent_keep_their_width_at_every_level(qapp):
    card = AgentCard()
    widths = set()
    for level in AgentLevel:
        card.set_level(level)
        widths.add(card.level_menu.width())
    assert len(widths) == 1
    card.deleteLater()


# --- Recent (FIB-1414), and the clamped Why (FIB-1412) ---------------------------


def test_what_the_agent_does_shows_in_the_card_s_now_line(qapp):
    from types import SimpleNamespace

    from fibsem.acting import AGENT
    from fibsem.control import Control
    from fibsem.ui.agent.activity import AgentActivity
    from fibsem.ui.widgets.status_bar import FibsemStatusBar

    bar = FibsemStatusBar()
    activity = AgentActivity(bar)
    control = Control()
    activity.set_microscope(SimpleNamespace(control=control))
    chip = AgentLevelChip()
    activity.now_changed.connect(chip.set_now)
    chip.set_level(AgentLevel.DRIVE)
    with control.take(AGENT, "move stage relative"):
        assert "move stage relative" in _labels(chip.card)
    assert "move stage relative" not in _labels(chip.card)
    activity.set_microscope(None)
    chip.deleteLater()
    bar.deleteLater()


def test_a_new_request_asks_for_attention_and_opens_nothing(qapp, monkeypatch):
    """FIB-1501: the card opening itself brought the window forward and took
    focus from the operator's browser. A new request or suggestion asks the
    system for attention instead, once per new one."""
    from PyQt5.QtWidgets import QApplication, QMainWindow

    alerts = []
    monkeypatch.setattr(
        QApplication, "alert", lambda widget, ms=0: alerts.append(widget)
    )
    window = QMainWindow()
    chip = AgentLevelChip()
    window.statusBar().addPermanentWidget(chip)
    window.show()
    chip.set_level(AgentLevel.ASSIST)

    chip.set_request(_request())
    assert not chip.popup.isVisible()
    assert alerts == [window]
    # The same request again (the window refreshes it) asks nothing more.
    chip.set_request(_request())
    chip.set_request(_request(status="applying"))
    assert alerts == [window]

    chip.set_request(_request(request_id="cd34"))
    assert len(alerts) == 2 and not chip.popup.isVisible()
    window.close()


def test_a_hidden_chip_asks_for_nothing(qapp, monkeypatch):
    from PyQt5.QtWidgets import QApplication

    alerts = []
    monkeypatch.setattr(
        QApplication, "alert", lambda widget, ms=0: alerts.append(widget)
    )
    chip = AgentLevelChip()
    chip.set_level(AgentLevel.ASSIST)
    chip.set_request(_request())
    assert not chip.popup.isVisible()
    assert alerts == []
    chip.deleteLater()


def test_the_border_is_the_agent_s_purple_while_it_drives(main_ui, host):
    def shown():
        return main_ui.run_border.shown

    main_ui.run_border.set_state("supervised")
    main_ui.agent_host.chip.level_chosen.emit(AgentLevel.DRIVE)
    assert shown() == "agent"
    for state in ("idle", "automated", "pending", "waiting"):
        main_ui.run_border.set_state(state)
        assert shown() == "agent"
    # Stopping is the one state shown over the agent's.
    main_ui.run_border.set_state("stopping")
    assert shown() == "stopping"

    main_ui.run_border.set_state("supervised")
    main_ui.agent_host.chip.level_chosen.emit(AgentLevel.ASSIST)
    assert shown() == "supervised"

    # The border's switch turns the purple off too.
    main_ui.agent_host.chip.level_chosen.emit(AgentLevel.DRIVE)
    enabled = main_ui.run_border.on
    main_ui.run_border.set_on(False)
    assert shown() == "idle"
    main_ui.run_border.set_on(enabled)
    assert shown() == "agent"

    # No App Server, no agent: the run's own colour.
    host.running = False
    main_ui.agent_host.refresh()
    assert shown() == "supervised"
    main_ui.run_border.set_state("idle")


def test_the_agent_s_workflow_stop_turns_the_border_red(main_ui, qapp):
    """The agent's workflow_stop goes through the app's own stop, from the App
    Server's thread, and the border shows stopping as for the operator's Stop."""
    import threading

    from fibsem.applications.autolamella.server import AutoLamellaAppContext

    ui = main_ui
    runner = ui.session.runner
    # Running, as far as a stop can tell: the window hears this runner's stop.
    runner._thread = threading.current_thread()
    main_ui.run_border.set_state("automated")
    answers = []
    try:
        context = AutoLamellaAppContext(ui.session)
        server = threading.Thread(
            target=lambda: answers.append(context.stop_workflow())
        )
        server.start()
        server.join(timeout=5)
        # Painted on the GUI thread, not the server's: nothing until it runs.
        assert main_ui.run_border.shown == "automated"
        qapp.processEvents()
        assert answers == [{"available": True, "stopped": True}]
        assert main_ui.run_border.shown == "stopping"
    finally:
        runner._thread = None
        main_ui.run_border.state = None
        main_ui.run_border.set_state("idle")


def test_the_coincidence_viewer_follows_the_border_switch(
    main_ui, tmp_path, monkeypatch
):
    """Show Workflow Border hides the coincidence viewer's border too, when it is
    turned off and when the viewer opens with it off."""
    from fibsem import config as cfg
    from fibsem import utils

    # The switch saves itself: to a file of this test's, not the user's.
    monkeypatch.setattr(cfg, "USER_PREFERENCES_PATH", str(tmp_path / "prefs.yaml"))
    from fibsem.applications.autolamella.structures import Experiment
    from fibsem.applications.autolamella.ui.coincidence.viewer import (
        FluorescenceCoincidenceViewerWidget,
    )

    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    viewer = FluorescenceCoincidenceViewerWidget(
        microscope=microscope,
        experiment=Experiment(path=str(tmp_path), name="border-switch"),
    )
    enabled, driving = main_ui.run_border.on, main_ui.run_border.agent_driving

    def shown():
        return viewer.run_border.shown

    try:
        main_ui.run_border.agent_driving = True
        main_ui.menu_bar._on_border_toggle(True)
        viewer.run_border.set_state("supervised")
        # Opened: it takes the switch and the agent's purple from the window.
        main_ui.run_border.mirror_to(viewer.run_border)
        assert shown() == "agent"
        main_ui.menu_bar._on_border_toggle(False)
        assert shown() == "idle"
        main_ui.menu_bar._on_border_toggle(True)
        assert shown() == "agent"
    finally:
        main_ui.run_border.agent_driving = driving
        main_ui.menu_bar._on_border_toggle(enabled)
        viewer.deleteLater()


def test_the_chip_pulses_while_the_agent_acts_and_ends_a_whole_cycle(qapp):
    from fibsem.ui.agent.level_chip import (
        PULSE_MS,
        PULSE_STEP_MS,
    )
    from fibsem.ui.agent.style import LEVEL_STYLES

    chip = AgentLevelChip()
    chip.set_level(AgentLevel.DRIVE)
    assert not chip.pulsing
    chip.set_now("focus sweep")
    assert chip.pulsing

    steps = PULSE_MS // PULSE_STEP_MS
    for _ in range(steps // 2):
        chip._on_pulse()
    # Halfway, the fill is at its faintest.
    assert chip.styleSheet() != LEVEL_STYLES[AgentLevel.DRIVE]
    assert chip.text() == "Agent: Driving"

    # The action ends: the cycle runs out, then the chip is still.
    chip.set_now(None)
    assert chip.pulsing
    for _ in range(steps - steps // 2):
        chip._on_pulse()
    assert not chip.pulsing
    assert chip.styleSheet() == LEVEL_STYLES[AgentLevel.DRIVE]

    # A request waiting shows over the pulse.
    chip.set_now("move stage relative")
    chip.set_request(_request())
    chip._on_pulse()
    assert chip.text() == "Agent asks"
    chip.deleteLater()


def test_the_chip_names_the_step_it_is_running():
    """Round 10 (FIB-1492): "applying" read as writing a config, whatever the
    step was."""
    from fibsem.ui.agent.level_chip import chip_request_text

    def running(tool):
        return chip_request_text(
            {"status": "applying", "actions": [{"tool": tool}], "results": []}
        )

    assert running("milling_run_task") == "Agent: milling 1/1"
    assert running("stage_move_relative") == "Agent: moving 1/1"
    assert running("beam_acquire_image") == "Agent: imaging 1/1"
    assert running("item_update") == "Agent: running 1/1"
