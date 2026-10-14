"""The agent chip (FIB-1035): the agent's level in the status bar, set in two
clicks, shown whenever the App Server runs."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.ui.agent_level_chip import (  # noqa: E402
    AgentLevelChip,
    last_heard_text,
    request_failure_text,
)
from fibsem.server.levels import AgentLevel  # noqa: E402


def _menu_names(chip):
    return [a.text() for a in chip.menu_.actions() if a.isCheckable()]


def test_the_chip_starts_watching_and_offers_every_level(qapp):
    chip = AgentLevelChip()
    assert chip.level is AgentLevel.WATCH
    assert chip.text() == "Agent: Watching"
    assert _menu_names(chip) == ["Off", "Watch", "Assist", "Drive"]
    checked = [a.text() for a in chip.menu_.actions() if a.isChecked()]
    assert checked == ["Watch"]
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


def test_each_level_row_has_its_line_and_a_click_chooses_it(qapp):
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QLabel

    chip = AgentLevelChip()
    chosen = []
    chip.level_chosen.connect(chosen.append)
    by_name = {a.text(): a for a in chip.menu_.actions()}
    row = by_name["Drive"].defaultWidget()
    labels = [label.text() for label in row.findChildren(QLabel)]
    assert "Drive" in labels and "Full control" in labels
    QTest.mouseClick(row, Qt.LeftButton)
    assert chosen == [AgentLevel.DRIVE]
    chip.deleteLater()


def test_stop_agent_is_first_whenever_the_agent_can_act(qapp):
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
        assert chip.stop_action.isVisible() is visible, level
        visible_actions = [a for a in chip.menu_.actions() if a.isVisible()]
        assert (visible_actions[0] is chip.stop_action) is visible
    chip.stop_action.trigger()
    assert stopped == [True]
    chip.deleteLater()


def test_the_menu_offers_to_connect_an_agent(qapp):
    chip = AgentLevelChip()
    clicked = []
    chip.connect_clicked.connect(lambda: clicked.append(True))
    assert chip.menu_.actions()[-1].text() == "Connect an agent…"
    chip.connect_action.trigger()
    assert clicked == [True]
    chip.deleteLater()


def test_the_menu_opens_the_event_log(qapp):
    chip = AgentLevelChip()
    clicked = []
    chip.event_log_clicked.connect(lambda: clicked.append(True))
    assert chip.menu_.actions()[-2] is chip.event_log_action
    assert chip.event_log_action.text() == "Open event log"
    chip.event_log_action.trigger()
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
    assert last_heard_text(5, "claude-code/2.1.3") == "claude-code · last heard 5 s ago"
    chip.deleteLater()


# --- the agent's request (FIB-1283) -------------------------------------------


def _request(status="pending", results=(), request_id="ab12"):
    return {
        "request_id": request_id,
        "status": status,
        "reason": "the trench floor is high on 01",
        "actions": [
            {
                "tool": "update_item_task_config",
                "what": "01 · Rough Milling",
                "change": "depth: 6e-07 → 8e-07",
            },
            {
                "tool": "requeue_task",
                "what": "01",
                "change": "Queue Rough Milling to run again",
            },
            {
                "tool": "move_stage_relative",
                "what": "Stage",
                "change": "Move by x 1 µm",
            },
        ],
        "results": list(results),
    }


def _labels(widget):
    from PyQt5.QtWidgets import QLabel

    return [
        label.text()
        for label in widget.findChildren(QLabel)
        if label.isVisibleTo(widget)
    ]


def test_a_waiting_request_turns_the_chip_and_tops_the_menu(qapp):
    chip = AgentLevelChip()
    chip.set_level(AgentLevel.ASSIST)
    assert not chip.request_action.isVisible()
    chip.set_request(_request())
    assert chip.text() == "Agent asks · 3"
    assert chip.request_action.isVisible()
    assert chip.menu_.actions()[0] is chip.request_action
    shown = _labels(chip.request_block)
    assert "01 · Rough Milling" in shown and "depth: 6e-07 → 8e-07" in shown
    assert "Why: the trench floor is high on 01" in shown
    chip.set_request(None)
    assert chip.text() == "Agent: Assisting"
    assert not chip.request_action.isVisible()
    chip.deleteLater()


def test_approve_and_reject_say_which_request_and_why(qapp):
    chip = AgentLevelChip()
    chip.set_level(AgentLevel.ASSIST)
    chip.set_request(_request())
    approved, rejected = [], []
    chip.approve_clicked.connect(approved.append)
    chip.reject_clicked.connect(lambda rid, why: rejected.append((rid, why)))
    block = chip.request_block

    block._approve.click()
    assert approved == ["ab12"]

    block._reject.click()
    assert block._reason.isVisibleTo(block) and not block._approve.isVisibleTo(block)
    # Re-rendered, the actions are shown once.
    assert _labels(block).count("Move by x 1 µm") == 1
    block._back.click()
    assert block._approve.isVisibleTo(block)
    block._reject.click()
    block._reason.setText("leave 01, I'll check it")
    block._confirm_reject.click()
    assert rejected == [("ab12", "leave 01, I'll check it")]
    chip.deleteLater()


def test_an_approved_request_shows_its_progress(qapp):
    chip = AgentLevelChip()
    chip.set_level(AgentLevel.ASSIST)
    chip.set_request(_request("applying", results=[{"index": 0, "ok": True}]))
    assert chip.text() == "Agent: applying 2/3"
    block = chip.request_block
    shown = _labels(block)
    assert "✓" in shown and "…" in shown and "○" in shown
    assert "Approved by you. Applying 2 of 3…" in shown
    assert not block._approve.isVisibleTo(block)
    chip.deleteLater()


def test_a_refused_request_is_said_in_one_line():
    failed = _request(
        "failed",
        results=[
            {"index": 0, "ok": True},
            {
                "index": 1,
                "ok": False,
                "status_code": 409,
                "error": {"message": "Rough Milling is running on 01"},
            },
        ],
    )
    assert request_failure_text(failed) == (
        "1 of 3 applied; refused at 2: Rough Milling is running on 01"
    )


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
        self.requests = _Store()

    def agent_seconds_since_seen(self):
        return self.auth.seconds_since_seen()

    def stop_agent(self, microscope):
        from fibsem.applications.autolamella.server.hosting import AppServerHost

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


def test_stop_agent_drops_to_watch_and_is_the_operator_s_on_the_record(main_ui, host):
    from fibsem import utils
    from fibsem.applications.autolamella.event_recording import EventRecorder

    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    ui = main_ui.autolamella_ui
    previous = ui._event_recorder
    recorder = EventRecorder(microscope, default_actor="operator")
    ui._event_recorder = recorder
    try:
        main_ui.agent_chip.level_chosen.emit(AgentLevel.DRIVE)
        main_ui.agent_chip.stop_clicked.emit()
        assert host.auth.level is AgentLevel.WATCH and host.auth.stopped
        assert main_ui.agent_chip.text() == "Agent: Watching"

        events = recorder.buffer.events_since(0)["events"]
        kinds = [(e["kind"], e["payload"]) for e in events]
        assert kinds == [
            ("agent_level_set", {"previous": "watch", "level": "drive"}),
            (
                "agent_stopped",
                {"previous": "drive", "level": "watch", "stopped": None},
            ),
        ]
        assert all(e["actor"] == "operator" for e in events)
    finally:
        ui._event_recorder = previous
        recorder.close()
        microscope.disconnect()


def test_the_window_shows_the_request_and_passes_on_the_decision(main_ui, host):
    host.auth.set_level(AgentLevel.ASSIST)
    host.requests.requests = [_request()]
    main_ui._refresh_agent_chip()
    chip = main_ui.agent_chip
    assert chip.text() == "Agent asks · 3"
    # A change from the store's worker thread reaches the chip without a poll.
    assert len(host.requests.listeners) == 1

    chip.approve_clicked.emit("ab12")
    assert host.requests.decided == [("approve", "ab12")]
    assert chip.text() == "Agent: applying 1/3"

    host.requests.requests = [_request("pending", request_id="cd34")]
    main_ui._refresh_agent_chip()
    chip.reject_clicked.emit("cd34", "not now")
    assert host.requests.decided[-1] == ("reject", "cd34", "not now")
    assert chip.text() == "Agent: Assisting"


def test_a_refused_request_stays_on_the_status_bar_once(main_ui, host):
    failed = _request(
        "failed",
        results=[
            {
                "index": 0,
                "ok": False,
                "status_code": 409,
                "error": {"message": "the workflow is in control"},
            }
        ],
        request_id="ef56",
    )
    host.auth.set_level(AgentLevel.ASSIST)
    host.requests.requests = [failed]
    main_ui._refresh_agent_chip()
    assert main_ui.status_bar.failure == (
        "0 of 3 applied; refused at 1: the workflow is in control"
    )
    main_ui.status_bar.dismiss_failure()
    main_ui._refresh_agent_chip()
    assert main_ui.status_bar.failure is None
