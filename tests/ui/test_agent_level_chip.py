"""The agent chip (FIB-1035): the agent's level in the status bar, set in two
clicks, shown whenever the App Server runs."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.ui.agent_card import AgentCard  # noqa: E402
from fibsem.applications.autolamella.ui.agent_level_chip import (  # noqa: E402
    AgentLevelChip,
    last_heard_text,
    request_outcome,
)
from fibsem.server.levels import AgentLevel  # noqa: E402
from fibsem.structures import BeamType  # noqa: E402


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


def test_the_card_opens_the_event_log_and_offers_to_connect_an_agent(qapp):
    seconds = [None]
    chip = AgentLevelChip(last_heard=lambda: seconds[0])
    logs, connects = [], []
    chip.event_log_clicked.connect(lambda: logs.append(True))
    chip.connect_clicked.connect(lambda: connects.append(True))
    chip.refresh_last_heard()
    card = chip.card
    assert card.connect_link.isVisibleTo(card)
    card.connect_link.click()
    card.event_log_link.click()
    assert connects == [True] and logs == [True]
    # Once heard from, the card says who and when instead of offering to connect.
    seconds[0] = 4.6
    chip.refresh_last_heard()
    assert not card.connect_link.isVisibleTo(card)
    assert card.heard_label.text() == "Agent · last heard 4 s ago"
    chip.deleteLater()


def test_the_last_heard_line():
    assert last_heard_text(None) == "Agent · not heard from yet"
    assert last_heard_text(200) == "Agent · last heard 3 min ago"
    assert last_heard_text(5, "claude-code/2.1.3") == "claude-code · last heard 5 s ago"


def test_with_nothing_waiting_the_card_says_what_the_agent_is_doing(qapp):
    card = AgentCard()
    card.set_now("stable move")
    shown = _labels(card)
    assert "stable move" in shown
    # Nothing waits, so no "Waiting for you".
    assert "WAITING FOR YOU" not in shown
    card.deleteLater()


def test_now_shows_under_the_title_while_a_request_waits_too(qapp):
    # FIB-1414: at Assist the card showed the request and nothing of the mill
    # still running.
    card = AgentCard()
    card.set_request(_request())
    assert not card.now_row.isVisibleTo(card)  # nothing runs, a request waits
    card.set_now("Milling 'trench' · stage 2 of 6")
    assert card.now_row.isVisibleTo(card)
    assert card.now_label.text() == "Milling 'trench' · stage 2 of 6"
    card.set_request(None)
    card.set_now(None)
    # Nothing runs: no Now line at all (FIB-1426).
    assert not card.now_row.isVisibleTo(card)
    card.deleteLater()


def test_a_tall_card_keeps_its_rows_packed_at_the_top(qapp):
    # FIB-1427: a large floating window spread Now, Recent and the footer apart.
    card = AgentCard()
    card.resize(900, 740)
    card.show()
    card.set_now("Milling 'trench'")
    short = AgentCard()
    short.set_now("Milling 'trench'")
    short.resize(900, short.sizeHint().height())
    short.show()
    qapp.processEvents()
    for tall_one, short_one in (
        (card.recent_link, short.recent_link),
        (card.level_menu, short.level_menu),
    ):
        assert tall_one.mapTo(card, tall_one.rect().topLeft()).y() == (
            short_one.mapTo(short, short_one.rect().topLeft()).y()
        )
    card.deleteLater()
    short.deleteLater()


def test_the_image_shows_beside_the_steps_and_follows_the_card_s_width(qapp):
    # Patrick, 2026-10-15 (layout E): the steps and Why left, the image right,
    # wider as the card is; stacked under them with details open.
    from PyQt5.QtGui import QImage

    from fibsem.applications.autolamella.ui.agent_card import PICTURE_MIN

    card = AgentCard()
    card.resize(780, 400)
    card.show()
    card.set_request(_request())
    assert card.image_label is None
    image = QImage(1024, 683, QImage.Format_RGB32)
    legend = [("#8b5cf6", "target", "move target")]
    card.set_image(image, "FIB · 07:41", legend=legend)
    narrow = card.image_label.pixmap().width()
    assert narrow >= PICTURE_MIN
    assert card.image_label.toolTip() == "FIB · 07:41 · click to enlarge"
    # Wider: the image takes the width the steps don't need.
    card.resize(1000, 400)
    assert card.image_label.pixmap().width() == narrow + 220
    # Details open: stacked, the image the card's width.
    card.details_link.click()
    margins = card.layout().contentsMargins()
    assert card.image_label.pixmap().width() == 1000 - margins.left() - margins.right()
    card.image_label.enlarge()
    assert card.image_label.window_.isVisible()
    card.image_label.window_.close()
    # A suggestion's image is its own: the request's stays with the request.
    card.set_image(None, suggestion=True)
    assert card.image_label is not None
    card.set_image(None)
    assert card.image_label is None
    # No image to draw on: why, in its place (FIB-1424).
    card.set_image(None, "No FIB image of 05-pale-lark yet to draw on")
    qapp.processEvents()
    assert "No FIB image of 05-pale-lark yet to draw on" in _labels(card)
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
                "change": "Move by x 1 µm",
            },
        ],
        "results": list(results),
    }


def test_a_waiting_request_turns_the_chip_and_fills_the_card(qapp):
    chip = AgentLevelChip()
    chip.set_level(AgentLevel.ASSIST)
    chip.set_request(_request())
    # One request waits at a time: no count.
    assert chip.text() == "Agent asks"
    shown = _labels(chip.card)
    assert "01 · Rough Milling" in shown
    assert "Rough Mill 01 · trench depth 0.6 µm → 0.8 µm" in shown
    assert "the trench floor is high on 01" in shown
    chip.set_request(None)
    assert chip.text() == "Agent: Assisting"
    assert "Rough Mill 01 · trench depth 0.6 µm → 0.8 µm" not in _labels(chip.card)
    chip.deleteLater()


def test_show_details_opens_every_value(qapp):
    card = AgentCard()
    card.set_request(_request())
    assert "0.6 µm" not in _labels(card)
    card.details_link.click()
    assert {"Setting", "0.6 µm", "0.8 µm"} <= set(_labels(card))
    card.details_link.click()
    assert "0.6 µm" not in _labels(card)
    card.deleteLater()


def test_approve_and_reject_say_which_request_and_why(qapp):
    chip = AgentLevelChip()
    chip.set_level(AgentLevel.ASSIST)
    chip.set_request(_request())
    approved, rejected = [], []
    chip.approve_clicked.connect(approved.append)
    chip.reject_clicked.connect(lambda rid, why: rejected.append((rid, why)))
    card = chip.card

    card.approve_button.click()
    assert approved == ["ab12"]

    card.reject_button.click()
    assert card.reason_edit is not None and card.approve_button is None
    # Re-rendered, the actions are shown once.
    assert _labels(card).count("Move by x 1 µm") == 1
    card.back_button.click()
    assert card.approve_button is not None
    card.reject_button.click()
    card.reason_edit.setText("leave 01, I'll check it")
    card.confirm_reject_button.click()
    assert rejected == [("ab12", "leave 01, I'll check it")]
    chip.deleteLater()


def test_an_approved_request_shows_its_progress(qapp):
    chip = AgentLevelChip()
    chip.set_level(AgentLevel.ASSIST)
    chip.set_request(_request("applying", results=[{"index": 0, "ok": True}]))
    assert chip.text() == "Agent: applying 2/3"
    card = chip.card
    shown = _labels(card)
    assert "✓" in shown and "…" in shown and "○" in shown
    assert "Approved by you. Running 2 of 3…" in shown
    assert card.approve_button is None
    chip.deleteLater()


def test_an_ended_request_leaves_the_card(qapp):
    """FIB-1400: nothing of an ended request stays behind."""
    chip = AgentLevelChip()
    chip.set_level(AgentLevel.ASSIST)
    chip.set_request(_request("applying", results=[{"index": 0, "ok": True}] * 2))
    chip.set_request(None)
    shown = _labels(chip.card)
    assert not any("Approved by you" in text for text in shown)
    chip.deleteLater()


_SUGGESTION = {
    "question": "Is the lamella position good?",
    "answer": "Continue",
    "reason": "the trench is centred on 01",
}


def test_a_suggestion_turns_the_chip_and_the_card_offers_accept(qapp):
    chip = AgentLevelChip()
    chip.set_level(AgentLevel.ASSIST)
    chip.set_suggestion(_SUGGESTION)
    assert chip.text() == "Agent suggests"
    card = chip.card
    assert card.accept_button is not None and card.approve_button is None
    assert {
        "Agent suggests an answer",
        "Is the lamella position good?",
        "Answer: Continue",
        "the trench is centred on 01",
    } <= set(_labels(card))

    accepted = []
    chip.accept_clicked.connect(lambda: accepted.append(True))
    chip.toggle_card()
    card.accept_button.click()
    assert accepted == [True]
    assert not chip.popup.isVisible()

    chip.set_suggestion(None)
    assert chip.text() == "Agent: Assisting"
    assert card.accept_button is None


def test_a_suggestion_can_be_dismissed_with_why(qapp):
    # FIB-1424: Dismiss beside Accept, an optional why, then Dismiss.
    chip = AgentLevelChip()
    chip.set_level(AgentLevel.ASSIST)
    chip.set_suggestion(_SUGGESTION)
    card = chip.card
    dismissed = []
    chip.dismiss_clicked.connect(dismissed.append)
    card.dismiss_button.click()  # "Dismiss…": asks why
    assert card.reason_edit is not None and card.accept_button is None
    card.back_button.click()
    assert card.reason_edit is None and card.accept_button is not None
    card.dismiss_button.click()
    card.reason_edit.setText("the trench is off to the left")
    card.dismiss_button.click()
    assert dismissed == ["the trench is off to the left"]
    chip.set_suggestion(None)
    assert card.dismiss_button is None and card.reason_edit is None


def test_a_request_shows_before_a_suggestion(qapp):
    chip = AgentLevelChip()
    chip.set_suggestion(_SUGGESTION)
    chip.set_request(_request())
    assert chip.text() == "Agent asks"
    assert chip.card.approve_button is not None
    assert chip.card.accept_button is None
    chip.set_request(None)
    assert chip.text() == "Agent suggests"


def test_the_card_stacks_when_narrow(qapp):
    card = AgentCard()
    card.set_request(_request())
    card.show()  # a hidden widget's resize event waits until it is shown
    card.resize(820, card.sizeHint().height())
    assert card.wide
    card.resize(380, card.sizeHint().height())
    assert not card.wide
    card.deleteLater()


def test_how_a_request_ended_is_said_in_one_line():
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
    assert request_outcome(failed) == (
        "Agent's request failed at 2 of 3: rough Milling is running on 01. "
        "Nothing else ran.",
        "error",
    )
    assert request_outcome(_request("applied")) == (
        "Agent's request applied: Rough Mill 01 · trench depth 0.6 µm → 0.8 µm, "
        "and 2 more",
        "success",
    )
    assert request_outcome(_request("cancelled"))[1] == "info"
    assert request_outcome(_request("rejected")) is None
    assert request_outcome(_request("pending")) is None


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
    assert chip.text() == "Agent asks"
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


def test_how_a_request_ended_is_a_toast_once_not_the_status_bar(
    main_ui, host, monkeypatch
):
    from fibsem.ui import notification_service

    shown = []
    monkeypatch.setattr(
        notification_service,
        "show",
        lambda message, kind="info", temporary=False, action=None: shown.append(
            (message, kind, action is not None)
        ),
    )
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
    main_ui._refresh_agent_chip()
    host.requests.requests = [failed]
    main_ui._refresh_agent_chip()
    main_ui._refresh_agent_chip()
    assert shown == [
        (
            "Agent's request failed at 1 of 3: the workflow is in control. "
            "Nothing else ran.",
            "error",
            True,
        )
    ]
    assert main_ui.status_bar.failure is None


def test_the_window_shows_a_suggestion_on_the_card_and_says_when_it_is_dropped(
    main_ui, host, monkeypatch
):
    from fibsem.ui import notification_service

    shown = []
    monkeypatch.setattr(
        notification_service,
        "show",
        lambda message, kind="info", temporary=False, action=None: shown.append(
            (message, kind)
        ),
    )
    host.auth.set_level(AgentLevel.ASSIST)
    main_ui._refresh_agent_chip()
    prompt = main_ui.autolamella_ui
    prompt.suggestion_changed.emit(_SUGGESTION)
    assert main_ui.agent_chip.text() == "Agent suggests"
    assert main_ui.agent_chip.suggestion == _SUGGESTION

    # Taken down by an answer that took it: no toast.
    prompt.suggestion_changed.emit(None)
    main_ui._on_question_event(
        "prompt_answered", {"answered_by": "operator", "suggested_by": "agent"}
    )
    assert shown == []

    # Dropped by the operator answering otherwise: a toast says so.
    prompt.suggestion_changed.emit(_SUGGESTION)
    prompt.suggestion_changed.emit(None)
    main_ui._on_question_event("prompt_answered", {"answered_by": "operator"})
    assert shown == [
        ("Agent's suggestion dropped: you answered the question yourself", "info")
    ]
    assert main_ui.agent_chip.suggestion is None


def test_the_window_draws_a_waiting_request_s_mill_on_the_card_and_takes_it_down(
    main_ui, host
):
    rectangle = {
        "type": "Rectangle",
        "width": 2e-6,
        "height": 1e-6,
        "depth": 2e-7,
        "centre_x": 0.0,
        "centre_y": 0.0,
    }
    waiting = _request(request_id="mill1")
    waiting["actions"] = [
        {
            "tool": "service_call_command",
            "arguments": {
                "service": "milling",
                "command": "prepare",
                "kwargs": {"patterns": [rectangle]},
            },
            "what": "Milling",
            "change": "Prepare: 1 rectangle",
        }
    ]
    from fibsem.structures import FibsemImage

    main_ui.view_controller.set_image(
        BeamType.ION, FibsemImage.generate_blank_image(resolution=(200, 100))
    )
    host.auth.set_level(AgentLevel.ASSIST)
    host.requests.requests = [waiting]
    main_ui._refresh_agent_chip()
    assert main_ui.agent_previews.request_picture is not None
    assert main_ui.agent_chip.card.image_label is not None

    host.requests.requests = [dict(waiting, status="rejected")]
    main_ui._refresh_agent_chip()
    assert main_ui.agent_previews.request_picture is None


def test_the_card_is_narrow_until_something_waits(qapp):
    from PyQt5.QtWidgets import QMainWindow

    from fibsem.applications.autolamella.ui.agent_card import STACK_BELOW
    from fibsem.applications.autolamella.ui.agent_level_chip import (
        NARROW_WIDTH,
        WIDE_WIDTH,
    )

    window = QMainWindow()
    chip = AgentLevelChip()
    window.statusBar().addPermanentWidget(chip)
    window.show()
    chip.set_level(AgentLevel.WATCH)
    chip.toggle_card()
    qapp.processEvents()
    assert NARROW_WIDTH <= chip.popup.width() < STACK_BELOW and not chip.card.wide

    chip.set_request(_request())
    qapp.processEvents()
    assert chip.popup.width() == WIDE_WIDTH and chip.card.wide

    chip.set_request(None)
    chip.set_suggestion({"question": "Pick the point", "answer": "Here"})
    qapp.processEvents()
    assert chip.popup.width() == WIDE_WIDTH

    chip.set_suggestion(None)
    qapp.processEvents()
    assert NARROW_WIDTH <= chip.popup.width() < STACK_BELOW and not chip.card.wide
    window.close()


def test_the_card_floats_in_its_own_window_and_goes_back_on_the_chip(qapp):
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QMainWindow

    window = QMainWindow()
    chip = AgentLevelChip()
    window.statusBar().addPermanentWidget(chip)
    window.show()
    chip.set_level(AgentLevel.ASSIST)
    chip.toggle_card()
    assert chip.popup.isVisible()

    chip.card.float_button.click()
    assert chip.floating and not chip.popup.isVisible()
    panel = chip.panel
    assert chip.card.parent() is panel
    # Over the app's window only, not over other apps.
    assert panel.windowFlags() & Qt.Tool and panel.parent() is window
    assert chip.card.float_button.toolTip() == "Put the card back on the chip"

    # The card works the same floating: a request shows, Approve says so, and
    # the window stays open.
    approved = []
    chip.approve_clicked.connect(approved.append)
    chip.set_request(_request())
    chip.card.approve_button.click()
    assert approved == ["ab12"] and chip.floating

    # Narrow, it stacks.
    panel.resize(420, 600)
    qapp.processEvents()
    assert not chip.card.wide
    narrow = panel.width()  # no narrower than the stacked card's minimum
    assert narrow < 600

    # Closing the window puts the card back; the chip opens it there again.
    panel.move(40, 50)
    panel.close()
    assert not chip.floating and chip.card.parent() is chip.popup
    chip.toggle_card()
    assert chip.popup.isVisible()

    # Floated again, it is where it was left.
    chip.card.float_button.click()
    assert chip.floating and panel.width() == narrow
    chip.card.float_button.click()  # the button puts it back too
    assert not chip.floating and chip.card.parent() is chip.popup
    window.close()


def test_show_details_gives_every_row_its_height_in_the_floating_card(qapp):
    # FIB-1413: a mill's six-row table was squeezed into two lines, rows over
    # each other. Rows are one line each, full height; a tall table scrolls.
    from PyQt5.QtWidgets import QLabel, QMainWindow, QScrollArea

    rows = [
        [f"Rectangle {i} (upper trench)", "10.00 µm × 5.00 µm", "1.00 µm", "(0, 0)"]
        for i in range(6)
    ]
    request = _request()
    request["actions"] = [
        {
            "tool": "milling_run",
            "what": "Milling",
            "change": "Run 'agent test mill'",
            "details": {
                "columns": ["Pattern", "Size", "Depth", "Centre"],
                "rows": rows,
            },
        }
    ]
    window = QMainWindow()
    chip = AgentLevelChip()
    window.statusBar().addPermanentWidget(chip)
    window.show()
    chip.set_level(AgentLevel.ASSIST)
    chip.toggle_card()
    chip.card.float_button.click()
    chip.set_request(request)
    chip.panel.resize(420, 300)
    qapp.processEvents()
    chip.card.details_link.click()
    qapp.processEvents()

    (table,) = chip.card.findChildren(QScrollArea, "agentDetails")
    cells = [
        label
        for label in table.widget().findChildren(QLabel)
        if label.text().startswith("Rectangle")
    ]
    assert len(cells) == 6
    tops = [cell.mapTo(table.widget(), cell.rect().topLeft()).y() for cell in cells]
    for cell, top, below in zip(cells, tops, tops[1:]):
        assert below >= top + cell.sizeHint().height()  # no row over the next
        assert not cell.wordWrap()
    # Six rows and the header fit without scrolling.
    assert table.viewport().height() >= table.widget().sizeHint().height()
    window.close()


def test_a_tall_details_table_scrolls_inside_the_card(qapp):
    from PyQt5.QtWidgets import QScrollArea

    card = AgentCard()
    request = _request()
    request["actions"][0]["details"]["rows"] = [
        [f"Setting {i}", "1", "2"] for i in range(30)
    ]
    card.set_request(request)
    card.details_link.click()
    (table,) = card.findChildren(QScrollArea, "agentDetails")
    assert table.height() < table.widget().sizeHint().height()
    card.deleteLater()


# --- the level as a menu, keep on top (FIB-1418) ---------------------------------


def test_the_level_is_drawn_as_the_chip_and_opens_a_menu_of_the_four(qapp):
    from PyQt5.QtWidgets import QLabel

    from fibsem.applications.autolamella.ui.agent_style import LEVEL_STYLES

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


def test_the_floating_card_can_be_kept_on_top_of_every_window(qapp):
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QMainWindow

    window = QMainWindow()
    chip = AgentLevelChip()
    window.statusBar().addPermanentWidget(chip)
    window.show()
    pin = chip.card.pin_button
    assert not pin.isVisibleTo(chip.card)  # on the chip, there is nothing to pin
    chip.float_card()
    panel = chip.panel
    assert pin.isVisibleTo(chip.card)
    assert not panel.windowFlags() & Qt.WindowStaysOnTopHint  # off at first
    pin.click()
    assert chip.keep_on_top and panel.windowFlags() & Qt.WindowStaysOnTopHint
    # A plain window: a tool window (an NSPanel on macOS) floats over other
    # apps only while the app is active (FIB-1427).
    assert panel.windowType() == Qt.Window
    assert panel.isVisible()
    # Kept for the session: put back and floated again, it is still on top.
    chip.dock_card()
    chip.float_card()
    assert panel.windowFlags() & Qt.WindowStaysOnTopHint and pin.isChecked()
    pin.click()
    assert not panel.windowFlags() & Qt.WindowStaysOnTopHint
    assert panel.windowType() == Qt.Tool and panel.parent() is window
    window.close()


# --- Recent (FIB-1414), and the clamped Why (FIB-1412) ---------------------------


def _row(text, outcome="done", tone="ok", level=""):
    from fibsem.applications.autolamella.ui.agent_recent import RecentRow

    return RecentRow("16:08", text, outcome, tone, level)


def _called(action, status=200, level="drive"):
    """The server's record of the agent's direct call (FIB-1426)."""
    return {"action": action, "status": status, "ok": status < 400, "level": level}


def test_recent_shows_at_drive_and_is_hidden_below_it_until_asked_for(qapp):
    card = AgentCard()
    logs = []
    card.event_log_clicked.connect(lambda: logs.append(True))
    card.set_recent([_row("Moved the stage: move relative")])
    card.set_level(AgentLevel.ASSIST)
    assert not card.recent_buttons
    # Folded: its newest line beside the link.
    assert card.recent_link.text() == "Recent ▸"
    assert card.recent_last.text() == "last: Moved the stage: move relative, done"
    card.set_level(AgentLevel.DRIVE)
    assert card.recent_link.text() == "Recent ▾"
    assert [b.text() for b in card.recent_buttons] == ["Moved the stage: move relative"]
    assert "16:08" in _labels(card)
    assert "done" in _labels(card)
    card.recent_buttons[0].click()
    assert logs == [True]
    # The operator's choice holds at that level for the session.
    card.recent_link.click()
    assert not card.recent_buttons
    card.set_level(AgentLevel.ASSIST)
    card.recent_link.click()
    assert card.recent_buttons
    card.set_level(AgentLevel.DRIVE)
    assert not card.recent_buttons
    card.deleteLater()


def test_a_long_why_is_clamped_with_more(qapp):
    card = AgentCard()
    reason = "The trench floor is high on 01 in the 16:02 FIB image. " * 8
    card.set_request({**_request(), "reason": reason})
    line = card.why_label.fontMetrics().lineSpacing()
    assert card.why_link is not None and card.why_link.text() == "more"
    assert card.why_label.maximumHeight() == 3 * line
    card.why_link.click()
    assert card.why_link.text() == "less"
    assert card.why_label.maximumHeight() > 3 * line
    card.set_request(_request())  # a short one: no clamp
    assert card.why_link is None
    card.deleteLater()


def test_at_drive_what_the_agent_does_shows_in_the_card_now_then_recent(qapp):
    """FIB-1414 done when: at Drive a move, a mill and an answer each show in
    the card, Now while running, then in Recent with how it ended."""
    from types import SimpleNamespace

    from fibsem.acting import AGENT, acting
    from fibsem.applications.autolamella.server.events import EventBuffer
    from fibsem.applications.autolamella.ui.agent_activity import AgentActivity
    from fibsem.control import Control
    from fibsem.ui.widgets.status_bar import FibsemStatusBar

    bar = FibsemStatusBar()
    activity = AgentActivity(bar)
    control = Control()
    activity.set_microscope(SimpleNamespace(control=control))
    chip = AgentLevelChip()
    activity.now_changed.connect(chip.set_now)
    activity.recent_changed.connect(chip.set_recent)
    chip.set_level(AgentLevel.DRIVE)
    buffer = EventBuffer(stamp=lambda kind, payload: {"actor": AGENT})
    activity.follow_events(buffer)
    with acting(AGENT), control.take(AGENT, "move stage relative"):
        assert "move stage relative" in _labels(chip.card)
        buffer.append("agent_called", _called("Stage · Move by x 2 µm"))
    buffer.append(
        "milling", {"milling_task_name": "agent test mill", "status": "completed"}
    )
    buffer.append(
        "prompt_answered", {"question_id": "q", "response": True, "answered_by": AGENT}
    )
    qapp.processEvents()
    texts = [b.text() for b in chip.card.recent_buttons]
    assert texts == [
        "Answered 'the run's question': yes",
        "Milled 'agent test mill'",
        "Stage · Move by x 2 µm",
    ]
    assert "move stage relative" not in _labels(chip.card)
    activity.follow_events(None)
    activity.set_microscope(None)
    chip.deleteLater()
    bar.deleteLater()


def test_at_drive_each_direct_call_shows_in_now_then_recent_with_its_level(qapp):
    """FIB-1426 done when: at Drive a move, a mill, a stop, an image and an
    overview each show in Now while running, then in Recent with a plain line,
    how it ended and its level."""
    from types import SimpleNamespace

    from fibsem.acting import AGENT, acting
    from fibsem.applications.autolamella.server.events import EventBuffer
    from fibsem.applications.autolamella.ui.agent_activity import AgentActivity
    from fibsem.control import Control
    from fibsem.ui.widgets.status_bar import FibsemStatusBar

    bar = FibsemStatusBar()
    activity = AgentActivity(bar)
    control = Control()
    activity.set_microscope(SimpleNamespace(control=control))
    chip = AgentLevelChip()
    activity.now_changed.connect(chip.set_now)
    activity.recent_changed.connect(chip.set_recent)
    chip.set_level(AgentLevel.DRIVE)
    buffer = EventBuffer(stamp=lambda kind, payload: {"actor": AGENT})
    activity.follow_events(buffer)

    def now():
        qapp.processEvents()
        return (
            chip.card.now_label.text()
            if chip.card.now_row.isVisibleTo(chip.card)
            else None
        )

    move, image = "Stage · Move by x 2 µm", "FIB · Take an image, 150 µm wide"
    overview = "FIB · Overview, 3 × 3 tiles"
    with acting(AGENT):
        with control.take(AGENT, move):
            assert now() == move
        buffer.append("agent_called", _called(move))
        with control.take(AGENT, image):
            assert now() == image
        buffer.append("agent_called", _called(image))
        run = {"service": "imaging", "run_id": "o", "action": overview}
        with control.take(AGENT, overview):
            buffer.append("command_started", dict(run, level="drive"))
            assert now() == overview
            assert chip.card.recent_buttons[0].text() == overview
            assert "running" in _labels(chip.card)
        buffer.append("command_ended", dict(run, state="done", level="drive"))
        with control.take(AGENT, "Milling · Run task 'trench'"):
            assert now() == "Milling · Run task 'trench'"
        buffer.append("milling", {"milling_task_name": "trench", "status": "cancelled"})
        buffer.append(
            "agent_stop",
            {"what": "the mill", "reason": "through the lamella", "level": "drive"},
        )
    assert now() is None
    rows = [(r.text, r.outcome, r.level) for r in activity.recent]
    assert rows == [
        ("Stopped the mill: through the lamella", "done", "drive"),
        ("Milled 'trench'", "stopped", "drive"),
        (overview, "done", "drive"),
        (image, "done", "drive"),
        (move, "done", "drive"),
    ]
    assert _labels(chip.card).count("Drive") == 5
    activity.follow_events(None)
    activity.set_microscope(None)
    chip.deleteLater()
    bar.deleteLater()


def test_a_row_s_outcome_changes_on_the_card(qapp):
    # Round 8 screenshots: an overview read "running" after it ended, since
    # the row was changed in place and the card saw the same rows.
    from fibsem.applications.autolamella.ui.agent_recent import RecentActions

    recent = RecentActions()
    run = {"service": "imaging", "run_id": "o", "action": "Overview", "level": "drive"}
    started = {"seq": 1, "kind": "command_started", "actor": "agent", "payload": run}
    recent.add(started)
    card = AgentCard()
    card.set_level(AgentLevel.DRIVE)
    card.set_recent(recent.rows)
    assert "running" in _labels(card)
    ended = dict(started, seq=2, kind="command_ended", payload=dict(run, state="done"))
    recent.add(ended)
    card.set_recent(recent.rows)
    assert "done" in _labels(card) and "running" not in _labels(card)
    card.deleteLater()
