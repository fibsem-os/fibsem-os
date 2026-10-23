"""What a fire-and-forget status update does to the interaction UI.

Everything the workflow says without needing an answer travels as a
``WorkflowStatusEvent`` on ``workflow_status_signal`` — the dict
``workflow_update_signal`` is gone. Events are emitted through the signal: a
direct (same-thread) connection runs the handler synchronously and propagates
exceptions to the emitter, so each test also pins that the signal is connected.
"""

import os

from tests.fixtures.main_window import main_window_fixture

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.workflows.status import Hold, HoldKind

ui = main_window_fixture(connect=True, name="ui")


def _status_event(msg="", workflow_info=None):
    from fibsem.workflows.status import (
        WorkflowStatusEvent,
    )

    return WorkflowStatusEvent(message=msg, workflow_info=workflow_info)


def test_a_status_event_writes_the_instruction_label(ui):
    # Through the signal, not the slot: a direct (same-thread) connection runs the
    # handler synchronously and propagates exceptions to the emitter, so this also
    # pins that the signal is actually connected.
    ui.run_relay.status.emit(_status_event("Milling: Preparing..."))

    assert ui.panel.prompt_bar.message.text() == "Milling: Preparing..."
    assert not ui.panel.prompt_bar.yes.isVisibleTo(ui)


def test_an_empty_message_clears_the_prompt(ui):
    # "" is the explicit clear — how a prompt comes down when a question ends.
    ui.panel.prompt_bar.ask("Continue?", pos="Continue", neg="Exit")

    ui.run_relay.status.emit(_status_event(""))

    assert not ui.panel.prompt_bar.message.isVisibleTo(ui)
    assert not ui.panel.prompt_bar.yes.isVisibleTo(ui)


def test_a_status_event_writes_the_workflow_information_label(ui):
    ui.run_relay.status.emit(_status_event("", workflow_info="Task 2/5: Rough Milling"))

    assert ui.panel.label_workflow_information.text() == "Task 2/5: Rough Milling"


def test_a_status_event_leaves_a_held_proposal_alone(ui):
    # Merely saying something must not release a hold. The polled flag is gone
    # entirely; the display state a held proposal sets must survive a status emit.
    ui.run_relay.hold = Hold(
        HoldKind.proposal, "decide the proposal on the Microscope tab"
    )

    ui.run_relay.status.emit(_status_event("Moving stage..."))

    assert ui.run_relay.hold is not None and ui.run_relay.hold.kind is HoldKind.proposal
    assert not hasattr(ui, "WAITING_FOR_UI_UPDATE")
    ui.run_relay.hold = None


def test_a_message_of_none_leaves_the_prompt_standing(ui):
    # None says nothing about the prompt bar. The window pings this signal for
    # chrome refreshes while a proposal is up — if the default took the bar
    # down, every proposal would erase itself as it appeared.
    ui.panel.prompt_bar.ask("Continue?", pos="Continue", neg="Exit")

    ui.run_relay.status.emit(_status_event(msg=None))

    assert ui.panel.prompt_bar.message.text() == "Continue?"
    assert ui.panel.prompt_bar.yes.isVisibleTo(ui)


def test_absent_workflow_info_shows_the_label_without_rewriting_it(ui):
    # Pinned behaviour, deliberate or not: set_current_workflow_message(None)
    # leaves the text alone but still makes the label visible.
    ui.panel.set_current_workflow_message("Task 2/5: Rough Milling", show=False)
    assert not ui.panel.label_workflow_information.isVisibleTo(ui)

    ui.run_relay.status.emit(_status_event("Moving stage..."))

    assert ui.panel.label_workflow_information.text() == "Task 2/5: Rough Milling"
    assert ui.panel.label_workflow_information.isVisibleTo(ui)
