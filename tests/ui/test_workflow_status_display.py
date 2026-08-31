"""What a fire-and-forget status update does to the interaction UI.

Everything the workflow says without needing an answer travels as a
``WorkflowStatusEvent`` on ``workflow_status_signal`` — the dict
``workflow_update_signal`` is gone. Events are emitted through the signal: a
direct (same-thread) connection runs the handler synchronously and propagates
exceptions to the emitter, so each test also pins that the signal is connected.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI


@pytest.fixture
def ui(qapp):
    """A real AutoLamellaUI, connected, without the main window that hosts it.

    Connected because the handler needs ``image_widget`` and
    ``milling_task_config_widget``, which ``update_microscope_ui()`` builds. The
    shipped default configuration is Demo, so this touches no hardware.
    """
    widget = AutoLamellaUI(parent_ui=None)
    widget.system_widget.connect_to_microscope()
    yield widget
    if widget.microscope is not None:
        widget.microscope.disconnect()
    widget.close()


def _status_event(msg="", workflow_info=None):
    from fibsem.applications.autolamella.workflows.tasks.status import (
        WorkflowStatusEvent,
    )

    return WorkflowStatusEvent(message=msg, workflow_info=workflow_info)


def test_a_status_event_writes_the_instruction_label(ui):
    # Through the signal, not the slot: a direct (same-thread) connection runs the
    # handler synchronously and propagates exceptions to the emitter, so this also
    # pins that the signal is actually connected.
    ui.workflow_status_signal.emit(_status_event("Milling: Preparing..."))

    assert ui.label_instructions.text() == "Milling: Preparing..."
    assert not ui.pushButton_yes.isVisibleTo(ui)


def test_an_empty_message_clears_the_prompt(ui):
    # "" is the explicit clear — how a prompt comes down when a question ends.
    ui.set_instructions_msg("Continue?", pos="Continue", neg="Exit")

    ui.workflow_status_signal.emit(_status_event(""))

    assert not ui.label_instructions.isVisibleTo(ui)
    assert not ui.pushButton_yes.isVisibleTo(ui)


def test_a_status_event_writes_the_workflow_information_label(ui):
    ui.workflow_status_signal.emit(
        _status_event("", workflow_info="Task 2/5: Rough Milling")
    )

    assert ui.label_workflow_information.text() == "Task 2/5: Rough Milling"


def test_a_status_event_leaves_a_pending_question_alone(ui):
    # The reason the channel exists: an answer belongs to one request, and merely
    # saying something must not complete it. The polled flag is gone entirely;
    # the display state a parked question sets must survive a status emit.
    ui.WAITING_FOR_USER_INTERACTION = True

    ui.workflow_status_signal.emit(_status_event("Moving stage..."))

    assert ui.WAITING_FOR_USER_INTERACTION is True
    assert not hasattr(ui, "WAITING_FOR_UI_UPDATE")
    ui.WAITING_FOR_USER_INTERACTION = False


def test_a_message_of_none_leaves_the_prompt_standing(ui):
    # None says nothing about the prompt. The responder pings this signal for
    # chrome refreshes while its question is up — if the default took the
    # prompt down, every question would erase itself as it appeared.
    ui.set_instructions_msg("Continue?", pos="Continue", neg="Exit")

    ui.workflow_status_signal.emit(_status_event(msg=None))

    assert ui.label_instructions.text() == "Continue?"
    assert ui.pushButton_yes.isVisibleTo(ui)


def test_absent_workflow_info_shows_the_label_without_rewriting_it(ui):
    # Pinned behaviour, deliberate or not: set_current_workflow_message(None)
    # leaves the text alone but still makes the label visible.
    ui.set_current_workflow_message("Task 2/5: Rough Milling", show=False)
    assert not ui.label_workflow_information.isVisibleTo(ui)

    ui.workflow_status_signal.emit(_status_event("Moving stage..."))

    assert ui.label_workflow_information.text() == "Task 2/5: Rough Milling"
    assert ui.label_workflow_information.isVisibleTo(ui)
