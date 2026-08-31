"""What a fire-and-forget status update does to the interaction UI, pinned pre-move.

Three of ``workflow_update_signal``'s emit families are fire-and-forget status:
``update_status_ui`` (message / workflow-info / status-bar text), the task manager's
lifecycle reports, and ``ask_user``'s trailing prompt-clear. They are about to move to
their own signal — continuing the split ``queue_changed_signal`` started — so this
pins what the display half does today, before the channel moves under it.

The payloads here are built by hand to match the emit sites (``update_status_ui`` at
``workflows/ui.py`` and ``TaskManager._emit_status``), and the handler is called
directly rather than through the signal: an exception escaping a queued slot is a
process abort (FIB-329), which would take the pytest summary with it.
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


def _status_payload(msg, workflow_info=None, status_bar=None):
    """The dict ``update_status_ui`` puts on the wire, byte for byte."""
    return {"msg": msg, "workflow_info": workflow_info, "status_bar": status_bar}


def test_a_status_message_writes_the_instruction_label(ui):
    ui.handle_workflow_update(_status_payload("Milling: Preparing..."))

    assert ui.label_instructions.text() == "Milling: Preparing..."
    assert ui.label_instructions.isVisibleTo(ui)


def test_a_status_message_carries_no_question_so_the_buttons_hide(ui):
    # A question showed its buttons first; a plain status update must take them
    # down — status payloads carry no pos/neg, and absent means "no question".
    ui.set_instructions_msg("Continue?", pos="Continue", neg="Exit")
    assert ui.pushButton_yes.isVisibleTo(ui)

    ui.handle_workflow_update(_status_payload("Milling: Preparing..."))

    assert not ui.pushButton_yes.isVisibleTo(ui)
    assert not ui.pushButton_no.isVisibleTo(ui)


def test_an_empty_message_clears_the_prompt(ui):
    # ask_user's trailing emit: {"msg": ""} takes the answered question down.
    ui.set_instructions_msg("Continue?", pos="Continue", neg="Exit")

    ui.handle_workflow_update({"msg": ""})

    assert ui.label_instructions.text() == ""
    assert not ui.label_instructions.isVisibleTo(ui)


def test_workflow_info_writes_the_workflow_information_label(ui):
    ui.handle_workflow_update(
        _status_payload("", workflow_info="Task 2/5: Rough Milling")
    )

    assert ui.label_workflow_information.text() == "Task 2/5: Rough Milling"
    assert ui.label_workflow_information.isVisibleTo(ui)


def test_absent_workflow_info_shows_the_label_without_rewriting_it(ui):
    # Today's behaviour, deliberate or not: set_current_workflow_message(None)
    # leaves the text alone but still makes the label visible. The channel move
    # must not silently change it either way.
    ui.set_current_workflow_message("Task 2/5: Rough Milling", show=False)
    assert not ui.label_workflow_information.isVisibleTo(ui)

    ui.handle_workflow_update(_status_payload("Moving stage..."))

    assert ui.label_workflow_information.text() == "Task 2/5: Rough Milling"
    assert ui.label_workflow_information.isVisibleTo(ui)


def test_a_status_update_releases_a_pending_ui_update_wait(ui):
    # The defect the move removes, pinned as it stands: the flag is scoped to the
    # signal, not to a request, so a mere status emit releases whoever is blocked
    # in a `while WAITING_FOR_UI_UPDATE` loop. Once status moves off this signal,
    # this inverts: a status update must leave a pending wait alone.
    ui.WAITING_FOR_UI_UPDATE = True

    ui.handle_workflow_update(_status_payload("Moving stage..."))

    assert ui.WAITING_FOR_UI_UPDATE is False


# ── the new channel ──────────────────────────────────────────────────────────────
# workflow_status_signal carries the same display behaviour as the dict payloads
# above (which update_status_ui still emits, until it converts too), minus the one
# defect: an emission can no longer release a blocked waiter.


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


def test_a_status_event_clears_the_prompt_when_it_says_nothing(ui):
    ui.set_instructions_msg("Continue?", pos="Continue", neg="Exit")

    ui.workflow_status_signal.emit(_status_event(""))

    assert not ui.label_instructions.isVisibleTo(ui)
    assert not ui.pushButton_yes.isVisibleTo(ui)


def test_a_status_event_writes_the_workflow_information_label(ui):
    ui.workflow_status_signal.emit(
        _status_event("", workflow_info="Task 2/5: Rough Milling")
    )

    assert ui.label_workflow_information.text() == "Task 2/5: Rough Milling"


def test_a_status_event_leaves_a_pending_wait_alone(ui):
    # The inversion of test_a_status_update_releases_a_pending_ui_update_wait, and
    # the reason the channel exists: a Future-style handshake (and until then, the
    # flag) belongs to one request, and merely saying something must not complete it.
    ui.WAITING_FOR_UI_UPDATE = True
    ui.WAITING_FOR_USER_INTERACTION = True

    ui.workflow_status_signal.emit(_status_event("Moving stage..."))

    assert ui.WAITING_FOR_UI_UPDATE is True
    assert ui.WAITING_FOR_USER_INTERACTION is True
    ui.WAITING_FOR_UI_UPDATE = False
    ui.WAITING_FOR_USER_INTERACTION = False
