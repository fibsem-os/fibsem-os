"""What `AutoLamellaUI.handle_workflow_update` must survive being handed.

`workflow_update_signal` has no declared contract. It has 13 emit sites and 12 of
them pass an opaque variable, so what a payload carries cannot be established by
reading the code -- only by running it.

That would be a cosmetic problem in most slots. Here it is not: PyQt5 calls Qt's
`qFatal()` whenever a Python exception escapes a slot invoked from C++, which a
queued signal from the workflow thread is. So a missing key is not a blank label,
it is `SIGABRT` -- the run dies mid-workflow and nothing reaches the logfile
(FIB-329). Measured on this handler while writing these tests: emitting a payload
without `msg` through the signal exits **134**, with no traceback in the log.

It has happened. A queue edit put a payload without `msg` on this signal and killed
the app on every queue action; see `test_a_queue_edit_never_reaches_the_workflow_update_slot`.
That was fixed at the emitting end, one emitter at a time, and pinned by a test
that every payload carries `msg`. This is the other end, which is the end that
holds for emitters nobody has written yet.

These tests call the handler directly rather than emitting. A test that went
through the signal could not *fail* -- it would abort the pytest process, taking
the summary with it and naming no test.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI


@pytest.fixture
def ui(qapp):
    """A real AutoLamellaUI, connected, without the main window that hosts it.

    Connected because the handler needs `image_widget` and
    `milling_task_config_widget`, which `update_microscope_ui()` builds. The
    shipped default configuration is Demo, so this touches no hardware.
    """
    widget = AutoLamellaUI(parent_ui=None)
    widget.system_widget.connect_to_microscope()
    yield widget
    if widget.microscope is not None:
        widget.microscope.disconnect()
    widget.close()


def test_a_payload_without_a_message_does_not_raise(ui):
    """The regression. Indexing `msg` here aborts the process, not the label."""
    ui.handle_workflow_update({"status": {}})


def test_a_payload_without_a_message_clears_the_prompt(ui):
    """Not merely "does not raise": an update that says nothing about the prompt
    must not leave the previous question standing.

    `set_instructions_msg("")` hides the label rather than showing a blank line,
    which is what one emit site sends `{"msg": ""}` deliberately to do -- so the
    default and the explicit clear take the same path.
    """
    ui.set_instructions_msg("Continue?", "Yes", "No")

    ui.handle_workflow_update({"status": {}})

    assert ui.label_instructions.text() == ""
    assert not ui.label_instructions.isVisible()


def test_a_message_is_still_shown_when_one_is_sent(ui):
    """The default must not have swallowed the normal path."""
    ui.handle_workflow_update({"msg": "Select a lamella position", "pos": "Continue"})

    assert ui.label_instructions.text() == "Select a lamella position"
    assert ui.pushButton_yes.text() == "Continue"


def test_the_handler_releases_the_workflow_thread(ui):
    """Every emitter sets `WAITING_FOR_UI_UPDATE` and then blocks on it in a
    `while ... time.sleep(0.5)` loop, so the flag is the workflow thread's only way
    out of that wait. It is cleared on the last line of the handler -- after the
    message -- so anything that raises on the way there strands the caller as well
    as killing the process."""
    ui.WAITING_FOR_UI_UPDATE = True

    ui.handle_workflow_update({"status": {}})

    assert ui.WAITING_FOR_UI_UPDATE is False
