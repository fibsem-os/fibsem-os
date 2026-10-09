"""The shared status bar's one line (FIB-1188): an operation, else the run, else the
instruction -- and the real windows feed it.

`showMessage` gave whatever spoke last the whole bar: a queue confirmation replaced
the run's own line until the next report, and a second widget for the stage words
made two messages at once. The bar holds each source apart and shows the one that
matters now.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.ui.widgets.status_bar import FibsemStatusBar  # noqa: E402

INSTRUCTION = "Create or load an experiment to begin."


@pytest.fixture
def slot(qapp):
    slot = FibsemStatusBar()
    slot.set_instruction(INSTRUCTION)
    yield slot
    slot.deleteLater()


def test_with_nothing_running_it_is_the_instruction(slot):
    assert slot.showing == "instruction"
    assert slot.text == INSTRUCTION


def test_an_operation_covers_the_instruction_and_gives_it_back(slot):
    slot.set_stage_activity("Moving to the SEM orientation…")
    assert slot.text == "Stage move moving to the SEM orientation…"
    slot.set_stage_activity(None)
    assert slot.text == INSTRUCTION


def test_a_run_shows_its_task_and_where_it_is(slot):
    slot.set_run("lamella-02 › Polishing", None, "3 of 4")
    assert slot.text == "lamella-02 › Polishing · 3 of 4"
    slot.set_run(None)
    assert slot.text == INSTRUCTION


def test_an_operation_outranks_the_run_and_the_run_comes_back(slot):
    slot.set_run("lamella-02 › Polishing", None, "3 of 4")
    slot.set_operation("Stage move", "acquiring images…")
    assert slot.showing == "operation"
    slot.set_operation(None)
    assert slot.text == "lamella-02 › Polishing · 3 of 4"


def test_a_run_step_keeps_its_task(slot):
    slot.set_run("lamella-02 › Polishing", None, "3 of 4")
    slot.set_run_step("Waiting on 1 decision: review.")
    assert slot.text == "lamella-02 › Polishing Waiting on 1 decision: review. · 3 of 4"
    slot.set_run_step("")
    assert slot.text == "lamella-02 › Polishing · 3 of 4"


def test_a_step_before_the_first_task_is_the_workflows(slot):
    slot.set_run_step("Waiting until 14:00 to start Polishing on lamella-02.")
    assert slot.text.startswith("Workflow Waiting until 14:00")


def test_the_host_places_its_actions_on_the_right(slot):
    from PyQt5.QtWidgets import QPushButton

    button = QPushButton("Run Workflow")
    slot.add_action(button)
    assert button.parent() is slot
    assert slot.text == INSTRUCTION, "the line is unaffected"


def test_a_new_instruction_waits_behind_a_running_line(slot):
    slot.set_run("lamella-02 › Polishing")
    slot.set_instruction("Add a lamella to begin.")
    assert slot.showing == "run"
    slot.set_run(None)
    assert slot.text == "Add a lamella to begin."


# --- the real windows ----------------------------------------------------------


@pytest.fixture
def no_quit(qapp):
    # AutoLamella's closeEvent ends in app.quit(), which latches on the shared test
    # QApplication (see test_mainui_workflow_status.py).
    original_quit = qapp.quit
    qapp.quit = lambda: None
    try:
        yield
    finally:
        qapp.quit = original_quit


def _close(window, qapp):
    window.close()
    window.deleteLater()
    qapp.processEvents()


def test_autolamella_says_the_stage_move_on_the_left(no_quit, qapp):
    from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module

    window = module.AutoLamellaSingleWindowUI()
    slot = window.status_bar
    assert window.statusBar() is slot, "the window's own bar, not a second one"
    assert slot.showing == "instruction", "the setup ladder is on the line"
    window.view_controller.report_activity("Moving the stage…")
    assert slot.text == "Stage move moving the stage…"
    window.view_controller.report_activity(None)
    assert slot.showing == "instruction"
    assert window.status_bar.currentMessage() == "", "nothing covers the line"
    _close(window, qapp)


def test_a_queue_confirmation_is_a_toast_not_the_line(no_quit, qapp, monkeypatch):
    from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module
    from fibsem.ui import notification_service

    shown = []
    monkeypatch.setattr(
        notification_service, "show_toast", lambda msg, *a, **k: shown.append(msg)
    )
    window = module.AutoLamellaSingleWindowUI()
    window.status_bar.set_run("lamella-02 › Polishing", None, "3 of 4")
    window._show_queue_message("Moved lamella-03 up.")
    assert shown == ["Moved lamella-03 up."]
    assert window.status_bar.text == "lamella-02 › Polishing · 3 of 4"
    _close(window, qapp)


def test_fibsem_says_the_stage_move_too(qapp):
    from fibsem.ui.FibsemUI import FibsemUI

    window = FibsemUI()
    window.view_controller.report_activity("Moving the stage vertically…")
    assert window.statusBar() is window.status_bar
    assert window.status_bar.text == "Stage move moving the stage vertically…"
    window.view_controller.report_activity(None)
    assert window.status_bar.text == ""
    _close(window, qapp)
