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


@pytest.fixture(autouse=True)
def quick_outcomes(monkeypatch):
    """An outcome stays 2 s in the app; 20 ms here, so a test can watch it go."""
    from fibsem.ui.widgets import status_bar

    monkeypatch.setattr(status_bar, "OUTCOME_MS", 20)


def _outlast_the_outcome():
    from PyQt5.QtCore import QEventLoop, QTimer

    loop = QEventLoop()
    QTimer.singleShot(80, loop.quit)
    loop.exec_()


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


# --- milling progress ------------------------------------------------------------


def _stage_started(stage=1, total=3, name="Rough Mill"):
    from fibsem.milling.progress import MillingProgress, MillingProgressStatus

    return MillingProgress(
        MillingProgressStatus.STAGE_STARTED,
        stage_name=name,
        current_stage=stage,
        total_stages=total,
    )


def _strategy_says(message="Running Rough Mill..."):
    """What `strategy/standard.py` sends once a stage is under way: its words, no
    figures."""
    from fibsem.milling.progress import MillingProgress, MillingProgressStatus

    return MillingProgress(
        MillingProgressStatus.STAGE_UPDATE, message=message, stage_name="Rough Mill"
    )


def _tick(remaining=30.0, estimated=60.0):
    """A backend's tick: figures, no words."""
    from fibsem.milling.progress import MillingProgress, MillingProgressStatus

    return MillingProgress(
        MillingProgressStatus.STAGE_UPDATE,
        remaining_time=remaining,
        estimated_time=estimated,
    )


def _finished():
    from fibsem.milling.progress import MillingProgress, MillingProgressStatus

    return MillingProgress(MillingProgressStatus.TASK_FINISHED)


def test_milling_shows_its_stage_then_its_countdown(slot):
    slot._on_milling_progress(_stage_started())
    assert slot.text == "Preparing: Rough Mill stage 2 of 3"
    assert slot.fraction is None, "no bar until there is a figure to draw"
    slot._on_milling_progress(_strategy_says())
    slot._on_milling_progress(_tick())
    assert slot.fraction == pytest.approx(0.5)
    assert slot.text == "Running Rough Mill... stage 2 of 3 50% · 30s left", (
        "the strategy's words, kept over the backend's wordless tick"
    )


def test_milling_ending_says_so_then_gives_the_line_back(slot):
    slot._on_milling_progress(_stage_started())
    slot._on_milling_progress(_tick())
    slot._on_milling_progress(_finished())
    assert slot.text == "Milling done"
    assert slot.fraction is None
    _outlast_the_outcome()
    assert slot.text == INSTRUCTION


def test_a_failed_mill_is_red_and_says_why(slot):
    from fibsem.milling.progress import MillingProgress, MillingProgressStatus

    slot._on_milling_progress(
        MillingProgress(MillingProgressStatus.TASK_FAILED, error="the column tripped")
    )
    assert slot.text == "Milling failed: the column tripped"
    assert "#d04040" in slot._step.styleSheet()
    _outlast_the_outcome()
    assert "#d04040" not in slot._step.styleSheet()


def test_a_cancelled_mill_is_not_red(slot):
    from fibsem.milling.progress import MillingProgress, MillingProgressStatus

    slot._on_milling_progress(MillingProgress(MillingProgressStatus.TASK_CANCELLED))
    assert slot.text == "Milling cancelled"
    assert "#d04040" not in slot._step.styleSheet()


def test_new_progress_is_not_cut_short_by_an_old_outcome(slot):
    slot._on_milling_progress(_finished())
    slot._on_milling_progress(_stage_started())
    _outlast_the_outcome()
    assert slot.text == "Preparing: Rough Mill stage 2 of 3"


def test_milling_in_a_run_sits_under_the_task(slot):
    slot.set_run("lamella-02 › Rough Milling", None, "3 of 5")
    slot._on_milling_progress(_stage_started())
    slot._on_milling_progress(_strategy_says())
    slot._on_milling_progress(_tick())
    assert slot.text == (
        "lamella-02 › Rough Milling Running Rough Mill... · stage 2 of 3"
        " 50% · 30s left · 3 of 5"
    )
    slot._on_milling_progress(_finished())
    assert slot.text == "lamella-02 › Rough Milling Milling · done · 3 of 5"
    _outlast_the_outcome()
    assert slot.text == "lamella-02 › Rough Milling · 3 of 5"


def test_a_stage_move_outranks_milling_and_hides_its_bar(slot):
    slot._on_milling_progress(_tick())
    slot.set_stage_activity("Moving the stage…")
    assert slot.fraction is None
    slot.set_stage_activity(None)
    assert slot.fraction == pytest.approx(0.5)


# --- spot burn -------------------------------------------------------------------


def test_a_spot_burn_counts_its_spots_and_its_time(slot):
    from fibsem.imaging.spot import SpotBurnProgress, SpotBurnStatus

    slot._on_spot_burn_progress(
        SpotBurnProgress(
            status=SpotBurnStatus.BURNING,
            current_point=2,
            total_points=5,
            total_remaining_time=30.0,
            total_estimated_time=50.0,
        )
    )
    assert slot.text == "Spot burn spot 2 of 5 40% · 30s left"
    assert slot.fraction == pytest.approx(0.4)


def test_a_spot_burn_that_fails_says_why_in_red(slot):
    from fibsem.imaging.spot import SpotBurnProgress, SpotBurnStatus

    slot._on_spot_burn_progress(
        SpotBurnProgress(status=SpotBurnStatus.FAILED, error="beam blanked")
    )
    assert slot.text == "Spot burn failed: beam blanked"
    assert "#d04040" in slot._step.styleSheet()


# --- a run's failures, and a run that waits -------------------------------------


def test_a_failure_stays_with_a_way_there_and_dismiss(slot):
    slot.show_failure(
        "Run finished", "1 of 4 failed · 02-civil-cub › Polishing: no focus peak"
    )
    assert slot.showing == "failure"
    assert (
        slot.text
        == "Run finished 1 of 4 failed · 02-civil-cub › Polishing: no focus peak"
    )
    assert "#d04040" in slot._step.styleSheet()
    assert not slot._details_btn.isHidden() and not slot._dismiss_btn.isHidden()
    _outlast_the_outcome()
    assert slot.showing == "failure", "a failure is not an outcome: it stays"
    slot._dismiss_btn.click()
    assert slot.text == INSTRUCTION
    assert slot._details_btn.isHidden() and slot._dismiss_btn.isHidden()


def test_show_in_workflow_asks_the_host(slot):
    """Where the failures are is the host's to say: the bar only asks."""
    asked = []
    slot.failure_details_requested.connect(lambda: asked.append(True))
    slot.show_failure("Run finished", "2 of 4 failed")
    assert slot._details_btn.text() == "Show"
    slot._details_btn.click()
    assert asked == [True]


def test_a_stage_move_passes_over_a_failure_and_gives_it_back(slot):
    slot.show_failure("Run finished", "1 of 4 failed")
    slot.set_stage_activity("Moving the stage…")
    assert slot.showing == "operation"
    assert slot._dismiss_btn.isHidden()
    slot.set_stage_activity(None)
    assert slot.showing == "failure"


def test_a_held_run_says_what_releases_it_and_counts_the_wait(slot, monkeypatch):
    from fibsem.ui.widgets import status_bar

    now = [1000.0]
    monkeypatch.setattr(status_bar.time, "monotonic", lambda: now[0])
    slot.set_run("01-fancy-mite › Rough Milling", None, "3 of 5")
    slot.set_waiting("review the milling patterns")
    assert slot.text == (
        "01-fancy-mite › Rough Milling review the milling patterns · waiting 0:00 · 3 of 5"
    )
    now[0] += 134
    slot.set_waiting("review the milling patterns")  # the same hold keeps its clock
    assert "waiting 2:14" in slot.text
    slot.set_waiting(None)
    assert slot.text == "01-fancy-mite › Rough Milling · 3 of 5"


def test_a_failed_task_names_itself_while_the_run_moves_on(slot):
    slot.set_run("03-next-one › Polishing", None, "4 of 5")
    slot.show_outcome(
        "02-civil-cub › Polishing",
        "failed: no focus peak",
        failed=True,
        standalone=True,
    )
    assert slot.text == "02-civil-cub › Polishing failed: no focus peak"
    _outlast_the_outcome()
    assert slot.text == "03-next-one › Polishing · 4 of 5"


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


def test_autolamella_shows_milling_from_its_microscope(no_quit, qapp):
    """Through the microscope's own signal, which the bar subscribes to itself; and a
    disconnect lets it go."""
    from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module

    window = module.AutoLamellaSingleWindowUI()
    window.autolamella_ui.system_widget.connect_to_microscope()
    microscope = window.autolamella_ui.microscope
    microscope.milling_progress_signal.emit(_tick())
    assert window.status_bar.fraction == pytest.approx(0.5)
    assert not hasattr(window, "milling_progress_bar"), "one place for it"
    assert not hasattr(window, "progress_widget"), "tiles and spot burn too"

    window.status_bar.set_microscope(None)
    microscope.milling_progress_signal.emit(_tick(remaining=15.0))
    assert window.status_bar.fraction is None
    microscope.disconnect()
    _close(window, qapp)


def test_fibsem_shows_milling_from_its_microscope(qapp):
    from fibsem.ui.FibsemUI import FibsemUI

    window = FibsemUI()
    window.system_widget.connect_to_microscope()
    window.microscope.milling_progress_signal.emit(_stage_started())
    assert window.status_bar.text == "Preparing: Rough Mill stage 2 of 3"
    _close(window, qapp)


def _report(
    status, item="02-civil-cub", task="Polishing", error=None, queue_items=None
):
    from fibsem.applications.autolamella.workflows.tasks.status import (
        WorkflowStatusEvent,
        WorkflowStatusUpdate,
    )

    return WorkflowStatusEvent(
        report=WorkflowStatusUpdate(
            task_name=task,
            item_name=item,
            status=status,
            error_message=error,
            queue_position=2,
            queue_total=4,
            queue_items=queue_items,
        )
    )


def test_a_run_that_fails_a_task_leaves_a_line_until_the_next_run(
    no_quit, qapp, monkeypatch
):
    from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus
    from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module
    from fibsem.ui import notification_service

    toasts = []
    monkeypatch.setattr(
        notification_service, "show_toast", lambda msg, *a, **k: toasts.append(msg)
    )
    window = module.AutoLamellaSingleWindowUI()
    # A report locks the protocol editor, whose UI the first connect builds.
    window.autolamella_ui.system_widget.connect_to_microscope()
    signal = window.autolamella_ui.workflow_status_signal
    from fibsem.applications.autolamella.workflows.tasks.queue import WorkItem

    def _queue(second):
        return [
            WorkItem("01-fancy-mite", "Polishing", AutoLamellaTaskStatus.Completed),
            WorkItem("02-civil-cub", "Polishing", second),
            WorkItem("03-gentle-elk", "Polishing"),
        ]

    signal.emit(
        _report(
            AutoLamellaTaskStatus.InProgress,
            queue_items=_queue(AutoLamellaTaskStatus.InProgress),
        )
    )
    signal.emit(
        _report(
            AutoLamellaTaskStatus.Failed,
            error="no focus peak",
            queue_items=_queue(AutoLamellaTaskStatus.Failed),
        )
    )
    assert window.status_bar.text == "02-civil-cub › Polishing failed: no focus peak"

    window._on_workflow_finished()
    bar = window.status_bar
    assert bar.showing == "failure"
    assert bar.text == (
        "Run finished 1 of 4 failed · 02-civil-cub › Polishing: no focus peak"
    )
    assert "Workflow finished." not in toasts, "the line says it, not a toast"

    # Show: the Workflow tab, the failed row selected.
    window.tab_widget.setTabEnabled(
        window.tab_widget.indexOf(window._workflow_tab_container), True
    )
    bar._details_btn.click()
    assert window.tab_widget.currentWidget() is window._workflow_tab_container
    assert window.workflow_timeline._outer._selected_index == 1

    signal.emit(_report(AutoLamellaTaskStatus.InProgress, item="01-fancy-mite"))
    assert bar.failure is None, "the next run starts clean"
    window._on_workflow_finished()
    assert bar.failure is None
    assert toasts[-1] == "Workflow finished."
    window.autolamella_ui.microscope.disconnect()
    _close(window, qapp)


def test_a_held_run_says_so_on_the_line(no_quit, qapp):
    from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus
    from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module
    from fibsem.applications.autolamella.workflows.tasks.status import (
        Hold,
        HoldKind,
        WorkflowStatusEvent,
    )

    window = module.AutoLamellaSingleWindowUI()
    ui = window.autolamella_ui
    ui.system_widget.connect_to_microscope()
    ui.workflow_status_signal.emit(_report(AutoLamellaTaskStatus.InProgress))
    ui.hold = Hold(HoldKind.question, "answer the question on the Microscope tab")
    ui.workflow_status_signal.emit(WorkflowStatusEvent())
    assert window.status_bar.text.startswith(
        "02-civil-cub › Polishing answer the question on the Microscope tab · waiting"
    )
    ui.hold = None
    ui.workflow_status_signal.emit(WorkflowStatusEvent())
    assert "waiting" not in window.status_bar.text
    ui.microscope.disconnect()
    _close(window, qapp)


def test_the_timeline_shows_its_first_failed_row(qapp):
    from fibsem.applications.autolamella.ui.workflow_timeline_widget import (
        StepStatus,
        TimelineStep,
        WorkflowTimelineWidget,
    )

    timeline = WorkflowTimelineWidget()
    timeline.set_steps(
        [
            TimelineStep("01 · Polishing", StepStatus.COMPLETED),
            TimelineStep("02 · Polishing", StepStatus.FAILED),
            TimelineStep("03 · Polishing", StepStatus.FAILED),
        ]
    )
    assert timeline.show_first(StepStatus.FAILED) == 1
    assert timeline._selected_index == 1
    assert timeline.show_first(StepStatus.SKIPPED) is None
    timeline.deleteLater()


def test_a_long_failure_elides_rather_than_widening_the_window(no_quit, qapp):
    """A plain label's minimum is its text: one long failure set the status bar's
    minimum, and the bar set the window's, so the whole app grew wider.

    Measured against a short failure, not against no failure: the failure line's
    Show and Dismiss have widths of their own, which differ by platform font. What
    must not move the window is the length of the message."""
    from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module

    window = module.AutoLamellaSingleWindowUI()
    window.resize(1200, 800)
    window.show()
    window.status_bar.show_failure("Run finished", "1 of 5 failed · no peak")
    qapp.processEvents()
    short = window.minimumSizeHint().width()

    reason = "the stage refused the move: " + "outside the travel range " * 40
    window.status_bar.show_failure("Run finished", f"1 of 5 failed · {reason}")
    qapp.processEvents()

    # Not `==`: on CI's fonts the two differ by a few pixels either way as the bar
    # relays out. The bug this pins made it thousands of pixels wider.
    assert window.minimumSizeHint().width() <= short
    assert window.width() == 1200
    step = window.status_bar._step
    assert step.text().endswith("travel range "), "the whole text is kept"
    assert step.toolTip() == step.text(), "and readable on hover"
    assert not window.status_bar._dismiss_btn.isHidden()
    _close(window, qapp)


def test_a_short_line_still_gets_its_whole_text(slot):
    slot.resize(1000, 30)
    slot.show()
    slot.set_run("01-fancy-mite › Rough Milling", None, "3 of 5")
    from PyQt5.QtWidgets import QApplication

    QApplication.processEvents()
    assert slot._what.width() >= slot._what.sizeHint().width() - 1, "not elided"
