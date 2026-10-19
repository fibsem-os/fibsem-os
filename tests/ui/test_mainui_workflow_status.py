"""What the main window does with a fire-and-forget status update.

The main window consumes workflow status twice over: transient status-bar text
(still on `workflow_update_signal` until `update_status_ui` converts) and the
task manager's lifecycle reports (timeline feed, the "Workflow: ..." message,
run/stop buttons), which arrive as `WorkflowStatusEvent` on
`workflow_status_signal`. Written pinning the dict path before the move; the
lifecycle tests now hold the same observable behaviour on the new channel, plus
that the dict channel ignores a straggling status key.

This is the first test to construct the real main window offscreen. The only
thing that ever prevented it was the napari Minimap tab, whose vispy canvas called
`glGetIntegerv` with no GL context and took the process down with SIGSEGV; the
fixture used to stub that one method out. The tab is gone (#585/#586), so every
tab here is real.

This fixture deliberately sets no border state first, so a handler that reads
one before the window has set it fails the first status these tests deliver to a
fresh window (an AttributeError in a queued slot is a process abort, FIB-329).
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.imaging.spot import SpotBurnProgress, SpotBurnStatus
from fibsem.workflows.status import (
    Hold,
    HoldKind,
    TaskStatus,
    WorkflowStatusUpdate,
)


@pytest.fixture(scope="module")
def main_ui(qapp):
    from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module

    window = module.AutoLamellaSingleWindowUI()
    # First connect builds the protocol editor's full UI (its lock path runs on
    # every lifecycle report); Demo, so no hardware.
    window.autolamella_ui.system_widget.connect_to_microscope()
    yield window
    if window.autolamella_ui.microscope is not None:
        window.autolamella_ui.microscope.disconnect()
    # closeEvent ends in app.quit() — the shipped quit-freeze workaround. On the
    # shared test QApplication that latches an interrupt (no event loop is running
    # to consume it), and every later QEventLoop.exec_() in the pytest process
    # then returns immediately — which silently broke every worker test that ran
    # after this module. Run the real close cleanup with quit stubbed out.
    original_quit = qapp.quit
    qapp.quit = lambda: None
    try:
        window.close()
    finally:
        qapp.quit = original_quit


def test_a_status_event_report_shows_the_run_on_the_status_bar(main_ui):
    from fibsem.workflows.status import (
        WorkflowStatusEvent,
    )

    report = WorkflowStatusUpdate(
        task_name="Polishing",
        item_name="lamella-02",
        status=TaskStatus.InProgress,
        queue_position=3,
        queue_total=4,
        queue_items=None,
    )

    # Through the embedded window's signal: a direct connection runs the handler
    # synchronously, so this also pins that the main window is connected to it.
    main_ui.autolamella_ui.workflow_status_signal.emit(
        WorkflowStatusEvent(report=report)
    )

    assert main_ui.status_bar.text == "lamella-02 › Polishing · 3 of 4"
    assert main_ui.status_bar.showing == "run"
    assert main_ui.workflow_controls.stop_btn.isVisibleTo(main_ui)
    assert not main_ui.workflow_controls.run_btn.isVisibleTo(main_ui)


def test_a_status_event_carries_transient_status_bar_text(main_ui):
    from fibsem.workflows.status import (
        WorkflowStatusEvent,
    )

    main_ui.autolamella_ui.workflow_status_signal.emit(
        WorkflowStatusEvent(status_bar="Scheduled start in 4 s")
    )

    assert "Scheduled start in 4 s" in main_ui.status_bar.text


def test_a_status_event_snapshot_reaches_the_windows_own_timeline(main_ui):
    # The tests above use queue_items=None, which tells the timeline to leave its
    # rows alone — so none of them would notice the handler dropping the
    # update_from_status call. This one hands over a real snapshot and reads the
    # window's own timeline rows back.
    from fibsem.workflows.queue import WorkItem
    from fibsem.workflows.status import (
        WorkflowStatusEvent,
    )

    items = [
        WorkItem(
            item_name="lamella-03",
            task_name="Trench",
            status=TaskStatus.InProgress,
        ),
        WorkItem(item_name="lamella-04", task_name="Trench"),
    ]
    report = WorkflowStatusUpdate(
        task_name="Trench",
        item_name="lamella-03",
        status=TaskStatus.InProgress,
        queue_items=items,
    )

    main_ui.autolamella_ui.workflow_status_signal.emit(
        WorkflowStatusEvent(report=report)
    )

    rows = [(i.item_name, i.task_name) for i in main_ui.workflow_tab.timeline._items]
    assert rows == [("lamella-03", "Trench"), ("lamella-04", "Trench")]

    # Reconcile back down to zero rows so the module-scoped window carries no
    # timeline state into other tests (an empty snapshot means exactly that).
    main_ui.autolamella_ui.workflow_status_signal.emit(
        WorkflowStatusEvent(
            report=WorkflowStatusUpdate(queue_items=[]),
        )
    )
    assert main_ui.workflow_tab.timeline._items == []


def test_a_status_event_refreshes_the_waiting_indicators(main_ui):
    # The attention button is driven by _refresh_workflow_indicators, which must
    # run from the status handler too: a status event arriving while a question
    # is pending must not take the waiting chrome down — and one arriving after
    # the answer must.
    from fibsem.workflows.status import (
        WorkflowStatusEvent,
    )

    main_ui.autolamella_ui.hold = Hold(
        HoldKind.proposal, "answer the question on the Microscope tab"
    )
    main_ui.autolamella_ui.workflow_status_signal.emit(WorkflowStatusEvent())
    assert main_ui.workflow_controls.attention_btn.isVisibleTo(main_ui)

    main_ui.autolamella_ui.hold = None
    main_ui.autolamella_ui.workflow_status_signal.emit(WorkflowStatusEvent())
    assert not main_ui.workflow_controls.attention_btn.isVisibleTo(main_ui)


# --- spot burn progress ----------------------------------------------------
#
# The same slot, reached by a different signal. `spot_burn_progress_signal` was typed
# in #595, and the main window's handler kept one line that read the report as the dict
# it used to be -- `ddict.get("finished")`. `SpotBurnProgress` is a frozen dataclass with
# no `.get`, so the first report of every supervised burn raised AttributeError inside a
# queued slot, which PyQt5 turns into a process abort: the application vanished the
# moment burning started, on v0.5.2rc1.
#
# Both tests below fail on that line, and neither needs hardware: the handler is called
# directly, exactly as the queued connection would call it. It is the status bar's
# since FIB-1188, which decodes the microscope's reports itself.


def test_a_burning_report_renders_without_taking_the_app_down(main_ui):
    """The crash. Reaching the assertion at all is most of what this tests."""
    report = SpotBurnProgress(
        status=SpotBurnStatus.BURNING,
        current_point=2,
        total_points=5,
        total_remaining_time=30.0,
        total_estimated_time=50.0,
    )

    main_ui.status_bar._on_spot_burn_progress(report)

    assert main_ui.status_bar.fraction == pytest.approx(0.4)
    # Under the run's task if this module's window has one going, on its own if not.
    assert "Spot burn" in main_ui.status_bar.text
    assert main_ui.status_bar.text.endswith("spot 2 of 5 40% · 30s left")


def test_a_terminal_report_is_recognised_as_terminal(main_ui):
    """The other half of the same line: `.get("finished")` decided when to schedule the
    bar's reset, so a fix that stopped the crash but misread the outcome would leave the
    bar full forever. Read through `status.is_terminal`, which is what both consumers of
    this signal use."""
    main_ui.status_bar._on_spot_burn_progress(
        SpotBurnProgress(
            status=SpotBurnStatus.FINISHED, current_point=5, total_points=5
        )
    )

    assert "Spot burn" in main_ui.status_bar.text
    assert main_ui.status_bar.text.endswith("done")

    assert SpotBurnStatus.FINISHED.is_terminal
    assert not SpotBurnStatus.BURNING.is_terminal


def test_an_event_without_status_bar_text_leaves_the_bar_alone(main_ui):
    from fibsem.workflows.status import (
        WorkflowStatusEvent,
    )

    main_ui.status_bar.set_run_step("previous message")
    before = main_ui.status_bar.text

    main_ui.autolamella_ui.workflow_status_signal.emit(
        WorkflowStatusEvent(message="not for the status bar")
    )

    assert main_ui.status_bar.text == before


def test_the_lamella_sub_tab_is_called_history_not_review(main_ui):
    """The lamella's per-task record is History, beside Protocol; the window
    has no Review tab."""
    container = main_ui._lamella_tab_container
    from PyQt5.QtWidgets import QTabWidget

    tabs = container.findChild(QTabWidget)
    labels = [tabs.tabText(i) for i in range(tabs.count())]
    assert labels == ["Protocol", "History"]
    main_labels = [
        main_ui.tab_widget.tabText(i) for i in range(main_ui.tab_widget.count())
    ]
    assert "Review" not in main_labels
