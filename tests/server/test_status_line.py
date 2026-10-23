"""The run's line and the microscope's progress on GET /app/workflow/status
(FIB-1542): what the window's status bar shows, for a reader with no window."""

from types import SimpleNamespace

import pytest

from fibsem.acting import AGENT, OPERATOR
from fibsem.control import Control
from fibsem.fm.progress import (
    FluorescenceAcquisitionProgress,
    FluorescenceAcquisitionStatus,
)
from fibsem.milling.progress import MillingProgress, MillingProgressStatus
from fibsem.server.app_context import AppContext
from fibsem.workflows.progress_line import (
    FM_STALE_S,
    OUTCOME_S,
    ProgressLine,
    ProgressLines,
)
from fibsem.workflows.runner import RunLine
from fibsem.workflows.status import (
    Hold,
    HoldKind,
    TaskStatus,
    WorkflowStatusEvent,
    WorkflowStatusUpdate,
)
from tests.run_state import stub_session


def _report(status=TaskStatus.InProgress, **kwargs):
    fields = dict(task_name="Mill Rough", item_name="01-lamella", status=status)
    fields.update(kwargs)
    return WorkflowStatusEvent(report=WorkflowStatusUpdate(**fields))


HOLD = Hold(HoldKind.proposal, "answer Mill Rough on 01-lamella")


def test_the_line_keeps_the_place_the_step_and_the_hold():
    line = RunLine()
    line.take_status(_report(queue_position=3, queue_total=12, iteration=2))
    line.take_step("Aligning")
    assert (line.report.queue_position, line.report.iteration) == (3, 2)
    assert line.step == "Aligning"
    # The run's own line wins while it says something; "" gives the task's back.
    line.take_status(WorkflowStatusEvent(status_bar="Loading grid G2..."))
    assert line.step == "Loading grid G2..."
    line.take_status(WorkflowStatusEvent(status_bar=""))
    assert line.step == "Aligning"
    line.take_hold(HOLD)
    since = line.held_since
    assert since is not None
    line.take_hold(HOLD)
    assert line.held_since == since, "the same hold keeps its clock"
    line.take_hold(None)
    assert line.hold is None and line.held_since is None
    # A new task has said no step yet.
    line.take_status(_report(task_name="Mill Polish"))
    assert line.step == ""


def test_failures_outlive_the_run_and_a_new_run_clears_them():
    line = RunLine()
    line.take_status(_report(TaskStatus.Failed, error_message="no focus peak"))
    line.take_hold(HOLD)
    line.ended()
    assert line.hold is None and line.step == ""
    (failure,) = line.failures
    assert failure.what == "01-lamella › Mill Rough"
    assert failure.message == "no focus peak"
    line.reset()
    assert line.failures == []


@pytest.fixture
def context():
    session = stub_session(is_running=True)
    session.microscope = SimpleNamespace(control=Control())
    return AppContext(session)


def test_the_status_carries_the_run_line(context):
    line = context._session.runner.line
    line.take_status(_report(queue_position=3, queue_total=12, iteration=1))
    line.take_step("Aligning")
    line.take_hold(HOLD)
    workflow = context.status().workflow
    assert workflow.queue_position == 3
    assert workflow.iteration == 1
    assert workflow.step == "Aligning"
    assert workflow.waiting_on == HOLD.releases
    assert workflow.waiting_since
    assert workflow.failed is None


def test_the_status_carries_the_last_failure_after_the_run(context):
    runner = context._session.runner
    runner.line.take_status(_report(TaskStatus.Failed, error_message="a"))
    runner.line.take_status(_report(TaskStatus.Failed, error_message="b"))
    runner.line.ended()
    runner.is_running = False
    workflow = context.status().workflow
    assert workflow.queue_position is None and workflow.step is None
    assert (workflow.failed.message, workflow.failed.count) == ("b", 2)


def test_a_stalled_run_says_why(context):
    runner = context._session.runner
    runner.is_running = False
    runner.last_note = "Gave up waiting on the operator."
    failed = context.status().workflow.failed
    assert (failed.what, failed.message) == (
        "Run stalled",
        "Gave up waiting on the operator.",
    )


def test_who_holds_the_microscope_when_it_is_not_the_run(context):
    control = context._session.microscope.control
    assert context.status().in_control is None
    with control.take(OPERATOR, "Move to the SEM orientation"):
        held = context.status().in_control
        assert (held.by, held.action) == (OPERATOR, "Move to the SEM orientation")


class _Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def _mill(status, **kwargs):
    return MillingProgress(status=status, milling_run_id="run-1", **kwargs)


def test_milling_progress_reads_as_the_status_bar_says_it():
    clock = _Clock()
    lines = ProgressLines(clock)
    lines.take(
        "milling",
        _mill(
            MillingProgressStatus.STAGE_STARTED,
            message="Rough Mill",
            current_stage=1,
            total_stages=3,
        ),
    )
    lines.take(
        "milling",
        _mill(
            MillingProgressStatus.STAGE_UPDATE, estimated_time=100, remaining_time=58
        ),
    )
    line = lines.current
    assert line.label == "Rough Mill"
    assert line.step == "stage 2 of 3", "a tick keeps the stage"
    assert line.fraction == pytest.approx(0.42)
    assert line.numbers.startswith("42% · ")
    lines.take("milling", _mill(MillingProgressStatus.TASK_FINISHED))
    assert lines.current == ProgressLine("Milling", "done", outcome=True)
    clock.now += OUTCOME_S + 0.1
    assert lines.current is None, "an outcome lasts a moment"


def test_the_agent_s_own_mill_is_the_agent_s():
    lines = ProgressLines()
    control = Control()
    lines.follow(SimpleNamespace(control=control))
    with control.take(AGENT, "mill"):
        lines.take(
            "milling",
            _mill(MillingProgressStatus.STAGE_STARTED, milling_task_name="trench"),
        )
    assert lines.current.label == "Agent"
    assert lines.current.step == "trench"


def test_a_fluorescence_line_goes_stale():
    clock = _Clock()
    lines = ProgressLines(clock)
    lines.take(
        "fluorescence",
        FluorescenceAcquisitionProgress(
            status=FluorescenceAcquisitionStatus.ACQUIRING_ZSTACK,
            channel="GFP",
            zlevel=3,
            total_zlevels=10,
        ),
    )
    assert lines.current.step == "GFP · plane 3 of 10"
    clock.now += FM_STALE_S + 0.1
    assert lines.current is None
    # Finished, with no fluorescence line up, says nothing.
    lines.take(
        "fluorescence",
        FluorescenceAcquisitionProgress(status=FluorescenceAcquisitionStatus.FINISHED),
    )
    assert lines.current is None


def test_the_status_follows_the_microscope_s_progress(context):
    signal = SimpleNamespace(slots=[])
    signal.connect = signal.slots.append
    signal.disconnect = signal.slots.remove
    microscope = context._session.microscope
    microscope.milling = SimpleNamespace(progress=SimpleNamespace(changed=signal))
    assert context.status().progress is None
    (slot,) = signal.slots
    slot(_mill(MillingProgressStatus.STAGE_STARTED, message="Polish"))
    assert context.status().progress.label == "Polish"
