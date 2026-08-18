"""Workflow-level assembly of the per-task estimates (FIB-666).

The per-task numbers are tested in test_task_duration_estimates.py; these are about
what the workflow does with them -- ordering, holds, and which rows are flagged.
"""

from datetime import datetime, timedelta

import pytest

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskDescription,
    AutoLamellaTaskProtocol,
    AutoLamellaWorkflowConfig,
    Experiment,
    Lamella,
)
from fibsem.applications.autolamella.workflows.tasks.fiducial import MillFiducialTaskConfig
from fibsem.applications.autolamella.workflows.tasks.select_position import (
    SelectMillingPositionTaskConfig,
)
from fibsem.applications.autolamella.workflows.workflow_estimate import estimate_workflow

NOW = datetime(2026, 8, 18, 14, 0, 0)


def _experiment(tmp_path, n_lamella: int = 2, scheduled=None, supervised=()) -> Experiment:
    """Two tasks over n lamella, with the protocol wired to match."""
    exp = Experiment(path=str(tmp_path), name="estimate-test")
    for i in range(n_lamella):
        lamella = Lamella(path=str(tmp_path), number=i + 1, petname=f"{i + 1:02d}-lamella")
        lamella.task_config["Setup Lamella Position"] = SelectMillingPositionTaskConfig()
        lamella.task_config["Mill Fiducial"] = MillFiducialTaskConfig()
        exp.add_lamella(lamella)

    # task_protocol is None on a fresh Experiment -- "must be set externally"
    exp.task_protocol = AutoLamellaTaskProtocol()
    exp.task_protocol.workflow_config = AutoLamellaWorkflowConfig(
        tasks=[
            AutoLamellaTaskDescription(
                name=name,
                supervise=name in supervised,
                required=True,
                scheduled_at=(scheduled or {}).get(name),
            )
            for name in ("Setup Lamella Position", "Mill Fiducial")
        ]
    )
    return exp


def _names(exp) -> list:
    return [lam.name for lam in exp.positions]


# ── the breakdown ────────────────────────────────────────────────────────────

def test_a_row_per_task_in_the_order_given(tmp_path):
    exp = _experiment(tmp_path)
    est = estimate_workflow(exp, ["Mill Fiducial", "Setup Lamella Position"], _names(exp), now=NOW)
    assert [t.name for t in est.tasks] == ["Mill Fiducial", "Setup Lamella Position"]


def test_a_row_totals_every_lamella_it_runs_on(tmp_path):
    exp = _experiment(tmp_path, n_lamella=3)
    est = estimate_workflow(exp, ["Mill Fiducial"], _names(exp), now=NOW)
    row = est.tasks[0]
    single = exp.positions[0].task_config["Mill Fiducial"].estimated_duration
    assert row.lamella_count == 3
    assert row.seconds == pytest.approx(3 * single)


def test_step_count_matches_the_queue(tmp_path):
    """One item per (task, lamella), the same shape build_from_matrix produces."""
    exp = _experiment(tmp_path, n_lamella=3)
    est = estimate_workflow(exp, ["Mill Fiducial", "Setup Lamella Position"], _names(exp), now=NOW)
    assert est.step_count == 6


def test_a_lamella_without_a_config_for_the_task_is_not_counted(tmp_path):
    """It has no duration to offer, and inventing one is worse than omitting it."""
    exp = _experiment(tmp_path, n_lamella=2)
    del exp.positions[1].task_config["Mill Fiducial"]
    est = estimate_workflow(exp, ["Mill Fiducial"], _names(exp), now=NOW)
    assert est.tasks[0].lamella_count == 1


def test_only_the_named_lamella_are_included(tmp_path):
    exp = _experiment(tmp_path, n_lamella=3)
    est = estimate_workflow(exp, ["Mill Fiducial"], [exp.positions[0].name], now=NOW)
    assert est.tasks[0].lamella_count == 1


# ── the totals ───────────────────────────────────────────────────────────────

def test_work_seconds_is_the_sum_of_the_rows(tmp_path):
    exp = _experiment(tmp_path)
    est = estimate_workflow(exp, ["Mill Fiducial", "Setup Lamella Position"], _names(exp), now=NOW)
    assert est.work_seconds == pytest.approx(sum(t.seconds for t in est.tasks))


def test_finish_is_now_plus_the_work_when_nothing_is_scheduled(tmp_path):
    exp = _experiment(tmp_path)
    est = estimate_workflow(exp, ["Mill Fiducial"], _names(exp), now=NOW)
    assert est.expected_finish == NOW + timedelta(seconds=est.work_seconds)
    assert est.hold_seconds == 0.0


def test_supervised_work_still_counts_towards_the_total(tmp_path):
    """A supervised task still mills, images and moves -- only the pause is unbounded,
    so dropping the whole task would understate the total by the work it does."""
    plain = _experiment(tmp_path)
    supervised = _experiment(tmp_path, supervised=("Mill Fiducial",))
    tasks = ["Mill Fiducial"]
    assert estimate_workflow(supervised, tasks, _names(supervised), now=NOW).work_seconds == (
        pytest.approx(estimate_workflow(plain, tasks, _names(plain), now=NOW).work_seconds)
    )


def test_supervised_rows_are_flagged_and_countable(tmp_path):
    exp = _experiment(tmp_path, supervised=("Mill Fiducial",))
    est = estimate_workflow(exp, ["Mill Fiducial", "Setup Lamella Position"], _names(exp), now=NOW)
    assert [t.name for t in est.supervised_tasks] == ["Mill Fiducial"]
    assert est.waits_for_a_human is True


def test_a_workflow_with_no_supervision_says_so(tmp_path):
    exp = _experiment(tmp_path)
    est = estimate_workflow(exp, ["Mill Fiducial"], _names(exp), now=NOW)
    assert est.waits_for_a_human is False
    assert est.supervised_tasks == []


# ── scheduling makes it a walk, not a sum ────────────────────────────────────

def test_a_scheduled_task_holds_the_workflow_until_its_time(tmp_path):
    """now + Σ would be wrong by the length of the wait."""
    at = NOW + timedelta(hours=4)
    exp = _experiment(tmp_path, scheduled={"Mill Fiducial": at})
    est = estimate_workflow(exp, ["Mill Fiducial"], _names(exp), now=NOW)

    assert est.hold_seconds == pytest.approx(4 * 3600)
    assert est.expected_finish == at + timedelta(seconds=est.work_seconds)
    assert est.expected_finish > NOW + timedelta(seconds=est.work_seconds)


def test_the_hold_is_not_counted_as_work(tmp_path):
    at = NOW + timedelta(hours=4)
    scheduled = _experiment(tmp_path, scheduled={"Mill Fiducial": at})
    plain = _experiment(tmp_path)
    assert estimate_workflow(scheduled, ["Mill Fiducial"], _names(scheduled), now=NOW).work_seconds == (
        pytest.approx(estimate_workflow(plain, ["Mill Fiducial"], _names(plain), now=NOW).work_seconds)
    )


def test_a_scheduled_time_already_past_holds_nothing(tmp_path):
    exp = _experiment(tmp_path, scheduled={"Mill Fiducial": NOW - timedelta(hours=1)})
    est = estimate_workflow(exp, ["Mill Fiducial"], _names(exp), now=NOW)
    assert est.hold_seconds == 0.0
    assert est.expected_finish == NOW + timedelta(seconds=est.work_seconds)


def test_the_hold_is_measured_from_where_the_earlier_tasks_left_the_clock(tmp_path):
    """The task ahead of it has already consumed part of the wait."""
    at = NOW + timedelta(hours=4)
    exp = _experiment(tmp_path, scheduled={"Mill Fiducial": at})
    est = estimate_workflow(exp, ["Setup Lamella Position", "Mill Fiducial"], _names(exp), now=NOW)
    setup_seconds = est.tasks[0].seconds
    assert est.hold_seconds == pytest.approx(4 * 3600 - setup_seconds)


def test_scheduled_rows_are_flagged(tmp_path):
    at = NOW + timedelta(hours=4)
    exp = _experiment(tmp_path, scheduled={"Mill Fiducial": at})
    est = estimate_workflow(exp, ["Mill Fiducial", "Setup Lamella Position"], _names(exp), now=NOW)
    assert [t.name for t in est.scheduled_tasks] == ["Mill Fiducial"]
    assert est.tasks[0].scheduled_at == at


def test_a_timezone_aware_schedule_does_not_raise(tmp_path):
    """A hand-edited protocol may carry one; the workflow loop normalises it too."""
    from datetime import timezone

    aware = (NOW + timedelta(hours=4)).replace(tzinfo=timezone.utc)
    exp = _experiment(tmp_path, scheduled={"Mill Fiducial": aware})
    est = estimate_workflow(exp, ["Mill Fiducial"], _names(exp), now=NOW)
    assert est.tasks[0].scheduled_at.tzinfo is None


# ── nothing to do ────────────────────────────────────────────────────────────

def test_an_empty_workflow_finishes_now(tmp_path):
    exp = _experiment(tmp_path)
    est = estimate_workflow(exp, [], _names(exp), now=NOW)
    assert est.tasks == []
    assert est.work_seconds == 0.0
    assert est.expected_finish == NOW
    assert est.step_count == 0
