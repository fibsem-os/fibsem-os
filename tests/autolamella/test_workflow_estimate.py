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


# ── the live queue ────────────────────────────────────────────────────────────
# `estimate_queue` answers the same question as `estimate_workflow` from the other end:
# against a queue that is part-way through rather than an experiment about to start.

from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus  # noqa: E402
from fibsem.applications.autolamella.workflows.tasks.queue import WorkItem  # noqa: E402
from fibsem.applications.autolamella.workflows.workflow_estimate import (  # noqa: E402
    estimate_queue,
)


def _items(*spec) -> list:
    """(lamella, task) or (lamella, task, status) triples, in queue order."""
    return [
        WorkItem(lamella_name=s[0], task_name=s[1],
                 status=s[2] if len(s) > 2 else AutoLamellaTaskStatus.NotStarted)
        for s in spec
    ]


def _flat(seconds: float):
    return lambda item: seconds


def test_an_empty_queue_finishes_now():
    est = estimate_queue([], _flat(60.0), now=NOW)
    assert est.remaining_seconds == 0.0
    assert est.expected_finish == NOW


def test_pending_items_are_summed():
    est = estimate_queue(_items(("01", "A"), ("02", "A")), _flat(60.0), now=NOW)
    assert est.remaining_seconds == pytest.approx(120.0)
    assert est.expected_finish == NOW + timedelta(seconds=120)


def test_work_already_done_costs_nothing():
    """The whole point of asking again mid-workflow: the finish converges on its own,
    without anything being re-learned from how the finished rows actually went."""
    items = _items(("01", "A", AutoLamellaTaskStatus.Completed),
                   ("02", "A", AutoLamellaTaskStatus.Failed),
                   ("03", "A", AutoLamellaTaskStatus.Skipped),
                   ("04", "A", AutoLamellaTaskStatus.Cancelled),
                   ("05", "A"))
    est = estimate_queue(items, _flat(60.0), now=NOW)
    assert est.remaining_seconds == pytest.approx(60.0)


def test_an_item_with_no_estimate_contributes_nothing():
    """A lamella with no config for the task has no duration to offer, and inventing
    one would be worse than leaving it out."""
    items = _items(("01", "A"), ("02", "A"))
    seconds_for = lambda item: 60.0 if item.lamella_name == "01" else None
    assert estimate_queue(items, seconds_for, now=NOW).remaining_seconds == pytest.approx(60.0)


# ── the running task ──────────────────────────────────────────────────────────

def test_the_running_task_contributes_only_what_is_left_of_it():
    items = _items(("01", "A", AutoLamellaTaskStatus.InProgress), ("02", "A"))
    est = estimate_queue(items, _flat(100.0), now=NOW, active_elapsed=40.0)
    assert est.active_remaining == pytest.approx(60.0)
    assert est.remaining_seconds == pytest.approx(160.0)
    assert est.active_estimate_spent is False


def test_a_task_past_its_estimate_contributes_zero_not_a_negative():
    """The total must not go backwards. It slides later by a second per second instead,
    which is the honest answer once the task's end is unknown -- a total that stalls or
    goes negative is the FIB-522 failure."""
    items = _items(("01", "A", AutoLamellaTaskStatus.InProgress), ("02", "A"))
    est = estimate_queue(items, _flat(100.0), now=NOW, active_elapsed=250.0)
    assert est.active_remaining == 0.0
    assert est.active_estimate_spent is True
    assert est.remaining_seconds == pytest.approx(100.0)  # the pending item, and no more


def test_nothing_running_reports_no_active_remaining():
    est = estimate_queue(_items(("01", "A")), _flat(60.0), now=NOW)
    assert est.active_remaining is None
    assert est.active_estimate_spent is False


def test_a_running_task_with_no_elapsed_yet_is_not_guessed_at():
    """Before the first tick there is nothing to subtract from, so the task is left out
    rather than counted at its full estimate and then jumping."""
    items = _items(("01", "A", AutoLamellaTaskStatus.InProgress))
    assert estimate_queue(items, _flat(100.0), now=NOW).active_remaining is None


# ── holds ─────────────────────────────────────────────────────────────────────

def test_a_scheduled_task_pushes_the_finish_out():
    at = NOW + timedelta(hours=4)
    est = estimate_queue(_items(("01", "A")), _flat(60.0), schedule={"A": at}, now=NOW)
    assert est.hold_seconds == pytest.approx(4 * 3600)
    assert est.expected_finish == at + timedelta(seconds=60)


def test_the_hold_is_wall_clock_not_work():
    """`remaining_seconds` answers "when do I come back" and `work_seconds` answers
    "how much is there left to do" -- the header quotes one or the other depending on
    whether a clock can be quoted at all."""
    at = NOW + timedelta(hours=4)
    est = estimate_queue(_items(("01", "A")), _flat(60.0), schedule={"A": at}, now=NOW)
    assert est.work_seconds == pytest.approx(60.0)
    assert est.remaining_seconds == pytest.approx(4 * 3600 + 60.0)


def test_a_task_holds_once_and_its_other_lamella_find_the_time_passed():
    """`_wait_until_scheduled` runs per queue *item*, so the first item of a scheduled
    task waits and the rest walk straight through. Charging the hold per item would
    quote four hours three times over."""
    at = NOW + timedelta(hours=4)
    items = _items(("01", "A"), ("02", "A"), ("03", "A"))
    est = estimate_queue(items, _flat(60.0), schedule={"A": at}, now=NOW)
    assert est.hold_seconds == pytest.approx(4 * 3600)
    assert est.expected_finish == at + timedelta(seconds=180)


def test_the_hold_is_measured_from_where_the_queue_has_got_to():
    """Work ahead of the scheduled task has already eaten part of the wait."""
    at = NOW + timedelta(hours=4)
    items = _items(("01", "Setup"), ("01", "A"))
    est = estimate_queue(items, _flat(600.0), schedule={"A": at}, now=NOW)
    assert est.hold_seconds == pytest.approx(4 * 3600 - 600.0)


def test_a_queue_scheduled_time_already_past_holds_nothing():
    items = _items(("01", "A"))
    est = estimate_queue(items, _flat(60.0),
                         schedule={"A": NOW - timedelta(hours=1)}, now=NOW)
    assert est.hold_seconds == 0.0
    assert est.expected_finish == NOW + timedelta(seconds=60)


def test_a_timezone_aware_schedule_in_the_queue_does_not_raise():
    from datetime import timezone
    aware = (NOW + timedelta(hours=4)).replace(tzinfo=timezone.utc)
    est = estimate_queue(_items(("01", "A")), _flat(60.0), schedule={"A": aware}, now=NOW)
    assert est.expected_finish is not None


# ── adding to a running queue ─────────────────────────────────────────────────
# Two figures that can disagree: the work added, and the delay it causes. The gap
# between them is the whole reason this is computed by difference rather than summed.

from fibsem.applications.autolamella.workflows.workflow_estimate import (  # noqa: E402
    estimate_addition,
)


def test_adding_work_to_a_plain_queue_delays_it_by_exactly_that_work():
    est = estimate_addition(_items(("01", "A")), [("02", "A")], _flat(60.0), now=NOW)
    assert est.work_seconds == pytest.approx(60.0)
    assert est.delay_seconds == pytest.approx(60.0)
    assert est.finish_after == est.finish_before + timedelta(seconds=60)


def test_the_work_added_does_not_depend_on_where_it_lands():
    items = _items(("01", "A"), ("02", "B"))
    at_end = estimate_addition(items, [("03", "A")], _flat(60.0), run_next=False, now=NOW)
    up_next = estimate_addition(items, [("03", "A")], _flat(60.0), run_next=True, now=NOW)
    assert at_end.work_seconds == pytest.approx(up_next.work_seconds)


def test_work_that_fits_inside_a_scheduled_hold_costs_no_delay():
    """The case the two figures exist for. The workflow was going to sit until 8pm
    anyway; work slotted in ahead of that is absorbed, so an hour of milling moves the
    finish by nothing. Summing the addition instead of differencing would say an hour."""
    at = NOW + timedelta(hours=4)
    items = _items(("01", "Scheduled"))
    est = estimate_addition(items, [("02", "Earlier")], _flat(3600.0),
                            run_next=True, schedule={"Scheduled": at}, now=NOW)
    assert est.work_seconds == pytest.approx(3600.0)
    assert est.delay_seconds == 0.0
    assert est.finish_after == est.finish_before


def test_the_same_work_after_the_hold_does_delay_it():
    """Same queue, same work, other end -- and now it costs its full duration. Position
    is a property of the hold, not of the work, which is why this cannot be a sum."""
    at = NOW + timedelta(hours=4)
    items = _items(("01", "Scheduled"))
    est = estimate_addition(items, [("02", "Later")], _flat(3600.0),
                            run_next=False, schedule={"Scheduled": at}, now=NOW)
    assert est.delay_seconds == pytest.approx(3600.0)


def test_run_next_lands_the_batch_ahead_of_the_pending_work_not_of_history():
    """`front=True` clamps to `_pending_start()`, so a completed item stays where it is
    and the addition cannot be placed before the running one."""
    from fibsem.applications.autolamella.workflows.workflow_estimate import _with_addition
    items = _items(("01", "A", AutoLamellaTaskStatus.Completed),
                   ("02", "A", AutoLamellaTaskStatus.InProgress),
                   ("03", "A"))
    order = [(i.lamella_name, i.task_name) for i in
             _with_addition(items, _items(("09", "New")), run_next=True)]
    assert order == [("01", "A"), ("02", "A"), ("09", "New"), ("03", "A")]


def test_added_items_with_no_estimate_are_counted_but_not_priced():
    """The dialog needs to know the difference between "no delay" and "cannot say"."""
    est = estimate_addition(_items(("01", "A")), [("02", "A"), ("03", "A")],
                            lambda item: None, now=NOW)
    assert est.total_count == 2
    assert est.priced_count == 0
    assert est.is_priced is False
    assert est.delay_seconds == 0.0


def test_a_partly_priced_addition_still_says_so():
    seconds_for = lambda item: 60.0 if item.lamella_name == "02" else None
    est = estimate_addition(_items(("01", "A")), [("02", "A"), ("03", "A")],
                            seconds_for, now=NOW)
    assert (est.priced_count, est.total_count) == (1, 2)
    assert est.is_priced is True
    assert est.work_seconds == pytest.approx(60.0)


def test_adding_to_an_empty_queue_is_the_whole_queue():
    est = estimate_addition([], [("01", "A")], _flat(60.0), now=NOW)
    assert est.work_seconds == pytest.approx(60.0)
    assert est.finish_after == NOW + timedelta(seconds=60)


def test_the_delay_never_goes_negative():
    """Two large sums differenced can land a hair below zero; `-0s` reads as a bug."""
    est = estimate_addition(_items(("01", "A")), [], _flat(60.0), now=NOW)
    assert est.delay_seconds == 0.0
    assert est.total_count == 0


def test_the_delay_is_exact_even_when_the_running_task_cannot_be_priced():
    """Both walks treat the active task identically, so whatever `active_elapsed` fails
    to account for cancels in the subtraction. The absolute finish may be optimistic;
    the delay is not."""
    items = _items(("01", "A", AutoLamellaTaskStatus.InProgress), ("02", "A"))
    known = estimate_addition(items, [("03", "A")], _flat(60.0), now=NOW,
                              active_elapsed=10.0)
    unknown = estimate_addition(items, [("03", "A")], _flat(60.0), now=NOW)
    assert known.delay_seconds == pytest.approx(unknown.delay_seconds)
    assert known.finish_after != unknown.finish_after
