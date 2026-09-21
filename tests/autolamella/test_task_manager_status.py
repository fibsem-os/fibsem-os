"""Tests for TaskManager status emission and skip logic.

Both used to be anchored to the task x lamella lists frozen at launch, which
broke as soon as the queue could be mutated: an added task raised ValueError
out of _emit_status, and an added lamella was silently skipped as
"not_required". Progress is now measured against the live queue instead.

These drive _emit_status/_should_skip against a real Experiment, real Lamellas
and a real task protocol. The only stand-ins are the two the environment cannot
provide: a microscope and the Qt UI.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskDescription,
    AutoLamellaTaskProtocol,
    AutoLamellaTaskState,
    AutoLamellaWorkflowConfig,
    DefectType,
    Experiment,
    Lamella,
)
from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus as Status
from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager


class RecordingUI:
    """The one stand-in here: a recorder in place of AutoLamellaUI.

    AutoLamellaUI needs a live napari viewer with docked widgets, so it cannot be
    built in a test. Everything else below is the real object. Deliberately not a
    QObject with real pyqtSignals — this module has no Qt dependency, and adding
    one would take it out of CI, where PyQt5 is not installed.
    """

    def __init__(self):
        # Everything the workflow says arrives on workflow_status_signal now
        # (the dict signal is gone).
        self.workflow_status_signal = _Recorder()
        self._task_manager = None  # _check_for_abort reads this


class _Recorder:
    def __init__(self):
        self.emitted: List = []

    def emit(self, payload) -> None:
        self.emitted.append(payload)


class NoMicroscope:
    """Enough microscope to construct a TaskManager and retract the objective.

    Experiment.register_metadata stamps the user and experiment ref onto it, so
    it has to be something attributes can be set on — None will not do.
    """

    fm = None


def make_lamella(
    experiment: Experiment, name: str, is_failure: bool = False, completed=()
) -> Lamella:
    lamella = Lamella(
        path=Path(experiment.path) / name,
        number=len(experiment.positions) + 1,
        petname=name,
    )
    if is_failure:
        lamella.defect.state = DefectType.FAILURE
    for task_name in completed:
        lamella.task_history.append(
            AutoLamellaTaskState(name=task_name, status=Status.Completed)
        )
    experiment.positions.append(lamella)
    return lamella


def make_experiment(
    tmp_path: Path,
    requirements: Optional[Dict[str, List[str]]] = None,
    lamella_names=("L1", "L2"),
) -> Experiment:
    """A real Experiment with a real task protocol.

    ``required=False`` throughout so nothing is ever "complete" — the completion
    hooks are a different test module's subject and would only add noise here.
    """
    reqs = requirements or {}
    experiment = Experiment(path=tmp_path, name="test-exp")
    experiment.task_protocol = AutoLamellaTaskProtocol(
        workflow_config=AutoLamellaWorkflowConfig(
            tasks=[
                AutoLamellaTaskDescription(
                    name=name,
                    required=False,
                    requires=reqs.get(name, []),
                )
                for name in ("Trench", "Undercut", "Polishing")
            ]
        )
    )
    for name in lamella_names:
        make_lamella(experiment, name)
    return experiment


def run_queue_with(manager: TaskManager, on_task=None):
    """Drive _run_queue with task execution stubbed out.

    ``on_task(task_name, lamella)`` runs in place of the real task and may
    mutate the queue — which is the mid-run editing case under test.
    """
    executed = []

    def _run_single_task(task_name, lamella):
        executed.append((lamella.name, task_name))
        lamella.task_state.status = Status.Completed
        if on_task is not None:
            on_task(task_name, lamella)
        return None

    manager.parent_ui._task_manager = manager
    manager._run_single_task = _run_single_task
    manager._run_queue()
    return executed


@pytest.fixture
def manager(tmp_path) -> TaskManager:
    m = TaskManager(
        microscope=NoMicroscope(),
        experiment=make_experiment(tmp_path),
        parent_ui=RecordingUI(),
    )
    m.queue.build_from_matrix(["Trench", "Undercut"], ["L1", "L2"])
    return m


def last_status(manager: TaskManager):
    return manager.parent_ui.workflow_status_signal.emitted[-1].report


# ── _emit_status ──────────────────────────────────────────────────────────────


def test_emit_reports_position_in_the_live_queue(manager):
    item = manager.queue.items[2]
    manager._emit_status(
        item=item,
        lamella=manager.experiment.get_lamella_by_name(item.lamella_name),
        status=Status.InProgress,
    )
    status = last_status(manager)
    assert status.queue_position == 3
    assert status.queue_total == 4


def test_emit_for_a_task_outside_the_launch_plan(manager):
    """Regression: this raised ValueError out of task_names.index()."""
    added = manager.queue.add("L1", "Polishing")
    assert added.task_name not in manager.queue.task_names

    manager._emit_status(
        item=added,
        lamella=manager.experiment.get_lamella_by_name("L1"),
        status=Status.InProgress,
    )
    status = last_status(manager)
    assert status.task_name == "Polishing"
    assert status.queue_position == 5
    assert status.queue_total == 5


def test_emit_for_a_lamella_outside_the_launch_plan(manager):
    added = manager.queue.add("L99", "Trench")
    manager._emit_status(
        item=added,
        lamella=make_lamella(manager.experiment, "L99"),
        status=Status.InProgress,
    )
    status = last_status(manager)
    assert status.item_name == "L99"
    # deprecated alias, dropped with the HookContext shims after v0.6 (FIB-464)
    assert status.lamella_name == "L99"
    assert status.queue_position == 5


def test_emit_position_follows_a_reorder(manager):
    item = manager.queue.items[3]
    manager.queue.move_to_front(item.id)
    manager._emit_status(
        item=item,
        lamella=manager.experiment.get_lamella_by_name(item.lamella_name),
        status=Status.InProgress,
    )
    assert last_status(manager).queue_position == 1


def test_emit_carries_the_launch_plan_as_context(manager):
    item = manager.queue.items[0]
    manager._emit_status(
        item=item,
        lamella=manager.experiment.get_lamella_by_name("L1"),
        status=Status.InProgress,
    )
    status = last_status(manager)
    assert status.task_names == ["Trench", "Undercut"]
    assert status.lamella_names == ["L1", "L2"]


def test_emit_includes_a_queue_snapshot(manager):
    item = manager.queue.items[0]
    manager._emit_status(
        item=item,
        lamella=manager.experiment.get_lamella_by_name("L1"),
        status=Status.InProgress,
    )
    snapshot = last_status(manager).queue_items
    assert len(snapshot) == 4
    snapshot[0].status = Status.Failed
    assert manager.queue.items[0].status is not Status.Failed


def test_emit_passes_through_error_and_skip_detail(manager):
    item = manager.queue.items[0]
    manager._emit_status(
        item=item,
        lamella=manager.experiment.get_lamella_by_name("L1"),
        status=Status.Failed,
        msg="boom",
        error_message="it broke",
        task_duration=12.5,
    )
    status = last_status(manager)
    assert status.error_message == "it broke"
    assert status.task_duration == 12.5
    assert manager.parent_ui.workflow_status_signal.emitted[-1].message == "boom"


def test_emit_is_a_noop_headless(tmp_path):
    experiment = make_experiment(tmp_path, lamella_names=["L1"])
    m = TaskManager(microscope=NoMicroscope(), experiment=experiment, parent_ui=None)
    m.queue.build_from_matrix(["Trench"], ["L1"])
    m._emit_status(
        item=m.queue.items[0],
        lamella=experiment.get_lamella_by_name("L1"),
        status=Status.InProgress,
    )  # must not raise


# ── _should_skip ──────────────────────────────────────────────────────────────


def test_lamella_outside_the_launch_selection_is_not_skipped(manager):
    """Regression: the old allow-list check skipped anything added mid-run."""
    lamella = make_lamella(manager.experiment, "L99")
    assert "L99" not in manager.queue.lamella_names
    assert manager._should_skip(lamella, "Trench") is None


def test_failed_lamella_is_skipped(manager):
    lamella = manager.experiment.get_lamella_by_name("L1")
    lamella.defect.state = DefectType.FAILURE
    assert manager._should_skip(lamella, "Trench") == "failure"


def test_missing_prerequisites_are_skipped(tmp_path):
    experiment = make_experiment(
        tmp_path, requirements={"Undercut": ["Trench"]}, lamella_names=["L1"]
    )
    m = TaskManager(
        microscope=NoMicroscope(), experiment=experiment, parent_ui=RecordingUI()
    )
    m.queue.build_from_matrix(["Trench", "Undercut"], ["L1"])
    lamella = experiment.get_lamella_by_name("L1")
    assert m._should_skip(lamella, "Undercut") == "missing_prereqs"

    lamella.task_history.append(
        AutoLamellaTaskState(name="Trench", status=Status.Completed)
    )
    assert m._should_skip(lamella, "Undercut") is None


def test_a_prerequisite_still_in_the_queue_defers_rather_than_skips(tmp_path):
    """The prerequisite check used to be terminal. With Trench re-queued behind
    Undercut, Undercut is passed over until Trench has run, then runs."""
    experiment = make_experiment(
        tmp_path, requirements={"Undercut": ["Trench"]}, lamella_names=["L1"]
    )
    m = TaskManager(
        microscope=NoMicroscope(), experiment=experiment, parent_ui=RecordingUI()
    )
    m.review_enabled = True
    m.queue.build_from_pairs([("L1", "Undercut"), ("L1", "Trench")])
    lamella = experiment.get_lamella_by_name("L1")
    assert m._defer_reason(lamella, "Undercut") == "prereq_pending"
    assert m._should_skip(lamella, "Undercut") == "missing_prereqs"

    def record_completion(task_name, lam):
        lam.task_history.append(
            AutoLamellaTaskState(name=task_name, status=Status.Completed)
        )

    executed = run_queue_with(m, record_completion)
    assert executed == [("L1", "Trench"), ("L1", "Undercut")]
    assert all(i.status is Status.Completed for i in m.queue.items)


def _one_lamella_manager(tmp_path):
    experiment = make_experiment(
        tmp_path, requirements={"Undercut": ["Trench"]}, lamella_names=["L1"]
    )
    m = TaskManager(
        microscope=NoMicroscope(), experiment=experiment, parent_ui=RecordingUI()
    )
    m.review_enabled = True
    return m, experiment.get_lamella_by_name("L1")


def test_an_old_success_does_not_satisfy_a_requirement_whose_rerun_failed(tmp_path):
    """FIB-1006: the latest run of a requirement is the answer."""
    m, lamella = _one_lamella_manager(tmp_path)
    m.queue.build_from_matrix(["Undercut"], ["L1"])
    lamella.task_history.append(
        AutoLamellaTaskState(name="Trench", status=Status.Completed)
    )
    lamella.task_history.append(
        AutoLamellaTaskState(name="Trench", status=Status.Failed)
    )
    assert lamella.has_completed_task("Trench"), "the old success is still history"
    assert m._should_skip(lamella, "Undercut") == "missing_prereqs"


def test_an_old_success_does_not_satisfy_a_requirement_whose_rerun_was_rejected(
    tmp_path,
):
    from fibsem.applications.autolamella.proposals import (
        Decision,
        DecisionOutcome,
        Proposal,
    )

    m, lamella = _one_lamella_manager(tmp_path)
    m.queue.build_from_matrix(["Undercut"], ["L1"])
    lamella.task_history.append(
        AutoLamellaTaskState(name="Trench", status=Status.Completed)
    )
    lamella.task_history.append(
        AutoLamellaTaskState(name="Trench", status=Status.AwaitingDecision)
    )
    lamella.proposals["Trench"] = Proposal(
        kind="task_result", provenance={"task_id": "run-2"}
    )
    assert m._defer_reason(lamella, "Undercut") == "awaiting_decision"

    m.experiment._decide(
        lamella.id,
        "Trench",
        Decision(
            outcome=DecisionOutcome.Rejected,
            author="human:op",
            reason="no",
            task_id="run-2",
        ),
    )

    assert m._defer_reason(lamella, "Undercut") is None
    assert m._should_skip(lamella, "Undercut") == "missing_prereqs"


def test_a_queued_rerun_of_a_requirement_outranks_its_old_success(tmp_path):
    """Undercut is selected before Trench's rerun: it waits for the rerun, and
    is skipped when the rerun fails, whatever the earlier Trench did."""
    m, lamella = _one_lamella_manager(tmp_path)
    lamella.task_history.append(
        AutoLamellaTaskState(name="Trench", status=Status.Completed)
    )
    m.queue.build_from_pairs([("L1", "Undercut"), ("L1", "Trench")])
    assert m._defer_reason(lamella, "Undercut") == "prereq_pending"

    def rerun_fails(task_name, lam):
        status = Status.Failed if task_name == "Trench" else Status.Completed
        lam.task_history.append(AutoLamellaTaskState(name=task_name, status=status))

    executed = run_queue_with(m, rerun_fails)

    assert executed == [("L1", "Trench")], "Undercut waited, then was skipped"
    (undercut,) = [i for i in m.queue.items if i.task_name == "Undercut"]
    assert undercut.status is Status.Skipped


def test_a_task_awaiting_a_decision_defers_its_consumer(tmp_path):
    """Trench ran and waits on a decision: it is not finished, so Undercut waits.
    It stays pending -- not Skipped -- and runs once the decision lands, which
    finishes Trench."""
    from fibsem.applications.autolamella.proposals import (
        Decision,
        DecisionOutcome,
        Proposal,
    )

    experiment = make_experiment(
        tmp_path, requirements={"Undercut": ["Trench"]}, lamella_names=["L1", "L2"]
    )
    m = TaskManager(
        microscope=NoMicroscope(), experiment=experiment, parent_ui=RecordingUI()
    )
    m.review_enabled = True
    experiment.task_protocol.options.review_wait = 0  # do not park on the review
    m.queue.build_from_matrix(["Undercut"], ["L1", "L2"])
    l1 = experiment.get_lamella_by_name("L1")
    l1.task_history.append(
        AutoLamellaTaskState(name="Trench", status=Status.AwaitingDecision)
    )
    l1.proposals["Trench"] = Proposal(
        kind="task_result", provenance={"task_id": "run-1"}
    )
    experiment.get_lamella_by_name("L2").task_history.append(
        AutoLamellaTaskState(name="Trench", status=Status.Completed)
    )
    assert m._defer_reason(l1, "Undercut") == "awaiting_decision"

    executed = run_queue_with(m)
    assert executed == [("L2", "Undercut")], "L1 waits, L2 does not"
    assert m.queue.has_pending_pair("L1", "Undercut")
    assert [(i.lamella_name, r) for i, r in m.deferred_items()] == [
        ("L1", "awaiting_decision")
    ]

    experiment.decide(
        l1.id,
        "Trench",
        Decision(outcome=DecisionOutcome.Confirmed, author="human:op", task_id="run-1"),
    )
    assert l1.has_completed_task("Trench"), "the decision finished it"
    assert m._defer_reason(l1, "Undercut") is None
    assert run_queue_with(m) == [("L1", "Undercut")]


def test_with_the_flag_off_nothing_is_deferred(tmp_path):
    """The whole path is behind proposer_reviewer_workflow_enabled: off, the
    prerequisite check is terminal exactly as it was."""
    experiment = make_experiment(
        tmp_path, requirements={"Undercut": ["Trench"]}, lamella_names=["L1"]
    )
    m = TaskManager(
        microscope=NoMicroscope(), experiment=experiment, parent_ui=RecordingUI()
    )
    m.review_enabled = False
    m.queue.build_from_pairs([("L1", "Undercut"), ("L1", "Trench")])
    executed = run_queue_with(m)
    assert executed == [("L1", "Trench")]
    assert [i.status for i in m.queue.items] == [Status.Skipped, Status.Completed]


def test_the_flag_reads_fail_closed(monkeypatch):
    from fibsem.applications.autolamella.workflows.tasks import manager as M

    class _Prefs:
        class features:
            proposer_reviewer_workflow_enabled = True

    monkeypatch.setattr(M.fibsem_cfg, "load_user_preferences", lambda: _Prefs())
    assert M.review_enabled() is True

    def boom():
        raise OSError("unreadable")

    monkeypatch.setattr(M.fibsem_cfg, "load_user_preferences", boom)
    assert M.review_enabled() is False


def test_a_rejected_task_is_failed_so_its_consumer_is_skipped(tmp_path):
    """Reject finishes the waiting task as Failed. Its consumer is then Skipped
    for a missing prerequisite, the ordinary rule; the lamella itself is not
    marked defective -- that stays a person's call."""
    from fibsem.applications.autolamella.proposals import (
        Decision,
        DecisionOutcome,
        Proposal,
    )

    experiment = make_experiment(
        tmp_path, requirements={"Undercut": ["Trench"]}, lamella_names=["L1"]
    )
    m = TaskManager(
        microscope=NoMicroscope(), experiment=experiment, parent_ui=RecordingUI()
    )
    m.queue.build_from_matrix(["Undercut"], ["L1"])
    l1 = experiment.get_lamella_by_name("L1")
    l1.task_history.append(
        AutoLamellaTaskState(name="Trench", status=Status.AwaitingDecision)
    )
    l1.proposals["Trench"] = Proposal(
        kind="task_result", provenance={"task_id": "run-1"}
    )
    experiment.decide(
        l1.id,
        "Trench",
        Decision(
            outcome=DecisionOutcome.Rejected,
            author="human:op",
            reason="no site",
            task_id="run-1",
        ),
    )
    assert l1.task_history[-1].status is Status.Failed
    assert l1.task_history[-1].status_message == "Rejected by op: no site"
    assert not l1.is_failure
    assert run_queue_with(m) == []
    assert [i.status for i in m.queue.items] == [Status.Skipped]
    reports = [e.report for e in m.parent_ui.workflow_status_signal.emitted if e.report]
    assert reports[-1].skip_reason == "missing_prereqs"


# ── stalled: drained with work waiting on a decision ─────────────────────────


def _review_manager(tmp_path, review_wait, hook_manager=None):
    """L1 has Trench awaiting a decision; Undercut requires it."""
    from fibsem.applications.autolamella.proposals import Proposal

    experiment = make_experiment(
        tmp_path, requirements={"Undercut": ["Trench"]}, lamella_names=["L1"]
    )
    experiment.task_protocol.options.review_wait = review_wait
    m = TaskManager(
        microscope=NoMicroscope(),
        experiment=experiment,
        parent_ui=RecordingUI(),
        hook_manager=hook_manager,
    )
    m.review_enabled = True
    m.queue.build_from_matrix(["Undercut"], ["L1"])
    l1 = experiment.get_lamella_by_name("L1")
    l1.task_history.append(
        AutoLamellaTaskState(name="Trench", status=Status.AwaitingDecision)
    )
    l1.proposals["Trench"] = Proposal(
        kind="task_result", provenance={"task_id": "run-1"}
    )
    return m, l1


def test_a_run_that_drains_on_pending_reviews_is_stalled_not_completed(tmp_path):
    from fibsem.hooks import FunctionHook, HookEvent, HookManager

    fired = []
    hooks = HookManager()
    hooks.register(
        FunctionHook(name="rec", events=list(HookEvent), callback=fired.append)
    )
    m, l1 = _review_manager(tmp_path, review_wait=0, hook_manager=hooks)

    assert run_queue_with(m) == []
    assert m.stalled is True
    assert [c.event for c in fired] == ["workflow_started", "workflow_stalled"]
    assert fired[-1].decisions_pending == 1
    assert m.queue.has_pending_pair("L1", "Undercut"), "nothing was retired"
    assert l1.proposals["Trench"].pending


def test_a_decision_during_the_wait_wakes_the_run(tmp_path, caplog):
    import logging
    import threading

    from fibsem.applications.autolamella.proposals import Decision, DecisionOutcome

    m, l1 = _review_manager(tmp_path, review_wait=10.0)
    caplog.set_level(logging.INFO)

    def decide_soon():
        time.sleep(0.3)
        # _decide, the unmarshalled inner: with a QApplication in the process
        # (the UI suites share it) decide() would park on a main thread this
        # test holds in run_queue_with. The wake-up is the subject, not the
        # marshalling, which tests/ui/test_decide_main_thread.py covers.
        m.experiment._decide(
            l1.id,
            "Trench",
            Decision(
                outcome=DecisionOutcome.Confirmed, author="human:op", task_id="run-1"
            ),
        )

    t = threading.Thread(target=decide_soon, daemon=True)
    started = time.monotonic()
    t.start()
    assert run_queue_with(m) == [("L1", "Undercut")]
    assert time.monotonic() - started < 5.0, "woke on the decision, not the timeout"
    assert m.stalled is False
    # the one place "why is nothing happening" is a fair question says so
    messages = [r.message for r in caplog.records]
    assert any(
        m_.startswith("Parked: 1 task(s) wait on 1 decision(s) in the Review tab")
        and "L1/Undercut" in m_
        and "giving up after" in m_
        for m_ in messages
    ), messages
    assert any(m_.startswith("A decision landed after") for m_ in messages)


def test_the_wait_is_measured_as_inactivity_and_gives_up(tmp_path, caplog):
    import logging

    m, _l1 = _review_manager(tmp_path, review_wait=0.5)
    caplog.set_level(logging.INFO)
    started = time.monotonic()
    assert run_queue_with(m) == []
    elapsed = time.monotonic() - started
    assert 0.4 < elapsed < 3.0
    assert m.stalled is True
    assert "pending" in m.stall_reason
    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        "Timed out after" in w and "waiting for a review" in w for w in warnings
    ), warnings


def test_stop_ends_the_wait_as_cancelled(tmp_path):
    import threading

    m, _l1 = _review_manager(tmp_path, review_wait=None)  # wait forever
    threading.Timer(0.3, m.stop).start()
    assert run_queue_with(m) == []
    assert m.is_stopped
    assert m.stalled is False


def test_unrunnable_work_with_nothing_to_unblock_it_exits_with_an_error(tmp_path):
    """Deferred, but on a prerequisite that is neither queued nor under review:
    a plan bug, not a stall to wait on."""
    experiment = make_experiment(
        tmp_path, requirements={"Undercut": ["Trench"]}, lamella_names=["L1"]
    )
    m = TaskManager(
        microscope=NoMicroscope(), experiment=experiment, parent_ui=RecordingUI()
    )
    m.review_enabled = True
    m.queue.build_from_matrix(["Undercut"], ["L1"])
    # Make the scan defer it, then remove what it was waiting on.
    m._is_deferred = lambda item: True
    m.deferred_items = lambda: [(m.queue.pending[0], "prereq_pending")]
    started = time.monotonic()
    assert run_queue_with(m) == []
    assert time.monotonic() - started < 2.0
    assert m.stalled is True and "no decision would change that" in m.stall_reason


def test_run_leaves_out_a_task_awaiting_a_decision(tmp_path):
    """Run re-runs completed pairs -- that is how a task is re-run -- but a task
    whose record waits in the Review tab is left out: running it again would
    supersede the proposal someone is about to decide."""
    experiment = make_experiment(tmp_path, lamella_names=["L1", "L2"])
    m = TaskManager(
        microscope=NoMicroscope(), experiment=experiment, parent_ui=RecordingUI()
    )
    experiment.get_lamella_by_name("L1").task_history.append(
        AutoLamellaTaskState(name="Trench", status=Status.AwaitingDecision)
    )
    experiment.get_lamella_by_name("L2").task_history.append(
        AutoLamellaTaskState(name="Trench", status=Status.Completed)
    )
    m._run_queue = lambda: None
    m.review_enabled = True
    m.run(["Trench", "Undercut"], ["L1", "L2"])
    assert [(i.lamella_name, i.task_name) for i in m.queue.items] == [
        ("L2", "Trench"),
        ("L1", "Undercut"),
        ("L2", "Undercut"),
    ]
    # With the Review surface off nobody can decide, so the exception is not
    # made and Run re-runs it, as it re-runs any other completed pair.
    m.review_enabled = False
    m.run(["Trench", "Undercut"], ["L1", "L2"])
    assert len(m.queue.items) == 4


def test_no_requirements_runs(manager):
    assert (
        manager._should_skip(manager.experiment.get_lamella_by_name("L1"), "Trench")
        is None
    )


# ── _run_queue: mid-run queue edits ───────────────────────────────────────────


def test_baseline_run_executes_the_whole_matrix(manager):
    assert run_queue_with(manager) == [
        ("L1", "Trench"),
        ("L2", "Trench"),
        ("L1", "Undercut"),
        ("L2", "Undercut"),
    ]


def test_task_added_mid_run_executes(manager):
    """The end-to-end fix: an out-of-plan task used to raise out of the loop."""
    added = []

    def on_task(task_name, lamella):
        if not added:
            added.append(manager.queue.add("L1", "Polishing"))

    executed = run_queue_with(manager, on_task)
    assert ("L1", "Polishing") in executed


def test_lamella_added_mid_run_executes(manager):
    """An out-of-plan lamella used to be silently skipped as not_required.

    It still has to exist in the experiment: queueing work for a name the
    experiment has never heard of is skipped with a warning, which a fake
    experiment that conjured lamellas on lookup used to hide.
    """
    make_lamella(manager.experiment, "L99")
    added = []

    def on_task(task_name, lamella):
        if not added:
            added.append(manager.queue.add("L99", "Trench"))

    executed = run_queue_with(manager, on_task)
    assert ("L99", "Trench") in executed
    assert all(i.status is not Status.Skipped for i in manager.queue.items)


def test_item_moved_to_front_mid_run_runs_next(manager):
    moved = []

    def on_task(task_name, lamella):
        if not moved:
            last = manager.queue.pending[-1]
            manager.queue.move_to_front(last.id)
            moved.append((last.lamella_name, last.task_name))

    executed = run_queue_with(manager, on_task)
    assert executed[1] == moved[0]


def test_item_removed_mid_run_never_executes(manager):
    removed = []

    def on_task(task_name, lamella):
        if not removed:
            target = manager.queue.pending[-1]
            manager.queue.remove(target.id)
            removed.append((target.lamella_name, target.task_name))

    executed = run_queue_with(manager, on_task)
    assert removed[0] not in executed
    assert len(executed) == 3


def test_status_bar_text_is_derivable_for_added_items(manager):
    """What AutoLamellaMainUI builds its status string from."""
    added = []

    def on_task(task_name, lamella):
        if not added:
            added.append(manager.queue.add("L99", "Polishing"))

    run_queue_with(manager, on_task)
    for event in manager.parent_ui.workflow_status_signal.emitted:
        status = event.report
        if status is None:
            continue
        assert status.queue_position is not None
        assert 1 <= status.queue_position <= status.queue_total


def test_a_parked_run_holds_the_window_and_says_who_releases_it(tmp_path):
    """Parked on decisions, the manager hands the window one Hold: kind review,
    the lamella/task pairs waiting, and the sentence that releases it. Gone
    again when the park ends; the give-up says it gave up, on the status bar
    too, so it never reads as a finish."""
    from fibsem.applications.autolamella.workflows.tasks.status import HoldKind

    experiment = make_experiment(
        tmp_path, requirements={"Undercut": ["Trench"]}, lamella_names=["L1", "L2"]
    )
    ui = RecordingUI()
    m = TaskManager(microscope=NoMicroscope(), experiment=experiment, parent_ui=ui)
    m.review_enabled = True
    experiment.task_protocol.options.review_wait = 0.2
    m.queue.build_from_matrix(["Undercut"], ["L1", "L2"])
    for name in ("L1", "L2"):
        experiment.get_lamella_by_name(name).task_history.append(
            AutoLamellaTaskState(name="Trench", status=Status.AwaitingDecision)
        )
    holds = []
    real = m._set_hold
    m._set_hold = lambda hold: (holds.append(hold), real(hold))  # type: ignore

    run_queue_with(m)

    parked, released = holds
    assert parked.kind is HoldKind.review_later
    assert parked.items == ("L1/Undercut", "L2/Undercut")
    assert parked.releases == "decide L1 and L2 in the Review tab"
    assert released is None and ui.hold is None
    bars = [e.status_bar for e in ui.workflow_status_signal.emitted if e.status_bar]
    assert bars[0] == "Parked on 2 decision(s): decide L1 and L2 in the Review tab."
    assert bars[-1].startswith("Workflow stalled: Timed out after")
    assert "waiting for a review: 2 decision(s) still pending." in bars[-1]
    assert bars[-1].endswith("Decide in the Review tab, then Run again.")
    assert m.stalled and m.closing_note() == bars[-1][len("Workflow stalled: ") :]


def test_the_names_a_hold_reads_out():
    from fibsem.applications.autolamella.workflows.tasks.manager import _named

    assert _named([]) == ""
    assert _named(["L1"]) == "L1"
    assert _named(["L1", "L2"]) == "L1 and L2"
    assert _named(["L1", "L2", "L3"]) == "L1, L2 and L3"
    assert _named(["L1", "L2", "L3", "L4"]) == "4 lamellae"
