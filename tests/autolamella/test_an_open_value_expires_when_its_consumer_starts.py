"""An automated task's value is open until the task that uses it starts.

Nobody was asked, so nobody confirms it: the value is live from the moment
it is proposed and can be corrected in the Review tab until then. When its
consumer starts it is recorded ``Unreviewed``: used as it stood, never
agreement. A run's end closes nothing: a result is consumed by nothing, so
it stays open until a person looks. A question a task is parked on, and a
supervised task's result the run is holding for, are not open and are never
expired.
"""

from pathlib import Path

import pytest

from fibsem.applications.autolamella.proposals import (
    POINT_OF_INTEREST,
    TASK_RESULT,
    Decision,
    DecisionOutcome,
    Proposal,
)
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskState,
    AutoLamellaTaskStatus,
    Experiment,
)
from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
from fibsem.structures import Point
from tests.autolamella.test_task_manager_status import (
    NoMicroscope,
    RecordingUI,
    make_experiment,
    run_queue_with,
)

RUN = "run-1"


def _proposal(kind=TASK_RESULT, **values) -> Proposal:
    return Proposal(kind=kind, values=values, provenance={"task_id": RUN})


@pytest.fixture
def experiment(tmp_path: Path) -> Experiment:
    return make_experiment(
        tmp_path, requirements={"Undercut": ["Trench"]}, lamella_names=["L1"]
    )


def test_an_open_value_is_listed_to_check_not_waiting(experiment):
    lamella = experiment.positions[0]
    proposal = lamella.record_proposal("Trench", _proposal())
    assert experiment.pending_proposals() == []
    assert [p for _i, _t, p in experiment.proposals_to_check()] == [proposal]


def test_expiring_closes_it_unreviewed_with_the_values_as_proposed(experiment):
    lamella = experiment.positions[0]
    proposal = lamella.record_proposal(
        "Trench", _proposal(POINT_OF_INTEREST, poi=Point(1e-6, 0))
    )
    heard = []
    experiment.decided.connect(lambda i, t: heard.append(t))

    assert experiment.expire_open(lamella.id, "Trench", "Undercut started") == 1

    d = proposal.current
    assert d.outcome is DecisionOutcome.Unreviewed
    assert d.values == {"poi": Point(1e-6, 0)} and d.reason == "Undercut started"
    assert d.proposal_id == proposal.id and d.task_id == RUN
    assert proposal.to_check and heard == ["Trench"]
    assert experiment.expire_open(lamella.id, "Trench", "again") == 0, "once"


def test_a_question_the_task_is_parked_on_is_not_expired(experiment):
    lamella = experiment.positions[0]
    proposal = lamella.record_proposal("Trench", _proposal())
    proposal.asking = True
    assert experiment.expire_open(lamella.id, "Trench", "x") == 0
    assert proposal.pending


def test_a_result_the_run_is_holding_for_is_not_expired(experiment):
    lamella = experiment.positions[0]
    lamella.task_history.append(
        AutoLamellaTaskState(
            name="Trench", task_id=RUN, status=AutoLamellaTaskStatus.AwaitingDecision
        )
    )
    proposal = lamella.record_proposal("Trench", _proposal())
    assert [p for _i, _t, p in experiment.pending_proposals()] == [proposal]
    assert experiment.expire_all_open("x") == 0
    assert proposal.pending


def test_a_correction_after_expiry_is_refused_as_a_late_edit(experiment):
    lamella = experiment.positions[0]
    proposal = lamella.record_proposal(
        "Trench", _proposal(POINT_OF_INTEREST, poi=Point(0, 0))
    )
    experiment.expire_open(lamella.id, "Trench", "Undercut started")
    late = experiment.decide(
        lamella.id,
        "Trench",
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={"poi": Point(1e-6, 0)},
            proposal_id=proposal.id,
        ),
    )
    assert not late.applied and "already decided" in late.reason
    look = experiment.decide(
        lamella.id,
        "Trench",
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            proposal_id=proposal.id,
        ),
    )
    assert look.applied and not proposal.to_check


def test_the_manager_expires_what_a_task_consumes_when_it_starts(tmp_path):
    """Trench leaves an open value; Undercut requires Trench, so the moment
    Undercut starts the value is used as it stands and recorded so."""
    experiment = make_experiment(
        tmp_path, requirements={"Undercut": ["Trench"]}, lamella_names=["L1"]
    )
    lamella = experiment.positions[0]
    m = TaskManager(
        microscope=NoMicroscope(), experiment=experiment, parent_ui=RecordingUI()
    )
    m.queue.build_from_matrix(["Trench", "Undercut"], ["L1"])
    seen = {}

    def on_task(task_name, lam):
        if task_name == "Trench":
            lam.record_proposal("Trench", _proposal())
            # what a real run leaves: the completed entry the requirement reads
            lam.task_history.append(
                AutoLamellaTaskState(
                    name="Trench", task_id=RUN, status=AutoLamellaTaskStatus.Completed
                )
            )
        if task_name == "Undercut":
            seen["trench_when_undercut_started"] = lam.proposal("Trench").current

    run_queue_with(m, on_task=on_task)

    d = seen["trench_when_undercut_started"]
    assert d is not None and d.outcome is DecisionOutcome.Unreviewed
    assert "Undercut started" in d.reason


def test_a_task_downstream_through_another_consumes_it_too(tmp_path):
    """Rough Milling requires Mill Fiducial, which requires Setup, and it is
    Setup's point that Rough Milling mills on. The point is closed when the
    downstream task starts, not only when the task that names Setup does."""
    experiment = make_experiment(
        tmp_path,
        requirements={"Undercut": ["Trench"], "Polishing": ["Undercut"]},
        lamella_names=["L1"],
    )
    lamella = experiment.positions[0]
    m = TaskManager(
        microscope=NoMicroscope(), experiment=experiment, parent_ui=RecordingUI()
    )
    lamella.record_proposal("Trench", _proposal(POINT_OF_INTEREST, poi=Point(0, 0)))
    for name in ("Trench", "Undercut"):
        lamella.task_history.append(
            AutoLamellaTaskState(
                name=name, task_id=RUN, status=AutoLamellaTaskStatus.Completed
            )
        )
    m.queue.build_from_matrix(["Polishing"], ["L1"])
    seen = {}

    def on_task(task_name, lam):
        seen["trench_when_polishing_started"] = lam.proposal("Trench").current

    run_queue_with(m, on_task=on_task)

    d = seen["trench_when_polishing_started"]
    assert d is not None and d.outcome is DecisionOutcome.Unreviewed
    assert "Polishing started" in d.reason
    assert m._upstream_of("Polishing") == ["Undercut", "Trench"]


def test_when_the_run_ends_a_result_and_a_value_both_stay_open(tmp_path):
    """Nothing consumes a result and nothing waits on it, so it is open until
    a person looks, however many runs end. A value is for the task that uses
    it, whichever run that is: Setup run on its own leaves its point open to
    correct before Rough Milling is run, which is the point of running it on
    its own."""
    experiment = make_experiment(
        tmp_path, requirements={"Undercut": ["Trench"]}, lamella_names=["L1"]
    )
    lamella = experiment.positions[0]
    m = TaskManager(
        microscope=NoMicroscope(), experiment=experiment, parent_ui=RecordingUI()
    )
    m.queue.build_from_matrix(["Trench"], ["L1"])

    def on_task(name, lam):
        lam.record_proposal("Trench", _proposal(POINT_OF_INTEREST, poi=Point(0, 0)))
        lam.record_proposal("Trench", _proposal())

    run_queue_with(m, on_task=on_task)

    point, result = lamella.current_proposals("Trench")
    assert result.kind == TASK_RESULT and result.pending, "still open"
    assert point.kind == POINT_OF_INTEREST and point.pending, "still open"
    for proposal in (point, result):
        assert experiment._is_open(lamella, "Trench", proposal)
    assert [p for _i, _t, p in experiment.proposals_to_check()] == [point, result]

    # A correction now is a plain confirm with values, as during the run.
    moved = experiment.decide(
        lamella.id,
        "Trench",
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={"poi": Point(1e-6, 0)},
            proposal_id=point.id,
        ),
    )
    assert moved.applied, moved.reason
    assert lamella.poi == Point(1e-6, 0)
