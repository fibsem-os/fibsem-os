"""An automated task's value is open until the task that uses it starts.

Nobody was asked, so nobody confirms it: the value is live from the moment
it is proposed and can be corrected in the Review tab until then. When its
consumer starts -- or the run ends with nothing to consume it -- it is
recorded ``Unreviewed``: used as it stood, never agreement. A question a
task is parked on, and a supervised task's result the run is holding for,
are not open and are never expired.
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
    assert experiment.expire_all_open("the run ended") == 0
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


def test_the_manager_expires_what_nothing_consumed_when_the_run_ends(tmp_path):
    experiment = make_experiment(tmp_path, lamella_names=["L1"])
    lamella = experiment.positions[0]
    m = TaskManager(
        microscope=NoMicroscope(), experiment=experiment, parent_ui=RecordingUI()
    )
    m.queue.build_from_matrix(["Polishing"], ["L1"])
    run_queue_with(
        m, on_task=lambda name, lam: lam.record_proposal("Polishing", _proposal())
    )
    d = lamella.proposal("Polishing").current
    assert d is not None and d.outcome is DecisionOutcome.Unreviewed
    assert "the run ended" in d.reason
