"""Questions and decisions on the experiment's record, in the event stream
(FIB-1034).

The app's own ``EventRecorder`` on Demo, watching a real ``Experiment``. Every
decision is made through the experiment's own methods -- ``decide`` for a
person or an agent, ``expire_open``, ``withdraw_proposal`` and
``record_unasked`` for the record itself -- and read back from the buffer.
"""

import os

import pytest
from psygnal.containers import EventedDict

from fibsem import utils
from fibsem.acting import OPERATOR
from fibsem.applications.autolamella.event_recording import EventRecorder
from fibsem.applications.autolamella.proposals import (
    POINT_OF_INTEREST,
    STATE,
    Decision,
    DecisionOutcome,
    Proposal,
)
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    AutoLamellaTaskStatus,
    Experiment,
)
from fibsem.structures import FibsemStagePosition, MicroscopeState, Point

TASK = "Setup Lamella Position"
RUN = "run-1"


@pytest.fixture(scope="module")
def microscope():
    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    yield microscope
    microscope.disconnect()


@pytest.fixture
def experiment(tmp_path):
    exp = Experiment(path=tmp_path, name="decisions")
    os.makedirs(exp.path, exist_ok=True)
    exp.task_protocol = AutoLamellaTaskProtocol()
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    lamella = exp.positions[0]
    lamella.task_state.name = TASK
    lamella.task_state.task_id = RUN
    lamella.task_state.status = AutoLamellaTaskStatus.Completed
    return exp


@pytest.fixture
def recorder(microscope, experiment):
    recorder = EventRecorder(microscope, default_actor=OPERATOR, experiment=experiment)
    yield recorder
    recorder.close()


def _events(recorder, *kinds):
    kinds = kinds or ("proposal_asked", "proposal_decided")
    return [e for e in recorder.buffer.events_since(0)["events"] if e["kind"] in kinds]


def _poi_proposal(lamella, x=0.0):
    """Setup's point of interest, left for the Review tab after its run."""
    proposal = Proposal(
        kind=POINT_OF_INTEREST,
        values={"poi": Point(x, 0.0)},
        provenance={"task_id": RUN, "proposer": "current-poi"},
    )
    lamella.proposals[TASK] = [proposal]
    return proposal


def _decide(experiment, proposal, author, values=None, via="review"):
    result = experiment.decide(
        experiment.positions[0].id,
        TASK,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author=author,
            values=values or {},
            via=via,
            task_id=RUN,
            proposal_id=proposal.id,
        ),
    )
    assert result.applied, result.reason
    return result


def test_a_decision_is_recorded_with_what_was_proposed_and_what_was_decided(
    experiment, recorder
):
    lamella = experiment.positions[0]
    proposal = _poi_proposal(lamella)

    _decide(experiment, proposal, "human:op", {"poi": Point(2e-6, -1e-6)})

    ((event),) = _events(recorder)
    assert (event["kind"], event["actor"]) == ("proposal_decided", "operator")
    payload = event["payload"]
    assert payload["item"] == {"id": lamella.id, "name": lamella.name}
    assert (payload["task"], payload["kind"]) == (TASK, POINT_OF_INTEREST)
    assert (payload["proposal_id"], payload["decision"]) == (proposal.id, 0)
    assert payload["proposed"]["poi"] == Point(0.0, 0.0).to_dict()
    assert payload["decided"]["poi"] == Point(2e-6, -1e-6).to_dict()
    assert (payload["outcome"], payload["author"], payload["via"]) == (
        "Confirmed",
        "human:op",
        "review",
    )
    assert "filled_in" not in payload
    assert lamella.poi.x == 2e-6  # and the decision still did what it does


def test_an_agent_s_decision_is_the_agent_s(experiment, recorder):
    proposal = _poi_proposal(experiment.positions[0])

    _decide(experiment, proposal, "agent:a-model", {"poi": Point(1e-6, 0.0)}, "server")

    ((event),) = _events(recorder)
    assert event["actor"] == "agent"
    assert (event["payload"]["author"], event["payload"]["via"]) == (
        "agent:a-model",
        "server",
    )


def test_a_question_is_recorded_and_so_is_the_position_it_was_confirmed_at(
    experiment, recorder
):
    """A state asked mid-run, confirmed "as it stands" from the prompt bar:
    the task fills in the position afterwards, and that is recorded too."""
    lamella = experiment.positions[0]
    lamella.task_state.status = AutoLamellaTaskStatus.InProgress
    asked_at = FibsemStagePosition(x=1e-3, y=0, z=0, r=0, t=0, name="asked")
    moved_to = FibsemStagePosition(x=1.002e-3, y=0, z=0, r=0, t=0, name="moved")
    proposal = Proposal(
        kind=STATE,
        values={"stage_position": asked_at},
        provenance={"task_id": RUN, "proposer": TASK, "message": "Confirm it"},
    )
    assert experiment.ask_proposal(lamella.id, TASK, proposal)

    _decide(experiment, proposal, "human:op", via="workflow")
    assert experiment.fill_in_decision(
        lamella.id, TASK, proposal.id, {"stage_position": moved_to}
    )

    asked, confirmed, filled = _events(recorder)
    assert (asked["kind"], asked["actor"]) == ("proposal_asked", "task")
    assert asked["payload"]["message"] == "Confirm it"
    assert asked["payload"]["proposed"]["stage_position"]["x"] == 1e-3
    assert (confirmed["kind"], confirmed["actor"]) == ("proposal_decided", "operator")
    assert confirmed["payload"]["decided"] == {}
    assert (filled["kind"], filled["actor"]) == ("proposal_decided", "operator")
    assert filled["payload"]["filled_in"] is True
    assert filled["payload"]["decision"] == confirmed["payload"]["decision"] == 0
    assert filled["payload"]["decided"]["stage_position"]["x"] == 1.002e-3


def test_what_nobody_decided_is_recorded_as_the_task_s(experiment, recorder):
    lamella = experiment.positions[0]
    # An automated value its consumer started on before anyone looked.
    open_one = _poi_proposal(lamella)
    assert experiment.expire_open(lamella.id, TASK, "Mill Rough started") == 1
    # A question nobody was asked: the task is not supervised.
    unasked = Proposal(
        kind=STATE,
        values={"stage_position": FibsemStagePosition(x=0, y=0, z=0, r=0, t=0)},
        provenance={"task_id": RUN, "proposer": TASK},
    )
    assert experiment.record_unasked(
        lamella.id, "Acquire Reference Image", unasked, "not supervised"
    )
    # A question whose run stopped before it was answered.
    lamella.task_state.status = AutoLamellaTaskStatus.InProgress
    withdrawn = Proposal(kind=STATE, values={}, provenance={"task_id": RUN})
    assert experiment.ask_proposal(lamella.id, TASK, withdrawn)
    assert experiment.withdraw_proposal(lamella.id, TASK, "the run stopped").applied

    decided = [e for e in _events(recorder) if e["kind"] == "proposal_decided"]
    assert [
        (e["payload"]["proposal_id"], e["payload"]["outcome"], e["actor"])
        for e in decided
    ] == [
        (open_one.id, "Unreviewed", "task"),
        (unasked.id, "Unreviewed", "task"),
        (withdrawn.id, "Withdrawn", "task"),
    ]
    unreviewed = decided[0]["payload"]
    assert unreviewed["decided"] == unreviewed["proposed"]  # used as proposed
    assert unreviewed["reason"] == "Mill Rough started"


def test_what_the_experiment_held_when_watched_is_not_recorded_again(
    microscope, experiment
):
    lamella = experiment.positions[0]
    earlier = _poi_proposal(lamella)
    _decide(experiment, earlier, "human:op", {"poi": Point(1e-6, 0.0)})
    recorder = EventRecorder(microscope, experiment=experiment)
    try:
        # A second look at the same proposal: its first decision was made
        # before the recorder watched, and is not new.
        _decide(experiment, earlier, "human:someone-else")

        ((event),) = _events(recorder)
        assert (event["payload"]["decision"], event["payload"]["author"]) == (
            1,
            "human:someone-else",
        )
    finally:
        recorder.close()


def test_the_experiment_watched_is_the_one_set(microscope, experiment, tmp_path):
    other = Experiment(path=tmp_path / "other", name="other")
    os.makedirs(other.path, exist_ok=True)
    recorder = EventRecorder(microscope, experiment=other)
    try:
        recorder.set_experiment(experiment.path, experiment)
        proposal = _poi_proposal(experiment.positions[0])
        _decide(experiment, proposal, "human:op", {"poi": Point(1e-6, 0.0)})
        assert len(_events(recorder)) == 1

        recorder.set_experiment(None)
        _decide(experiment, proposal, "human:op")  # a second look
        assert len(_events(recorder)) == 1, "recorded after it stopped watching"
    finally:
        recorder.close()


def test_a_decision_that_cannot_be_recorded_is_still_made(
    experiment, recorder, monkeypatch
):
    from fibsem.applications.autolamella import event_recording

    def fail(values):
        raise RuntimeError("cannot encode")

    monkeypatch.setattr(event_recording, "_encode_values", fail)
    lamella = experiment.positions[0]
    proposal = _poi_proposal(lamella)

    _decide(experiment, proposal, "human:op", {"poi": Point(3e-6, 0.0)})

    assert _events(recorder) == []
    assert lamella.poi.x == 3e-6
    assert proposal.current.values["poi"].x == 3e-6
