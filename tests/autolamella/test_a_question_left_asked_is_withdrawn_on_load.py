"""A question the task was waiting on when the app closed is withdrawn when
the experiment loads: nothing is waiting on it any more (FIB-1046).

``asking`` is the live flag and is not saved. The fact that the question was
put to someone is: ``ask_proposal`` stamps ``provenance["asked"]``. A pending
proposal with that stamp on disk was a question whose task is gone, so on
load it is withdrawn, lists as decided, and a decision on it is refused as
stale. A pending value nobody was asked about -- an automated task's point,
open to correct -- loads exactly as it was. Loading twice appends nothing.
"""

import os
from pathlib import Path

import pytest
from psygnal.containers import EventedDict

from fibsem.applications.autolamella.proposals import (
    DETECTION,
    POINT_OF_INTEREST,
    Decision,
    DecisionOutcome,
    Proposal,
    Standing,
)
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    AutoLamellaTaskStatus,
    Experiment,
    standing,
)
from fibsem.structures import MicroscopeState, Point

SETUP = "Setup Lamella Position"
ROUGH = "Mill Rough"


@pytest.fixture
def experiment(tmp_path):
    exp = Experiment(path=tmp_path, name="load-exp")
    os.makedirs(exp.path, exist_ok=True)
    exp.task_protocol = AutoLamellaTaskProtocol()
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    return exp


def _saved_and_loaded(experiment) -> Experiment:
    experiment.save()
    return Experiment.load(Path(experiment.path) / "experiment.yaml")


def _ask(experiment, task_name=ROUGH) -> Proposal:
    lamella = experiment.positions[0]
    lamella.task_state.name = task_name
    lamella.task_state.task_id = "run-1"
    lamella.task_state.status = AutoLamellaTaskStatus.InProgress
    question = Proposal(
        kind=DETECTION,
        values={"features": [{"name": "LamellaCentre", "px": Point(10, 20)}]},
        provenance={"task_id": "run-1", "proposer": task_name},
    )
    assert experiment.ask_proposal(lamella.id, task_name, question)
    return question


def test_a_question_left_asked_loads_withdrawn(experiment):
    question = _ask(experiment)
    assert question.asking and question.provenance["asked"] is True

    again = _saved_and_loaded(experiment)
    lamella = again.positions[0]
    back = lamella.proposal(ROUGH)

    assert back.id == question.id
    assert back.withdrawn and not back.asking
    assert back.current.reason == "the app closed before it was answered"
    assert standing(lamella, ROUGH, back) is Standing.Closed
    assert again.pending_proposals() == [] and again.proposals_to_check() == []


def test_a_decision_on_it_is_refused_as_stale(experiment):
    question = _ask(experiment)
    again = _saved_and_loaded(experiment)
    lamella = again.positions[0]
    lamella.task_state.status = AutoLamellaTaskStatus.Completed

    result = again.decide(
        lamella.id,
        ROUGH,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values=dict(question.values),
            proposal_id=question.id,
        ),
    )

    assert not result.applied and result.error_type == "stale_review", result


def test_a_value_nobody_was_asked_about_loads_as_it_was(experiment):
    """An automated task's point: pending, open to correct, no ``asked``."""
    lamella = experiment.positions[0]
    point = Proposal(
        kind=POINT_OF_INTEREST,
        values={"poi": Point(0.0, 0.0)},
        provenance={"task_id": "run-1", "proposer": "current-poi"},
    )
    lamella.record_proposal(SETUP, point)

    again = _saved_and_loaded(experiment)
    back = again.positions[0].proposal(SETUP)

    assert back.pending and standing(again.positions[0], SETUP, back) is Standing.Open


def test_loading_twice_appends_nothing(experiment):
    _ask(experiment)
    once = _saved_and_loaded(experiment)
    twice = _saved_and_loaded(once)

    assert len(twice.positions[0].proposal(ROUGH).decisions) == 1
