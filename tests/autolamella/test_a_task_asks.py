"""A task asks for a value it needs now with ``ask``: one way, on the record.

Headless here. Automated, nobody is asked: the proposal goes on the record with
an ``Unreviewed`` decision carrying the values the task went on to use.
Supervised, the task asks and waits as it would at a window, whoever can
answer, nobody included: an answer is never taken unreviewed. The operator at a window is in tests/ui.
"""

import os
import threading
import time
from pathlib import Path

import pytest
import yaml
from psygnal.containers import EventedDict

import fibsem.config as cfg
from fibsem import utils
from fibsem.acting import ActorKind
from fibsem.applications.autolamella.proposals import POINT_OF_INTEREST
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskDescription,
    AutoLamellaTaskProtocol,
    AutoLamellaWorkflowConfig,
    Experiment,
)
from fibsem.applications.autolamella.workflows.tasks import get_tasks
from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
from fibsem.applications.autolamella.workflows.tasks.rough import (
    MillRoughTask,
    MillRoughTaskConfig,
)
from fibsem.cancellation import StopEvent
from fibsem.structures import Point
from fibsem.workflows.attention import Attention
from fibsem.workflows.proposals import (
    PROPOSAL_KINDS,
    Decision,
    DecisionOutcome,
)

ROUGH = "Rough Milling"
CONFIG = os.path.join(cfg.CONFIG_PATH, "microscope-configuration.yaml")


class _AsksForAPoint(MillRoughTask):
    """A real task type that declares one question."""

    questions = (POINT_OF_INTEREST,)


class _AsksButTurnedOff(_AsksForAPoint):
    @classmethod
    def questions_for(cls, config):
        return ()


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo", config_path=CONFIG)
    yield microscope
    microscope.disconnect()


def _experiment(tmp_path: Path, microscope, attention=Attention.automated):
    exp = Experiment(path=tmp_path, name="ask-exp")
    exp.task_protocol = AutoLamellaTaskProtocol(
        workflow_config=AutoLamellaWorkflowConfig(
            tasks=[
                AutoLamellaTaskDescription(
                    name=ROUGH, required=True, attention=attention
                )
            ]
        )
    )
    os.makedirs(exp.path, exist_ok=True)
    exp.add_new_lamella(
        microscope.get_microscope_state(),
        EventedDict({ROUGH: MillRoughTaskConfig(task_name=ROUGH)}),
    )
    exp.positions[0].path.mkdir(parents=True, exist_ok=True)
    return exp


def _task(microscope, exp, cls=_AsksForAPoint, body=None, answerers=None):
    manager = TaskManager(microscope=microscope, experiment=exp, answerers=answerers)
    lamella = exp.positions[0]
    task = cls(
        microscope=microscope,
        config=lamella.task_config[ROUGH],
        lamella=lamella,
        task_manager=manager,
    )
    if body is not None:
        task._run = lambda: body(task)  # type: ignore[method-assign]
    return task


def test_an_undeclared_kind_is_refused_and_nothing_is_recorded(microscope, tmp_path):
    exp = _experiment(tmp_path, microscope)
    task = _task(microscope, exp, cls=MillRoughTask)
    with pytest.raises(ValueError, match="does not declare"):
        task.ask(POINT_OF_INTEREST, {"poi": Point(0, 0)})
    assert exp.positions[0].proposals == []


def test_a_value_the_kind_does_not_carry_is_refused(microscope, tmp_path):
    exp = _experiment(tmp_path, microscope)
    with pytest.raises(ValueError, match="does not carry"):
        _task(microscope, exp).ask(POINT_OF_INTEREST, {"nope": 1})


def test_with_nobody_to_ask_the_proposed_value_is_used_and_recorded_unreviewed(
    microscope, tmp_path
):
    """Automated: the answer is what was proposed, and the record says
    nobody looked."""
    exp = _experiment(tmp_path, microscope)
    lamella = exp.positions[0]
    seen = {}

    def body(task):
        seen["decision"] = task.ask(
            POINT_OF_INTEREST, {"poi": Point(1e-6, 2e-6)}, image="ref.tif"
        )

    _task(microscope, exp, body=body).run()

    decision = seen["decision"]
    assert decision.outcome is DecisionOutcome.Unreviewed
    assert decision.values == {"poi": Point(1e-6, 2e-6)}, "used as proposed"
    assert decision.actor.kind is ActorKind.task
    # On the record, before the run's own result, and decided already.
    question, result = lamella.proposals_from(ROUGH)
    assert question.kind == POINT_OF_INTEREST and question.unreviewed
    assert question.provenance["reference_image"] == "ref.tif"
    assert question.provenance["task_id"] == result.provenance["task_id"]
    assert not question.pending
    assert exp.pending_proposals() == [], "nobody is asked about it"
    assert lamella.latest_run_completed(ROUGH), "nothing waits on it"


def test_with_the_agent_to_answer_a_supervised_task_asks_and_waits(
    microscope, tmp_path
):
    """No window, but the agent can answer (at Drive): the question stands,
    pending, and the task carries on with the agent's decision."""
    exp = _experiment(tmp_path, microscope, attention=Attention.supervised)
    lamella = exp.positions[0]
    task = _task(microscope, exp, answerers=lambda: (ActorKind.agent,))
    seen = {}
    asking = threading.Thread(
        target=lambda: seen.update(
            decision=task.ask(POINT_OF_INTEREST, {"poi": Point(1e-6, 2e-6)})
        ),
        daemon=True,
    )
    asking.start()
    deadline = time.monotonic() + 10
    while (question := lamella.proposal(ROUGH)) is None or not question.pending:
        assert time.monotonic() < deadline, "the question was never asked"
        assert asking.is_alive(), "the task did not wait for an answer"
        time.sleep(0.01)
    assert question.pending and question.asking, "the task waits on it"

    result = exp.decide(
        lamella.id,
        ROUGH,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            actor="agent:x",
            proposal_id=question.id,
            values={"poi": Point(3e-6, 0)},
        ),
    )
    assert result.applied, result.reason
    asking.join(timeout=10)
    assert not asking.is_alive(), "the task went on once it was answered"

    decision = seen["decision"]
    assert decision.outcome is DecisionOutcome.Confirmed
    assert decision.values == {"poi": Point(3e-6, 0)}, "the agent's value"
    assert decision.actor.kind is ActorKind.agent


def _asking(task, values):
    """``task`` asks on a thread of its own; returns (thread, seen)."""
    seen = {}
    asking = threading.Thread(
        target=lambda: seen.update(decision=task.ask(POINT_OF_INTEREST, values)),
        daemon=True,
    )
    asking.start()
    return asking, seen


def _asked(lamella, asking):
    deadline = time.monotonic() + 10
    while (question := lamella.proposal(ROUGH)) is None or not question.pending:
        assert time.monotonic() < deadline, "the question was never asked"
        assert asking.is_alive(), "the task did not wait for an answer"
        time.sleep(0.01)
    return question


def test_with_nobody_to_answer_a_supervised_task_still_waits(microscope, tmp_path):
    """No window and no agent listening: the question waits for someone to
    come back, and the hold says who can answer as that changes."""
    exp = _experiment(tmp_path, microscope, attention=Attention.supervised)
    lamella = exp.positions[0]
    who = []
    task = _task(microscope, exp, answerers=lambda: tuple(who))
    holds = []
    task.task_manager._set_hold = lambda hold, note=None, message="": holds.append(
        (hold, note)
    )
    asking, seen = _asking(task, {"poi": Point(1e-6, 2e-6)})
    question = _asked(lamella, asking)

    time.sleep(0.3)
    assert asking.is_alive(), "nobody answered, so it is still waiting"
    assert question.pending and question.asking
    hold, note = holds[0]
    assert "nobody can answer" in hold.releases
    assert "connect the agent at Drive" in note

    who.append(ActorKind.agent)
    deadline = time.monotonic() + 10
    while len(holds) < 2:
        assert time.monotonic() < deadline, "the hold never said the agent is back"
        time.sleep(0.01)
    assert (
        holds[1][1] == f"{ROUGH} is waiting for the agent to answer on {lamella.name}."
    )

    result = exp.decide(
        lamella.id,
        ROUGH,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            actor="agent:x",
            proposal_id=question.id,
            values={"poi": Point(3e-6, 0)},
        ),
    )
    assert result.applied, result.reason
    asking.join(timeout=10)
    assert not asking.is_alive(), "the task went on once it was answered"
    assert seen["decision"].outcome is DecisionOutcome.Confirmed
    assert holds[-1] == (None, None), "and the hold came down"


def test_stop_withdraws_a_question_nobody_can_answer(microscope, tmp_path):
    exp = _experiment(tmp_path, microscope, attention=Attention.supervised)
    lamella = exp.positions[0]
    task = _task(microscope, exp, answerers=lambda: ())
    task._stop_event = StopEvent()
    errors = []

    def ask():
        try:
            task.ask(POINT_OF_INTEREST, {"poi": Point(0, 0)})
        except InterruptedError as error:
            errors.append(error)

    asking = threading.Thread(target=ask, daemon=True)
    asking.start()
    question = _asked(lamella, asking)
    task._stop_event.set()
    asking.join(timeout=10)

    assert errors, "Stop ends the wait"
    assert question.current.outcome is DecisionOutcome.Withdrawn


def test_a_question_a_setting_turns_off_is_recorded_not_asked(microscope, tmp_path):
    exp = _experiment(tmp_path, microscope, attention=Attention.supervised)
    decision = _task(microscope, exp, cls=_AsksButTurnedOff).ask(
        POINT_OF_INTEREST, {"poi": Point(0, 0)}
    )
    assert decision.outcome is DecisionOutcome.Unreviewed
    assert "turned off" in decision.reason


def test_unreviewed_is_not_something_a_decider_can_say(microscope, tmp_path):
    exp = _experiment(tmp_path, microscope)
    lamella = exp.positions[0]
    _task(microscope, exp).ask(POINT_OF_INTEREST, {"poi": Point(0, 0)})
    proposal = lamella.proposal(ROUGH)
    result = exp.decide(
        lamella.id,
        ROUGH,
        Decision(
            outcome=DecisionOutcome.Unreviewed,
            actor="operator:op",
            proposal_id=proposal.id,
        ),
    )
    assert not result.applied and result.error_type == "invalid_value"


def test_unreviewed_survives_the_file(microscope, tmp_path):
    exp = _experiment(tmp_path, microscope)
    _task(microscope, exp).ask(POINT_OF_INTEREST, {"poi": Point(3e-6, 0)})
    exp.save()
    again = Experiment.load(str(Path(exp.path, "experiment.yaml")))
    proposal = again.positions[0].proposal(ROUGH)
    assert proposal.unreviewed and proposal.current.values["poi"] == Point(3e-6, 0)
    raw = yaml.safe_load(Path(exp.path, "experiment.yaml").read_text())
    (saved,) = raw["positions"][0]["proposals"]
    assert saved["asked_by"] == f"task:{ROUGH}"
    assert saved["decisions"][0]["outcome"] == "Unreviewed"


def test_every_task_type_declares_kinds_the_record_knows():
    """A task says what it asks; what it says has to be a proposal kind."""
    for name, cls in get_tasks().items():
        for kind in cls.questions:
            assert kind in PROPOSAL_KINDS, f"{name} asks an unknown kind {kind!r}"


def test_a_question_and_the_runs_result_are_both_current(microscope, tmp_path):
    """Different kinds do not replace each other: the unreviewed question
    stays current beside the run's result, both decided Unreviewed when the
    run ends, and a later decision on either is refused as already decided."""
    exp = _experiment(tmp_path, microscope)
    lamella = exp.positions[0]
    _task(
        microscope,
        exp,
        body=lambda t: t.ask(POINT_OF_INTEREST, {"poi": Point(0, 0)}),
    ).run()
    question, result = lamella.proposals_from(ROUGH)
    assert lamella.current_proposals(ROUGH) == [question, result]
    assert question.unreviewed and result.unreviewed
    assert exp.pending_proposals() == []

    looked = exp.decide(
        lamella.id,
        ROUGH,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            actor="operator:op",
            proposal_id=question.id,
        ),
    )
    assert not looked.applied and looked.error_type == "invalid_value", looked
    assert "already decided" in looked.reason
    assert len(question.decisions) == 1 and len(result.decisions) == 1
