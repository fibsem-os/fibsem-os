"""A task asks for a value it needs now with ``ask``: one way, on the record.

Headless here, so nobody is there to ask: the proposal goes on the record with
an ``Unreviewed`` decision carrying the values the task went on to use, for
someone to check afterwards. The supervised path, where the run holds until
the decision lands, needs a window and is in tests/ui.
"""

import os
from pathlib import Path

import pytest
import yaml
from psygnal.containers import EventedDict

import fibsem.config as cfg
from fibsem import utils
from fibsem.applications.autolamella.proposals import (
    POINT_OF_INTEREST,
    PROPOSAL_KINDS,
    AuthorKind,
    Decision,
    DecisionOutcome,
)
from fibsem.applications.autolamella.structures import (
    Attention,
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
from fibsem.structures import Point

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


def _task(microscope, exp, cls=_AsksForAPoint, body=None):
    manager = TaskManager(microscope=microscope, experiment=exp, parent_ui=None)
    manager.review_enabled = True
    lamella = exp.positions[0]
    task = cls(
        microscope=microscope,
        config=lamella.task_config[ROUGH],
        lamella=lamella,
        parent_ui=None,
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
    assert exp.positions[0].proposals == {}


def test_a_value_the_kind_does_not_carry_is_refused(microscope, tmp_path):
    exp = _experiment(tmp_path, microscope)
    with pytest.raises(ValueError, match="does not carry"):
        _task(microscope, exp).ask(POINT_OF_INTEREST, {"nope": 1})


def test_with_nobody_to_ask_the_proposed_value_is_used_and_recorded_unreviewed(
    microscope, tmp_path
):
    """Automated, or supervised with no window: the answer is what was
    proposed, and the record says nobody looked."""
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
    assert decision.author.kind is AuthorKind.automated
    # On the record, before the run's own result, and listed to check.
    question, result = lamella.proposals[ROUGH]
    assert question.kind == POINT_OF_INTEREST and question.unreviewed
    assert question.provenance["reference_image"] == "ref.tif"
    assert question.provenance["task_id"] == result.provenance["task_id"]
    assert question.to_check and not question.pending
    assert any(p is question for _i, _t, p in exp.proposals_to_check()), (
        "listed to check, beside the run's result"
    )
    assert not lamella.is_awaiting_decision(ROUGH), "nothing waits on it"


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
            author="human:op",
            proposal_id=proposal.id,
        ),
    )
    assert not result.applied and result.error_type == "invalid_value"


def test_a_persons_later_look_clears_it_from_to_check(microscope, tmp_path):
    exp = _experiment(tmp_path, microscope)
    lamella = exp.positions[0]
    _task(microscope, exp).ask(POINT_OF_INTEREST, {"poi": Point(0, 0)})
    proposal = lamella.proposal(ROUGH)
    assert proposal.to_check
    result = exp.decide(
        lamella.id,
        ROUGH,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            proposal_id=proposal.id,
        ),
    )
    assert result.applied, result.reason
    assert not proposal.to_check and not proposal.unreviewed
    assert lamella.poi == Point(0, 0), "a look writes nothing"


def test_unreviewed_survives_the_file(microscope, tmp_path):
    exp = _experiment(tmp_path, microscope)
    _task(microscope, exp).ask(POINT_OF_INTEREST, {"poi": Point(3e-6, 0)})
    exp.save()
    again = Experiment.load(str(Path(exp.path, "experiment.yaml")))
    proposal = again.positions[0].proposal(ROUGH)
    assert proposal.unreviewed and proposal.current.values["poi"] == Point(3e-6, 0)
    raw = yaml.safe_load(Path(exp.path, "experiment.yaml").read_text())
    assert raw["positions"][0]["proposals"][ROUGH][0]["decisions"][0]["outcome"] == (
        "Unreviewed"
    )


def test_every_task_type_declares_kinds_the_record_knows():
    """A task says what it asks; what it says has to be a proposal kind."""
    for name, cls in get_tasks().items():
        for kind in cls.questions:
            assert kind in PROPOSAL_KINDS, f"{name} asks an unknown kind {kind!r}"
        assert isinstance(cls.sessions, tuple), name


def test_a_question_and_the_runs_result_are_both_current(microscope, tmp_path):
    """Different kinds do not replace each other: the unreviewed question is
    listed to check beside the run's result, and is decided by its own id
    without touching the result."""
    exp = _experiment(tmp_path, microscope)
    lamella = exp.positions[0]
    _task(
        microscope,
        exp,
        body=lambda t: t.ask(POINT_OF_INTEREST, {"poi": Point(0, 0)}),
    ).run()
    question, result = lamella.proposals[ROUGH]
    assert lamella.current_proposals(ROUGH) == [question, result]
    listed = [p for _i, _t, p in exp.proposals_to_check()]
    assert listed == [question, result]

    looked = exp.decide(
        lamella.id,
        ROUGH,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            proposal_id=question.id,
        ),
    )
    assert looked.applied, looked.reason
    assert not question.to_check and result.to_check, "the result is untouched"
    assert [p for _i, _t, p in exp.proposals_to_check()] == [result]
