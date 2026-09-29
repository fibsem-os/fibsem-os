"""The ``state`` kind: a task's position, confirmed before it goes on.

Acquire Reference Image is the first task to ask through ``ask``: with the
review preference on it asks a ``state`` question in place of its "Press
Continue when ready" prompt. Headless nobody is there, so the position goes
on the record as proposed, ``Unreviewed``. With the preference off nothing
changes: the old prompt, and no record.
"""

import math
import os
from pathlib import Path

import pytest
from psygnal.containers import EventedDict

import fibsem.config as cfg
from fibsem import utils
from fibsem.applications.autolamella.proposals import (
    STATE,
    TASK_RESULT,
    Decision,
    DecisionOutcome,
    compute_delta,
)
from fibsem.applications.autolamella.structures import (
    Attention,
    AutoLamellaTaskDescription,
    AutoLamellaTaskProtocol,
    AutoLamellaWorkflowConfig,
    Experiment,
)
from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
from fibsem.applications.autolamella.workflows.tasks.reference_image import (
    AcquireReferenceImageConfig,
    AcquireReferenceImageTask,
)
from fibsem.structures import FibsemStagePosition

REF = "Acquire Reference Image"
CONFIG = os.path.join(cfg.CONFIG_PATH, "microscope-configuration.yaml")


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo", config_path=CONFIG)
    yield microscope
    microscope.disconnect()


def _experiment(tmp_path: Path, microscope, attention=Attention.automated):
    exp = Experiment(path=tmp_path, name="state-exp")
    exp.task_protocol = AutoLamellaTaskProtocol(
        workflow_config=AutoLamellaWorkflowConfig(
            tasks=[
                AutoLamellaTaskDescription(name=REF, required=True, attention=attention)
            ]
        )
    )
    os.makedirs(exp.path, exist_ok=True)
    exp.add_new_lamella(
        microscope.get_microscope_state(),
        EventedDict({REF: AcquireReferenceImageConfig(task_name=REF)}),
    )
    lamella = exp.positions[0]
    lamella.path.mkdir(parents=True, exist_ok=True)
    lamella.milling_pose = microscope.get_microscope_state()
    return exp


def _task(microscope, exp, review_enabled=True) -> AcquireReferenceImageTask:
    manager = TaskManager(microscope=microscope, experiment=exp, parent_ui=None)
    manager.review_enabled = review_enabled
    lamella = exp.positions[0]
    return AcquireReferenceImageTask(
        microscope=microscope,
        config=lamella.task_config[REF],
        lamella=lamella,
        parent_ui=None,
        task_manager=manager,
    )


def test_the_task_declares_its_question():
    assert AcquireReferenceImageTask.questions == (STATE,)


def test_headless_the_position_goes_on_the_record_unreviewed(microscope, tmp_path):
    exp = _experiment(tmp_path, microscope)
    lamella = exp.positions[0]

    _task(microscope, exp).run()

    question, result = lamella.proposals[REF]
    assert question.kind == STATE and result.kind == TASK_RESULT
    assert question.unreviewed
    pose = question.values["stage_position"]
    assert isinstance(pose, FibsemStagePosition)
    assert question.current.values["stage_position"] == pose, "used as proposed"
    assert "Press Continue" in question.provenance["message"]
    assert not question.provenance["reference_image"], "asked before an image"


def test_with_the_preference_off_nothing_is_recorded(microscope, tmp_path):
    exp = _experiment(tmp_path, microscope, attention=Attention.supervised)
    _task(microscope, exp, review_enabled=False).run()
    assert [p.kind for p in exp.positions[0].proposals[REF]] == [TASK_RESULT]


def test_the_position_survives_the_file(microscope, tmp_path):
    exp = _experiment(tmp_path, microscope)
    _task(microscope, exp).run()
    pose = exp.positions[0].proposal(REF, STATE).values["stage_position"]
    exp.save()
    again = Experiment.load(str(Path(exp.path, "experiment.yaml")))
    read = again.positions[0].proposal(REF, STATE).values["stage_position"]
    assert isinstance(read, FibsemStagePosition)
    assert (read.x, read.y, read.z, read.r, read.t) == pytest.approx(
        (pose.x, pose.y, pose.z, pose.r, pose.t)
    )


def test_a_confirmed_position_is_checked_and_written_nowhere(microscope, tmp_path):
    """The task uses the position; the writer checks it and touches nothing,
    like a detection's features. A later look at an unreviewed one is
    accepted and carries no values."""
    from fibsem.applications.autolamella.proposals import (
        ValueRefused,
        prepare_values,
    )

    exp = _experiment(tmp_path, microscope)
    lamella = exp.positions[0]
    _task(microscope, exp).run()
    before = lamella.to_dict()
    moved = FibsemStagePosition(x=1e-3, y=2e-3, z=0.0, r=0.0, t=math.radians(5))
    write = prepare_values(exp, lamella, STATE, {"stage_position": moved})
    assert write.apply() == []
    after = lamella.to_dict()
    after.pop("proposals"), before.pop("proposals")
    assert after == before, "nothing on the lamella moved"
    with pytest.raises(ValueRefused):
        prepare_values(exp, lamella, STATE, {"stage_position": "here"})

    question = lamella.proposal(REF, STATE)
    looked = exp.decide(
        lamella.id,
        REF,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            proposal_id=question.id,
        ),
    )
    assert looked.applied, looked.reason
    assert not question.unreviewed and not question.to_check


def test_the_delta_between_positions_is_a_position():
    proposed = FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=0.0, t=0.0)
    confirmed = FibsemStagePosition(x=1e-6, y=-2e-6, z=0.0, r=0.0, t=0.1)
    delta = compute_delta(proposed, confirmed)
    assert isinstance(delta, FibsemStagePosition)
    assert (delta.x, delta.y, delta.t) == pytest.approx((1e-6, -2e-6, 0.1))
    assert compute_delta(FibsemStagePosition(x=None), confirmed) is None
