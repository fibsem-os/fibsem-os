"""The ``state`` kind: a task's position, confirmed before it goes on.

Acquire Reference Image was the first task to ask through ``ask``: it asks
a ``state`` question in place of its "Press Continue when ready" prompt,
whether the review preference is on or off. Headless nobody is there, so the
position goes on the record as proposed, ``Unreviewed``.
"""

import math
import os
from pathlib import Path

import pytest
from psygnal.containers import EventedDict

import fibsem.config as cfg
from fibsem import utils
from fibsem.applications.autolamella.structures import (
    Experiment,
)
from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
from fibsem.applications.autolamella.workflows.tasks.reference_image import (
    AcquireReferenceImageConfig,
    AcquireReferenceImageTask,
)
from fibsem.structures import FibsemStagePosition
from fibsem.workflows.attention import Attention
from fibsem.workflows.proposals import (
    STATE,
    Decision,
    DecisionOutcome,
    compute_delta,
)
from tests.protocols import planned, task_protocol

REF = "Acquire Reference Image"
CONFIG = os.path.join(cfg.CONFIG_PATH, "microscope-configuration.yaml")


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo", config_path=CONFIG)
    yield microscope
    microscope.disconnect()


def _experiment(tmp_path: Path, microscope, attention=Attention.automated):
    exp = Experiment(path=tmp_path, name="state-exp")
    exp.task_protocol = task_protocol(
        planned(name=REF, required=True, attention=attention)
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


def _task(microscope, exp) -> AcquireReferenceImageTask:
    manager = TaskManager(microscope=microscope, experiment=exp)
    lamella = exp.positions[0]
    return AcquireReferenceImageTask(
        microscope=microscope,
        config=lamella.task_config[REF],
        lamella=lamella,
        task_manager=manager,
    )


def test_the_task_declares_its_question():
    assert AcquireReferenceImageTask.asks == (STATE,)


def test_headless_the_position_goes_on_the_record_unreviewed(microscope, tmp_path):
    exp = _experiment(tmp_path, microscope)
    lamella = exp.positions[0]

    _task(microscope, exp).run()

    (question,) = lamella.proposals_from(REF)
    assert question.kind == STATE
    assert question.unreviewed
    pose = question.values["stage_position"]
    assert isinstance(pose, FibsemStagePosition)
    assert question.current.values["stage_position"] == pose, "used as proposed"
    assert "Press Continue" in question.provenance["message"]
    assert not question.provenance["reference_image"], "asked before an image"


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
    like a detection's features. Unasked, it is recorded unreviewed, and a
    later decision on it is refused as already decided."""
    from fibsem.workflows.proposals import ValueRefused, prepare_values

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
            actor="operator:op",
            proposal_id=question.id,
        ),
    )
    assert not looked.applied and looked.error_type == "invalid_value", looked
    assert question.unreviewed


def test_the_delta_between_positions_is_a_position():
    proposed = FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=0.0, t=0.0)
    confirmed = FibsemStagePosition(x=1e-6, y=-2e-6, z=0.0, r=0.0, t=0.1)
    delta = compute_delta(proposed, confirmed)
    assert isinstance(delta, FibsemStagePosition)
    assert (delta.x, delta.y, delta.t) == pytest.approx((1e-6, -2e-6, 0.1))
    assert compute_delta(FibsemStagePosition(x=None), confirmed) is None
