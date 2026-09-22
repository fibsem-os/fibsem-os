"""The two fluorescence tasks ask their confirmations as ``state`` questions.

Select Fluorescence Position confirms the stage and objective are at the
fluorescence pose; Acquire Fluorescence Image confirms the position before
its autofocus runs. With the review preference on each is a ``state``
question through ``ask`` in place of its "Press Continue" prompt; headless
nobody is there, so the position goes on the record as proposed,
``Unreviewed``. With the preference off nothing changes: the old prompt, and
no record.

Runs the real tasks on the simulated Arctis, which has a fluorescence
microscope; the autofocus and the acquisition are stubbed as the pose tests
stub them.
"""

import os
import types
from pathlib import Path

import pytest
from psygnal.containers import EventedDict

import fibsem.config as fconfig
from fibsem import utils
from fibsem.applications.autolamella.proposals import STATE, TASK_RESULT
from fibsem.applications.autolamella.structures import (
    Attention,
    AutoLamellaTaskDescription,
    AutoLamellaTaskProtocol,
    AutoLamellaWorkflowConfig,
    Experiment,
)
from fibsem.applications.autolamella.workflows.tasks import acquire_fluorescence as AF
from fibsem.applications.autolamella.workflows.tasks import (
    select_fluorescence_position as SF,
)
from fibsem.applications.autolamella.workflows.tasks.acquire_fluorescence import (
    AcquireFluorescenceImageConfig,
    AcquireFluorescenceImageTask,
)
from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
from fibsem.applications.autolamella.workflows.tasks.select_fluorescence_position import (
    SelectFluorescencePositionConfig,
    SelectFluorescencePositionTask,
)
from fibsem.fm.structures import ChannelSettings
from fibsem.structures import FibsemStagePosition

SELECT = "Select Fluorescence Position"
ACQUIRE = "Acquire Fluorescence Image"
CONFIG = os.path.join(fconfig.CONFIG_PATH, "sim-arctis-configuration.yaml")
OBJECTIVE = 0.006


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(config_path=CONFIG, setup_logging=False)
    if microscope.fm is None:
        pytest.skip("no fluorescence microscope in the simulator")
    microscope.fm._allow_unknown_orientations = True
    yield microscope
    microscope.disconnect()


def _experiment(tmp_path: Path, microscope, attention=Attention.automated):
    exp = Experiment(path=tmp_path, name="fm-exp")
    exp.task_protocol = AutoLamellaTaskProtocol(
        workflow_config=AutoLamellaWorkflowConfig(
            tasks=[
                AutoLamellaTaskDescription(
                    name=SELECT, required=True, attention=attention
                ),
                AutoLamellaTaskDescription(
                    name=ACQUIRE, required=True, attention=attention
                ),
            ]
        )
    )
    os.makedirs(exp.path, exist_ok=True)
    exp.add_new_lamella(
        microscope.get_microscope_state(),
        EventedDict(
            {
                SELECT: SelectFluorescencePositionConfig(task_name=SELECT),
                ACQUIRE: AcquireFluorescenceImageConfig(
                    task_name=ACQUIRE, channel_settings=[ChannelSettings(name="GFP")]
                ),
            }
        ),
    )
    lamella = exp.positions[0]
    lamella.path.mkdir(parents=True, exist_ok=True)
    lamella.milling_pose = microscope.get_microscope_state()
    lamella.fluorescence_pose = microscope.get_microscope_state()
    lamella.fluorescence_pose.objective_position = OBJECTIVE
    return exp


def _task(cls, name, microscope, exp, review_enabled=True):
    manager = TaskManager(microscope=microscope, experiment=exp, parent_ui=None)
    manager.review_enabled = review_enabled
    lamella = exp.positions[0]
    return cls(
        microscope=microscope,
        config=lamella.task_config[name],
        lamella=lamella,
        parent_ui=None,
        task_manager=manager,
    )


@pytest.fixture
def stubbed_acquisition(monkeypatch):
    """The autofocus and the z-stack are the slow parts and not the point."""
    monkeypatch.setattr(
        AF,
        "run_coarse_fine_autofocus",
        lambda *a, **k: types.SimpleNamespace(
            working_distance=OBJECTIVE, save=lambda *a, **k: None
        ),
    )
    monkeypatch.setattr(AF, "acquire_image", lambda **kwargs: None)


def test_both_tasks_declare_the_question():
    assert SelectFluorescencePositionTask.questions == (STATE,)
    assert AcquireFluorescenceImageTask.questions == (STATE,)


def test_select_position_headless_the_position_goes_on_the_record_unreviewed(
    microscope, tmp_path
):
    exp = _experiment(tmp_path, microscope)
    lamella = exp.positions[0]

    _task(SelectFluorescencePositionTask, SELECT, microscope, exp).run()

    question, result = lamella.proposals[SELECT]
    assert question.kind == STATE and result.kind == TASK_RESULT
    assert question.unreviewed
    pose = question.values["stage_position"]
    assert isinstance(pose, FibsemStagePosition)
    assert question.current.values["stage_position"] == pose, "used as proposed"
    assert "Move to fluorescence position" in question.provenance["message"]
    assert lamella.fluorescence_pose.objective_position == pytest.approx(OBJECTIVE), (
        "the objective is recorded on the pose, as before"
    )


def test_acquire_headless_the_position_before_autofocus_goes_on_the_record(
    microscope, tmp_path, stubbed_acquisition
):
    exp = _experiment(tmp_path, microscope)
    lamella = exp.positions[0]
    assert lamella.task_config[ACQUIRE].autofocus_settings.enabled

    _task(AcquireFluorescenceImageTask, ACQUIRE, microscope, exp).run()

    question = lamella.proposal(ACQUIRE, STATE)
    assert question is not None and question.unreviewed
    assert "Run autofocus" in question.provenance["message"]
    assert isinstance(question.values["stage_position"], FibsemStagePosition)


def test_with_the_preference_off_the_old_prompts_run_and_nothing_is_recorded(
    microscope, tmp_path, stubbed_acquisition, monkeypatch
):
    for cls in (SelectFluorescencePositionTask, AcquireFluorescenceImageTask):
        monkeypatch.setattr(cls, "validate", property(lambda self: True))
    asked = []
    monkeypatch.setattr(SF, "ask_user", lambda ui, msg, pos: asked.append((msg, pos)))
    monkeypatch.setattr(AF, "ask_user", lambda ui, msg, pos: asked.append((msg, pos)))
    exp = _experiment(tmp_path, microscope, attention=Attention.supervised)
    lamella = exp.positions[0]

    _task(SelectFluorescencePositionTask, SELECT, microscope, exp, False).run()
    _task(AcquireFluorescenceImageTask, ACQUIRE, microscope, exp, False).run()

    assert [pos for _m, pos in asked] == ["Continue", "Continue"]
    assert "Move to fluorescence position" in asked[0][0]
    assert "Run autofocus" in asked[1][0]
    assert [p.kind for p in lamella.proposals[SELECT]] == [TASK_RESULT]
    assert [p.kind for p in lamella.proposals[ACQUIRE]] == [TASK_RESULT]
