"""Select Position asks its two confirmations through ``ask``.

With the review preference on, the tilt to the milling angle and the move to
the milling position are ``state`` questions on the record, each with a
switch of its own in the task's settings (``confirm_tilt``,
``confirm_position``). Off, that one question is recorded and not asked.
Headless nobody is there, so both go on the record as proposed,
``Unreviewed``. With the preference off nothing changes: the prompts as they
have always been, and no record of them.

Runs the real task against the Demo microscope, which starts 38° from the
milling angle, so the tilt is asked.
"""

import os
from pathlib import Path

import pytest
from psygnal.containers import EventedDict

import fibsem.config as cfg
from fibsem import utils
from fibsem.applications.autolamella.proposals import (
    ALIGNMENT_AREA,
    DETECTION,
    POINT_OF_INTEREST,
    STATE,
)
from fibsem.applications.autolamella.structures import (
    Attention,
    AutoLamellaTaskDescription,
    AutoLamellaTaskProtocol,
    AutoLamellaWorkflowConfig,
    Experiment,
)
from fibsem.applications.autolamella.workflows.tasks import base as B
from fibsem.applications.autolamella.workflows.tasks import select_position as S
from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
from fibsem.applications.autolamella.workflows.tasks.rough import MillRoughTaskConfig
from fibsem.applications.autolamella.workflows.tasks.select_position import (
    SelectMillingPositionTask,
    SelectMillingPositionTaskConfig,
)
from fibsem.structures import FibsemRectangle, FibsemStagePosition

SETUP = "Setup Lamella Position"
ROUGH = "Rough Milling"
CONFIG = os.path.join(cfg.CONFIG_PATH, "microscope-configuration.yaml")
MILLING_ANGLE = 15.0


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo", config_path=CONFIG)
    yield microscope
    microscope.disconnect()


def _experiment(
    tmp_path: Path, microscope, attention=Attention.automated, **flags
) -> Experiment:
    exp = Experiment(path=tmp_path, name="asks-exp")
    exp.task_protocol = AutoLamellaTaskProtocol(
        workflow_config=AutoLamellaWorkflowConfig(
            tasks=[
                AutoLamellaTaskDescription(
                    name=SETUP, required=True, attention=attention
                ),
                AutoLamellaTaskDescription(name=ROUGH, required=True, requires=[SETUP]),
            ]
        )
    )
    os.makedirs(exp.path, exist_ok=True)
    exp.add_new_lamella(
        microscope.get_microscope_state(),
        EventedDict(
            {
                SETUP: SelectMillingPositionTaskConfig(
                    task_name=SETUP,
                    milling_angle=MILLING_ANGLE,
                    auto_milling_alignment=False,
                    use_autofocus=False,
                    select_poi=True,
                    **flags,
                ),
                ROUGH: MillRoughTaskConfig(task_name=ROUGH),
            }
        ),
    )
    lamella = exp.positions[0]
    lamella.path.mkdir(parents=True, exist_ok=True)
    lamella.milling_pose = microscope.get_microscope_state()
    return exp


def _task(microscope, exp, review_enabled=True) -> SelectMillingPositionTask:
    manager = TaskManager(microscope=microscope, experiment=exp, parent_ui=None)
    manager.review_enabled = review_enabled
    lamella = exp.positions[0]
    return SelectMillingPositionTask(
        microscope=microscope,
        config=lamella.task_config[SETUP],
        lamella=lamella,
        parent_ui=None,
        task_manager=manager,
    )


def _supervised_in_a_window_less_run(monkeypatch):
    """The task reads "supervised" off the window's protocol; headless there
    is no window, so a supervised run has to be stood up by hand."""
    monkeypatch.setattr(
        SelectMillingPositionTask, "validate", property(lambda self: True)
    )


# ---------------------------------------------------------------------------
# What it declares
# ---------------------------------------------------------------------------


def test_the_flags_take_the_position_question_out_of_the_declaration():
    on = SelectMillingPositionTaskConfig(task_name=SETUP)
    assert on.confirm_position and on.confirm_tilt, "on by default: today's prompts"
    assert on.confirm_alignment_area, "on by default too"
    assert SelectMillingPositionTask.questions_for(on) == (STATE, ALIGNMENT_AREA)
    one = SelectMillingPositionTaskConfig(task_name=SETUP, confirm_position=False)
    assert SelectMillingPositionTask.questions_for(one) == (STATE, ALIGNMENT_AREA), (
        "the tilt still asks"
    )
    off = SelectMillingPositionTaskConfig(
        task_name=SETUP,
        confirm_position=False,
        confirm_tilt=False,
        confirm_alignment_area=False,
    )
    assert SelectMillingPositionTask.questions_for(off) == ()
    walk = SelectMillingPositionTaskConfig(
        task_name=SETUP,
        confirm_position=False,
        confirm_tilt=False,
        confirm_alignment_area=False,
        auto_milling_alignment=True,
    )
    assert SelectMillingPositionTask.questions_for(walk) == (DETECTION,)


# ---------------------------------------------------------------------------
# With the preference on
# ---------------------------------------------------------------------------


def test_headless_both_confirmations_go_on_the_record_unreviewed(microscope, tmp_path):
    exp = _experiment(tmp_path, microscope)
    lamella = exp.positions[0]
    assert microscope.get_current_milling_angle() == pytest.approx(38.0), (
        "the Demo stage starts off the angle"
    )
    before = microscope.get_stage_position()

    _task(microscope, exp).run()

    tilt, position, area, point = lamella.proposals[SETUP]
    assert [p.kind for p in (tilt, position, area, point)] == [
        STATE,
        STATE,
        ALIGNMENT_AREA,
        POINT_OF_INTEREST,
    ], "the tilt, the position, the alignment area, then the point for afterwards"
    assert tilt.unreviewed and position.unreviewed
    assert point.pending, "open to correct until Rough Milling starts"

    assert "Tilt to the milling angle (15.0" in tilt.provenance["message"]
    assert "38.0" in tilt.provenance["message"], "says where the stage is"
    assert tilt.values["stage_position"].t == pytest.approx(before.t), (
        "proposed before the tilt"
    )
    assert lamella.milling_pose.stage_position.t != pytest.approx(before.t), (
        "and the pose stored at the end is after it"
    )
    assert tilt.provenance["reference_image"] == f"ref_{SETUP}_start_ib.tif"
    assert microscope.get_current_milling_angle() == pytest.approx(MILLING_ANGLE), (
        "and the task tilted, as it does when nobody answers"
    )

    assert "Double click the image" in position.provenance["message"]
    pose = position.values["stage_position"]
    assert isinstance(pose, FibsemStagePosition)
    assert position.current.values["stage_position"] == pose, "used as proposed"
    assert position.provenance["reference_image"] == f"ref_{SETUP}_post_tilt_ib.tif", (
        "sits on the image taken at the milling angle"
    )
    assert os.path.exists(
        os.path.join(str(lamella.path), position.provenance["reference_image"])
    )
    assert lamella.has_completed_task(SETUP), "automated: nothing waits"


def test_a_switched_off_confirmation_is_recorded_and_not_asked(
    microscope, tmp_path, monkeypatch
):
    """Off means that one question behaves as Automated -- and is still on
    the record, so the run can be read afterwards -- while the other is
    asked. Stood up as supervised so the difference is the switch alone."""
    _supervised_in_a_window_less_run(monkeypatch)
    exp = _experiment(
        tmp_path, microscope, attention=Attention.supervised, confirm_tilt=False
    )
    monkeypatch.setattr(S, "select_poi_ui", lambda **kwargs: None)
    lamella = exp.positions[0]

    _task(microscope, exp).run()

    tilt, position = [p for p in lamella.proposals[SETUP] if p.kind == STATE]
    assert tilt.unreviewed and "turned off" in tilt.current.reason
    assert position.unreviewed and "no window" in position.current.reason, (
        "asked, but headless there is nobody to answer"
    )


# ---------------------------------------------------------------------------
# With the preference off: the prompts as they have always been
# ---------------------------------------------------------------------------


def test_with_the_preference_off_the_old_prompts_run_and_nothing_is_recorded(
    microscope, tmp_path, monkeypatch
):
    _supervised_in_a_window_less_run(monkeypatch)
    exp = _experiment(tmp_path, microscope, attention=Attention.supervised)
    asked = []

    def _ask_user(parent_ui, msg, pos, neg=None):
        asked.append((pos, neg))
        return False  # Skip the tilt

    monkeypatch.setattr(S, "ask_user", _ask_user)
    monkeypatch.setattr(S, "select_poi_ui", lambda **kwargs: None)
    lamella = exp.positions[0]

    _task(microscope, exp, review_enabled=False).run()

    assert asked == [("Tilt", "Skip"), ("Continue", None)]
    assert microscope.get_current_milling_angle() == pytest.approx(38.0), (
        "Skip still skips"
    )
    assert [p.kind for p in lamella.proposals[SETUP]] == [POINT_OF_INTEREST], (
        "no record of either prompt"
    )


def test_with_the_preference_off_the_flags_still_switch_the_prompts_off(
    microscope, tmp_path, monkeypatch
):
    """A protocol author turns a prompt off for every build, not only for
    those with the preference on: off, the task tilts and carries on."""
    _supervised_in_a_window_less_run(monkeypatch)
    exp = _experiment(
        tmp_path,
        microscope,
        attention=Attention.supervised,
        confirm_tilt=False,
        confirm_position=False,
    )
    asked = []
    monkeypatch.setattr(S, "ask_user", lambda **kwargs: asked.append(kwargs))
    monkeypatch.setattr(S, "select_poi_ui", lambda **kwargs: None)

    _task(microscope, exp, review_enabled=False).run()

    assert asked == []
    assert microscope.get_current_milling_angle() == pytest.approx(MILLING_ANGLE)


# ---------------------------------------------------------------------------
# The alignment area (FIB-1053)
# ---------------------------------------------------------------------------


def test_headless_the_alignment_area_goes_on_the_record_unreviewed(
    microscope, tmp_path
):
    """Asked on the last FIB image, right before the alignment reference is
    taken in it; nobody there, so used as proposed."""
    exp = _experiment(tmp_path, microscope)
    lamella = exp.positions[0]
    proposed = FibsemRectangle(left=0.2, top=0.3, width=0.4, height=0.3)
    lamella.alignment_area = proposed

    _task(microscope, exp).run()

    area = lamella.proposal(SETUP, ALIGNMENT_AREA)
    assert area is not None and area.unreviewed
    assert area.values["alignment_area"] == proposed
    assert area.current.values["alignment_area"] == proposed, "used as proposed"
    assert "alignment area" in area.provenance["message"]
    assert area.provenance["reference_image"] == f"ref_{SETUP}_post_tilt_ib.tif", (
        "the last FIB image before the reference is taken"
    )
    assert lamella.alignment_area == proposed


def test_a_switched_off_alignment_area_is_recorded_and_not_asked(
    microscope, tmp_path, monkeypatch
):
    _supervised_in_a_window_less_run(monkeypatch)
    exp = _experiment(
        tmp_path,
        microscope,
        attention=Attention.supervised,
        confirm_alignment_area=False,
    )
    monkeypatch.setattr(S, "select_poi_ui", lambda **kwargs: None)
    lamella = exp.positions[0]

    _task(microscope, exp).run()

    area = lamella.proposal(SETUP, ALIGNMENT_AREA)
    assert area.unreviewed and "turned off" in area.current.reason
    assert ALIGNMENT_AREA not in SelectMillingPositionTask.questions_for(
        lamella.task_config[SETUP]
    )


def test_with_the_preference_off_the_alignment_area_prompt_runs_as_before(
    microscope, tmp_path, monkeypatch
):
    _supervised_in_a_window_less_run(monkeypatch)
    exp = _experiment(tmp_path, microscope, attention=Attention.supervised)
    lamella = exp.positions[0]
    asked = []

    def old_prompt(alignment_area, parent_ui, msg, validate):
        asked.append((alignment_area, msg, validate))
        return alignment_area

    monkeypatch.setattr(B, "update_alignment_area_ui", old_prompt)
    monkeypatch.setattr(S, "ask_user", lambda *a, **k: "Continue")
    monkeypatch.setattr(S, "select_poi_ui", lambda **kwargs: None)

    _task(microscope, exp, review_enabled=False).run()

    assert len(asked) == 1 and asked[0][2] is True
    assert "Drag to edit the Alignment Area" in asked[0][1]
    assert lamella.proposal(SETUP, ALIGNMENT_AREA) is None, "nothing recorded"
