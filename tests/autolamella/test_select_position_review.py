"""SelectMillingPositionTask asks for the point of interest once the position
is confirmed: a ``point_of_interest`` question on the record, answered at the
FIB view's marker or by an agent, whose decision writes the point through.

Runs the real task against the Demo microscope, through a real TaskManager,
so the attention is read the way the application reads it. No Qt: parent_ui
is None throughout.
"""

import os
import threading
import time
from pathlib import Path

from psygnal.containers import EventedDict

from fibsem.acting import ActorKind
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
from fibsem.applications.autolamella.workflows.tasks.rough import MillRoughTaskConfig
from fibsem.applications.autolamella.workflows.tasks.select_position import (
    SelectMillingPositionTask,
    SelectMillingPositionTaskConfig,
)
from fibsem.structures import Point
from fibsem.workflows.attention import Attention
from fibsem.workflows.proposals import (
    ALIGNMENT_AREA,
    MILLING,
    POINT_OF_INTEREST,
    STATE,
    Decision,
    DecisionOutcome,
)
from tests.answering import answering
from tests.fixtures.microscopes import demo_microscope
from tests.protocols import planned, task_protocol

SETUP = "Setup Lamella Position"
ROUGH = "Rough Milling"


def _in_run(relpath: str):
    """(task folder, file name) of an image in a run's own folder,
    ``<task>/<YYYYMMDD-HHMMSS>/<file>`` (FIB-1275)."""
    parts = Path(relpath).parts
    assert len(parts) == 3, f"{relpath} is not in a run's folder"
    return parts[0], parts[-1]


microscope = demo_microscope(scope="function")


def _experiment(
    tmp_path: Path, microscope, attention: Attention = Attention.automated
) -> Experiment:
    exp = Experiment(path=tmp_path, name="test-exp")
    exp.task_protocol = task_protocol(
        planned(name=SETUP, required=True, attention=attention),
        planned(name=ROUGH, required=True, requires=[SETUP]),
    )
    os.makedirs(exp.path, exist_ok=True)
    exp.add_new_lamella(
        microscope.get_microscope_state(),
        EventedDict(
            {
                SETUP: SelectMillingPositionTaskConfig(
                    task_name=SETUP,
                    auto_milling_alignment=False,
                    use_autofocus=False,
                    select_poi=True,
                ),
                ROUGH: MillRoughTaskConfig(task_name=ROUGH),
            }
        ),
    )
    lamella = exp.positions[0]
    lamella.path.mkdir(parents=True, exist_ok=True)
    lamella.milling_pose = microscope.get_microscope_state()
    return exp


def _task(microscope, exp: Experiment, deciders=None) -> SelectMillingPositionTask:
    manager = TaskManager(microscope=microscope, experiment=exp, deciders=deciders)
    lamella = exp.positions[0]
    return SelectMillingPositionTask(
        microscope=microscope,
        config=lamella.task_config[SETUP],
        lamella=lamella,
        task_manager=manager,
    )


def _run(task: SelectMillingPositionTask) -> None:
    """Run the task with someone there to answer what it asks inline: a
    supervised question waits for an answer, window or not."""
    with answering(task.task_manager.experiment):
        task.run()


def _points(lamella) -> list:
    """Setup's point-of-interest proposals, leaving out the position
    confirmations and alignment area it also records."""
    return [p for p in lamella.proposals_from(SETUP) if p.kind == POINT_OF_INTEREST]


def _rough_point(lamella):
    return lamella.task_config[ROUGH].milling["mill_rough"].stages[0].pattern.point


def test_supervised_with_the_agent_to_answer_the_point_is_asked_and_written(
    microscope, tmp_path
):
    """The question stands while the task waits; the decision moves the point
    and the patterns that follow it, and the task carries on to the end."""
    exp = _experiment(tmp_path, microscope, attention=Attention.supervised)
    lamella = exp.positions[0]
    config = lamella.task_config[SETUP]
    config.confirm_position = config.confirm_tilt = False
    config.confirm_alignment_area = False
    task = _task(microscope, exp, deciders=lambda: (ActorKind.agent,))
    rough_point = _rough_point(lamella)
    running = threading.Thread(target=task.run, daemon=True)
    running.start()
    deadline = time.monotonic() + 30
    while not (asked := [p for p in _points(lamella) if p.asking]):
        assert time.monotonic() < deadline, "the point was never asked"
        assert running.is_alive(), "the task did not wait for an answer"
        time.sleep(0.01)
    (question,) = asked
    assert question.values == {"poi": Point(0.0, 0.0)}, "the point as it stands"
    assert "Move the marker" in question.provenance["message"]
    assert _in_run(question.provenance["reference_image"])[0] == SETUP

    result = exp.decide(
        lamella.id,
        SETUP,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            actor="agent:x",
            proposal_id=question.id,
            values={"poi": Point(3e-6, 0.0)},
        ),
    )
    assert result.applied, result.reason
    running.join(timeout=30)
    assert not running.is_alive(), "the task went on once it was answered"

    assert lamella.poi == Point(3e-6, 0.0)
    assert _rough_point(lamella) != rough_point, "the patterns followed it"
    assert lamella.has_completed_task(SETUP), "nothing waits after it ends"
    assert task.task_manager._defer_reason(lamella, ROUGH) is None


def test_supervised_the_point_asked_is_the_one_already_set(microscope, tmp_path):
    exp = _experiment(tmp_path, microscope, attention=Attention.supervised)
    lamella = exp.positions[0]
    lamella.poi = Point(4e-6, -2e-6)  # set before Setup: correlation, a script

    _run(_task(microscope, exp))

    (point,) = _points(lamella)
    assert point.current.outcome is DecisionOutcome.Confirmed, "asked and answered"
    assert point.values == {"poi": Point(4e-6, -2e-6)}, "what was already set"
    assert point.values["poi"] is not lamella.poi, "a copy, not the live point"
    assert lamella.poi == Point(4e-6, -2e-6), "not undone by the step checking it"
    assert lamella.has_completed_task(SETUP)
    assert MILLING not in SelectMillingPositionTask.asks, "nothing milled"


def test_automated_the_point_is_recorded_unasked_and_nothing_waits(
    microscope, tmp_path
):
    """Nobody is asked: the questions and the result all go on the record
    Unreviewed, and nothing is left pending."""
    exp = _experiment(tmp_path, microscope)
    task = _task(microscope, exp)
    assert task.supervised is False
    lamella = exp.positions[0]

    _run(task)

    assert lamella.poi == Point(0.0, 0.0), "used as it stands"
    assert lamella.has_completed_task(SETUP)
    recorded = lamella.proposals_from(SETUP)
    assert [p.kind for p in recorded] == [
        STATE,
        STATE,
        POINT_OF_INTEREST,
        ALIGNMENT_AREA,
    ], "the proposals nobody was asked"
    assert all(p.unreviewed for p in recorded)
    assert exp.pending_proposals() == []


def test_with_select_poi_off_the_point_is_not_asked(microscope, tmp_path):
    exp = _experiment(tmp_path, microscope, attention=Attention.supervised)
    lamella = exp.positions[0]
    lamella.task_config[SETUP].select_poi = False

    _run(_task(microscope, exp))

    (point,) = _points(lamella)
    assert point.unreviewed and "turned off" in point.current.reason
    assert POINT_OF_INTEREST not in SelectMillingPositionTask.asks_for(
        lamella.task_config[SETUP]
    )


def test_attention_round_trips_through_the_protocol():
    protocol = task_protocol(
        planned(
            name=SETUP,
            attention=Attention.supervised,
            config=SelectMillingPositionTaskConfig(),
        )
    )
    saved = protocol.to_dict()
    assert saved["workflow_config"]["tasks"][0]["attention"] == "supervised"
    again = AutoLamellaTaskProtocol.from_dict(saved)
    assert again.get_attention(SETUP) is Attention.supervised
    assert again.get_attention("nope") is Attention.automated


def _load_protocol(**entry) -> AutoLamellaTaskProtocol:
    """A protocol file whose workflow entry for SETUP carries *entry*."""
    return AutoLamellaTaskProtocol.from_dict(
        {
            "name": "p",
            "tasks": {SETUP: {"task_type": "SELECT_MILLING_POSITION"}},
            "workflow": {"tasks": [{"name": SETUP, "required": True, **entry}]},
        }
    )


def _load_entry(**entry) -> Attention:
    """The attention a protocol file's workflow entry for SETUP reads as."""
    return _load_protocol(**entry).get_attention(SETUP)


def test_a_protocol_saved_with_the_third_mode_still_loads(caplog):
    """Development builds before 0.6.0 had a third mode, stored as ``review``
    and then ``review_later``: the operator deciding afterwards in the Review
    tab. It was Supervised with the wait in a different place, and reads as
    Supervised. A stored value goes through the one reader, so neither of
    those spellings nor one this build does not know takes the whole protocol
    down."""
    for old in ("review", "review_later"):
        assert _load_entry(attention=old) is Attention.supervised
    with caplog.at_level("WARNING"):
        assert _load_entry(attention="gate") is Attention.automated
    assert f"Unknown attention 'gate' on task {SETUP!r}" in caplog.text


def test_a_protocol_written_with_the_supervise_flag_still_loads():
    """v0.5.2 wrote ``supervise`` in place of ``attention``, so every protocol
    and experiment saved by it still loads. The ``review`` flag and the interim
    mode strings that briefly sat beside it never shipped, and are now dropped
    as any unknown key is, rather than mapped."""
    assert _load_entry(supervise=True) is Attention.supervised
    assert _load_entry(supervise=False) is Attention.automated
    assert _load_entry() is Attention.automated, "neither key: automated"
    assert _load_entry(review=True) is Attention.automated
    assert _load_entry(supervise=True, review="gate") is Attention.supervised, (
        "review is dropped; supervise still decides"
    )
    saved = _load_protocol(supervise=True).to_dict()
    assert "supervise" not in saved["workflow_config"]["tasks"][0]
    assert saved["workflow_config"]["tasks"][0]["attention"] == "supervised"
