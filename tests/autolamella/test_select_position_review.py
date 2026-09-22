"""SelectMillingPositionTask under review: it proposes the point of interest
and completes, instead of asking for it at the beam.

Runs the real task against the Demo microscope, through a real TaskManager
with the feature flag forced on, so the review property is read the way the
application reads it. No Qt: parent_ui is None throughout, which is also
what makes the non-review path a silent no-op today.
"""

import os
from pathlib import Path

import pytest
from psygnal.containers import EventedDict

import fibsem.config as cfg
from fibsem import utils
from fibsem.applications.autolamella.proposals import (
    POINT_OF_INTEREST,
    AuthorKind,
    Decision,
    DecisionOutcome,
)
from fibsem.applications.autolamella.structures import (
    Attention,
    AutoLamellaTaskDescription,
    AutoLamellaTaskProtocol,
    AutoLamellaTaskStatus,
    AutoLamellaWorkflowConfig,
    Experiment,
)
from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
from fibsem.applications.autolamella.workflows.tasks.rough import MillRoughTaskConfig
from fibsem.applications.autolamella.workflows.tasks.select_position import (
    CurrentPoiProposer,
    SelectMillingPositionTask,
    SelectMillingPositionTaskConfig,
    consumed_values,
)
from fibsem.structures import Point

SETUP = "Setup Lamella Position"
ROUGH = "Rough Milling"
CONFIG = os.path.join(cfg.CONFIG_PATH, "microscope-configuration.yaml")


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo", config_path=CONFIG)
    yield microscope
    microscope.disconnect()


def _experiment(
    tmp_path: Path, microscope, attention: Attention = Attention.automated
) -> Experiment:
    exp = Experiment(path=tmp_path, name="test-exp")
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


def _task(microscope, exp: Experiment, flag: bool) -> SelectMillingPositionTask:
    manager = TaskManager(microscope=microscope, experiment=exp, parent_ui=None)
    manager.review_enabled = flag
    lamella = exp.positions[0]
    return SelectMillingPositionTask(
        microscope=microscope,
        config=lamella.task_config[SETUP],
        lamella=lamella,
        parent_ui=None,
        task_manager=manager,
    )


def test_under_review_the_task_records_a_proposal_and_awaits_a_decision(
    microscope, tmp_path
):
    exp = _experiment(tmp_path, microscope, attention=Attention.supervised)
    task = _task(microscope, exp, flag=True)
    assert task.review is True
    lamella = exp.positions[0]
    rough_point = (
        lamella.task_config[ROUGH].milling["mill_rough"].stages[0].pattern.point
    )

    task.run()

    proposal = lamella.proposal(SETUP)
    assert proposal.kind == POINT_OF_INTEREST
    assert proposal.pending
    assert proposal.values == {"poi": Point(0.0, 0.0)}
    assert proposal.confidence is None and proposal.alternatives == []
    assert proposal.provenance["proposer"] == "current-poi"
    assert proposal.provenance["reference_image"] == (
        f"ref_{SETUP}_final_res_02_ib.tif"
    ), "the last final reference image: the tightest field of view, at the stored pose"
    assert not os.path.isabs(proposal.provenance["reference_image"]), (
        "relative to the lamella folder, so a moved experiment still resolves"
    )
    assert os.path.exists(
        os.path.join(str(lamella.path), proposal.provenance["reference_image"])
    )
    # Nothing was written through: the point and the patterns wait for a decision.
    assert lamella.poi == Point(0.0, 0.0)
    assert (
        lamella.task_config[ROUGH].milling["mill_rough"].stages[0].pattern.point
        == rough_point
    )
    assert lamella.milling_pose is not None, "the task did all of its work"
    assert lamella.is_awaiting_decision(SETUP) and not lamella.has_completed_task(
        SETUP
    ), "but it is not finished until someone decides"
    assert lamella.task_state.status is AutoLamellaTaskStatus.AwaitingDecision

    # The proposal is what the experiment file carries.
    exp.save()
    again = Experiment.load(Path(exp.path) / "experiment.yaml")
    assert again.positions[0].proposal(SETUP).pending


def test_the_proposal_gates_the_consumer_until_it_is_decided(microscope, tmp_path):
    exp = _experiment(tmp_path, microscope, attention=Attention.supervised)
    task = _task(microscope, exp, flag=True)
    task.run()
    manager = task.task_manager
    lamella = exp.positions[0]
    assert manager._defer_reason(lamella, ROUGH) == "awaiting_decision"

    result = exp.decide(
        lamella.id,
        SETUP,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={"poi": Point(3e-6, 0.0)},
            task_id=lamella.proposal(SETUP).task_id,
        ),
    )
    assert result.applied and result.delta["poi"] == Point(3e-6, 0.0)
    assert manager._defer_reason(lamella, ROUGH) is None
    assert lamella.poi == Point(3e-6, 0.0)


def test_the_proposal_carries_the_point_something_else_already_set(
    microscope, tmp_path
):
    """Correlation, a script or an agent may have positioned the point before
    Setup runs. The proposer proposes that point rather than the image centre,
    so recording the proposal and confirming it changes nothing: whatever set
    the point is not undone by the step that is meant to check it."""
    exp = _experiment(tmp_path, microscope, attention=Attention.supervised)
    lamella = exp.positions[0]
    lamella.poi = Point(4e-6, -2e-6)
    task = _task(microscope, exp, flag=True)

    task.run()

    proposal = lamella.proposal(SETUP)
    assert proposal.values == {"poi": Point(4e-6, -2e-6)}
    assert proposal.values["poi"] is not lamella.poi, "a copy, not the live point"
    assert lamella.poi == Point(4e-6, -2e-6), "and nothing was written through"


def test_a_deliberate_rerun_supersedes_a_decided_proposal(microscope, tmp_path):
    """Re-running Setup is a deliberate act: the operator gets a new proposal
    on the new image, and the old one -- with its decision -- stays on the
    record. The confirmed point stays on the lamella until the new decision."""
    exp = _experiment(tmp_path, microscope, attention=Attention.supervised)
    task = _task(microscope, exp, flag=True)
    task.run()
    lamella = exp.positions[0]
    exp.decide(
        lamella.id,
        SETUP,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={"poi": Point(1e-6, 1e-6)},
            task_id=lamella.proposal(SETUP).task_id,
        ),
    )
    decided = lamella.proposal(SETUP)

    _task(microscope, exp, flag=True).run()

    fresh = lamella.proposal(SETUP)
    assert fresh is not decided and fresh.pending
    assert fresh.values["poi"] == Point(1e-6, 1e-6), (
        "proposes the point the last decision left on the lamella"
    )
    assert lamella.proposals[SETUP] == [decided, fresh]
    assert decided.current.values["poi"] == Point(1e-6, 1e-6)
    assert lamella.poi == Point(1e-6, 1e-6)
    assert task.task_manager._defer_reason(lamella, ROUGH) == "awaiting_decision"


def test_without_the_flag_the_proposal_is_recorded_but_never_gates(
    microscope, tmp_path
):
    """The flag hides the Review surface, not the record. A protocol that says
    supervised runs ungated with it off: the proposal is recorded, open, and
    nothing defers."""
    exp = _experiment(tmp_path, microscope, attention=Attention.supervised)
    task = _task(microscope, exp, flag=False)
    assert task.review is False, "gate needs the flag"
    task.run()
    lamella = exp.positions[0]
    proposal = lamella.proposal(SETUP)
    assert proposal.kind == POINT_OF_INTEREST and proposal.pending
    assert task.task_manager._defer_reason(lamella, ROUGH) is None


def test_automated_the_value_is_open_until_its_consumer_starts(microscope, tmp_path):
    """Not gated, nobody asked inline: the proposal is recorded exactly as
    under a gate, its values are live from that moment, and it stays open to
    correct. Nobody confirms it; when the task that uses it starts it is
    closed as Unreviewed, which is never agreement. The run never waits."""
    exp = _experiment(tmp_path, microscope)
    task = _task(microscope, exp, flag=True)
    assert task.review is False
    lamella = exp.positions[0]
    heard = []
    exp.decided.connect(lambda item_id, task_name: heard.append(task_name))

    task.run()

    proposal = lamella.proposal(SETUP)
    assert proposal.pending, "open"
    assert proposal.values == {"poi": Point(0.0, 0.0)}, "the proposal is untouched"
    assert lamella.poi == Point(0.0, 0.0), "live as proposed"
    assert heard == [], "nothing was decided"
    assert task.task_manager._defer_reason(lamella, ROUGH) is None, "nothing waits"
    assert lamella.has_completed_task(SETUP)
    assert [t for _i, t, _p in exp.proposals_to_check()] == [SETUP]

    # A correction before the consumer starts is a plain confirm with values.
    moved = exp.decide(
        lamella.id,
        SETUP,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={"poi": Point(1e-6, 0.0)},
            proposal_id=proposal.id,
        ),
    )
    assert moved.applied, moved.reason
    assert lamella.poi == Point(1e-6, 0.0)
    assert heard == [SETUP]

    # A re-run of the task leaves the decided one on the record before it.
    _task(microscope, exp, flag=True).run()
    fresh = lamella.proposal(SETUP)
    assert fresh is not proposal and fresh.pending
    assert lamella.proposals[SETUP] == [proposal, fresh]

    # A re-run over an open one closes it as used-unreviewed, not dropped.
    _task(microscope, exp, flag=True).run()
    assert [p.unreviewed for p in lamella.proposals[SETUP]] == [False, True, False]
    assert "re-ran" in lamella.proposals[SETUP][1].current.reason


def test_supervised_the_inline_answer_is_the_decision(
    microscope, tmp_path, monkeypatch
):
    """The operator picked the point in the workflow's own question. That
    answer is the decision on the record, made in the workflow, and the delta
    against the proposer's point is captured without anyone opening the tab.
    Nothing is to check: a person already looked."""
    from fibsem.applications.autolamella.workflows.tasks import select_position as S

    monkeypatch.setattr(S, "select_poi_ui", lambda **kwargs: Point(2e-6, -1e-6))
    exp = _experiment(tmp_path, microscope)
    task = _task(microscope, exp, flag=True)
    lamella = exp.positions[0]

    task.run()

    proposal = lamella.proposal(SETUP)
    assert proposal.values == {"poi": Point(0.0, 0.0)}, "what the proposer said"
    d = proposal.current
    assert d.outcome is DecisionOutcome.Confirmed and d.via == "workflow"
    assert d.author.kind is AuthorKind.human
    assert d.values == {"poi": Point(2e-6, -1e-6)}
    assert proposal.delta()["poi"] == Point(2e-6, -1e-6)
    assert d.author.kind is not AuthorKind.automated, "a person decided it"
    assert lamella.poi == Point(2e-6, -1e-6), "applied inline, once"
    assert task.task_manager._defer_reason(lamella, ROUGH) is None


def test_a_value_exists_because_something_consumes_it(tmp_path, microscope):
    exp = _experiment(tmp_path, microscope, attention=Attention.supervised)
    lamella = exp.positions[0]
    assert consumed_values(lamella) == ["poi"]
    del lamella.task_config[ROUGH]
    assert consumed_values(lamella) == []
    task = _task(microscope, exp, flag=True)
    assert CurrentPoiProposer().propose(task) is None, "no consumer, no proposal"


def test_attention_round_trips_through_the_protocol():
    d = AutoLamellaTaskDescription(
        name=SETUP, required=True, attention=Attention.supervised
    )
    assert d.to_dict()["attention"] == "supervised"
    again = AutoLamellaTaskDescription.from_dict(d.to_dict())
    assert again.attention is Attention.supervised
    assert AutoLamellaTaskDescription(name=SETUP, attention="supervised").attention is (
        Attention.supervised
    ), "a string from a hand-edited file is the enum"
    cfg_ = AutoLamellaWorkflowConfig(tasks=[d])
    assert cfg_.get_attention(SETUP) is Attention.supervised
    assert cfg_.get_attention("nope") is Attention.automated


def test_a_protocol_saved_with_the_third_mode_still_loads(caplog):
    """Development builds before 0.6.0 had a third mode, stored as ``review``
    and then ``review_later``: the operator deciding afterwards in the Review
    tab. It was Supervised with the wait in a different place, and reads as
    Supervised. A stored value goes through the one reader, so neither of
    those spellings nor one this build does not know takes the whole protocol
    down."""
    load = AutoLamellaTaskDescription.from_dict
    base = {"name": SETUP, "required": True, "requires": []}
    for old in ("review", "review_later"):
        assert load({**base, "attention": old}).attention is Attention.supervised
        assert load({**base, "attention": old}).to_dict()["attention"] == (
            "supervised"
        ), "and it is written back as supervised"
    with caplog.at_level("WARNING"):
        assert load({**base, "attention": "gate"}).attention is Attention.automated
    assert f"Unknown attention 'gate' on task {SETUP!r}" in caplog.text


def test_a_protocol_written_with_the_supervise_flag_still_loads():
    """v0.5.2 wrote ``supervise`` in place of ``attention``, so every protocol
    and experiment saved by it still loads. The ``review`` flag and the interim
    mode strings that briefly sat beside it never shipped, and are now dropped
    as any unknown key is, rather than mapped."""
    load = AutoLamellaTaskDescription.from_dict
    base = {"name": SETUP, "required": True, "requires": []}
    assert load({**base, "supervise": True}).attention is Attention.supervised
    assert load({**base, "supervise": False}).attention is Attention.automated
    assert load(base).attention is Attention.automated, "neither key: automated"
    assert load({**base, "review": True}).attention is Attention.automated
    assert load({**base, "supervise": True, "review": "gate"}).attention is (
        Attention.supervised
    ), "review is dropped; supervise still decides"
    assert "supervise" not in load({**base, "supervise": True}).to_dict()
