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
    MILLING_SETUP,
    Decision,
    DecisionOutcome,
)
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskDescription,
    AutoLamellaTaskProtocol,
    AutoLamellaWorkflowConfig,
    Experiment,
)
from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
from fibsem.applications.autolamella.workflows.tasks.rough import MillRoughTaskConfig
from fibsem.applications.autolamella.workflows.tasks.select_position import (
    SelectMillingPositionTask,
    SelectMillingPositionTaskConfig,
    consumed_values,
    propose_milling_setup,
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


def _experiment(tmp_path: Path, microscope, review) -> Experiment:
    exp = Experiment(path=tmp_path, name="test-exp")
    exp.task_protocol = AutoLamellaTaskProtocol(
        workflow_config=AutoLamellaWorkflowConfig(
            tasks=[
                AutoLamellaTaskDescription(
                    name=SETUP, supervise=False, required=True, review=review
                ),
                AutoLamellaTaskDescription(
                    name=ROUGH, supervise=False, required=True, requires=[SETUP]
                ),
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


def test_under_review_the_task_records_a_proposal_and_completes(microscope, tmp_path):
    exp = _experiment(tmp_path, microscope, review=True)
    task = _task(microscope, exp, flag=True)
    assert task.review is True
    lamella = exp.positions[0]
    rough_point = (
        lamella.task_config[ROUGH].milling["mill_rough"].stages[0].pattern.point
    )

    task.run()

    proposal = lamella.proposals[SETUP]
    assert proposal.kind == MILLING_SETUP
    assert proposal.pending
    assert proposal.values == {"poi": Point(0.0, 0.0)}
    assert proposal.confidence is None and proposal.alternatives == []
    assert proposal.provenance["proposer"] == "current-poi"
    assert proposal.provenance["values"] == ["poi"]
    assert proposal.provenance["reference_image"] == (
        f"ref_{SETUP}_final_res_01_ib.tif"
    ), "the final reference image, the last thing acquired at the stored pose"
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
    assert lamella.has_completed_task(SETUP), "the task did all of its work"
    assert lamella.milling_pose is not None

    # The proposal is what the experiment file carries.
    exp.save()
    again = Experiment.load(Path(exp.path) / "experiment.yaml")
    assert again.positions[0].proposals[SETUP].pending


def test_the_proposal_gates_the_consumer_until_it_is_decided(microscope, tmp_path):
    exp = _experiment(tmp_path, microscope, review=True)
    task = _task(microscope, exp, flag=True)
    task.run()
    manager = task.task_manager
    lamella = exp.positions[0]
    assert manager._defer_reason(lamella, ROUGH) == "awaiting_review"

    result = exp.decide(
        lamella.id,
        SETUP,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={"poi": Point(3e-6, 0.0)},
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
    exp = _experiment(tmp_path, microscope, review=True)
    lamella = exp.positions[0]
    lamella.poi = Point(4e-6, -2e-6)
    task = _task(microscope, exp, flag=True)

    task.run()

    proposal = lamella.proposals[SETUP]
    assert proposal.values == {"poi": Point(4e-6, -2e-6)}
    assert proposal.values["poi"] is not lamella.poi, "a copy, not the live point"
    assert lamella.poi == Point(4e-6, -2e-6), "and nothing was written through"


def test_a_deliberate_rerun_supersedes_a_decided_proposal(microscope, tmp_path):
    """Re-running Setup is a deliberate act: the operator gets a new proposal
    on the new image, and the old one -- with its decision -- stays on the
    record. The confirmed point stays on the lamella until the new decision."""
    exp = _experiment(tmp_path, microscope, review=True)
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
        ),
    )
    decided = lamella.proposals[SETUP]

    _task(microscope, exp, flag=True).run()

    fresh = lamella.proposals[SETUP]
    assert fresh is not decided and fresh.pending
    assert fresh.values["poi"] == Point(1e-6, 1e-6), (
        "proposes the point the last decision left on the lamella"
    )
    assert fresh.superseded == [decided]
    assert decided.current.values["poi"] == Point(1e-6, 1e-6)
    assert lamella.poi == Point(1e-6, 1e-6)
    assert task.task_manager._defer_reason(lamella, ROUGH) == "awaiting_review"


def test_without_the_flag_the_proposal_is_recorded_but_never_gates(
    microscope, tmp_path
):
    """The flag hides the Review surface, not the record. A protocol that says
    review runs ungated with it off: the producer confirms its own proposal
    and nothing defers."""
    exp = _experiment(tmp_path, microscope, review=True)
    task = _task(microscope, exp, flag=False)
    assert task.review is False, "gate needs the flag"
    task.run()
    lamella = exp.positions[0]
    proposal = lamella.proposals[SETUP]
    assert proposal.kind == MILLING_SETUP and not proposal.pending
    assert proposal.current.author == "auto:current-poi"
    assert task.task_manager._defer_reason(lamella, ROUGH) is None


def test_automated_the_producer_confirms_its_own_proposal(microscope, tmp_path):
    """Not gated, nobody asked inline: the proposal is recorded exactly as
    under a gate, then confirmed as proposed by the producer, through the
    same decide path a person's confirm takes. The author says nobody looked;
    the run never waits."""
    exp = _experiment(tmp_path, microscope, review=False)
    task = _task(microscope, exp, flag=True)
    assert task.review is False
    lamella = exp.positions[0]
    heard = []
    exp.decided.connect(lambda item_id, task_name: heard.append(task_name))

    task.run()

    proposal = lamella.proposals[SETUP]
    assert not proposal.pending
    assert proposal.values == {"poi": Point(0.0, 0.0)}, "the proposal is untouched"
    assert proposal.current.outcome is DecisionOutcome.Confirmed
    assert proposal.current.author == "auto:current-poi"
    assert proposal.current.via == "workflow"
    assert proposal.current.values == proposal.values, "confirmed as proposed"
    assert proposal.delta() == {"poi": Point(0.0, 0.0)}
    assert heard == [SETUP], "the tab hears it like any other decision"
    assert task.task_manager._defer_reason(lamella, ROUGH) is None, "nothing waits"
    assert lamella.has_completed_task(SETUP)

    # A re-run supersedes the auto-confirmed proposal like a person's.
    _task(microscope, exp, flag=True).run()
    fresh = lamella.proposals[SETUP]
    assert fresh is not proposal and not fresh.pending
    assert fresh.superseded == [proposal]


def test_supervised_the_inline_answer_is_the_decision(
    microscope, tmp_path, monkeypatch
):
    """The operator picked the point in the workflow's own question. That
    answer is the decision on the record, made in the workflow, and the delta
    against the proposer's point is captured without anyone opening the tab.
    Nothing is to check: a person already looked."""
    from fibsem.applications.autolamella.workflows.tasks import select_position as S

    monkeypatch.setattr(S, "select_poi_ui", lambda **kwargs: Point(2e-6, -1e-6))
    exp = _experiment(tmp_path, microscope, review=False)
    task = _task(microscope, exp, flag=True)
    lamella = exp.positions[0]

    task.run()

    proposal = lamella.proposals[SETUP]
    assert proposal.values == {"poi": Point(0.0, 0.0)}, "what the proposer said"
    d = proposal.current
    assert d.outcome is DecisionOutcome.Confirmed and d.via == "workflow"
    assert d.author.startswith("human:")
    assert d.values == {"poi": Point(2e-6, -1e-6)}
    assert proposal.delta()["poi"] == Point(2e-6, -1e-6)
    assert not d.author.startswith("auto:"), "a person decided it"
    assert lamella.poi == Point(2e-6, -1e-6), "applied inline, once"
    assert task.task_manager._defer_reason(lamella, ROUGH) is None


def test_a_value_exists_because_something_consumes_it(tmp_path, microscope):
    exp = _experiment(tmp_path, microscope, review=True)
    lamella = exp.positions[0]
    assert consumed_values(lamella) == ["poi"]
    del lamella.task_config[ROUGH]
    assert consumed_values(lamella) == []
    assert propose_milling_setup(lamella, None) is None, "no consumer, no proposal"


def test_review_round_trips_through_the_protocol():
    d = AutoLamellaTaskDescription(
        name=SETUP, supervise=True, required=True, review=True
    )
    assert d.to_dict()["review"] is True
    again = AutoLamellaTaskDescription.from_dict(d.to_dict())
    assert again.review is True
    old = AutoLamellaTaskDescription.from_dict(
        {"name": SETUP, "supervise": True, "required": True, "requires": []}
    )
    assert old.review is False
    # one interim version wrote a mode string; it still loads
    for legacy, flag in (("gate", True), ("advise", False), ("off", False)):
        interim = AutoLamellaTaskDescription.from_dict(
            {"name": SETUP, "supervise": True, "required": True, "review": legacy}
        )
        assert interim.review is flag, legacy
    cfg_ = AutoLamellaWorkflowConfig(tasks=[d])
    assert cfg_.get_review(SETUP) is True and cfg_.get_review("nope") is False
