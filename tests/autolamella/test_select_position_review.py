"""SelectMillingPositionTask under review: it proposes the point of interest
and completes, instead of asking for it at the beam.

Runs the real task against the Demo microscope, through a real TaskManager
with the feature flag forced on, so the review property is read the way the
application reads it. No Qt: parent_ui is None throughout, which is also
what makes the non-review path a silent no-op today.

The point-of-interest step lives on the coincidence-alignment path, reached
from the SEM orientation with ``auto_milling_alignment`` on, so that is how the
task is run here. The alignment itself is the one thing stubbed: a real
ensure_coincident on the Demo takes a minute per run and is the coincidence
tests' subject, not this file's. The stubs return the real result types.
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


MILLING_ANGLE = 15.0  # deg; the Demo starts flat, so the task tilts to reach it


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo", config_path=CONFIG)
    yield microscope
    microscope.disconnect()


@pytest.fixture(autouse=True)
def cheap_coincidence(monkeypatch):
    """Stand in for the alignment step only, with its real result types."""
    from fibsem.alignment import coincidence, plotting

    def ensure(microscope, reference=None, on_progress=None, **_kw):
        return coincidence.CoincidenceAlignment(
            measurements=[], converged=True, reason=coincidence.REASON_CONVERGED
        )

    def tilt(microscope, target_stage_tilt, reference=None, on_progress=None, **_kw):
        return coincidence.TiltAlignment(
            tilts=[target_stage_tilt],
            alignments=[],
            converged=True,
            reason=coincidence.REASON_CONVERGED,
        )

    monkeypatch.setattr(coincidence, "ensure_coincident", ensure)
    monkeypatch.setattr(coincidence, "tilt_coincident", tilt)
    monkeypatch.setattr(plotting, "save_coincidence_diagnostics", lambda *a, **k: None)


def _experiment(tmp_path: Path, microscope, review: bool) -> Experiment:
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
                    milling_angle=MILLING_ANGLE,
                    auto_milling_alignment=True,
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
    assert proposal.provenance["proposer"] == "centre-of-image"
    assert proposal.provenance["values"] == ["poi"]
    assert proposal.provenance["reference_image"].endswith("_ib.tif")
    assert os.path.exists(proposal.provenance["reference_image"])
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


def test_a_decided_proposal_is_kept_when_the_task_runs_again(microscope, tmp_path):
    """A stall re-queued without Resume must not overwrite the reviewed answer."""
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

    assert lamella.proposals[SETUP] is decided
    assert not lamella.proposals[SETUP].pending
    assert lamella.poi == Point(1e-6, 1e-6)


def test_without_the_flag_or_without_review_nothing_is_proposed(microscope, tmp_path):
    exp = _experiment(tmp_path, microscope, review=True)
    task = _task(microscope, exp, flag=False)
    assert task.review is False
    task.run()
    assert exp.positions[0].proposals == {}

    exp = _experiment(tmp_path / "b", microscope, review=False)
    task = _task(microscope, exp, flag=True)
    assert task.review is False
    task.run()
    assert exp.positions[0].proposals == {}


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
    again = AutoLamellaTaskDescription.from_dict(d.to_dict())
    assert again.review is True
    old = AutoLamellaTaskDescription.from_dict(
        {"name": SETUP, "supervise": True, "required": True, "requires": []}
    )
    assert old.review is False
    cfg_ = AutoLamellaWorkflowConfig(tasks=[d])
    assert cfg_.get_review(SETUP) is True and cfg_.get_review("nope") is False
