"""Grid tasks propose their result and are decided, the way lamella tasks are.

Real overview tasks on the Arctis simulator, run through a real GridTaskManager:
every run records a task_result proposal on the grid pointing at the stitched
overview; automated, the task confirms its own; under review (the protocol says
so and the preference is on), the task ends AwaitingDecision and a decision
finishes it.
"""

import os
from pathlib import Path

import pytest
import yaml

import fibsem.config as cfg
from fibsem import utils
from fibsem.applications.autolamella.proposals import (
    TASK_RESULT,
    AuthorKind,
    Decision,
    DecisionOutcome,
)
from fibsem.applications.autolamella.structures import (
    Attention,
    AutoLamellaTaskProtocol,
    AutoLamellaTaskStatus,
    Experiment,
    GridTaskProtocol,
    Lamella,
)
from fibsem.applications.autolamella.task_outputs import latest_grid_output
from fibsem.applications.autolamella.workflows.tasks.grid import (
    BeamOverviewGridTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.grid.manager import (
    GridTaskManager,
)
from fibsem.structures import BeamType, ImageSettings, OverviewAcquisitionSettings

GRID = "Grid-01"
OVERVIEW = "overview_sem"


class _Signal:
    def emit(self, payload) -> None:
        pass


class _UI:
    """What the manager emits status on, in place of AutoLamellaUI."""

    def __init__(self):
        self.workflow_status_signal = _Signal()
        self._task_manager = None


def _small_settings() -> OverviewAcquisitionSettings:
    return OverviewAcquisitionSettings(
        image_settings=ImageSettings(
            resolution=(128, 128), hfw=200e-6, beam_type=BeamType.ELECTRON
        ),
        nrows=1,
        ncols=1,
    )


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(
        manufacturer="Demo",
        config_path=os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml"),
    )
    return microscope


@pytest.fixture
def experiment(tmp_path, microscope):
    exp = Experiment(path=tmp_path, name="exp")
    (tmp_path / "exp").mkdir()
    exp.task_protocol = AutoLamellaTaskProtocol()
    exp.grid_protocol.add(
        BeamOverviewGridTaskConfig(task_name=OVERVIEW, settings=_small_settings())
    )
    exp.sync_grids_from_inventory(microscope._stage)
    return exp


def _run(microscope, experiment, review_enabled: bool) -> GridTaskManager:
    ui = _UI()
    manager = GridTaskManager(microscope, experiment, parent_ui=ui)
    ui._task_manager = manager
    manager.review_enabled = review_enabled  # the preference, as read for a run
    manager.run([OVERVIEW], [GRID])
    return manager


# ---------------------------------------------------------------------------
# The config
# ---------------------------------------------------------------------------


class TestAttentionOnTheConfig:
    def test_defaults_to_automated_and_is_not_a_form_parameter(self):
        config = BeamOverviewGridTaskConfig(task_name=OVERVIEW)
        assert config.attention is Attention.automated
        assert "attention" not in config.parameters

    def test_round_trips_through_the_protocol(self):
        protocol = GridTaskProtocol()
        protocol.add(
            BeamOverviewGridTaskConfig(task_name=OVERVIEW, attention=Attention.review)
        )
        data = yaml.safe_load(yaml.safe_dump(protocol.to_dict()))
        assert data["tasks"][OVERVIEW]["attention"] == "review"
        again = GridTaskProtocol.from_dict(data)
        assert again.task_config[OVERVIEW].attention is Attention.review

    def test_an_unknown_value_reads_as_automated_and_keeps_the_task(self):
        data = BeamOverviewGridTaskConfig(task_name=OVERVIEW).to_dict()
        data["attention"] = "sometimes"
        again = GridTaskProtocol.from_dict({"tasks": {OVERVIEW: data}})
        assert again.task_config[OVERVIEW].attention is Attention.automated

    def test_a_protocol_written_before_attention_loads_as_automated(self):
        data = BeamOverviewGridTaskConfig(task_name=OVERVIEW).to_dict()
        del data["attention"]
        again = GridTaskProtocol.from_dict({"tasks": {OVERVIEW: data}})
        assert again.task_config[OVERVIEW].attention is Attention.automated


# ---------------------------------------------------------------------------
# A run proposes
# ---------------------------------------------------------------------------


class TestAGridTaskProposes:
    def test_automated_confirms_its_own_result(self, microscope, experiment):
        _run(microscope, experiment, review_enabled=True)
        grid = experiment.get_grid_by_name(GRID)

        assert grid.task_history[-1].status is AutoLamellaTaskStatus.Completed
        proposal = grid.proposals[OVERVIEW]
        assert proposal.kind == TASK_RESULT
        assert not proposal.pending
        assert proposal.decisions[-1].author.kind is AuthorKind.automated

    def test_the_proposal_points_at_the_stitched_overview_not_the_thumbnail(
        self, microscope, experiment
    ):
        _run(microscope, experiment, review_enabled=False)
        grid = experiment.get_grid_by_name(GRID)

        reference = grid.proposals[OVERVIEW].provenance["reference_image"]
        overview = latest_grid_output(experiment, grid, OVERVIEW)
        assert reference.endswith(".tif") and "thumbnail" not in reference
        path = Path(experiment.item_path(grid)) / reference
        # the record keeps a forward slash on every platform; compare the files
        assert path.exists() and os.path.normpath(path) == os.path.normpath(overview)

    def test_under_review_the_task_waits_and_a_decision_finishes_it(
        self, microscope, experiment
    ):
        experiment.grid_protocol.task_config[OVERVIEW].attention = Attention.review
        _run(microscope, experiment, review_enabled=True)
        grid = experiment.get_grid_by_name(GRID)

        assert grid.is_awaiting_decision(OVERVIEW)
        assert grid.task_state.status is AutoLamellaTaskStatus.AwaitingDecision
        assert grid.proposals[OVERVIEW].pending
        assert not grid.has_completed_task(OVERVIEW)

        result = experiment.decide(
            grid.id,
            OVERVIEW,
            Decision(outcome=DecisionOutcome.Confirmed, author="human:op"),
        )
        assert result.applied is True
        assert grid.has_completed_task(OVERVIEW)

    def test_review_in_the_protocol_runs_automated_while_the_preference_is_off(
        self, microscope, experiment
    ):
        experiment.grid_protocol.task_config[OVERVIEW].attention = Attention.review
        _run(microscope, experiment, review_enabled=False)
        grid = experiment.get_grid_by_name(GRID)

        assert grid.task_history[-1].status is AutoLamellaTaskStatus.Completed
        assert not grid.proposals[OVERVIEW].pending

    def test_a_failed_task_proposes_its_failure_and_stays_failed(
        self, microscope, experiment
    ):
        experiment.grid_protocol.task_config[OVERVIEW].attention = Attention.review
        # an orientation the stage does not have: a real failure, inside the
        # task and before any beam
        experiment.grid_protocol.task_config[OVERVIEW].orientation = "NOWHERE"
        grid = experiment.get_grid_by_name(GRID)

        _run(microscope, experiment, review_enabled=True)

        assert grid.task_history[-1].status is AutoLamellaTaskStatus.Failed
        proposal = grid.proposals[OVERVIEW]
        assert proposal.provenance["failure"]
        assert proposal.provenance["reference_image"] == ""
        assert not grid.is_awaiting_decision(OVERVIEW), "a failure does not wait"


def test_item_path_is_a_lamellas_own_and_a_grids_derived(tmp_path, experiment):
    grid = experiment.get_grid_by_name(GRID)
    assert experiment.item_path(grid) == experiment.grid_path(grid)
    lamella = Lamella(path=tmp_path / "lam-01", number=1, petname="lam-01")
    assert experiment.item_path(lamella) == Path(lamella.path)
