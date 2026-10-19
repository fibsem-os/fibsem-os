"""Grid tasks propose and are decided, the way lamella tasks are.

Real overview tasks on the Arctis simulator, run through a real GridTaskManager:
every run records a proposal on the grid pointing at the stitched overview,
and its values are used as proposed. Supervised (the protocol says so) asks
its questions while it runs; nothing waits on a decision once it has ended.

An overview proposes ``overview_positions`` -- where the lamellae go on this
grid -- rather than a bare result, because that is the question the image is
acquired to answer, and confirming it creates them (FIB-1015). These tests
confirm with none placed, which is a real answer and keeps them about the
run's lifecycle rather than about placing.
"""

import os
from pathlib import Path

import pytest
import yaml

import fibsem.config as cfg
from fibsem import utils
from fibsem.acting import ActorKind
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
    Lamella,
)
from fibsem.structures import BeamType, ImageSettings, OverviewAcquisitionSettings
from fibsem.workflows.attention import Attention
from fibsem.workflows.grid import (
    BeamOverviewGridTaskConfig,
)
from fibsem.workflows.grid.manager import (
    LOAD_ENTRY_NAME,
    GridTaskManager,
)
from fibsem.workflows.grid.protocol import GridTaskProtocol
from fibsem.workflows.outputs import latest_item_output
from fibsem.workflows.proposals import (
    OVERVIEW_POSITIONS,
    Decision,
    DecisionOutcome,
    Standing,
)
from fibsem.workflows.review import standing
from fibsem.workflows.status import TaskStatus
from tests.run_state import stub_session

GRID = "Grid-01"
OTHER_GRID = "Grid-02"
OVERVIEW = "overview_sem"
LATER = "overview_fib"


class _Signal:
    def emit(self, payload) -> None:
        pass


class _UI:
    """What the manager emits status on, in place of AutoLamellaUI."""

    def __init__(self):
        self.workflow_status_signal = _Signal()
        self.session = stub_session()


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
    exp.sync_grids_from_inventory(microscope.sample_handling)
    return exp


def _run(microscope, experiment) -> GridTaskManager:
    ui = _UI()
    manager = GridTaskManager(microscope, experiment, parent_ui=ui)
    ui.session.runner.manager = manager
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
            BeamOverviewGridTaskConfig(
                task_name=OVERVIEW, attention=Attention.supervised
            )
        )
        data = yaml.safe_load(yaml.safe_dump(protocol.to_dict()))
        assert data["tasks"][OVERVIEW]["attention"] == "supervised"
        again = GridTaskProtocol.from_dict(data)
        assert again.task_config[OVERVIEW].attention is Attention.supervised

    def test_a_protocol_saved_before_the_rename_still_loads(self):
        """The mode was stored as ``review`` on main before 0.6.0. It never
        shipped, but grid protocols saved from those builds carry it."""
        protocol = GridTaskProtocol()
        protocol.add(
            BeamOverviewGridTaskConfig(
                task_name=OVERVIEW, attention=Attention.supervised
            )
        )
        data = yaml.safe_load(yaml.safe_dump(protocol.to_dict()))
        data["tasks"][OVERVIEW]["attention"] = "review"
        again = GridTaskProtocol.from_dict(data)
        assert again.task_config[OVERVIEW].attention is Attention.supervised

    def test_an_unknown_value_reads_as_automated_and_keeps_the_task(self):
        data = BeamOverviewGridTaskConfig(task_name=OVERVIEW).to_dict()
        data["attention"] = "sometimes"
        again = GridTaskProtocol.from_dict({"tasks": {OVERVIEW: data}})
        assert again.task_config[OVERVIEW].attention is Attention.automated

    def test_requires_round_trips_and_is_not_a_form_parameter(self):
        protocol = GridTaskProtocol()
        protocol.add(BeamOverviewGridTaskConfig(task_name=OVERVIEW))
        protocol.add(BeamOverviewGridTaskConfig(task_name=LATER, requires=[OVERVIEW]))
        assert "requires" not in protocol.task_config[LATER].parameters
        again = GridTaskProtocol.from_dict(
            yaml.safe_load(yaml.safe_dump(protocol.to_dict()))
        )
        assert again.requirements(LATER) == [OVERVIEW]
        assert again.requirements(OVERVIEW) == []
        assert again.requirements("not a task") == []

    def test_a_malformed_requires_reads_as_none_and_keeps_the_task(self):
        data = BeamOverviewGridTaskConfig(task_name=LATER).to_dict()
        data["requires"] = OVERVIEW  # a string, not a list
        again = GridTaskProtocol.from_dict({"tasks": {LATER: data}})
        assert again.requirements(LATER) == []

    def test_a_protocol_written_before_attention_loads_as_automated(self):
        data = BeamOverviewGridTaskConfig(task_name=OVERVIEW).to_dict()
        del data["attention"]
        again = GridTaskProtocol.from_dict({"tasks": {OVERVIEW: data}})
        assert again.task_config[OVERVIEW].attention is Attention.automated


# ---------------------------------------------------------------------------
# A run proposes
# ---------------------------------------------------------------------------


class TestAGridTaskProposes:
    def test_automated_leaves_its_positions_open_for_the_task_that_uses_them(
        self, microscope, experiment
    ):
        """Nobody is asked and nobody confirms: the positions are used as
        proposed and stay open to correct until the task that consumes them
        starts. The run ending does not close them; a run of the overview on
        its own is how they get looked at before screening."""
        _run(microscope, experiment)
        grid = experiment.get_grid_by_name(GRID)

        assert grid.task_history[-1].status is TaskStatus.Completed
        proposal = grid.proposal(OVERVIEW)
        assert proposal.kind == OVERVIEW_POSITIONS
        assert standing(grid, OVERVIEW, proposal) is Standing.Open

    def test_the_proposal_points_at_the_stitched_overview_not_the_thumbnail(
        self, microscope, experiment
    ):
        _run(microscope, experiment)
        grid = experiment.get_grid_by_name(GRID)

        reference = grid.proposal(OVERVIEW).provenance["reference_image"]
        overview = latest_item_output(experiment, grid, OVERVIEW)
        assert reference.endswith(".tif") and "thumbnail" not in reference
        path = Path(experiment.item_path(grid)) / reference
        # the record keeps a forward slash on every platform; compare the files
        assert path.exists() and os.path.normpath(path) == os.path.normpath(overview)

    def test_supervised_the_task_completes_and_its_positions_can_be_decided(
        self, microscope, experiment
    ):
        """Nothing waits once the task has ended: it is Completed, its
        positions used as proposed, and a decision on them is a record."""
        experiment.grid_protocol.task_config[OVERVIEW].attention = Attention.supervised
        _run(microscope, experiment)
        grid = experiment.get_grid_by_name(GRID)

        assert grid.task_state.status is TaskStatus.Completed
        assert grid.has_completed_task(OVERVIEW)
        assert grid.proposal(OVERVIEW).pending
        assert grid.proposal(OVERVIEW).task_id == grid.task_history[-1].task_id

        result = experiment.decide(
            grid.id,
            OVERVIEW,
            Decision(
                outcome=DecisionOutcome.Confirmed,
                actor="operator:op",
                values={"positions": []},
                task_id=grid.proposal(OVERVIEW).task_id,
            ),
        )
        assert result.applied is True
        assert grid.has_completed_task(OVERVIEW)
        assert not grid.proposal(OVERVIEW).pending

    def test_a_failed_task_proposes_its_failure_and_stays_failed(
        self, microscope, experiment
    ):
        experiment.grid_protocol.task_config[OVERVIEW].attention = Attention.supervised
        # an orientation the stage does not have: a real failure, inside the
        # task and before any beam
        experiment.grid_protocol.task_config[OVERVIEW].orientation = "NOWHERE"
        grid = experiment.get_grid_by_name(GRID)

        _run(microscope, experiment)

        assert grid.task_history[-1].status is TaskStatus.Failed
        proposal = grid.proposal(OVERVIEW)
        assert proposal.provenance["failure"]
        assert proposal.provenance["reference_image"] == ""


def test_item_path_is_a_lamellas_own_and_a_grids_derived(tmp_path, experiment):
    grid = experiment.get_grid_by_name(GRID)
    assert experiment.item_path(grid) == experiment.grid_path(grid)
    lamella = Lamella(path=tmp_path / "lam-01", number=1, petname="lam-01")
    assert experiment.item_path(lamella) == Path(lamella.path)


# ---------------------------------------------------------------------------
# A task that requires another (FIB-1002 PR 4)
# ---------------------------------------------------------------------------


def _with_later_task(experiment, requires=(OVERVIEW,)):
    """The SEM overview supervised, then an automated FIB overview that
    requires it (a stand-in for a task that uses the overview)."""
    experiment.grid_protocol.task_config[OVERVIEW].attention = Attention.supervised
    experiment.grid_protocol.add(
        BeamOverviewGridTaskConfig(
            task_name=LATER,
            requires=list(requires),
            orientation="FIB",
            settings=OverviewAcquisitionSettings(
                image_settings=ImageSettings(
                    resolution=(128, 128), hfw=200e-6, beam_type=BeamType.ION
                ),
                nrows=1,
                ncols=1,
            ),
        )
    )


def _manager(microscope, experiment, hook_manager=None) -> GridTaskManager:
    return GridTaskManager(microscope, experiment, hook_manager=hook_manager)


def _ran(grid, task_name):
    return [t for t in grid.task_history if t.name == task_name]


class TestATaskThatRequiresAnother:
    def test_a_supervised_requirement_holds_nothing_once_it_has_ended(
        self, microscope, experiment
    ):
        """The supervised overview completes; the task that requires it runs
        straight after, with no decision in between."""
        _with_later_task(experiment)
        manager = _manager(microscope, experiment)

        manager.run([OVERVIEW, LATER], [GRID])

        grid = experiment.get_grid_by_name(GRID)
        assert grid.has_completed_task(OVERVIEW)
        assert [t.status for t in _ran(grid, LATER)] == [TaskStatus.Completed]
        assert not manager.stalled and manager.deferred_items() == []

    def test_a_requirement_queued_later_runs_first(self, microscope, experiment):
        """Selected in the other order, the task waits for its requirement's
        turn (prereq_pending) instead of skipping over it."""
        _with_later_task(experiment)
        experiment.grid_protocol.task_config[OVERVIEW].attention = Attention.automated
        manager = _manager(microscope, experiment)

        manager.run([LATER, OVERVIEW], [GRID])

        grid = experiment.get_grid_by_name(GRID)
        assert grid.has_completed_task(OVERVIEW) and grid.has_completed_task(LATER)
        assert _ran(grid, OVERVIEW)[-1].end_timestamp <= (
            _ran(grid, LATER)[-1].start_timestamp
        )

    def test_a_failed_requirement_skips_the_task_without_waiting(
        self, microscope, experiment
    ):
        _with_later_task(experiment)
        # an orientation the stage does not have: the overview fails in the task
        experiment.grid_protocol.task_config[OVERVIEW].orientation = "NOWHERE"
        manager = _manager(microscope, experiment)

        manager.run([OVERVIEW, LATER], [GRID])

        grid = experiment.get_grid_by_name(GRID)
        assert _ran(grid, OVERVIEW)[-1].status is TaskStatus.Failed
        assert _ran(grid, LATER) == []
        assert not manager.stalled


# ---------------------------------------------------------------------------
# The latest run of a requirement counts (FIB-1006)
# ---------------------------------------------------------------------------


class TestTheLatestRunOfARequirementCounts:
    def _succeed_once(self, microscope, experiment):
        _with_later_task(experiment)
        experiment.grid_protocol.task_config[OVERVIEW].attention = Attention.automated
        _manager(microscope, experiment).run([OVERVIEW], [GRID])
        grid = experiment.get_grid_by_name(GRID)
        assert grid.has_completed_task(OVERVIEW)
        return grid

    def test_a_failed_rerun_skips_the_task_that_requires_it(
        self, microscope, experiment
    ):
        grid = self._succeed_once(microscope, experiment)
        experiment.grid_protocol.task_config[OVERVIEW].orientation = "NOWHERE"

        _manager(microscope, experiment).run([OVERVIEW, LATER], [GRID])

        assert _ran(grid, OVERVIEW)[-1].status is TaskStatus.Failed
        assert _ran(grid, LATER) == [], "the old success did not license it"

    def test_a_rerun_queued_after_its_consumer_is_waited_for(
        self, microscope, experiment
    ):
        grid = self._succeed_once(microscope, experiment)
        experiment.grid_protocol.task_config[OVERVIEW].orientation = "NOWHERE"
        manager = _manager(microscope, experiment)

        manager.run([LATER, OVERVIEW], [GRID])

        assert _ran(grid, OVERVIEW)[-1].status is TaskStatus.Failed
        assert _ran(grid, LATER) == [], "waited for the rerun, then skipped"
        (later,) = [i for i in manager.queue.items if i.task_name == LATER]
        assert later.status is TaskStatus.Skipped


# ---------------------------------------------------------------------------
# A grid is loaded for work that can run (FIB-1005)
# ---------------------------------------------------------------------------


def _loaded(microscope, name):
    return microscope.sample_handling.holder.find_slot_by_grid_name(name) is not None


class TestAGridIsLoadedForWorkThatCanRun:
    def test_no_exchange_when_every_selected_task_will_be_skipped(
        self, microscope, experiment
    ):
        """Grid-01 is in; on Grid-02 the FIB overview requires an SEM overview
        that never ran. Nothing there can run, so Grid-01 stays in."""
        _with_later_task(experiment)
        _manager(microscope, experiment).run([OVERVIEW], [GRID])
        assert _loaded(microscope, GRID)
        second = experiment.get_grid_by_name(OTHER_GRID)

        manager = _manager(microscope, experiment)
        manager.run([LATER], [OTHER_GRID])

        assert _loaded(microscope, GRID) and not _loaded(microscope, OTHER_GRID)
        assert _ran(second, LOAD_ENTRY_NAME) == [], "no exchange was attempted"
        statuses = {i.task_name: i.status for i in manager.queue.items}
        assert statuses == {
            LOAD_ENTRY_NAME: TaskStatus.Skipped,
            LATER: TaskStatus.Skipped,
        }
        assert not manager.stalled

    def test_a_load_with_nothing_behind_it_still_loads(self, microscope, experiment):
        _with_later_task(experiment)
        manager = _manager(microscope, experiment)
        manager.queue.build_from_pairs([(OTHER_GRID, LOAD_ENTRY_NAME)])
        manager._run_queue()
        assert _loaded(microscope, OTHER_GRID)
