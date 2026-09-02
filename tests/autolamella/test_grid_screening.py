"""'Screen all grids': inventory, every present grid, the protocol, one call.

On the Arctis simulator that is three exchanges; on a fixed holder with two
calibrated, occupied slots it is the same call with none.
"""

import os

import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus as Status
from fibsem.applications.autolamella.workflows.tasks.grid import (
    BeamOverviewGridTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.grid.manager import (
    LOAD_ENTRY_NAME,
    GridTaskManager,
)
from fibsem.applications.autolamella.workflows.tasks.grid.screening import (
    screen_grids,
    screening_plan,
)
from fibsem.microscopes._stage import (
    SampleGrid,
    SlotCalibration,
    _create_sample_stage,
)
from fibsem.structures import (
    BeamType,
    FibsemStagePosition,
    ImageSettings,
    OverviewAcquisitionSettings,
)


def _small_settings(beam=BeamType.ELECTRON) -> OverviewAcquisitionSettings:
    return OverviewAcquisitionSettings(
        image_settings=ImageSettings(resolution=(128, 128), hfw=200e-6, beam_type=beam),
        nrows=1,
        ncols=1,
    )


@pytest.fixture
def arctis():
    microscope, _ = utils.setup_session(
        manufacturer="Demo",
        config_path=os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml"),
    )
    return microscope


@pytest.fixture
def fixed_holder():
    """A plain Demo with a two-slot holder, both calibrated and occupied."""
    microscope, _ = utils.setup_session(manufacturer="Demo")
    microscope.stage_is_compustage = False
    microscope._stage = _create_sample_stage(microscope)
    holder = microscope._stage.holder
    for i, (name, x) in enumerate([("grid-aspen", -4e-3), ("grid-birch", 4e-3)]):
        slot = holder.slots[f"Slot-{i + 1:02d}"]
        slot.position = FibsemStagePosition(name=slot.name, x=x, y=0, z=4e-3, r=0, t=0)
        slot.calibration = SlotCalibration("SEM", 35.0, 0.0, "2026-09-02T11:24:09", "t")
        slot.loaded_grid = SampleGrid(name=name)
    return microscope


@pytest.fixture
def experiment(tmp_path):
    exp = Experiment(path=tmp_path, name="exp")
    (tmp_path / "exp").mkdir()
    exp.task_protocol = AutoLamellaTaskProtocol()
    exp.grid_protocol.add(
        BeamOverviewGridTaskConfig(task_name="overview_sem", settings=_small_settings())
    )
    exp.grid_protocol.add(
        BeamOverviewGridTaskConfig(
            task_name="overview_fib",
            orientation="FIB",
            settings=_small_settings(BeamType.ION),
        )
    )
    return exp


@pytest.fixture
def stub_tasks(monkeypatch):
    """Stand in for the task itself, so these tests are about the composition."""
    executed = []

    def _run_single_task(self, task_name, grid):
        executed.append((grid.name, task_name))
        grid.task_state.name = task_name
        grid.task_state.status = Status.Completed
        return None

    monkeypatch.setattr(GridTaskManager, "_run_single_task", _run_single_task)
    return executed


class TestOnTheAutoloader:
    def test_inventories_then_runs_the_protocol_on_every_grid(
        self, arctis, experiment, stub_tasks
    ):
        assert experiment.grids == []
        manager = screen_grids(arctis, experiment)
        assert [g.name for g in experiment.grids] == ["Grid-01", "Grid-02", "Grid-03"]
        assert stub_tasks == [
            (g, t)
            for g in ("Grid-01", "Grid-02", "Grid-03")
            for t in ("overview_sem", "overview_fib")
        ]
        assert all(i.status is Status.Completed for i in manager.queue.items)
        for grid in experiment.grids:
            assert [t.name for t in grid.task_history if t.name == LOAD_ENTRY_NAME] == [
                LOAD_ENTRY_NAME
            ]

    def test_screening_again_adds_no_records(self, arctis, experiment, stub_tasks):
        screen_grids(arctis, experiment)
        screen_grids(arctis, experiment)
        assert len(experiment.grids) == 3

    def test_the_plan_is_what_a_confirmation_shows(self, arctis, experiment):
        plan = screening_plan(arctis, experiment)
        assert plan[:3] == [
            ("Grid-01", "load"),
            ("Grid-01", "overview_sem"),
            ("Grid-01", "overview_fib"),
        ]
        assert len(plan) == 9
        assert screening_plan(arctis, experiment, ["overview_fib"]) == [
            (g, s)
            for g in ("Grid-01", "Grid-02", "Grid-03")
            for s in ("load", "overview_fib")
        ]

    def test_a_grid_that_will_not_load_does_not_stop_the_others(
        self, arctis, experiment, stub_tasks
    ):
        arctis._stage.loader.fail_next_exchange = True
        manager = screen_grids(arctis, experiment, task_names=["overview_sem"])
        assert stub_tasks == [("Grid-02", "overview_sem"), ("Grid-03", "overview_sem")]
        assert [i.status for i in manager.queue.items if i.item_name == "Grid-01"] == [
            Status.Failed,
            Status.Skipped,
        ]


class TestOnAFixedHolder:
    def test_every_grid_is_present_and_nothing_is_exchanged(
        self, fixed_holder, experiment, stub_tasks
    ):
        manager = screen_grids(fixed_holder, experiment, task_names=["overview_sem"])
        assert [g.name for g in experiment.grids] == ["grid-aspen", "grid-birch"]
        assert stub_tasks == [
            ("grid-aspen", "overview_sem"),
            ("grid-birch", "overview_sem"),
        ]
        # the load steps complete without an exchange, and leave no entry
        loads = [i for i in manager.queue.items if i.task_name == LOAD_ENTRY_NAME]
        assert [i.status for i in loads] == [Status.Completed, Status.Completed]
        for grid in experiment.grids:
            assert not any(t.name == LOAD_ENTRY_NAME for t in grid.task_history)

    def test_nothing_to_screen_is_an_empty_run(self, experiment, stub_tasks):
        microscope, _ = utils.setup_session(manufacturer="Demo")
        microscope.stage_is_compustage = False
        microscope._stage = _create_sample_stage(microscope)
        for slot in microscope._stage.holder.slots.values():
            slot.loaded_grid = None
        manager = screen_grids(microscope, experiment)
        assert experiment.grids == [] and stub_tasks == []
        assert manager.queue.items == []
