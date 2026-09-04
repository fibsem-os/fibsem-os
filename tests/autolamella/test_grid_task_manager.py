"""The grid run loop, on the Arctis simulator (a compustage with a Demo autoloader).

Grid-outer: a grid is exchanged once and all of its tasks run while it is in the
beam. A grid that will not load is recorded, skipped, and the run continues; a
task that fails fails only itself. Most tests stub the task itself, as the lamella
manager's tests do, so they exercise the loop and not the microscope; the last
runs real overview tasks end to end.
"""

import os
from typing import List

import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
    GridRecord,
)
from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus as Status
from fibsem.applications.autolamella.task_outputs import grid_outputs
from fibsem.applications.autolamella.workflows.tasks.grid import (
    BeamOverviewGridTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.grid.manager import (
    LOAD_ENTRY_NAME,
    SKIP_GRID_NOT_FOUND,
    SKIP_GRID_NOT_LOADED,
    GridTaskManager,
    plan_grid_run,
    run_grid_tasks,
)
from fibsem.applications.autolamella.workflows.tasks.status import WorkflowStatusEvent
from fibsem.structures import BeamType, ImageSettings, OverviewAcquisitionSettings

GRIDS = ["Grid-01", "Grid-02", "Grid-03"]  # what the sim magazine holds


class _Recorder:
    def __init__(self):
        self.emitted: List = []

    def emit(self, payload) -> None:
        self.emitted.append(payload)


class RecordingUI:
    """A recorder in place of AutoLamellaUI, which needs a live viewer."""

    def __init__(self):
        self.workflow_status_signal = _Recorder()
        self._task_manager = None  # _check_for_abort reads this

    @property
    def reports(self):
        return [
            e.report
            for e in self.workflow_status_signal.emitted
            if isinstance(e, WorkflowStatusEvent) and e.report is not None
        ]

    @property
    def workflow_info(self):
        return [
            e.workflow_info
            for e in self.workflow_status_signal.emitted
            if isinstance(e, WorkflowStatusEvent) and e.workflow_info is not None
        ]


def _small_settings(beam=BeamType.ELECTRON) -> OverviewAcquisitionSettings:
    return OverviewAcquisitionSettings(
        image_settings=ImageSettings(resolution=(128, 128), hfw=200e-6, beam_type=beam),
        nrows=1,
        ncols=1,
    )


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(
        manufacturer="Demo",
        config_path=os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml"),
    )
    assert microscope._stage.loader is not None
    assert [e.name for e in microscope._stage.grid_inventory() if e.present] == GRIDS
    return microscope


@pytest.fixture
def experiment(tmp_path, microscope):
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
    exp.sync_grids_from_inventory(microscope._stage)
    assert [g.name for g in exp.grids] == GRIDS
    return exp


def run_with_stub(manager: GridTaskManager, task_names, grid_names, on_task=None):
    """Drive the loop with the task itself stubbed out.

    ``on_task(task_name, grid)`` runs in place of the real task; it may raise, or
    stop the manager, which are the paths under test.
    """
    executed = []

    def _run_single_task(task_name, grid):
        executed.append((grid.name, task_name))
        grid.task_state.name = task_name
        try:
            if on_task is not None:
                on_task(task_name, grid)
        except Exception as e:
            grid.task_state.status = Status.Failed
            grid.task_state.status_message = str(e)
            return e
        grid.task_state.status = Status.Completed
        grid.task_state.status_message = ""
        return None

    manager.parent_ui._task_manager = manager
    manager._run_single_task = _run_single_task
    manager.run(task_names, grid_names)
    return executed


@pytest.fixture
def manager(microscope, experiment) -> GridTaskManager:
    return GridTaskManager(microscope, experiment, parent_ui=RecordingUI())


def load_entries(grid: GridRecord):
    return [t for t in grid.task_history if t.name == LOAD_ENTRY_NAME]


class TestOrderAndLoading:
    def test_the_plan_shows_each_load_ahead_of_that_grids_tasks(self):
        assert plan_grid_run(
            ["overview_sem", "overview_fib"], ["Grid-01", "Grid-02"]
        ) == [
            ("Grid-01", LOAD_ENTRY_NAME),
            ("Grid-01", "overview_sem"),
            ("Grid-01", "overview_fib"),
            ("Grid-02", LOAD_ENTRY_NAME),
            ("Grid-02", "overview_sem"),
            ("Grid-02", "overview_fib"),
        ]

    def test_the_queue_is_the_plan(self, manager):
        run_with_stub(manager, ["overview_sem"], ["Grid-01", "Grid-02"])
        assert [(i.item_name, i.task_name) for i in manager.queue.items] == [
            ("Grid-01", LOAD_ENTRY_NAME),
            ("Grid-01", "overview_sem"),
            ("Grid-02", LOAD_ENTRY_NAME),
            ("Grid-02", "overview_sem"),
        ]
        assert manager.queue.task_names == ["overview_sem"]
        assert manager.queue.item_names == ["Grid-01", "Grid-02"]

    def test_runs_grid_outer(self, manager):
        executed = run_with_stub(
            manager, ["overview_sem", "overview_fib"], ["Grid-01", "Grid-02"]
        )
        assert executed == [
            ("Grid-01", "overview_sem"),
            ("Grid-01", "overview_fib"),
            ("Grid-02", "overview_sem"),
            ("Grid-02", "overview_fib"),
        ]

    def test_each_grid_is_exchanged_once_and_the_exchange_is_recorded(
        self, manager, experiment, microscope
    ):
        run_with_stub(manager, ["overview_sem", "overview_fib"], ["Grid-01", "Grid-02"])
        for name in ("Grid-01", "Grid-02"):
            (entry,) = load_entries(experiment.get_grid_by_name(name))
            assert entry.status is Status.Completed
            assert "Slot-01" in entry.status_message
            assert entry.end_timestamp is not None
        # the last grid is left loaded; nothing unloads at the end of a run
        assert microscope._stage.loaded_grids[0].name == "Grid-02"

    def test_a_grid_already_in_the_beam_is_not_loaded_again(
        self, manager, experiment, microscope
    ):
        microscope._stage.ensure_loaded("Grid-03")
        run_with_stub(manager, ["overview_sem"], ["Grid-03"])
        assert load_entries(experiment.get_grid_by_name("Grid-03")) == []
        # the planned load step still completes: the grid is where it needs to be
        load, _ = manager.queue.items
        assert (load.task_name, load.status) == (LOAD_ENTRY_NAME, Status.Completed)

    def test_a_task_loads_its_grid_even_if_the_load_step_was_removed(
        self, manager, experiment, microscope
    ):
        def on_task(task_name, grid):
            assert microscope._stage.loaded_grids[0].name == grid.name

        # remove Grid-02's planned load before the run reaches it
        def drop_second_load(task_name, grid):
            on_task(task_name, grid)
            if grid.name == "Grid-01":
                load = next(
                    i for i in manager.queue.pending if i.item_name == "Grid-02"
                )
                assert manager.queue.remove(load.id).ok

        run_with_stub(
            manager, ["overview_sem"], ["Grid-01", "Grid-02"], drop_second_load
        )
        (entry,) = load_entries(experiment.get_grid_by_name("Grid-02"))
        assert entry.status is Status.Completed

    def test_defaults_to_every_grid_in_the_experiment(self, manager):
        executed = run_with_stub(manager, ["overview_sem"], None)
        assert [g for g, _ in executed] == GRIDS


class TestFailureIsolation:
    def test_a_failed_load_skips_that_grids_tasks_and_the_run_continues(
        self, manager, experiment, microscope
    ):
        microscope._stage.loader.fail_next_exchange = True  # Grid-01's exchange
        executed = run_with_stub(
            manager, ["overview_sem", "overview_fib"], ["Grid-01", "Grid-02"]
        )
        assert executed == [("Grid-02", "overview_sem"), ("Grid-02", "overview_fib")]

        failed = experiment.get_grid_by_name("Grid-01")
        (entry,) = load_entries(failed)
        assert entry.status is Status.Failed
        assert "Simulated" in entry.status_message
        # the grid's own tasks never ran, so it has no task entries at all
        assert [t.name for t in failed.task_history] == [LOAD_ENTRY_NAME]

        statuses = [(i.item_name, i.task_name, i.status) for i in manager.queue.items]
        assert statuses == [
            ("Grid-01", LOAD_ENTRY_NAME, Status.Failed),
            ("Grid-01", "overview_sem", Status.Skipped),
            ("Grid-01", "overview_fib", Status.Skipped),
            ("Grid-02", LOAD_ENTRY_NAME, Status.Completed),
            ("Grid-02", "overview_sem", Status.Completed),
            ("Grid-02", "overview_fib", Status.Completed),
        ]
        (load_report,) = [
            r
            for r in manager.parent_ui.reports
            if r.task_name == LOAD_ENTRY_NAME and r.status is Status.Failed
        ]
        assert load_report.item_name == "Grid-01"
        assert "Simulated" in load_report.error_message
        skipped = [r for r in manager.parent_ui.reports if r.status is Status.Skipped]
        assert {r.skip_reason for r in skipped} == {SKIP_GRID_NOT_LOADED}
        assert all("Simulated" in r.error_message for r in skipped)
        assert manager.parent_ui.workflow_info[-1] == (
            "Grid workflow complete: 1 of 2 grids run, 1 could not be loaded."
        )

    def test_a_failed_load_is_not_retried_within_the_run(self, manager, microscope):
        attempts = []
        stage = microscope._stage
        original = stage.ensure_loaded

        def counting(name):
            attempts.append(name)
            return original(name)

        stage.ensure_loaded = counting
        stage.loader.fail_next_exchange = True
        run_with_stub(manager, ["overview_sem", "overview_fib"], ["Grid-01"])
        assert attempts == ["Grid-01"]

    def test_a_failed_task_does_not_fail_the_grid(self, manager, experiment):
        def on_task(task_name, grid):
            if task_name == "overview_sem":
                raise RuntimeError("beam off")

        executed = run_with_stub(
            manager, ["overview_sem", "overview_fib"], ["Grid-01"], on_task
        )
        assert executed == [("Grid-01", "overview_sem"), ("Grid-01", "overview_fib")]
        assert [i.status for i in manager.queue.items] == [
            Status.Completed,  # load
            Status.Failed,
            Status.Completed,
        ]
        assert manager.parent_ui.workflow_info[-1] == (
            "Grid workflow complete: 1 of 1 grids run, 1 task failed."
        )

    def test_an_unknown_grid_is_skipped_with_a_reason(self, manager):
        executed = run_with_stub(manager, ["overview_sem"], ["Grid-99", "Grid-01"])
        assert executed == [("Grid-01", "overview_sem")]
        skipped = [r for r in manager.parent_ui.reports if r.status is Status.Skipped]
        assert [(r.item_name, r.task_name) for r in skipped] == [
            ("Grid-99", LOAD_ENTRY_NAME),
            ("Grid-99", "overview_sem"),
        ]
        assert {r.skip_reason for r in skipped} == {SKIP_GRID_NOT_FOUND}


class TestStopAndStatus:
    def test_stop_ends_the_run_at_the_next_task_boundary(self, manager):
        def on_task(task_name, grid):
            manager.stop()

        executed = run_with_stub(
            manager, ["overview_sem", "overview_fib"], ["Grid-01", "Grid-02"], on_task
        )
        assert executed == [("Grid-01", "overview_sem")]
        # load + first task consumed; the other four still pending
        assert manager.queue.counts == (4, 6)
        assert manager.parent_ui.workflow_info[-1] == "Grid workflow cancelled by user."
        # The rows that never ran say nothing about the load, rather than
        # repeating the last load's outcome.
        df = manager.build_run_summary_dataframe()
        assert df.loc[df.task_status == "NotStarted", "loaded"].isna().all()
        assert df.loc[df.task_status != "NotStarted", "loaded"].notna().all()

    def test_reports_name_the_grid_and_track_the_queue(self, manager):
        run_with_stub(manager, ["overview_sem"], ["Grid-01", "Grid-02"])
        reports = [
            r for r in manager.parent_ui.reports if r.task_name != LOAD_ENTRY_NAME
        ]
        assert [(r.item_name, r.status) for r in reports] == [
            ("Grid-01", Status.InProgress),
            ("Grid-01", Status.Completed),
            ("Grid-02", Status.InProgress),
            ("Grid-02", Status.Completed),
        ]
        assert [r.queue_position for r in reports] == [2, 2, 4, 4]
        assert reports[-1].queue_total == 4
        assert [i.task_name for i in reports[-1].queue_items] == [
            LOAD_ENTRY_NAME,
            "overview_sem",
            LOAD_ENTRY_NAME,
            "overview_sem",
        ]

    def test_run_summary_has_one_row_per_attempt(self, manager, microscope):
        microscope._stage.loader.fail_next_exchange = True
        run_with_stub(manager, ["overview_sem"], ["Grid-01", "Grid-02"])
        df = manager.build_run_summary_dataframe()
        assert list(df.columns) == [
            "grid_name",
            "task_name",
            "task_status",
            "loaded",
            "completed_at",
            "duration",
        ]
        assert df["grid_name"].tolist() == ["Grid-01", "Grid-01", "Grid-02", "Grid-02"]
        assert df["task_name"].tolist() == [LOAD_ENTRY_NAME, "overview_sem"] * 2
        assert df["task_status"].tolist() == [
            "Failed",
            "Skipped",
            "Completed",
            "Completed",
        ]
        assert df["loaded"].tolist() == [False, False, True, True]


class TestEndToEnd:
    def test_real_overviews_land_under_each_grid(
        self, microscope, experiment, tmp_path
    ):
        manager = run_grid_tasks(
            microscope, experiment, grid_names=["Grid-01", "Grid-02"]
        )
        assert all(i.status is Status.Completed for i in manager.queue.items)
        for name in ("Grid-01", "Grid-02"):
            grid = experiment.get_grid_by_name(name)
            assert [t.name for t in grid.task_history] == [
                LOAD_ENTRY_NAME,
                "overview_sem",
                "overview_fib",
            ]
            for role in ("overview_sem", "overview_fib"):
                (path,) = grid_outputs(experiment, grid, role)
                assert path.startswith(str(tmp_path / "exp" / "grids" / name / role))
        # the run saved as it went: the record on disk carries the history
        loaded = Experiment.load(tmp_path / "exp" / "experiment.yaml")
        assert len(loaded.get_grid_by_name("Grid-02").task_history) == 3
