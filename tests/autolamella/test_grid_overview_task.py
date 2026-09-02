"""The beam overview grid task, end to end on the simulator.

One task for both beams, an orientation to acquire in, the standard overview
settings. It starts from the grid's calibrated slot position re-expressed for the
orientation, refuses a grid that is not reachable, and records the stitched image
and a thumbnail by role under the grid's own directory.
"""

import threading

import numpy as np
import pytest
import yaml

from fibsem import utils
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    AutoLamellaTaskStatus,
    Experiment,
    GridRecord,
)
from fibsem.applications.autolamella.task_outputs import (
    grid_outputs,
    latest_grid_output,
)
from fibsem.applications.autolamella.workflows.tasks.grid import (
    GRID_TASK_REGISTRY,
    BeamOverviewGridTask,
    BeamOverviewGridTaskConfig,
    run_grid_task,
)
from fibsem.applications.autolamella.workflows.tasks.grid.imaging import (
    acquire_beam_overview,
    write_thumbnail,
)
from fibsem.microscopes._stage import SampleGrid, SlotCalibration, _create_sample_stage
from fibsem.structures import (
    BeamType,
    FibsemStagePosition,
    ImageSettings,
    OverviewAcquisitionSettings,
)


def _small_settings(
    beam=BeamType.ELECTRON, rows=2, cols=2
) -> OverviewAcquisitionSettings:
    return OverviewAcquisitionSettings(
        image_settings=ImageSettings(resolution=(128, 128), hfw=200e-6, beam_type=beam),
        nrows=rows,
        ncols=cols,
    )


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo")
    microscope.stage_is_compustage = False
    microscope._stage = _create_sample_stage(microscope)
    slot = microscope._stage.holder.slots["Slot-01"]
    slot.position = FibsemStagePosition(
        name="Slot-01", x=-4e-3, y=1e-3, z=4e-3, r=0.0, t=0.61
    )
    slot.calibration = SlotCalibration("SEM", 35.0, 0.0, "2026-09-02T11:24:09", "test")
    slot.loaded_grid = SampleGrid(name="grid-aspen")
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
    exp.add_grid(GridRecord(name="grid-aspen"))
    return exp


class TestConfig:
    def test_registered(self):
        assert GRID_TASK_REGISTRY["BEAM_OVERVIEW_GRID"] is BeamOverviewGridTask

    def test_role_follows_the_beam_in_the_settings(self):
        assert BeamOverviewGridTaskConfig().role == "overview_sem"
        assert (
            BeamOverviewGridTaskConfig(settings=_small_settings(BeamType.ION)).role
            == "overview_fib"
        )

    def test_round_trips_through_protocol_yaml(self, experiment, tmp_path):
        experiment.save(save_protocol=True)
        loaded = Experiment.load(tmp_path / "exp" / "experiment.yaml")
        fib = loaded.grid_protocol.task_config["overview_fib"]
        assert isinstance(fib, BeamOverviewGridTaskConfig)
        assert fib.orientation == "FIB"
        assert fib.beam_type is BeamType.ION
        assert fib.settings.nrows == 2 and fib.settings.image_settings.hfw == 200e-6
        data = yaml.safe_load((tmp_path / "exp" / "protocol.yaml").read_text())
        assert data["grid_tasks"]["order"] == ["overview_sem", "overview_fib"]


class TestReachability:
    def test_refuses_a_grid_that_is_not_in_a_slot(self, microscope, experiment):
        grid = experiment.add_grid(GridRecord(name="grid-nowhere"))
        with pytest.raises(RuntimeError, match="not in a holder slot"):
            run_grid_task(microscope, "overview_sem", experiment, grid)
        assert grid.task_state.status is AutoLamellaTaskStatus.Failed

    def test_refuses_an_uncalibrated_slot(self, microscope, experiment):
        slot = microscope._stage.holder.slots["Slot-01"]
        slot.position = None
        slot.calibration = None
        grid = experiment.get_grid_by_name("grid-aspen")
        with pytest.raises(RuntimeError, match="Calibrate slot positions"):
            run_grid_task(microscope, "overview_sem", experiment, grid)

    def test_centre_is_the_slot_re_expressed_for_the_orientation(
        self, microscope, experiment
    ):
        grid = experiment.get_grid_by_name("grid-aspen")
        task = BeamOverviewGridTask(
            microscope,
            experiment.grid_protocol.task_config["overview_fib"],
            grid,
            experiment,
        )
        centre = task.grid_centre()
        fib = microscope.get_orientation("FIB")
        assert centre.r == pytest.approx(fib.r) and centre.t == pytest.approx(fib.t)
        # a half turn about the stage axis mirrors x: the slot at -4 mm is reached at
        # +4 mm once the stage has rotated to the FIB orientation
        assert abs(centre.x) == pytest.approx(4e-3, abs=1e-6)
        sem = task.microscope.get_target_position(task.slot.position, "SEM")
        assert sem.x == pytest.approx(-4e-3, abs=1e-6)


class TestRun:
    def test_records_the_stitched_image_and_a_thumbnail(
        self, microscope, experiment, tmp_path
    ):
        grid = experiment.get_grid_by_name("grid-aspen")
        run_grid_task(microscope, "overview_sem", experiment, grid)

        entry = grid.task_history[-1]
        assert entry.status is AutoLamellaTaskStatus.Completed
        assert set(entry.outputs) == {"overview_sem", "overview_sem_thumbnail"}
        (overview,) = grid_outputs(experiment, grid, "overview_sem")
        assert overview.startswith(
            str(tmp_path / "exp" / "grids" / "grid-aspen" / "overview_sem")
        )
        assert overview.endswith(".tif")
        thumbnail = latest_grid_output(experiment, grid, "overview_sem_thumbnail")
        assert thumbnail.endswith("-thumbnail.png")
        from PIL import Image

        with Image.open(thumbnail) as im:
            assert max(im.size) <= 512

    def test_fib_overview_lands_under_its_own_role_and_task(
        self, microscope, experiment
    ):
        grid = experiment.get_grid_by_name("grid-aspen")
        run_grid_task(microscope, "overview_fib", experiment, grid)
        (overview,) = grid_outputs(experiment, grid, "overview_fib")
        assert "/overview_fib/" in overview
        assert grid_outputs(experiment, grid, "overview_sem") == []

    def test_stage_is_where_it_started_afterwards(self, microscope, experiment):
        before = microscope.get_stage_position()
        grid = experiment.get_grid_by_name("grid-aspen")
        run_grid_task(microscope, "overview_sem", experiment, grid)
        after = microscope.get_stage_position()
        # the runner restores its start, which is the grid centre the task moved to
        centre = microscope.get_target_position(
            microscope._stage.holder.slots["Slot-01"].position, "SEM"
        )
        assert after.x == pytest.approx(centre.x, abs=1e-6)
        assert after.x != pytest.approx(before.x, abs=1e-6)

    def test_two_runs_do_not_overwrite_each_other(self, microscope, experiment):
        grid = experiment.get_grid_by_name("grid-aspen")
        run_grid_task(microscope, "overview_sem", experiment, grid)
        import time

        time.sleep(1.1)  # the name carries the time of day, to the second
        run_grid_task(microscope, "overview_sem", experiment, grid)
        assert len(grid_outputs(experiment, grid, "overview_sem")) == 2

    def test_a_stop_before_the_run_is_a_cancellation(self, microscope, experiment):
        class Manager:
            abort_token = threading.Event()
            hook_manager = None

            @property
            def should_abort(self):
                return self.abort_token.is_set()

            def hook_run_context(self):
                return {}

        manager = Manager()
        manager.abort_token.set()
        grid = experiment.get_grid_by_name("grid-aspen")
        with pytest.raises((InterruptedError, Exception)):
            run_grid_task(
                microscope, "overview_sem", experiment, grid, task_manager=manager
            )
        assert grid.task_state.status is AutoLamellaTaskStatus.Cancelled

    def test_the_operation_is_callable_standalone(self, microscope, tmp_path):
        image = acquire_beam_overview(
            microscope,
            _small_settings(),
            microscope.get_stage_position(),
            tmp_path / "standalone",
            stem="ov",
        )
        assert image.filepath and "/standalone/ov-" in image.filepath
        assert image.data.ndim == 2


def test_write_thumbnail_bounds_the_size_and_is_atomic(tmp_path):
    data = (np.random.rand(1200, 900) * 255).astype(np.uint8)
    path = write_thumbnail(data, tmp_path / "t" / "thumb.png")
    from PIL import Image

    with Image.open(path) as im:
        assert max(im.size) == 512
    assert [p.name for p in (tmp_path / "t").iterdir()] == ["thumb.png"]
