"""The fluorescence overview grid task, on the Arctis simulator (a compustage with an FM).

It travels to the FM, inserts the objective if it has to, acquires the tileset
centred on the grid's slot re-expressed for the FM, records the mosaic and a
channel-composite thumbnail, and puts the objective back how it found it.
"""

import os

import pytest
import yaml

import fibsem.config as cfg
from fibsem import utils
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.applications.autolamella.workflows.tasks.grid import (
    FluorescenceOverviewGridTask,
    FluorescenceOverviewGridTaskConfig,
    get_grid_tasks,
    run_grid_task,
)
from fibsem.fm.acquisition import OverviewDestination, acquire_fm_tiled_overview
from fibsem.fm.structures import ChannelSettings, OverviewParameters, ZParameters
from fibsem.services.sample_handling import BUILTIN_SLOT_NAME
from fibsem.structures import InsertableDeviceState, SampleGrid
from fibsem.workflows.grids import GridRecord
from fibsem.workflows.outputs import (
    item_outputs,
    latest_item_output,
)
from fibsem.workflows.status import TaskStatus


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(
        manufacturer="Demo",
        config_path=os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml"),
    )
    assert microscope._fm_is_a_pose() and microscope.fm is not None
    # the working slot is the origin by construction; put a grid in it
    microscope.sample_handling.holder.slots[BUILTIN_SLOT_NAME].loaded_grid = SampleGrid(
        name="grid-aspen"
    )
    return microscope


@pytest.fixture
def experiment(tmp_path):
    exp = Experiment(path=tmp_path, name="exp")
    (tmp_path / "exp").mkdir()
    exp.task_protocol = AutoLamellaTaskProtocol()
    exp.grid_protocol.add(
        FluorescenceOverviewGridTaskConfig(
            task_name="overview_fm",
            channels=[
                ChannelSettings(name="GFP", color="green"),
                ChannelSettings(name="mCherry", color="red"),
            ],
            overview=OverviewParameters(rows=1, cols=2),
        )
    )
    exp.add_grid(GridRecord(name="grid-aspen"))
    return exp


class TestConfig:
    def test_registered(self):
        assert get_grid_tasks()["FM_OVERVIEW_GRID"] is FluorescenceOverviewGridTask

    def test_round_trips_through_protocol_yaml_with_a_channel_list(
        self, experiment, tmp_path
    ):
        config = experiment.grid_protocol.task_config["overview_fm"]
        config.overview.use_zstack = True
        config.zparams = ZParameters(zmin=-2e-6, zmax=2e-6, zstep=1e-6)
        experiment.save(save_protocol=True)
        data = yaml.safe_load((tmp_path / "exp" / "protocol.yaml").read_text())
        saved = data["grid_tasks"]["tasks"]["overview_fm"]
        assert [c["name"] for c in saved["channels"]] == ["GFP", "mCherry"]
        assert saved["overview"]["rows"] == 1

        loaded = Experiment.load(tmp_path / "exp" / "experiment.yaml")
        again = loaded.grid_protocol.task_config["overview_fm"]
        assert isinstance(again, FluorescenceOverviewGridTaskConfig)
        assert [type(c).__name__ for c in again.channels] == ["ChannelSettings"] * 2
        assert again.channels[1].color == "red"
        assert again.overview.use_zstack is True
        assert again.zparams.zstep == 1e-6


class TestRun:
    def test_records_mosaic_and_composite_thumbnail(
        self, microscope, experiment, tmp_path
    ):
        grid = experiment.get_grid_by_name("grid-aspen")
        run_grid_task(microscope, "overview_fm", experiment, grid)

        entry = grid.task_history[-1]
        assert entry.status is TaskStatus.Completed
        assert set(entry.outputs) == {"overview_fm", "overview_fm_thumbnail"}
        (mosaic,) = item_outputs(experiment, grid, "overview_fm")
        assert mosaic.startswith(
            str(tmp_path / "exp" / "grids" / "grid-aspen" / "overview_fm")
        )
        assert mosaic.endswith(".ome.tiff")
        thumbnail = latest_item_output(experiment, grid, "overview_fm_thumbnail")
        from PIL import Image

        with Image.open(thumbnail) as im:
            assert im.mode == "RGB" and max(im.size) <= 512

    def test_objective_is_inserted_for_the_run_and_returned(
        self, microscope, experiment
    ):
        grid = experiment.get_grid_by_name("grid-aspen")
        assert microscope.fm.objective.state is not InsertableDeviceState.INSERTED
        run_grid_task(microscope, "overview_fm", experiment, grid)
        assert (
            microscope.fm.objective.state is not InsertableDeviceState.INSERTED
        )  # put back how it was found

    def test_an_already_inserted_objective_is_left_in(self, microscope, experiment):
        microscope.fm.objective.insert()
        grid = experiment.get_grid_by_name("grid-aspen")
        run_grid_task(microscope, "overview_fm", experiment, grid)
        assert microscope.fm.objective.state is InsertableDeviceState.INSERTED

    def test_centre_is_the_slot_at_the_fm(self, microscope, experiment):
        grid = experiment.get_grid_by_name("grid-aspen")
        task = FluorescenceOverviewGridTask(
            microscope,
            experiment.grid_protocol.task_config["overview_fm"],
            grid,
            experiment,
        )
        centre = task.grid_centre()
        fm = microscope.get_orientation("FM")
        assert centre.t == pytest.approx(fm.t)

    def test_refuses_without_an_fm(self, experiment, tmp_path):
        plain, _ = utils.setup_session(manufacturer="Demo")
        assert plain.fm is None
        grid = experiment.get_grid_by_name("grid-aspen")
        with pytest.raises(RuntimeError, match="no fluorescence microscope"):
            run_grid_task(plain, "overview_fm", experiment, grid)
        assert grid.task_state.status is TaskStatus.Failed


class TestProgressEndsTheRun:
    """The FM runner reports up to the stitch; the operation says how it ended, so
    the window's status bar does not stay on "Stitching tiles" for ever."""

    def test_a_finished_run_reports_saving_then_finished(self, microscope, experiment):
        from fibsem.imaging.tiling.progress import MODALITY_FLUORESCENCE, TiledStatus

        reports = []
        microscope.imaging.tiled_progress.connect(reports.append)
        grid = experiment.get_grid_by_name("grid-aspen")
        run_grid_task(microscope, "overview_fm", experiment, grid)
        statuses = [r.status for r in reports if r.modality == MODALITY_FLUORESCENCE]
        assert statuses[-2:] == [TiledStatus.SAVING, TiledStatus.FINISHED]
        assert TiledStatus.STITCHING in statuses

    def test_a_stopped_run_reports_cancelled(self, microscope, experiment):
        import threading

        from fibsem.imaging.tiling.progress import TiledStatus

        reports = []
        microscope.imaging.tiled_progress.connect(reports.append)
        stop = threading.Event()
        stop.set()
        grid = experiment.get_grid_by_name("grid-aspen")
        config = experiment.grid_protocol.task_config["overview_fm"]
        # Where the task would have put the stage: the runner refuses to image
        # anywhere but the FM pose, which would report Failed, not Cancelled.
        microscope.move_to_device("FM")
        directory = experiment.grid_path(grid) / "overview_fm"
        directory.mkdir(parents=True, exist_ok=True)
        with pytest.raises(Exception):
            acquire_fm_tiled_overview(
                microscope,
                config.channels,
                config.overview,
                OverviewDestination.create(str(directory)),
                stop_event=stop,
            )
        assert reports[-1].status is TiledStatus.CANCELLED
