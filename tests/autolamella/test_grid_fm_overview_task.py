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
    FluorescenceOverviewGridTask,
    FluorescenceOverviewGridTaskConfig,
    run_grid_task,
)
from fibsem.fm.structures import ChannelSettings, OverviewParameters, ZParameters
from fibsem.microscopes._stage import SampleGrid


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(
        manufacturer="Demo",
        config_path=os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml"),
    )
    assert microscope.stage_is_compustage and microscope.fm is not None
    # the working slot is the origin by construction; put a grid in it
    microscope._stage.holder.slots["Slot-01"].loaded_grid = SampleGrid(
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
        assert GRID_TASK_REGISTRY["FM_OVERVIEW_GRID"] is FluorescenceOverviewGridTask

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
        assert entry.status is AutoLamellaTaskStatus.Completed
        assert set(entry.outputs) == {"overview_fm", "overview_fm_thumbnail"}
        (mosaic,) = grid_outputs(experiment, grid, "overview_fm")
        assert mosaic.startswith(
            str(tmp_path / "exp" / "grids" / "grid-aspen" / "overview_fm")
        )
        assert mosaic.endswith(".ome.tiff")
        thumbnail = latest_grid_output(experiment, grid, "overview_fm_thumbnail")
        from PIL import Image

        with Image.open(thumbnail) as im:
            assert im.mode == "RGB" and max(im.size) <= 512

    def test_objective_is_inserted_for_the_run_and_returned(
        self, microscope, experiment
    ):
        grid = experiment.get_grid_by_name("grid-aspen")
        assert microscope.fm.objective.state != "Inserted"
        run_grid_task(microscope, "overview_fm", experiment, grid)
        assert microscope.fm.objective.state != "Inserted"  # put back how it was found

    def test_an_already_inserted_objective_is_left_in(self, microscope, experiment):
        microscope.fm.objective.insert()
        grid = experiment.get_grid_by_name("grid-aspen")
        run_grid_task(microscope, "overview_fm", experiment, grid)
        assert microscope.fm.objective.state == "Inserted"

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
        assert grid.task_state.status is AutoLamellaTaskStatus.Failed
