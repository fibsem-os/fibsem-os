"""A grid overview larger than the stage can reach, on the Arctis simulator.

The compustage travels +/-999.9 um in x and only +/-377.8 um in y about the
working slot. The tile runners refuse a grid with any tile past that, but only at
acquire time, which for a grid task is after the exchange and again on every grid
(FIB-1131). The task now turns those tiles off first, acquires the rest, and says
in the run's history which it skipped; it fails only when no tile is in reach.
"""

import os
from pathlib import Path

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
    FluorescenceOverviewGridTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.grid.imaging import (
    disable_tiles,
    reachable_overview,
)
from fibsem.applications.autolamella.workflows.tasks.grid.manager import (
    run_grid_tasks,
)
from fibsem.fm.structures import ChannelSettings, OverviewParameters
from fibsem.structures import BeamType, ImageSettings, OverviewAcquisitionSettings

ROWS_OUT_OF_REACH = "(0,0), (0,1), (0,2), (0,3), (3,0), (3,1), (3,2), (3,3)"


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(
        manufacturer="Demo",
        config_path=os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml"),
    )
    yield microscope
    microscope.disconnect()


@pytest.fixture
def experiment(tmp_path, microscope):
    exp = Experiment(path=tmp_path, name="exp")
    (tmp_path / "exp").mkdir()
    exp.task_protocol = AutoLamellaTaskProtocol()
    exp.sync_grids_from_inventory(microscope._stage)
    return exp


def _beam(rows, cols, hfw, beam=BeamType.ELECTRON) -> OverviewAcquisitionSettings:
    # square tiles, so a row is as tall as a column is wide
    return OverviewAcquisitionSettings(
        image_settings=ImageSettings(resolution=(64, 64), hfw=hfw, beam_type=beam),
        nrows=rows,
        ncols=cols,
    )


def _run(microscope, experiment, config):
    experiment.grid_protocol.add(config)
    run_grid_tasks(microscope, experiment, [config.task_name], ["Grid-01"])
    grid = experiment.get_grid_by_name("Grid-01")
    (entry,) = [t for t in grid.task_history if t.name == config.task_name]
    return grid, entry


@pytest.mark.parametrize(
    "orientation, beam", [("SEM", BeamType.ELECTRON), ("FIB", BeamType.ION)]
)
def test_tiles_out_of_reach_are_skipped_and_named(
    microscope, experiment, orientation, beam
):
    """4 x 4 tiles of 300 um: rows 0 and 3 sit at +/-405 um in y, past the
    travel. They used to fail the whole overview; now the middle rows are taken."""
    config = BeamOverviewGridTaskConfig(
        task_name="overview",
        orientation=orientation,
        settings=_beam(4, 4, 300e-6, beam),
    )
    grid, entry = _run(microscope, experiment, config)
    assert entry.status is Status.Completed
    assert entry.status_message == (
        "Finished; 8 of 16 tiles were past the stage's reach and skipped: "
        + ROWS_OUT_OF_REACH
    )
    (image,) = entry.outputs[config.role]
    assert (experiment.grid_path(grid) / image).is_file()
    # the protocol itself is left as it was
    assert experiment.grid_protocol.task_config["overview"].settings.tile_mask is None


def test_the_report_says_which_tiles_were_skipped(microscope, experiment, tmp_path):
    pytest.importorskip("reportlab")
    from fibsem.applications.autolamella.tools.grid_report_pdf import (
        generate_grid_report,
    )

    config = BeamOverviewGridTaskConfig(
        task_name="overview", settings=_beam(4, 4, 300e-6)
    )
    _run(microscope, experiment, config)
    path = generate_grid_report(
        experiment, output_path=str(tmp_path / "report.pdf"), compress=False
    )
    text = Path(path).read_bytes()
    # the line wraps in the overview's facts column, so in pieces
    assert b"Finished; 8 of 16 tiles were" in text
    assert b"past the stage's reach and skipped:" in text


def test_an_overview_in_reach_is_untouched(microscope, experiment):
    config = BeamOverviewGridTaskConfig(
        task_name="overview", settings=_beam(3, 3, 300e-6)
    )
    _, entry = _run(microscope, experiment, config)
    assert entry.status is Status.Completed
    assert entry.status_message == "Finished"


def test_an_overview_with_no_tile_in_reach_fails_and_says_so(microscope, experiment):
    """2 x 2 tiles of 1 mm: every centre is 450 um off in y."""
    config = BeamOverviewGridTaskConfig(
        task_name="overview", settings=_beam(2, 2, 1e-3)
    )
    _, entry = _run(microscope, experiment, config)
    assert entry.status is Status.Failed
    assert "None of the 4 tiles" in entry.status_message


def test_a_mask_already_set_is_kept_and_not_counted(microscope):
    """A tile the operator turned off is not "skipped": only the reachable
    question's answers are reported, on top of the mask already there."""
    settings = _beam(4, 4, 300e-6)
    settings.tile_mask = disable_tiles(None, 4, 4, [(1, 1)])
    centre = microscope._stage.holder.slots["Slot-01"].position
    masked, skipped = reachable_overview(microscope, settings, centre)
    assert len(skipped) == 8 and (1, 1) not in skipped
    assert sum(not on for row in masked.tile_mask for on in row) == 9
    assert settings.tile_mask[0][0] is True  # the caller's settings are not rewritten


def test_a_fluorescence_overview_skips_its_tiles_out_of_reach(microscope, experiment):
    """A column of 10 tiles of the simulator's 102 um field: the top and bottom
    tiles are past the y travel. One column keeps the run short."""
    config = FluorescenceOverviewGridTaskConfig(
        task_name="fm",
        channels=[ChannelSettings(name="GFP", color="green")],
        overview=OverviewParameters(rows=10, cols=1),
    )
    _, entry = _run(microscope, experiment, config)
    assert entry.status is Status.Completed, entry.status_message
    assert entry.status_message == (
        "Finished; 2 of 10 tiles were past the stage's reach and skipped: (0,0), (9,0)"
    )
