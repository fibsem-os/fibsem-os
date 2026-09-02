"""Grids · Results: what a grid's runs recorded, read off its history."""

import os

import numpy as np
import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    AutoLamellaTaskState,
    AutoLamellaTaskStatus,
    Experiment,
    GridRecord,
)
from fibsem.applications.autolamella.ui.grid_results_widget import (
    GridResultsWidget,
    latest_runs,
    thumbnail_for,
)
from fibsem.applications.autolamella.ui.grids_tab_widget import GridsTabWidget
from fibsem.applications.autolamella.workflows.tasks.grid import (
    BeamOverviewGridTaskConfig,
    FluorescenceOverviewGridTaskConfig,
)
from fibsem.imaging.thumbnail import write_thumbnail


def entry(name, status, message="", outputs=None, seconds=60):
    state = AutoLamellaTaskState(name=name, status=status, status_message=message)
    state.end_timestamp = state.start_timestamp + seconds
    state.outputs = outputs or {}
    return state


@pytest.fixture
def experiment(tmp_path):
    exp = Experiment(path=tmp_path, name="exp")
    (tmp_path / "exp").mkdir()
    exp.task_protocol = AutoLamellaTaskProtocol()
    exp.grid_protocol.add(BeamOverviewGridTaskConfig(task_name="overview_sem"))
    exp.grid_protocol.add(
        BeamOverviewGridTaskConfig(task_name="overview_fib", orientation="FIB")
    )
    exp.grid_protocol.add(FluorescenceOverviewGridTaskConfig(task_name="overview_fm"))
    return exp


@pytest.fixture
def grid(experiment):
    """A grid with a load, a recorded SEM overview, a failed FM overview, and the FIB
    overview never run."""
    grid = experiment.add_grid(GridRecord(name="grid-elm", description="HeLa, batch B"))
    root = experiment.grid_path(grid)
    thumb = write_thumbnail(
        (np.random.rand(300, 450) * 255).astype(np.uint8),
        root / "overview_sem" / "overview-thumbnail.png",
    )
    assert os.path.isfile(thumb)
    (root / "overview_sem" / "overview.tif").write_bytes(b"")  # the full overview
    grid.task_history += [
        entry(
            "load", AutoLamellaTaskStatus.Completed, "Loaded into Slot-01.", seconds=218
        ),
        entry(
            "overview_sem",
            AutoLamellaTaskStatus.Completed,
            outputs={
                "overview_sem": ["overview_sem/overview.tif"],
                "overview_sem_thumbnail": ["overview_sem/overview-thumbnail.png"],
            },
        ),
        entry(
            "overview_fm", AutoLamellaTaskStatus.Failed, "autofocus timeout on tile 7"
        ),
    ]
    return grid


def test_latest_run_per_task_leaves_the_load_out(grid):
    assert list(latest_runs(grid)) == ["overview_sem", "overview_fm"]


def test_thumbnail_is_read_off_the_recorded_outputs(experiment, grid):
    sem = latest_runs(grid)["overview_sem"]
    assert thumbnail_for(experiment, grid, sem).endswith("overview-thumbnail.png")
    assert thumbnail_for(experiment, grid, latest_runs(grid)["overview_fm"]) is None


def test_nothing_selected(qapp, experiment):
    widget = GridResultsWidget()
    widget.set_experiment(experiment)
    assert not widget.empty_label.isHidden()
    assert widget.name_label.isHidden()


def test_rows_follow_the_history_with_images_where_recorded(qapp, experiment, grid):
    widget = GridResultsWidget()
    widget.set_experiment(experiment)
    widget.set_grid(grid)
    assert widget.name_label.text() == "grid-elm"
    assert widget.name_label.toolTip().endswith("grid-elm")
    assert widget.subtitle_label.text().startswith(
        "HeLa, batch B · overview_sem, completed at"
    )

    load, sem, fm = widget.rows
    assert load.state.name == "load" and load.tile is None
    assert load.detail_label.text() == "Loaded into Slot-01."
    assert sem.tile is not None and sem.tile.pixmap() is not None
    # the tile opens the full overview, not the thumbnail it shows
    assert sem.image.endswith("overview_sem/overview.tif")
    assert sem.tile._filepath == sem.image
    assert fm.tile is None
    assert fm.status_label.text() == "Failed"
    assert fm.detail_label.text() == "autofocus timeout on tile 7"


def test_a_grid_never_run(qapp, experiment):
    grid = experiment.add_grid(GridRecord(name="grid-oak"))
    widget = GridResultsWidget()
    widget.set_experiment(experiment)
    widget.set_grid(grid)
    assert widget.subtitle_label.text() == "No completed tasks"
    assert len(widget.rows) == 1 and "Nothing has run" in widget.rows[0].text()


def test_the_grids_tab_follows_card_selection(qapp, experiment, grid):
    tab = GridsTabWidget(synchronous=True)
    tab.set_experiment(experiment)
    tab.cards._on_card_clicked(grid)
    assert tab.results_widget.grid is grid
    tab.cards._on_card_clicked(grid)
    assert tab.results_widget.grid is None
