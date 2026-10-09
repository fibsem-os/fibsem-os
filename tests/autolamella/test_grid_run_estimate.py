"""How long a grid run will take (FIB-1134).

The per-task figures on the grid task configs, the exchange the loader declares, and
the run-level assembly the confirmation, the timeline and Add to queue quote. As in
test_timing.py, the shape is pinned rather than the constants, except where a test
says it holds the estimate to a measured run.
"""

import os
from copy import deepcopy
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

import fibsem.config as cfg
from fibsem import timing, utils
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.applications.autolamella.workflows.tasks.grid.fluorescence import (
    FluorescenceOverviewGridTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.grid.imaging import (
    BeamOverviewGridTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.grid.manager import (
    LOAD_ENTRY_NAME,
    plan_grid_run,
)
from fibsem.applications.autolamella.workflows.workflow_estimate import (
    estimate_grid_run,
    grid_item_seconds,
)
from fibsem.autofunctions.autofocus import AutoFocusSettings, FocusSweepPass
from fibsem.drivers.autoscript.microscope import AutoscriptSampleLoader
from fibsem.fm.structures import ChannelSettings, OverviewParameters
from fibsem.structures import (
    AutoContrastMode,
    AutoFocusMode,
    BeamType,
    ImageSettings,
    OverviewAcquisitionSettings,
)
from tests.fixtures.demo_stage import demo_grid_loader

NOW = datetime(2026, 10, 5, 14, 0, 0)


def _overview(nrows=3, ncols=3, **kwargs) -> OverviewAcquisitionSettings:
    return OverviewAcquisitionSettings(
        image_settings=ImageSettings(
            resolution=(1536, 1024), dwell_time=1e-6, hfw=500e-6
        ),
        nrows=nrows,
        ncols=ncols,
        **kwargs,
    )


# ── a beam overview ──────────────────────────────────────────────────────────


def test_a_beam_overview_is_a_move_and_a_frame_per_tile_and_the_way_back():
    settings = _overview()
    per_tile = timing.OVERVIEW_TILE_MOVE_S + timing.image_cost(settings.image_settings)
    assert timing.beam_overview_cost(settings) == pytest.approx(
        9 * per_tile + timing.stage_move_cost(1)
    )


def test_a_beam_overview_counts_the_enabled_tiles_only():
    mask = [[True, False, False], [False, True, False], [False, False, False]]
    masked = timing.beam_overview_cost(_overview(tile_mask=mask))
    assert masked == pytest.approx(timing.beam_overview_cost(_overview(1, 2)))
    assert timing.beam_overview_cost(_overview(tile_mask=[[False] * 3] * 3)) == 0


def test_a_beam_overview_pays_for_the_focus_and_contrast_it_asks_for():
    sweep = AutoFocusSettings(
        passes=[FocusSweepPass(search_range=10e-6, step_size=1e-6)]
    )
    one_sweep = timing.beam_autofocus_cost(sweep)
    assert one_sweep > 0
    base = timing.beam_overview_cost(_overview(autofocus_settings=sweep))
    for mode, sweeps in [
        (AutoFocusMode.ONCE, 1),
        (AutoFocusMode.EACH_ROW, 3),
        (AutoFocusMode.EACH_TILE, 9),
    ]:
        cost = timing.beam_overview_cost(
            _overview(autofocus_mode=mode, autofocus_settings=sweep)
        )
        assert cost == pytest.approx(base + sweeps * one_sweep), mode

    once = timing.beam_overview_cost(_overview(autocontrast_mode=AutoContrastMode.ONCE))
    each = timing.beam_overview_cost(
        _overview(autocontrast_mode=AutoContrastMode.EACH_TILE)
    )
    assert base < once < each


def test_the_beam_overview_task_adds_the_move_to_its_grid():
    config = BeamOverviewGridTaskConfig(task_name="SEM", settings=_overview())
    assert config.estimated_duration == pytest.approx(
        timing.stage_move_cost(1) + timing.beam_overview_cost(config.settings)
    )


# ── a fluorescence overview ──────────────────────────────────────────────────


def _fm(**overview) -> FluorescenceOverviewGridTaskConfig:
    return FluorescenceOverviewGridTaskConfig(
        task_name="FM",
        channels=[ChannelSettings(name="A", exposure_time=0.1)],
        overview=OverviewParameters(rows=3, cols=3, **overview),
    )


def test_a_fluorescence_overview_puts_the_objective_in_and_takes_it_out():
    config = _fm()
    fixed = (
        2 * timing.stage_move_cost(1)
        + timing.OBJECTIVE_INSERT_S
        + timing.OBJECTIVE_FOCUS_MOVE_S
        + timing.OBJECTIVE_RETRACT_S
    )
    tiles = config.estimated_duration - fixed
    # each tile: the measured move, the exposure and the camera's overhead
    assert tiles == pytest.approx(
        9 * (timing.OVERVIEW_TILE_MOVE_S + 0.1 + 2.3), rel=1e-3
    )


def test_a_fluorescence_overview_grows_with_tiles_channels_and_focus():
    base = _fm().estimated_duration
    assert _fm(tile_mask=[[True, False, False]] * 3).estimated_duration < base

    two = _fm()
    two.channels.append(ChannelSettings(name="B", exposure_time=0.1))
    assert two.estimated_duration > base

    focused = _fm(autofocus_mode=AutoFocusMode.EACH_TILE)
    focused.autofocus_settings = AutoFocusSettings()
    once = _fm(autofocus_mode=AutoFocusMode.ONCE)
    once.autofocus_settings = AutoFocusSettings()
    assert base < once.estimated_duration < focused.estimated_duration


# ── the exchange ─────────────────────────────────────────────────────────────


def test_each_loader_declares_what_an_exchange_costs():
    assert (
        demo_grid_loader(SimpleNamespace(), exchange_delay=5.0).exchange_seconds == 10
    )
    assert AutoscriptSampleLoader(SimpleNamespace()).exchange_seconds == 180


@pytest.fixture
def arctis():
    microscope, _ = utils.setup_session(
        manufacturer="Demo",
        config_path=os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml"),
    )
    return microscope


@pytest.fixture
def experiment(tmp_path, arctis):
    exp = Experiment(path=tmp_path, name="exp")
    (tmp_path / "exp").mkdir()
    exp.task_protocol = AutoLamellaTaskProtocol()
    exp.grid_protocol.add(
        BeamOverviewGridTaskConfig(task_name="SEM", settings=_overview())
    )
    fib = _overview(3, 6)
    fib.image_settings.beam_type = BeamType.ION
    exp.grid_protocol.add(
        BeamOverviewGridTaskConfig(task_name="FIB", orientation="FIB", settings=fib)
    )
    exp.grid_protocol.add(_fm())
    exp.sync_grids_from_inventory(arctis._stage)
    return exp


# ── the run ──────────────────────────────────────────────────────────────────


def test_a_three_grid_run_prices_an_exchange_per_grid_and_each_task_per_grid(
    experiment, arctis
):
    """The Arctis simulator, three grids, none loaded: three exchanges, as the plan
    has a load for each, and SEM, FIB and FM on every grid."""
    grids = [g.name for g in experiment.grids]
    assert len(grids) == 3
    loader = arctis._stage.loader
    plan = plan_grid_run(["SEM", "FIB", "FM"], grids)
    exchanges = sum(1 for _, step in plan if step == LOAD_ENTRY_NAME)

    est = estimate_grid_run(
        experiment, ["SEM", "FIB", "FM"], grids, exchanges, loader.exchange_seconds, NOW
    )

    assert [(t.name, t.lamella_count) for t in est.tasks] == [
        (LOAD_ENTRY_NAME, 3),
        ("SEM", 3),
        ("FIB", 3),
        ("FM", 3),
    ]
    assert est.tasks[0].seconds == pytest.approx(3 * loader.exchange_seconds)
    config = experiment.grid_protocol.task_config["FIB"]
    assert est.tasks[2].seconds == pytest.approx(3 * config.estimated_duration)
    assert est.work_seconds == pytest.approx(sum(t.seconds for t in est.tasks))
    assert est.expected_finish == NOW + timedelta(seconds=est.work_seconds)
    assert est.step_count == len(plan)


def test_a_fixed_holder_prices_no_exchange(experiment):
    grids = [g.name for g in experiment.grids]
    est = estimate_grid_run(experiment, ["SEM"], grids, 0, 0.0, NOW)
    assert [t.name for t in est.tasks] == ["SEM"]
    # and a loader with every grid loaded quotes no row for it either
    est = estimate_grid_run(experiment, ["SEM"], grids, 0, 180.0, NOW)
    assert [t.name for t in est.tasks] == ["SEM"]


def test_a_task_the_protocol_does_not_have_is_left_out(experiment):
    grids = [g.name for g in experiment.grids]
    est = estimate_grid_run(experiment, ["SEM", "gone"], grids, 0, 0.0, NOW)
    assert [t.name for t in est.tasks] == ["SEM"]


def test_a_grid_step_is_priced_by_its_task_and_a_load_by_the_loader(experiment):
    grids = [g.name for g in experiment.grids]
    seconds_for = grid_item_seconds(experiment, 180.0, loaded=[grids[0]])
    sem = experiment.grid_protocol.task_config["SEM"].estimated_duration
    assert seconds_for(grids[1], "SEM") == pytest.approx(sem)
    assert seconds_for(grids[1], LOAD_ENTRY_NAME) == 180.0
    # the grid on the stage loads without an exchange
    assert seconds_for(grids[0], LOAD_ENTRY_NAME) == 0.0
    assert seconds_for(grids[1], "gone") is None
    assert seconds_for("not-a-grid", "SEM") is None


# ── against the bench ────────────────────────────────────────────────────────


def test_the_estimate_is_not_shorter_than_the_arctis_bench_run():
    """FIB-893, 2026-10-02: an exchange took about 3 minutes, a 3 x 3 SEM overview
    at 500 um about 3 and a 3 x 6 FIB overview at 200 um about 4, both at
    1536 x 1024. The estimate errs long; this says it still does once the
    constants are refit."""
    sem = BeamOverviewGridTaskConfig(task_name="SEM", settings=_overview())
    fib_settings = _overview(3, 6)
    fib_settings.image_settings.hfw = 200e-6
    fib_settings.image_settings.beam_type = BeamType.ION
    fib = BeamOverviewGridTaskConfig(
        task_name="FIB", orientation="FIB", settings=fib_settings
    )

    assert AutoscriptSampleLoader(SimpleNamespace()).exchange_seconds >= 3 * 60
    assert sem.estimated_duration >= 3 * 60
    assert fib.estimated_duration >= 4 * 60
    # and not wildly so on the SEM, which the tile move was taken from
    assert sem.estimated_duration < 1.25 * 3 * 60


def test_the_shipped_grid_tasks_have_an_estimate():
    grid_tasks = AutoLamellaTaskProtocol.load(
        cfg.AUTOLAMELLA_TASK_PROTOCOL_PATH
    ).grid_tasks
    for name in grid_tasks.ordered_task_names:
        config = deepcopy(grid_tasks.task_config[name])
        assert config.estimated_duration > 60, name
