"""The simulated stage error (`sim.stage_error`): off by default; when on, the stage
reads back where it was sent while the sample scene renders where it really went."""

import os

import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.drivers.demo.stage_error import StageMoveError
from fibsem.structures import (
    BeamType,
    FibsemStagePosition,
    ImageSettings,
    OverviewAcquisitionSettings,
)

ARCTIS_CONFIG = os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml")


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo", config_path=ARCTIS_CONFIG)
    yield microscope
    microscope.disconnect()


def _turn_on(microscope, **kwargs) -> StageMoveError:
    error = StageMoveError(**kwargs)
    error.start_at(microscope.get_stage_position())
    microscope.stage.sim_error = error
    return error


def test_the_shipped_arctis_config_turns_it_on(microscope):
    error = microscope.stage.sim_error
    assert error is not None
    assert error.scale == (1.06, 1.09)
    assert error.jitter == 0.01


def test_it_is_off_without_the_block():
    microscope, _ = utils.setup_session(manufacturer="Demo")
    try:
        assert "stage_error" not in (microscope.system.sim or {})
        assert microscope.stage.sim_error is None
        position = microscope.get_stage_position()
        assert microscope._connection.scene_stage_position() == position
    finally:
        microscope.disconnect()


@pytest.mark.parametrize(
    "config, expected",
    [
        (None, None),
        ({"enabled": False, "scale": [1.06, 1.09]}, None),
        ({"enabled": True, "scale": [1.06, 1.09]}, (1.06, 1.09)),
        ({"enabled": True, "scale": 1.05}, (1.05, 1.05)),
        ({"enabled": True, "scale": {"x": 1.02, "y": 0.98}}, (1.02, 0.98)),
    ],
)
def test_the_config_block(config, expected):
    error = StageMoveError.from_config(config)
    assert (error.scale if error else None) == expected


def test_a_move_reads_back_as_sent_but_goes_further(microscope):
    _turn_on(microscope, scale=(1.06, 1.09))
    start = microscope.get_stage_position()
    actual_start = microscope._connection.scene_stage_position()

    microscope.move_stage_relative(FibsemStagePosition(x=100e-6, y=50e-6))

    reported = microscope.get_stage_position()
    assert reported.x - start.x == pytest.approx(100e-6)
    assert reported.y - start.y == pytest.approx(50e-6)
    actual = microscope._connection.scene_stage_position()
    assert actual.x - actual_start.x == pytest.approx(106e-6)
    assert actual.y - actual_start.y == pytest.approx(54.5e-6)

    # back where it started is the same place: the scale is about the start
    microscope.move_stage_absolute(start)
    back = microscope._connection.scene_stage_position()
    assert back.x == pytest.approx(actual_start.x)
    assert back.y == pytest.approx(actual_start.y)


def test_the_jitter_is_drawn_fresh_each_move_and_does_not_add_up():
    error = StageMoveError(jitter=0.01, seed=1)
    origin = FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=0.0, t=0.0)
    error.start_at(origin)
    target = FibsemStagePosition(x=100e-6, y=0.0, z=0.0, r=0.0, t=0.0)
    landings = []
    for _ in range(200):
        error.moved(origin, target)
        landings.append(error.actual(target).x)
        error.moved(target, origin)
    misses = [x - 100e-6 for x in landings]
    assert len(set(misses)) == len(misses)
    assert max(abs(m) for m in misses) < 5e-6  # 5 sigma of 1 um
    assert abs(sum(misses) / len(misses)) < 0.3e-6
    # y did not move, so it has no jitter
    assert error.actual(target).y == 0.0


def test_an_axis_that_does_not_move_keeps_its_jitter():
    error = StageMoveError(jitter=0.01, seed=2)
    a = FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=0.0, t=0.0)
    error.start_at(a)
    b = FibsemStagePosition(x=100e-6, y=0.0, z=0.0, r=0.0, t=0.0)
    error.moved(a, b)
    landed = error.actual(b).x
    tilted = FibsemStagePosition(x=100e-6, y=0.0, z=0.0, r=0.0, t=0.2)
    error.moved(b, tilted)
    assert error.actual(tilted).x == landed


def test_the_layout_check_sees_the_scale_error_on_a_tiled_overview(
    microscope, tmp_path
):
    from fibsem.imaging.tiled import acquire_tiled_overview

    _turn_on(microscope, scale=(1.06, 1.09), jitter=0.005, seed=0)
    image_settings = ImageSettings(
        beam_type=BeamType.ELECTRON,
        hfw=150e-6,
        resolution=(512, 512),
        dwell_time=1e-6,
        autocontrast=False,
        save=False,
    )
    result = acquire_tiled_overview(
        microscope,
        OverviewAcquisitionSettings(
            image_settings=image_settings, nrows=3, ncols=3, overlap=0.2
        ),
        str(tmp_path),
        "ov",
    )
    summary = result.layout["summary"]
    assert summary["x"]["step_fraction"] == pytest.approx(1.06, abs=0.015)
    assert summary["y"]["step_fraction"] == pytest.approx(1.09, abs=0.015)
    assert result.layout["warning"] is not None


def test_a_long_move_misses_by_no_more_than_the_cap():
    error = StageMoveError(jitter=0.01, jitter_max=2e-6, seed=4)
    origin = FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=0.0, t=0.0)
    error.start_at(origin)
    far = FibsemStagePosition(x=2e-3, y=0.0, z=0.0, r=0.0, t=0.0)
    misses = []
    for _ in range(200):
        error.moved(origin, far)
        misses.append(error.actual(far).x - 2e-3)
    # 1% of 2 mm would be 20 um; capped at 2 um sigma
    assert max(abs(m) for m in misses) < 10e-6
