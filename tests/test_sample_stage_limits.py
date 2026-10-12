"""Stage limits have one home: the stage device's `position.limits`.

They used to come from `FibsemMicroscope._get_axis_limits`, a vendor read on the
microscope class that the stage drivers also called back into, so one read had two
homes in two units. On Tescan the two disagreed: the device reported every axis
unlimited while the sample stage got the base class's made-up table. Then the sample
stage kept a copy in degrees, taken once at connect; every reader only checks x and
y, so they read the device now and the sample stage has no limits of its own.
"""

import os

import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.devices.stage import axis_limits_from_degrees, axis_limits_to_degrees
from fibsem.drivers.demo.simulator import STAGE_LIMITS_COMPUSTAGE, STAGE_LIMITS_DEFAULT


@pytest.fixture(
    scope="module",
    params=[
        ("microscope-configuration.yaml", STAGE_LIMITS_DEFAULT),
        ("sim-arctis-configuration.yaml", STAGE_LIMITS_COMPUSTAGE),
    ],
    ids=["flat-stage", "compustage"],
)
def connected(request):
    configuration, table = request.param
    microscope, _ = utils.setup_session(
        config_path=os.path.join(cfg.CONFIG_PATH, configuration),
        setup_logging=False,
    )
    return microscope, table


def test_the_sample_stage_has_no_limits_of_its_own(connected):
    microscope, _ = connected
    assert not hasattr(microscope._stage, "limits")


def test_the_stage_device_reports_the_simulator_table(connected):
    microscope, table = connected
    limits = axis_limits_to_degrees(microscope.stage.position.limits)
    assert set(limits) == set(table)
    for axis, expected in table.items():
        assert (limits[axis].min, limits[axis].max) == pytest.approx(
            (expected.min, expected.max)
        )


def test_the_rotation_capability_follows_the_device_axes(connected):
    microscope, _ = connected
    assert microscope.system.stage.rotation is ("r" in microscope.stage.axes)


def test_the_two_conversions_are_inverses():
    round_trip = axis_limits_to_degrees(axis_limits_from_degrees(STAGE_LIMITS_DEFAULT))
    for axis, limit in STAGE_LIMITS_DEFAULT.items():
        assert (round_trip[axis].min, round_trip[axis].max) == pytest.approx(
            (limit.min, limit.max)
        )
