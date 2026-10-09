"""The sample stage's limits are the stage device's, not a second read.

They used to come from `FibsemMicroscope._get_axis_limits`, a vendor read on the
microscope class that the stage drivers also called back into, so one read had two
homes in two units. On Tescan the two disagreed: the device reported every axis
unlimited while the sample stage got the base class's made-up table. Now the stage
device is the only source, and the sample stage keeps its units (metres, and degrees
for r and t).
"""

import math
import os
from types import SimpleNamespace

import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.devices.stage import axis_limits_from_degrees, axis_limits_to_degrees
from fibsem.drivers.demo.simulator import STAGE_LIMITS_COMPUSTAGE, STAGE_LIMITS_DEFAULT
from fibsem.microscopes._stage import _stage_device_limits
from fibsem.structures import RangeLimit


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


def test_the_sample_stage_reads_the_stage_device(connected):
    microscope, _ = connected
    assert microscope._stage.limits == axis_limits_to_degrees(
        microscope.stage.position.limits
    )


def test_the_sample_stage_keeps_degrees_for_rotations(connected):
    microscope, table = connected
    limits = microscope._stage.limits
    assert set(limits) == set(table)
    for axis, expected in table.items():
        assert (limits[axis].min, limits[axis].max) == pytest.approx(
            (expected.min, expected.max)
        )


def test_the_rotation_capability_follows_the_device_axes(connected):
    microscope, _ = connected
    assert microscope.system.stage.rotation is ("r" in microscope.stage.axes)


def test_no_stage_device_means_unknown_limits():
    assert _stage_device_limits(SimpleNamespace(stage=None)) == {}


def test_an_unlimited_device_reads_as_unlimited():
    """Tescan's stage reports every axis unlimited; the sample stage now says the same
    instead of inventing a range."""
    unlimited = RangeLimit(min=-math.inf, max=math.inf)
    stage = SimpleNamespace(
        position=SimpleNamespace(limits={"x": unlimited, "t": unlimited})
    )
    assert _stage_device_limits(SimpleNamespace(stage=stage)) == {
        "x": unlimited,
        "t": unlimited,
    }


def test_the_two_conversions_are_inverses():
    round_trip = axis_limits_to_degrees(axis_limits_from_degrees(STAGE_LIMITS_DEFAULT))
    for axis, limit in STAGE_LIMITS_DEFAULT.items():
        assert (round_trip[axis].min, round_trip[axis].max) == pytest.approx(
            (limit.min, limit.max)
        )
