"""The needle's axes come from its position limits, as the stage's do (FIB-1247)."""

import math
from copy import deepcopy

import pytest

from fibsem import utils
from fibsem.devices.core import UNLIMITED, AxisLimitError, ParameterMetadata
from fibsem.devices.manipulator import Manipulator, ManipulatorLimitError
from fibsem.devices.stage import StageLimitError
from fibsem.structures import (
    FibsemManipulatorPosition,
    InsertableDeviceState,
    RangeLimit,
)

LIMITS = {
    "x": RangeLimit(min=-1e-3, max=1e-3),
    "y": RangeLimit(min=-1e-3, max=1e-3),
    "z": RangeLimit(min=0.0, max=2e-3),
    "r": UNLIMITED,
}


class BoundedNeedle(Manipulator):
    """A needle with bounded x, y and z, and a rotation it cannot bound."""

    def __init__(self, limits=LIMITS):
        super().__init__()
        self.limits = limits
        self.sim_position = FibsemManipulatorPosition()
        self.moves = []

    def read_position(self):
        return deepcopy(self.sim_position)

    def read_state(self):
        return InsertableDeviceState.INSERTED

    def metadata_position(self):
        return ParameterMetadata(limits=self.limits)

    def _move_absolute(self, position):
        self.moves.append(position)
        self.sim_position = deepcopy(position)

    def _move_relative(self, delta):
        self.moves.append(delta)
        self.sim_position = self.sim_position + delta


def test_the_axes_are_the_keys_of_the_position_limits():
    needle = BoundedNeedle().connect()
    assert list(needle.axes) == ["x", "y", "z", "r"]
    assert "t" not in needle.axes
    assert needle.axes.z.limits == RangeLimit(min=0.0, max=2e-3)
    assert needle.axes.r.unit == "rad"
    assert math.isinf(needle.axes.r.limits.max)


def test_an_axis_is_a_view_of_the_position():
    needle = BoundedNeedle().connect()
    needle.move_absolute(FibsemManipulatorPosition(x=1e-4, y=0, z=1e-4))
    assert needle.axes.x.value == pytest.approx(1e-4)
    assert needle.axes.x.cached == pytest.approx(1e-4)


def test_a_driver_that_gives_no_limits_has_x_y_z_unbounded():
    microscope, _ = utils.setup_session(manufacturer="Demo", setup_logging=False)
    try:
        needle = microscope.manipulator_device
        assert list(needle.axes) == ["x", "y", "z"]
        assert all(math.isinf(needle.axes[a].limits.max) for a in needle.axes)
        assert microscope.is_available("manipulator_rotation") is False
    finally:
        microscope.disconnect()


def test_an_absolute_move_outside_the_limits_is_refused():
    needle = BoundedNeedle().connect()
    with pytest.raises(ManipulatorLimitError, match=r"x=0.002 not in"):
        needle.move_absolute(FibsemManipulatorPosition(x=2e-3, y=0, z=1e-4))
    assert needle.moves == []


def test_a_relative_move_is_checked_from_where_the_needle_is():
    needle = BoundedNeedle().connect()
    needle.move_absolute(FibsemManipulatorPosition(x=0, y=0, z=1.5e-3))
    with pytest.raises(ManipulatorLimitError, match=r"z=0.0025"):
        needle.move_relative(FibsemManipulatorPosition(x=0, y=0, z=1e-3))
    assert len(needle.moves) == 1
    needle.move_relative(FibsemManipulatorPosition(x=0, y=0, z=1e-4))
    assert needle.sim_position.z == pytest.approx(1.6e-3)


def test_an_axis_the_needle_does_not_have_is_not_checked():
    needle = BoundedNeedle().connect()
    needle.move_absolute(FibsemManipulatorPosition(x=0, y=0, z=0, t=10.0))
    assert needle.sim_position.t == 10.0


def test_the_rotation_is_available_when_the_driver_gives_r_limits():
    microscope, _ = utils.setup_session(manufacturer="Demo", setup_logging=False)
    try:
        microscope.manipulator_device = BoundedNeedle().connect()
        assert microscope.is_available("manipulator_rotation") is True
        assert microscope.is_available("manipulator_tilt") is False
    finally:
        microscope.disconnect()


def test_both_limit_errors_are_axis_limit_errors():
    assert issubclass(ManipulatorLimitError, AxisLimitError)
    assert issubclass(StageLimitError, AxisLimitError)
    assert issubclass(AxisLimitError, ValueError)
