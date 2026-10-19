"""FibsemAxesPosition: the shared maths of stage and manipulator positions (FIB-1247).

The dict pins were written against the two classes before they shared a base, so they
check that the base changed nothing on disk or on the wire: lamella YAML, image
metadata and the device server all go through ``to_dict``/``from_dict``.
"""

import math

import numpy as np
import pytest

from fibsem.structures import (
    FibsemAxesPosition,
    FibsemManipulatorPosition,
    FibsemRectangle,
    FibsemStagePosition,
    Point,
    RangeLimit,
)

# -- dict pins ---------------------------------------------------------------


def test_stage_to_dict_is_unchanged():
    position = FibsemStagePosition(
        name="p",
        x=1,
        y=2e-3,
        z=np.float64(3e-3),
        r=0.5,
        t=None,
        coordinate_system="RAW",
    )
    data = position.to_dict()
    assert data == {
        "name": "p",
        "x": 1.0,
        "y": 2e-3,
        "z": 3e-3,
        "r": 0.5,
        "t": None,
        "coordinate_system": "RAW",
    }
    assert list(data) == ["name", "x", "y", "z", "r", "t", "coordinate_system"]
    assert all(type(data[a]) is float for a in "xyzr")


def test_stage_dict_round_trip():
    position = FibsemStagePosition(
        name="p", x=1e-3, y=None, z=3e-3, r=0.5, t=0.1, coordinate_system="RAW"
    )
    assert FibsemStagePosition.from_dict(position.to_dict()) == position


def test_stage_from_dict_without_name():
    data = {"x": 1, "y": 2, "z": 3, "r": 0, "t": 0, "coordinate_system": None}
    assert FibsemStagePosition.from_dict(data).name is None


def test_manipulator_to_dict_is_unchanged():
    position = FibsemManipulatorPosition(x=1, y=2e-3, z=3e-3, coordinate_system="raw")
    data = position.to_dict()
    assert data == {
        "x": 1.0,
        "y": 2e-3,
        "z": 3e-3,
        "r": 0.0,
        "t": 0.0,
        "coordinate_system": "RAW",
    }
    assert list(data) == ["x", "y", "z", "r", "t", "coordinate_system"]


def test_manipulator_dict_round_trip():
    position = FibsemManipulatorPosition(x=1e-3, y=2e-3, z=3e-3, r=0.1, t=0.2)
    assert FibsemManipulatorPosition.from_dict(position.to_dict()) == position


def test_manipulator_defaults_are_unchanged():
    assert FibsemManipulatorPosition() == FibsemManipulatorPosition(0, 0, 0, 0, 0, None)
    assert FibsemManipulatorPosition(x=1).y == 0.0


def test_numpy_scalars_become_plain_numbers():
    for cls in (FibsemStagePosition, FibsemManipulatorPosition):
        position = cls(x=np.float64(1.5), r=np.float32(0.5))
        assert type(position.x) is float
        assert type(position.r) is float


def test_both_classes_share_the_base():
    assert issubclass(FibsemStagePosition, FibsemAxesPosition)
    assert issubclass(FibsemManipulatorPosition, FibsemAxesPosition)
    assert FibsemStagePosition.AXES == ("x", "y", "z", "r", "t")
    assert FibsemManipulatorPosition.AXES == ("x", "y", "z", "r", "t")


# -- arithmetic --------------------------------------------------------------


def test_stage_add_keeps_self_where_other_is_none():
    a = FibsemStagePosition(
        name="a", x=1, y=2, z=3, r=0.1, t=0.2, coordinate_system="RAW"
    )
    b = FibsemStagePosition(x=1, t=0.1)
    result = a + b
    assert result == FibsemStagePosition(
        x=2, y=2, z=3, r=0.1, t=pytest.approx(0.3), coordinate_system="RAW"
    )
    assert result.name is None


def test_add_keeps_none_where_self_is_none():
    """An unknown axis plus an offset stays unknown, so an absolute move leaves it."""
    a = FibsemStagePosition(x=None, y=1)
    assert (a + FibsemStagePosition(x=1, y=1)) == FibsemStagePosition(x=None, y=2)


def test_sub_is_none_safe():
    a = FibsemStagePosition(x=3, y=None, z=1, coordinate_system="RAW")
    b = FibsemStagePosition(x=1, y=1, z=None)
    assert a - b == FibsemStagePosition(x=2, y=None, z=None, coordinate_system="RAW")


def test_manipulator_add_and_sub():
    a = FibsemManipulatorPosition(x=1, y=2, z=3, coordinate_system="STAGE")
    b = FibsemManipulatorPosition(x=1, y=1, z=1)
    assert a + b == FibsemManipulatorPosition(x=2, y=3, z=4, coordinate_system="STAGE")
    assert a - b == FibsemManipulatorPosition(x=0, y=1, z=2, coordinate_system="STAGE")
    assert type(a + b) is FibsemManipulatorPosition


def test_manipulator_add_no_longer_crashes_on_none():
    a = FibsemManipulatorPosition(x=1, y=2, z=3)
    assert (a + FibsemManipulatorPosition(x=1, y=None, z=None)).y == 2


def test_arithmetic_refuses_the_other_position_type():
    with pytest.raises(TypeError):
        FibsemStagePosition(x=1) + FibsemManipulatorPosition(x=1)
    with pytest.raises(TypeError):
        FibsemManipulatorPosition(x=1) - FibsemStagePosition(x=1)


# -- distance ----------------------------------------------------------------


def test_distance_defaults_to_linear_axes_euclidean():
    a = FibsemStagePosition(x=0, y=0, z=0, r=0, t=0)
    b = FibsemStagePosition(x=3e-3, y=4e-3, z=0, r=1.0, t=1.0)
    assert a.distance(b) == pytest.approx(5e-3)


def test_distance_metrics():
    a = FibsemStagePosition(x=0, y=0, z=0)
    b = FibsemStagePosition(x=3, y=-4, z=1)
    assert a.distance(b, metric="euclidean") == pytest.approx(math.sqrt(26))
    assert a.distance(b, metric="manhattan") == pytest.approx(8)
    assert a.distance(b, metric="max") == pytest.approx(4)
    with pytest.raises(ValueError):
        a.distance(b, metric="chebyshev")


def test_distance_skips_none_axes():
    a = FibsemStagePosition(x=0, y=None, z=0)
    b = FibsemStagePosition(x=3, y=100, z=4)
    assert a.distance(b) == pytest.approx(5)
    assert FibsemStagePosition().distance(b) == 0.0


def test_distance_wraps_angular_axes():
    a = FibsemStagePosition(r=np.deg2rad(359.9), t=0)
    b = FibsemStagePosition(r=np.deg2rad(0.1), t=0)
    assert a.distance(b, axes=["r"]) == pytest.approx(np.deg2rad(0.2))


def test_distance_rejects_unknown_axes():
    with pytest.raises(ValueError):
        FibsemStagePosition(x=0).distance(FibsemStagePosition(x=1), axes=["q"])


def test_euclidean_distance_is_unchanged():
    a = FibsemStagePosition(x=1, y=None, z=0, r=0.0, t=0.0)
    b = FibsemStagePosition(x=4, y=7, z=4, r=3.0, t=1.0)
    assert a.euclidean_distance(b) == pytest.approx(5)
    assert type(a.euclidean_distance(b)) is float


def test_manipulator_has_distance():
    a = FibsemManipulatorPosition(x=0, y=0, z=0)
    assert a.distance(FibsemManipulatorPosition(x=3e-6, y=4e-6)) == pytest.approx(5e-6)


# -- closeness ---------------------------------------------------------------


def test_is_close_per_axis_tolerance():
    a = FibsemStagePosition(x=0, y=0, z=0, r=0, t=0)
    assert a.is_close(FibsemStagePosition(x=5e-7, y=-5e-7, z=0, r=0, t=0))
    assert not a.is_close(FibsemStagePosition(x=2e-6, y=0, z=0, r=0, t=0))
    assert a.is_close(FibsemStagePosition(x=2e-6, y=0, z=0, r=0, t=0), tol=1e-5)


def test_is_close_wraps_rotation():
    a = FibsemStagePosition(x=0, y=0, z=0, r=2 * np.pi - 1e-9, t=0)
    b = FibsemStagePosition(x=0, y=0, z=0, r=1e-9, t=0)
    assert a.is_close(b)
    assert a.is_close2(b, axes=["r"])


def test_is_close_is_false_when_an_axis_is_unknown():
    a = FibsemStagePosition(x=0, y=0, z=0, r=0, t=0)
    assert not a.is_close(FibsemStagePosition(x=0, y=0, z=0, r=None, t=0))
    assert not a.is_close2(FibsemStagePosition(x=0, y=0, z=0, r=None, t=0))
    assert a.is_close2(FibsemStagePosition(x=0, y=0, z=None), axes=["x", "y"])


def test_is_close2_checks_only_named_axes():
    a = FibsemStagePosition(x=0, y=0, z=0, r=0, t=0)
    b = FibsemStagePosition(x=0, y=0, z=1, r=1, t=1)
    assert a.is_close2(b, tol=1e-6, axes=["x", "y"])
    assert not a.is_close2(b, tol=1e-6)
    with pytest.raises(ValueError):
        a.is_close2(b, axes=["x", "w"])


def test_manipulator_is_close():
    a = FibsemManipulatorPosition(x=1e-3)
    assert a.is_close(FibsemManipulatorPosition(x=1e-3 + 1e-9))
    assert not a.is_close(FibsemManipulatorPosition(x=2e-3))


# -- limits and display ------------------------------------------------------


def test_is_within_limits():
    limits = {"x": RangeLimit(min=-1e-3, max=1e-3), "t": RangeLimit(min=0, max=1)}
    assert FibsemStagePosition(x=0, t=0.5).is_within_limits(limits)
    assert not FibsemStagePosition(x=2e-3, t=0.5).is_within_limits(limits)
    assert FibsemStagePosition(x=2e-3, t=0.5).is_within_limits(limits, axes=["t"])
    assert FibsemStagePosition(x=None, t=None).is_within_limits(limits)
    assert not FibsemManipulatorPosition(x=2e-3).is_within_limits(limits)


def test_pretty_strings_are_unchanged():
    position = FibsemStagePosition(x=1e-3, y=-2e-3, z=None, r=np.pi, t=np.pi / 6)
    assert position.pretty == "X:1.00mm, Y:-2.00mm, Z:None, R:180.0°, T:30.0°"
    assert position.pretty_string == "X:1.00, Y:-2.00, Z:None, R:180.0, T:30.0"
    assert position.pretty_orientation == "R:180.0, T:30.0"
    assert FibsemStagePosition(x=1e-3, y=2e-3, z=3e-3)._scale_repr(1e3, 1) == (
        "x:1.0, y:2.0, z:3.0"
    )
    assert FibsemManipulatorPosition(x=1e-3).pretty == (
        "X:1.00mm, Y:0.00mm, Z:0.00mm, R:0.0°, T:0.0°"
    )


@pytest.mark.parametrize(
    "cls, data",
    [
        (Point, {"x": 1.0}),
        (FibsemStagePosition, {"x": 0, "y": 0, "z": 0, "r": 0, "t": 0}),
        (FibsemManipulatorPosition, {"x": 0, "y": 0, "z": 0, "r": 0}),
        (FibsemRectangle, {"left": 0, "top": 0, "width": 1}),
    ],
)
def test_a_position_read_without_one_of_its_keys_is_refused(cls, data):
    with pytest.raises(ValueError, match="missing"):
        cls.from_dict(data)
