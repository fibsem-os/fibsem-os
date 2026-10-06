"""The stage as a device, on the Demo backend: axes and limits, refused moves,
signals, the stage resource and commands."""

import math
import os
import threading

import pytest

from fibsem import utils
from fibsem.config import CONFIG_PATH
from fibsem.devices import (
    STAGE_RESOURCE,
    ParameterReadOnly,
    ParameterUnavailable,
    StageLimitError,
)
from fibsem.structures import FibsemStagePosition, RangeLimit

# A simulated Arctis, whose stage is a compustage.
_ARCTIS_CONFIG = os.path.join(CONFIG_PATH, "sim-arctis-configuration.yaml")


def _demo_stage(config_path=None):
    microscope, _ = utils.setup_session(
        manufacturer="Demo", config_path=config_path, setup_logging=False
    )
    return microscope.stage


@pytest.fixture
def stage():
    return _demo_stage()


# -- the new API ---------------------------------------------------------------------


def test_the_driver_lists_the_axes_with_their_limits_in_si_units(stage):
    assert list(stage.axes) == ["x", "y", "z", "r", "t"]
    assert stage.axes.x.unit == "m" and stage.axes.t.unit == "rad"
    assert stage.axes.z.limits == RangeLimit(min=0.0, max=40e-3)
    assert stage.axes["z"] is stage.axes.z
    # _get_axis_limits gives r and t in degrees; the axes carry radians
    t = stage.axes.t.limits
    assert (t.min, t.max) == pytest.approx((math.radians(-10), math.radians(90)))
    r = stage.position.limits["r"]
    assert (r.min, r.max) == pytest.approx((-2 * math.pi, 2 * math.pi))
    assert stage.describe()["position"]["limits"]["z"] == {"min": 0.0, "max": 40e-3}


def test_a_compustage_has_no_rotation_axis_and_cannot_link():
    stage = _demo_stage(_ARCTIS_CONFIG)
    assert list(stage.axes) == ["x", "y", "z", "t"]
    assert "r" not in stage.axes
    with pytest.raises(AttributeError, match="no 'r' axis"):
        stage.axes.r
    assert "linked" not in stage.parameters
    with pytest.raises(ParameterUnavailable):
        stage.linked
    assert stage.commands["link"].available is False
    assert stage.commands["home"].available is True


def test_an_axis_is_a_view_of_the_position_with_no_read_of_its_own(stage):
    stage.sim_position.t = 0.25
    assert stage.axes.t.value == 0.25  # a live read, of the whole position
    stage.sim_position.t = 0.5
    assert stage.axes.t.cached == 0.25  # no read: the last position read
    assert stage.position.cached.t == 0.25


def test_state_parameters_are_read_only(stage):
    assert set(stage.parameters) == {"position", "homed", "linked"}
    for name in stage.parameters:
        assert stage.parameters[name].settable is False, name
    with pytest.raises(ParameterReadOnly):
        stage.position.value = FibsemStagePosition(x=0.0)


def test_move_absolute_moves_and_returns_the_read_back_position(stage):
    result = stage.move_absolute(FibsemStagePosition(x=1e-3, z=2e-3))
    assert result == stage.sim_position
    assert (result.x, result.z) == (1e-3, 2e-3)
    assert stage.axes.x.cached == 1e-3


def test_a_move_outside_the_limits_is_refused_and_nothing_moves(stage):
    before = stage.position.get_value()
    with pytest.raises(StageLimitError, match=r"x=0.5 not in"):
        stage.move_absolute(FibsemStagePosition(x=0.5))
    with pytest.raises(StageLimitError, match=r"z=0.05 not in"):
        stage.move_relative(FibsemStagePosition(z=50e-3))
    assert stage.position.get_value() == before


def test_a_move_emits_position_and_axis_changes(stage):
    stage.position.get_value()
    positions, xs, ys = [], [], []
    stage.position.changed.connect(positions.append)
    stage.axes.x.changed.connect(xs.append)
    stage.axes.y.changed.connect(ys.append)

    stage.move_relative(FibsemStagePosition(x=2e-4))

    assert len(positions) == 1 and positions[0].x == pytest.approx(2e-4)
    assert xs == [pytest.approx(2e-4)]
    assert ys == []  # an axis that didn't move stays quiet


def test_home_and_link_are_commands_that_report_the_result(stage):
    stage.sim_homed = False
    stage.sim_linked = False
    assert stage.home() is True and stage.homed.cached is True
    assert stage.link() is True and stage.linked.cached is True
    assert set(stage.commands) == {"home", "link", "move_absolute", "move_relative"}
    assert (
        stage.commands["move_absolute"].signature
        == "(position: 'FibsemStagePosition') -> 'FibsemStagePosition'"
    )


def test_a_move_holds_the_stage_resource(stage):
    lock = stage.resources.lock(STAGE_RESOURCE)
    moved = threading.Event()
    with lock:
        worker = threading.Thread(
            target=lambda: (
                stage.move_relative(FibsemStagePosition(x=1e-5)),
                moved.set(),
            )
        )
        worker.start()
        assert not moved.wait(0.2)  # blocked while another holder has the stage
    worker.join(5)
    assert moved.is_set()
