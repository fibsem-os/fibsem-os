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
    assert stage.axes.z.limits == RangeLimit(min=-40e-3, max=40e-3)
    assert stage.axes["z"] is stage.axes.z
    # the simulator's table gives r and t in degrees; the axes carry radians
    t = stage.axes.t.limits
    assert (t.min, t.max) == pytest.approx((math.radians(-60), math.radians(90)))
    r = stage.position.limits["r"]
    assert (r.min, r.max) == pytest.approx((-2 * math.pi, 2 * math.pi))
    assert stage.describe()["position"]["limits"]["z"] == {"min": -40e-3, "max": 40e-3}


def test_is_homed_reads_the_homed_parameter(stage):
    assert stage.is_homed is True
    stage.sim_homed = False
    assert stage.is_homed is False


def test_is_homed_is_none_for_a_stage_that_cannot_say(stage, monkeypatch):
    parameters = {k: v for k, v in stage.parameters.items() if k != "homed"}
    monkeypatch.setattr(type(stage), "parameters", property(lambda _: parameters))
    assert stage.is_homed is None


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


def test_readback_jitter_below_the_tolerance_does_not_signal(stage):
    """A real stage's readback wanders by nanometres between reads. Every read
    signalling would redraw the overview and fill the event feed while nothing moved."""
    stage.position.get_value()
    positions = []
    stage.position.changed.connect(positions.append)

    stage.sim_position = stage.sim_position + FibsemStagePosition(x=1e-9, t=1e-9)
    read = stage.position.get_value()

    assert positions == []
    assert stage.position.cached == read  # the cache still holds the exact read


def test_drift_below_the_tolerance_per_read_still_signals_once_it_adds_up(stage):
    """Compared with what was last signalled, not with the read before, so a slow
    drift is seen once it passes the tolerance."""
    stage.position.get_value()
    positions = []
    stage.position.changed.connect(positions.append)

    for _ in range(4):
        stage.sim_position = stage.sim_position + FibsemStagePosition(x=4e-7)
        stage.position.get_value()

    assert len(positions) == 1


def test_home_and_link_are_commands_that_report_the_result(stage):
    stage.sim_homed = False
    stage.sim_linked = False
    assert stage.home() is True and stage.homed.cached is True
    assert stage.link() is True and stage.linked.cached is True
    assert set(stage.commands) == {
        "home",
        "link",
        "move_absolute",
        "move_relative",
    }
    assert set(stage.queries) == {"pose_table"}
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


@pytest.mark.parametrize(
    "a, b, same",
    [
        (FibsemStagePosition(x=0, t=0), FibsemStagePosition(x=5e-7, t=5e-7), True),
        (FibsemStagePosition(x=0, t=0), FibsemStagePosition(x=2e-6, t=0), False),
        (FibsemStagePosition(x=0, r=None), FibsemStagePosition(x=1e-9, r=None), True),
        (FibsemStagePosition(x=0, r=None), FibsemStagePosition(x=0, r=0.0), False),
        (
            FibsemStagePosition(x=0, coordinate_system="RAW"),
            FibsemStagePosition(x=0, coordinate_system="SPECIMEN"),
            False,
        ),
    ],
)
def test_positions_are_the_same_within_the_tolerance(a, b, same):
    from fibsem.devices.core import _same

    assert _same(a, b) is same


def test_the_microscope_has_no_stage_signal_or_cache_of_its_own():
    """A stage move is on the stage device; the microscope keeps no second copy."""
    microscope, _ = utils.setup_session(manufacturer="Demo", setup_logging=False)
    positions = []
    microscope.stage.position.changed.connect(positions.append)

    microscope.stage.move_relative(FibsemStagePosition(x=2e-4))

    assert len(positions) == 1
    assert microscope.stage.position.last_known == positions[0]
    assert not hasattr(microscope, "stage_position_changed")
    assert not hasattr(microscope, "_stage_position")
