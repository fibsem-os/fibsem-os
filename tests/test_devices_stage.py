"""The stage as a device, on the Demo backend.

Three kinds of test. Parity: every stage key through the key router gives what the
untouched old call gives. The reverse direction: a Demo microscope whose old stage
methods ask the device behaves exactly like one that doesn't. And the new API:
axes and limits, refused moves, signals, the stage resource and commands.
"""

import logging
import math
import threading

import pytest

from fibsem import utils
from fibsem.devices import (
    STAGE_COMMAND_ROUTES,
    STAGE_RESOURCE,
    STAGE_ROUTES,
    KeyRouter,
    ParameterReadOnly,
    ParameterUnavailable,
    StageLimitError,
)
from fibsem.devices.drivers.demo import bind_demo_beams, bind_demo_stage
from fibsem.microscope import _records_stage_move
from fibsem.microscopes.simulator import DemoMicroscope
from fibsem.structures import FibsemStagePosition, RangeLimit


def _demo(compustage: bool = False):
    microscope, _ = utils.setup_session(manufacturer="Demo")
    microscope.stage_is_compustage = compustage
    return microscope


def _router(microscope):
    stage = bind_demo_stage(microscope)
    return KeyRouter(microscope, bind_demo_beams(microscope), stage=stage), stage


@pytest.fixture
def microscope():
    return _demo()


@pytest.fixture
def stage(microscope):
    return bind_demo_stage(microscope)


# -- parity: the old keys through the router behave exactly as before ---------------


@pytest.mark.parametrize("compustage", [False, True])
def test_router_get_matches_old_get_for_every_stage_key(compustage):
    microscope = _demo(compustage)
    router, _ = _router(microscope)
    for key in STAGE_ROUTES:
        assert router.get(key) == microscope.get(key), key


@pytest.mark.parametrize("compustage", [False, True])
@pytest.mark.parametrize("key", sorted(STAGE_COMMAND_ROUTES))
def test_router_set_of_a_stage_verb_does_what_the_old_set_does(compustage, key, caplog):
    old, new = _demo(compustage), _demo(compustage)
    for m in (old, new):
        m.stage_system.is_homed = False
        m.stage_system.is_linked = False
    router, _ = _router(new)

    with caplog.at_level(logging.INFO):
        old.set(key, True)
        old_log = [r.getMessage() for r in caplog.records if r.levelno >= logging.INFO]
        caplog.clear()
        router.set(key, True)
        new_log = [r.getMessage() for r in caplog.records if r.levelno >= logging.INFO]

    for state in ("stage_homed", "stage_linked"):
        assert router.get(state) == old.get(state), state
    assert new_log == old_log


def test_router_set_of_a_read_only_stage_key_falls_through_to_the_old_warning(
    microscope, caplog
):
    router, _ = _router(microscope)
    before = microscope.get_stage_position()
    with caplog.at_level(logging.WARNING):
        router.set("stage_position", FibsemStagePosition(x=1e-3))
    assert "Unknown key: stage_position" in caplog.text
    assert microscope.get_stage_position() == before


# -- the reverse direction: old methods ask the device ------------------------------


class DeviceBackedDemo(DemoMicroscope):
    """What ``DemoMicroscope`` becomes once its stage keys move: the old stage
    methods keep their names, signatures, returns and recording, and ask the device.

    Only the methods whose Demo implementation touches the stage are here. The base
    class's ``get_stage_position``, ``home`` and ``link_stage`` already go through
    ``get``/``set``, so the router covers them.
    """

    router = None  # until the devices are bound; binding reads through the old API

    def _use_devices(self) -> None:
        self.stage_device = bind_demo_stage(self)
        self.router = KeyRouter(self, bind_demo_beams(self), stage=self.stage_device)

    def get(self, key, beam_type=None):
        if self.router is None:
            return super().get(key, beam_type)
        return self.router.get(key, beam_type)

    def set(self, key, value, beam_type=None):
        if self.router is None:
            return super().set(key, value, beam_type)
        self.router.set(key, value, beam_type)

    @_records_stage_move
    def move_stage_absolute(self, position):
        self.stage_device.move_through(position)
        return self.get_stage_position()

    @_records_stage_move
    def move_stage_relative(self, position):
        self.stage_device.move_through(position, relative=True)
        return self.get_stage_position()

    def home(self):
        return self.stage_device.home()


def _unhomed(microscope):
    """Unhome and unlink the stage, so home and link have something to do. The stage
    device copies this state when it is built."""
    microscope.stage_system.is_homed = False
    microscope.stage_system.is_linked = False
    return microscope


def _device_backed(compustage: bool = False) -> DeviceBackedDemo:
    microscope = _unhomed(_demo(compustage))
    microscope.__class__ = DeviceBackedDemo  # same connected state, the new methods
    microscope._use_devices()
    return microscope


def _script(microscope):
    """The old stage API, as scripts and the UI call it today, on an unhomed stage."""
    emitted, recorded = [], []
    microscope.stage_position_changed.connect(emitted.append)
    microscope.record_signal.connect(
        lambda kind, payload: recorded.append((kind, payload))
    )
    stage = microscope._stage
    out = [
        microscope.get_stage_position(),
        stage.position,
        stage.axes,
        microscope.move_stage_absolute(FibsemStagePosition(x=1e-3, y=-2e-3)),
        microscope.move_stage_relative(FibsemStagePosition(z=1e-4, t=0.1)),
        # the old API has no limit check on Demo, and still has none
        microscope.move_stage_absolute(FibsemStagePosition(x=0.5)),
        stage.move_absolute(FibsemStagePosition(x=0.0)),
        stage.orientation,
        stage.is_homed,
        microscope.home(),
        stage.is_homed,
        microscope.link_stage(),
        microscope.get("stage_linked"),
        microscope.get_stage_position(),
    ]
    events = [
        (p["move"], p["request"], p["start"], p["end"], p["error"])
        for kind, p in recorded
        if kind == "stage_moved"
    ]
    return out, emitted, events


@pytest.mark.parametrize("compustage", [False, True])
def test_old_stage_api_is_unchanged_when_it_asks_the_device(compustage):
    old_out, old_emitted, old_events = _script(_unhomed(_demo(compustage)))
    new_out, new_emitted, new_events = _script(_device_backed(compustage))
    assert new_out == old_out
    assert new_emitted == old_emitted
    assert new_events == old_events and len(new_events) == 4


def test_the_old_signal_and_the_new_one_fire_together():
    microscope = _device_backed()
    old, new = [], []
    microscope.stage_position_changed.connect(old.append)
    microscope.stage_device.position.changed.connect(new.append)
    microscope.get_stage_position()
    microscope.move_stage_relative(FibsemStagePosition(x=1e-4))
    assert old == new and len(new) == 1


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
    stage = bind_demo_stage(_demo(compustage=True))
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
    stage.state.position.t = 0.25
    assert stage.axes.t.value == 0.25  # a live read, of the whole position
    stage.state.position.t = 0.5
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
    assert result == stage.state.position
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
    stage.state.is_homed = False
    stage.state.is_linked = False
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
