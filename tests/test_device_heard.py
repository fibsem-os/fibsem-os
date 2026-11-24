"""When a device was last heard from: noted from the calls a run makes, never
raising into them or holding them up."""

import pytest

from fibsem.devices import heard as h
from fibsem.devices.heard import heard_call, heard_of


class Part:
    name = "stage"


class ParameterOutOfRange(ValueError):
    """Named as the device layer's refusal."""


def test_a_call_that_comes_back_is_the_last_heard():
    part, now = Part(), [10.0]
    with heard_call(part, "move", clock=lambda: now[0]):
        assert [c.call for c in heard_of(part).running()] == ["move"]
        now[0] = 12.0
    assert heard_of(part).last_ok == 12.0
    assert heard_of(part).running() == []


def test_a_driver_failure_is_kept_and_passed_on_unchanged():
    part = Part()
    error = RuntimeError("AutoScript lost the connection")
    with pytest.raises(RuntimeError) as raised:
        with heard_call(part, "move"):
            raise error
    assert raised.value is error
    last = heard_of(part).last_error
    assert last.call == "move" and "lost the connection" in last.message
    assert heard_of(part).last_ok is None


def test_a_refusal_says_nothing_of_the_hardware():
    part = Part()
    with pytest.raises(ParameterOutOfRange):
        with heard_call(part, "set exposure_time"):
            raise ParameterOutOfRange("over the limit")
    assert heard_of(part).last_error is None


def test_a_failure_to_note_never_stops_the_call(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("the recorder is broken")

    monkeypatch.setattr(h.Heard, "_start", broken)
    monkeypatch.setattr(h.Heard, "_end", broken)
    part, ran = Part(), []
    with heard_call(part, "move"):
        ran.append(True)
    assert ran == [True]
    with pytest.raises(KeyError):  # the call's own error, not the recorder's
        with heard_call(part, "move"):
            raise KeyError("its own")


def test_a_part_that_keeps_no_attributes_is_still_called():
    class Slotted:
        __slots__ = ()

    with heard_call(Slotted(), "move"):
        pass


def test_device_reads_writes_and_commands_are_heard(monkeypatch):
    monkeypatch.setenv("FIBSEM_SIM_NO_DELAY", "1")
    from fibsem import utils

    microscope, _ = utils.setup_session()
    try:
        devices = microscope.devices
        name, device = next(
            (n, d) for n, d in devices.items() if getattr(d, "_bound", None)
        )
        parameter = next(iter(device._bound.values()))
        parameter.get_value()
        assert heard_of(device).last_ok is not None, name
        stage = devices["stage"]
        before = heard_of(stage).last_ok
        microscope.get_stage_position()
        microscope.move_stage_relative(type(microscope.get_stage_position())(x=0))
        assert heard_of(stage).last_ok is not None
        assert heard_of(stage).last_ok != before
    finally:
        microscope.disconnect()
