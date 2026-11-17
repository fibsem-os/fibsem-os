"""A numpy value written to a parameter, or passed to a command, reaches the driver as
the plain Python it holds, on every write path."""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import pytest

from fibsem.devices import Device, DeviceParameter, command
from fibsem.devices.wire import to_wire
from fibsem.structures.settings import Point

BUILTINS = (bool, int, float, str, type(None))


def _all_builtin(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return all(_all_builtin(v) for v in value)
    if isinstance(value, dict):
        return all(_all_builtin(k) and _all_builtin(v) for k, v in value.items())
    return type(value) in BUILTINS


class Recorder(Device):
    exposure = DeviceParameter(float)
    binning = DeviceParameter(int)
    size = DeviceParameter(tuple)

    def __init__(self) -> None:
        super().__init__("recorder")
        self.written: List[Any] = []
        self.called: List[Tuple[Any, ...]] = []
        for name, value in (("exposure", 0.1), ("binning", 1), ("size", (8, 8))):
            self.bind(name, read=lambda v=value: v, write=self.written.append)

    @command
    def move(self, position: float, steps: Any = None) -> None:
        self.called.append((position, steps))

    @command(long=True)
    def sweep(self, positions: Any) -> None:
        self.called.append((positions,))


NUMPY_VALUES = [
    ("exposure", np.float64(0.2), 0.2),
    ("exposure", np.float32(0.5), 0.5),
    ("binning", np.int64(2), 2),
    ("size", (np.int64(16), np.int32(32)), (16, 32)),
]


@pytest.mark.parametrize("name,value,expected", NUMPY_VALUES)
def test_set_value_writes_plain_python(name, value, expected):
    device = Recorder()
    getattr(device, name).set_value(value)
    assert device.written == [expected]
    assert _all_builtin(device.written[0])


@pytest.mark.parametrize("name,value,expected", NUMPY_VALUES)
def test_write_through_writes_plain_python(name, value, expected):
    device = Recorder()
    getattr(device, name).write_through(value)
    assert device.written == [expected]
    assert _all_builtin(device.written[0])


def test_a_numpy_bool_is_still_refused_for_a_number():
    with pytest.raises(TypeError):
        Recorder().binning.set_value(np.bool_(True))


def test_command_arguments_arrive_as_plain_python():
    device = Recorder()
    device.move(np.float64(1e-6), steps=np.arange(3))
    device.sweep(np.linspace(0, 1, 3))
    assert device.called == [(1e-6, [0, 1, 2]), ([0.0, 0.5, 1.0],)]
    assert all(_all_builtin(call) for call in device.called)


def test_commands_keep_their_signature():
    assert str(Recorder().commands["move"].signature) == (
        "(position: 'float', steps: 'Any' = None) -> 'None'"
    )
    assert Recorder().commands["sweep"].long


def test_the_wire_never_carries_numpy():
    assert _all_builtin(to_wire((np.int64(1), np.float64(2.5))))
    assert _all_builtin(to_wire(Point(np.float64(1.0), np.float64(2.0))))
