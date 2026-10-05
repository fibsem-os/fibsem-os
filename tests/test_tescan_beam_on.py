"""TESCAN reports whether a column is on as True or False, not as a status code.

`Beam.GetStatus()` returns a status code. Returned as is, a non-zero code for a
column that is warming up or in transition read as "on", so `turn_beams_on` would
skip turning it on.

No hardware or Tescan SDK required: the microscope is created without __init__.
"""

import enum
import threading
from types import SimpleNamespace

import pytest

from fibsem.microscopes.tescan import TescanMicroscope
from fibsem.structures import BeamType


class FakeStatus(enum.IntEnum):
    BeamOff = 0
    BeamOn = 1
    Transition = 1000


class FakeBeam:
    Status = FakeStatus

    def __init__(self, status: FakeStatus):
        self.status = status

    def GetStatus(self) -> FakeStatus:
        return self.status


def make_microscope(status: FakeStatus) -> TescanMicroscope:
    m = object.__new__(TescanMicroscope)
    m._connection_lock = threading.RLock()
    column = SimpleNamespace(Beam=FakeBeam(status))
    m._get_beam = lambda beam_type: column
    return m


@pytest.mark.parametrize(
    "status, expected",
    [
        (FakeStatus.BeamOn, True),
        (FakeStatus.BeamOff, False),
        (FakeStatus.Transition, False),
    ],
)
@pytest.mark.parametrize("beam_type", [BeamType.ELECTRON, BeamType.ION])
def test_on_is_a_bool(status, expected, beam_type):
    m = make_microscope(status)
    on = m._get("on", beam_type)
    assert on is expected
