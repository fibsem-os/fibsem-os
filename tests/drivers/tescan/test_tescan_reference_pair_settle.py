"""TESCAN pauses before an ion image that directly follows an electron image.

The pause used to live in the shared `acquire.take_reference_images` behind a manufacturer
check (FIB-1118). It is now the driver's, keyed on the previous requested acquisition.

The pause is on the connection, which every beam of one microscope shares. No hardware
or Tescan SDK required.
"""

from typing import List

import pytest

from fibsem.drivers.tescan import connection as tescan
from fibsem.drivers.tescan.connection import TescanConnection
from fibsem.structures.settings import BeamType


@pytest.fixture()
def sleeps(monkeypatch) -> List[float]:
    recorded: List[float] = []
    monkeypatch.setattr(tescan.time, "sleep", recorded.append)
    return recorded


def make_connection() -> TescanConnection:
    return TescanConnection(sdk=None, ip_address="localhost", port=8300)


def test_an_ion_image_after_an_electron_image_waits(sleeps):
    m = make_connection()
    m.settle_after_electron_image(BeamType.ELECTRON)
    m.settle_after_electron_image(BeamType.ION)
    assert sleeps == [tescan.TESCAN_ELECTRON_TO_ION_SETTLE_TIME]


@pytest.mark.parametrize(
    "sequence",
    [
        [BeamType.ION],
        [BeamType.ELECTRON],
        [BeamType.ELECTRON, BeamType.ELECTRON],
        [BeamType.ION, BeamType.ION],
        [BeamType.ION, BeamType.ELECTRON],
    ],
)
def test_no_other_sequence_waits(sleeps, sequence):
    m = make_connection()
    for beam_type in sequence:
        m.settle_after_electron_image(beam_type)
    assert sleeps == []


def test_each_connection_tracks_its_own_last_beam(sleeps):
    a, b = make_connection(), make_connection()
    a.settle_after_electron_image(BeamType.ELECTRON)
    b.settle_after_electron_image(BeamType.ION)
    assert sleeps == []
