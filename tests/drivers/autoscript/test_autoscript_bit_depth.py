"""The AutoScript beam's bit depth, whichever shape xT answers it in.

Some xT versions answer ``scanning.bit_depth`` as a setting with ``value``, others
as a plain int. Checked against fakes of both shapes, not on hardware.
"""

from types import SimpleNamespace

import pytest

from fibsem.drivers.autoscript.devices import AutoscriptBeam


class _Setting:
    def __init__(self, value: int) -> None:
        self.value = value


def _beam(bit_depth) -> AutoscriptBeam:
    """An AutoscriptBeam over a fake vendor beam, built without a connection."""
    vendor = SimpleNamespace(scanning=SimpleNamespace(bit_depth=bit_depth))

    class _FakeBeam(AutoscriptBeam):
        _beam = property(lambda self: vendor)

    return object.__new__(_FakeBeam)


@pytest.mark.parametrize("bit_depth", [_Setting(8), 8], ids=["setting", "plain-int"])
def test_bit_depth_reads_either_shape(bit_depth):
    assert _beam(bit_depth).read_bit_depth() == 8


def test_bit_depth_writes_a_setting():
    beam = _beam(_Setting(8))
    beam.write_bit_depth(16)
    assert beam._beam.scanning.bit_depth.value == 16


def test_bit_depth_writes_a_plain_int():
    beam = _beam(8)
    beam.write_bit_depth(16)
    assert beam._beam.scanning.bit_depth == 16
