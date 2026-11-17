"""A beam's resolution reads as the tuple it is declared as, so what was read can be
written back."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fibsem.structures.settings import BeamType
from tests.fixtures import fake_drivers
from tests.fixtures.fake_drivers import FAKE_DRIVERS


@pytest.mark.parametrize("beam_type", [BeamType.ELECTRON, BeamType.ION])
@pytest.mark.parametrize("driver", sorted(FAKE_DRIVERS))
def test_a_read_resolution_writes_back(driver, beam_type, monkeypatch):
    microscope = fake_drivers.connect(driver, monkeypatch)
    resolution = microscope.beams[beam_type].parameters.get("resolution")
    if resolution is None:
        pytest.skip(f"{driver}'s {beam_type.name} beam has no resolution")
    if driver == "Tescan":  # read from the last image, which held a list
        microscope.beams[beam_type]._cache.resolution = [1536, 1024]
    read = resolution.get_value()
    assert isinstance(read, tuple)
    if resolution.settable:
        assert resolution.set_value(read) == read
        assert resolution.get_value() == read


def test_autoscript_reads_a_tuple():
    from fibsem.drivers.autoscript.devices import AutoscriptBeam

    beam = SimpleNamespace(
        _beam=SimpleNamespace(
            scanning=SimpleNamespace(resolution=SimpleNamespace(value="1536x1024"))
        )
    )
    assert AutoscriptBeam.read_resolution(beam) == (1536, 1024)


def test_odemis_reads_a_tuple():
    from fibsem.drivers.odemis.devices import OdemisBeam

    client = SimpleNamespace(get_resolution=lambda channel: [768, 512])
    beam = SimpleNamespace(_client=client, channel="electron")
    assert OdemisBeam.read_resolution(beam) == (768, 512)
