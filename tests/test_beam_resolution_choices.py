"""A beam's resolution choices are what its scan can be set to, and its dwell time
and field of view have limits where the driver knows them.

Thermo's, from the scan, are in ``tests/test_autoscript_beam.py``.
"""

import os

import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.devices.beam import STANDARD_RESOLUTIONS
from fibsem.structures import BeamType

E, I = BeamType.ELECTRON, BeamType.ION


def _demo():
    microscope, _ = utils.setup_session(manufacturer="Demo", setup_logging=False)
    return microscope


@pytest.mark.parametrize("beam_type", [E, I])
def test_demo_offers_the_standard_resolutions_and_its_ranges(beam_type):
    beam = _demo().beams[beam_type]
    assert list(beam.resolution.choices) == list(STANDARD_RESOLUTIONS)
    assert beam.dwell_time.limits is not None
    assert beam.hfw.limits is not None


def test_the_old_wrapper_still_writes_without_checking():
    """``set_resolution`` writes as the old key did; only the device API checks."""
    microscope = _demo()
    microscope.set_resolution((1024, 1024), E)
    assert tuple(microscope.get_resolution(E)) == (1024, 1024)


def test_a_resolution_off_the_list_is_refused():
    beam = _demo().beams[E]
    with pytest.raises(ValueError, match="1024"):
        beam.resolution.set_value((1024, 1024))


def test_tescan_offers_the_standard_resolutions_read_only(monkeypatch):
    """Tescan's scan resolution is not settable on its own: the choices are listed
    for acquisition, and the parameter stays read-only."""
    from tests.fixtures.tescan_sdk import connect

    system = utils.load_microscope_configuration(
        os.path.join(cfg.CONFIG_PATH, "tescan-configuration.yaml")
    ).system
    microscope, _ = connect(monkeypatch, system)
    for beam in microscope.beams.values():
        assert list(beam.resolution.choices) == list(STANDARD_RESOLUTIONS)
        assert not beam.resolution.settable
