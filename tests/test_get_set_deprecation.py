"""The string-key ``microscope.get``/``set`` warns when called from outside the
microscope classes, and the named wrappers built on it do not."""

import warnings

import pytest

from fibsem import utils
from fibsem.structures import BeamType


@pytest.fixture(scope="module")
def microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo")
    yield microscope
    microscope.disconnect()


def test_get_warns_and_names_the_beam_parameter(microscope):
    with pytest.warns(DeprecationWarning) as record:
        microscope.get("current", BeamType.ION)
    message = str(record[0].message)
    assert message.startswith('microscope.get("current", ...) is deprecated')
    assert 'microscope.beams[BeamType.ION].parameters["current"]' in message
    # the warning points at the caller, not at fibsem/microscope.py
    assert record[0].filename == __file__


def test_set_warns_and_names_the_stage_parameter(microscope):
    with pytest.warns(DeprecationWarning, match=r'microscope\.get\("stage_position"'):
        microscope.get("stage_position")
    hfw = microscope.get_field_of_view(BeamType.ELECTRON)
    with pytest.warns(DeprecationWarning, match=r'microscope\.set\("hfw"'):
        microscope.set("hfw", hfw, BeamType.ELECTRON)


def test_named_wrappers_do_not_warn(microscope):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        hfw = microscope.get_field_of_view(BeamType.ELECTRON)
        microscope.set_field_of_view(hfw, BeamType.ELECTRON)
        microscope.get_beam_current(BeamType.ION)
        microscope.get_stage_position()
        microscope.get_microscope_state()
