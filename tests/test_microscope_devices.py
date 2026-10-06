"""``microscope.devices``: every device a microscope built, by name.

The typed attributes (``beams``, ``stage``, ``chamber_device``, ``manipulator_device``,
``gis_device``, ``fm_devices``) are views of it, so the two can't disagree.
"""

import os
from types import MappingProxyType

import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.structures import BeamType

OFFSET_FM = os.path.join(cfg.CONFIG_PATH, "sim-iflm-configuration.yaml")


def _demo(config_path=None):
    kwargs = {"config_path": config_path} if config_path else {"manufacturer": "Demo"}
    microscope, _ = utils.setup_session(setup_logging=False, **kwargs)
    return microscope


def test_every_device_is_in_the_map_by_name():
    microscope = _demo(OFFSET_FM)

    assert set(microscope.devices) == {
        "fm",
        "camera",
        "light_source",
        "filter_set",
        "objective",
        "electron",
        "ion",
        "stage",
        "chamber",
        "manipulator",
        "gis",
    }


def test_the_typed_attributes_are_the_same_devices():
    microscope = _demo(OFFSET_FM)
    devices = microscope.devices

    assert devices["stage"] is microscope.stage
    assert devices["electron"] is microscope.beams[BeamType.ELECTRON]
    assert devices["ion"] is microscope.beams[BeamType.ION]
    assert devices["chamber"] is microscope.chamber_device
    assert devices["manipulator"] is microscope.manipulator_device
    assert devices["gis"] is microscope.gis_device
    assert devices["fm"] is microscope.fm_devices["fm"]
    assert devices["camera"] is microscope.fm_devices["camera"]
    assert list(microscope.fm_devices) == [
        "fm",
        "camera",
        "light_source",
        "filter_set",
        "objective",
    ]


def test_the_map_is_read_only():
    microscope = _demo()

    with pytest.raises(TypeError):
        microscope.devices["stage"] = None
    with pytest.raises(AttributeError):
        microscope.devices = {}


def test_assigning_a_typed_attribute_changes_the_map():
    microscope = _demo()
    stage = microscope.stage

    microscope.stage = None
    assert "stage" not in microscope.devices

    microscope.stage = stage
    assert microscope.devices["stage"] is stage

    electron = microscope.beams[BeamType.ELECTRON]
    microscope.beams = MappingProxyType({BeamType.ELECTRON: electron})
    assert "ion" not in microscope.devices
    assert dict(microscope.beams) == {BeamType.ELECTRON: electron}


def test_several_gas_injectors_are_each_in_the_map():
    """Thermo builds one injector per port and names the one a caller means."""
    microscope = _demo()
    one, two = object(), object()

    microscope.gis_devices = {"Pt dep": one, "Multichem": two}
    microscope.gis_device = two

    assert microscope.devices["Pt dep"] is one
    assert microscope.devices["Multichem"] is two
    assert dict(microscope.gis_devices) == {"Pt dep": one, "Multichem": two}
    assert microscope.gis_device is two
    assert "gis" not in microscope.devices, "the Demo injector is replaced"


def test_each_microscope_has_its_own_map():
    first, second = _demo(), _demo()
    assert first.devices["stage"] is not second.devices["stage"]
