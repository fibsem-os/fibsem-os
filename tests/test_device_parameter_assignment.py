"""Assigning to a device's parameter is refused: it would only hide the parameter
on that object and write nothing to the device."""

import pytest

from fibsem.devices.entries import disconnect_devices
from fibsem.server.devices import configured_devices, served_fm_settings


def test_assigning_a_parameter_is_refused_and_says_how_to_write_it():
    built, cleanups = configured_devices(served_fm_settings("fm"))
    try:
        light = built["light_source"]
        with pytest.raises(AttributeError, match=r"power\.set_value"):
            light.power = 0.2
        light.power.set_value(0.2)
        assert light.power.get_value() == pytest.approx(0.2)
    finally:
        disconnect_devices(built, cleanups)
