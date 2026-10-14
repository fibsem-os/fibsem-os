"""What a device says about where it came from and how to show it.

Each device built from a configuration entry records the driver that built it
(`Device.driver`), and its description carries that and each parameter's display
hint, so a client with no copy of the class (an agent) reads both. A plugin imports
everything it needs from `fibsem.devices`.
"""

import pytest

from fibsem import utils
from fibsem.devices import Device, Display, Parameter
from fibsem.devices.display import display_to_dict
from fibsem.structures import BeamType


@pytest.fixture(scope="module")
def microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo", setup_logging=False)
    yield microscope
    microscope.disconnect()


def test_each_built_device_names_the_driver_that_built_it(microscope):
    assert microscope.beams[BeamType.ELECTRON].driver == "Demo"
    assert microscope.devices["stage"].driver == "Demo"


def test_a_device_built_by_hand_names_no_driver():
    assert Device("knife").driver is None


def test_a_description_carries_each_parameters_display_hint(microscope):
    described = microscope.beams[BeamType.ELECTRON].describe()
    hfw = microscope.beams[BeamType.ELECTRON].hfw.display
    assert described["hfw"]["display"] == display_to_dict(hfw)
    assert "label" in described["hfw"]["display"]


def test_a_parameter_without_a_hint_has_no_display_key():
    class Knife(Device):
        angle = Parameter(float, unit="rad")

        def read_angle(self):
            return 0.0

    assert "display" not in Knife("knife").connect().describe()["angle"]


def test_a_composite_hint_is_one_dict_per_field():
    hint = {"x": Display(label="X", scale=1e6), "y": Display(advanced=True)}
    assert display_to_dict(hint) == {
        "x": {"label": "X", "scale": 1e6},
        "y": {"advanced": True},
    }


def test_the_device_server_describes_the_driver_and_the_hints(microscope):
    pytest.importorskip("fastapi")
    from fibsem.server.devices import describe_device

    described = describe_device(microscope.beams[BeamType.ELECTRON])
    assert described["driver"] == "Demo"
    assert "display" in described["parameters"]["hfw"]


def test_a_plugin_imports_what_it_needs_from_fibsem_devices():
    import fibsem.devices as devices

    for name in ("Scanner", "Role", "Display", "query", "command", "Parameter"):
        assert name in devices.__all__ and hasattr(devices, name)
