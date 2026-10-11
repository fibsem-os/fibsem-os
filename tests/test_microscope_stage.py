"""``microscope.stage`` is the stage device, and ``stage_device`` its earlier name.

The vendor stage objects that used the name on Thermo and Odemis are private
(``_vendor_stage``); their own tests drive them there.
"""

from fibsem import utils
from fibsem.devices.stage import Stage
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import FibsemStagePosition


def _demo():
    microscope, _ = utils.setup_session(manufacturer="Demo", setup_logging=False)
    return microscope


def test_the_stage_is_the_stage_device():
    microscope = _demo()
    assert isinstance(microscope.stage, Stage)
    assert microscope.stage_device is microscope.stage
    assert microscope.get_stage_position() == microscope.stage.position.get_value()


def test_stage_device_is_an_alias_both_ways():
    microscope = _demo()
    device = microscope.stage
    microscope.stage_device = None
    assert microscope.stage is None
    microscope.stage_device = device
    assert microscope.stage is device


def _bare_microscope() -> FibsemMicroscope:
    """A microscope whose backend builds no devices."""
    bare = type("Bare", (FibsemMicroscope,), {})
    bare.__abstractmethods__ = frozenset()
    return bare.__new__(bare)


def test_a_backend_without_a_stage_device_has_none():
    microscope = _bare_microscope()
    assert microscope.stage is None
    assert dict(microscope.devices) == {}


def test_the_stage_keys_route_to_the_stage():
    microscope = _demo()
    microscope.move_stage_absolute(FibsemStagePosition(x=1e-4, y=0.0))
    assert microscope.get("stage_position") == microscope.stage.position.cached
