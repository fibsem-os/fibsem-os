"""Display hints: declared once per device parameter, as a dataclass field declares them."""

from math import pi

import pytest

from fibsem import utils
from fibsem.devices.beam import Beam
from fibsem.devices.core import DeviceParameter
from fibsem.devices.fm import Camera
from fibsem.devices.stage import Stage
from fibsem.structures.params import Display
from fibsem.structures.settings import BeamType


def test_the_beam_declares_its_hints_and_a_bound_parameter_has_them():
    assert Beam.hfw.display.scale == 1e6
    microscope, _ = utils.setup_session(manufacturer="Demo", setup_logging=False)
    beam = microscope.beams[BeamType.ELECTRON]
    assert beam.parameters["hfw"].display is Beam.hfw.display
    assert beam.parameters["scan_rotation"].display.unit == "°"


def test_a_composite_value_has_a_hint_per_field():
    assert set(Stage.position.display) == {"x", "y", "z", "r", "t"}
    assert Stage.position.display["t"].scale == pytest.approx(180 / pi)


def test_a_backend_keeps_the_hint_when_it_redeclares_the_parameter():
    class _Camera(Camera):
        gain = DeviceParameter(float, doc="This camera's gain.")

    assert _Camera.gain.display == Camera.gain.display


def test_a_backend_cannot_show_a_parameter_differently():
    with pytest.raises(TypeError, match="display"):

        class _Camera(Camera):
            gain = DeviceParameter(float, display=Display("Gain", scale=1))


def test_an_explicit_display_unit_wins_over_the_scale_prefix():
    pytest.importorskip("PyQt5")
    from fibsem.ui.widgets.form_builder import parameter_spec

    assert parameter_spec(Beam.scan_rotation).suffix == "°"  # not "mrad"
    assert parameter_spec(Beam.hfw).suffix == "µm"
