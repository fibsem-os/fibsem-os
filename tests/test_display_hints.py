"""Display hints: declared once per device parameter, in field_meta's vocabulary."""

from math import pi

import pytest

from fibsem import utils
from fibsem.devices.beam import Beam
from fibsem.devices.core import Parameter
from fibsem.devices.display import Display
from fibsem.devices.fm import Camera
from fibsem.devices.stage import Stage
from fibsem.structures import BeamType


def test_a_hint_is_field_metadata():
    hint = Display("Field of View", scale=1e6, step=50.0, decimals=1)
    assert hint.as_field_metadata("m") == {
        "label": "Field of View",
        "scale": 1e6,
        "unit": "m",
        "step": 50.0,
        "decimals": 1,
    }
    assert Display(unit="%", advanced=True).as_field_metadata() == {
        "display_unit": "%",
        "advanced": True,
    }


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
        gain = Parameter(float, doc="This camera's gain.")

    assert _Camera.gain.display == Camera.gain.display


def test_a_backend_cannot_show_a_parameter_differently():
    with pytest.raises(TypeError, match="display"):

        class _Camera(Camera):
            gain = Parameter(float, display=Display("Gain", scale=1))


def test_an_explicit_display_unit_wins_over_the_scale_prefix():
    pytest.importorskip("PyQt5")
    from fibsem.ui.widgets.form_builder import display_suffix

    rotation = Beam.scan_rotation.display.as_field_metadata("rad")
    assert display_suffix(rotation) == "°"  # the prefix rule would give "mrad"
    assert display_suffix(Beam.hfw.display.as_field_metadata("m")) == "µm"
