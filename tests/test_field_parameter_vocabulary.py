"""A dataclass field and a device parameter are the same ``Parameter``.

Its meaning -- SI unit, limits in SI, choices, a doc -- and its look, a ``Display``.
``field_specs`` reads them off a dataclass as one ``Parameter`` per field, which is
what every form reads.
"""

from dataclasses import dataclass, field
from typing import Optional

import pytest

from fibsem.structures import (
    Display,
    FibsemMillingSettings,
    Parameter,
    RangeLimit,
    field_specs,
)


@dataclass
class Probe:
    width: float = field(
        default=10e-6,
        metadata=Parameter(
            unit="m",
            limits=RangeLimit(1e-6, 1e-3),
            doc="The width.",
            display=Display("Width", scale=1e6, step=0.1, decimals=2),
        ).meta(),
    )
    mode: str = field(
        default="a",
        metadata=Parameter(choices=["a", "b"], display=Display("Mode")).meta(),
    )
    count: Optional[int] = field(
        default=None, metadata=Parameter(doc="How many.").meta()
    )
    shapes: list = field(
        default_factory=list, metadata=Parameter(display=Display(hidden=True)).meta()
    )
    current: float = field(
        default=1e-9, metadata=Parameter(device_parameter="beam.current").meta()
    )
    hfw: float = field(
        default=150e-6, metadata=Parameter(device_parameter="beam.hfw").meta()
    )


def test_si_limits_are_shown_through_the_scale():
    width = field_specs(Probe)["width"]
    assert width.limits == RangeLimit(1e-6, 1e-3)
    assert width.shown_range == pytest.approx((1.0, 1000.0))
    assert (width.label, width.scale, width.display.step, width.display.decimals) == (
        "Width",
        1e6,
        0.1,
        2,
    )
    assert (width.unit, width.suffix) == ("m", "µm")
    assert width.doc == "The width."


def test_choices_doc_and_hidden():
    specs = field_specs(Probe)
    assert specs["mode"].choices == ("a", "b")
    assert specs["count"].doc == "How many."
    assert specs["shapes"].hidden is True


def test_type_is_the_annotation():
    specs = field_specs(Probe)
    assert specs["width"].type is float
    assert specs["count"].type == Optional[int]


def test_a_bound_parameter_without_a_range_takes_the_microscope_choices():
    """The current is picked from what the instrument offers; the field of view
    is a number within the beam's range."""
    specs = field_specs(Probe)
    assert specs["current"].instrument_choices
    assert not specs["hfw"].instrument_choices
    # and it is shown as the beam shows it
    assert (specs["hfw"].scale, specs["hfw"].unit) == (1e6, "m")


def test_an_instrument_report_replaces_the_limits_and_choices():
    from fibsem.devices.core import ParameterMetadata

    width = field_specs(Probe)["width"]
    reported = width.with_reported(ParameterMetadata(limits=RangeLimit(2e-6, 5e-6)))
    assert reported.shown_range == pytest.approx((2.0, 5.0))
    # an unbounded report keeps the declared limits
    unbounded = ParameterMetadata(limits=RangeLimit(float("-inf"), float("inf")))
    assert width.with_reported(unbounded).limits == width.limits


@pytest.mark.parametrize("key", ["tooltip", "label", "minimum", "items"])
def test_the_older_keys_are_gone(key):
    with pytest.raises(TypeError):
        Parameter(**{key: 1}).meta()


@pytest.mark.parametrize("key", ["name", "type"])
def test_a_field_gives_its_own_name_and_type(key):
    with pytest.raises(ValueError, match="from the field"):
        Parameter(**{key: "x" if key == "name" else float}).meta()


def test_limits_and_display_are_typed():
    with pytest.raises(TypeError, match="RangeLimit"):
        Parameter(limits=(0, 1)).meta()
    with pytest.raises(TypeError, match="Display"):
        Parameter(display={"label": "A"}).meta()


def test_no_gui_default_in_the_milling_settings():
    """``"default": 150.0`` (µm) on hfw read as the field's default to anything
    describing the dataclass from its fields (pydantic), not 150e-6 m."""
    hfw = field_specs(FibsemMillingSettings)["hfw"]
    assert hfw.limits == RangeLimit(20e-6, 950e-6)
    assert hfw.shown_range == pytest.approx((20.0, 950.0))


# --- one Parameter for fields and devices -------------------------------------


def test_a_device_parameter_is_a_parameter():
    """What a form or a description reads off a device parameter is what it reads
    off a field: the device class only adds the binding."""
    from fibsem.devices.beam import Beam
    from fibsem.structures import Parameter

    assert isinstance(Beam.current, Parameter)
    assert field_specs(Probe)["current"].unit == Beam.current.unit


def test_a_device_declaring_a_plain_parameter_is_refused():
    """A plain Parameter on a device would bind nothing and read as absent."""
    from fibsem.devices.core import Device
    from fibsem.structures import Parameter

    with pytest.raises(TypeError, match="DeviceParameter"):

        class Gauge(Device):
            pressure = Parameter(float, unit="Pa")


def test_a_device_parameter_names_its_device():
    with pytest.raises(ValueError, match="beam.current"):
        Parameter(device_parameter="current").meta()


def test_a_bound_field_reports_its_device_parameter_name():
    spec = field_specs(Probe)["current"]
    assert (spec.device_parameter, spec.bound_name) == ("beam.current", "current")
