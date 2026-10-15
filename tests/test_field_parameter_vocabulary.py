"""A dataclass field says what a device Parameter says (fibsem.devices.core).

Its meaning -- SI unit, limits in SI, choices, a doc -- and its look, a ``Display``.
``field_specs`` reads them off a dataclass as one ``FieldSpec`` per field, which is
what every form reads.
"""

from dataclasses import dataclass, field
from typing import Optional

import pytest

from fibsem.display import Display
from fibsem.structures import (
    FibsemMillingSettings,
    RangeLimit,
    field_meta,
    field_specs,
)


@dataclass
class Probe:
    width: float = field(
        default=10e-6,
        metadata=field_meta(
            unit="m",
            limits=RangeLimit(1e-6, 1e-3),
            doc="The width.",
            display=Display("Width", scale=1e6, step=0.1, decimals=2),
        ),
    )
    mode: str = field(
        default="a", metadata=field_meta(choices=["a", "b"], display=Display("Mode"))
    )
    count: Optional[int] = field(default=None, metadata=field_meta(doc="How many."))
    shapes: list = field(
        default_factory=list, metadata=field_meta(display=Display(hidden=True))
    )
    current: float = field(
        default=1e-9, metadata=field_meta(microscope_parameter="current")
    )
    hfw: float = field(default=150e-6, metadata=field_meta(microscope_parameter="hfw"))


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


@pytest.mark.parametrize("key", ["tooltip", "label", "minimum", "items", "type"])
def test_the_older_keys_are_gone(key):
    with pytest.raises(TypeError):
        field_meta(**{key: 1})


def test_limits_and_display_are_typed():
    with pytest.raises(TypeError, match="RangeLimit"):
        field_meta(limits=(0, 1))
    with pytest.raises(TypeError, match="Display"):
        field_meta(display={"label": "A"})


def test_no_gui_default_in_the_milling_settings():
    """``"default": 150.0`` (µm) on hfw read as the field's default to anything
    describing the dataclass from its fields (pydantic), not 150e-6 m."""
    hfw = field_specs(FibsemMillingSettings)["hfw"]
    assert hfw.limits == RangeLimit(20e-6, 950e-6)
    assert hfw.shown_range == pytest.approx((20.0, 950.0))
