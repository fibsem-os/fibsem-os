"""A dataclass field says what a device Parameter says (fibsem.devices.core).

Its meaning -- SI unit, limits in SI, choices, a doc -- and its look, a ``Display``.
The forms still read the flat keys (label, tooltip, minimum and maximum in display
units, items), so ``get_fields_with_metadata`` fills those from the new ones.
"""

from dataclasses import dataclass, field
from typing import Optional

import pytest

from fibsem.display import Display
from fibsem.structures import (
    FibsemMillingSettings,
    RangeLimit,
    field_meta,
    get_fields_with_metadata,
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
    width = get_fields_with_metadata(Probe)["width"]
    assert width["minimum"] == pytest.approx(1.0)
    assert width["maximum"] == pytest.approx(1000.0)
    assert (width["label"], width["scale"], width["step"], width["decimals"]) == (
        "Width",
        1e6,
        0.1,
        2,
    )
    assert width["unit"] == "m"
    assert width["tooltip"] == "The width."


def test_choices_doc_and_hidden_fill_the_flat_keys():
    metadata = get_fields_with_metadata(Probe)
    assert metadata["mode"]["items"] == ["a", "b"]
    assert metadata["count"]["tooltip"] == "How many."
    assert metadata["shapes"]["hidden"] is True


def test_type_comes_from_the_annotation():
    metadata = get_fields_with_metadata(Probe)
    assert metadata["width"]["type"] is float
    assert metadata["count"]["type"] is int  # Optional unwrapped


def test_a_bound_parameter_without_a_range_takes_the_microscope_choices():
    """The current is picked from what the instrument offers; the field of view
    is a number within the beam's range, so it has no items."""
    metadata = get_fields_with_metadata(Probe)
    assert metadata["current"]["items"] == "dynamic"
    assert metadata["hfw"]["items"] is None
    # and it is shown as the beam shows it
    assert metadata["hfw"]["scale"] == 1e6


@pytest.mark.parametrize(
    "keys",
    [
        {"doc": "a", "tooltip": "b"},
        {"choices": [1], "items": [1]},
        {"limits": RangeLimit(0, 1), "minimum": 0},
        {"display": Display("A"), "label": "A"},
    ],
)
def test_saying_one_thing_both_ways_is_refused(keys):
    with pytest.raises(TypeError, match="give it once"):
        field_meta(**keys)


def test_limits_and_display_are_typed():
    with pytest.raises(TypeError, match="RangeLimit"):
        field_meta(limits=(0, 1))
    with pytest.raises(TypeError, match="Display"):
        field_meta(display={"label": "A"})


def test_no_gui_default_in_the_milling_settings():
    """``"default": 150.0`` (µm) on hfw read as the field's default to anything
    describing the dataclass from its fields (pydantic), not 150e-6 m."""
    for name, metadata in get_fields_with_metadata(FibsemMillingSettings).items():
        assert "default" not in metadata, name
    hfw = get_fields_with_metadata(FibsemMillingSettings)["hfw"]
    assert hfw["minimum"] == pytest.approx(20.0)
    assert hfw["maximum"] == pytest.approx(950.0)
