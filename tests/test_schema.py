"""JSON schemas from what a value declares (fibsem.schema).

An agent has no copy of the classes, so a command's arguments reach it as JSON
Schema: the dataclasses they hold under ``$defs``, each field with the SI unit,
limits, choices and doc the GUI forms read from the same ``Parameter``.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from fibsem.schema import arguments_schema, parameter_schema, value_schema
from fibsem.structures import (
    Display,
    FibsemMillingSettings,
    Parameter,
    Point,
    RangeLimit,
    param,
)


class Mode(Enum):
    FAST = "fast"
    SLOW = "slow"


@dataclass
class Recipe:
    """A probe recipe."""

    width: float = param(
        10e-6,
        unit="m",
        limits=RangeLimit(1e-6, 1e-3),
        doc="The width.",
        display=Display("Width", scale=1e6),
    )
    mode: Mode = Mode.FAST
    centre: Point = field(default_factory=Point)
    repeats: Optional[int] = None


class Probe:
    def mill(self, recipe: Recipe, depth: float = 1e-6, *, tags: List[str]) -> None:
        """Mill it."""

    def move(self, offset: Tuple[float, float]) -> None:
        """Move it."""


def test_a_field_carries_its_unit_limits_doc_and_display():
    recipe = value_schema(Recipe, {})

    width = recipe["properties"]["width"]
    assert width == {
        "type": "number",
        "default": 10e-6,
        "description": "The width.",
        "unit": "m",
        "minimum": 1e-6,
        "maximum": 1e-3,
        "display": {"label": "Width", "scale": 1e6},
    }
    assert recipe["description"] == "A probe recipe."


def test_enums_optionals_and_nested_dataclasses():
    defs = {}
    properties = value_schema(Recipe, defs)["properties"]

    assert properties["mode"] == {"$ref": "#/$defs/Mode", "default": "fast"}
    assert defs["Mode"]["enum"] == ["fast", "slow"]
    assert properties["repeats"] == {
        "anyOf": [{"type": "integer"}, {"type": "null"}],
        "default": None,
    }
    assert set(properties["centre"]["properties"]) >= {"x", "y"}


def test_the_schema_is_what_the_codec_reads():
    """Saved keys, tagged classes and fields never read, as `fibsem.codec` has them."""
    from fibsem.milling.base import FibsemMillingStage
    from fibsem.structures import FibsemImageMetadata

    defs = {}
    stage = value_schema(FibsemMillingStage, defs)
    assert "reference_image" not in stage["properties"], "never read, so not sent"
    variants = defs["BasePattern"]["oneOf"]
    rectangle = next(
        v for v in variants if v["properties"]["name"] == {"const": "Rectangle"}
    )
    assert "width" in rectangle["properties"], "each pattern its own fields"
    metadata = value_schema(FibsemImageMetadata, {})
    assert "image" in metadata["properties"], "saved under its key"
    assert "image_settings" not in metadata["properties"]


def test_a_command_lists_its_arguments_with_their_defs():
    schema = arguments_schema(Probe.mill)

    assert set(schema["properties"]) == {"recipe", "depth", "tags"}
    assert schema["required"] == ["recipe", "tags"]
    assert schema["properties"]["depth"] == {"type": "number", "default": 1e-6}
    assert schema["properties"]["tags"] == {
        "type": "array",
        "items": {"type": "string"},
    }
    assert schema["properties"]["recipe"]["title"] == "Recipe"
    assert "Mode" in schema["$defs"]


def test_a_fixed_tuple_is_an_array_of_that_length():
    offset = arguments_schema(Probe.move)["properties"]["offset"]
    assert offset["prefixItems"] == [{"type": "number"}, {"type": "number"}]
    assert (offset["minItems"], offset["maxItems"]) == (2, 2)


def test_a_bound_field_names_the_device_parameter_for_its_live_limits():
    """The milling current's limits are the instrument's, so the schema points at
    the beam parameter (list_devices has its live limits) rather than inventing
    any; its unit comes from the beam."""
    settings = value_schema(FibsemMillingSettings, {})
    current = settings["properties"]["milling_current"]

    assert current["device_parameter"] == "beam.current"
    assert current["unit"] == "A"
    assert "minimum" not in current


def test_a_device_parameter_reads_the_same_way():
    from fibsem.devices.beam import Beam

    schema = parameter_schema(Beam.hfw, {})
    assert schema["type"] == "number"
    assert schema["unit"] == "m"


def test_a_parameter_with_choices_is_an_enum():
    schema = parameter_schema(Parameter(type=str, choices=("a", "b")), {})
    assert schema == {"type": "string", "enum": ["a", "b"]}


def test_a_field_nothing_reads_says_so():
    from fibsem.structures import MillingAlignment

    properties = value_schema(MillingAlignment, {})["properties"]
    assert properties["interval"]["not_used"]
    assert "not_used" not in properties["enabled"]
