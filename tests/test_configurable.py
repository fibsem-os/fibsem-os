"""A Configurable dataclass reads its Parameters with ``.params`` and describes
itself with ``describe()``; its values stay plain."""

import dataclasses
from dataclasses import dataclass

import pytest

from fibsem.structures.params import (
    Configurable,
    RangeLimit,
    distance_param,
    field_specs,
    param,
)
from fibsem.structures.settings import ImageSettings, Point


@dataclass
class Probe(Configurable):
    width: float = param(10e-6, distance_param("Width", "The width."))
    count: int = param(3, limits=RangeLimit(1, 9), doc="How many.")
    point: Point = param(
        parameter=distance_param("Point", "Where."), default_factory=Point
    )


def test_values_stay_plain():
    probe = Probe()

    assert probe.width == 10e-6
    assert dataclasses.asdict(probe)["count"] == 3


def test_params_by_attribute_name_and_order():
    assert Probe.params.width.limits == distance_param("W", "d").limits
    assert Probe().params["count"].limits == RangeLimit(1, 9)
    assert list(Probe.params) == ["width", "count", "point"]
    assert dict(Probe.params.items()) == field_specs(Probe)


def test_params_is_not_a_field():
    assert "params" not in {f.name for f in dataclasses.fields(Probe)}
    with pytest.raises(AttributeError, match="no field"):
        Probe.params.missing


def test_describe_is_a_json_schema():
    schema = Probe.describe()

    assert schema["title"] == "Probe"
    assert schema["properties"]["width"]["unit"] == "m"
    assert schema["properties"]["count"]["maximum"] == 9
    assert schema["properties"]["point"]["title"] == "Point"


def test_a_shared_struct_is_configurable():
    assert ImageSettings.params.hfw.device_parameter == "beam.hfw"


def _configurable_classes():
    from fibsem.applications.autolamella.workflows.tasks import BUILTIN_TASKS
    from fibsem.milling.patterning import get_pattern, get_pattern_names
    from fibsem.milling.strategy import get_strategies

    classes = [task.config_cls for task in BUILTIN_TASKS.values()]
    classes += [type(get_pattern(name)) for name in get_pattern_names()]
    classes += [strategy.config_class for strategy in get_strategies().values()]
    return classes


def test_no_built_in_config_has_a_field_named_params():
    """A field named ``params`` would hide the class's Parameters."""
    for cls in _configurable_classes():
        assert issubclass(cls, Configurable), cls
        assert "params" not in {f.name for f in dataclasses.fields(cls)}, cls
