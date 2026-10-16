"""A field declares a Parameter, and anything else in its metadata is reported.

A dataclass field says what its value means and how a form shows it with
``param(...)``, whose Parameter is the one shape a device parameter uses too. The
Parameter's constructor refuses a misspelled or older key (`tooltip`, `minimum`,
`items`) when the module is imported (FIB-384, FIB-531). What is left is a raw
metadata dict, which a plugin may still write: none of its keys are read, and
each is warned about once, so a field doesn't silently lose its tooltip.
"""

import logging
from dataclasses import dataclass, field, fields

import fibsem.structures as fstructures
from fibsem.structures import (
    PARAMETER_KEY,
    Display,
    Parameter,
    RangeLimit,
    field_specs,
    param,
)


@dataclass
class MetadataProbe:
    """One field per declaration a form has to cope with."""

    raw: float = field(default=1.0, metadata={"doc": "a raw dict", "unit": "m"})
    declared: float = param(1.0, doc="declared", unit="m")
    neither: float = field(default=1.0)


def test_a_declared_parameter_is_read():
    spec = field_specs(MetadataProbe)["declared"]

    assert (spec.name, spec.type, spec.doc, spec.unit) == (
        "declared",
        float,
        "declared",
        "m",
    )


def test_a_raw_dict_is_not_read():
    """One way to declare a field, not two accepted forever."""
    spec = field_specs(MetadataProbe)["raw"]

    assert (spec.doc, spec.unit) == ("", None)


def test_a_field_declaring_nothing_gets_the_defaults():
    spec = field_specs(MetadataProbe)["neither"]

    assert (spec.doc, spec.unit, spec.limits, spec.filepath) == (
        "",
        None,
        None,
        False,
    )
    assert spec.display == Display()


def test_a_raw_key_is_told_how_to_declare_it(caplog):
    """Out-of-tree configs are the ones that still write raw dicts, and their
    authors cannot see this vocabulary: the warning says what to write instead."""
    fstructures._warned_metadata_keys.clear()
    with caplog.at_level(logging.WARNING):
        field_specs(MetadataProbe)

    messages = [
        r.message for r in caplog.records if "declares metadata key" in r.message
    ]
    assert len(messages) == 2
    assert all("param(default, unit=..., doc=...)" in m for m in messages)
    assert any("'doc'" in m for m in messages)


def test_an_unreadable_metadata_key_is_reported_once(caplog):
    """Warned once, because form metadata is re-read on every rebuild and this
    would otherwise repeat on every form refresh."""

    @dataclass
    class Misspelled:
        speed: float = field(default=1.0, metadata={"dco": "typo"})

    fstructures._warned_metadata_keys.clear()
    with caplog.at_level(logging.WARNING):
        field_specs(Misspelled)
        field_specs(Misspelled)

    warnings = [r for r in caplog.records if "dco" in r.message]
    assert len(warnings) == 1


def test_a_declared_parameter_is_never_reported(caplog):
    @dataclass
    class Fine:
        a: float = param(1.0, doc="t", unit="m")
        b: str = param("", filepath=True, display=Display("L"))

    fstructures._warned_metadata_keys.clear()
    with caplog.at_level(logging.WARNING):
        field_specs(Fine)

    assert [
        r.message for r in caplog.records if "declares metadata key" in r.message
    ] == []


def _in_tree_classes():
    from fibsem.applications.autolamella.workflows.tasks import BUILTIN_TASKS
    from fibsem.milling.patterning import get_pattern, get_pattern_names
    from fibsem.milling.strategy import get_strategies

    classes = [task.config_cls for task in BUILTIN_TASKS.values()]
    classes += [type(get_pattern(name)) for name in get_pattern_names()]
    classes += [strategy.config_class for strategy in get_strategies().values()]
    return classes


def test_in_tree_classes_declare_no_unreadable_keys(caplog):
    """Every built-in task config, pattern and strategy config. Catches a raw
    dict in a class no test opens a form for.

    Built-ins only (not `get_tasks()`), so this doesn't pass or fail on whatever
    plugins happen to be installed.
    """
    fstructures._warned_metadata_keys.clear()
    with caplog.at_level(logging.WARNING):
        for cls in _in_tree_classes():
            field_specs(cls)

    offenders = [
        r.message for r in caplog.records if "declares metadata key" in r.message
    ]
    assert offenders == []


def test_param_keeps_the_parameter_under_one_key():
    (declared,) = fields(_Declared)

    assert list(declared.metadata) == [PARAMETER_KEY]
    assert declared.metadata[PARAMETER_KEY] == Parameter(
        doc="Width", limits=RangeLimit(0.0, 1.0)
    )
    assert declared.default == 2.0


@dataclass
class _Declared:
    width: float = param(2.0, doc="Width", limits=RangeLimit(0.0, 1.0))


def test_param_keeps_choices_as_a_tuple():
    """A list of choices would make two equal declarations compare unequal."""

    @dataclass
    class Choosing:
        mode: str = param("a", choices=["a", "b"])

    assert field_specs(Choosing)["mode"].choices == ("a", "b")


def test_param_overrides_a_shared_parameter():
    shared = Parameter(unit="m", doc="Shared", display=Display("Width"))

    @dataclass
    class Overriding:
        width: float = param(1.0, shared, doc="Own")
        items: list = param(parameter=shared, default_factory=list)

    specs = field_specs(Overriding)
    assert (specs["width"].unit, specs["width"].doc) == ("m", "Own")
    assert specs["items"].doc == "Shared"
    assert Overriding().items == []


def test_the_shared_milling_parameters_are_parameters():
    """These helpers declare roughly a hundred pattern and strategy fields, so one
    wrong declaration here renders blank in every field that uses it."""
    from fibsem.milling import properties

    with_label = ("distance_param", "angle_param")
    shared = [
        helper("Label", "Doc") if name in with_label else helper()
        for name, helper in vars(properties).items()
        if name.endswith("_param") and callable(helper)
    ]
    assert len(shared) >= 7
    assert all(isinstance(declared, Parameter) for declared in shared)
