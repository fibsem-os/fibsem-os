"""One metadata vocabulary, and a diagnostic for anything outside it.

`FIELD_METADATA_KEYS` is what a dataclass field may declare: a
``Parameter``'s vocabulary (unit, limits in SI, choices, doc, device_parameter,
filepath) and a `Display` for how it is shown. The older flat keys (`tooltip`,
`label`, `minimum` in display units, `items`, ...) are **not** read, and neither
are the older spellings before them (`help`, `units`; FIB-384). What makes that
safe rather than silent is the warning: a field still declaring one is told what
replaced it, instead of simply losing its tooltip.

`field_meta` closes the same hole from the other end (FIB-531): it spells the
vocabulary out as keyword arguments, so a declaration written through it fails at
import instead of rendering blank. The warning is still what serves the plugins
we cannot convert, which keep writing raw dicts.
"""

import inspect
import logging
from dataclasses import dataclass, field

import pytest

import fibsem.structures as fstructures
from fibsem.structures import (
    FIELD_METADATA_KEYS,
    RENAMED_METADATA_KEYS,
    Display,
    RangeLimit,
    field_meta,
    field_specs,
)


def keyword_only_parameters():
    return [
        name
        for name, p in inspect.signature(field_meta).parameters.items()
        if p.kind is inspect.Parameter.KEYWORD_ONLY
    ]


@dataclass
class MetadataProbe:
    """One field per spelling situation the forms have to cope with."""

    superseded: float = field(
        default=1.0, metadata={"tooltip": "old key", "minimum": 0.0, "units": "m"}
    )
    canonical: float = field(
        default=1.0, metadata=field_meta(doc="current key", unit="m")
    )
    neither: float = field(default=1.0)


def test_no_renamed_key_is_still_current():
    for old in RENAMED_METADATA_KEYS:
        assert old not in FIELD_METADATA_KEYS, f"{old!r} is both superseded and current"


def test_a_superseded_key_is_not_resolved():
    """The old keys are gone, not aliased: one vocabulary, not two accepted
    forever. A field still declaring `tooltip` renders without one -- and is warned
    about, which is what makes dropping them safe rather than silent."""
    spec = field_specs(MetadataProbe)["superseded"]

    assert spec.doc == ""
    assert spec.unit is None
    assert spec.limits is None


def test_the_current_keys_are_read():
    spec = field_specs(MetadataProbe)["canonical"]

    assert spec.doc == "current key"
    assert spec.unit == "m"


def test_a_field_declaring_nothing_gets_the_defaults():
    spec = field_specs(MetadataProbe)["neither"]

    assert (spec.doc, spec.unit, spec.limits, spec.filepath) == (
        "",
        None,
        None,
        False,
    )
    assert spec.display == Display()


def test_a_superseded_key_is_told_what_replaced_it(caplog):
    """A generic "no form reads this" line is useless to someone with `tooltip`.

    Out-of-tree configs are the ones that still declare the old keys, and their
    authors cannot see this vocabulary. The warning names the replacement so the
    fix is obvious from the log alone.
    """
    fstructures._warned_metadata_keys.clear()
    with caplog.at_level(logging.WARNING):
        field_specs(MetadataProbe)

    messages = [r.message for r in caplog.records]
    assert any("'tooltip'" in m and "declare doc instead" in m for m in messages)
    assert any("'minimum'" in m and "limits, in SI" in m for m in messages)
    assert any("'units'" in m and "declare unit instead" in m for m in messages)


def test_an_unreadable_metadata_key_is_reported_once(caplog):
    """A mis-keyed field is otherwise completely silent.

    The form renders, the value is right, and the label or suffix is simply
    missing -- which is an afternoon of confusion for a plugin author who has no
    way to discover the vocabulary. Warned once, because form metadata is re-read
    on every rebuild and this would otherwise repeat on every keystroke-driven
    form refresh.
    """

    @dataclass
    class Misspelled:
        speed: float = field(default=1.0, metadata={"dco": "typo"})

    fstructures._warned_metadata_keys.clear()
    with caplog.at_level(logging.WARNING):
        field_specs(Misspelled)
        field_specs(Misspelled)

    warnings = [r for r in caplog.records if "dco" in r.message]
    assert len(warnings) == 1
    assert "which no form reads" in warnings[0].message


def test_a_known_key_is_never_reported(caplog):
    @dataclass
    class Fine:
        a: float = field(default=1.0, metadata={"doc": "t", "unit": "m"})
        b: str = field(default="", metadata={"filepath": True, "display": Display("L")})

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
    """Every built-in task config, pattern and strategy config, read through the
    real vocabulary. Catches a typo in a class no test opens a form for.

    Built-ins only (not `get_tasks()`), so this doesn't pass or fail on whatever
    plugins happen to be installed. Plugin configs are exactly the ones we cannot
    fix from here -- the warning is what serves them.
    """
    fstructures._warned_metadata_keys.clear()
    with caplog.at_level(logging.WARNING):
        for cls in _in_tree_classes():
            field_specs(cls)

    offenders = [
        r.message for r in caplog.records if "declares metadata key" in r.message
    ]
    assert offenders == []


# --- field_meta: the same vocabulary, enforced at import time (FIB-531) -------


def test_the_constructor_accepts_exactly_the_vocabulary():
    """The signature *is* the documentation, so it must not drift from the keys.

    A key in the vocabulary but not the signature cannot be declared through
    `field_meta` at all; one in the signature but not the vocabulary would be
    accepted here and then warned about at render time.
    """
    assert keyword_only_parameters() == list(FIELD_METADATA_KEYS)


def test_every_parameter_reaches_the_result():
    """A parameter added to the signature but missed in the body would accept its
    value and silently drop it, which is precisely the bug class `field_meta`
    exists to prevent."""
    # the typed ones take a real value of their type
    typed = {
        "limits": RangeLimit(0.0, 1.0),
        "display": Display("A"),
        "choices": ("a", "b"),
        "device_parameter": "beam.current",
    }
    for name in keyword_only_parameters():
        marker = typed.get(name, object())
        assert field_meta(**{name: marker}) == {name: marker}, name


@pytest.mark.parametrize("key", ["help", "units", "tooltip", "label", "minimum"])
def test_a_superseded_key_is_rejected_at_import(key):
    """FIB-384's bug is unrepresentable through this door: a raw dict takes
    `tooltip` happily and renders without one; here it is a TypeError, raised
    while the module is being imported."""
    with pytest.raises(TypeError, match=key):
        field_meta(**{key: "old"})


def test_a_misspelled_key_is_rejected_at_import():
    with pytest.raises(TypeError, match="dco"):
        field_meta(dco="typo")


def test_unset_keys_are_omitted():
    assert field_meta(doc="Width") == {"doc": "Width"}


def test_an_explicit_false_is_kept():
    """`filepath=False` means not a path, not "unspecified"."""
    assert field_meta(filepath=False) == {"filepath": False}


def test_the_shared_milling_metadata_declares_only_known_keys():
    """The highest-leverage place to check keys: these helpers declare roughly a
    hundred pattern and strategy fields, so one typo here renders blank in every
    field that uses it."""
    from fibsem.milling import properties

    with_label = ("distance_metadata", "angle_metadata")
    shared = [
        helper("Label", "Doc") if name in with_label else helper()
        for name, helper in vars(properties).items()
        if name.endswith("_metadata") and callable(helper)
    ]
    assert len(shared) >= 7

    for meta in shared:
        unknown = set(meta) - set(FIELD_METADATA_KEYS)
        assert not unknown, unknown
