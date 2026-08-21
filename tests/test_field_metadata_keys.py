"""One metadata vocabulary, and a diagnostic for anything outside it.

`DEFAULT_FIELD_METADATA` is the vocabulary every config form reads. The
AutoLamella task configs used to spell two of those keys differently -- `help`
for `tooltip`, `units` for `unit` -- and the form rendering them read only those
spellings, so a field written for one form rendered half-blank in the other:
right value, no tooltip, no unit suffix, nothing logged.

All in-tree declarations were converted and the old spellings are **not**
accepted (FIB-384). What makes that safe rather than silent is the warning: a
field still declaring a superseded key is told which key replaced it, instead of
simply losing its tooltip.

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
    DEFAULT_FIELD_METADATA,
    RENAMED_METADATA_KEYS,
    field_meta,
    get_fields_with_metadata,
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
        default=1.0, metadata={"help": "old spelling", "units": "m", "scale": 1e6}
    )
    canonical: float = field(
        default=1.0, metadata={"tooltip": "current spelling", "unit": "m"}
    )
    neither: float = field(default=1.0, metadata={"label": "No text at all"})


def test_every_renamed_key_points_at_a_real_one():
    """The diagnostic map cannot advise a key the vocabulary does not define."""
    for old, new in RENAMED_METADATA_KEYS.items():
        assert new in DEFAULT_FIELD_METADATA, f"{old!r} -> unknown key {new!r}"
        assert old not in DEFAULT_FIELD_METADATA, (
            f"{old!r} is both superseded and current"
        )


def test_filepath_is_part_of_the_vocabulary():
    """Both milling forms read it, so it belongs in the canonical dict.

    It was read but never listed, which made it undiscoverable to exactly the
    plugin authors this vocabulary exists for. The default has to stay falsy:
    both forms select the file-picker control with a truthy ``m.get("filepath")``,
    so a truthy default would turn every string field into a file picker.
    """
    assert "filepath" in DEFAULT_FIELD_METADATA
    assert not DEFAULT_FIELD_METADATA["filepath"]
    assert not get_fields_with_metadata(MetadataProbe)["neither"]["filepath"]


def test_a_superseded_key_is_not_resolved():
    """The old spellings are gone, not aliased.

    This is the whole point of the decision: one vocabulary, not two accepted
    forever. A field still declaring `help` renders without a tooltip -- and is
    warned about, which is what makes dropping them safe rather than silent.
    """
    meta = get_fields_with_metadata(MetadataProbe)["superseded"]

    assert meta["tooltip"] is None
    assert meta["unit"] is None


def test_the_current_spellings_are_read():
    meta = get_fields_with_metadata(MetadataProbe)["canonical"]

    assert meta["tooltip"] == "current spelling"
    assert meta["unit"] == "m"


def test_a_field_declaring_neither_spelling_gets_the_default():
    meta = get_fields_with_metadata(MetadataProbe)["neither"]

    assert meta["tooltip"] is None
    assert meta["unit"] is None


def test_a_superseded_key_is_told_what_replaced_it(caplog):
    """A generic "no form reads this" line is useless to someone with `help`.

    Out-of-tree configs are the ones that still declare the old spellings, and
    their authors cannot see this vocabulary. The warning names the replacement
    so the fix is obvious from the log alone.
    """
    fstructures._warned_metadata_keys.clear()
    with caplog.at_level(logging.WARNING):
        get_fields_with_metadata(MetadataProbe)

    messages = [r.message for r in caplog.records]
    assert any("'help'" in m and "renamed to 'tooltip'" in m for m in messages)
    assert any("'units'" in m and "renamed to 'unit'" in m for m in messages)


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
        speed: float = field(default=1.0, metadata={"toolip": "typo"})

    fstructures._warned_metadata_keys.clear()
    with caplog.at_level(logging.WARNING):
        get_fields_with_metadata(Misspelled)
        get_fields_with_metadata(Misspelled)

    warnings = [r for r in caplog.records if "toolip" in r.message]
    assert len(warnings) == 1
    assert "which no form reads" in warnings[0].message


def test_a_known_key_is_never_reported(caplog):
    """Every key in the vocabulary, including ones added late like `filepath`."""

    @dataclass
    class Fine:
        a: float = field(default=1.0, metadata={"tooltip": "t", "unit": "m"})
        b: str = field(default="", metadata={"filepath": True, "label": "L"})

    fstructures._warned_metadata_keys.clear()
    with caplog.at_level(logging.WARNING):
        get_fields_with_metadata(Fine)

    assert [
        r.message for r in caplog.records if "declares metadata key" in r.message
    ] == []


def test_in_tree_configs_declare_no_unreadable_keys(caplog):
    """Every built-in task config, rendered through the real vocabulary.

    Catches a typo in a config that no test opens a form for -- which is most of
    them.

    Deliberately `BUILTIN_TASKS` rather than `get_tasks()`: the latter includes
    whatever plugins happen to be pip-installed, so this would pass or fail on
    the contents of the developer's environment. Plugin configs are exactly the
    ones we cannot fix from here -- the warning is what serves them.
    """
    from fibsem.applications.autolamella.workflows.tasks import BUILTIN_TASKS

    fstructures._warned_metadata_keys.clear()
    with caplog.at_level(logging.WARNING):
        for task in BUILTIN_TASKS.values():
            get_fields_with_metadata(task.config_cls)

    offenders = [
        r.message for r in caplog.records if "declares metadata key" in r.message
    ]
    assert offenders == []


# --- field_meta: the same vocabulary, enforced at import time (FIB-531) -------


def test_the_constructor_accepts_exactly_the_vocabulary():
    """The signature *is* the documentation, so it must not drift from the dict.

    A key in the vocabulary but not the signature cannot be declared through
    `field_meta` at all; one in the signature but not the vocabulary would be
    accepted here and then warned about at render time. Either way the two
    disagree about what a valid key is.
    """
    assert keyword_only_parameters() == list(DEFAULT_FIELD_METADATA)


def test_every_parameter_reaches_the_result():
    """Guards the `locals()` trick that builds the returned dict.

    Spelling the keys out a second time inside the body is what this avoids: a
    parameter added to the signature but missed there would accept its value and
    silently drop it, which is precisely the bug class `field_meta` exists to
    prevent -- and it would be invisible to every other test here.
    """
    marker = object()
    for name in keyword_only_parameters():
        assert field_meta(**{name: marker}) == {name: marker}, name


def test_a_superseded_spelling_is_rejected_at_import():
    """The headline: FIB-384's bug is unrepresentable through this door.

    A raw dict takes `help` happily and renders without a tooltip. Here it is a
    TypeError, raised while the module is being imported -- before any form is
    built and long before anyone squints at a blank label.
    """
    with pytest.raises(TypeError, match="help"):
        field_meta(help="old spelling")
    with pytest.raises(TypeError, match="units"):
        field_meta(units="m")


def test_a_misspelled_key_is_rejected_at_import():
    with pytest.raises(TypeError, match="toolip"):
        field_meta(toolip="typo")


def test_unset_keys_are_omitted():
    """The result has to be the dict you would have written by hand.

    Returning all nineteen keys with None would look equivalent -- the merge in
    `get_fields_with_metadata` fills them anyway -- but it would override
    whatever a base or a struct-level DEFAULT_METADATA had supplied, turning a
    conversion into a behaviour change.
    """
    assert field_meta(label="Width") == {"label": "Width"}


def test_omitting_a_key_renders_the_same_as_declaring_it_None():
    """Why converting a declaration that said `"scale": None` is not a change.

    Dropping an explicitly-None key does alter the raw `field.metadata` -- and
    six pattern and strategy fields lost one that way, via the shared angle
    metadata. It is invisible because every form reads the merged
    `field_metadata`, which refills the key from the vocabulary either way. A
    reader going to `field.metadata` directly with a non-None fallback would be
    able to tell, so this pins the equivalence the conversion relied on.
    """

    @dataclass
    class Omitted:
        angle: float = field(default=0.0, metadata=field_meta(unit="deg"))

    @dataclass
    class ExplicitNone:
        angle: float = field(default=0.0, metadata={"unit": "deg", "scale": None})

    assert "scale" not in Omitted.__dataclass_fields__["angle"].metadata
    assert (
        get_fields_with_metadata(Omitted)["angle"]
        == get_fields_with_metadata(ExplicitNone)["angle"]
    )


def test_an_explicit_false_is_kept():
    """`hidden=False` means hidden=False, not "unspecified".

    Filtering on falsiness rather than None would drop it, and a field could then
    no longer turn off a flag its base had turned on.
    """
    assert field_meta(hidden=False) == {"hidden": False}


def test_it_produces_the_dict_it_replaces():
    """Converting a declaration must not change what the form renders."""

    @dataclass
    class Constructed:
        width: float = field(default=1.0, metadata=field_meta(unit="m", scale=1e6))

    @dataclass
    class Literal:
        width: float = field(default=1.0, metadata={"unit": "m", "scale": 1e6})

    assert (
        get_fields_with_metadata(Constructed)["width"]
        == get_fields_with_metadata(Literal)["width"]
    )


def test_a_base_is_extended_and_overridden():
    """`field_meta(BASE, ...)` has to mean exactly `{**BASE, ...}`.

    Patterns and strategies are overwhelmingly declared by extending a shared
    dict, so a constructor that could not express that would be unusable at most
    of the sites that need it.
    """
    base = {"unit": "m", "scale": 1e6, "label": "Distance"}

    assert field_meta(base, label="Overtilt", minimum=0.1) == {
        "unit": "m",
        "scale": 1e6,
        "label": "Overtilt",
        "minimum": 0.1,
    }


def test_a_base_cannot_launder_an_unreadable_key():
    """Otherwise the base is a hole straight through the check.

    `field_meta(some_raw_dict)` would look like a converted declaration while
    carrying the same typo the raw dict had.
    """
    with pytest.raises(TypeError, match="toolip"):
        field_meta({"toolip": "typo"})


def test_a_base_naming_a_superseded_key_says_what_replaced_it():
    with pytest.raises(TypeError, match="renamed to 'tooltip'"):
        field_meta({"help": "old spelling"})


def test_a_spread_base_rejects_a_keyword_it_already_supplies():
    """The stricter idiom, and the reason to prefer it where nothing overrides.

    `{"type": Point, **DISTANCE}` silently resolves to the spread's `type` --
    that exact collision is live in `patterns2.BasePattern.point`, which declares
    `Point` and renders as a float. Spread as arguments, Python refuses it.
    """
    base = {"unit": "m", "scale": 1e6}

    with pytest.raises(TypeError, match="unit"):
        field_meta(**base, unit="µm")


def test_the_shared_pattern_metadata_declares_only_known_keys():
    """The highest-leverage place to check keys.

    These handful of dicts are spread into roughly a hundred pattern and strategy
    fields, so one typo here renders blank in every field that inherits it.
    Declaring them through `field_meta` is what currently guarantees this, but
    the guarantee is what matters, so this asserts the keys directly.
    """
    from fibsem.milling import properties

    shared = [
        value
        for name, value in vars(properties).items()
        if name.startswith("DEFAULT_") and name.endswith("_METADATA")
    ]
    assert shared, "expected the shared pattern metadata dicts to be importable"

    for meta in shared:
        unknown = set(meta) - set(DEFAULT_FIELD_METADATA)
        assert not unknown, unknown
