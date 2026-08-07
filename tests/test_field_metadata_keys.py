"""One metadata vocabulary, two spellings accepted.

`DEFAULT_FIELD_METADATA` is the canonical set, and the pattern, strategy and
milling-settings forms all read it. The AutoLamella task-parameters form grew its
own spellings for two of those keys -- `help` for `tooltip`, `units` for `unit` --
and reads only those, so a field written for one form renders half-blank in the
other: right value, no tooltip, no unit suffix, nothing logged.

Both spellings are now accepted everywhere (FIB-384). Deliberately *not* a
rename-and-drop: installed out-of-tree plugins declare `help` and `units` between
them dozens of times, and dropping the aliases would silently strip their
tooltips and suffixes.
"""

import logging
from dataclasses import dataclass, field
from typing import ClassVar

import pytest

from fibsem.structures import (
    DEFAULT_FIELD_METADATA,
    METADATA_KEY_ALIASES,
    get_fields_with_metadata,
    metadata_value,
)
import fibsem.structures as fstructures


@dataclass
class MetadataProbe:
    """One field per spelling situation the forms have to cope with."""

    task_dialect: float = field(
        default=1.0, metadata={"help": "task spelling", "units": "m", "scale": 1e6}
    )
    canonical: float = field(
        default=1.0, metadata={"tooltip": "canonical spelling", "unit": "m"}
    )
    both_spellings: float = field(
        default=1.0, metadata={"tooltip": "canonical wins", "help": "alias loses"}
    )
    neither: float = field(default=1.0, metadata={"label": "No text at all"})


def test_every_alias_points_at_a_canonical_key():
    """The alias map cannot name a key the vocabulary does not define."""
    for alias, canonical in METADATA_KEY_ALIASES.items():
        assert canonical in DEFAULT_FIELD_METADATA, f"{alias!r} -> unknown key {canonical!r}"
        assert alias not in DEFAULT_FIELD_METADATA, f"{alias!r} is both an alias and canonical"


def test_the_task_dialect_fills_the_canonical_keys():
    """A config written with `help`/`units` renders in the canonical-reading forms."""
    meta = get_fields_with_metadata(MetadataProbe)["task_dialect"]

    assert meta["tooltip"] == "task spelling"
    assert meta["unit"] == "m"


def test_the_aliases_survive_for_readers_that_still_use_them():
    """Normalising must not remove the original keys.

    The task-parameters form reads a field's raw metadata in places, and existing
    plugins declare the alias spellings. Filling the canonical key is additive.
    """
    meta = get_fields_with_metadata(MetadataProbe)["task_dialect"]

    assert meta["help"] == "task spelling"
    assert meta["units"] == "m"


def test_canonical_wins_when_a_field_declares_both():
    """Otherwise the winner would depend on dict iteration order."""
    meta = get_fields_with_metadata(MetadataProbe)["both_spellings"]

    assert meta["tooltip"] == "canonical wins"


def test_a_field_declaring_neither_spelling_gets_the_default():
    meta = get_fields_with_metadata(MetadataProbe)["neither"]

    assert meta["tooltip"] is None
    assert meta["unit"] is None


@pytest.mark.parametrize(
    "metadata,expected",
    [
        ({"tooltip": "canonical"}, "canonical"),
        ({"help": "alias"}, "alias"),
        ({"tooltip": "canonical", "help": "alias"}, "canonical"),
        ({}, None),
        ({"tooltip": None, "help": "alias"}, "alias"),
    ],
    ids=["canonical", "alias", "both", "neither", "canonical_is_none"],
)
def test_metadata_value_reads_either_spelling(metadata, expected):
    """For consumers reading raw field metadata rather than the normalised dict."""
    assert metadata_value(metadata, "tooltip") == expected


def test_metadata_value_returns_the_default_rather_than_none():
    assert metadata_value({}, "unit", "") == ""


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
    """Including the aliases and the keys read by forms but absent from the dict."""

    @dataclass
    class Fine:
        a: float = field(default=1.0, metadata={"help": "alias", "units": "m"})
        b: str = field(default="", metadata={"filepath": True, "tooltip": "t"})

    fstructures._warned_metadata_keys.clear()
    with caplog.at_level(logging.WARNING):
        get_fields_with_metadata(Fine)

    assert [r.message for r in caplog.records if "no form reads" in r.message] == []


def test_in_tree_configs_declare_no_unreadable_keys(caplog):
    """Every registered task config, rendered through the real vocabulary.

    Catches a typo in a config that no test opens a form for -- which is most of
    them.
    """
    from fibsem.applications.autolamella.workflows.tasks import get_tasks

    fstructures._warned_metadata_keys.clear()
    with caplog.at_level(logging.WARNING):
        for task in get_tasks().values():
            get_fields_with_metadata(task.config_cls)

    offenders = [r.message for r in caplog.records if "no form reads" in r.message]
    assert offenders == []
