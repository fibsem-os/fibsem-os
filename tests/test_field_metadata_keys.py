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
"""

import logging
from dataclasses import dataclass, field

import pytest

from fibsem.structures import (
    DEFAULT_FIELD_METADATA,
    RENAMED_METADATA_KEYS,
    get_fields_with_metadata,
)
import fibsem.structures as fstructures


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
        assert old not in DEFAULT_FIELD_METADATA, f"{old!r} is both superseded and current"


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

    assert [r.message for r in caplog.records if "declares metadata key" in r.message] == []


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

    offenders = [r.message for r in caplog.records if "declares metadata key" in r.message]
    assert offenders == []
