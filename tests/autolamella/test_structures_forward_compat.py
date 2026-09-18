"""A protocol written by a newer build must load in this one.

Live incident (2026-09-01): a protocol.yaml carrying a field this build did
not know ('supervisor') made the whole protocol refuse to load — the
experiment quickloaded without it. Task descriptions now keep known fields
only, the same rule AutoLamellaTaskState already follows."""

import logging

import pytest

from fibsem.applications.autolamella.structures import (
    Attention,
    AutoLamellaTaskDescription,
    attention_from,
)


def test_unknown_fields_from_the_future_are_ignored():
    task = AutoLamellaTaskDescription.from_dict(
        {
            "name": "Mill Fiducial",
            "supervise": True,
            "required": True,
            "requires": [],
            "supervisor": "agent",  # a future build's field
            "entirely_new_thing": {"nested": 1},
        }
    )
    assert task.name == "Mill Fiducial"
    assert task.attention is Attention.supervised


def test_none_still_produces_a_blank_description():
    task = AutoLamellaTaskDescription.from_dict(None)
    assert task.name == ""


def test_one_reader_takes_every_form_a_stored_attention_has_had():
    """Lamella and grid task configs read the same field through the same
    function: an attention's own value, or the supervise bool that stood in
    its place in v0.5.2 and in the per-stage supervision before it."""
    assert attention_from(True) is Attention.supervised
    assert attention_from(False) is Attention.automated
    assert attention_from("review") is Attention.review
    assert attention_from(Attention.supervised) is Attention.supervised


def test_an_attention_this_build_does_not_know_reads_as_automated(caplog):
    """One unreadable field must not drop the whole task, so a value from a
    newer build -- or an interim spelling that never shipped -- is automated
    and says so, naming where it was."""
    with caplog.at_level(logging.WARNING):
        assert attention_from("gate", "grid task 'Acquire Overview'") is (
            Attention.automated
        )
    assert "Unknown attention 'gate' on grid task 'Acquire Overview'" in caplog.text


def test_a_bad_attention_in_code_still_raises():
    """The tolerance is the loaders', not the language's: a typo in a literal
    is a bug and stays one."""
    with pytest.raises(ValueError):
        Attention("supervized")
