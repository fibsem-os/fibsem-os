"""A protocol written by a newer build must load in this one.

Live incident (2026-09-01): a protocol.yaml carrying a field this build did
not know ('supervisor') made the whole protocol refuse to load — the
experiment quickloaded without it. A workflow entry and a task config keep
known fields only, the same rule TaskState already follows."""

import logging

import pytest

from fibsem.applications.autolamella.structures import AutoLamellaTaskProtocol
from fibsem.workflows.attention import Attention, attention_from

FIDUCIAL = "Mill Fiducial"


def _protocol(entry, config=None) -> AutoLamellaTaskProtocol:
    return AutoLamellaTaskProtocol.from_dict(
        {
            "name": "p",
            "tasks": {FIDUCIAL: {"task_type": "MILL_FIDUCIAL", **(config or {})}},
            "workflow": {"tasks": [entry]},
        }
    )


def test_unknown_fields_from_the_future_are_ignored():
    protocol = _protocol(
        {
            "name": FIDUCIAL,
            "supervise": True,
            "required": True,
            "requires": [],
            "from_a_newer_build": "value",
            "entirely_new_thing": {"nested": 1},
        },
        config={"from_a_newer_build": "value"},
    )
    assert protocol.workflow_config.workflow == [FIDUCIAL]
    assert protocol.get_attention(FIDUCIAL) is Attention.supervised


def test_a_null_entry_still_produces_a_blank_task():
    protocol = _protocol(None)
    assert [t.name for t in protocol.workflow_config.tasks] == [""]


def test_one_reader_takes_every_form_a_stored_attention_has_had():
    """Lamella and grid task configs read the same field through the same
    function: an attention's own value, or the supervise bool that stood in
    its place in v0.5.2 and in the per-stage supervision before it."""
    assert attention_from(True) is Attention.supervised
    assert attention_from(False) is Attention.automated
    assert attention_from("review") is Attention.supervised
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
