"""A protocol written by a newer build must load in this one.

Live incident (2026-09-01): a protocol.yaml carrying a field this build did
not know ('supervisor') made the whole protocol refuse to load — the
experiment quickloaded without it. Task descriptions now keep known fields
only, the same rule AutoLamellaTaskState already follows."""

from fibsem.applications.autolamella.structures import AutoLamellaTaskDescription


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
    assert task.supervise is True


def test_none_still_produces_a_blank_description():
    task = AutoLamellaTaskDescription.from_dict(None)
    assert task.name == ""
