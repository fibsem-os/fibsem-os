"""The typed payloads of the kinds agents and operators are told about."""

from dataclasses import fields

from fibsem.hooks import HookContext, HookEvent
from fibsem.workflows.events import EventBuffer, kinds
from fibsem.workflows.events.payloads import (
    PAYLOADS,
    LevelSet,
    ProposalHold,
    TaskEvent,
    read_payload,
    unknown_fields,
)


def test_every_typed_kind_is_registered():
    assert set(PAYLOADS) <= set(kinds.KINDS)


def test_a_task_event_names_every_field_a_hook_context_carries():
    # The lifecycle kinds are written from HookContext: a field added there is
    # a field the typed payload must name.
    hook = {f.name for f in fields(HookContext)} - {"event"}
    assert TaskEvent.names() == hook


def test_reading_is_forgiving_of_old_and_new_files():
    held = ProposalHold.read({"item_name": "01", "from_the_future": 1})
    assert held == ProposalHold(item_name="01")
    assert ProposalHold.read(None) == ProposalHold()


def test_a_record_reads_as_its_kind_s_payload():
    buffer = EventBuffer()
    buffer.append(kinds.AGENT_LEVEL_SET, {"level": "drive", "previous": "assist"})
    buffer.append(kinds.EVIDENCE, {"evidence_id": "e1"})
    level, evidence = buffer.events_since(0)["events"]

    assert read_payload(level) == LevelSet(level="drive", previous="assist")
    assert read_payload(evidence) is None


def test_a_closed_kind_with_a_field_nobody_named_is_caught():
    assert unknown_fields(kinds.AGENT_LEVEL_SET, {"level": "x", "mood": 1}) == ["mood"]
    # An open kind carries extras (a run's labels).
    assert unknown_fields(kinds.COMMAND_ENDED, {"run_id": "r", "label": 1}) == []
    assert unknown_fields(kinds.EVIDENCE, {"anything": 1}) == []


def test_the_hook_event_names_are_the_typed_lifecycle_kinds():
    lifecycle = {kind for kind, cls in PAYLOADS.items() if cls is TaskEvent}
    hooked = {event.value for event in HookEvent} & set(kinds.KINDS)
    assert hooked <= lifecycle | {kinds.TASK_STEP}
