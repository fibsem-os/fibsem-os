"""The registry of event kinds (fibsem.workflows.event_kinds)."""

import re
from pathlib import Path

from fibsem.hooks import HookEvent
from fibsem.workflows import event_kinds
from fibsem.workflows.event_kinds import GROUPS, KINDS, LIVE_KINDS, group_of

ROOT = Path(__file__).resolve().parents[1]


def test_every_kind_has_a_known_group_and_a_constant():
    constants = {
        value
        for name, value in vars(event_kinds).items()
        if name.isupper() and isinstance(value, str)
    }
    for kind in KINDS.values():
        assert kind.group in GROUPS, kind
        assert kind.name in constants, kind
        assert kind.about, kind


def test_every_lifecycle_event_is_registered():
    assert {event.value for event in HookEvent} <= set(KINDS)


def test_live_kinds_are_the_live_group():
    assert LIVE_KINDS == {k for k, v in KINDS.items() if v.group == "live"}
    assert "milling_progress" in LIVE_KINDS
    assert "milling" not in LIVE_KINDS


def test_an_unregistered_kind_is_other():
    assert group_of("not_a_kind") == "other"


def test_every_kind_the_events_page_names_is_registered():
    source = (ROOT / "web/src/lib/eventText.ts").read_text(encoding="utf-8")
    named = set(re.findall(r'case "([a-z_]+)"', source))
    named |= set(re.findall(r'^\s+"([a-z_]+)",$', source, re.MULTILINE))
    named -= {group for group in GROUPS}
    assert named, "found no kinds in eventText.ts"
    assert named <= set(KINDS), sorted(named - set(KINDS))
