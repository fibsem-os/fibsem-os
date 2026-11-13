"""The registry of event kinds (fibsem.workflows.events.kinds)."""

import ast
import re
from pathlib import Path

from fibsem.hooks import HookEvent
from fibsem.workflows.events import kinds as event_kinds
from fibsem.workflows.events.kinds import GROUPS, KINDS, LIVE_KINDS, group_of

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


def test_the_events_page_has_every_group():
    source = (ROOT / "web/src/lib/eventText.ts").read_text(encoding="utf-8")
    listed = re.search(r"export const GROUPS = \[(.*?)\] as const", source, re.S)
    assert listed is not None
    assert re.findall(r'\["([a-z]+)",', listed.group(1)) == list(GROUPS)


# The modules that read the record by kind.
READERS = (
    "fibsem/ai/agent/watch.py",
    "fibsem/alerts/rules.py",
    "fibsem/workflows/history_rows.py",
    "fibsem/applications/autolamella/tools/event_tables.py",
    "fibsem/applications/autolamella/tools/replay.py",
)


def _kind_literals(path):
    """The string literals compared with an event's kind in *path*."""
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        sides = [node.left, *node.comparators]
        if not any("kind" in ast.unparse(side).lower() for side in sides):
            continue
        for side in sides:
            elements = (
                side.elts
                if isinstance(side, (ast.Tuple, ast.List, ast.Set))
                else [side]
            )
            for element in elements:
                if isinstance(element, ast.Constant) and isinstance(element.value, str):
                    yield element.lineno, element.value


def test_readers_name_event_kinds_by_the_registry_s_constants():
    # A renamed kind then breaks at import, not silently in one reader.
    found = [
        f"{path}:{line} {value!r}"
        for path in READERS
        for line, value in _kind_literals(path)
        if value in KINDS
    ]
    assert not found, found
