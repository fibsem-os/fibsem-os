"""The workflow border QSS is generated, not hand-written (FIB-679).

Two widgets draw this border -- the AutoLamella main window and the coincidence
viewer -- and the rules used to be spelled out once per widget. That is how the
``agent`` state came to exist in one copy and not the other, so a coincidence run
driven by an agent fell through to no rule at all.

These tests are deliberately string-level and import no Qt: ``fibsem.ui.stylesheets``
pulls in only ``os`` and the palette, so the guard runs anywhere the package does.
"""
import pathlib
import re

from fibsem.ui.stylesheets import (
    BORDER_STATE_COLOURS,
    BORDER_WIDTH_PX,
    border_stylesheet,
)

RULE = re.compile(
    r'QFrame#(?P<name>\w+)\[borderState="(?P<state>\w+)"\]\s*'
    r'\{\s*border:\s*(?P<px>\d+)px solid (?P<colour>#[0-9A-Fa-f]{6});\s*\}'
)

# Every widget that owns a border frame. Adding one here is the cheap way to keep
# it in the generated set rather than growing a second copy of the rules.
BORDER_FRAME_NAMES = ["workflow_border_frame", "coincidence_border_frame"]


def test_emits_exactly_one_rule_per_state():
    for name in BORDER_FRAME_NAMES:
        rules = RULE.findall(border_stylesheet(name))
        assert len(rules) == len(BORDER_STATE_COLOURS)


def test_each_rule_carries_the_declared_colour_and_width():
    for name in BORDER_FRAME_NAMES:
        found = {
            m.group("state"): (m.group("px"), m.group("colour").upper())
            for m in RULE.finditer(border_stylesheet(name))
        }
        expected = {
            state: (str(BORDER_WIDTH_PX), colour.upper())
            for state, colour in BORDER_STATE_COLOURS.items()
        }
        assert found == expected


def test_selector_targets_the_requested_object_name():
    sheet = border_stylesheet("some_other_frame")
    assert {m.group("name") for m in RULE.finditer(sheet)} == {"some_other_frame"}
    assert "workflow_border_frame" not in sheet


def test_every_border_frame_gets_the_same_states():
    """The drift this replaced: one frame had `agent`, the other did not."""
    per_frame = [
        {m.group("state") for m in RULE.finditer(border_stylesheet(name))}
        for name in BORDER_FRAME_NAMES
    ]
    assert all(states == per_frame[0] for states in per_frame)


def test_border_rules_are_not_hand_written_anywhere_else():
    """`[borderState="` appears only in a QSS selector, never in widget code.

    Widgets set the property with ``setProperty("borderState", ...)``, which does
    not match. So a hit outside stylesheets.py means someone has written a second
    copy of the rules -- exactly what FIB-679 removed.
    """
    root = pathlib.Path(__file__).resolve().parents[2] / "fibsem"
    owner = root / "ui" / "stylesheets.py"
    offenders = [
        str(path.relative_to(root.parent))
        for path in root.rglob("*.py")
        if path != owner and '[borderState="' in path.read_text(encoding="utf-8")
    ]
    assert offenders == [], (
        "border rules duplicated outside fibsem/ui/stylesheets.py: "
        + ", ".join(offenders)
    )
