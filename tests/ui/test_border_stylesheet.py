"""The generated border QSS carries every state, for every frame that draws one.

The single-source guard -- that these rules are not hand-written a second time --
lives in tests/test_border_stylesheet_single_source.py, which imports nothing and so
still runs on CI. This module needs the real ``fibsem.ui`` package, and importing any
part of it pulls in Qt through ``fibsem/ui/__init__.py``, so it skips without PyQt5.
"""

import re

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("napari")

from fibsem.ui.stylesheets import (  # noqa: E402
    BORDER_STATE_COLOURS,
    BORDER_WIDTH_PX,
    border_stylesheet,
)

RULE = re.compile(
    r'QFrame#(?P<name>\w+)\[borderState="(?P<state>\w+)"\]\s*'
    r"\{\s*border:\s*(?P<px>\d+)px solid (?P<colour>#[0-9A-Fa-f]{6});\s*\}"
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
