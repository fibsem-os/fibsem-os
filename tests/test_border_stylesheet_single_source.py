"""The workflow border QSS is written in exactly one module (FIB-679).

Two widgets draw this border -- the AutoLamella main window and the coincidence
viewer -- and the rules used to be spelled out once per widget. That is how the
``agent`` state came to exist in one copy and not the other, so a coincidence run
driven by an agent fell through to no rule at all.

Source-level on purpose. ``fibsem.ui.__init__`` eagerly imports Qt widgets, so any
``from fibsem.ui...`` import needs PyQt5 -- which CI deliberately does not install
(see the ``[ui]`` extra note in python-package.yml). Reading the files instead keeps
this guard running on every version in the matrix, which is the whole point: it is
what stops the rules being copied a third time. The behavioural tests that need a
real widget live in tests/ui/test_border_stylesheet.py and skip without PyQt5.
"""

import ast
import pathlib
import re

FIBSEM = pathlib.Path(__file__).resolve().parents[1] / "fibsem"
OWNER = FIBSEM / "ui" / "stylesheets.py"

SET_STATE = re.compile(r'_set_border_state\(\s*"(?P<state>\w+)"')


def _styled_states():
    """Keys of BORDER_STATE_COLOURS, read from the source rather than imported."""
    tree = ast.parse(OWNER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "BORDER_STATE_COLOURS"
            for t in node.targets
        ):
            return {key.value for key in node.value.keys}
    raise AssertionError("BORDER_STATE_COLOURS not found in " + str(OWNER))


def test_border_rules_are_not_hand_written_outside_the_stylesheet_module():
    """``[borderState="`` appears only in a QSS selector, never in widget code.

    Widgets set the property with ``setProperty("borderState", ...)``, which does
    not match this string. So a hit anywhere but the owner module means someone
    has written a second copy of the rules.
    """
    offenders = [
        str(path.relative_to(FIBSEM.parent))
        for path in FIBSEM.rglob("*.py")
        if path != OWNER and '[borderState="' in path.read_text(encoding="utf-8")
    ]
    assert offenders == [], (
        "border rules duplicated outside fibsem/ui/stylesheets.py: "
        + ", ".join(offenders)
    )


def test_the_owner_module_still_generates_them():
    """Guard against the above passing vacuously if the generator is removed."""
    source = OWNER.read_text(encoding="utf-8")
    assert "def border_stylesheet(" in source
    assert "BORDER_STATE_COLOURS" in source
    assert '[borderState="' in source


def test_every_state_the_code_sets_has_a_rule_to_paint_it():
    """A state with no rule paints no border at all.

    That is not hypothetical: `agent` was set on the coincidence frame while only
    the main window's copy of the sheet declared it, so those runs showed nothing.
    FIB-679 removed the duplication; this stops the same gap reappearing from the
    other direction, where a new state is set in code and never given a colour.

    Only quoted literals are matched -- `_set_border_state(initial_state)` and the
    supervised/automated ternary pass variables, and their values appear as literals
    at the sites that build them.

    The reverse (a rule no code sets) is deliberately not asserted: landing a state's
    styling ahead of its wiring is a normal way to ship groundwork.
    """
    styled = _styled_states()
    used = {}
    for path in FIBSEM.rglob("*.py"):
        for match in SET_STATE.finditer(path.read_text(encoding="utf-8")):
            used.setdefault(match.group("state"), set()).add(
                str(path.relative_to(FIBSEM.parent))
            )
    unstyled = {
        state: sorted(paths) for state, paths in used.items() if state not in styled
    }
    assert unstyled == {}, f"states set in code with no border rule: {unstyled}"
