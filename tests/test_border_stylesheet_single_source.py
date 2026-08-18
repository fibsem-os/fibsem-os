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
import pathlib

FIBSEM = pathlib.Path(__file__).resolve().parents[1] / "fibsem"
OWNER = FIBSEM / "ui" / "stylesheets.py"


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
