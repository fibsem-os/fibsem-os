"""No widget should re-declare a colour that already exists in stylesheets.py.

Five hand-built dialogs had each pasted the same napari-dark palette locally, and
four of those values already existed in ``stylesheets.py`` under shade-derived
names -- ``#262930`` was ``GRAY_BACKGROUND_COLOR``, ``#d04040`` was
``DEFECT_RED_COLOR``. Nothing detects that: the copies agree until someone
adjusts one of them, and then two dialogs disagree with the rest of the app for
no reason anybody can see in a diff.

So the rule is enforced rather than documented: if a colour is already in
stylesheets.py, import it.
"""

import re
from pathlib import Path

import pytest

WIDGETS = Path(__file__).resolve().parents[2] / "fibsem" / "ui" / "widgets"
STYLESHEETS = Path(__file__).resolve().parents[2] / "fibsem" / "ui" / "stylesheets.py"

_DECLARATION = re.compile(r'^(_?[A-Z][A-Z_0-9]*) *= *"(#[0-9a-fA-F]{3,8})"', re.M)


def _shared_colours():
    """{value: [names]} for every colour constant in stylesheets.py."""
    shared = {}
    for name, value in _DECLARATION.findall(STYLESHEETS.read_text()):
        shared.setdefault(value.lower(), []).append(name)
    return shared


def _widget_modules():
    return sorted(p for p in WIDGETS.glob("*.py") if p.name != "__init__.py")


def test_stylesheets_itself_has_the_palette():
    """Guard the guard: a typo in the regex would make every case below vacuous."""
    shared = _shared_colours()
    assert shared.get("#262930")  # SURFACE_COLOR / GRAY_BACKGROUND_COLOR
    assert shared.get("#1e2027")  # PANEL_COLOR
    assert len(shared) > 20


@pytest.mark.parametrize("module", _widget_modules(), ids=lambda p: p.name)
def test_no_widget_redeclares_a_shared_colour(module):
    shared = _shared_colours()

    offenders = [
        f"{name} = {value!r} is already {' / '.join(shared[value.lower()])}"
        for name, value in _DECLARATION.findall(module.read_text())
        if value.lower() in shared
    ]

    assert offenders == [], (
        f"{module.name} re-declares colours that stylesheets.py already defines. "
        f"Import them instead: " + "; ".join(offenders)
    )
