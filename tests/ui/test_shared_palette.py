"""Nothing should re-declare a colour that already exists in stylesheets.py.

Six hand-built dialogs had each pasted the same napari-dark palette locally, and
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

PACKAGE = Path(__file__).resolve().parents[2] / "fibsem"
STYLESHEETS = PACKAGE / "ui" / "stylesheets.py"

_DECLARATION = re.compile(r'^(_?[A-Z][A-Z_0-9]*) *= *"(#[0-9a-fA-F]{3,8})"', re.M)


def _shared_colours():
    """{value: [names]} for every colour constant in stylesheets.py."""
    shared = {}
    for name, value in _DECLARATION.findall(STYLESHEETS.read_text()):
        shared.setdefault(value.lower(), []).append(name)
    return shared


def _modules():
    """Modules for which importing the palette is free.

    Everything under ``fibsem/ui``, plus anything elsewhere that already imports
    ``stylesheets``. Not the whole package, and the boundary is not arbitrary:
    ``fibsem/ui/__init__`` imports the widget layer, so ``import
    fibsem.ui.stylesheets`` from outside costs 133 fibsem modules instead of the
    one it looks like. Applying this rule to ``correlation/fit_diagnostics.py``
    tripled that module's import weight, and ``correlation.util`` imports it at
    module level -- so every correlation user would have paid for the entire UI.

    Scoped to ``fibsem/ui/widgets/*.py`` alone at first, which is how a sixth
    full copy of the palette in ``fibsem/ui/fm/widgets`` went unnoticed: a
    boundary drawn where the search happened to start is not one the next dialog
    will respect.
    """
    modules = []
    for path in PACKAGE.rglob("*.py"):
        if path == STYLESHEETS:
            continue
        if PACKAGE / "ui" in path.parents or "stylesheets" in path.read_text():
            modules.append(path)
    return sorted(modules)


def test_stylesheets_itself_has_the_palette():
    """Guard the guard: a typo in the regex would make every case below vacuous."""
    shared = _shared_colours()
    assert shared.get("#262930")  # SURFACE_COLOR / GRAY_BACKGROUND_COLOR
    assert shared.get("#1e2027")  # PANEL_COLOR
    assert len(shared) > 20


@pytest.mark.parametrize("module", _modules(), ids=lambda p: p.name)
def test_no_module_redeclares_a_shared_colour(module):
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
