"""Nothing should re-declare a colour that already exists in the shared palette.

Six hand-built dialogs had each pasted the same napari-dark palette locally, and
four of those values already existed under shade-derived names -- ``#262930``
was ``GRAY_BACKGROUND_COLOR``, ``#d04040`` was ``DEFECT_RED_COLOR``. Nothing
detects that: the copies agree until someone adjusts one of them, and then two
dialogs disagree with the rest of the app for no reason anybody can see in a
diff.

So the rule is enforced rather than documented: if a colour is already in the
palette, import it.

The palette itself now lives in ``fibsem/ui/tokens.py``; ``stylesheets.py`` is
still read for declarations because it re-exports the palette and is the module
most callers import a colour *from*, so a colour reintroduced there is the same
duplicate this test exists to catch.
"""

import re
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[2] / "fibsem"
STYLESHEETS = PACKAGE / "ui" / "stylesheets.py"
TOKENS = PACKAGE / "ui" / "tokens.py"

# The files that are allowed to declare a palette colour. Anything else that
# declares one is duplicating them.
PALETTE_SOURCES = (TOKENS, STYLESHEETS)

# Matches `NAME = "#hex"` and the wrapped form `NAME = QColor("#hex")`.
#
# The wrapper is not cosmetic: matching only the bare form left this test blind
# to a whole file. tile_mask_widget.py declared five colours as QColor("...")
# -- four of them exact copies of SURFACE_COLOR, PANEL_COLOR, BORDER_COLOR and
# ACCENT_COLOR -- and the build stayed green, which is the exact outcome this
# test exists to prevent.
_DECLARATION = re.compile(
    r'^(_?[A-Z][A-Z_0-9]*) *= *(?:[A-Za-z_][\w.]* *\( *)?"(#[0-9a-fA-F]{3,8})"',
    re.M,
)


def _shared_colours():
    """{value: [names]} for every colour constant in the palette modules."""
    shared = {}
    for source in PALETTE_SOURCES:
        for name, value in _DECLARATION.findall(source.read_text(encoding="utf-8")):
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

    Third instance of that, and the reason this now matches any ``ui`` package
    rather than the one under ``fibsem/ui``: FIB-558/559/560 moved 25 widgets to
    ``fibsem/applications/autolamella/ui``. Anchored at ``fibsem/ui``, this walk
    silently stopped covering 22 files -- the ones that happened not to mention
    ``stylesheets`` stayed in only by accident. The test kept passing, with less
    to test. ``tests/test_ui_extra_isolation.py`` had already learned this and
    matches on a ``ui`` path component; do the same here.
    """
    modules = []
    for path in PACKAGE.rglob("*.py"):
        if path in PALETTE_SOURCES:
            continue
        rel = path.relative_to(PACKAGE)
        if "ui" in rel.parts or "stylesheets" in path.read_text(encoding="utf-8"):
            modules.append(path)
    return sorted(modules)


def test_the_palette_is_where_this_test_looks_for_it():
    """Guard the guard: a typo in the regex, or a palette that moved out from
    under these paths, would make every case below vacuous rather than failing."""
    shared = _shared_colours()
    assert shared.get("#262930")  # SURFACE_COLOR / GRAY_BACKGROUND_COLOR
    assert shared.get("#1e2027")  # PANEL_COLOR
    assert len(shared) > 20


@pytest.mark.parametrize("module", _modules(), ids=lambda p: p.name)
def test_no_module_redeclares_a_shared_colour(module):
    shared = _shared_colours()

    offenders = [
        f"{name} = {value!r} is already {' / '.join(shared[value.lower()])}"
        for name, value in _DECLARATION.findall(module.read_text(encoding="utf-8"))
        if value.lower() in shared
    ]

    assert offenders == [], (
        f"{module.name} re-declares colours that stylesheets.py already defines. "
        f"Import them instead: " + "; ".join(offenders)
    )
