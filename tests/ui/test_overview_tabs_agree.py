"""The two overview tabs draw the same things, so they must draw them the same way.

Checked against the modules' *source*, not by importing them, for two reasons: importing
either pulls in PyQt5, which CI does not install — and this is exactly the check worth
having where the rest of the UI tests are skipped — and what is being guarded against is
someone writing a colour down a second time, which is a property of the text.

They had already drifted once. The FIB/SEM tab used palette values while the fluorescence
one asked matplotlib for `"yellow"` and `"red"`, which are pure #ffff00 and #ff0000: the
two most saturated colours available, on the two shapes that are meant to be background
context. And the specimen grid's radius was the same 1000 µm under two names in two files.

Run directly:
    python -m pytest tests/ui/test_overview_tabs_agree.py
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BEAM = REPO_ROOT / "fibsem" / "ui" / "widgets" / "overview_widget.py"
FM = REPO_ROOT / "fibsem" / "ui" / "fm" / "widgets" / "fm_overview_widget.py"
TOKENS = REPO_ROOT / "fibsem" / "ui" / "tokens.py"
OVERLAYS = (
    REPO_ROOT / "fibsem" / "ui" / "widgets" / "canvas" / "overlays" / "minimap_overlays.py"
)

# What both tabs draw of the sample holder, and where the one definition lives.
SHARED_COLOURS = ["SLOT_COLOUR", "STAGE_LIMITS_COLOUR", "GRID_BOUNDARY_COLOUR"]

# matplotlib's named colours for the two that drifted. Matched as quoted strings so the
# word "red" in a comment or a label does not trip this.
RAW_NAMED = re.compile(r"""color\s*=\s*["'](red|yellow|blue|green)["']""")

# Any hex colour literal. The palette is the place for these.
HEX_LITERAL = re.compile(r'"#[0-9a-fA-F]{6}"')


@pytest.fixture(scope="module")
def sources():
    return {"beam": BEAM.read_text(encoding="utf-8"), "fm": FM.read_text(encoding="utf-8")}


def _assigned_names(source: str) -> set:
    """Module-level names the source binds by assignment."""
    tree = ast.parse(source)
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def test_the_palette_holds_the_shared_colours():
    """Beside the position markers, which were moved there for the same reason and whose
    comment says so."""
    tokens = TOKENS.read_text(encoding="utf-8")
    for name in SHARED_COLOURS:
        assert re.search(rf"^{name}\s*=", tokens, re.M), f"{name} is not in the palette"


@pytest.mark.parametrize("tab", ["beam", "fm"])
@pytest.mark.parametrize("name", SHARED_COLOURS)
def test_neither_tab_defines_a_shared_colour_of_its_own(tab, name, sources):
    """Redefining one is how the two stopped agreeing the first time. Importing it is
    fine; assigning it is what this catches."""
    assert name not in _assigned_names(sources[tab]), (
        f"{tab} assigns {name} rather than importing it from the palette"
    )


@pytest.mark.parametrize("tab", ["beam", "fm"])
def test_no_tab_asks_matplotlib_for_a_named_colour(tab, sources):
    """`color="red"` is pure #ff0000 — the most saturated red there is, and nothing else
    in the app shouts like it. It was drawing the grid boundary."""
    found = RAW_NAMED.findall(sources[tab])
    assert not found, f"{tab} passes raw named colours: {found}"


@pytest.mark.parametrize("tab", ["beam", "fm"])
def test_no_tab_carries_a_hex_colour_literal(tab, sources):
    """Everything either tab draws is either a palette token or built from one. A hex
    literal here is a fourth spelling of a colour that already has a name."""
    found = HEX_LITERAL.findall(sources[tab])
    assert not found, f"{tab} carries hex colour literals: {found}"


def test_the_grid_radius_is_defined_once():
    """It was `GRID_RADIUS_M` on one tab and `GRID_BOUNDARY_RADIUS` on the other: the
    same 1000 µm, in two files, under two names, with nothing to keep them equal."""
    overlays = OVERLAYS.read_text(encoding="utf-8")
    assert re.search(r"^GRID_BOUNDARY_RADIUS_M\s*=", overlays, re.M)

    for tab, source in (("beam", BEAM.read_text(encoding="utf-8")), ("fm", FM.read_text(encoding="utf-8"))):
        assigned = _assigned_names(source)
        strays = {n for n in assigned if "GRID_RADIUS" in n or "BOUNDARY_RADIUS" in n}
        assert not strays, f"{tab} defines its own grid radius: {strays}"
