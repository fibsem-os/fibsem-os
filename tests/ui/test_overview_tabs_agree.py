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
    REPO_ROOT
    / "fibsem"
    / "ui"
    / "widgets"
    / "canvas"
    / "overlays"
    / "minimap_overlays.py"
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
    return {
        "beam": BEAM.read_text(encoding="utf-8"),
        "fm": FM.read_text(encoding="utf-8"),
    }


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

    for tab, source in (
        ("beam", BEAM.read_text(encoding="utf-8")),
        ("fm", FM.read_text(encoding="utf-8")),
    ):
        assigned = _assigned_names(source)
        strays = {n for n in assigned if "GRID_RADIUS" in n or "BOUNDARY_RADIUS" in n}
        assert not strays, f"{tab} defines its own grid radius: {strays}"


# ── the geometry, not just the colours ───────────────────────────────────

STAGE_CONTEXT = (
    REPO_ROOT / "fibsem" / "ui" / "widgets" / "canvas" / "overlays" / "stage_context.py"
)

# The shapes both tabs draw for stage context. Built once, in `stage_context`, and
# neither tab may build its own -- which is what let the fluorescence copy drift into
# gating the travel box on the stage type, sizing the boundary in stage axes rather than
# along the surface, and handing a frame the `r=None` a holder file leaves (FIB-698).
SHARED_BUILDERS = ("limit_shapes", "boundary_shapes", "slot_shapes", "context_shapes")


def test_the_stage_context_shapes_are_built_in_one_place():
    defined = {
        n.name
        for n in ast.parse(STAGE_CONTEXT.read_text(encoding="utf-8")).body
        if isinstance(n, ast.FunctionDef)
    }
    assert set(SHARED_BUILDERS) <= defined


@pytest.mark.parametrize("tab", ["beam", "fm"])
def test_neither_tab_builds_its_own_stage_context_shapes(tab, sources):
    """A `ShapeSpec` for stage context, constructed in a tab, is the drift starting again.

    Each tab still builds specs of its own -- the beam tab's gridbars and both tabs'
    position markers -- so this looks for the *labels* the shared builders own rather
    than for `ShapeSpec` outright.
    """
    source = sources[tab]
    for label in ("Stage Limits", "Grid Boundary"):
        assert f'"{label}"' not in source and f"'{label}'" not in source, (
            f"the {tab} tab writes the {label!r} shape itself; "
            "`stage_context` is where both tabs get it"
        )


def test_the_fluorescence_tab_sizes_nothing_with_frame_length():
    """`frame.length()` is a pure scale, so anything lying *on the sample* sized with it
    is the same size in every view -- and therefore right in at most one of them.

    This tab did exactly that for the grid boundary, which is why it drew a circle where
    the beam tab draws an ellipse. `stage_context.canvas_span` is the one that applies
    the view's foreshortening.

    **The beam tab is deliberately not checked.** It still has three, all in the gridbar
    lattice, and its own comment says they are wrong in the same way -- a square lattice
    on the sample is not square in a tilted view, so at the milling pose the horizontal
    bars sit about four times too far apart. That is the rest of FIB-615, left on
    purpose. Asserting it here would be asserting something known to be false, and the
    test would have to be deleted to make the tree green rather than fixed.
    """
    assert "frame.length(" not in FM.read_text(encoding="utf-8"), (
        "the fluorescence tab sizes something with frame.length(); if it lies on the "
        "sample surface it wants stage_context.canvas_span()"
    )


# ── the settings column is stacked the same way on both tabs ─────────────────


def _added_widgets(source: str) -> dict:
    """Every ``addWidget`` call, keyed by the function that makes it.

    Returns ``{function_name: [(lineno, described_argument), ...]}`` in source order.
    The argument is described rather than evaluated: ``self.status`` becomes
    ``"self.status"`` and a bare local becomes its name, which is all the ordering
    questions below need.
    """

    def describe(node):
        if isinstance(node, ast.Attribute):
            return f"{describe(node.value)}.{node.attr}"
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Call):
            return describe(node.func)
        return ""

    calls: dict = {}
    tree = ast.parse(source)
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef):
            continue
        found = [
            (node.lineno, describe(node.args[0]))
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "addWidget"
            and node.args
        ]
        if found:
            calls[function.name] = sorted(found)
    return calls


def _action_column(source: str) -> list:
    """The arguments of the layout that stacks the acquire buttons, in order.

    Located by finding the function that adds the `buttons` container, since both tabs
    build one and neither adds it anywhere else.
    """
    for _, added in _added_widgets(source).items():
        names = [name for _, name in added]
        if "buttons" in names:
            return names
    raise AssertionError("neither tab appears to lay out its acquire buttons")


@pytest.mark.parametrize("tab", ["beam", "fm"])
def test_the_status_is_above_the_acquire_button(tab, sources):
    """Both tabs report a stage move on their status label (FIB-765), and reported it in
    opposite places: below the button on the FIB/SEM side, above it on the fluorescence
    one. Same message, mirrored layout.

    Above, so the button is the last thing in the column -- the ordinary place for a
    commit action, and it means the button does not shift as the readouts above it
    appear and disappear through a run.
    """
    column = _action_column(sources[tab])
    status = next(name for name in column if name.endswith(("status", "label_status")))

    assert column.index(status) < column.index("buttons"), (
        f"the {tab} tab puts its status below the acquire button: {column}"
    )


@pytest.mark.parametrize("tab", ["beam", "fm"])
def test_the_progress_is_above_the_acquire_button(tab, sources):
    """Same argument, and the one that keeps the two readouts together: a progress bar
    on one side of the button and a status line on the other is the worst of the three
    arrangements, whichever side each ends up on."""
    column = _action_column(sources[tab])
    progress = next(name for name in column if "progress" in name)

    assert column.index(progress) < column.index("buttons"), (
        f"the {tab} tab puts its progress below the acquire button: {column}"
    )
