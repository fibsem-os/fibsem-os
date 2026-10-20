"""Both overview modalities are wired into every lifecycle point the other is.

The two used to be separate tabs. They are now two pages of one Overview tab
(`AutoLamellaOverviewContainerTab`, FIB-780), and both host tabs stay built and
subscribed under the stack. The window tells the container; the container tells
both pages. What these check is that fan-out, and that the window goes through it:
a page left out of a connection, a loaded experiment, a workflow lock or a
selection keeps showing something stale rather than raising.

Checked statically, against the source: this is one of the few CI runs without
PyQt5. The behaviour behind these calls is tested for real in
`test_overview_tab_host.py`, which builds the tabs against a live simulator.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WINDOW = REPO_ROOT / "fibsem" / "applications" / "autolamella" / "ui" / "window.py"
PREFERENCES_DIALOG = REPO_ROOT / "fibsem" / "ui" / "widgets" / "preferences_dialog.py"
CONTAINER_SOURCE = (
    REPO_ROOT
    / "fibsem"
    / "applications"
    / "autolamella"
    / "ui"
    / "overview"
    / "container.py"
)


@pytest.fixture(scope="module")
def window_source() -> str:
    return WINDOW.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def container_tree() -> ast.Module:
    return ast.parse(CONTAINER_SOURCE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def self_calls(window_source):
    """Map of `self.<method>` -> the window methods that call it."""
    tree = ast.parse(window_source)
    calls: dict = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and isinstance(inner.func.value, ast.Name)
                and inner.func.value.id == "self"
            ):
                calls.setdefault(inner.func.attr, set()).add(node.name)
    return calls


def _method(tree: ast.Module, cls: str, name: str) -> ast.FunctionDef:
    owner = next(
        n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls
    )
    for node in owner.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    pytest.fail(f"{cls} has no {name}; this test is stale")


def _reads_self(node: ast.AST, attr: str) -> bool:
    """Whether *node* reads `self.<attr>` as an attribute, not in a comment."""
    return any(
        isinstance(inner, ast.Attribute)
        and inner.attr == attr
        and isinstance(inner.value, ast.Name)
        and inner.value.id == "self"
        for inner in ast.walk(node)
    )


def test_the_overview_tab_is_created(window_source):
    assert "self.add_overview_tab()" in window_source, (
        "the tab is defined but never added to the window"
    )


def test_the_window_has_no_names_of_its_own_for_the_pages(window_source):
    """The window reaches the pages through the container, so a call it makes reaches
    both, and nothing it tells one page alone can drift from the other (FIB-765)."""
    assert "fm_overview_tab" not in window_source
    assert "beam_overview_tab" not in window_source


def test_the_container_is_rebuilt_on_every_connection(self_calls):
    """The builder hands each tab the current microscope; a tab not told about a
    reconnection goes on reading geometry from an instrument nobody is driving."""
    callers = self_calls.get("_refresh_overview_microscope", set())
    assert "_on_microscope_changed" in callers, (
        "the overview tabs are not rebuilt on a connection; each would go on driving "
        "the previous microscope"
    )


@pytest.mark.parametrize(
    "method",
    ["refresh_microscope", "refresh_experiment", "refresh_positions", "set_selected"],
)
def test_the_container_tells_both_pages(method, container_tree):
    node = _method(container_tree, "AutoLamellaOverviewContainerTab", method)
    assert _reads_self(node, "_tabs"), f"{method} no longer fans out over both tabs"


def test_a_selection_on_one_page_reaches_the_other(container_tree):
    node = _method(
        container_tree, "AutoLamellaOverviewContainerTab", "_on_tab_selected"
    )
    assert _reads_self(node, "_tabs")


def test_the_lock_is_derived_from_both_pages(container_tree):
    """One overview must not drive the stage while the other acquires (FIB-706), so
    both report to the locks and the locks set both."""
    init = _method(container_tree, "OverviewLocks", "__init__")
    connected = ast.get_source_segment(CONTAINER_SOURCE.read_text(), init) or ""
    assert "beam.acquiring_changed.connect(self.apply)" in connected
    assert "fluorescence.acquiring_changed.connect(self.apply)" in connected
    apply = _method(container_tree, "OverviewLocks", "apply")
    assert _reads_self(apply, "_beam") and _reads_self(apply, "_fluorescence")
    built = _method(container_tree, "AutoLamellaOverviewContainerTab", "__init__")
    assert any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "OverviewLocks"
        for n in ast.walk(built)
    ), "the container never locks its pages"


def test_a_run_locks_the_overviews(window_source):
    assert "self.overview_tab.locks.set_run_allows(False)" in window_source
    assert "self.overview_tab.locks.set_run_allows(True)" in window_source


def test_the_retired_flag_is_gone_everywhere(window_source):
    """`overview_canvas_tab` was retired rather than flipped.

    It gated the *new* tab. Keeping the name while pointing it at the old one would leave
    a preferences file nobody can read: the same key meaning opposite things depending on
    which version wrote it. A new name also gets the new default on every install,
    including the ones with a saved preferences file -- a changed default alone reaches
    only fresh ones.

    So it has to be gone from the flags, from the dialog and from the window together: a
    leftover reader would be reading a field that no longer exists.
    """
    import fibsem.config as fibsem_cfg

    assert not hasattr(fibsem_cfg.FeatureFlags(), "overview_canvas_tab")
    assert "overview_canvas_tab" not in window_source
    # Read from disk, not imported. This is one of the few CI runs *without* PyQt5, and
    # `import preferences_dialog` pulls it in -- taking the whole module down at
    # collection, which is the failure the static approach here exists to avoid.
    assert "overview_canvas_tab" not in PREFERENCES_DIALOG.read_text(encoding="utf-8")


def test_the_overview_tab_is_not_behind_any_flag(window_source):
    """It replaced a tab that shipped to everyone with an FM (FIB-611). Putting the merged
    tab behind a flag would take the fluorescence overview away from every user who had
    not turned that flag on."""
    import re

    body = re.search(r"def add_overview_tab\(.*?\n(?=    def )", window_source, re.S)
    assert body, "add_overview_tab is gone; this test is stale"
    assert "features." not in body.group(0), (
        "the Overview tab reads a feature flag; it ships to everyone"
    )


def test_the_napari_tab_and_its_flag_are_gone(window_source):
    """The Minimap tab was the fallback while the Overview tab earned its place; it has
    now been removed, and `features.napari_overview_tab` with it.

    Flag and tab together, in one test, because either alone is a defect: a flag with
    nothing to gate is a checkbox that does nothing, and a tab still built with no flag
    to hide it would come back for everyone.

    `_sub_from_dict` drops unknown keys, so a saved preferences file still carrying
    `napari_overview_tab: true` loads without complaint -- checked here rather than
    assumed, because the failure would be at every startup for exactly the users who had
    turned it on.
    """
    import fibsem.config as fibsem_cfg

    assert not hasattr(fibsem_cfg.FeatureFlags(), "napari_overview_tab")
    for gone in (
        "add_minimap_tab",
        "FibsemMinimapWidget",
        "napari_overview_tab",
        '"Minimap"',
    ):
        assert gone not in window_source, f"{gone} survived the tab removal"
    # Read from disk rather than imported: this is one of the few CI runs *without*
    # PyQt5, and importing the dialog would take the module down at collection.
    assert "napari_overview" not in PREFERENCES_DIALOG.read_text(encoding="utf-8")

    stale = {"features": {"napari_overview_tab": True}}
    assert fibsem_cfg.UserPreferences.from_dict(stale) is not None


def test_the_window_no_longer_imports_napari(window_source):
    """The tab was the window's only use of it.

    Worth pinning: napari stays installed for labelling and segmentation, so an import
    creeping back here would cost nothing visible and would quietly re-couple the main
    window to it (FIB-407).
    """
    assert "import napari" not in window_source


# ── the two host tabs' shared base ───────────────────────────────────────

UI_DIR = REPO_ROOT / "fibsem" / "applications" / "autolamella" / "ui"
HOST_TABS = {
    "BeamOverviewTab": UI_DIR / "overview/beam.py",
    "FluorescenceOverviewTab": UI_DIR / "overview/fluorescence.py",
}


def _class_def(path: Path, name: str) -> ast.ClassDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(
        n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == name
    )


def test_the_two_host_tabs_share_one_base():
    """Both drive an experiment the same way, so neither owns that plumbing.

    Read from source rather than imported, like everything else in this file: this is
    the one UI test module CI actually runs, because it needs no PyQt5. An earlier
    version of this test imported the classes and turned the whole run red on every
    platform.

    The first pass moved only the methods whose code was byte-identical (FIB-697); the
    second moved the ones that differed solely in *which pose they read*, behind the
    `_pose_of` hook (FIB-709). What is left below is the whole shared surface, and a
    subclass defining any of it again is a subclass that can drift.

    `_on_move_requested` is deliberately absent: dragging a marker writes one pose on
    one tab and derives both on the other, after asking. Those are different operations
    sharing a name.
    """
    shared = {
        "is_available",
        "is_acquiring",
        "microscope",
        "experiment",
        "refresh_experiment",
        "refresh_microscope",
        "refresh_positions",
        "set_selected",
        "set_interactive",
        "_drop_overview",
        "_on_list_selection",
        "_on_marker_clicked",
        "_on_move_to_requested",
        "_on_add_requested",
        "_on_remove_requested",
    }

    for name, path in HOST_TABS.items():
        cls = _class_def(path, name)
        bases = {b.id for b in cls.bases if isinstance(b, ast.Name)}
        assert "AutoLamellaOverviewTabBase" in bases, f"{name} bases are {bases}"

        defined = {
            n.name
            for n in cls.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        again = defined & shared
        assert not again, (
            f"{name} defines {sorted(again)} again; the base is what stops the two tabs "
            "drifting apart"
        )
        # The pose is the difference between the tabs, so each must still answer it --
        # inheriting `_pose_of` would raise on the first lamella, and a subclass that
        # somehow satisfied it generically would be marking one tab's poses on both.
        for hook in ("_pose_of", "_build_overview", "_on_move_requested"):
            assert hook in defined, f"{name} does not answer {hook}"

        # Both signals are declared once, on the base. A subclass redeclaring one
        # shadows it, and the window connects to whichever it happens to see first.
        assigned = {
            t.id
            for n in cls.body
            if isinstance(n, ast.Assign)
            for t in n.targets
            if isinstance(t, ast.Name)
        }
        assert not assigned & {"availability_changed", "lamella_selected"}, (
            f"{name} redeclares a signal the base owns"
        )


def test_the_base_owns_the_signals_the_window_connects_to():
    base = _class_def(UI_DIR / "overview/base.py", "AutoLamellaOverviewTabBase")
    assigned = {
        t.id
        for n in base.body
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name)
    }
    assert {"availability_changed", "lamella_selected"} <= assigned


def test_a_lamella_added_anywhere_reaches_both_canvases(window_source):
    """The window's position-change handler re-marks the Overview tab, which re-marks
    both pages. Neither page subscribes to the experiment itself, so this handler is
    the only thing that sees a lamella added, removed or re-posed anywhere (FIB-709)."""
    tree = ast.parse(window_source)
    handler = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef)
            and n.name == "_refresh_overview_positions"
        ),
        None,
    )
    assert handler is not None, (
        "the window has no `_refresh_overview_positions`; if it was renamed, this test "
        "and the events wired to it need to follow"
    )
    body = ast.get_source_segment(window_source, handler) or ""
    assert "self.overview_tab.refresh_positions()" in body

    for event in ("inserted", "removed", "changed"):
        assert f"events.{event}.connect" in window_source, (
            f"nothing follows the experiment's `{event}` event any more"
        )
