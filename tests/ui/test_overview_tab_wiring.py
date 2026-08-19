"""The Overview (Canvas) tab is wired into every lifecycle point its twin is.

Checked *statically*, against the window's source, because the window cannot be built
here: it still constructs the napari minimap, and a `napari.Viewer` segfaults under the
offscreen platform (measured, exit 139 — the same reason FIB-405 could not verify the
standalone host headless). A test that tried would take the whole run down.

That is a weaker check than constructing it, and it is chosen deliberately over no check
at all. What it catches is the failure that actually happens with a tab added beside an
existing one: it gets created, and then someone forgets to tell it about a connection, a
loaded experiment, a workflow lock or a selection — each of which is silent, and each of
which leaves the tab showing something stale rather than raising.

The behaviour behind these calls is tested for real in `test_overview_tab_host.py`,
which builds the tab against a live simulator.

Run directly:
    python -m pytest tests/ui/test_overview_tab_wiring.py
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WINDOW = REPO_ROOT / "fibsem" / "applications" / "autolamella" / "ui" / "AutoLamellaMainUI.py"

# The tab the new one was modelled on, and the calls that keep it in step with the
# window. Anything the fluorescence tab is told, the beam one has to be told too --
# they are the same shape of thing in the same window.
TWIN = "fm_overview_tab"
NEW = "overview_canvas_tab"


@pytest.fixture(scope="module")
def window_source() -> str:
    return WINDOW.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def self_calls(window_source):
    """Map of `self.<method>` -> the window methods that call it.

    Separate from `methods_calling`, which records calls *through* an attribute. The
    per-tab builders are plain self-calls, and they are what a connection reaches.
    """
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


@pytest.fixture(scope="module")
def methods_calling(window_source):
    """Map of `attribute.method` -> the window methods that call it."""
    tree = ast.parse(window_source)
    calls: dict = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Call):
                continue
            func = inner.func
            if not isinstance(func, ast.Attribute):
                continue
            owner = func.value
            # `self.<attr>.<method>(...)`, and the `getattr(self, "<attr>", None)` form
            # the window uses where the tab may not exist yet.
            name = None
            if isinstance(owner, ast.Attribute) and isinstance(owner.value, ast.Name):
                if owner.value.id == "self":
                    name = owner.attr
            if name is None:
                continue
            calls.setdefault(f"{name}.{func.attr}", set()).add(node.name)
    return calls


def _stem(tab_attribute: str) -> str:
    """`fm_overview_tab` -> `fm_overview`, the name the window builds its methods from."""
    return tab_attribute[: -len("_tab")]


def _is_own_method(caller: str, tab_attribute: str) -> bool:
    """Whether a window method belongs to *this* tab rather than being shared.

    Each tab has a `_apply_<stem>_visibility` that builds it, and a
    `_on_<stem>_lamella_selected` that its own list raises. Neither has a counterpart
    obligation on the other tab -- the first is the per-tab builder, and the second
    must *not* re-select the list that raised it. So they are excluded from the
    comparison rather than treated as calls the other tab is missing.
    """
    return _stem(tab_attribute) in caller


def test_the_new_tab_is_created(window_source):
    assert "self.add_overview_canvas_tab()" in window_source, (
        "the tab is defined but never added to the window"
    )


@pytest.mark.parametrize(
    "method",
    ["refresh_microscope", "refresh_experiment", "set_interactive", "set_selected"],
)
def test_the_new_tab_is_told_what_its_twin_is_told(method, methods_calling):
    """Each of these is a way the window's state changes underneath a tab.

    Missing one is silent: the tab keeps showing the previous experiment's lamellae, or
    stays interactive through a workflow that has taken the instrument.
    """
    twin_callers = methods_calling.get(f"{TWIN}.{method}", set())
    new_callers = methods_calling.get(f"{NEW}.{method}", set())
    assert twin_callers, f"the twin no longer calls {method}; this test is stale"

    # Every window method that tells the twin has to tell the new tab, not merely one
    # of them. `refresh_microscope` in particular is reached from two places -- a
    # preferences change and a connection -- and only the second matters for a tab that
    # holds its microscope for life. Asserting "called from somewhere" let a mutation
    # removing the connection path pass.
    missing = {
        caller
        for caller in twin_callers - new_callers
        if not _is_own_method(caller, TWIN) and not _is_own_method(caller, NEW)
    }
    assert not missing, (
        f"{NEW}.{method}() is not called from {sorted(missing)}, where "
        f"{TWIN}.{method}() is"
    )


def test_the_new_tab_is_rebuilt_wherever_its_twin_is(self_calls):
    """The per-tab builders are self-calls, so the attribute-based check above cannot
    see them -- each tab's builder is excluded there as its own method.

    This is the one that matters most: the twin's builder is what hands a tab the
    current microscope, and a tab that holds its microscope for life and is not told
    about a reconnection goes on reading geometry from an instrument nobody is driving.
    Both call sites count -- a preferences change and a connection -- and a mutation
    removing only the connection one survived until this existed.

    The twin's builder is `_refresh_fm_overview_microscope` since FIB-609 shipped that
    tab to everyone with an FM: it no longer answers a preference, only "is there an
    instrument". The assertion below is a superset check, so the new tab having the
    extra preference-driven call site of its own is fine -- it still has a flag.
    """
    twin = self_calls.get("_refresh_fm_overview_microscope", set())
    new = self_calls.get("_apply_overview_canvas_visibility", set())
    assert twin, "the twin's builder is no longer called; this test is stale"
    missing = {c for c in twin - new if "fm_overview" not in c and "overview_canvas" not in c}
    assert not missing, (
        f"_apply_overview_canvas_visibility() is not called from {sorted(missing)}, "
        "where the fluorescence tab's builder is"
    )


def test_every_selection_handler_reaches_the_new_tab(methods_calling):
    """Selection is synced from several handlers, one per list. A tab left out of one
    of them updates from three lists and not the fourth, which reads as a bug in
    whichever list was clicked."""
    twin_handlers = methods_calling.get(f"{TWIN}.set_selected", set())
    new_handlers = methods_calling.get(f"{NEW}.set_selected", set())
    # A tab's own handler is excluded on both sides: it must *not* re-select the list
    # that raised the selection, which would move it out from under a click that is
    # still happening. Each tab is therefore expected to be absent from its own.
    missing = {
        caller
        for caller in twin_handlers - new_handlers
        if not _is_own_method(caller, TWIN) and not _is_own_method(caller, NEW)
    }
    assert not missing, f"the new tab is not synced from: {sorted(missing)}"


def test_the_new_tab_is_behind_its_own_feature_flag(window_source):
    """Beside the napari tab rather than replacing it, so it has to be possible to not
    see it at all.

    Read from the window's own preferences, not from a module-level `FEATURE_*` global.
    FIB-609 removed five of those; the one it kept exists because its caller is a widget
    constructor with no preferences to hand, which is not the case here.
    """
    assert "self._preferences.features.overview_canvas_tab" in window_source
    assert "FEATURE_OVERVIEW_CANVAS_TAB_ENABLED" not in window_source, (
        "the flag went back to a module global; read the preference directly"
    )


def test_the_flag_reaches_the_widget_and_not_just_the_tab_bar(window_source):
    """Off has to mean "not built", not "built and hidden".

    The widget subscribes to the microscope for its lifetime, so one left behind goes on
    working for a tab nobody can open. `set_enabled` is what carries the flag past the
    tab bar; the behaviour it buys is tested in `test_overview_tab_host.py`.
    """
    import re

    body = re.search(
        r"def _apply_overview_canvas_visibility\(.*?\n(?=    def )",
        window_source,
        re.S,
    )
    assert body, "_apply_overview_canvas_visibility is gone; this test is stale"
    body = body.group(0)
    assert "set_enabled(enabled)" in body, (
        "the flag only reaches setTabVisible; the widget is still built when it is off"
    )


def test_the_flag_exists_and_is_off_by_default():
    """Off by default: the napari Overview tab is the one people are relying on until
    this has had bench time."""
    import fibsem.config as fibsem_cfg

    assert fibsem_cfg.FeatureFlags().overview_canvas_tab is False


def test_the_flag_survives_a_preferences_round_trip(tmp_path):
    """A flag that does not persist cannot be turned on by the person who wants it,
    which is the only way this tab gets exercised before the swap.

    Through `to_dict`/`from_dict` rather than through `apply_feature_flags`, which no
    longer carries this one — saving and reloading is what the preferences dialog
    actually does, and it is the step that would silently drop an unknown field.
    """
    import fibsem.config as fibsem_cfg

    prefs = fibsem_cfg.UserPreferences()
    prefs.features.overview_canvas_tab = True
    reloaded = fibsem_cfg.UserPreferences.from_dict(prefs.to_dict())
    assert reloaded.features.overview_canvas_tab is True


def test_the_napari_overview_tab_is_untouched(window_source):
    """This change adds a tab; it does not remove one. The old tab is what a user falls
    back to, and the swap is its own change."""
    assert "self.add_minimap_tab()" in window_source
    assert "FibsemMinimapWidget(" in window_source


def test_the_done_state_in_the_status_bar_clears_itself(window_source):
    """The shared progress widget shows "Done" when a tiled run finishes, and something
    has to take it away again.

    That used to be `_on_tile_acquisition_finished`, which is connected to the *napari
    minimap's* own signal -- so a run driven from the rebuilt tab left a green "Done"
    bar in the status bar for the rest of the session. Scheduling the hide from the
    progress signal instead covers every producer of it.
    """
    import ast

    tree = ast.parse(window_source)
    handler = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_on_tile_acquisition_progress"
    )
    body = ast.dump(handler)
    assert "reset_if_finished" in body, (
        "nothing hides the Done state; it stays until another operation replaces it"
    )
    assert "singleShot" in body, "the hide has to be delayed, or Done is never seen"


# ── the two host tabs' shared base ───────────────────────────────────────

UI_DIR = REPO_ROOT / "fibsem" / "applications" / "autolamella" / "ui"
HOST_TABS = {
    "AutoLamellaOverviewTab": UI_DIR / "autolamella_overview_tab.py",
    "AutoLamellaFluorescenceOverviewTab": UI_DIR / "autolamella_fluorescence_overview_tab.py",
}


def _class_def(path: Path, name: str) -> ast.ClassDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(n for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == name)


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
    shared = {"is_available", "is_acquiring", "microscope", "experiment",
              "refresh_experiment", "refresh_microscope", "refresh_positions",
              "set_selected", "set_interactive", "_drop_overview",
              "_on_list_selection", "_on_marker_clicked", "_on_move_to_requested",
              "_on_add_requested", "_on_remove_requested"}

    for name, path in HOST_TABS.items():
        cls = _class_def(path, name)
        bases = {b.id for b in cls.bases if isinstance(b, ast.Name)}
        assert "AutoLamellaOverviewTabBase" in bases, f"{name} bases are {bases}"

        defined = {n.name for n in cls.body
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
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
        assigned = {t.id for n in cls.body if isinstance(n, ast.Assign)
                    for t in n.targets if isinstance(t, ast.Name)}
        assert not assigned & {"availability_changed", "lamella_selected"}, (
            f"{name} redeclares a signal the base owns"
        )


def test_the_base_owns_the_signals_the_window_connects_to():
    base = _class_def(UI_DIR / "overview_tab_base.py", "AutoLamellaOverviewTabBase")
    assigned = {t.id for n in base.body if isinstance(n, ast.Assign)
                for t in n.targets if isinstance(t, ast.Name)}
    assert {"availability_changed", "lamella_selected"} <= assigned


def test_a_lamella_added_anywhere_reaches_both_canvases(window_source):
    """The window's position-change handler has to re-mark *both* overviews.

    Neither tab subscribes to the experiment itself -- deliberately, since a tab is
    rebuilt when the microscope changes and a subscription holding the old one's bound
    method would be stale and undisconnectable -- so this handler is the only thing that
    can see a lamella added, removed or re-posed anywhere in the window.

    It reached only the fluorescence tab. The beam tab papered over its own edits by
    re-marking inline, and went stale for everything else: a lamella marked on the FM
    overview, added from the Microscope tab, or moved by a workflow, none of which
    appeared on the FIB/SEM canvas (FIB-709). That is the mirror of the bug
    `_wire_position_events` was written to fix, and it survived because the parity test
    above lists the calls the window makes *directly* on each tab, and this one is made
    through a handler.
    """
    tree = ast.parse(window_source)
    handler = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == "_refresh_overview_positions"),
        None,
    )
    assert handler is not None, (
        "the window has no `_refresh_overview_positions`; if it was renamed, this test "
        "and the events wired to it need to follow"
    )

    # Read as text rather than by attribute access: the handler loops over tab names, so
    # `self.<tab>.refresh_positions` never appears as an attribute chain to walk.
    body = ast.get_source_segment(window_source, handler) or ""
    for tab in (TWIN, NEW):
        assert tab in body, f"{tab} is not re-marked when the experiment's lamellae change"
    assert "refresh_positions" in body

    # And the handler has to be what the experiment's events actually reach.
    for event in ("inserted", "removed", "changed"):
        assert f"events.{event}.connect" in window_source, (
            f"nothing follows the experiment's `{event}` event any more"
        )


def test_both_tabs_tell_the_window_when_they_start_acquiring(window_source):
    """The cross-tab stage lock is derived from both tabs, so both have to report.

    `test_overview_tab_host.py` checks the rule itself against real widgets, with the
    signal connected by the fixture. That fixture cannot notice production forgetting to
    connect it -- which is the likely failure, since the connection lives in each tab's
    builder rather than anywhere the two are handled together (FIB-706).
    """
    for tab in (TWIN, NEW):
        assert f"self.{tab}.acquiring_changed.connect(" in window_source, (
            f"{tab} never tells the window it is acquiring, so the other overview is "
            "free to drive the stage during its run"
        )
    assert window_source.count("acquiring_changed.connect(self._apply_overview_locks)") == 2, (
        "the tabs report to something other than the one place that derives the lock"
    )
