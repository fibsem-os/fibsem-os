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
    return WINDOW.read_text()


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

    This is the one that matters most: `_apply_*_visibility` is what hands a tab the
    current microscope, and a tab that holds its microscope for life and is not told
    about a reconnection goes on reading geometry from an instrument nobody is driving.
    Both call sites count -- a preferences change and a connection -- and a mutation
    removing only the connection one survived until this existed.
    """
    twin = self_calls.get("_apply_fm_overview_visibility", set())
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
