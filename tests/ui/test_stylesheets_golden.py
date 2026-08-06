"""Every public string constant in ``stylesheets.py`` renders exactly as recorded.

The QSS constants interpolate the shared palette via f-strings. QSS is full of
braces, every one of which has to be doubled, and a missed pair silently drops a
rule instead of raising -- the widget then renders unstyled, which no other test
here would notice. Plain strings could not fail that way, so this guard exists
because of the move to f-strings, not in spite of it.

Recorded as hashes rather than full text: this only has to answer "did the
rendered output change", and the *what* is already readable in the
``stylesheets.py`` diff of the same commit. 16 hex chars is 64 bits, far more
than enough to catch an accidental edit to 68 fixed strings.

When a change is deliberate, regenerate with::

    python tests/ui/test_stylesheets_golden.py

which prints a replacement block to paste over ``_GOLDEN`` below. It is a manual
paste on purpose -- a snapshot test that regenerates itself in place teaches you
to accept diffs without reading them.

``stylesheets`` is loaded without executing ``fibsem/ui/__init__.py``, which
imports the widget layer and so needs napari/PyQt5. CI installs only ``.[test]``,
so importing it the ordinary way would skip this test in exactly the job it is
meant to protect. Stubbing the package also pins a property worth keeping:
``stylesheets`` stays a leaf module that costs nothing to import.
"""

import contextlib
import hashlib
import importlib
import re
import sys
import types
from pathlib import Path

import pytest

UI_DIR = Path(__file__).resolve().parents[2] / "fibsem" / "ui"

# `{BORDER_COLOR}` surviving into the output means a brace pair was doubled that
# should have been a substitution. Qt drops the malformed rule silently.
_UNSUBSTITUTED = re.compile(r"\{[A-Z_][A-Z_0-9]*\}")

# Stands in for the absolute icons directory, which is checkout-specific.
_ICONS_PLACEHOLDER = "<ICONS_DIR>"

# An absolute path anywhere else in a rendered sheet would make the recorded
# value machine-specific the same way, so it is rejected rather than absorbed.
_ABSOLUTE_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|/(?:Users|home|opt|tmp|private|var)/)")

_GOLDEN = {
    "ACCENT_COLOR": "5791fda913e64277",
    "AUTOMATED_COLOR": "af19d595dbe7669e",
    "BLUE_PUSHBUTTON_STYLE": "890ed98ead9a0205",
    "BORDER_COLOR": "7a03f94fa9d76551",
    "CANVAS_BG": "2a9c18bc4e8c918a",
    "CHECKBOX_STYLE": "ee1027525c91f5ec",
    "DATETIME_EDIT_STYLESHEET": "825dde4ebfb50eb7",
    "DEFECT_ORANGE_COLOR": "67fe9e13198b7262",
    "DEFECT_RED_COLOR": "fb3bf12cd8f86d6f",
    "DISABLED_PUSHBUTTON_STYLE": "f46d46fcd500df08",
    "ERROR_COLOR": "fb3bf12cd8f86d6f",
    "FAILED_PROGRESS_BAR_STYLESHEET": "fabca21e8cc7cc6e",
    "GRAY_BACKGROUND_COLOR": "a929f4a7412bb9bf",
    "GRAY_CANVAS_COLOR": "7b2817bf9b80e66d",
    "GRAY_CONSOLE_COLOR": "21dcede6e69a9179",
    "GRAY_FOREGROUND_COLOR": "c6d5381eadc65897",
    "GRAY_HIGHLIGHT_COLOR": "33ea1238c672179b",
    "GRAY_ICON_COLOR": "000319f1e13f2318",
    "GRAY_PRIMARY_COLOR": "84a9594fa9bb054b",
    "GRAY_PUSHBUTTON_STYLE": "9934ee7d147d3784",
    "GRAY_SECONDARY_COLOR": "fa319cb18c0863fe",
    "GRAY_TEXT_COLOR": "1aec4a9d078e736f",
    "GRAY_WHITE_COLOR": "07ee686b3ce29f11",
    "GREEN_COLOR": "af19d595dbe7669e",
    "GREEN_PUSHBUTTON_STYLE": "b2e20b07daad5c73",
    "INDETERMINATE_PROGRESS_BAR_STYLESHEET": "1c7dfa7f1eb2fdc7",
    "LABEL_INSTRUCTIONS_STYLE": "a2e40af45042a73e",
    "LIST_WIDGET_STYLESHEET": "6f6a2df891282951",
    "MESSAGE_BOX_STYLESHEET": "5a9960a342c4e43c",
    "MILLING_PROGRESS_BAR_STYLESHEET": "621feff03128b0b6",
    "NAPARI_STYLE": "769c3aa4de2aa9e2",
    "OK_COLOR": "af19d595dbe7669e",
    "ORANGE_COLOR": "30abb8ea87c6abf8",
    "ORANGE_PUSHBUTTON_STYLE": "8bf8fa4f934839fe",
    "PANEL_COLOR": "72dc7efbf8fb183b",
    "PRIMARY_ACCENT": "9d494816709b31fd",
    "PRIMARY_BUTTON_STYLESHEET": "35e174e8aebf68e3",
    "PRIMARY_COLOR": "df784d4ffdeccf99",
    "PRIMARY_COLOR_HOVER": "d258e9a18382ced5",
    "PRIMARY_COLOR_PRESSED": "16d7510927e7a94b",
    "PROGRESS_BAR_BLUE_STYLE": "af78c8a18384dc35",
    "PROGRESS_BAR_GREEN_STYLE": "6038107aacdc9137",
    "PURPLE_COLOR": "05887be54e2eca6d",
    "RED_COLOR": "429aea9dfaea3849",
    "RED_PUSHBUTTON_STYLE": "021c6482a192073e",
    "ROW_ALT_COLOR": "b1fb8ff122c786ad",
    "RUN_WORKFLOW_BUTTON_STYLESHEET": "3629ecdf117e3a90",
    "SECONDARY_BUTTON_STYLESHEET": "4b9058e9dee0c937",
    "SEMANTIC_ERROR_COLOR": "429aea9dfaea3849",
    "SEMANTIC_ERROR_HOVER_COLOR": "e5c544982c6cf445",
    "SEMANTIC_ERROR_PRESSED_COLOR": "bf645a12cedea175",
    "SEMANTIC_WARNING_COLOR": "c08177246f540adb",
    "STATUS_BAR_STYLESHEET": "581c32dced024b25",
    "STOP_WORKFLOW_BUTTON_STYLESHEET": "fbcea4b6541e229e",
    "SUPERVISION_STATUS_AUTOMATED_STYLESHEET": "c860f4842a08a937",
    "SUPERVISION_STATUS_SUPERVISED_STYLESHEET": "9b4781f153f381d4",
    "SURFACE_COLOR": "a929f4a7412bb9bf",
    "TEXT_COLOR": "587ee5b584c2dab2",
    "TEXT_MUTED_COLOR": "440517a362cb0d63",
    "TEXT_STRONG_COLOR": "bc075fd6b7676cb2",
    "TOOLBUTTON_ICON_STYLESHEET": "2866396d9958a9ff",
    "TOOLTIP_STYLESHEET": "727d5f9cfeb958e3",
    "USER_ATTENTION_BUTTON_STYLESHEET": "95b019cd38820c21",
    "WARN_COLOR": "481492ba043ddaf3",
    "WHITE_ICON_COLOR": "f2074b6cef37b673",
    "WHITE_PUSHBUTTON_STYLE": "eb89a0f5ebaa41c8",
    "WORKFLOW_BORDER_STYLESHEET": "4edc3709c235a8f0",
    "YELLOW_PUSHBUTTON_STYLE": "8e04fc3d506dc6e9",
}


def _ui_modules():
    return [n for n in sys.modules if n == "fibsem.ui" or n.startswith("fibsem.ui.")]


@contextlib.contextmanager
def _leaf_import():
    """Import ``fibsem.ui.stylesheets`` with a stub for its parent package.

    Every ``fibsem.ui*`` entry is saved and restored, rather than a fixed list,
    so that a module imported transitively (``stylesheets`` re-exports
    ``napari_style``, which may re-export more later) cannot be left behind
    pointing at the stub for a later test that wants the real package.
    """
    saved = {name: sys.modules[name] for name in _ui_modules()}
    try:
        for name in saved:
            del sys.modules[name]
        package = types.ModuleType("fibsem.ui")
        package.__path__ = [str(UI_DIR)]
        sys.modules["fibsem.ui"] = package
        yield importlib.import_module("fibsem.ui.stylesheets")
    finally:
        for name in _ui_modules():
            del sys.modules[name]
        sys.modules.update(saved)


def _digest(value):
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def _render():
    """{name: value} for every public module-level string constant.

    Two sheets embed ``_ICONS_DIR`` -- an absolute path -- in ``url(...)`` rules
    for the spinbox and combobox arrows, so their rendered text differs between
    a developer checkout and CI. Substituting a placeholder keeps the rest of
    those sheets pinned while making the recorded value machine-independent;
    hashing the raw text pins the checkout it was generated on and nothing else.
    """
    with _leaf_import() as stylesheets:
        icons_dir = stylesheets._ICONS_DIR
        values = {}
        for name in dir(stylesheets):
            if name.startswith("_"):
                continue
            value = getattr(stylesheets, name)
            if isinstance(value, str):
                values[name] = value.replace(icons_dir, _ICONS_PLACEHOLDER)
        return values


@pytest.fixture(scope="module")
def rendered():
    return _render()


def test_golden_table_is_populated():
    """Guard the guard: an emptied _GOLDEN would collect nothing below and pass."""
    assert len(_GOLDEN) > 60, f"_GOLDEN looks truncated ({len(_GOLDEN)} entries)"
    assert "NAPARI_STYLE" in _GOLDEN


def test_no_constant_was_added_or_removed(rendered):
    removed = sorted(set(_GOLDEN) - set(rendered))
    added = sorted(set(rendered) - set(_GOLDEN))

    assert not removed, (
        "constants disappeared from stylesheets.py -- other modules import "
        f"these by name: {', '.join(removed)}"
    )
    assert not added, (
        f"new constants are not in _GOLDEN: {', '.join(added)}. "
        "Regenerate with: python tests/ui/test_stylesheets_golden.py"
    )


@pytest.mark.parametrize("name", sorted(_GOLDEN))
def test_constant_renders_unchanged(rendered, name):
    assert _digest(rendered[name]) == _GOLDEN[name], (
        f"{name} renders differently than recorded -- see this commit's "
        "stylesheets.py diff for what changed. If it is deliberate, regenerate "
        "with: python tests/ui/test_stylesheets_golden.py"
    )


def test_no_rendered_sheet_carries_an_absolute_path(rendered):
    """The recorded values have to mean the same thing on every checkout.

    ``_ICONS_DIR`` is normalised away in ``_render``; anything else absolute
    would pin the machine the hashes were generated on, so CI would fail with a
    digest mismatch that looks like a style regression and is not one.
    """
    offenders = {
        name: _ABSOLUTE_PATH.search(value).group()
        for name, value in rendered.items()
        if _ABSOLUTE_PATH.search(value)
    }
    assert offenders == {}, (
        "a rendered stylesheet embeds an absolute path, which differs between "
        f"checkouts: {offenders}. Normalise it in _render() as _ICONS_DIR is."
    )


def test_no_unsubstituted_token_placeholders(rendered):
    offenders = {
        name: sorted(set(found))
        for name, value in rendered.items()
        if (found := _UNSUBSTITUTED.findall(value))
    }
    assert offenders == {}, (
        "a brace pair was doubled where a token substitution was intended, so "
        f"the placeholder reached the rendered QSS: {offenders}"
    )


if __name__ == "__main__":
    print("_GOLDEN = {")
    for _name, _value in sorted(_render().items()):
        print(f'    "{_name}": "{_digest(_value)}",')
    print("}")
