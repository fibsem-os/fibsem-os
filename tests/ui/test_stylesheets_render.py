"""The QSS constants render to usable stylesheets.

``stylesheets.py`` builds its QSS with f-strings so the rules can interpolate the
shared palette. That buys a real hazard: QSS is made of braces, an f-string needs
every literal one doubled, and the two ways of getting it wrong are not equally
loud.

    QProgressBar {{ color: {TEXT_COLOR}; }}      ->  color: #d6d6d6
    QProgressBar {{ color: {{TEXT_COLOR}}; }}    ->  color: {TEXT_COLOR}

Both are valid Python. The second raises nothing, and Qt discards a rule it
cannot parse without reporting it, so the widget simply renders unstyled. These
tests exist for that second line.

Deliberately *not* a snapshot of the rendered text. Pinning all 68 values caught
accidental colour edits -- a risk that predates the f-strings and that nothing
else was testing -- at the cost of regenerating a table after every intentional
restyle. The hazard the f-strings actually introduced is structural, so it is
checked structurally and there is nothing to keep up to date.

``stylesheets`` is loaded without executing ``fibsem/ui/__init__.py``, which
imports the widget layer and so needs napari/PyQt5. CI installs only ``.[test]``,
so importing it the ordinary way would skip these in exactly the job they are
meant to protect. Stubbing the package also pins a property worth keeping:
``stylesheets`` stays a leaf module that costs nothing to import.
"""

import contextlib
import importlib
import re
import sys
import types
from pathlib import Path

import pytest

UI_DIR = Path(__file__).resolve().parents[2] / "fibsem" / "ui"

# `{BORDER_COLOR}` surviving into the output means a brace pair was doubled that
# should have been a substitution.
_UNSUBSTITUTED = re.compile(r"\{[A-Z_][A-Z_0-9]*\}")

# Stands in for the absolute icons directory, which is checkout-specific.
_ICONS_PLACEHOLDER = "<ICONS_DIR>"

# Anything else absolute would mean a sheet renders differently on two machines.
_ABSOLUTE_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|/(?:Users|home|opt|tmp|private|var)/)")


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


@pytest.fixture(scope="module")
def rendered():
    """{name: value} for every public module-level string constant.

    Two sheets embed ``_ICONS_DIR`` -- an absolute path -- in ``url(...)`` rules
    for the spinbox and combobox arrows. Substituting a placeholder is what lets
    the absolute-path check below mean "no path that should not be here".
    """
    with _leaf_import() as stylesheets:
        icons_dir = stylesheets._ICONS_DIR
        return {
            name: getattr(stylesheets, name).replace(icons_dir, _ICONS_PLACEHOLDER)
            for name in dir(stylesheets)
            if not name.startswith("_") and isinstance(getattr(stylesheets, name), str)
        }


def test_the_stylesheets_import_and_render(rendered):
    """Smoke test, and the guard on the two below.

    A brace left undoubled around anything Python can parse as an expression
    fails here, at import -- that is the loud half of the hazard. It also stops
    the checks below passing vacuously: they iterate over this mapping, so an
    empty one would make them both trivially true.
    """
    # Loose on purpose: this is here to catch an import that produced nothing,
    # not to pin the count, which would need editing every time a sheet is
    # added or retired.
    assert len(rendered) > 40, f"only {len(rendered)} constants -- did the import half-fail?"
    assert rendered["NAPARI_STYLE"].strip().startswith("QWidget")
    assert "#" in rendered["BORDER_COLOR"]


def test_no_unsubstituted_token_placeholders(rendered):
    """The quiet half: a doubled brace where a substitution was meant."""
    offenders = {
        name: sorted(set(found))
        for name, value in rendered.items()
        if (found := _UNSUBSTITUTED.findall(value))
    }
    assert offenders == {}, (
        "a brace pair was doubled where a token substitution was intended, so "
        f"the placeholder reached the rendered QSS: {offenders}"
    )


def test_no_rendered_sheet_carries_an_absolute_path(rendered):
    """A sheet must mean the same thing on every checkout.

    ``_ICONS_DIR`` is normalised away in the fixture; anything else absolute is
    a path that leaked in from the machine the tests happen to run on.
    """
    offenders = {
        name: _ABSOLUTE_PATH.search(value).group()
        for name, value in rendered.items()
        if _ABSOLUTE_PATH.search(value)
    }
    assert offenders == {}, (
        "a rendered stylesheet embeds an absolute path, which differs between "
        f"checkouts: {offenders}. Normalise it in the fixture as _ICONS_DIR is."
    )
