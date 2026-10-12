"""The colour tokens import without Qt, and the UI still sees the same ones (FIB-1287).

Headless code -- a saved figure, the image export -- needs the palette, and used
to copy colours out of it by hand because importing anything under ``fibsem.ui``
loads the widgets.
"""

import subprocess
import sys

import pytest


def test_the_tokens_import_without_loading_qt():
    # A fresh interpreter: anything this test session already imported would hide it.
    code = (
        "import sys; import fibsem.theme.tokens as t; "
        "assert t.CANVAS_BG; "
        "qt = sorted(m for m in sys.modules if m.split('.')[0] in "
        "('PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'qtpy', 'napari') or m == 'fibsem.ui'); "
        "print(','.join(qt))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "", f"importing the tokens loaded {out.stdout}"


def test_the_ui_tokens_are_the_same_names_and_values():
    pytest.importorskip("PyQt5")
    import fibsem.theme.tokens as theme
    import fibsem.ui.tokens as ui

    public = {n for n in vars(theme) if not n.startswith("_")}
    assert {n for n in public if getattr(ui, n, None) is getattr(theme, n)} == public
