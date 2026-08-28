"""Every instruction names something that exists, and nothing names nothing.

Three of the five used to point at menus that were never there — "Connection ->
Connect to Microscope" and "Experiment -> Add Lamella" name no menu at all, and the
File entries are "New Experiment" and "Load Experiment", not "Create / Load
Experiment". Guidance that sends someone hunting for a menu is worse than silence,
and nothing failed when it drifted, because a string cannot be wrong on its own.

So the menu paths are checked against the menus the window actually builds, and the
table is checked against the states that actually render it. Both read the source:
`AutoLamellaSingleWindowUI` builds a napari viewer and a vispy quad view and cannot
be constructed under the offscreen platform.
"""

from __future__ import annotations

import ast
import inspect
import os
import re
import textwrap

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.ui.AutoLamellaMainUI import (  # noqa: E402
    AutoLamellaSingleWindowUI,
)
from fibsem.applications.autolamella.ui.AutoLamellaUI import INSTRUCTIONS  # noqa: E402

# "(File -> Load Protocol)" / "(File → Load Protocol)" -- the menu, then the entry.
MENU_PATH = re.compile(r"\(([A-Z][A-Za-z ]*?)\s*(?:->|→)\s*([^)]+)\)")


def _source(func) -> ast.Module:
    return ast.parse(textwrap.dedent(inspect.getsource(func)))


def _first_string_args(func, name: str) -> set[str]:
    """Every literal first argument to `name(...)` or `.name(...)` in the source.

    Both call shapes matter here: menus are added with `menu_bar.addMenu("File")`
    (an attribute) and actions are built with `QAction("Load Protocol", self)`
    (a plain name).
    """
    out = set()
    for node in ast.walk(_source(func)):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        called = (
            node.func.attr
            if isinstance(node.func, ast.Attribute)
            else getattr(node.func, "id", None)
        )
        arg = node.args[0]
        if (
            called == name
            and isinstance(arg, ast.Constant)
            and isinstance(arg.value, str)
        ):
            out.add(arg.value)
    return out


def test_every_menu_path_names_a_real_menu():
    menus = _first_string_args(AutoLamellaSingleWindowUI._create_menu_bar, "addMenu")
    actions = _first_string_args(AutoLamellaSingleWindowUI._create_menu_bar, "QAction")
    assert "File" in menus, "sanity: the menu bar builder was not read correctly"
    assert actions, "sanity: no QAction labels found in the menu bar builder"

    for key, msg in INSTRUCTIONS.items():
        for menu, entry in MENU_PATH.findall(msg):
            assert menu in menus, (
                f"INSTRUCTIONS[{key!r}] sends the user to a {menu!r} menu; "
                f"the window builds {sorted(menus)}"
            )
            assert entry.strip() in actions, (
                f"INSTRUCTIONS[{key!r}] names a {entry.strip()!r} entry that no "
                f"QAction in the menu bar is called"
            )


def test_no_instruction_goes_unused():
    """A message no state can reach is a message nobody maintains.

    Three ready-state entries outlived the code that showed them, still describing
    a Waffle-method flow that no longer renders them.
    """
    used = {
        node.slice.value
        for node in ast.walk(_source(AutoLamellaSingleWindowUI._update_instructions))
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "INSTRUCTIONS"
        and isinstance(node.slice, ast.Constant)
    }

    assert used, "sanity: no INSTRUCTIONS lookups found in _update_instructions"
    assert set(INSTRUCTIONS) == used, (
        f"unused: {sorted(set(INSTRUCTIONS) - used)}; "
        f"missing: {sorted(used - set(INSTRUCTIONS))}"
    )


def test_instructions_say_what_to_do_without_asking_permission():
    """House style: the UI is not requesting a favour, and it is not shouting."""
    for key, msg in INSTRUCTIONS.items():
        assert not msg.lower().startswith("please"), f"INSTRUCTIONS[{key!r}]"
        assert msg == msg[0] + msg[1:].replace("  ", " "), f"INSTRUCTIONS[{key!r}]"
        assert msg.endswith("."), f"INSTRUCTIONS[{key!r}] should be a sentence"
