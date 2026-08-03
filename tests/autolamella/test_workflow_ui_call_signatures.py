"""Every call into the workflow UI helpers must match its signature (FIB-454).

`MillCoincidentTask` passed `coincidence_milling=True` to `ask_user`, which has never had
that parameter. It raised `TypeError` at the hand-off to the operator — after moving the
stage and objective, applying the milling config and acquiring — and survived because the
workflow path is rarely exercised: in a two-day log covering ten coincidence mills, the
task never ran once.

These helpers are called from workflow tasks that need hardware and a GUI, so nothing
imports them under test and a plain signature mismatch is invisible until an operator hits
it mid-run. Checking the call sites statically costs nothing and catches the whole class.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import List, Tuple

import pytest

from fibsem.applications.autolamella.workflows import ui as workflow_ui

# Helpers worth checking: module-level functions in workflows/ui.py that tasks call with
# keywords. Resolved dynamically so a new helper is covered without touching this test.
_HELPERS = {
    name: obj
    for name, obj in vars(workflow_ui).items()
    if inspect.isfunction(obj) and not name.startswith("_")
    and obj.__module__ == workflow_ui.__name__
}

_TASKS_DIR = Path(workflow_ui.__file__).parent / "tasks"


def _call_sites() -> List[Tuple[Path, int, str, ast.Call]]:
    """Every call to a workflows.ui helper made from a workflow task module."""
    sites = []
    for path in sorted(_TASKS_DIR.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover - a broken file is another test's problem
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _called_helper_name(node.func)
            if name in _HELPERS:
                sites.append((path, node.lineno, name, node))
    return sites


def _called_helper_name(func: ast.expr):
    """The helper being called, or None if this is not a direct call to one.

    `self.update_status_ui(...)` is a *method* that wraps the module-level function of the
    same name with different parameter names (`message=` vs `msg=`). Matching on the
    attribute alone reported it as a signature mismatch when it is simply a different
    callable, so calls on self/cls are excluded.
    """
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        value = func.value
        if isinstance(value, ast.Name) and value.id in {"self", "cls"}:
            return None
        return func.attr
    return None


def test_the_helpers_are_discoverable():
    """Guard against the discovery above silently finding nothing, which would make every
    other test in this file vacuously pass."""
    assert "ask_user" in _HELPERS, f"ask_user not discovered; found {sorted(_HELPERS)}"
    assert _call_sites(), "no workflow-ui call sites found — discovery is broken"


@pytest.mark.parametrize(
    "path,lineno,name,call",
    [pytest.param(*s, id=f"{s[0].name}:{s[1]}:{s[2]}") for s in _call_sites()],
)
def test_call_site_matches_signature(path, lineno, name, call):
    """Bind each call's arguments against the real signature.

    Positional args are passed as placeholders — only arity and keyword *names* matter
    here, not types. `bind_partial` because a task may legitimately rely on defaults.
    """
    signature = inspect.signature(_HELPERS[name])
    args = [object()] * len(call.args)
    if any(isinstance(a, ast.Starred) for a in call.args):
        pytest.skip("call uses *args; arity cannot be checked statically")
    kwargs = {kw.arg: object() for kw in call.keywords if kw.arg is not None}
    if any(kw.arg is None for kw in call.keywords):
        pytest.skip("call uses **kwargs; keywords cannot be checked statically")

    try:
        signature.bind_partial(*args, **kwargs)
    except TypeError as exc:
        pytest.fail(
            f"{path.name}:{lineno} calls {name}({', '.join(sorted(kwargs))}) "
            f"but the signature is {name}{signature} — {exc}"
        )
