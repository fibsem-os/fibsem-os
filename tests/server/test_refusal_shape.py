"""A refusal is one shape from the context to the route: a `Refusal`."""

import ast
from pathlib import Path

import pytest

import fibsem
from fibsem.server.responses import Refusal, refusal_of

ROOT = Path(fibsem.__file__).parent
CONTEXTS = [
    "server/app_context.py",
    "applications/autolamella/server/context.py",
    "applications/volume/server.py",
]


def test_an_applied_edit_is_its_own_answer():
    result = {"applied": True, "changes": []}
    assert refusal_of(result) is result


def test_a_stale_edit_is_a_stale_config_refusal():
    refusal = refusal_of({"applied": False, "stale": True})
    assert isinstance(refusal, Refusal)
    assert (refusal.error_type, refusal.status) == ("stale_config", 409)


def test_an_invalid_patch_carries_its_path():
    refusal = refusal_of({"applied": False, "invalid_patch": "no", "path": "a.b"})
    assert (refusal.error_type, refusal.status) == ("invalid_patch", 422)
    assert refusal.detail == {
        "error_type": "invalid_patch",
        "message": "no",
        "path": "a.b",
    }


def test_an_error_keeps_its_type_and_names():
    refusal = refusal_of(
        {
            "applied": False,
            "error": "running",
            "error_type": "task_running",
            "task_names": ["a"],
        }
    )
    assert (refusal.error_type, refusal.status) == ("task_running", 409)
    assert refusal.names == {"task_names": ["a"]}
    assert refusal_of({"applied": False, "error": "gone"}).status == 404


def test_a_refusal_passes_through():
    refusal = Refusal("invalid", "no")
    assert refusal_of(refusal) is refusal


@pytest.mark.parametrize("path", CONTEXTS)
def test_no_context_writes_a_refusal_by_hand(path):
    """A context refuses with `refused` (or `refusal_of`), never a dict the
    route would have to recognise."""
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    by_hand = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key in node.keys
        if isinstance(key, ast.Constant)
        and key.value in ("error_type", "stale", "invalid_patch")
    ]
    assert not by_hand, f"{path}: refusal dicts at lines {by_hand}"
