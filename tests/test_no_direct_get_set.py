"""No new direct ``microscope.get("key")`` / ``microscope.set("key", ...)`` calls.

The string-key ``get``/``set`` is being deprecated in favour of the devices
(``microscope.beams[beam_type].parameters[...]``, ``microscope.stage``, ...) and the
named wrappers (``get_working_distance``, ``set_preset``, ...). This keeps the number of
callers going down: code outside the microscope classes and the device router
uses those instead.
"""

import ast
from pathlib import Path
from typing import Iterator, List

ROOT = Path(__file__).resolve().parents[1]

SCANNED = ["fibsem", "example", "docs", "scripts"]

# The old API itself, its backends and the router that sends its keys to devices.
ALLOWED = [
    "fibsem/microscope.py",
    "fibsem/microscopes/",
    "fibsem/devices/",
]


def _receiver_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _python_files() -> Iterator[Path]:
    for top in SCANNED:
        base = ROOT / top
        if base.is_dir():
            yield from sorted(base.rglob("*.py"))


def _direct_calls(path: Path) -> List[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []
    found = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("get", "set")
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            continue
        receiver = _receiver_name(node.func.value)
        if receiver.lower().endswith("microscope") or receiver == "scope":
            found.append(
                f"{path.relative_to(ROOT).as_posix()}:{node.lineno}: "
                f'{receiver}.{node.func.attr}("{node.args[0].value}", ...)'
            )
    return found


def test_no_direct_get_set_calls():
    offenders = []
    for path in _python_files():
        rel = path.relative_to(ROOT).as_posix()
        if any(rel == a or rel.startswith(a) for a in ALLOWED):
            continue
        offenders.extend(_direct_calls(path))
    assert not offenders, (
        "Use the device or the named wrapper instead of the string-key get/set:\n"
        + "\n".join(offenders)
    )
