"""Imports of fibsem inside functions don't grow.

An import inside a function usually works around an import cycle, or is there by
habit. Either way it hides a dependency from the top of the module, so the count
per module is held to what ``local_imports_allowed.txt`` lists. A new one is a
decision: break the cycle (move what both modules need to where both can import
it), or add the module to the list with its new count. When a module needs fewer,
lower its number. Imports under ``if TYPE_CHECKING:`` don't run, so they don't
count.
"""

import ast
from collections import Counter
from pathlib import Path

import fibsem

FIBSEM = Path(fibsem.__file__).parent
ALLOWED = Path(__file__).with_name("local_imports_allowed.txt")


def _is_fibsem(node) -> bool:
    if isinstance(node, ast.Import):
        return any(
            a.name == "fibsem" or a.name.startswith("fibsem.") for a in node.names
        )
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        return node.level > 0 or module == "fibsem" or module.startswith("fibsem.")
    return False


def local_imports(path: Path) -> int:
    """How many fibsem imports in *path* run inside a function."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    type_checking = {
        id(child)
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.dump(node.test)
        for child in ast.walk(node)
    }
    seen = set()
    for function in ast.walk(tree):
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            for node in ast.walk(function):
                if id(node) not in type_checking and _is_fibsem(node):
                    seen.add(id(node))
    return len(seen)


def counts() -> Counter:
    found = Counter()
    for path in sorted(FIBSEM.rglob("*.py")):
        n = local_imports(path)
        if n:
            found[path.relative_to(FIBSEM.parent).as_posix()] = n
    return found


def allowed() -> Counter:
    listed = Counter()
    for line in ALLOWED.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            path, n = line.split()
            listed[path] = int(n)
    return listed


def test_no_module_imports_fibsem_inside_more_functions_than_listed():
    limit = allowed()
    over = [
        f"{path}: {n} (listed {limit[path]})"
        for path, n in sorted(counts().items())
        if n > limit[path]
    ]
    assert over == [], (
        "fibsem imported inside a function beyond tests/local_imports_allowed.txt; "
        "break the cycle, or list the new count:\n" + "\n".join(over)
    )
