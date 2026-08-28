"""The core of fibsem must import without the `ui` extra installed.

`pip install fibsem` gets napari, PyQt5 and qtawesome only if you ask for `.[ui]`.
CI asks for `.[test]`. So a module-level import of one of those from code on a non-UI
path does not degrade gracefully -- it raises at *collection* time, which reads as a
pile of unrelated tests failing rather than as one bad import.

That has happened: FIB-390 hoisted a deferred `matplotlib_scalebar` import into a
module header during an extraction and broke every CI job on every Python version.
The guard written in response lived in `test_tiling_package.py` and was parametrised
over the tiling package alone, so it protected the module that had just been fixed and
nothing else. `applications/autolamella/tools/reporting.py` then acquired the same
module-level import and kept it, because nothing was looking.

`matplotlib_scalebar` has since been promoted to a core dependency (FIB-562) -- it was
the wrong thing to gate, being a matplotlib plugin the non-UI plotting code needed --
so that particular breakage cannot recur. The rule it motivated still holds for the
packages that really are UI-only.

This file is the general version. It walks the tree rather than naming modules, so a
new file on a non-UI path is covered the day it is written.

The pattern being protected is deferral, not avoidance -- an optional dependency
imported inside the one function that needs it is correct and is what these tests
allow. Only module scope is checked.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Set, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIBSEM = REPO_ROOT / "fibsem"

# The distributions declared in the `ui` extra of pyproject.toml, plus the two that
# arrive transitively with napari and are equally absent without it.
#
# `matplotlib_scalebar` is deliberately NOT here. It used to be, and it was the one
# package on this list that half the core drew on -- tile plotting, pattern plotting
# and the report figures all wanted a scalebar, so each deferred the import into a
# function body purely to satisfy this rule. It is a core dependency now, so those
# deferrals are a choice rather than a requirement.
UI_ONLY_PACKAGES = (
    "napari",
    "PyQt5",
    "pyqt5",
    "qtawesome",
    "superqt",
    "vispy",
    "qtpy",
)

# Importing anything under fibsem.ui pulls Qt in transitively -- `fibsem/ui/__init__.py`
# imports every widget eagerly (FIB-352), so even a leaf module costs the whole tree.
UI_PACKAGE_PREFIX = "fibsem.ui"

# Modules that violate the rule and are not being fixed here. Each entry needs a
# reason; an allow-list without one is just a disabled test.
KNOWN_VIOLATIONS: set = set()


def _is_ui_path(rel: str) -> bool:
    return rel.startswith("ui/") or "/ui/" in rel


def _core_modules() -> List[str]:
    """Every non-UI, non-test module under fibsem/, relative to the package root."""
    out = []
    for path in sorted(FIBSEM.rglob("*.py")):
        rel = path.relative_to(FIBSEM).as_posix()
        parts = rel.split("/")
        if "__pycache__" in parts or "tests" in parts or "test" in parts:
            continue
        if path.name.startswith("test"):
            continue
        if _is_ui_path(rel):
            continue
        out.append(rel)
    return out


CORE_MODULES = _core_modules()


def _module_level_imports(rel: str) -> Set[str]:
    """Imports at module scope only -- the ones that cost something to `import`.

    Anything nested inside a function or an `if TYPE_CHECKING:` block is excluded on
    purpose: that is the pattern this file protects, not a violation of it.
    """
    source = (FIBSEM / rel).read_text(encoding="utf-8", errors="ignore")
    names: Set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
    return names


def _offenders(rel: str) -> Tuple[List[str], List[str]]:
    imports = _module_level_imports(rel)
    ui_dists = sorted(m for m in imports if m.split(".")[0] in UI_ONLY_PACKAGES)
    ui_pkg = sorted(
        m
        for m in imports
        if m == UI_PACKAGE_PREFIX or m.startswith(UI_PACKAGE_PREFIX + ".")
    )
    return ui_dists, ui_pkg


def test_the_scan_actually_found_modules():
    """A walk that silently matches nothing would make every test below vacuous."""
    assert len(CORE_MODULES) > 100, f"only found {len(CORE_MODULES)} core modules"


@pytest.mark.parametrize("rel", CORE_MODULES)
def test_no_module_level_import_of_a_ui_only_dependency(rel):
    if rel in KNOWN_VIOLATIONS:
        pytest.skip("known violation, see KNOWN_VIOLATIONS")
    offending, _ = _offenders(rel)
    assert not offending, (
        f"fibsem/{rel} imports {offending} at module level. Those live in the `ui` "
        f"extra, which CI does not install -- defer the import into the function that "
        f"needs it, as fibsem/imaging/tiling/plotting.py does."
    )


@pytest.mark.parametrize("rel", CORE_MODULES)
def test_no_module_level_import_of_the_ui_package(rel):
    if rel in KNOWN_VIOLATIONS:
        pytest.skip("known violation, see KNOWN_VIOLATIONS")
    _, offending = _offenders(rel)
    assert not offending, (
        f"fibsem/{rel} imports {offending} at module level. fibsem.ui pulls in Qt "
        f"transitively, so importing it from a non-UI path costs the whole widget "
        f"tree -- defer it, or invert the dependency so the UI calls this module."
    )


@pytest.mark.parametrize("rel", sorted(KNOWN_VIOLATIONS))
def test_known_violations_are_still_violations(rel):
    """Delete the entry when the module is fixed, rather than leaving it to rot.

    An allow-list nobody prunes eventually hides a regression behind an entry that
    describes something already repaired.
    """
    assert (FIBSEM / rel).exists(), f"{rel} is gone -- drop it from KNOWN_VIOLATIONS"
    ui_dists, ui_pkg = _offenders(rel)
    assert ui_dists or ui_pkg, (
        f"fibsem/{rel} no longer imports UI code at module level -- "
        f"drop it from KNOWN_VIOLATIONS"
    )
