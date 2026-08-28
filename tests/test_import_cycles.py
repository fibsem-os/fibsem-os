"""Import-order regressions.

`fibsem.microscope` imports `fibsem.fm.microscope` at module level, which runs
`fibsem/fm/__init__.py` and lands in `fibsem.fm.acquisition`. So anything
`fm.acquisition` imports at module scope is imported while `fibsem.microscope` is
only half-built, whenever the chain *started* somewhere that pulls in
`fibsem.microscope`.

That became a real failure when `fm.acquisition` began importing
`fibsem.imaging.tiling.geometry`: importing any `fibsem.imaging` submodule runs
`fibsem/imaging/__init__.py`, which star-imports `fibsem.imaging.spot`, which
imported `FibsemMicroscope` -- a name defined *after* the line that started the
chain. The AutoLamella UI stopped importing at all.

The fix was to make `spot.py` import it for type-checking only, which it always
was: one annotation, in a module that already has `from __future__ import
annotations`. That makes the whole `fibsem.imaging` package microscope-free.

Both properties are pinned below: the entry points, and the package boundary that
keeps them working.

Run in subprocesses -- import order cannot be tested in-process, because whatever a
previous test imported has already populated sys.modules.
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

import fibsem

# Entry points that reach fibsem.microscope on the way in. The AutoLamella UI is the
# one users actually hit, but it needs the `ui` extra, so it is not what CI relies on.
#
# `fibsem.fm.acquisition` is in this list precisely because it passes even when the
# cycle is present: it imports `fibsem.microscope` fully first, so it cannot observe
# the half-initialised state. A check written against it reports success while the
# application is broken, which is how this shipped.
ENTRY_POINTS = [
    "fibsem.milling.tasks",
    "fibsem.milling",
    "fibsem.microscope",
    "fibsem.fm.acquisition",
]


# The subprocess has to import the *same* checkout pytest itself imported, and by
# default it does not. Two things conspire: `tests/conftest.py` chdirs into a
# `tmp_path`, so `python -c` puts that empty directory on `sys.path[0]` rather than the
# repo; and an editable install then resolves `fibsem` to whichever checkout it was
# installed from. In a git worktree those are different trees, so every guard below
# silently measured the *other* checkout -- passing or failing according to what that
# one happened to have checked out, which is not a property of the code under test.
_REPO_ROOT = Path(fibsem.__file__).resolve().parent.parent


def _run(code: str):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(_REPO_ROOT), env.get("PYTHONPATH", "")) if part
    )
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )


def test_the_subprocess_measures_the_checkout_under_test():
    """Without this, a green run below proves nothing.

    Every other test in this file asks a subprocess what it imported. If that
    subprocess resolves a *different* `fibsem` -- the one the editable install points
    at, rather than the one pytest loaded -- then the guards describe some other
    checkout. That is not hypothetical: it happened in a worktree, where the milling
    guard flipped red because the sibling checkout had switched branches, with nothing
    changed here at all.
    """
    result = _run("import fibsem; print(fibsem.__file__)")
    assert result.returncode == 0, result.stderr[-1500:]
    measured = Path(result.stdout.strip().splitlines()[-1]).resolve()
    assert measured == Path(fibsem.__file__).resolve(), (
        f"the subprocess imported {measured}, but pytest imported "
        f"{Path(fibsem.__file__).resolve()} -- every guard below is describing the "
        "wrong tree"
    )


@pytest.mark.parametrize("module", ENTRY_POINTS)
def test_entry_point_imports_without_a_circular_import(module):
    result = _run(f"import {module}")
    assert "circular import" not in result.stderr, (
        f"importing {module} hit a circular import:\n{result.stderr[-1500:]}"
    )
    assert result.returncode == 0, (
        f"importing {module} failed:\n{result.stderr[-1500:]}"
    )


def test_imaging_does_not_pull_in_the_microscope():
    """The property that keeps the cycle closed.

    `fibsem/imaging/__init__.py` star-imports its submodules, so one runtime
    `fibsem.microscope` import anywhere under `fibsem.imaging` makes every
    `fibsem.imaging.*` import drag the microscope in -- and the microscope imports
    `fibsem.fm.microscope`, which reaches back into `fibsem.imaging`.
    """
    result = _run(
        "import sys, warnings; warnings.filterwarnings('ignore')\n"
        "import fibsem.imaging\n"
        "print('fibsem.microscope' in sys.modules)"
    )
    assert result.returncode == 0, result.stderr[-1500:]
    assert result.stdout.strip().splitlines()[-1] == "False", (
        "importing fibsem.imaging now pulls in fibsem.microscope again. Something "
        "under fibsem/imaging/ imports it at runtime; if it is only needed for an "
        "annotation, put it under TYPE_CHECKING as fibsem/imaging/spot.py does."
    )


def test_spot_imports_the_microscope_for_typing_only():
    """Stated statically, so the reason survives even if the runtime check moves.

    `FibsemMicroscope` appears once in spot.py, as a parameter annotation, and the
    module has `from __future__ import annotations` -- so it never needs the real
    object at runtime.
    """
    source = (Path(fibsem.__file__).parent / "imaging" / "spot.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)

    module_level = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "fibsem.microscope" not in module_level, (
        "spot.py imports fibsem.microscope at runtime. It is used only in an "
        "annotation; keep it under TYPE_CHECKING."
    )


def test_milling_does_not_pull_in_the_microscope():
    """The same property, for `fibsem.milling` -- established by FIB-797.

    `fibsem/milling/__init__.py` imports `base` and `core`, so one runtime
    `fibsem.microscope` import in either makes *every* `fibsem.milling.*` import drag
    the microscope in. That is a cycle rather than merely slow, because
    `fibsem.microscope` imports `fibsem.milling.progress` to type
    `milling_progress_signal`: the microscope module is only half-built when the milling
    package runs, and `FibsemMicroscope` is not bound yet.
    """
    result = _run(
        "import sys, warnings; warnings.filterwarnings('ignore')\n"
        "import fibsem.milling\n"
        "print('fibsem.microscope' in sys.modules)"
    )
    assert result.returncode == 0, result.stderr[-1500:]
    assert result.stdout.strip().splitlines()[-1] == "False", (
        "importing fibsem.milling now pulls in fibsem.microscope again. Something "
        "under fibsem/milling/ imports it at runtime; if it is only needed for an "
        "annotation, put it under TYPE_CHECKING as fibsem/milling/base.py does."
    )


@pytest.mark.parametrize(
    "module_path", [("milling", "base.py"), ("milling", "core.py")]
)
def test_milling_imports_the_microscope_for_typing_only(module_path):
    """Stated statically, so the reason survives even if the runtime check moves.

    `FibsemMicroscope` appears in both files only as a parameter annotation, and both
    carry `from __future__ import annotations` -- so neither needs the real object.
    """
    source = (Path(fibsem.__file__).parent.joinpath(*module_path)).read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)

    module_level = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "fibsem.microscope" not in module_level, (
        f"{module_path[-1]} imports fibsem.microscope at runtime. It is used only in "
        "annotations; keep it under TYPE_CHECKING."
    )
    assert "fibsem.acquire" not in module_level, (
        f"{module_path[-1]} imports fibsem.acquire at runtime, and fibsem.acquire "
        "imports fibsem.microscope -- which is the other half of the same cycle."
    )
