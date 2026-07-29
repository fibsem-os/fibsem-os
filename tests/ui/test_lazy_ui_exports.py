"""The ``fibsem.ui`` package must not drag napari in just to be imported.

Its ``__init__`` used to import all eight widgets eagerly, one of which imports napari, so
reaching *any* module under ``fibsem.ui`` — including Qt-free leaves — pulled napari into the
graph. The exports are resolved lazily now (PEP 562).

The first four tests import nothing but ``fibsem.ui`` itself and so **run on CI**, which
installs neither PyQt5 nor napari — that is the regression worth guarding. The two that need a
real widget class are skipped there, like every other UI test. The skip is a decorator rather
than a module-level ``importorskip`` precisely so the CI-relevant tests still run.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

try:  # only the last two tests need Qt
    import PyQt5  # noqa: F401

    HAS_QT = True
except ImportError:
    HAS_QT = False

needs_qt = pytest.mark.skipif(not HAS_QT, reason="needs PyQt5 (not installed on CI)")


def _probe(body: str) -> str:
    """Run *body* in a fresh interpreter and return its last stdout line.

    Runs from the repo root so ``fibsem`` resolves to *this* checkout — an editable install
    otherwise maps it to the main working copy, and the probe would silently test that tree
    instead. Only the last line is returned because importing ``fibsem`` prints a
    configuration banner.
    """
    result = subprocess.run(
        [sys.executable, "-c", body],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT), "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, f"probe failed:\n{result.stdout}\n{result.stderr}"
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert lines, f"probe produced no output:\n{result.stderr}"
    return lines[-1].strip()


def test_importing_the_package_needs_neither_napari_nor_qt():
    """`import fibsem.ui` must be free of both — the CI-relevant guarantee."""
    out = _probe(
        "import sys, fibsem.ui;"
        "print('napari' in sys.modules, 'PyQt5' in sys.modules)"
    )
    assert out == "False False", f"expected no napari and no Qt, got {out!r}"


def test_unknown_attribute_raises_attribute_error():
    """A typo must not surface as an ImportError from some unrelated widget."""
    out = _probe(
        "import fibsem.ui\n"
        "try:\n"
        "    fibsem.ui.NoSuchWidget\n"
        "except AttributeError:\n"
        "    print('AttributeError')\n"
    )
    assert out == "AttributeError"


def test_dir_still_advertises_the_exports():
    out = _probe(
        "import fibsem.ui;"
        "d = dir(fibsem.ui);"
        "print(all(n in d for n in ('FibsemMinimapWidget', 'MillingTaskViewerWidget', 'DETECTION_AVAILABLE')))"
    )
    assert out == "True"


def test_detection_available_is_a_plain_bool():
    """Computed on first access now; it must still be a bool, not a lazy proxy."""
    out = _probe("from fibsem.ui import DETECTION_AVAILABLE as d; print(type(d).__name__)")
    assert out == "bool"


@needs_qt
def test_reexport_resolves_to_the_class():
    out = _probe(
        "import inspect;"
        "from fibsem.ui import FibsemMovementWidget as W;"
        "print(inspect.isclass(W))"
    )
    assert out == "True"


@needs_qt
def test_reexport_is_the_class_even_when_the_submodule_shadows_it():
    """Each widget class shares its name with the module that defines it.

    Importing ``fibsem.ui.FibsemMovementWidget`` binds that *module* onto the package under
    the same name, which bypasses PEP 562's ``__getattr__`` entirely — the attribute is no
    longer missing. The eager ``__init__`` used to overwrite it with the class; the lazy one
    restores that guarantee by unwrapping the module on access. Without it, callers get a
    module and ``FibsemMovementWidget(...)`` fails with "module is not callable".
    """
    out = _probe(
        "import inspect\n"
        "import fibsem.ui.FibsemMovementWidget  # binds the MODULE onto the package\n"
        "from fibsem.ui import FibsemMovementWidget as W\n"
        "print(inspect.isclass(W))\n"
    )
    assert out == "True", "a shadowing submodule leaked through the re-export"
