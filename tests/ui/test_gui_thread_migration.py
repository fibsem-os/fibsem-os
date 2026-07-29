"""Guard the GUI's background-threading migration onto ``FunctionWorker``.

``fibsem.ui.qt.threading`` is tested on its own in ``test_qt_threading.py``. What this file
protects is the *migration*: every GUI background job goes through ``FunctionWorker`` rather
than a bare ``threading.Thread``.

That matters because a raw thread is invisible to Qt. It swallows exceptions into the
interpreter's default handler instead of surfacing them, and it offers no signal that is
marshalled back onto the GUI thread — which is how widget updates end up being made from a
worker thread. ``FunctionWorker`` logs every failure and delivers ``returned`` / ``errored`` /
``finished`` on the thread that created it.

Deliberately *not* covered: the driver layer (``fibsem/microscope.py``, ``fibsem/fm/microscope.py``)
and the workflow hooks (``workflows/tasks/hooks.py``). Those are Qt-free by design, and
``FunctionWorker`` is a ``QObject`` — pulling it in there would drag Qt into non-GUI code.

These checks read source rather than importing it, deliberately: CI installs neither PyQt5 nor
napari, and ``fibsem/ui/__init__.py`` eagerly imports every widget, so *any* import under
``fibsem.ui`` pulls in napari and would skip the whole file there. Reading the files keeps the
guard running on CI, where it is most useful. ``FunctionWorker``'s runtime behaviour — including
the ``start`` / ``is_alive`` / ``join`` surface these call sites rely on — is covered by
``test_qt_threading.py``, which is import-guarded and runs locally.
"""
import ast
from pathlib import Path
from typing import List, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Directories whose background work must be Qt-observable.
GUI_ROOTS = [
    REPO_ROOT / "fibsem" / "ui",
    REPO_ROOT / "fibsem" / "correlation" / "ui",
    REPO_ROOT / "fibsem" / "applications" / "autolamella" / "ui",
]

# The FunctionWorker implementation itself owns the one legitimate raw thread.
ALLOWED = {REPO_ROOT / "fibsem" / "ui" / "qt" / "threading.py"}

# Widgets whose background jobs were migrated off raw threads; each must still route
# through FunctionWorker, so a half-revert is caught.
MIGRATED_MODULES = [
    "fibsem/ui/FMAcquisitionWidget.py",
    "fibsem/ui/FibsemMinimapWidget.py",
    "fibsem/ui/widgets/milling_widget.py",
    "fibsem/ui/widgets/fluorescence_control_widget.py",
    "fibsem/ui/widgets/fluorescence_coincidence_viewer_widget.py",
    "fibsem/applications/autolamella/ui/AutoLamellaUI.py",
]


def _gui_python_files() -> List[Path]:
    files = []
    for root in GUI_ROOTS:
        if root.exists():
            files.extend(p for p in root.rglob("*.py") if p not in ALLOWED)
    return sorted(files)


def _raw_thread_constructions(path: Path) -> List[Tuple[int, str]]:
    """Return (lineno, source-ish) for every ``threading.Thread(...)`` construction."""
    tree = ast.parse(path.read_text(), filename=str(path))
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # threading.Thread(...)
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "Thread"
            and isinstance(func.value, ast.Name)
            and func.value.id == "threading"
        ):
            hits.append((node.lineno, "threading.Thread"))
        # from threading import Thread; Thread(...)
        elif isinstance(func, ast.Name) and func.id == "Thread":
            hits.append((node.lineno, "Thread"))
    return hits


def test_no_raw_threads_in_gui_code():
    """No GUI module may start a bare thread — background work goes through FunctionWorker."""
    offenders = []
    for path in _gui_python_files():
        for lineno, what in _raw_thread_constructions(path):
            offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {what}(...)")
    assert not offenders, (
        "GUI background work must use fibsem.ui.qt.threading.FunctionWorker, which logs "
        "failures and delivers its signals on the GUI thread:\n  " + "\n  ".join(offenders)
    )


@pytest.mark.parametrize("relpath", MIGRATED_MODULES)
def test_migrated_modules_use_function_worker(relpath: str):
    """Each migrated widget still imports and uses FunctionWorker."""
    src = (REPO_ROOT / relpath).read_text()
    assert "from fibsem.ui.qt.threading import" in src, f"{relpath} lost its FunctionWorker import"
    assert "FunctionWorker(" in src, f"{relpath} no longer constructs a FunctionWorker"


def test_non_gui_layers_are_left_alone():
    """FunctionWorker is a QObject, so Qt-free layers must keep their plain threads."""
    for relpath in (
        "fibsem/microscope.py",
        "fibsem/fm/microscope.py",
        "fibsem/applications/autolamella/workflows/tasks/hooks.py",
    ):
        src = (REPO_ROOT / relpath).read_text()
        assert "FunctionWorker" not in src, (
            f"{relpath} is Qt-free by design; importing FunctionWorker would drag Qt into it"
        )
