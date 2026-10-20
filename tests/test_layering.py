"""The core never imports the UI.

Headless runs, the App Server and the device server use the core with no Qt. A Qt
import inside it (even a guarded one) ties the core to the window, so the rule is
checked from source: no module in the core packages imports ``fibsem.ui``, an app's
``ui`` package, Qt or napari. Two more edges are kept as they are today: the UI reaches
hardware through the microscope, never a driver, an app's App Server routes never
import the UI, and the shared UI (``fibsem.ui``) never imports an app. Imports under ``if TYPE_CHECKING:`` don't run, so they are allowed.
"""

import ast
from pathlib import Path

import pytest

import fibsem

FIBSEM = Path(fibsem.__file__).parent

CORE = [
    "ai",
    "alignment",
    "autofunctions",
    "codec",
    "correlation",
    "detection",
    "devices",
    "drivers",
    "fm",
    "geometry",
    "imaging",
    "milling",
    "server",
    "services",
    "workflows",
]

UI_MODULES = ("fibsem.ui", "PyQt5", "PyQt6", "qtpy", "superqt", "napari")


def _core_files():
    for package in CORE:
        yield from sorted((FIBSEM / package).rglob("*.py"))
    for app in sorted((FIBSEM / "applications").iterdir()):
        if app.is_dir():
            yield from sorted(
                path
                for path in app.rglob("*.py")
                if "ui" not in path.relative_to(app).parts[:1]
            )


def _is_ui(module: str) -> bool:
    if any(module == name or module.startswith(name + ".") for name in UI_MODULES):
        return True
    parts = module.split(".")
    return parts[:2] == ["fibsem", "applications"] and parts[3:4] == ["ui"]


def _ui_files():
    yield from sorted((FIBSEM / "ui").rglob("*.py"))
    for app in sorted((FIBSEM / "applications").glob("*/ui")):
        yield from sorted(app.rglob("*.py"))


def _is_driver(module: str) -> bool:
    return module == "fibsem.drivers" or module.startswith("fibsem.drivers.")


def _imports(path: Path, banned):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    type_checking = {
        id(child)
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.dump(node.test)
        for child in ast.walk(node)
    }
    for node in ast.walk(tree):
        if id(node) in type_checking:
            continue
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module]
        else:
            continue
        for name in names:
            if banned(name):
                yield f"{path.relative_to(FIBSEM.parent)}:{node.lineno} imports {name}"


def test_the_core_imports_no_ui():
    found = [line for path in _core_files() for line in _imports(path, _is_ui)]
    assert found == [], "\n".join(found)


def test_the_ui_imports_no_driver():
    found = [line for path in _ui_files() for line in _imports(path, _is_driver)]
    assert found == [], "\n".join(found)


def _is_app(module: str) -> bool:
    return module == "fibsem.applications" or module.startswith("fibsem.applications.")


def test_the_shared_ui_imports_no_app():
    found = [
        line
        for path in sorted((FIBSEM / "ui").rglob("*.py"))
        for line in _imports(path, _is_app)
    ]
    assert found == [], "\n".join(found)


def test_app_servers_import_no_ui():
    servers = sorted((FIBSEM / "applications").glob("*/server"))
    assert servers, "no app server package found"
    found = [
        line
        for server in servers
        for path in sorted(server.rglob("*.py"))
        for line in _imports(path, _is_ui)
    ]
    assert found == [], "\n".join(found)


@pytest.mark.parametrize(
    "module, ui",
    [
        ("fibsem.ui.notification_service", True),
        ("PyQt5.QtCore", True),
        ("napari", True),
        ("fibsem.applications.autolamella.ui.window", True),
        ("fibsem.uiutils", False),
        ("fibsem.services.milling", False),
    ],
)
def test_what_counts_as_ui(module, ui):
    assert _is_ui(module) is ui
