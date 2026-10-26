"""Each optional extra's packages stay in the folders that need them.

``pip install fibsem`` must import everywhere outside those folders. Two checks keep
it so: no module-level import of an extra's package outside its home (the UI extra
has its own, ``test_ui_extra_isolation.py``), and every module outside the homes
imports in a fresh interpreter. The second is the one that catches a core module
reaching an extra through another fibsem module; it only proves anything on an
install without the extras, which is what CI's build matrix is.
"""

import ast
import importlib.metadata
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest

from fibsem import extras

FIBSEM = Path(__file__).resolve().parents[1] / "fibsem"

# extra: (its top-level import names, the folders under fibsem/ that may import them)
HOMES = {
    "ml": (
        ("torch", "segmentation_models_pytorch", "transformers", "onnx", "onnxruntime"),
        ("segmentation/",),
    ),
    "mcp": (("mcp",), ("ai/agent/",)),
    "agent": (("claude_agent_sdk",), ("ai/agent/",)),
}


def _modules() -> List[str]:
    return sorted(
        path.relative_to(FIBSEM).as_posix()
        for path in FIBSEM.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def _is_ui(rel: str) -> bool:
    return rel.startswith("ui/") or "/ui/" in rel


def _in_a_home(rel: str) -> bool:
    return _is_ui(rel) or any(
        rel.startswith(folder) for _, folders in HOMES.values() for folder in folders
    )


def _module_level_imports(rel: str) -> List[str]:
    tree = ast.parse((FIBSEM / rel).read_text(encoding="utf-8"))
    names = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.append(node.module)
    return names


MODULES = _modules()


@pytest.mark.parametrize("extra", sorted(HOMES))
def test_an_extras_packages_are_imported_only_in_its_home(extra):
    packages, folders = HOMES[extra]
    offenders = [
        f"fibsem/{rel}: {name}"
        for rel in MODULES
        if not rel.startswith(folders)
        for name in _module_level_imports(rel)
        if name.split(".")[0] in packages
    ]
    assert not offenders, (
        f"Module-level imports of the [{extra}] extra outside "
        f"{', '.join('fibsem/' + f for f in folders)}: {offenders}. Import it inside "
        "the function that needs it, or move the code into that folder."
    )


def test_every_module_outside_the_homes_imports():
    modules = [
        "fibsem." + rel[: -len(".py")].replace("/", ".").removesuffix(".__init__")
        for rel in MODULES
        # A __main__ module runs its program when imported.
        if not _in_a_home(rel) and not rel.endswith("__main__.py")
    ]
    assert len(modules) > 300, f"only found {len(modules)} modules"
    script = (
        "import importlib, sys\n"
        "for name in sys.stdin.read().split():\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "    except BaseException as e:\n"
        "        print(f'{name}: {type(e).__name__}: {e}')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        input="\n".join(modules),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    assert not result.stdout, (
        "These modules don't import on this install, so they would break a plain "
        f"`pip install fibsem`:\n{result.stdout}"
    )


def test_an_extra_that_includes_another_lists_both_packages():
    names = {requirement.name for requirement in extras._requirements("agent")}
    assert {"mcp", "claude-agent-sdk"} <= names


def test_the_message_names_what_is_missing_and_how_to_install_it(monkeypatch):
    real_version = importlib.metadata.version

    def version(name):
        if name == "claude-agent-sdk":
            raise importlib.metadata.PackageNotFoundError(name)
        return real_version(name) if name != "mcp" else "2.0"

    monkeypatch.setattr(importlib.metadata, "version", version)
    assert extras.missing_packages("agent") == ["claude-agent-sdk"]
    assert extras.missing_extra_message("agent", "fibsem-agent") == (
        "fibsem-agent needs the [agent] extra (claude-agent-sdk not installed): "
        'pip install "fibsem[agent]"'
    )


def test_an_installed_extra_has_no_message():
    # pytest runs from the [test] extra, so it is installed wherever this runs.
    assert extras.missing_extra_message("test", "the tests") is None
