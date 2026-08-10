"""The tiling geometry layers must stay free of the microscope and the UI.

This is the property the package exists to provide, and it is one convenient import
away from being lost.

Checked *statically*, against what the modules declare, rather than by importing them
and inspecting sys.modules. A runtime check cannot distinguish a module's own
dependencies from its parent package's: `fibsem/imaging/__init__.py` does
`from fibsem.imaging.tiled import *`, so importing any submodule of `fibsem.imaging`
loads the microscope regardless of what that submodule needs. Removing that
star-import would be a public API change and does not belong here.

Note that the original motivation for this split -- an import cycle supposedly
preventing `fibsem.fm` from importing `fibsem.imaging.tiled` at all -- turned out not
to exist. `fibsem.fm.acquisition` can import it today, because the back-edge
(`fibsem.microscope` -> `fibsem.fm.microscope`) is a submodule import that never needs
the partially-initialised package's attributes. The reason to share one definition is
the reason it always was: the two tilers had already drifted into a shipped bug (#226).
"""

import ast
from pathlib import Path

import pytest

import fibsem

# Resolved from the package rather than the working directory, which pytest does not
# guarantee.
FIBSEM_ROOT = Path(fibsem.__file__).parent

PURE_MODULES = [
    "imaging/tiling/geometry.py",
    "imaging/tiling/reprojection.py",
    "imaging/tiling/plotting.py",
]

FORBIDDEN_PREFIXES = ("fibsem.microscope", "fibsem.ui", "napari")


def _declared_imports(relative_path: str) -> set:
    """Every module named by an import statement anywhere in the file."""
    source = (FIBSEM_ROOT / relative_path).read_text()
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
    return names


def _module_level_imports(relative_path: str) -> set:
    """Only imports at module scope -- the ones that cost something to `import`.

    Imports nested inside a function are deliberately excluded: deferring an
    optional dependency to the one function that needs it is the pattern being
    protected here, not a violation of it.
    """
    source = (FIBSEM_ROOT / relative_path).read_text()
    names = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
    return names


@pytest.mark.parametrize("path", PURE_MODULES)
def test_layers_declare_no_microscope_or_ui_import(path):
    offending = sorted(
        m for m in _declared_imports(path)
        if m.startswith(FORBIDDEN_PREFIXES)
    )
    assert not offending, (
        f"{path} imports {offending}. These layers stay independent of the microscope "
        f"and the UI so the fluorescence tiler can share them, and so that consuming "
        f"tile geometry does not cost a driver stack."
    )


def test_the_runner_module_still_imports_the_microscope():
    """The complement: tiled.py legitimately needs it, so its import is not a
    leftover to be 'cleaned up' by moving the runner as well."""
    assert "fibsem.microscope" in _declared_imports("imaging/tiled.py")


def test_tiled_still_exports_the_moved_names():
    """Existing importers must not notice the move.

    `fibsem/imaging/__init__.py` star-imports tiled, and 15 modules import from it,
    including a visual harness at fibsem/imaging/tests/.
    """
    import fibsem.imaging.tiled as tiled

    for name in [
        "TilePosition",
        "compute_tile_grid",
        "order_tiles",
        "validate_tile_stage_positions",
        "plot_tile_positions",
        "plot_stage_positions_on_image",
        "plot_minimap",
        "calculate_reprojected_stage_position",
        "calculate_reprojected_stage_position2",
        "reproject_stage_positions_onto_image",
        "reproject_stage_positions_onto_image2",
        "POSITION_COLOURS",
        "_spiral_order",
        "_transform_position",
        "_inverse_y_corrected_stage_movement",
    ]:
        assert hasattr(tiled, name), f"tiled.{name} disappeared in the move"


def test_the_package_exposes_the_public_surface():
    import fibsem.imaging.tiling as tiling

    assert set(tiling.__all__) <= set(dir(tiling))
    for name in tiling.__all__:
        assert getattr(tiling, name) is not None


# The `ui`-extra import guard that used to live here is now repo-wide, in
# tests/test_ui_extra_isolation.py. It was parametrised over this package alone, which
# protected the module FIB-390 had just broken and nothing else -- reporting.py
# acquired the same module-level import afterwards and kept it. The general version
# walks the tree, so this package stays covered without naming it.
_MOVED_TO = "tests/test_ui_extra_isolation.py"


def test_the_ui_extra_guard_still_exists():
    """This package relies on that guard; a rename should fail here, not go quiet."""
    assert (Path(__file__).resolve().parents[1] / _MOVED_TO).exists(), (
        f"{_MOVED_TO} is missing -- the `ui` extra import guard for this package "
        f"(FIB-390) went with it."
    )
