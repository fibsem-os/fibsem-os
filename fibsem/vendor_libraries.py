"""Vendor libraries a driver imports from a folder outside this Python environment.

A driver's setup names such a library (``DriverSetup.library``): the packages it may
import from the folder and where the folder usually is. A copy installed in this
environment always wins. Otherwise the folder, on this PC, is the driver's environment
variable when set, else this PC's install preferences
(``InstallPreferences.vendor_libraries``), else the first of the driver's candidate
folders that holds one of its packages (the newest, where several do, so an upgrade
installed in a new folder is taken up). ``search`` asks the other Pythons on PATH. Only the named
packages are served from it, by a finder on ``sys.meta_path``. The folder never goes on
``sys.path``, so nothing else in it, such as an older numpy or packages compiled for
another Python, is imported by accident.
"""

from __future__ import annotations

import glob
import importlib.metadata
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec, PathFinder
from typing import Dict, List, Optional, Sequence, Tuple

from fibsem import config as cfg
from fibsem.drivers.registry import driver_setups
from fibsem.drivers.setup import SetupLibrary

SOURCE_ENV = "environment"
SOURCE_PREFERENCE = "preference"
SOURCE_FOUND = "found"
SOURCE_INSTALLED = "installed"


@dataclass(frozen=True)
class LibraryFolder:
    """Where a driver's library is on this PC, and what is in it."""

    folder: Optional[str]
    """None when nothing says where it is and no candidate folder holds it."""

    source: Optional[str] = None
    """What gave the folder: ``SOURCE_ENV``, ``SOURCE_PREFERENCE``, ``SOURCE_FOUND``,
    or ``SOURCE_INSTALLED`` where this environment has the library itself."""

    found: Tuple[str, ...] = ()
    """The library's packages the folder holds."""

    version: Optional[str] = None
    """The version its package metadata gives, where it has any."""

    @property
    def ok(self) -> bool:
        return bool(self.folder) and bool(self.found)


def libraries() -> Dict[str, SetupLibrary]:
    """Each driver that declares a library, by driver name, in registry order."""
    return {
        name: setup.library
        for name, setup in driver_setups()
        if setup.library is not None
    }


def held(folder: str, packages: Sequence[str]) -> Tuple[str, ...]:
    """The names in *packages* that *folder* holds, as a package or a module."""
    if not folder or not os.path.isdir(folder):
        return ()
    return tuple(name for name in packages if PathFinder.find_spec(name, [folder]))


def version(folder: str, packages: Sequence[str]) -> Optional[str]:
    """The version of the first of *packages* whose metadata is in *folder*."""
    versions = {}
    try:
        for dist in importlib.metadata.distributions(path=[folder]):
            name = (dist.metadata["Name"] or "").lower().replace("-", "_")
            versions.setdefault(name, dist.version)
    except Exception as e:  # noqa: BLE001 - a broken dist-info is no version
        logging.debug(f"Could not read package metadata in {folder}: {e}")
    return next((versions[n.lower()] for n in packages if n.lower() in versions), None)


def _version_key(text: Optional[str]) -> Tuple[int, ...]:
    """A version's leading numbers (``"1.14.0.post1"`` -> (1, 14, 0)), to sort by."""
    numbers = []
    for part in (text or "").split("."):
        digits = re.match(r"\d+", part)
        if digits is None:
            break
        numbers.append(int(digits.group()))
    return tuple(numbers)


def _newest_first(folders: List[Tuple[str, Optional[str]]]) -> List[str]:
    ordered = sorted(folders, key=lambda item: _version_key(item[1]), reverse=True)
    return [folder for folder, _ in ordered]


def candidates(library: SetupLibrary) -> List[str]:
    """The library's candidate folders that hold one of its packages, newest first."""
    matches = []
    for pattern in library.candidates:
        for folder in sorted(
            glob.glob(os.path.expandvars(os.path.expanduser(pattern)))
        ):
            if folder not in [m[0] for m in matches] and held(folder, library.packages):
                matches.append((folder, version(folder, library.packages)))
    return _newest_first(matches)


def _describe(folder: str, source: str, library: SetupLibrary) -> LibraryFolder:
    return LibraryFolder(
        folder=folder,
        source=source,
        found=held(folder, library.packages),
        version=version(folder, library.packages),
    )


def installed(library: SetupLibrary) -> Optional[str]:
    """The folder this environment imports the library's first package from, if it
    has one of its own; that copy is imported whatever folder is set."""
    if not library.packages:
        return None
    try:
        spec = PathFinder.find_spec(library.packages[0])
    except Exception as e:  # noqa: BLE001 - a package that breaks on inspection
        logging.debug(f"Could not look for {library.packages[0]}: {e}")
        return None
    return _spec_folder(spec)


def _spec_folder(spec: Optional[ModuleSpec]) -> Optional[str]:
    if spec is None:
        return None
    if spec.submodule_search_locations:
        return os.path.dirname(list(spec.submodule_search_locations)[0])
    return os.path.dirname(spec.origin) if spec.origin else None


def check(folder: str, library: SetupLibrary) -> LibraryFolder:
    """What *folder* holds of *library*, as typed into a form."""
    return _describe(folder.strip(), SOURCE_PREFERENCE, library)


def resolve(driver: str, library: SetupLibrary) -> LibraryFolder:
    """Where *driver*'s library is on this PC: this environment's own copy, its
    environment variable, this PC's preference, then its candidate folders."""
    own = installed(library)
    if own:
        return _describe(own, SOURCE_INSTALLED, library)
    if library.env and os.environ.get(library.env):
        return _describe(os.environ[library.env], SOURCE_ENV, library)
    folders = cfg.load_user_preferences().install.vendor_libraries or {}
    preferred = folders.get(driver)
    if preferred:
        return _describe(preferred, SOURCE_PREFERENCE, library)
    found = candidates(library)
    if found:
        return _describe(found[0], SOURCE_FOUND, library)
    return LibraryFolder(folder=None)


def save_folder(driver: str, folder: str) -> None:
    """Keep *folder* as *driver*'s library on this PC; an empty one clears it."""

    def change(preferences: cfg.UserPreferences) -> None:
        folders = preferences.install.vendor_libraries = dict(
            preferences.install.vendor_libraries or {}
        )
        if folder.strip():
            folders[driver] = folder.strip()
        else:
            folders.pop(driver, None)

    cfg.update_user_preferences(change)


# Asks a Python where it imports a package from, without importing it. Kept to what
# Python 3.5 runs: AutoScript's own environment can be that old.
_WHERE_SCRIPT = (
    "import importlib.util, os, sys\n"
    "s = importlib.util.find_spec(sys.argv[1])\n"
    "if s is not None:\n"
    "    p = list(s.submodule_search_locations or []) or [s.origin]\n"
    "    print(os.path.dirname(p[0]))\n"
)


def pythons_on_path() -> List[str]:
    """Each Python on PATH, in PATH order, as ``where python`` lists them. The
    Windows Store stand-in (WindowsApps), which opens the Store, is left out."""
    names = ("python.exe",) if os.name == "nt" else ("python", "python3")
    found: List[str] = []
    seen = set()
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if not directory or "windowsapps" in directory.lower():
            continue
        for name in names:
            path = shutil.which(name, path=directory)
            if path is None:
                continue
            real = os.path.normcase(os.path.realpath(path))
            if real not in seen:
                seen.add(real)
                found.append(path)
    return found


def search(library: SetupLibrary, timeout: float = 10.0) -> List[LibraryFolder]:
    """Folders holding *library*, from asking each other Python on PATH where it
    imports the library's first package from; newest first. Slow (each Python is
    started), so only on request."""
    if not library.packages:
        return []
    ours = os.path.normcase(os.path.realpath(sys.executable))
    folders: List[Tuple[str, Optional[str]]] = []
    for python in pythons_on_path():
        if os.path.normcase(os.path.realpath(python)) == ours:
            continue
        try:
            result = subprocess.run(
                [python, "-I", "-c", _WHERE_SCRIPT, library.packages[0]],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (OSError, subprocess.SubprocessError) as e:
            logging.debug(f"Could not ask {python} for the {library.label}: {e}")
            continue
        folder = result.stdout.strip()
        if result.returncode == 0 and folder and held(folder, library.packages):
            if folder not in [f for f, _ in folders]:
                folders.append((folder, version(folder, library.packages)))
    return [_describe(f, SOURCE_FOUND, library) for f in _newest_first(folders)]


class _LibraryFinder(MetaPathFinder):
    """Finds only *packages*, and only in *folder*. Their submodules are found
    through the package's own path, as usual."""

    def __init__(self, folder: str, packages: Sequence[str]):
        self.folder = folder
        self.packages = frozenset(packages)

    def find_spec(self, fullname, path=None, target=None) -> Optional[ModuleSpec]:
        if path is not None or fullname not in self.packages:
            return None
        return PathFinder.find_spec(fullname, [self.folder])


_FINDERS: Dict[str, _LibraryFinder] = {}


def serve(
    driver: str, library: SetupLibrary, folder: Optional[str] = None
) -> Optional[str]:
    """Make *driver*'s library importable: its packages from *folder*, else from the
    resolved folder. Returns the folder, or None where this environment has the
    library itself and nothing needs serving.
    Raises ``ImportError`` when no folder holds it, saying where it was looked for."""
    if folder is None and installed(library):
        return None
    found = check(folder, library) if folder else resolve(driver, library)
    if not found.ok:
        where = f"in {found.folder}" if found.folder else "anywhere"
        hints = ["choose its folder in Preferences or the guided setup"]
        if library.env:
            hints.append(f"set {library.env}")
        raise ImportError(
            f"The {library.label} was not found {where}: " + ", or ".join(hints) + "."
        )
    previous = _FINDERS.pop(driver, None)
    if previous in sys.meta_path:
        sys.meta_path.remove(previous)
    finder = _LibraryFinder(found.folder, library.packages)
    _FINDERS[driver] = finder
    # last, so a package installed in this environment wins over the folder's copy
    sys.meta_path.append(finder)
    version_text = f" {found.version}" if found.version else ""
    logging.info(
        f"{driver}: the {library.label}{version_text} is served from {found.folder}."
    )
    return found.folder
