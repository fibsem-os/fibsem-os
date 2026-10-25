"""Vendor libraries a driver imports from a folder outside this Python environment.

A driver's setup names such a library (``DriverSetup.library``): the packages it may
import from the folder and where the folder usually is. Which folder, on this PC, is
the driver's environment variable when set, else this PC's install preferences
(``InstallPreferences.vendor_libraries``), else the first of the driver's candidate
folders that holds one of its packages (the newest, where several do). Only the named
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


@dataclass(frozen=True)
class LibraryFolder:
    """Where a driver's library is on this PC, and what is in it."""

    folder: Optional[str]
    """None when nothing says where it is and no candidate folder holds it."""

    source: Optional[str] = None
    """What gave the folder: ``SOURCE_ENV``, ``SOURCE_PREFERENCE`` or ``SOURCE_FOUND``."""

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
    wanted = {name.lower() for name in packages}
    try:
        for dist in importlib.metadata.distributions(path=[folder]):
            name = (dist.metadata["Name"] or "").lower().replace("-", "_")
            if name in wanted:
                return dist.version
    except Exception as e:  # noqa: BLE001 - a broken dist-info is no version
        logging.debug(f"Could not read package metadata in {folder}: {e}")
    return None


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


def check(folder: str, library: SetupLibrary) -> LibraryFolder:
    """What *folder* holds of *library*, as typed into a form."""
    return _describe(folder.strip(), SOURCE_PREFERENCE, library)


def resolve(driver: str, library: SetupLibrary) -> LibraryFolder:
    """Where *driver*'s library is on this PC: its environment variable, this PC's
    preference, then its candidate folders."""
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


def serve(driver: str, library: SetupLibrary, folder: Optional[str] = None) -> str:
    """Make *driver*'s library importable: its packages from *folder*, else from the
    resolved folder. Returns the folder. Raises ``ImportError`` when no folder holds
    it, saying where it was looked for."""
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
    logging.info(f"{driver}: the {library.label} is served from {found.folder}.")
    return found.folder
