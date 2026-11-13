"""Where copies of experiments are kept: one small interface in front of a
folder (this PC's disk or a shared drive) and, later, remote storage.

A store holds files by a relative name with ``/`` between its parts. The
experiment being worked on always stays on its own PC's disk; a store only
ever receives copies (`fibsem.copy_up`), so nothing waits on one to run.
"""

from __future__ import annotations

import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterator, Optional, Protocol

__all__ = ["FolderStore", "Kept", "Store", "valid_name"]


@dataclass(frozen=True)
class Kept:
    """A file in a store: its size, and when its source was last changed."""

    name: str
    size: int
    mtime: float


def valid_name(name: str) -> bool:
    """A relative name with no way out of the store."""
    parts = PurePosixPath(name).parts
    return (
        bool(parts)
        and "\\" not in name
        and not PurePosixPath(name).is_absolute()
        and all(part not in ("..", ".") and ":" not in part for part in parts)
    )


class Store(Protocol):
    def exists(self, name: str) -> bool: ...

    def stat(self, name: str) -> Optional[Kept]: ...

    def list(self, prefix: str = "") -> Iterator[Kept]: ...

    def save(self, name: str, source: Path) -> Kept: ...

    def read(self, name: str) -> bytes: ...


class FolderStore:
    """A store that is a folder: a local one, or a shared drive's."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _path(self, name: str) -> Path:
        if not valid_name(name):
            raise ValueError(f"Not a name in a store: {name!r}")
        return self.root.joinpath(*PurePosixPath(name).parts)

    def available(self) -> bool:
        """Whether the folder is there now: a shared drive may not be."""
        return self.root.is_dir()

    def exists(self, name: str) -> bool:
        return self._path(name).is_file()

    def stat(self, name: str) -> Optional[Kept]:
        try:
            found = self._path(name).stat()
        except OSError:
            return None
        return Kept(name, found.st_size, found.st_mtime)

    def list(self, prefix: str = "") -> Iterator[Kept]:
        start = self._path(prefix) if prefix else self.root
        for folder, _, files in os.walk(start):
            for file in sorted(files):
                path = Path(folder) / file
                try:
                    found = path.stat()
                except OSError:
                    continue
                name = path.relative_to(self.root).as_posix()
                yield Kept(name, found.st_size, found.st_mtime)

    def save(self, name: str, source: Path) -> Kept:
        """Copy *source* in as *name*, keeping its modified time. It is
        written beside its place and moved in, so a reader never sees half a
        file and a copy that is cut short leaves the old one."""
        path = self._path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.part")
        try:
            shutil.copy2(source, part)
            os.replace(part, path)
        finally:
            if part.exists():
                part.unlink()
        found = path.stat()
        return Kept(name, found.st_size, found.st_mtime)

    def read(self, name: str) -> bytes:
        return self._path(name).read_bytes()
