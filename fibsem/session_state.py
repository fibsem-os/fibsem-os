"""What an instrument was left at: the state a session restores.

One of five kinds of persisted state, and the one that changes constantly:

- the **microscope configuration** -- what the instrument is, its calibration, its
  defaults -- changes rarely and is written by setup and calibration actions;
- **session state** (this module) -- what was on screen when the application closed
  -- is autosaved as the operator works;
- **application configuration** -- the operator's preferences -- belongs to the
  user and PC, not to an instrument;
- a **protocol** is a method, meant to be portable across instruments;
- an **experiment** is a run, and keeps copies of what it ran with.

Session state is scoped to one instrument configuration, so it lives in one file per
configuration -- `config/session/<configuration file name>.yaml` -- and switching
configurations can never restore one instrument's grids or positions onto another.

**What belongs here.** Session state holds last-used values only where no preference
home exists. The acquire tab's imaging settings are not here, because
`defaults.imaging` in the microscope configuration is their home: a restored copy
would shadow it, and editing the defaults would appear to do nothing. The FM working
state is here because it has no other home. A new section should be able to say why
it has no home elsewhere.

**Who writes.** The application does; scripts and headless runs read but never write,
so a script cannot rewrite what the operator left on screen. A store is read-only
unless it is constructed with `writable=True`.

**How a write is kept safe.** Several writers share the file -- the FM autosave fires
about once a second, the holder occupancy is written on a grid swap -- so each write
re-reads the file immediately before writing, replaces only its own section, and
replaces the file atomically. Sections other writers own are carried across
untouched, including ones this version does not know.

A section name may be dotted -- `fm.working`, `fm.recent_channels` -- to name one
key inside a section. Writing it replaces that key and nothing else in the section,
so two writers of one subsystem's state do not overwrite each other either.

**How a read is kept safe.** A missing file or section reads as empty. A file that
cannot be parsed reads as empty too, is logged, and is moved aside on the next write
rather than overwritten, so whatever it held can still be recovered by hand.
"""

import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import yaml

from fibsem import config as cfg

SESSION_STATE_VERSION: int = 1

# Next to the other state the application writes, not beside the configuration
# file: the shipped configurations live inside the installed package.
SESSION_STATE_DIRECTORY: str = os.path.join(cfg.CONFIG_PATH, "session")

# One lock per file, for writers in the same process: the read-modify-write below
# must not interleave between two threads writing different sections.
_LOCKS: Dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(str(path), threading.Lock())


_ABSENT = object()


def _get(data: dict, name: str) -> Any:
    """The value at a dotted *name*, or `_ABSENT`."""
    node: Any = data
    for part in name.split("."):
        if not isinstance(node, dict) or part not in node:
            return _ABSENT
        node = node[part]
    return node


def _set(data: dict, name: str, value: Any) -> None:
    """Set the value at a dotted *name*, creating (or replacing non-mapping)
    parents on the way; sibling keys are left as they are."""
    *parents, leaf = name.split(".")
    node = data
    for part in parents:
        if not isinstance(node.get(part), dict):
            node[part] = {}
        node = node[part]
    node[leaf] = value


def session_state_path(
    configuration_path: Union[str, Path], directory: Optional[str] = None
) -> Path:
    """Where the session state for a configuration lives: named by its file name."""
    stem = Path(configuration_path).stem
    return Path(directory or SESSION_STATE_DIRECTORY) / f"{stem}.yaml"


class SessionState:
    """The session state of one instrument configuration.

    Args:
        configuration_path: the microscope configuration file the session runs with.
            `None` (a session not started from a file) gives a store that reads as
            empty and never writes.
        writable: whether this store may write. The application passes `True`;
            scripts leave it at `False`.
        directory: where the session files live; defaults to
            `config/session/`. Tests point it elsewhere.
    """

    def __init__(
        self,
        configuration_path: Optional[Union[str, Path]],
        writable: bool = False,
        directory: Optional[str] = None,
    ) -> None:
        self.path: Optional[Path] = (
            session_state_path(configuration_path, directory)
            if configuration_path
            else None
        )
        self.writable = writable and self.path is not None

    # ---- reading ----------------------------------------------------------

    def _read(self) -> Optional[dict]:
        """The whole file, `{}` when absent, or `None` when it cannot be parsed."""
        if self.path is None or not self.path.exists():
            return {}
        try:
            with open(self.path, "r") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            logging.warning(f"Session state {self.path} could not be read: {e}")
            return None
        if data is None:
            return {}
        if not isinstance(data, dict):
            logging.warning(f"Session state {self.path} is not a mapping; ignoring it.")
            return None
        return data

    def load_section(
        self,
        name: str,
        default: Any = None,
        migrate: Optional[Callable[[], Any]] = None,
    ) -> Any:
        """The stored value of a section, or *default* when there is none.

        *migrate* is called when the section is absent, to import it from wherever
        it lived before this store existed. Its result is returned, and written
        into the store when this store may write -- so an old file is read once,
        and never deleted. A migration that raises is logged and treated as
        having found nothing.
        """
        value = _get(self._read() or {}, name)
        if value is not _ABSENT:
            return value
        if migrate is not None:
            try:
                migrated = migrate()
            except Exception as e:
                logging.warning(f"Could not import session state '{name}': {e}")
                migrated = None
            if migrated is not None:
                self.save_section(name, migrated)
                return migrated
        return default

    # ---- writing ----------------------------------------------------------

    def save_section(self, name: str, value: Any) -> bool:
        """Replace one section, leaving every other section as it is on disk.

        Returns whether anything was written. A read-only store writes nothing.
        """
        if not self.writable:
            return False
        from fibsem.utils import _plain

        with _lock_for(self.path):
            data = self._read()
            if data is None:
                self._set_aside_unreadable()
                data = {}
            data["version"] = SESSION_STATE_VERSION
            _set(data, name, _plain(value))
            try:
                self._replace(data)
            except Exception as e:
                logging.warning(
                    f"Could not save session state '{name}' to {self.path}: {e}"
                )
                return False
        return True

    def _replace(self, data: dict) -> None:
        """Write the whole file atomically: a reader never sees half of it."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # `version` first, then the sections in the order they were added.
        ordered = {"version": data.pop("version"), **data}
        # Written beside the file and swapped in, so a write cut short -- a crash,
        # a kill, a power cut during the FM autosave -- leaves the old file, never a
        # truncated one. A plain `open`, not `mkstemp`: that creates the file 0600,
        # and `os.replace` keeps the mode, which on a PC shared between operator
        # accounts would lock the next operator out. The lock above covers this
        # process's threads, and only the application writes, so the name needs
        # only the process to be unique.
        temp = str(self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp"))
        try:
            with open(temp, "w") as f:
                yaml.safe_dump(ordered, f, sort_keys=False, indent=4)
            os.replace(temp, self.path)
        except BaseException:
            if os.path.exists(temp):
                os.remove(temp)
            raise

    def _set_aside_unreadable(self) -> None:
        """Keep an unparseable file for recovery by hand, rather than overwrite it."""
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        kept = self.path.with_name(f"{self.path.name}.unreadable-{stamp}")
        try:
            os.replace(self.path, kept)
            logging.warning(f"Unreadable session state moved aside to {kept}")
        except OSError as e:
            logging.warning(f"Could not move unreadable {self.path} aside: {e}")


def session_state_for(microscope, writable: bool = False) -> SessionState:
    """The session state for the configuration *microscope* was connected with."""
    return SessionState(
        getattr(microscope, "configuration_path", None), writable=writable
    )
