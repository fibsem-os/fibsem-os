"""An index of the experiments and agent sessions on this PC's disk.

It answers "which experiments are there, on which instrument, and which agent
sessions worked on them" without opening every folder. It is derived: the
experiment folders and the session folders are the record, and `rebuild`
makes the whole index again from them. Nothing reads the index to run, so a
missing or stale one costs a slower answer and never a run.

One SQLite file under the state folder. A shared copy of the experiment
folders is indexed the same way, by giving its folder as a root.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from fibsem import paths
from fibsem.config import peek_experiment

__all__ = ["Index", "files_named", "index_file"]

log = logging.getLogger(__name__)

EXPERIMENT_FILE = "experiment.yaml"
SESSION_FILE = "session.json"
# How far below a root an experiment's folder may sit.
MAX_DEPTH = 4
# Bumped when the tables change: an older file is made again, not migrated.
VERSION = 1

_TABLES = """
CREATE TABLE IF NOT EXISTS experiments (
    path TEXT PRIMARY KEY,
    mtime REAL NOT NULL,
    id TEXT,
    name TEXT NOT NULL,
    app TEXT,
    created_at TEXT,
    num_items INTEGER NOT NULL,
    available INTEGER NOT NULL,
    instrument_serial TEXT,
    instrument_model TEXT,
    operator TEXT,
    software_version TEXT
);
CREATE INDEX IF NOT EXISTS experiments_id ON experiments (id);
CREATE TABLE IF NOT EXISTS sessions (
    path TEXT PRIMARY KEY,
    mtime REAL NOT NULL,
    session_id TEXT,
    experiment_id TEXT,
    instrument_id TEXT,
    instrument_serial TEXT,
    playbook TEXT,
    goal TEXT,
    started TEXT
);
CREATE INDEX IF NOT EXISTS sessions_experiment ON sessions (experiment_id);
"""


def index_file() -> Path:
    return paths.state_root() / "index.sqlite"


def files_named(root: Path, name: str, depth: int = MAX_DEPTH) -> Iterator[Path]:
    """Each file called *name* at most *depth* folders below *root*. A folder
    that holds one is not searched further: an experiment's own folders are
    its images, not more experiments."""
    try:
        entries = sorted(os.scandir(root), key=lambda entry: entry.name)
    except OSError:
        return
    found = next((e for e in entries if e.name == name and e.is_file()), None)
    if found is not None:
        yield Path(found.path)
        return
    if depth <= 0:
        return
    for entry in entries:
        if entry.is_dir(follow_symlinks=False) and not entry.name.startswith("."):
            yield from files_named(Path(entry.path), name, depth - 1)


def _mtime(path: Path) -> Optional[float]:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _experiment(path: Path, mtime: float) -> Dict[str, Any]:
    found = peek_experiment(str(path))
    return {
        "path": str(path),
        "mtime": mtime,
        "id": found.id,
        "name": found.name,
        "app": found.app,
        "created_at": found.created_at.isoformat() if found.created_at else None,
        "num_items": found.num_items,
        "available": int(found.available),
        "instrument_serial": found.instrument_serial,
        "instrument_model": found.instrument_model,
        "operator": found.operator,
        "software_version": found.software_version,
    }


def _session(path: Path, mtime: float) -> Dict[str, Any]:
    try:
        with open(path) as f:
            record = json.load(f)
    except (OSError, ValueError):
        record = {}
    if not isinstance(record, dict):
        record = {}
    runs = record.get("runs") or [{}]
    first = runs[0] if isinstance(runs[0], dict) else {}

    def text(key: str) -> Optional[str]:
        value = record.get(key)
        return str(value) if value else None

    return {
        "path": str(path),
        "mtime": mtime,
        "session_id": text("session_id"),
        "experiment_id": text("experiment_id"),
        "instrument_id": text("instrument_id"),
        "instrument_serial": text("instrument_serial"),
        "playbook": text("playbook"),
        "goal": text("goal"),
        "started": str(first["started"]) if first.get("started") else None,
    }


class Index:
    """The index in *path* (this install's file by default)."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path is not None else index_file()
        self._lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        if db.execute("PRAGMA user_version").fetchone()[0] != VERSION:
            db.executescript(
                "DROP TABLE IF EXISTS experiments; DROP TABLE IF EXISTS sessions;"
            )
            db.execute(f"PRAGMA user_version = {VERSION}")
        db.executescript(_TABLES)
        return db

    # -- writing ---------------------------------------------------------------

    def rebuild(
        self,
        experiments: Iterable[Path] = (),
        sessions: Iterable[Path] = (),
    ) -> Dict[str, int]:
        """Forget everything and read the folders again: ``{experiments,
        sessions}``, how many of each it holds after."""
        with self._lock, closing(self._connect()) as db, db:
            db.execute("DELETE FROM experiments")
            db.execute("DELETE FROM sessions")
        return self.refresh(experiments, sessions)

    def refresh(
        self,
        experiments: Iterable[Path] = (),
        sessions: Iterable[Path] = (),
    ) -> Dict[str, int]:
        """Read what is new or changed under the given folders, and forget
        what is gone from them: ``{experiments, sessions}``, how many of each
        the index holds after. A row from a folder not given is kept."""
        with self._lock, closing(self._connect()) as db, db:
            for table, name, read, roots in (
                ("experiments", EXPERIMENT_FILE, _experiment, experiments),
                ("sessions", SESSION_FILE, _session, sessions),
            ):
                for root in roots:
                    self._refresh(db, table, Path(root), name, read)
            return {
                table: db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("experiments", "sessions")
            }

    def _refresh(self, db, table: str, root: Path, name: str, read) -> None:
        under = str(root).rstrip("/\\") + os.sep
        known = {
            row["path"]: row["mtime"]
            for row in db.execute(
                f"SELECT path, mtime FROM {table} WHERE substr(path, 1, ?) = ?",
                (len(under), under),
            )
        }
        seen = set()
        for path in files_named(root, name):
            mtime = _mtime(path)
            if mtime is None:
                continue
            seen.add(str(path))
            if known.get(str(path)) == mtime:
                continue
            row = read(path, mtime)
            keys = ", ".join(row)
            marks = ", ".join(f":{key}" for key in row)
            db.execute(f"INSERT OR REPLACE INTO {table} ({keys}) VALUES ({marks})", row)
        for gone in set(known) - seen:
            db.execute(f"DELETE FROM {table} WHERE path = ?", (gone,))

    # -- reading ---------------------------------------------------------------

    def _rows(self, query: str, values: List[Any]) -> List[Dict[str, Any]]:
        with self._lock, closing(self._connect()) as db:
            return [dict(row) for row in db.execute(query, values)]

    def experiments(
        self,
        app: Optional[str] = None,
        instrument_serial: Optional[str] = None,
        text: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """The experiments, newest first, narrowed by what is given; *text*
        is looked for in the name and the operator."""
        where, values = _where(app=app, instrument_serial=instrument_serial)
        if text:
            where.append("(name LIKE ? OR operator LIKE ?)")
            values += [f"%{text}%"] * 2
        return self._rows(
            "SELECT * FROM experiments"
            + (" WHERE " + " AND ".join(where) if where else "")
            + " ORDER BY created_at IS NULL, created_at DESC, path",
            values,
        )

    def sessions(
        self,
        experiment_id: Optional[str] = None,
        instrument_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """The agent sessions, newest first, narrowed by what is given."""
        where, values = _where(experiment_id=experiment_id, instrument_id=instrument_id)
        return self._rows(
            "SELECT * FROM sessions"
            + (" WHERE " + " AND ".join(where) if where else "")
            + " ORDER BY started IS NULL, started DESC, path",
            values,
        )


def _where(**given: Optional[str]):
    where = [f"{key} = ?" for key, value in given.items() if value is not None]
    return where, [value for value in given.values() if value is not None]
