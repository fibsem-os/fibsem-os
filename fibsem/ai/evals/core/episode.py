"""An episode: the record of one run, real or simulated, in one folder an eval
can be cut from. It holds a copy of the experiment (its ``events.jsonl``, its
images, its configs) and of each agent session's folder (``session.json``, the
transcript the agent was shown, its log and usage), with ``episode.yaml``
naming where they came from. Nothing in it is changed: paths the record names
are read through :meth:`Episode.local`, which finds them in the copy wherever
the run kept them. `fibsem.ai.evals.core.segments` cuts it into cases.
"""

import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml

from fibsem.util import timestamps

log = logging.getLogger(__name__)

FILE = "episode.yaml"
FORMAT = 0
EXPERIMENT = "experiment"
SESSIONS = "sessions"
EVENTS = "events.jsonl"
SESSION_RECORD = "session.json"


class EpisodeError(RuntimeError):
    pass


# -- making one ---------------------------------------------------------------------


def _session_id(folder: Path) -> Optional[str]:
    try:
        return json.loads((folder / SESSION_RECORD).read_text()).get("session_id")
    except (OSError, ValueError, AttributeError):
        return None


def session_ids(events: Iterable[Dict[str, Any]]) -> List[str]:
    """The agent sessions an experiment's events name, in the order they
    first appear."""
    found: Dict[str, None] = {}
    for event in events:
        payload = event.get("payload") or {}
        if event.get("actor") == "agent" and payload.get("session_id"):
            found.setdefault(str(payload["session_id"]), None)
    return list(found)


def find_sessions(ids: Sequence[str], places: Sequence[Path]) -> Dict[str, Path]:
    """Each session's folder among *places*: a session folder itself (its
    ``session.json`` names it), or a folder of them (``<date>-<id>``, as
    ``~/.fibsem/agent/sessions`` keeps them)."""
    found: Dict[str, Path] = {}
    for place in map(Path, places):
        candidates = [place] + (sorted(place.iterdir()) if place.is_dir() else [])
        for folder in candidates:
            if not folder.is_dir():
                continue
            sid = _session_id(folder)
            if sid is None:
                sid = next((i for i in ids if folder.name.endswith(f"-{i}")), None)
            if sid in ids and sid not in found:
                found[sid] = folder
    return found


def _copy(source: Path, target: Path, link: bool) -> None:
    if link:
        shutil.copytree(source, target, copy_function=_link_or_copy)
    else:
        shutil.copytree(source, target)


def _link_or_copy(source: str, target: str) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def pack(
    experiment: Path,
    out: Path,
    sessions: Sequence[Path] = (),
    link: bool = False,
    note: Optional[str] = None,
) -> "Episode":
    """Copy *experiment* and the agent sessions its events name (found in
    *sessions*) into a new episode folder *out*. With *link*, files are hard
    linked where the disk allows, so a large experiment costs no space."""
    experiment, out = Path(experiment), Path(out)
    if not (experiment / EVENTS).is_file():
        raise EpisodeError(f"{experiment} has no {EVENTS}: not an experiment folder")
    if out.exists():
        raise EpisodeError(f"{out} exists: give a new folder")
    events = read_events(experiment / EVENTS)
    ids = session_ids(events)
    folders = find_sessions(ids, sessions)
    out.mkdir(parents=True)
    _copy(experiment, out / EXPERIMENT / experiment.name, link)
    kept = []
    for sid in ids:
        entry: Dict[str, Any] = {"id": sid}
        if sid in folders:
            _copy(folders[sid], out / SESSIONS / sid, link)
            entry["from"] = str(folders[sid])
            record = _read_json(folders[sid] / SESSION_RECORD) or {}
            for key in ("model", "fibsem", "playbook", "playbook_hash", "mode"):
                if record.get(key) is not None:
                    entry[key] = record[key]
        else:
            entry["missing"] = True
            log.warning("session %s: no folder found; its transcript is not kept", sid)
        kept.append(entry)
    data = {
        "format": FORMAT,
        "packed_at": timestamps.now().isoformat(),
        "experiment": {"name": experiment.name, "from": str(experiment)},
        "sessions": kept,
    }
    if note:
        data["note"] = note
    (out / FILE).write_text(yaml.safe_dump(data, sort_keys=False))
    return Episode(out, data)


# -- reading one --------------------------------------------------------------------


def read_events(path: Path) -> List[Dict[str, Any]]:
    events = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                events.append(json.loads(line))
            except ValueError:
                log.debug("skipping a line of %s that is not JSON", path)
    return sorted(events, key=lambda e: e.get("timestamp") or 0)


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _parts(recorded: str) -> Sequence[str]:
    if "\\" in recorded or (len(recorded) > 1 and recorded[1] == ":"):
        return PureWindowsPath(recorded).parts
    return PurePosixPath(recorded).parts


@dataclass
class Episode:
    path: Path
    data: Dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> "Episode":
        path = Path(path)
        try:
            data = yaml.safe_load((path / FILE).read_text()) or {}
        except OSError:
            raise EpisodeError(f"{path} has no {FILE}: not an episode folder")
        return cls(path, data)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def experiment(self) -> Path:
        return self.path / EXPERIMENT / self.data["experiment"]["name"]

    def events(self) -> List[Dict[str, Any]]:
        return read_events(self.experiment / EVENTS)

    def session(self, sid: str) -> Optional[Path]:
        folder = self.path / SESSIONS / sid
        return folder if folder.is_dir() else None

    def local(self, recorded: Optional[str]) -> Optional[Path]:
        """Where a path the record names is in this episode: one inside the
        experiment, however the run's machine spelled it (``C:\\...`` or
        ``/Users/...``), or relative to the experiment. None for a path
        outside it, or one the copy doesn't hold."""
        if not recorded:
            return None
        parts = list(_parts(str(recorded)))
        name = self.data["experiment"]["name"]
        if name in parts:
            rest = parts[len(parts) - 1 - parts[::-1].index(name) + 1 :]
        elif (
            parts
            and not PurePosixPath(str(recorded)).is_absolute()
            and ":" not in parts[0]
        ):
            rest = parts
        else:
            return None
        path = self.experiment.joinpath(*rest) if rest else self.experiment
        return path if path.exists() else None
