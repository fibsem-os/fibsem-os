"""This PC's agent sessions and experiments, and the links between them, read
from their own folders.

Nothing here writes or keeps a copy: each folder's files stay the record, so
the list can't go stale, and every caller (the gateway, the scoring page, the
hub) gets the same answer. A session's status is what its record says, and
for a run that never wrote its end, whether its process still runs.
"""

import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import yaml

from fibsem import paths
from fibsem.ai.agent import manager
from fibsem.config import recent_experiment_paths
from fibsem.util import timestamps
from fibsem.util.process import same_process

logger = logging.getLogger(__name__)

EXPERIMENT_FILE = "experiment.yaml"
EVENTS_FILE = "events.jsonl"
# The session's own record of its conversation.
EVENTS_IN_SESSION = os.path.join("agent", EVENTS_FILE)
# In a session's agent/ folder: the experiment it works on.
LINK = os.path.join("agent", "experiment.json")

# A session's status.
RUNNING = "running"
# Its manager will start or resume it once its app answers: the app went
# away, the operator stopped the app (parked), or a start waits for its app.
WAITING = "waiting"
ENDED = "ended"
# Its gateway was closed on purpose with the session unfinished.
LEFT = "left"
# Its process is gone without writing how it ended.
CRASHED = "crashed"


@dataclass
class ExperimentLink:
    path: str
    name: Optional[str] = None
    id: Optional[str] = None


@dataclass
class SessionRow:
    session_id: str
    folder: str
    status: str
    reason: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    # When the session last wrote its record, its events or its output.
    last_active_at: Optional[str] = None
    playbook: Optional[str] = None
    model: Optional[str] = None
    goal: Optional[str] = None
    level: Optional[str] = None
    spent_usd: Optional[float] = None
    by_gateway: bool = False
    experiment: Optional[ExperimentLink] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentRow:
    path: str
    name: str
    id: Optional[str] = None
    updated_at: Optional[str] = None
    session_ids: List[str] = field(default_factory=list)
    # False for an experiment a session here linked that isn't on this PC
    # (moved, deleted, or on another machine): known by its link only.
    on_this_pc: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _alive(run: dict) -> bool:
    """Whether *run*'s process is still running: the same process, not a
    reused pid."""
    return same_process(run.get("pid"), run.get("pid_started"))


def session_status(
    record: dict, managed: dict, alive: Callable[[dict], bool] = _alive
) -> Tuple[str, Optional[str]]:
    """A session's status and, once it has ended, why, from its record
    (``session.json``, or empty for a start still waiting for its app) and
    its manager's (``manager.json``, or empty when run by hand).

    A manager's ``ended`` is the last word; until it writes one, a session it
    runs whose last run ended waits for it (the app went, or was stopped and
    the agent parked)."""
    runs = record.get("runs") or []
    last = runs[-1] if runs else {}
    if runs and "ended" not in last and alive(last):
        return RUNNING, None
    if managed.get("ended"):
        return ENDED, managed["ended"]
    if managed.get("left") is not None:
        return LEFT, "the gateway was closed"
    if managed and not runs:
        return WAITING, "waiting for its app to start"
    if "ended" not in last:
        return CRASHED, None
    if managed and last.get("resumable", True):
        return WAITING, last.get("reason")
    return ENDED, last.get("reason")


def _playbook(record: dict) -> Optional[str]:
    playbook = record.get("playbook")
    return playbook.get("name") if isinstance(playbook, dict) else playbook


def _spent(runs: List[dict]) -> Optional[float]:
    """The session's spend: each run writes the CLI's running total for the
    session, so the largest, as the harness counts it (``_spent_before``)."""
    spent = [
        run["spent_usd"]
        for run in runs
        if isinstance(run.get("spent_usd"), (int, float))
        and not isinstance(run.get("spent_usd"), bool)
    ]
    return round(max(spent), 2) if spent else None


def _last_active(folder: Path) -> Optional[str]:
    times = []
    for name in (manager.RECORD, EVENTS_IN_SESSION, manager.OUTPUT):
        try:
            times.append((folder / name).stat().st_mtime)
        except OSError:
            continue
    return timestamps.iso_from_posix(max(times)) if times else None


def _folders(root: Path) -> List[Path]:
    """Every session folder under *root*, newest first: with its record, or
    with only its manager's (a start waiting for its app)."""
    if not root.is_dir():
        return []
    found = [
        path
        for path in root.iterdir()
        if (path / manager.RECORD).is_file() or (path / manager.MANAGED).is_file()
    ]
    return sorted(found, key=lambda path: path.name, reverse=True)


def _id_from_name(folder: Path) -> Optional[str]:
    """The session id in a folder's name (`manager.folder_name`)."""
    parts = folder.name.split("-", 4)
    return parts[4] if len(parts) == 5 and parts[4] else None


def _link(folder: Path, session_id: str, roots: Iterable[Path]) -> Optional[dict]:
    """The experiment link in *folder*, or in the same session's folder under
    another of *roots* (the agent's files follow the state root)."""
    found = manager.read_json(folder / LINK)
    if found.get("path"):
        return found
    for root in roots:
        other = manager.find_folder(session_id, root)
        if other is not None and other != folder:
            found = manager.read_json(other / LINK)
            if found.get("path"):
                return found
    return None


def _roots(root: Optional[Path]) -> List[Path]:
    if root is not None:
        return [root]
    found = [manager.sessions_dir()]
    state = paths.state_root() / "agent" / "sessions"
    if state != found[0]:
        found.append(state)
    return found


def list_sessions(
    limit: Optional[int] = None,
    root: Optional[Path] = None,
    alive: Callable[[dict], bool] = _alive,
) -> List[SessionRow]:
    """This PC's agent sessions, newest first, with each one's status and the
    experiment it worked on."""
    roots = _roots(root)
    rows: List[SessionRow] = []
    for folder in _folders(roots[0]):
        if limit is not None and len(rows) >= limit:
            break
        record = manager.read_json(folder / manager.RECORD)
        managed = manager.read_json(folder / manager.MANAGED)
        session_id = record.get("session_id") or (
            _id_from_name(folder) if managed and not record else None
        )
        if not session_id:
            continue
        about = managed.get("about") or {}
        runs = record.get("runs") or [{}]
        status, reason = session_status(record, managed, alive)
        link = _link(folder, session_id, roots[1:])
        rows.append(
            SessionRow(
                session_id=session_id,
                folder=str(folder),
                status=status,
                reason=reason,
                started_at=runs[0].get("started") or about.get("started"),
                ended_at=runs[-1].get("ended") if status == ENDED else None,
                last_active_at=_last_active(folder),
                playbook=_playbook(record) or about.get("playbook"),
                model=record.get("model"),
                goal=record.get("goal"),
                level=about.get("level_now") or about.get("level"),
                spent_usd=_spent(runs),
                by_gateway=bool(managed),
                experiment=(
                    ExperimentLink(link["path"], link.get("name"), link.get("id"))
                    if link
                    else None
                ),
            )
        )
    return rows


# Each experiment.yaml's name and id, kept until the file changes: an
# experiment.yaml can take seconds to parse and pages ask often.
_read: Dict[str, Tuple[Tuple[int, int], Tuple[Optional[str], Optional[str]]]] = {}


def _name_and_id(
    path: str, stat: os.stat_result
) -> Tuple[Optional[str], Optional[str]]:
    stamp = (stat.st_mtime_ns, stat.st_size)
    kept = _read.get(path)
    if kept is not None and kept[0] == stamp:
        return kept[1]
    try:
        with open(path, encoding="utf-8") as f:
            saved = yaml.safe_load(f)
    except (OSError, yaml.YAMLError, UnicodeDecodeError):
        logger.warning("could not read %s", path, exc_info=True)
        saved = None
    saved = saved if isinstance(saved, dict) else {}
    found = (saved.get("name"), str(saved["_id"]) if saved.get("_id") else None)
    _read[path] = (stamp, found)
    return found


def _experiment(folder: str) -> Optional[Tuple[float, ExperimentRow]]:
    """When the experiment in *folder* was last written, and its row; None
    when it has no experiment.yaml."""
    path = os.path.join(folder, EXPERIMENT_FILE)
    try:
        stat = os.stat(path)
    except OSError:
        return None
    name, experiment_id = _name_and_id(path, stat)
    times = [stat.st_mtime]
    try:
        times.append(os.stat(os.path.join(folder, EVENTS_FILE)).st_mtime)
    except OSError:
        pass
    return max(times), ExperimentRow(
        path=folder,
        name=name or os.path.basename(folder.rstrip("/\\")) or folder,
        id=experiment_id,
        updated_at=timestamps.iso_from_posix(max(times)),
    )


def _recent_folders() -> List[str]:
    try:
        return [os.path.dirname(p) for p in recent_experiment_paths()]
    except Exception:  # noqa: BLE001 - a bad prefs file must not hide the sessions'
        logger.warning("could not read the recent experiments", exc_info=True)
        return []


def _path_key(folder: str) -> str:
    return os.path.normcase(os.path.normpath(folder))


def list_experiments(
    sessions: Optional[List[SessionRow]] = None,
    recent: Callable[[], List[str]] = _recent_folders,
) -> List[ExperimentRow]:
    """The experiments that are recent on this PC or that a session here
    worked on, most recently updated first, each with the sessions that
    worked on it. One row per experiment id (by folder for an experiment with
    none), so a moved experiment is listed once. A linked experiment that
    isn't on this PC is kept, known by its link, after the ones that are.
    *sessions* is `list_sessions()` when not given."""
    sessions = list_sessions() if sessions is None else sessions
    rows: Dict[str, Tuple[float, ExperimentRow]] = {}
    # Each folder seen, to the key of its row.
    keys: Dict[str, str] = {}

    def add(folder: str, link: Optional[ExperimentLink] = None) -> str:
        path_key = _path_key(folder)
        if path_key in keys:
            return keys[path_key]
        found = _experiment(folder)
        if found is not None:
            experiment_id = found[1].id or (link.id if link else None)
        elif link is not None:
            experiment_id = link.id
        else:
            return ""
        key = f"id:{experiment_id}" if experiment_id else f"path:{path_key}"
        keys[path_key] = key
        kept = rows.get(key)
        if found is not None and (kept is None or not kept[1].on_this_pc):
            if kept is not None:
                found[1].session_ids = kept[1].session_ids
            rows[key] = found
        elif kept is None:
            rows[key] = (
                float("-inf"),
                ExperimentRow(
                    path=folder,
                    name=link.name or os.path.basename(folder.rstrip("/\\")) or folder,
                    id=link.id,
                    on_this_pc=False,
                ),
            )
        return key

    for folder in recent():
        add(folder)
    for row in sessions:
        if row.experiment is None:
            continue
        key = add(row.experiment.path, row.experiment)
        if key and row.session_id not in rows[key][1].session_ids:
            rows[key][1].session_ids.append(row.session_id)
    found = sorted(rows.values(), key=lambda kept: kept[0], reverse=True)
    return [row for _, row in found]
