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
from fibsem.util.process import pid_alive

logger = logging.getLogger(__name__)

EXPERIMENT_FILE = "experiment.yaml"
EVENTS_FILE = "events.jsonl"
# In a session's agent/ folder: the experiment it works on.
LINK = os.path.join("agent", "experiment.json")

# A session's status.
RUNNING = "running"
# Its app went away and its manager will resume it when the app is back.
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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _alive(run: dict) -> bool:
    """Whether *run*'s process is still running."""
    return pid_alive(run.get("pid"))


def session_status(
    record: dict, managed: dict, alive: Callable[[dict], bool] = _alive
) -> Tuple[str, Optional[str]]:
    """A session's status and, once it has ended, why, from its record
    (``session.json``) and its manager's (``manager.json``, or empty)."""
    runs = record.get("runs") or [{}]
    last = runs[-1]
    if "ended" not in last:
        if alive(last):
            return RUNNING, None
        if managed.get("ended"):
            return ENDED, managed["ended"]
        if managed.get("left") is not None:
            return LEFT, "the gateway was closed"
        return CRASHED, None
    if managed.get("ended"):
        return ENDED, managed["ended"]
    if (
        managed
        and last.get("exit_code") == manager.GONE
        and last.get("resumable", True)
    ):
        return WAITING, last.get("reason")
    return ENDED, last.get("reason")


def _playbook(record: dict) -> Optional[str]:
    playbook = record.get("playbook")
    return playbook.get("name") if isinstance(playbook, dict) else playbook


def _spent(runs: List[dict]) -> Optional[float]:
    spent = [run["spent_usd"] for run in runs if "spent_usd" in run]
    return round(sum(spent), 2) if spent else None


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
    rows = []
    for folder in manager.folders(roots[0])[:limit]:
        record = manager.read_json(folder / manager.RECORD)
        session_id = record.get("session_id")
        if not session_id:
            continue
        managed = manager.read_json(folder / manager.MANAGED)
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


def list_experiments(
    sessions: Optional[List[SessionRow]] = None,
    recent: Callable[[], List[str]] = _recent_folders,
) -> List[ExperimentRow]:
    """The experiments on this PC that are recent or that a session worked on,
    still on disk, most recently updated first, each with the sessions that
    worked on it. *sessions* is `list_sessions()` when not given."""
    sessions = list_sessions() if sessions is None else sessions
    worked: Dict[str, List[str]] = {}
    folders: Dict[str, str] = {}
    for folder in recent():
        folders.setdefault(os.path.normcase(os.path.normpath(folder)), folder)
    for row in sessions:
        if row.experiment is None:
            continue
        key = os.path.normcase(os.path.normpath(row.experiment.path))
        folders.setdefault(key, row.experiment.path)
        worked.setdefault(key, []).append(row.session_id)
    rows = []
    for key, folder in folders.items():
        found = _experiment(folder)
        if found is None:
            continue
        found[1].session_ids = worked.get(key, [])
        rows.append(found)
    rows.sort(key=lambda found: found[0], reverse=True)
    return [row for _, row in rows]
