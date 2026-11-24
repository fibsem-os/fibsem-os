"""Where an agent session's own files go: its plan, its transcript, its
review list and its handoff notes.

They are the agent's record, not the experiment's: what ran the experiment,
kept apart from the data the experiment made. Each session's files live in
its own folder under the state root
(``~/.fibsem/agent/sessions/<date>-<id>/agent/``), from its first line to its
last, whether or not an experiment is open.

One session works on one experiment: the first one open while it runs (opening
one is the first step the agent proposes at intake, never a rule), kept even
if the operator then opens another. The two are linked both ways by id, and
nothing is copied: the session's folder names its experiment
(``agent/experiment.json``), and the experiment's record says the session
joined it (``agent_session_joined``, with the session's folder).

Experiments from before kept their sessions' files in their own ``agent/``
folder. Those are left where they are and still read as earlier sessions'
(`AgentFiles.earlier`).
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from fibsem import paths
from fibsem.ai.agent import manager

logger = logging.getLogger(__name__)

FOLDER = "agent"
PREFIXES = ("plan-", "transcript-", "handoff-", "review-")
# The pictures its transcript shows, ``pictures/<session id>-<line>.<ext>``.
PICTURES = "pictures"
# In the session's agent/ folder: the experiment the session works on.
LINK = "experiment.json"
# On the experiment's record: a session took it.
JOINED = "agent_session_joined"


def sessions_root() -> Path:
    """The folder each session's own folder is in, under the state root."""
    return paths.state_root() / "agent" / "sessions"


def session_folder(session_id: str) -> Path:
    """Session *session_id*'s agent/ folder, made or not."""
    return manager.session_folder(session_id, sessions_root()) / FOLDER


def _read_link(folder: Path) -> Dict[str, Any]:
    try:
        data = json.loads((folder / LINK).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _same(a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b:
        return False
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.normpath(a) == os.path.normpath(b)


class AgentFiles:
    """Each session's folder, and the experiment it took."""

    def __init__(
        self,
        experiment_path: Callable[[], Optional[str]],
        folder: Optional[Callable[[str], Path]] = None,
        joined: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        experiment_id: Optional[Callable[[], Optional[str]]] = None,
    ) -> None:
        self._experiment_path = experiment_path
        self._folder = folder or session_folder
        # Puts the link on the experiment's record (the app's record_event).
        self._joined = joined
        self._experiment_id = experiment_id or (lambda: None)
        # session id -> the experiment it took, as read from its link.
        self._taken: Dict[str, str] = {}

    def _open(self) -> Optional[str]:
        try:
            path = self._experiment_path()
        except Exception:  # noqa: BLE001 - no experiment is the answer
            logger.debug("could not read the open experiment", exc_info=True)
            return None
        return path if path and os.path.isdir(path) else None

    def experiment(self, session_id: Optional[str]) -> Optional[str]:
        """The experiment session *session_id* works on: the one it took,
        else the one open now, which it takes. None while none is open."""
        if not session_id:
            return self._open()
        taken = self._taken.get(session_id)
        if taken is None:
            taken = _read_link(self._folder(session_id)).get("path")
        if taken and os.path.isdir(taken):
            self._taken[session_id] = taken
            return taken
        found = self._open()
        if found is not None:
            self._take(session_id, found)
        return found

    def _take(self, session_id: str, experiment: str) -> None:
        """Session *session_id* takes *experiment*: each names the other."""
        self._taken[session_id] = experiment
        folder = self._folder(session_id)
        link = {
            "session_id": session_id,
            "path": experiment,
            "name": Path(experiment).name,
            "taken": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        try:
            experiment_id = self._experiment_id()
        except Exception:  # noqa: BLE001 - the path still links them
            experiment_id = None
        if experiment_id:
            link["id"] = experiment_id
        try:
            folder.mkdir(parents=True, exist_ok=True)
            (folder / LINK).write_text(json.dumps(link, indent=2), encoding="utf-8")
        except OSError:
            logger.warning("could not link %s to %s", folder, experiment, exc_info=True)
        if self._joined is not None:
            try:
                self._joined(
                    JOINED, {"session_id": session_id, "folder": str(folder.parent)}
                )
            except Exception:  # noqa: BLE001 - the session's link still holds
                logger.warning("could not record the session joining", exc_info=True)

    def folder(self, session_id: Optional[str]) -> Optional[Path]:
        """Where session *session_id*'s files go: its own folder, which first
        takes the open experiment if it has none. None without a session."""
        if not session_id:
            return None
        self.experiment(session_id)
        return self._folder(session_id)

    def sessions(self, experiment: Optional[str]) -> List[Tuple[str, Path]]:
        """The sessions that worked on *experiment*, as (session id, agent/
        folder), newest first: from their links, then the experiment's own
        agent/ folder for sessions from before (one folder for all)."""
        found: List[Tuple[float, str, Path]] = []
        root = sessions_root()
        if experiment and root.is_dir():
            for link in root.glob(f"*/{FOLDER}/{LINK}"):
                data = _read_link(link.parent)
                session_id = data.get("session_id")
                if not session_id or not _same(data.get("path"), experiment):
                    continue
                try:
                    found.append((link.stat().st_mtime, session_id, link.parent))
                except OSError:
                    continue
        found.sort(reverse=True)
        return [(session_id, folder) for _, session_id, folder in found]

    def folders(self, session_id: Optional[str]) -> List[Path]:
        """Every agent/ folder of *session_id*'s experiment (the open one
        without a session): the session's own first, then the other
        sessions' on it, then the experiment's own from before."""
        experiment = self.experiment(session_id) if session_id else self._open()
        found: List[Path] = [self._folder(session_id)] if session_id else []
        found += [f for other, f in self.sessions(experiment) if other != session_id]
        if experiment:
            found.append(Path(experiment) / FOLDER)
        return [folder for folder in found if folder.is_dir()]

    def earlier(
        self, session_id: Optional[str], prefix: str, suffix: str
    ) -> List[Tuple[str, Path]]:
        """The other sessions' files named *prefix*<id>*suffix* on
        *session_id*'s experiment, newest first, as (session id, file)."""
        found: Dict[str, Tuple[float, Path]] = {}
        for folder in self.folders(session_id):
            for path in folder.glob(f"{prefix}*{suffix}"):
                other = path.name[len(prefix) : len(path.name) - len(suffix)]
                if not other or other == session_id or other in found:
                    continue
                try:
                    found[other] = (path.stat().st_mtime, path)
                except OSError:
                    continue
        ordered = sorted(found.items(), key=lambda kv: kv[1][0], reverse=True)
        return [(other, path) for other, (_, path) in ordered]

    def find(self, session_id: str, name: str) -> Optional[Path]:
        """File *name* of session *session_id*: in its own folder, else (a
        session from before) in its experiment's agent/ folder."""
        own = self._folder(session_id) / name
        if own.exists():
            return own
        experiment = self.experiment(session_id)
        if experiment:
            old = Path(experiment) / FOLDER / name
            if old.exists():
                return old
        return None
